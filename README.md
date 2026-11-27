# Axiya model fixes

Axiya (https://axiya.io) runs vision-language models locally in the browser, on WebGPU, through a fork of ONNX Runtime Web.
Getting there meant fixing problems you do not see with cloud AI.
This repo holds the records of those fixes. It was cut out of Axiya's private repo, and the commits keep their original dates and messages.

## The two fixes I am proudest of

### 1. Gemma 4 E4B and E2B: a weight about 64 times too small

In Google's quantization-aware (QAT) Gemma 4 E4B and E2B checkpoints, one per-layer embedding tensor (`per_layer_model_projection`) is about 64 times smaller than in the base model (E4B RMS 0.000384 vs 0.02476).
The RMSNorm after it has eps 1e-6, and the signal falls close to eps, so the norm shrinks that branch.
The fix scales the tensor up (x64 for E4B, x53.65 for E2B). In exact arithmetic the RMSNorm cancels the scale, so only the eps effect goes away.

| My document benchmark | Before | After |
|---|---|---|
| Math transcription | 14/96 | 54/96 |
| General text transcription | 217/304 | 286/304 |

- Script: `browser-prototype/scripts/e4b-pleproj-patch.py` (commit `e624af8`, 2026-09-27)
- The published models carry the fix. Their cards list the scale:
  - https://huggingface.co/parkingsoman/gemma-4-e4b-it-onnx-webgpu ("PLE projection (F16, x64.0)")
  - https://huggingface.co/parkingsoman/gemma-4-e2b-it-onnx-webgpu ("x53.65")

### 2. Intel gen-9 GPUs: every second 16-bit number read as 0

On an older Intel laptop (Windows, Chrome, D3D12), the model wrote random words.
An op-by-op comparison with the Mac (`tools/gpu-debug/vision_taps.py`, `compare_taps.py`) found the first wrong op: every odd f16 value read as 0.
A 19-kernel WGSL probe (`tools/gpu-debug/f16-store-test.html`) narrowed it down. When a shader picks one element out of a 16-bit storage load, the driver returns 0 for odd elements. Stores, whole-vector reads and 32-bit reads are exact.
The fix makes every GPU step read the full 32-bit word, which holds two numbers, and then split out the number it needs.

| Check against the Mac output | Before | After |
|---|---|---|
| Vision similarity (median cosine) | 0.45 | 0.999996 |
| Top-1 match | 0/8 | 8/8 |

- Fix: commit `885da74` (2026-09-28), in `browser-prototype/reference/onnxruntime-webgpu.patch`
- The build refuses a fork patch without it: `browser-prototype/scripts/build-ort-fork.sh`

## Other fixes in this history

| Problem | Fix | Commit |
|---|---|---|
| Qualcomm Adreno laptop: garbage text, then a freeze | Every GPU runs the same kernel programs ("gpu-neutral") | `3db61bb` |
| Windows GPU reset on long submits | Flash-attention prefill runs in slices, each its own submit | `3ad9f92` |
| Speculative decoding gave different text from greedy decoding | Every decode kernel runs one program for any pass size | `b1670f2`, `c7c1526` |
| About 1.1 GB of GPU memory leaked on load | Upload weights in 16 MB pieces and wait for the queue every 64 MB | `1b299dc` |

The full engineering log is `browser-prototype/docs/LOWMEM.md`.
