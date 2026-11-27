# Axiya model fixes

Axiya (https://axiya.io) runs vision-language models locally in the browser on WebGPU, through a fork of ONNX Runtime Web.
This repo holds the records of the model and GPU bugs I found and fixed to make that work.
It was cut out of Axiya's private repo (code name BlinkFix), and the commits keep their original dates and messages.

Credit: I directed the work, chose what to test and checked the results. Claude (Opus) coding sessions wrote much of the code and the commit messages. That is why 80 of the commits carry a Co-Authored-By line.

## 1. Google's Gemma 4 QAT checkpoints: one weight is about 64 times too small

In Google's quantization-aware-trained (QAT) Gemma 4 E4B checkpoint, `per_layer_model_projection.weight` has RMS 0.000384, against 0.02476 in the base model.
The RMSNorm after it has eps 1e-6 (a small constant added for stability). With the small weight, the signal falls below eps, so the norm shrinks the whole per-layer branch: to about 0.24x on image tokens and 0.64x on text tokens.
E2B's QAT checkpoint has the same defect, less strongly.

The fix: scale that one tensor up (x64 for E4B, x53.65 for E2B). The norm cancels any scale, so only the eps effect goes away.

Effect on one scanned math-paper page (llama.cpp, the released QAT Q4_0 file, 8 sampled draws per row):

| Model file | Formula anchors | Text anchors |
|---|---|---|
| QAT as released | 14/96 | 217/304 |
| Same file, only this tensor x64 | **54/96** | **286/304** |
| Same file, tensor replaced by the base model's | 53/96 | |

A control run changes only eps (to 1e-12) and recovers the same score as x64, so eps is the cause, not precision.

- Full report and a repro script that reads Google's own checkpoints: [`audit/gemma4-qat-ple/`](audit/gemma4-qat-ple/)
- Patch script: `browser-prototype/scripts/e4b-pleproj-patch.py` (commit `e624af8`, 2026-09-27)
- My published models carry the fix. Their cards list the scale: [E4B](https://huggingface.co/parkingsoman/gemma-4-e4b-it-onnx-webgpu) ("x64.0"), [E2B](https://huggingface.co/parkingsoman/gemma-4-e2b-it-onnx-webgpu) ("x53.65")

## 2. Intel gen-9 GPUs: every second 16-bit number reads as 0

On my older HP laptop (Intel gen-9 integrated GPU, Windows, Chrome, D3D12), the model wrote random words.
An op-by-op comparison against the Mac (`browser-prototype/tools/gpu-debug/vision_taps.py`, `compare_taps.py`) found the first wrong step: every odd 16-bit value read as 0.
A 19-kernel WGSL probe (`browser-prototype/tools/gpu-debug/f16-store-test.html`) narrowed it down. When a shader picks one element out of a 16-bit load, that GPU returns 0 for odd elements. Stores, whole-vector reads and 32-bit reads are exact.
The fix makes every GPU step read the full 32-bit word, which holds two numbers, and split out the number it needs.

| Check | Before | After |
|---|---|---|
| HP vs Mac: vision similarity (median cosine) | 0.45 | **0.999996** |
| HP vs Mac: top-1 match on 8 checks | 0/8 | **8/8** |
| Mac: vision taps still bit-identical | | 141/141 |
| Mac: cost (decoder GPU time) | | +2.3% |

- Fix: commit `885da74` (2026-09-28), in `browser-prototype/reference/onnxruntime-webgpu.patch`. The record is in `browser-prototype/docs/LOWMEM.md`, section "Garbage text: odd f16 elements read as 0". The raw HP captures are not in this repo.
- The build refuses a fork patch without it: `browser-prototype/scripts/build-ort-fork.sh`

## Other fixes in this history

| Problem | Fix | Commit |
|---|---|---|
| Intel HP: Windows reset the GPU (`DXGI_ERROR_DEVICE_HUNG`) on long jobs | Flash-attention prefill runs in slices, each its own submit | `3ad9f92` |
| Qualcomm laptop: garbage, then a freeze | Every GPU runs the same kernel programs. Checked on the Mac only, not yet on the Qualcomm laptop | `3db61bb` |
| Speculative decoding gave different text from one-token decoding | Every decode kernel runs one program for any pass size | `b1670f2`, `c7c1526` |
| About 1.1 GB stayed in Chrome's GPU process after loading the 12B model | Upload weights in 16 MB pieces and wait for the queue every 64 MB. Reverted once (`0528971`) after two machine crashes, then reapplied | `1b299dc` |
