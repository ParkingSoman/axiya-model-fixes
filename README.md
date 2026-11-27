# Axiya model fixes

**In short:** Axiya runs AI models inside the web browser, on the user's own computer. That surfaced two bugs, which I found and fixed:
1. Google's compact Gemma 4 models have one set of internal numbers about 64 times smaller than in Google's regular versions, which makes them read pages worse. Scaling those numbers up took one test page from 14 to 54 formulas read correctly, out of 96 (12 formulas, 8 tries each).
2. On my older HP laptop, the Intel gen-9 graphics (chip or driver) read every second 16-bit number as 0, so the model wrote random words. Reading numbers in pairs fixed it: the laptop's output now matches a Mac's (similarity 0.999996, where 1 = identical).

The weight sizes can be rerun by anyone (see the Colab link below). The page scores (14/96, 217/304) and the Intel numbers are my own measurements; the test page, its scorer and the raw laptop captures are not public.

The rest of this page gives the details.

Axiya (https://axiya.io) runs vision-language models locally in the browser on WebGPU, through a fork of ONNX Runtime Web.
This repo holds the records of the model and GPU bugs I found and fixed to make that work.
It was cut out of Axiya's private repo (code name BlinkFix), with the original commit history. Git dates are set by the author; the Hugging Face model cards below, created 2026-09-28, give an independent date for fix 1.

## 1. Google's Gemma 4 QAT checkpoints: one weight is about 64 times smaller than in the base model

In Google's quantization-aware-trained (QAT) Gemma 4 E4B checkpoint, `per_layer_model_projection.weight` has RMS 0.000384, against 0.02476 in the base model.
The RMSNorm after it has eps 1e-6 (a small constant added for stability). With the small weight, the signal falls below eps, so the norm shrinks the whole per-layer branch: to about 0.24x on image tokens and 0.64x on text tokens.
E2B's QAT checkpoint has the same defect, less strongly.

The fix: scale that one tensor up (x64 for E4B, x53.65 for E2B). The norm cancels any scale, so only the eps effect goes away.

Effect on one scanned math-paper page (llama.cpp, the released QAT Q4_0 file, 8 sampled draws per row). An anchor is one formula or one stretch of text that must come out exactly right; the page has 12 formula anchors and 38 text anchors, each read 8 times:

| Model file | Formula anchors | Text anchors |
|---|---|---|
| QAT as released | 14/96 | 217/304 |
| Same file, only this tensor x64 | **54/96** | **286/304** |
| Same file, tensor replaced by the base model's | 53/96 | |

A control run changes only eps (to 1e-12) and matches the x64 fix (28/48 each, on 4 draws), so eps is the cause, not precision. Why the weight is small is not known. The report's guess is a folded runtime scale.

- Full report and a repro script that reads Google's own checkpoints: [`audit/gemma4-qat-ple/`](audit/gemma4-qat-ple/)
- [Open in Colab](https://colab.research.google.com/github/ParkingSoman/axiya-model-fixes/blob/main/audit/gemma4-qat-ple/rerun.ipynb)
- Rerun on any computer in about 1 minute (measured: 37 s for E4B, 34 s for E2B on a Mac CPU, after the packages are installed): `pip install torch "transformers>=5.5" huggingface_hub safetensors` then `python audit/gemma4-qat-ple/repro_ple_eps.py`
- Reported to Google on the checkpoint's Hugging Face page (2026-11-27): https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized/discussions/7
- Patch script: `browser-prototype/scripts/e4b-pleproj-patch.py` (commit `e624af8`, 2026-09-27)
- My published models carry the fix. Their cards list the scale: [E4B](https://huggingface.co/parkingsoman/gemma-4-e4b-it-onnx-webgpu) ("x64.0"), [E2B](https://huggingface.co/parkingsoman/gemma-4-e2b-it-onnx-webgpu) ("x53.65")

## 2. Intel gen-9 GPU on my HP laptop: every second 16-bit number reads as 0

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

- Run the probe in Chrome: https://parkingsoman.github.io/axiya-model-fixes/browser-prototype/tools/gpu-debug/f16-store-test.html . It needs a GPU with WebGPU `shader-f16`, and prints `wrongOdd` / `wrongEven` counts for each of 19 kernels. In September 2026 (Chrome 154) the single-element 16-bit reads failed on my HP laptop. On the same laptop in Chrome 158 (2026-11-27), and on an Apple M5 Mac, all 19 kernels report 0 wrong, so Chrome or Intel's driver has since fixed it. The September failure is from my own record; the fix in my runtime stays for older versions.
- Fix: commit `885da74` (2026-09-28), in `browser-prototype/reference/onnxruntime-webgpu.patch`. The record is in `browser-prototype/docs/LOWMEM.md`, section "Garbage text: odd f16 elements read as 0". The raw HP captures are not in this repo.
- The build refuses a fork patch without it: `browser-prototype/scripts/build-ort-fork.sh`

## Other fixes in this history

| Problem | Fix | Commit |
|---|---|---|
| Intel HP: Windows reset the GPU (`DXGI_ERROR_DEVICE_HUNG`) on long jobs | Flash-attention prefill runs in slices, each its own submit | `3ad9f92` |
| Qualcomm laptop: garbage, then a freeze | Every GPU runs the same kernel programs. Checked on the Mac only, not yet on the Qualcomm laptop | `3db61bb` |
| Speculative decoding gave different text from one-token decoding | Every decode kernel runs one program for any pass size | `b1670f2`, `c7c1526` |
| About 1.1 GB stayed in Chrome's GPU process after loading the 12B model | Upload weights in 16 MB pieces and wait for the queue every 64 MB. Reverted once (`0528971`) after two machine crashes, then reapplied | `1b299dc` |
