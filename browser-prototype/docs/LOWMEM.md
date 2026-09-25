# Gemma 4 12B on 16 GB computers (lowmem)

Plan: `~/.claude/plans/for-reference-on-the-eager-crayon.md`. Branch
`lowmem/old-laptop`, from `fix2/final-review` (`118b5d6`).

Every number says where it comes from:

| Tag | Meaning |
|---|---|
| **[Mac]** | Measured on this Mac (MacBook Air M5, 24 GB). |
| **[E]** | Estimate. Not measured. |

The memory margins below the Windows budget (default ≤ 7.0 GB, fallback ≤ 5.5 GB)
are guesses. No Windows machine has run this build.

## Setup

| Item | Value |
|---|---|
| Model package | `/Users/shonusengupta/BlinkFix-models/gemma-12b-q4/` (`decoder.onnx` `a6c6f4ea…`, the fix2 trim graph) |
| Fork dist | `ort-fork-dist-pm-a`, copied to `.local/ort-fork-dist` |
| Browser profile (model cached) | `agent-a1a20012219348d7d/browser-prototype/evidence/gemma12b-profile` |
| Degraded scans | `evidence/lowmem/scans/` (made by `make_scans.py` there): openstax and mathfuncs page 1, 150 DPI, grey, 1.3° and −1.8° skew, blur, noise, specks, JPEG q45, image-only PDF |
| Unit tests | 895 tests, 884 pass, 9 known failures (the baseline) |

## Stage 0: baseline at budget 1120

### Token gate [Mac]

Engine check `STEPS=gate`, greedy, one-call prompt, kept prefix, budget 1120,
PDF.js render. File `evidence/lowmem/gate-stage0.json`.

| Page | Tokens | Prefill s | Decode tok/s | GPU peak GB (vmmap) |
|---|---|---|---|---|
| smoke | 25 | 8.9 | 9.5 | 9.52 |
| mathfuncs | 352 | 9.1 | 11.1 | 9.63 |
| openstax | 1,173 | 9.7 | 10.7 | 9.73 |
| long page 13_pg501 | 2,713 | 10.0 | 10.0 | 9.73 |
| scan-openstax (degraded) | 1,169 | 10.6 | 10.0 | 9.73 |
| scan-mathfuncs (degraded) | 358 | 10.9 | 9.9 | 9.73 |

- The 4 gate pages give the same token ids as fix2's gate (`gate-trim256-3.json`,
  `gate-trim256-dense.json`): IDENTICAL on all 4 (`scripts/lowmem-gate-compare.mjs`).
- The GPU peak is the engine check's, not the app's (the app adds the
  speculative drafter and the UI).

## Phase A1: the MTP drafter that sees the image [Mac]

Native llama.cpp (build 10180), target `gemma-4-12b-it-qat-q4_0.gguf` (the GGUF
the app package is built from), drafter `gemma-4-12b-it-qat-assistant-q4_0.gguf`
(Google's QAT assistant). The app's system prompt and user text, the page image
at 1,120 image tokens, greedy, thinking off. Script `scripts/lowmem-a1-mtp.mjs`,
summary `scripts/lowmem-a1-summary.mjs`, file `evidence/lowmem/a1/mtp.json`.
Pages: smoke, mathfuncs, openstax, the long page, and the 2 degraded scans.

| Draft length n | Tokens | Target passes | Tokens per pass (tau) | Drafts accepted | Decode tok/s (M5) | Same text as n=0 |
|---|---|---|---|---|---|---|
| 0 (no drafter) | 5,117 | 5,117 | 1.00 | – | 14.2 | – |
| 2 | 5,117 | 1,995 | **2.56** | 78% | 18.1 | 6/6 |
| 4 | 5,117 | 1,489 | **3.44** | 61% | 16.9 | 6/6 |
| 8 (stopped early, 3 pages) | 1,532 | 243 | 6.30 | 66% | 15.8 | 3/3 |

| Page | tau n=2 | tau n=4 |
|---|---|---|
| smoke | 3.13 | 4.17 |
| mathfuncs | 2.92 | 4.68 |
| openstax | 2.85 | 4.29 |
| long page (dense 2-column) | 2.22 | 2.59 |
| scan-openstax (degraded) | 2.82 | 4.34 |
| scan-mathfuncs (degraded) | 2.90 | 4.58 |

- **A1 passes.** The kill line was 1.5 tokens per pass. The drafter works as
  well on the degraded image-only scans as on the clean pages, because it reads
  the target's KV and so sees the image.
- The output is the same text as without the drafter on every page.
- On the M5 the wall-clock gain is small (1.27x at n=2), because this GPU has
  spare bandwidth at batch 1 and the drafter's own passes cost time. On a
  bandwidth-bound laptop GPU, one target pass costs about as much as reading
  the weights once, so tokens per pass is the number that matters.
  Estimate for Iris Xe: 4 to 4.5 tok/s greedy × about 2.2 to 2.6 ≈ 9 to 11 tok/s [E].
- The dense long page is the weak case (2.2 to 2.6).
- Not measured: the tesseract drafter on the same pages uses a different
  harness; see the queue-1 spec run below.

### Today's tesseract drafter on the same pages [Mac]

Engine check `STEPS=spec SPEC_ONLY=spec` (the app's speculative path), budget
1120, file `evidence/lowmem/spec-stage0.json`.

| Page | Tokens | Passes | Tokens per pass | MTP drafter, n=2 (A1) |
|---|---|---|---|---|
| openstax | 1,193 | 720 | 1.66 | 2.85 |
| scan-openstax (degraded) | 1,166 | 778 | 1.50 | 2.82 |
| long page, app run (below) | 2,702 | 1,595 | 1.69 | 2.22 |

The MTP drafter gives 1.3x to 1.9x more tokens per pass than the tesseract
drafter, and it needs no OCR on the CPU.

## Stage 0: the app on the long page [Mac]

`scripts/gemma12b-app-check.mjs`, default build (speculative tesseract drafter),
budget 1120, Convert at 15 s. File `evidence/lowmem/app-stage0-long.json`.

| Measure | Value |
|---|---|
| GPU footprint at Convert (model loaded) | 8.40 GB |
| GPU peak (vmmap, end of page) | **9.93 GB** (fix2 measured 9.93 GB) |
| First text after Convert | 13.8 s |
| Output tokens | 2,702 (greedy gives 2,713: the known speculative mismatch, C1) |
| Decode | 11.7 tok/s |

## Phase A3: one weight buffer, several sessions [Mac]

Two tiny graphs (`tools/lowmem/make_a3_graphs.py`): a 4-bit `MatMulNBits`
(K 3840, N 4096) with the weight and scales as **graph inputs**, one with a
Relu after it. The page uploads the weight once (`writeBuffer`), wraps it with
`Tensor.fromGpuBuffer` and runs both sessions on it. Script
`scripts/lowmem-a3.mjs`, file `evidence/lowmem/a3/result.json`.

| Check | Result |
|---|---|
| Output vs numpy (fp32) | max abs diff 0.012, 0.17% of the largest value (fp16 rounding) |
| Session 1 run twice | the same bits |
| Relu session | equals max(session 1, 0) |
| Node placement (profiling) | every node on WebGPU |
| GPU buffers ORT created at or above the scales' size during the runs | none (no copy of the weight) |
| `writeBuffer`, 8 × 128 MB | 11.9 GB/s (7.0 on the first, cold, round) |

**A3 passes.** One GPU weight buffer can feed several sessions without a copy.
That is what the fallback (B4) and a layer-split decoder (B3) need.

## Stage 0: where the GPU memory goes, long page [Mac]

Engine check `STEPS=mem`, greedy, budget 1120, the long page (a 240-DPI render,
2,719 tokens). The page's GPU buffer tracker gives live bytes by creation tag.
File `evidence/lowmem/mem-stage0-long.json`. (This run took 18.5 min, because
`vmmap` every 0.4 s slowed decode to 2.5 tok/s. Use that sampling on short
pages only.)

| Item | MiB | How |
|---|---|---|
| Decoder weights (6.13 GB body + the 1.07 GB 8-bit token table) | 6,953 | buffers made while the decoder session loads |
| Vision weights | 146 | same, vision session |
| Kept prefix KV (829 tokens) | 336 (272 real data) | the prefix tensors; ORT rounds each buffer up |
| ORT cache kept from load (prefix prefill, warm-up) | about 350 | `load:prefix+warm` minus the prefix KV |
| Page KV at the first token | 672 (368 real data) | the page's cache tensors |
| Page buffers still live after the page ended | 623 (prefill) + 907 (decode) = **1,530** | `page1` and `page1:decode` tags at the end, with the page's KV already freed |
| Tracker peak / GPU-process peak | 9,377 / 9,728 (vmmap) | |

What this says:

- About 1.5 GB of the peak is ORT's buffer cache holding freed buffers. The
  decode part (907 MiB) grows with every KV merge: each merge makes new, larger
  stable tensors for the 8 global layers, so a freed buffer never fits the next
  request. B1 (one fixed size for those tensors) targets exactly this.
- ORT rounds KV buffers up by about 1.8x (672 MiB held for 368 MiB of data).
- The 8-bit token table is 1.07 GB of the weights. B2 moves it off the GPU.

## Phase A2: a 4-bit LM head plus an exact 8-bit re-check [Mac, CPU]

`tools/gemma12b/lm_head_cert.py`. Hidden states `h` (after the final norm) come
from `llama-embedding` on the QAT GGUF, teacher-forced on the chat prompt plus
each gate page's greedy text, **without the image** (a conservative stand-in:
the model is less sure without the page, so margins are smaller). The 8-bit
table is read from the app package. The 4-bit copy is block-32 with an fp16
scale and a 4-bit zero point (`asym`), or symmetric (`sym`).

Certificate at each position: the exact 8-bit best among the top K rows by
4-bit logit must beat an upper bound on every other row,
`l4_j + Σ_b ‖h_b‖·‖w8_jb − w4_jb‖` (per block; the per-row bound is looser).

Validation first (`evidence/lowmem/a2/cert-val.json`): on 192 positions of
llama.cpp's own greedy text, `argmax(W8·h)` equals the next token 99.5% of the
time, so the hidden states are right.

Gate pages, 5,789 positions (`evidence/lowmem/a2/cert-gate.json`), `asym`:

| K | Certified (per-block bound) | Certified (per-row bound) | True 8-bit argmax inside the 4-bit top K |
|---|---|---|---|
| 16 | 88.7% | 79.4% | 100% |
| 64 | 93.0% | 85.1% | 100% |
| 256 | 95.9% | 89.3% | 100% |

- **A2 fails the 99% bar, so B2 as written is dropped.** A worst-case bound
  cannot prove what the 4-bit head gets right in practice. A larger K helps
  slowly. At every failing position, a full 8-bit pass on the CPU would cost
  about 1 GFLOP per token.
- In practice, the 4-bit head put the true 8-bit argmax in its top 16 at every
  one of the 5,789 + 192 positions. A head that re-checks only the top 16
  would match here, but nothing proves it for every page. That is a
  non-exact change, so it is the user's decision (saving about 0.55 GB of GPU
  memory).

## C5: an AI edit pauses a page instead of restarting it

Before: an AI edit terminated the worker. The model loaded again for the edit,
and the page ran again from the start (a second load and a full prefill).

Now (gemma12b build, `InferenceQueue` option `pausable`):

| Step | What happens |
|---|---|
| Edit arrives | The queue sends `pause` to the worker. |
| Worker | The engine stops between two decode steps, keeps the page's KV cache and streamer, and answers the page request with status `paused`. |
| Edit | Runs on the same worker, with the model loaded. |
| Page | The queue sends it again with `resumeFrom`. The engine feeds the last id on the kept cache and continues greedily. The live text continues; it does not restart. |
| Fallbacks | A worker that answers `pause-unsupported`, or no answer within 120 s, gets the old path (terminate, run again). A cancelled paused page is dropped in the worker (`dropSuspended`); a new page also drops an old paused one. |

Files: `src/engine.js`, `src/gemma12b/engine.js`, `src/gemma12b/generate.js`,
`src/gemma12b/speculative.js`, `src/inference.worker.js`, `src/app.js`.
Tests: `tests/queue-pause.test.mjs` (6), `tests/gemma12b-engine.test.mjs` (3 new).

Real check [Mac] (`STEPS=pause`, `evidence/lowmem/c5-pause.json`): mathfuncs at
budget 1120, paused at token 100, a text-only edit ran while the paused page
held its 100 tokens, then the page resumed. **The 352 ids are identical to the
Stage 0 gate**, and the figure-crop info (`pageCrops`) is the same.

A resume decodes greedily (no drafter). The ids are the same by construction.
The paused page's KV stays in GPU memory while the edit runs.
