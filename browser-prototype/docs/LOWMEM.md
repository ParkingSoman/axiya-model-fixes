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

## Try it locally

```
cd browser-prototype
npm start
```

It checks the ONNX Runtime build and the model files, builds the app (the
canonical path), serves the model files on port 4190 (or reuses a server that
already serves them), serves the app on port 4173 and opens Chrome. An earlier
`npm start` is replaced; a program that is not ours on 4173 is left alone and
the app uses the next free port. The first load copies the model (8.3 GB) into
the browser from the local server. Ctrl+C stops it.

## Status (2026-09-25)

| Stage | Status | Where it stands |
|---|---|---|
| Stage 0 baseline | done | gate at budget 1120; memory split measured |
| A0 budget | done | the user kept 1120 |
| A1 MTP drafter (llama.cpp) | done | 2.56 / 3.44 tokens a pass (n = 2 / 4) |
| A2 certificate | done, failed the bar | the user chose the hybrid head instead |
| A3 shared weight buffers | done | works, no copy |
| B1 fixed short context | done (measured; not used) | global band built and tested, off in the app (0.1 GB saved, ~6% slower); buffer-cache modes: bucket best; vision session kept (0.15 GB) |
| B2 hybrid LM head | done | GPU peak -0.3 to -0.4 GB; exact re-check ~2 ms a token; app build flag `VITE_LM_HYBRID=1` |
| B3 decoder in parts | done (experiment) | no per-boundary cost once warmed (`docs/LOWMEM-STREAM.md`; code on branch `lowmem/stream-canonical`, not merged) |
| B4 single fallback | superseded | user decision 2026-09-25: streamed layers (hc30) + MTP + hybrid head is the only path; no fallback |
| C1 speculative = one-token decoding | done | kernel choices independent of the pass size on every GPU (fork dist `inv`); same-cache check 4/4 pages greedy, 2/2 pages sampled, also on the streamed path; no runtime self-check, near-tie guard off |
| Decoding | done | the model's recommended sampling (user decision 2026-09-25), repeatable per output position; see the section below |
| C2 MTP drafter | done | app build flag `VITE_SPECULATIVE=mtp` |
| C3 display pacer | done | 8 tok/s; text stays on its page |
| C4 page 1 before Convert | done | first text about 1.5 s after Convert; Convert reuses the prefetch's page-1 render |
| C4 lookahead (next page while reading) | done | display-side page queue (user decision): page events wait behind their text in the pacer, and the editor keeps a page live until its page-complete reaches the display. App, three-pages.pdf: page 2 saved at 41.5 s, its live text still grew to 1,087 characters at 55.9 s while page 3 decoded; cancel drops the queue |
| C5 pause instead of restart | done | |
| Live edit (branch `lowmem/live-edit`) | done | math and closed figures editable while streaming; each math item checked by MathJax as it closes, a flagged one repaired mid-page; WCAG figure rules (about +0.3 GB GPU peak: mathfuncs app 7.17 GB, 8.78 tok/s). See `docs/LIVE-EDIT.md` |

The canonical path (default build since 2026-09-25): 30 layers streamed
(layout `hc30`), MTP, hybrid LM head.

| Page (app, Mac) | GPU peak | Decode | First text after Convert |
|---|---|---|---|
| smoke | 6.66 GB | 8.0 tok/s | 1.7 s |
| mathfuncs | 6.86 GB | 9.7 tok/s | 1.65 s |
| three-pages.pdf (3 pages) | 6.86 GB | | 2.0 s |
| long page, engine check (2,741 tokens) | 7.07 GB | 4.75 tok/s | |

The long page ends slightly above 7.0 GB: the peak still grows with page
length (6.76 GB at 600 tokens). Iris Xe speed is estimated only (2 to 4.5
tok/s, `docs/LOWMEM-STREAM.md`).

Open finding: vision features vary at rounding level with the session's
history (C1 section). The kernel is not found.

Deviations from the plan (each is a ruling in the ledger,
`.superpowers/sdd/for-reference-on-the-eager-crayon/progress.md`): C1 is judged
from one prefilled cache, not across processes; the near-tie guard uses a lower
bound on the gap with the hybrid head; the gate reference was re-taken with the
fixed fork build (`gate-binv.json`).

## Setup

| Item | Value |
|---|---|
| Model package | `BlinkFix-models/gemma-12b-q4/` (local model folder) (`decoder.onnx` `a6c6f4ea…`, the fix2 trim graph) |
| Fork dist | `ort-fork-dist-inv` (before: `binv`, `pm-a`), its 4 served files in `reference/ort-fork-dist/` (in git since 2026-09-27); pinned by `reference/ort-fork-dist.json` |
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
  non-exact change, so it is my decision (saving about 0.55 GB of GPU
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

## Phase A0: image budget 560 vs 1120 — text accuracy [Mac]

Native llama.cpp (the same QAT model, the app's prompts, greedy, thinking off,
image tokens fixed at the budget, 4 pages at a time), `scripts/lowmem-a0-llama.mjs`.
Scored with the pinned olmOCR scorer (`f7cfe4c`), `tools/lowmem/a0_score.py`.
Files: `evidence/lowmem/a0/llama-budget{560,1120}.json`, `score-summary.json`.

Subset, fixed before any result (my 15-minute rule cut it twice): the
first 2 PDFs of each bench-v2 category (12) plus openstax, 80 olmOCR checks.
Each page is capped at 1,500 output tokens at both budgets.

| Budget | Checks passed (figure blocks removed) | Raw text | Pages at the cap |
|---|---|---|---|
| 1120 | **50 / 80** | 49 / 80 | 4 |
| 560 | **29 / 80** | 29 / 80 | 6 (adds a table page and the degraded mathfuncs scan, which normally ends at about 358 tokens) |

Paired: 22 checks pass only at 1120, 1 only at 560 (sign test p < 0.001).

| Category | 560 | 1120 |
|---|---|---|
| table_tests | 9 / 17 | 16 / 17 |
| multi_column | 2 / 10 | 7 / 10 |
| continuity (math, tables) | 9 / 21 | 15 / 21 |
| long_tiny_text | 3 / 15 | 6 / 15 |
| headers_footers | 1 / 5 | 2 / 5 |
| old_scans | 5 / 12 | 4 / 12 |

| Check type | 560 | 1120 |
|---|---|---|
| table | 9 / 24 | 19 / 24 |
| order | 5 / 17 | 10 / 17 |
| present | 9 / 27 | 13 / 27 |
| math | 2 / 4 | 4 / 4 |
| absent | 4 / 8 | 4 / 8 |

Caveats:
- Small sample (about 10 checks per category). It shows only a large effect.
  The effect here is large.
- llama.cpp resizes some pages differently from the app (test-4's figure box
  sits about 65/1000 higher than the app's), so the app-engine check below
  confirms the table loss on the app's own path.
- The 1,500-token cap cuts the longest pages at both budgets.

App-engine confirmation (the app's own image path; engine check `STEPS=gate`,
greedy, 1,500-token cap; `evidence/lowmem/a0app/`): the 2 table pages that 560
lost most on in llama.cpp.

| Budget | olmOCR checks passed (18) | discoverworld table | 0091c5b pg3 table |
|---|---|---|---|
| 1120 | **12** | cap (1,500) | ends at 1,462 tokens |
| 560 | **6** | cap (1,500) | cap (1,500); an extra empty table column; a wrong cell ("22 (14)" where 1120 reads "15 (9)") |

These pages were chosen because llama.cpp showed the loss there, so this
confirms the direction on the app's path, not the size.

## C1: why speculative runs differ from greedy [Mac]

Engine check `STEPS=vcheck` (`evidence/lowmem/c1-vcheck.json`): the long page's
ids teacher-forced; at positions 1,200 to 2,440 the last row of a k-token verify
pass (k = 2, 4, 8, 16) is compared with a one-token step on the same cache.

| Measure | Value |
|---|---|
| Logits that differ (of 262,144) | about 215,000 to 230,000 at every position and k |
| Largest difference | 0.41 (k = 2 at position 2,000); typical 0.11 to 0.30 |
| Argmax changed at the checked positions | no (their top-2 gaps were 0.69 to 7.7) |

So at long contexts the GPU sums depend on the number of rows in a pass (the
Phase 6 bit-identity was measured on shorter pages). A near tie smaller than
about 0.4 can flip, which explains the different words on the long page.

First fix tried: a near-tie guard (`speculativeGenerate` option `nearTie`, ε =
1.0). A verify row whose top two logits are closer than ε decides nothing, and a
one-token step decides that token. It was not enough. With the guard, MTP runs
still differed from greedy (openstax at token 1,071, the long page at token
119). The per-pass trace (`SPEC_TRACE=1`) shows why: at both points the guard
stopped the pass, and then the one-token step picked another token than greedy,
at a gap of 0.031. The one-token step is not the greedy path, because earlier
multi-row passes had written slightly different cache entries.

### The cause: two kernels whose sums depended on the pass size

A tiny-graph test (`scripts/lowmem-batchinv.html`, graph from
`tools/lowmem/make_attn_graph.py`) runs one layer's attention core as a k-token
verify pass and as the one-token steps greedy takes. It compares each token's
rows bit for bit. 84 attention cases (sliding and global shapes, stable length
1,101 to 3,000, recent length 0 to 50, k = 2 to 8) and 8 value-norm cases.

| Kernel (ORT fork) | What depended on the pass size | Fix |
|---|---|---|
| `SplitMaskedSoftmax` | The vector width (1, 2 or 4) came from the row length. A verify pass has more (masked) columns than a one-token step, so a row's sums were grouped differently. | Always scalar columns. Thread i sums columns i, i + 64, ... Masked extra columns add exactly 0. |
| `ReduceMean` (shared path) | 256 threads for one output, 64 for more. A global layer's value norm is one row of 512 per token, so a one-token step had one output. | Always 64 threads. |

| Build | Attention cases identical | Value-norm cases identical |
|---|---|---|
| `ort-fork-dist-pm-a` (before) | 13 of 84 | 6 of 8 |
| `ort-fork-dist-binv` (after) | 84 of 84 | 8 of 8 |

The recent-part MatMuls also change their program with the column count
(vec4 or not). The test shows that their per-row results do not change.
The norms (`RMSNormalization`, `SkipSimplifiedLayerNormalization`) and
`MatMulNBits` were already made pass-size independent in the fork.

Both fixes are in `reference/onnxruntime-webgpu.patch`. The fork source is at
`BlinkFix-models/tmp-g12/ort-fork-src`, and the new dist is
`BlinkFix-models/ort-fork-dist-binv` (JSPI and asyncify rebuilt).

### Every kernel choice independent of the pass size (fork dist `inv`)

WebGPU gives the same bits for the same program and inputs on one GPU. The
bug was that ORT chose the program by the number of rows, so a k-token pass and
k one-token steps could run different code. After the two fixes above, an audit
of the kernels on the decode path found more such choices, on non-Apple GPUs:

| Kernel | Choice by row count (stock / binv) | Now |
|---|---|---|
| MatMul | tile shape 4x1 up to 8 rows, 4x4 above | 4x1 up to 256 rows |
| MatMul, Intel | subgroup program from 64 rows (its tile also by rows) | only above 256 rows |
| MatMul, subgroup-matrix GPUs | used when the rows fill a tile | only above 256 rows |
| MatMul, Intel Xe2/Xe3 | split-K with atomic adds (order not deterministic) | never |
| SplitMaskedSoftmax | 256 threads for one row | always 64 |

256 rows covers every decode and verify pass (up to 16 tokens x 16 query heads
of a global layer). Above that (page prefill) the fast paths stay. The
`MatMulNBits` weights path was already one program for 1 to 16 rows. Norms,
elementwise ops and rotary do not depend on the row count.

Checks on this Mac with `ort-fork-dist-inv`: 320 of 320 attention cases
identical; MTP equal to greedy from one cache on mathfuncs (420 tokens) and
openstax (600), with the near-tie guard on and off. The Intel and subgroup-matrix
paths could not run on this Mac. Since 2026-09-28 they are gone: see
"GPU-neutral kernels" below. The load-time self-check
was removed (user decision), and the near-tie guard is off by default (it only
cost speed: mathfuncs 10.9 against 12.5 tok/s).

### Decode check with the fixed build [Mac]

Greedy and MTP (guard on) on the two pages that differed, 1,080 tokens each:

| Page | MTP vs greedy, old build | MTP vs greedy, fixed build |
|---|---|---|
| openstax | differs at 1,071 | identical (1,080) |
| long page | differs at 119 | identical (1,080) |

The fixed build changes greedy's rounding, so greedy output can change at near
ties. On the long page the first token changed ("# MALAY STATES" became "MALAY
STATES"). Its top-2 gap under the new build is 0.25, so it is a near tie. The
plan counts this as exact: no weight change and no approximation of the math.
The gate reference is re-taken with the fixed build (`gate-binv.json`).

### A third cause: results depend on the session's history [Mac]

The 3-run check across processes still failed on mathfuncs (index 351, all 3
runs). Probes (all in `scripts/gemma12b-engine-check.mjs`):

| Probe | Result |
|---|---|
| `rowcheck`: one row as a 1-row step, as row 3 of a 9-row pass, and after a rollback | bit-identical (0 differing logits at 3 positions) |
| `draftercheck`: cache before and after 3 MTP draft plans | 0 of 192 tensors changed |
| `prefixcheck`: kept prefix before and after page runs | unchanged; repeated greedy runs identical |
| `repeat`: image features and prefill logits, same page twice in one session | identical |
| the same, across processes with different loads | features differ |
| vision graph alone (`lowmem-visioncheck.html`), cold, warm, after other GPU work | identical |

Features across two loads (with and without the load's warm-up): 0.08% of
values differ, by at most 0.031 (1 to 2 float16 steps). This carries into the
prefill logits (up to 0.75). With no buffer reuse at all (`BUFFER_CACHE=
storage=lazyRelease,uniform=lazyRelease,default=lazyRelease`) it still happens,
so stale memory is not the cause. The vision graph alone is deterministic, so
the effect needs the full engine's history. The kernel is not found yet
(open item). It is rounding-level, which the plan counts as exact, but it makes
comparisons between processes unfair.

### C1 result: MTP vs greedy from one cache [Mac]

`STEPS=specvsgreedy`: the page is prefilled once; greedy and each MTP run decode
from forks of that cache, so all start from the same bits.

| Page | Greedy tokens | MTP runs (guard on) | Guard off |
|---|---|---|---|
| smoke | 25 | 3 of 3 identical | |
| mathfuncs | 352 | 3 of 3 identical | |
| openstax | 1,173 | 3 of 3 identical | |
| long page (to 1,500 tokens) | 1,500 | 3 of 3 identical | identical |

Speed in these runs: MTP about 20 tok/s on smoke, 24 on mathfuncs, 16.8 on
openstax, 10.9 on the long page (greedy 11.1, 11.4, 9.4, 9.2).

## Decoding: the model's recommended sampling [Mac]

User decision (2026-09-25): decode with the model's own settings, not greedy.

| Setting | Value | Source |
|---|---|---|
| temperature | 1.0 | `generation_config.json` |
| top_k | 64 | `generation_config.json` |
| top_p | 0.95 | `generation_config.json` |
| soft cap | 30 | `config.json` (`final_logit_softcapping`) |
| seed | 0 | engine |

Why: greedy decoding looped on page 4 of a user PDF. It wrote a footnote rule
as `\_` without end, up to the token cap. llama.cpp loops the same way at the
same input, and turning off the sliding trim changed nothing. This is the
known greedy failure mode ("neural text degeneration"). With sampling, page 4
ends by itself after 1,074 tokens and the text is correct.

How it stays repeatable and exact:

- Output position i draws one number `positionUniform(seed, i)`. The token at
  a position is a fixed function of that position's logits. The same page
  gives the same text every time (page 4: the same 1,074 token ids in two
  runs, two sessions).
- MTP keeps the output. A verify row and a one-token step at the same
  position see the same logits (C1) and draw the same number.
- The hybrid head samples the exact 8-bit logits. It stops checking rows once
  the top-p set is certain. Bounds on the top-64 weight sum decide that. Then
  it runs the same pick code as the full computation.
- `sampling: null` (engine option; harness `GREEDY=1`) gives argmax. Greedy
  references such as `gate-binv.json` need `GREEDY=1`.

Same-cache check (`STEPS=specvsgreedy`, both runs sampled, streamed path,
hybrid head, cap 600):

| Page | Tokens | MTP = one-token | MTP tok/s | Head ms a token |
|---|---|---|---|---|
| mathfuncs | 419 (end token) | yes | 7.99 | 27 |
| openstax | 600 (cap) | yes | 10.08 | 17 |

(Head times measured before the last two head speed-ups below.)

Page 4 (engine check, gate, streamed path, MTP, hybrid head):

| Build | Tokens | Stop | Decode tok/s | Head time |
|---|---|---|---|---|
| greedy | 3,000 | cap (loop) | 13.5 | |
| sampled, first head | 200 of a run | cap | 2.2 | 349 ms a token |
| sampled, head stops at a certain top-p set | 1,074 | end token | 7.9 | 27.6 s |
| sampled, heap and bound-first scans | 1,074 (same ids) | end token | 8.8 | 12.7 s |

App through `npm start` (page 4 of the user PDF, `scripts/gemma12b-app-check.mjs`
with `APP_URL=http://127.0.0.1:4173`, service workers on, Convert 20 s after
the page opens; `evidence/lowmem/app/npmstart-sample-p4*.json`):

| Run | First text after Convert | Page saved | GPU peak | Text |
|---|---|---|---|---|
| fresh profile (model copied first) | 74 s | 215 s | 7.07 GB | ends by itself, no loop |
| same profile again | 7.9 s | 144 s | 6.96 GB | the same text as run 1 |

The app text equals the engine check's text character for character. No
console errors. On the second run the model is still loading when Convert is
clicked, so page 1 is not prefetched yet (the greedy build: 5.4 s on
mathfuncs in the same setup).

Exact rows per token on page 4: 673 tokens needed 128 rows or fewer, 180
needed up to 1,024, 159 up to 8,192, 62 more. The last group is most of the
head time: those tokens have a wide top-p set, and the 4-bit bounds are
loose. Speeding up the exact row itself (WASM SIMD) is the next step if the
head matters.

## B2: the hybrid LM head [Mac]

`decoder.hb.onnx` + `lmhead.json`, engine option `lmHybrid`, worker build flag
`VITE_LM_HYBRID=1` (the 8-bit table is copied once into OPFS and read by row).

| Page | GPU peak, GPU head | GPU peak, hybrid | Greedy tok/s, GPU head | hybrid |
|---|---|---|---|---|
| smoke | 9.52 GB | 9.22 GB | 11.4 | 10.2 |
| mathfuncs | 9.73 GB | 9.32 GB | 11.1 | 9.3 |
| openstax | 9.73 GB | 9.42 GB | 8.6 | 8.0 |
| long page | 9.73 GB | 9.42 GB | 8.4 | 8.0 |

(Speeds measured before the re-check was made faster: it cost 6 to 9 ms a
token then, about 2 ms now.)

- Hybrid greedy differs from GPU-head greedy at mathfuncs index 351 (hybrid
  gap 1.5) and long page index 279 (gap 0.013). The re-check works in float64 on
  the same 8-bit weights; the GPU head computes its dot products in float16
  (fork comment in `matmul_nbits.cc`). Both are the same math, rounded
  differently.
- MTP on the hybrid head is identical to hybrid greedy (same-cache check,
  mathfuncs and openstax, guard on and off).
- With MTP the guard asks for a gap on every verify row. An exact gap cost ~327
  exact rows a call; the guard now reads a lower bound instead (`boundOnly`):
  ~64 rows, ~2 ms a call. A stop only defers the token to an exact one-row step.

## App run: MTP + hybrid head [Mac]

`VITE_SPECULATIVE=mtp VITE_LM_HYBRID=1` build, `CONVERT_AT_MS=30000`
(evidence/lowmem/app/). Run before the bound-only guard.

| Page | Tokens | Decode tok/s | First text after Convert | GPU peak |
|---|---|---|---|---|
| smoke | 25 (eos) | 12.0 | | 9.42 GB |
| mathfuncs | 352 (eos) | 16.4 | 10.7 s | 9.63 GB |

- The re-check took 5.8 s of mathfuncs' 21.5 s decode (fixed since, above).
- First text came 10.7 s after Convert: the page-1 prefill never ran. Two
  causes, both fixed (commit 47354dd): the screen hook had been edited in the
  generated `public/screens/` (overwritten by `scripts/prepare.mjs`), and a PDF
  chosen before the load started was dropped (`src/prefetch-gate.js` now keeps
  it). After the fix: first text 1.3 s (smoke) and 1.6 s (mathfuncs) after
  Convert; mathfuncs decode 19.3 tok/s with the bound-only guard.
- The prefetched mathfuncs run went past index 351 (420 tokens, a figure block
  at the end) where the earlier app runs stopped: the same near tie, decided by
  the session-history effect above, not by the prefetch path (the text is
  identical up to that point).

## B1: the global layers' cache in bands [Mac]

Each KV merge made slightly larger global tensors, so a freed one never fit
the next request and ORT's buffer cache grew with the page. Engine option
`globalBand` (kvNames, PastCache.extra, kv-merge.js `global_start`): a global
tensor holds a whole number of bands, with the unused slots masked at the
front. Checks: a CPU run of the merge graph (onnxruntime-node) gives the
expected layout; the fake engine gives the same ids with 7 tensor sizes
instead of 25; on the GPU, MTP equals greedy from one cache (mathfuncs,
openstax to 600 tokens).

| Long page, canonical path | Tokens | GPU footprint / peak | MTP tok/s |
|---|---|---|---|
| no band | 2,741 | 6.96 / 7.07 GB | 4.75 |
| band 1,024 | 2,720 | 6.86 / 6.96 GB | 4.47 |

The band saved about 0.1 GB and the peak still grew, so another source of
growth remains (not found). One run each; the speed difference may be noise.
The band is also not bit-identical to the unbanded cache: the masked front
slots add exact zeros, but they move the real slots to other columns, so the
softmax and P·V sums group them differently (2,720 against 2,741 tokens on the
long page). Greedy and MTP stay identical to each other with or without it.
The app does not use the band.

The vision session stays loaded (0.15 GB). Freeing it after each encode would
cost a reload per page (0.14 to 0.34 s), and in bucket mode its freed buffers
likely stay in ORT's cache, so the GPU footprint may not drop [E, not measured].

Buffer-cache modes on the canonical path (openstax, 600 tokens): bucket
6.66 / 6.76 GB, 13.9 tok/s (kept); simple 10.5 GB; lazyRelease 5.2 GB at rest
but a 9.0 GB peak.

## B3 / B4: the decoder in parts, and weights streamed through a GPU ring [Mac]

`tools/gemma12b/split_decoder.py` cuts `decoder.onnx` into consecutive layer-group
graphs that read the same weight shards (no weight bytes copied). With
`--stream-parts`, a part's layer weights become graph inputs, and
`decoder.stream.json` lists where their bytes are. `loadGemma12b` options
`decoderParts` and `streamParts` run the parts in order (the residual stream is
passed on the GPU) and write streamed weights into one of two GPU buffer slots
before each run (`Tensor.fromGpuBuffer`, as A3 showed).

B4 check (`evidence/lowmem/b4-stream2.json`): layers 0 and 1 streamed (123 MB
each, 246 MB a token), layers 2-23 and 24-47 resident, budget 1120, greedy.

| Page | Tokens | Same ids as the Stage 0 gate | Decode |
|---|---|---|---|
| smoke | 25 | IDENTICAL | – |
| mathfuncs | 352 | IDENTICAL | 6.4 tok/s (baseline 11.1) |

- The output is exact: streaming changes where the weights live, not the math.
- B3 alone (the plain cut at layer 24, 2 parts, nothing streamed;
  `evidence/lowmem/b3-split24.json`): IDENTICAL on smoke and mathfuncs, but
  mathfuncs decodes at 8.8 tok/s (113 ms a token against about 90 ms whole).
  So each part boundary costs about 20 ms a token. Suspects: the mask and
  position sub-graphs that every part rebuilds (some of their nodes run on the
  CPU) and per-run overhead. One part per streamed layer would be far too slow;
  this must be profiled and cut before the fallback relies on many parts.
- Decode is 66 ms a token slower. The 246 MB upload explains about 20 ms at
  12 GB/s; the rest is the four parts' per-run overhead. Not yet profiled.
- This test build holds the 1.07 GB token table twice (part 0 for the embedding
  lookup, part 3 for the LM head), so its GPU peak (10.85 GB) is not a memory
  result. The real build must share one copy (a graph input fed from one GPU
  buffer, as A3 showed) or move it to the CPU (the hybrid LM head).

## C2: the MTP drafter in the app engine [Mac]

`tools/gemma12b/mtp_drafter.py` builds `drafter.onnx` (263 MB, 4-bit) from the QAT
assistant GGUF; `tools/gemma12b/mtp_drafter_check.py` checks it on the CPU against
a numpy reference. The target runs as one part with a hidden-state output
(`decoder.h0.onnx`, the same weights). `src/gemma12b/mtp-drafter.js` drafts from the
target's hidden row and its layer 46/47 cache; the embedding row of each drafted
token is read on the CPU from the cached decoder shard. The near-tie guard is on
(ε = 1.0). Build: `VITE_SPECULATIVE=mtp`.

Engine check `STEPS=spec SPEC_DRAFTER=mtp` (`evidence/lowmem/c2-mtp.json`), budget
1120, PNG pages, greedy then speculative in one session:

| Page | Same ids as greedy | Tokens per target pass | Drafts accepted | Decode, MTP | Decode, greedy | Near-tie stops |
|---|---|---|---|---|---|---|
| mathfuncs (first page of the session) | IDENTICAL 423/423 | 3.92 | 73% | 2.8 tok/s | 9.7 tok/s | 22 of 108 passes |
| openstax (second page) | IDENTICAL 1,183/1,183 | 4.08 | 66% | **14.1 tok/s** | 7.7 tok/s | 41 of 290 passes |

- On mathfuncs the first verify passes of each new row count took about 2 s
  (shader compilation). On the second page every shape was compiled: 1.8x.
- A draft step costs about 27 ms (38.7 s for 1,409 drafts on openstax). The
  target pass costs 105 ms with 1 row and 390 ms with 9 rows on this GPU.
- Rerun with the verify shapes compiled at load (`engine.loadStats.warmUp`: 8
  shapes and one drafter step, 1.6 s), `evidence/lowmem/c2-mtp-warm.json`:

  | Page | Same ids as greedy | MTP | Greedy | Speed-up | Draft step | Near-tie stops |
  |---|---|---|---|---|---|---|
  | mathfuncs | IDENTICAL 423/423 | 18.4 tok/s | 11.1 tok/s | 1.66x | 19.8 ms (0.7 embedding read) | 22 of 108 |
  | openstax | IDENTICAL 1,183/1,183 | 13.8 tok/s | 7.7 tok/s | 1.79x | 25.5 ms (2.8 embedding read) | 41 of 290 |

- The draft step is mostly the drafter session's run (per-run overhead; its
  weights are only 0.27 GB). On a bandwidth-bound laptop GPU the target pass
  dominates, so this matters less there. Estimate for Iris Xe: target pass about
  220 ms, 4 draft steps about 60 ms, about 3.9 tokens a pass: about 14 tok/s [E].
- Ideas to cut the draft step later: run all k draft steps in one session call
  (needs the target's token table as a shared GPU input); fewer near-tie stops
  with a smaller ε (the largest measured difference was 0.41).

## GPU-neutral kernels (2026-09-28)

A Surface Laptop 7 (Windows, Qualcomm Adreno) wrote garbage from the first
character with Small, and froze. The fork let kernels pick programs by GPU
brand and subgroup width, so that laptop ran programs no Mac had run:

| Kernel | Apple | Qualcomm (before) |
|---|---|---|
| Flash attention (E-model vision) | shared-memory path, 128-lane prefill tile | subgroup path, Qualcomm output tile, 64-lane tile |
| Wide-tile MatMulNBits (prefill) | `subgroupShuffle`, min size 32 | `subgroupShuffle`, min size 64 |
| MatMul above 256 rows | generic | generic (Intel: Intel program) |

The fork (tree `~/BlinkFix-models/tmp-g12/ort-fork-src`, branch `gpu-neutral`,
commit 7fab419) now:

- gives kernels an empty adapter info (`WebGpuContext::AdapterInfo()` returns
  `kernel_adapter_info_`),
- creates the device without `Subgroups`, `SubgroupSizeControl` and
  subgroup-matrix, so every kernel takes its non-subgroup branch,
- clamps the limits kernels choose from to the WebGPU defaults
  (`GpuNeutralLimits`: 16384 B workgroup storage, 8 storage buffers, 256
  invocations).

`scripts/build-ort-fork.sh` refuses a reference patch that loses any of this
(check GN). Bits can still differ in the last places between GPU compilers
(Metal uses relaxed math); that changes a token only at a near tie.

Checks on the M5 (`evidence/gpu-neutral/`):

| Check | Old dist | Neutral dist |
|---|---|---|
| `lowmem-batchinv.html?model=e4b` | 528 / 528 | 528 / 528 |
| `lowmem-batchinv.html` (12B attention) | 92 / 92 | 92 / 92 |
| E4B vision + decoder vs CPU reference, 8 positions | 8 / 8 top-1 | same logits, bit for bit |
| E4B prefill / decode step | 5.9-6.3 s / 36-38 ms | 6.0-6.1 s / 36-37 ms |
| App, handwritten 2-page PDF | - | Small, Medium, Large all readable |

Longest single dispatch on the M5: E2B vision flash attention, about 137 ms
(16 per page). The Surface check (`gpu-check.html`) records it there.


## Windows GPU: hang, lag and garbage on an Intel gen-9 laptop (2026-09-28)

Plan: `~/.claude/plans/so-i-tried-concurrent-wind.md`. Test machine: my
HP laptop (Intel gen-9 integrated GPU, 16 GB, Chrome 154, D3D12), driven remotely from
the Mac. Evidence: `evidence/windows-hp/`, `evidence/f16-read/`.

### Tools

| Tool | What it does |
|---|---|
| `?gpuTrace=1` (`src/gpu-trace.js`) | Times every GPU submit in the model worker; logs long submits, GPU loss with the submits in flight, and run counts. `?gpuTracePost=1` sends the lines to `npm start`'s collector (`scripts/gpu-trace-collector.mjs`). |
| `?gpuProfile=1` / `=decode` | ORT profiler: every dispatch of a session's first long run, or per-program GPU time up to the 20th decode step. |
| `gpu-check.html?imageBudget=280` | The GPU check at a small page image, so a slow GPU stays under the Windows 2 s limit. Reference `src/gpu-check/ref-e2b-280.json` (M5). The verdict now reads the real fields (it said "matches" for every run before). |
| `gpu-check.html?taps=1` | Small's vision graph with every layer-0 op and every attention/norm output as an extra output (`tools/gpu-debug/vision_taps.py`); `tools/gpu-debug/compare_taps.py` compares two machines op by op. |
| `tools/gpu-debug/f16-store-test.html` | 19 tiny WGSL kernels: which ways of storing or loading f16 lose values on this GPU. |

`?gpuSubmit=N` cannot reach ORT: ORT web builds the WebGPU context from the EP
options before it reads session config, and has no option for
`maxNumPendingDispatches`. Every run above used 16 dispatches per submit.

### Garbage text: odd f16 elements read as 0

On the HP, Small wrote random words in many scripts. The Mac wrote the page.

| Step | Result |
|---|---|
| Vision taps, HP vs M5 | first wrong op: patch embedder position `Gather_22/23`; every odd f16 column read as 0 (243 of 384; the rest are 0 in the table). `MatMul_7` bit-exact. |
| `f16-store-test` | stores exact; picking one element out of a 16-bit storage load (scalar, vec2/vec4 with an index, or `.y`) gives 0 for odd elements; whole-vector use, workgroup staging, and u32 loads + `bitcast` are exact. |
| Fix (fork branch `gpu-f16-read`, ee2ba08) | scalar f16 inputs are `array<u32>`; every f16 getter bitcasts a 32-bit value (`shader_variable.cc PackedF16Get`); `Where`'s broadcast path reads through the getters. Every GPU runs these programs. `build-ort-fork.sh` refuses a patch without it (F16R). |

| Check | Result |
|---|---|
| HP `gpu-check.html?imageBudget=280` | before: vision cosine median 0.45, 0/8 top-1; after: 0.999996, 8/8, "matches the Mac" |
| HP app, Small at 560 | writes the page (same kind of sampling slips as the M5 at 560) |
| M5 vision taps, new vs old fork | 141/141 bit-identical |
| M5 `lowmem-batchinv.html?model=e4b` / 12B | 528/528, 92/92 |
| M5 gpu-check at 280 vs the old-fork reference | 8/8 top-1; probability difference ≤ 0.009 (last-bit rounding in a few decoder kernels) |
| M5 GPU time, same work (`?gpuProfile=decode`) | decoder +2.3%, drafter +4.2%; small scalar-read kernels slower (Cast x1.45, Concat x2.4, Tile x2.2) |

No public report of this bug was found (onnxruntime #29807 and #24442 report
wrong output on Intel GPUs without a cause).

### Hang and lag: GPU submits too long for a slow GPU

| Run (HP, Small) | Result |
|---|---|
| imageBudget 1120 | `DXGI_ERROR_DEVICE_HUNG` in vision: one 16-dispatch submit with the flash-attention dispatch ran over 4.5 s. Chrome then gave no WebGPU adapter to any site until it restarted. |
| imageBudget 560 | no loss; the 16 vision submits took 1.46-1.90 s each (under the 2 s limit) |

HP/M5 time per program (Small at 280, profiled, matched dispatch by
dispatch): vision flash attention K 48 (367 ms vs 7.7 ms), vision fp16 MatMul
20, prefill fp16 MatMul 17, `MatMulNBitsWideTile` 5-7. At 1120 one
attention dispatch (M5 ~137 ms) takes about 6.6 s on the HP.

The page's frame-gap probe (rAF) saw 18 ms gaps while 400 ms GPU submits ran,
so it cannot see this lag; screenshots of the HP are the evidence instead.

### Phase 2: slices, own submits, and a bound on submits in flight

Fork branch `gpu-slices` (from `gpu-f16-read`). Pinned on 2026-09-29 at
54634e0 (dist `ort-fork-dist-slices`); see "Phase 2 result" below. The
checkpoint notes that follow are kept as written:

| Commit | Change |
|---|---|
| c81b25e | flash-attention prefill split into slices of whole workgroups, at most 2^31 query·key·head-size products each, each in its own submit (`ProgramBase::SetOwnSubmit`). Output bit-identical (M5 taps 147/147). |
| 8f7b7a9 | `MatMul` and `MatMulNBits` dispatches with at least 2^31 multiply-adds submitted on their own |
| 0197db4 | diagnostic overrides read with quoted names (the ORT glue is Closure-minified; unquoted `globalThis` names were renamed, so earlier override runs were invalid) |
| 4f78d42 | `Flush` waits for the oldest submit when more than 2 are in flight |

Page flags (measurement only): `?gpuSliceLog2=N`, `?gpuInFlight=N`,
`?gpuSubmit=N`. The page's GPU wait probe (a canvas readback every 2 s) shows
how long Chrome's own GPU work waits.

| HP, Small | Pinned f16r | gpu-slices 4f78d42 |
|---|---|---|
| 1120 | `DEVICE_HUNG` in vision | no loss; longest submit 390 ms (bound not yet tested at 1120) |
| 560 vision wall (unprofiled) | 20.5 s | 37.4-37.7 s |
| 560 GPU wait, model load | 11.8 s | 20 ms |
| 560 GPU wait, vision | 19.9-32.7 s (133 s at 1120) | 46-59 ms |

Profiled runs are excluded from the cost attribution: timestamp queries at
every pass boundary cost 1.8x on the HP by themselves (pinned f16r profiled at
560: 36.3 s). Unprofiled: attention slices alone +11 s, all splitters +17 s,
not additive; the cause is not identified yet.

One attention workgroup at 1120 walks all 10080 keys (~190 ms on the HP), so
no slice can be shorter without changing the reduction order (not allowed:
exact output). The plan's 50 ms target is replaced by the GPU wait number the
user accepts.

### Phase 2 result (2026-09-29): pinned, no measurable cost on the Mac

The final constants are the same on every GPU:

| Constant | Value |
|---|---|
| Flash-attention slice | 2^32 query·key·head-size products, own submit |
| Matmul own submit | 2^31 multiply-adds or more |
| Submits in flight | 16 |

The "Mac cost" measured earlier was a measuring error. The vision and prefill
walls end when `run()` returns, and `run()` returns before the GPU finishes
when ORT queues work ahead. A Dawn trace (`CATS=... app-trace.mjs`) and a
macOS `sample` of Chrome's GPU process show this:

- At 4 dispatches per submit, Chrome's GPU main thread waits 8.1 s in
  `-[MTLCommandQueue commandBuffer]` for one of Metal's 64 command-buffer
  slots. So the GPU is the bottleneck, and Chrome only waits for it.
- In 0.5 s buckets, the GPU is busy from vision start to prefill end at both
  16 and 4 dispatches per submit.

The honest measure is the Convert click to "page done" (`app-ab.mjs`,
`pageMs`; headless Chrome on the M5, Small, `scan-mathfuncs.pdf`):

| Build | Round 1 | Round 2 |
|---|---|---|
| Pinned f16r | 73.1 s | 71.0 s |
| gpu-slices, final constants | 75.0 s | 71.0 s |
| gpu-slices, all splitters and the bound off, 16 per submit | 69.8 s | 71.1 s |
| same, 4 per submit | 71.0 s | 71.4 s |

The differences stay inside the run-to-run noise. The HP "vision wall" costs
in the table above have the same flaw.

Research (2026-09-29): Windows resets the GPU when the driver cannot pause the
work within about 2 s. The driver decides where it can pause (between submits,
between dispatches, or finer), and this differs by machine. The only pause
point that every driver respects is a submit boundary, so the slices keep
their own submits. Sources: Microsoft's TDR and GPU preemption pages. Other
browser runtimes use a fixed count of dispatches per submit (ORT 16, TF.js 15,
TVM 32, llama.cpp 64). None of them splits one long dispatch. WebLLM has an
open hang report on an Intel UHD 630 (web-llm #489).

Gates on the M5 with the pinned slices dist: `lowmem-batchinv.html?model=e4b`
528/528, `lowmem-batchinv.html` (12B) 92/92, `npm test` 1196 pass.


## tile9 and key step 8 (2026-09-30): pinned for the GLM-OCR engine

Pinned dist: `ort-fork-dist-tile9k8` (fork branch `mnb-tile9` on `gpu-slices`).

| Commit | Change |
|---|---|
| 0c8a61d | The float `MatMulNBitsBatched` kernel keeps tiles of up to 9 rows. Rows 10-16 take a second tile. Before: 16 register accumulators in one tile. |
| 0176080 | Flash-attention prefill with 32-bit inputs steps 8 keys at a time. 16-bit inputs (Gemma) keep their step. |

Why: on the Intel gen-9 HP the 16-accumulator kernel spilled registers. One GLM-OCR decode step took 227 ms. The image reader's attention (f32, 16 heads x 64, 2464 tokens) was most of the 41 s image time.

Matmul bench (GLM-OCR decoder weights, 363.7 MB, f32, ms per pass by row count):

| GPU | Tile | 1 row | 3 | 5 | 7 | 9 |
|---|---|---|---|---|---|---|
| Intel gen-9 | 16 (before) | 125.8 | 132.5 | 158.0 | 171.0 | 193.3 |
| Intel gen-9 | 9 (pinned) | 38.0 | 49.4 | 63.8 | 76.1 | 90.6 |
| Apple M5 | 16 (before) | 6.7 | 7.9 | 8.8 | 9.0 | 9.6 |
| Apple M5 | 9 (pinned) | 4.3 | 4.1 | 5.3 | 6.4 | 7.5 |

Attention bench (one image-reader layer, ms):

| GPU | Key step 32 (before) | Key step 8 (pinned) | Largest difference |
|---|---|---|---|
| Intel gen-9 | 1061.9 | 483.6 | 1.0e-7 |
| Apple M5 | 28.5 | 22.5 | 1.5e-7 |

GLM-OCR engine, one page (`twocol.png`, 342 tokens, 3 drafts per pass):

| GPU | Image reader | Prompt | Decode |
|---|---|---|---|
| Intel gen-9, tile9 only | 41.3 s | 5.5 s | 13.7 tok/s |
| Intel gen-9, tile9k8 | 28.3 s | 5.0 s | 19.5 tok/s |
| Apple M5, tile9k8 | 1.9 s | 0.6 s | 215.9 tok/s |

Bits: the Gemma path is unchanged. Gates on the Apple M5 with this dist: `lowmem-batchinv.html?model=e4b` 528/528, 12B 92/92. The key-step change moves the online-softmax rescale points, so f32 attention may differ in the last bits. Output of the GLM-OCR engine with drafts equals plain decoding token for token on both machines.

Sources: `~/BlinkFix-models/glm-ocr-engine/results/` (`mnb-*-tiles-f32.json`, `attnbench-*-k.json`, `engine-*8-tile9k8-summary.json`), gate files in `~/BlinkFix-models/ort-fork-dist-tile9k8/`.

## ovis-speed (2026-10-01): pinned for the OvisOCR2 engine

Pinned dist: `ort-fork-dist-ovis` (fork branch `ovis-speed` on `mnb-tile9`, commit 5297cec).
The wasm build-info string names cb6d3a7: the tree was built just before the final commit, whose content it equals (`BUILD-NOTE.txt` in the dist).

| Change | What it does |
|---|---|
| LinearAttention `state_window` on WebGPU | The op keeps the recurrent state after each of the last W tokens (W <= 16). State tensors are (W, B, H, dk, dv). The op reads only slot W-1 of the past state. |
| Chunked LinearAttention prefill | A pass of more than 16 tokens runs two programs (`LinearAttentionChunkPrep`, `LinearAttentionChunkScan`) instead of one token after another. With a state window, the last W tokens still take the token loop. Passes of 1-16 tokens keep the token loop. |
| Tiled fp16 attention (opt-in) | A session with the config entry `webgpu.flash_attention_tiled=1` runs the 16-bit, head-64 MultiHeadAttention prefill as `FlashAttentionTiled`. Only the OvisOCR2 image reader sets it (lock `runtime.visionAttention: 'tiled'`). Every GPU runs the same program. |

Default programs are unchanged. Gates on the Apple M5 (JSPI): `lowmem-batchinv.html?model=e4b` 528/528, 12B 92/92 (`~/BlinkFix-models/ort-fork-dist-ovis/gates/`).

| Check | Result |
|---|---|
| One LinearAttention node, 1-16 tokens with a state window | bit-identical rows (`la-test.html`) |
| Chunked prefill vs float64 reference | largest difference 3e-8 |
| 20-token greedy text, chunked prefill + tiled image reader | same as the tile9k8 dist (`hp-prefill-ovis-tiled.json`) |

Intel gen-9 HP, OvisOCR2 1 MP page:

| Part | tile9k8 | ovis-speed |
|---|---|---|
| Image reader, fp16 | 20.3 s (attention 12.9 s) | 14.6 s with the tiled opt-in (attention 7.7 s) |
| Prompt step, 1099 tokens, q4 | 11.5 s (LinearAttention 3.0 s) | 9.9 s (LinearAttention 0.93 s) |
| Decode step q4 | 104-117 ms | same |

Sources: `~/BlinkFix-models/ocr-probes/results/` (`hp-qv-pinned-prof.json`, `hp-qv-tiled-v2.json`, `hp-prefill-pinned.json`, `hp-prefill-ovis1.json`, `hp-la-test.json`), `~/BlinkFix-models/ocr-probes/README.md` section 11.
Known latent bug (default FlashAttention, not new): a nonzero scalar f16 attention bias is read wrong. No model in the app hits it. The tiled program reads it right.

## ovis-speed2 (2026-10-01): two opt-in decode programs for the OvisOCR2 decoder and drafter

Pinned dist: `ort-fork-dist-ovis2` (fork branch `ovis-speed2` on `ovis-speed`, commits 5854696 and 7f33d0f; `BUILD-NOTE.txt` in the dist).
`reference/onnxruntime-webgpu.patch` is the dist's `SOURCE.patch` (v1.30.0..ovis-speed2).

| Session config entry | Program | What it does |
|---|---|---|
| `webgpu.matmulnbits_block=1` | `MatMulNBitsBlock` | q4, block 32, f32 MatMulNBits for 1-16 rows. One thread takes one whole 32-value block per step. |
| `webgpu.linear_attention_decode=1` | `LinearAttentionDecode` | LinearAttention passes of 1-16 tokens run as a gated-delta token loop. |

Both are off unless the session sets the entry. Default programs are unchanged, so the Gemma path keeps its bits. Gates on the Apple M5 (JSPI): `lowmem-batchinv.html?model=e4b` 528/528, 12B 92/92 (`~/BlinkFix-models/ort-fork-dist-ovis2/gates/`).

The app turns both on for the OvisOCR2 decoder and drafter sessions only (not the image reader, not the embedder). The switch is the lock's `runtime.decoderKernels: ['matmulnbits_block', 'linear_attention_decode']`; `decoderSessionExtra` in `src/ovis-ocr/decode.js` maps the names to the entries. Same on every GPU.

Intel gen-9 HP, OvisOCR2 real engine (1 MP, q4), token ids identical to the ovis build:

| Part | ovis | ovis-speed2 |
|---|---|---|
| q4 matmuls per decode step | 66 ms | 41 ms (output head 8.1 -> 15.2 GB/s) |
| K=0 decode step | 112 ms (8.9 tok/s) | 79 ms (12.6 tok/s) |
| K=2 verify pass | 197 ms (12.3 tok/s) | 175 ms (13.7 tok/s) |
| First token | about 24 s | unchanged |

Sources: `~/BlinkFix-models/ocr-probes/README.md` section 13, `results/hp-engine-before.json`, `results/hp-engine-block-ladec.json`.
