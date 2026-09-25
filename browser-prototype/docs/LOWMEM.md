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

## Phase A0: image budget 560 vs 1120 — text accuracy [Mac]

Native llama.cpp (the same QAT model, the app's prompts, greedy, thinking off,
image tokens fixed at the budget, 4 pages at a time), `scripts/lowmem-a0-llama.mjs`.
Scored with the pinned olmOCR scorer (`f7cfe4c`), `tools/lowmem/a0_score.py`.
Files: `evidence/lowmem/a0/llama-budget{560,1120}.json`, `score-summary.json`.

Subset, fixed before any result (the user's 15-minute rule cut it twice): the
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
- First text came 10.7 s after Convert: the page-1 prefill did not help in
  this run. Not investigated yet (the harness may choose the file only at
  Convert time).

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
