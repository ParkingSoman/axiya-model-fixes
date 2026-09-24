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
