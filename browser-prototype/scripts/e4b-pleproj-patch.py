"""Make a copy of a Gemma 4 E2B/E4B QAT GGUF with per_layer_model_proj.weight scaled.

Why: in Google's QAT checkpoints (E4B and E2B) this F16 tensor is about 64x smaller
than in the base checkpoint (E4B RMS 0.000384 vs 0.02476). The runtime multiplies its
output by hidden_size**-0.5 and applies an RMSNorm with eps 1e-6. With the QAT weights
the mean square before that norm is about 1e-6, so the eps shrinks the per-layer
embedding branch (most on image tokens). Scaling the weight by 64 moves it out of
the eps regime. In exact arithmetic the change is otherwise a no-op (the RMSNorm
removes the scale).

Usage (never on the original file; the script copies first):
  python scripts/e4b-pleproj-patch.py SRC.gguf DST.gguf [--factor 64]
"""
import argparse, shutil
import numpy as np
import gguf

ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--factor', type=float, default=64.0)
a = ap.parse_args()
if a.src == a.dst:
    raise SystemExit('dst must differ from src')
shutil.copyfile(a.src, a.dst)
r = gguf.GGUFReader(a.dst, mode='r+')
t = next(t for t in r.tensors if t.name == 'per_layer_model_proj.weight')
if t.data.dtype != np.float16:
    raise SystemExit(f'expected F16, got {t.data.dtype}')
before = float(np.sqrt((t.data.astype(np.float32) ** 2).mean()))
t.data[...] = (t.data.astype(np.float32) * a.factor).astype(np.float16)
r.data.flush()
after = float(np.sqrt((t.data.astype(np.float32) ** 2).mean()))
print(f'per_layer_model_proj.weight RMS {before:.6f} -> {after:.6f} (x{a.factor})')
