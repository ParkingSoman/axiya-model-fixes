"""Make a copy of a Gemma 4 E2B/E4B QAT GGUF with per_layer_model_proj.weight scaled.

Why: in Google's QAT checkpoints (E4B and E2B) this F16 tensor is about 64x smaller
than in the base checkpoint (E4B RMS 0.000384 vs 0.02476). The runtime multiplies its
output by hidden_size**-0.5 and applies an RMSNorm with eps 1e-6. With the QAT weights
the mean square before that norm is about 1e-6, so the eps shrinks the per-layer
embedding branch (most on image tokens). Scaling the weight by 64 moves it out of
the eps regime. In exact arithmetic the change is otherwise a no-op (the RMSNorm
removes the scale).

F16 (Google's files) and Q4_0/Q8_0 (e.g. Unsloth's UD-Q4_K_XL, where the block
scales are multiplied) are supported.

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
qt = gguf.GGMLQuantizationType(t.tensor_type)
if qt == gguf.GGMLQuantizationType.F16:
    rms = lambda: float(np.sqrt((t.data.astype(np.float32) ** 2).mean()))
    before = rms()
    t.data[...] = (t.data.astype(np.float32) * a.factor).astype(np.float16)
elif qt in (gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q8_0):
    # Block formats with one f16 scale d first in each block: scale d only (the
    # integer codes stay, so the file keeps the QAT lattice).
    bsize = gguf.GGML_QUANT_SIZES[qt][1]
    blocks = t.data.reshape(-1, bsize)
    d = blocks[:, :2].copy().view(np.float16)
    rms = lambda: float(np.sqrt((gguf.quants.dequantize(t.data, qt).astype(np.float32) ** 2).mean()))
    before = rms()
    new_d = (d.astype(np.float32) * a.factor).astype(np.float16)
    if not np.isfinite(new_d).all(): raise SystemExit('scale overflow in f16')
    blocks[:, :2] = new_d.view(np.uint8).reshape(-1, 2)
else:
    raise SystemExit(f'unsupported type {qt.name}')
r.data.flush()
after = rms()
print(f'per_layer_model_proj.weight ({qt.name}) RMS {before:.6f} -> {after:.6f} (x{a.factor})')
