"""Compare two gpu-check.html?taps=1 results op by op (Windows GPU plan).

  python tools/gpu-debug/compare_taps.py <reference log> <other log>

Each log holds a 'GPUTAPS <json>' line (scripts/gpu-trace-collector.mjs).
For every tap, in graph order: the largest absolute difference, that
difference relative to the reference's largest value, and the cosine of the
sampled rows. The first tap far from the reference names the op that breaks.
"""
import base64, json, sys
import numpy as np

def load(path):
    line = [l for l in open(path) if l.startswith('GPUTAPS ')][-1]
    return json.loads(line[len('GPUTAPS '):])

DT = {'float16': np.float16, 'float32': np.float32}

def rows(tap, key):
    return np.frombuffer(base64.b64decode(tap[key]), dtype=DT[tap['type']]).astype(np.float64)

ref, oth = load(sys.argv[1]), load(sys.argv[2])
other = {t['name']: t for t in oth['taps']}
print(f"{'tap':70} {'max|ref|':>10} {'max|diff|':>10} {'rel':>8} {'cos':>9} nan")
for t in ref['taps']:
    o = other.get(t['name'])
    if t['type'] not in DT or not o:
        continue
    a = np.concatenate([rows(t, 'head'), rows(t, 'middle')])
    b = np.concatenate([rows(o, 'head'), rows(o, 'middle')])
    bad = int(np.sum(~np.isfinite(b)))
    fa, fb = np.nan_to_num(a), np.nan_to_num(b)
    diff = float(np.max(np.abs(fa - fb))) if len(a) else 0.0
    top = float(np.max(np.abs(fa))) if len(a) else 0.0
    cos = float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-30))
    flag = '  <==' if (cos < 0.999 or bad) else ''
    print(f"{t['name'][-70:]:70} {top:10.4g} {diff:10.4g} {diff / (top + 1e-30):8.2g} {cos:9.6f} {bad}{flag}")
