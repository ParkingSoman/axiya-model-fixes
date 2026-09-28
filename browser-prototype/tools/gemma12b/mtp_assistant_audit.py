"""Audit a Gemma 4 QAT assistant (MTP drafter) checkpoint for the E4B QAT release defect.

The E4B QAT main checkpoint had one tensor about 64x too small (per_layer_model_projection).
The value before its RMSNorm (eps 1e-6) fell near eps, and the norm cut the branch. This tool
looks for the same fault in an assistant checkpoint:

1. Tensor scales: RMS of every tensor in the QAT assistant against the base (non-QAT)
   assistant. Informational: Google retrains the QAT assistant, so ratios far from 1 are
   normal. A tensor ratio alone does not prove a fault.
2. Starved norms (the decisive test): one fp32 forward of the assistant on REAL inputs, the
   target's final-norm hidden rows (llama-server --embeddings --pooling none on the fixed
   target GGUF) and the target's scaled token embeddings of the next ids. For every RMSNorm
   it records the mean square of its input. gain = sqrt(ms / (ms + eps)) is the factor the
   eps takes off. The defect in the main checkpoint showed gain 0.37 on image tokens. A healthy
   norm has gain 1.0000. K/V: random with the target's layer 22/23 key-norm scale (values unit
   RMS, as v_norm gives); they only reach the norms through o_proj.
3. Lattice: the Q4_0 GGUF against the QAT bf16 weights (relative error per tensor). QAT
   weights sit on the Q4_0 grid, so llama-quantize Q4_0 should be exact or nearly so.

  python mtp_assistant_audit.py --qat DIR --base DIR --target-gguf F --hidden H.npy --ids I.npy \
      [--q4 assistant-q4_0.gguf] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

EPS = 1e-6


def load(dir_: Path) -> dict:
    f = safe_open(str(dir_ / 'model.safetensors'), 'pt')
    return {k: f.get_tensor(k).float().numpy() for k in f.keys()}


def rms_rows(x):
    return np.sqrt(np.mean(x.astype(np.float64) ** 2, -1))


def forward_norm_stats(W: dict, cfg: dict, emb: np.ndarray, hid: np.ndarray, knorm: dict, rng) -> dict:
    """fp32 forward of the assistant over rows of (embedding, hidden); mean square before each RMSNorm."""
    t = cfg['text_config']
    heads, hd, ghd, kvh = t['num_attention_heads'], t['head_dim'], t['global_head_dim'], t['num_key_value_heads']
    stats = {}

    def norm(name, x, w):
        ms = np.mean(x.astype(np.float64) ** 2, -1)
        s = stats.setdefault(name, [])
        s.extend(ms.reshape(-1).tolist())
        return (x / np.sqrt(ms[..., None] + EPS) * w).astype(np.float32)

    x = np.concatenate([emb, hid], -1) @ W['pre_projection.weight'].T  # [R, H]
    R, T = x.shape[0], 64
    for i, kind in enumerate(t['layer_types']):
        p = f'model.layers.{i}.'
        d = hd if kind == 'sliding_attention' else ghd
        h = norm(f'l{i}.input_layernorm', x, W[p + 'input_layernorm.weight'])
        q = (h @ W[p + 'self_attn.q_proj.weight'].T).reshape(R, heads, d)
        q = norm(f'l{i}.q_norm', q, W[p + 'self_attn.q_norm.weight'])
        kw = knorm['sliding' if kind == 'sliding_attention' else 'global']
        K = rng.standard_normal((kvh, T, d)).astype(np.float32)
        K = K / np.sqrt(np.mean(K ** 2, -1, keepdims=True)) * kw
        V = rng.standard_normal((kvh, T, d)).astype(np.float32)
        V = V / np.sqrt(np.mean(V ** 2, -1, keepdims=True))
        g = heads // kvh
        out = np.zeros((R, heads, d), np.float32)
        for hh in range(heads):
            s = q[:, hh] @ K[hh // g].T
            s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
            out[:, hh] = s @ V[hh // g]
        o = out.reshape(R, -1) @ W[p + 'self_attn.o_proj.weight'].T
        x = x + norm(f'l{i}.post_attention_layernorm', o, W[p + 'post_attention_layernorm.weight'])
        h = norm(f'l{i}.pre_feedforward_layernorm', x, W[p + 'pre_feedforward_layernorm.weight'])
        gt = h @ W[p + 'mlp.gate_proj.weight'].T
        ge = 0.5 * gt * (1 + np.tanh(np.sqrt(2 / np.pi) * (gt + 0.044715 * gt ** 3)))
        dn = (ge * (h @ W[p + 'mlp.up_proj.weight'].T)) @ W[p + 'mlp.down_proj.weight'].T
        x = (x + norm(f'l{i}.post_feedforward_layernorm', dn, W[p + 'post_feedforward_layernorm.weight'])) * W[p + 'layer_scalar']
    y = norm('final_norm', x, W['model.norm.weight'])
    out = {}
    for k, v in stats.items():
        v = np.asarray(v)
        gain = np.sqrt(v / (v + EPS))
        out[k] = {'ms_min': float(v.min()), 'ms_median': float(np.median(v)), 'gain_min': float(gain.min())}
    return out, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qat', type=Path, required=True)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--target-gguf', type=Path, required=True)
    ap.add_argument('--hidden', type=Path, required=True)
    ap.add_argument('--ids', type=Path, required=True)
    ap.add_argument('--q4', type=Path)
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    import gguf
    cfg = json.loads((args.qat / 'config.json').read_text())
    Wq, Wb = load(args.qat), load(args.base)
    report = {'tensors': {}, 'norms': {}, 'lattice': {}}
    for k in sorted(Wq):
        if Wq[k].dtype.kind != 'f' or k not in Wb:
            continue
        a, b = np.sqrt(np.mean(Wq[k].astype(np.float64) ** 2)), np.sqrt(np.mean(Wb[k].astype(np.float64) ** 2))
        report['tensors'][k] = {'rms_qat': float(a), 'rms_base': float(b), 'ratio': float(a / b) if b else None}
    # Real inputs: hidden row i produced id i+1; the drafter's first step reads (embed(id i+1), hidden i).
    tg = {t.name: t for t in gguf.GGUFReader(str(args.target_gguf)).tensors}
    hid = np.load(args.hidden).astype(np.float32)
    ids = np.load(args.ids)
    n = min(len(hid) - 1, 512)
    te = tg['token_embd.weight']
    rows = np.unique(ids[1:n + 1])
    table = {}
    raw = np.asarray(te.data)
    for r in rows:
        table[int(r)] = gguf.quants.dequantize(raw[r:r + 1], te.tensor_type).reshape(-1)
    D = hid.shape[1]
    emb = np.stack([table[int(r)] for r in ids[1:n + 1]]) * np.float32(np.sqrt(D))
    kn = {kind: gguf.quants.dequantize(np.asarray(tg[f'blk.{L}.attn_k_norm.weight'].data), tg[f'blk.{L}.attn_k_norm.weight'].tensor_type).reshape(-1)
          for kind, L in (('sliding', 22), ('global', 23))}
    report['inputs'] = {'rows': int(n), 'hidden_rms_median': float(np.median(rms_rows(hid[:n]))), 'embed_rms_median': float(np.median(rms_rows(emb)))}
    ys = {}
    for name, W in (('qat', Wq), ('base', Wb)):
        report['norms'][name], ys[name] = forward_norm_stats(W, cfg, emb, hid[:n], kn, np.random.default_rng(3))
    if args.q4:
        g4 = {t.name: t for t in gguf.GGUFReader(str(args.q4)).tensors}
        names = {'nextn.pre_projection.weight': 'pre_projection.weight', 'nextn.post_projection.weight': 'post_projection.weight', 'token_embd.weight': 'model.embed_tokens.weight'}
        for i in range(len(cfg['text_config']['layer_types'])):
            for a, b in (('attn_q', 'self_attn.q_proj'), ('attn_output', 'self_attn.o_proj'), ('ffn_gate', 'mlp.gate_proj'), ('ffn_up', 'mlp.up_proj'), ('ffn_down', 'mlp.down_proj')):
                names[f'blk.{i}.{a}.weight'] = f'model.layers.{i}.{b}.weight'
        for gname, hname in names.items():
            t = g4[gname]
            w = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).reshape(Wq[hname].shape)
            report['lattice'][gname] = {'type': t.tensor_type.name, 'rel_err': float(np.linalg.norm(w - Wq[hname]) / np.linalg.norm(Wq[hname]))}
    worst = {k: min(v['gain_min'] for v in report['norms'][k].values()) for k in report['norms']}
    report['worst_gain'] = worst
    report['verdict'] = 'no starved norm (every gain >= 0.999)' if worst['qat'] >= 0.999 else 'STARVED NORM: see norms.qat'
    for k in ('qat', 'base'):
        print(k, 'worst gain', f"{worst[k]:.6f}")
        for nm, v in report['norms'][k].items():
            print(f'  {nm:34s} ms_min {v["ms_min"]:.3e} ms_median {v["ms_median"]:.3e} gain_min {v["gain_min"]:.6f}')
    for g, v in report['lattice'].items():
        print(f'lattice {g:32s} {v["type"]:6s} rel_err {v["rel_err"]:.2e}')
    print(report['verdict'])
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
