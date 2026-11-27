#!/usr/bin/env python
"""
Repro: in google/gemma-4-E4B-it-qat-q4_0-unquantized, `per_layer_model_projection.weight` is ~64x smaller
than in google/gemma-4-E4B-it. With transformers' own Gemma4 code, the per-layer projection value before
`per_layer_projection_norm` then has a mean square BELOW rms_norm_eps (1e-6), so the RMSNorm no longer
normalizes: it shrinks the branch to ~0.24x (image tokens) / ~0.64x (text tokens) of its intended size.

Needs: torch, transformers>=5.5 (Gemma4), huggingface_hub, safetensors. No HF token needed: the base and QAT repos are public (gated=False).
Downloads only the few tensors it needs (HTTP range reads, ~60 MB for text mode; +~400 MB with --image).
CPU only, < 3 GB RAM.

  python repro_ple_eps.py                    # text tokens
  python repro_ple_eps.py --image page.png   # + image tokens through the real vision tower (max_soft_tokens 1120)
  python repro_ple_eps.py --size E2B         # same check on E2B
"""
import argparse, json, struct, types
import numpy as np, torch
from torch import nn
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextModel, Gemma4RMSNorm, Gemma4VisionModel, Gemma4MultimodalEmbedder)
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig, Gemma4TextConfig

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="E4B")
ap.add_argument("--image", default=None)
ap.add_argument("--text", default="Transcribe this page to Markdown. Keep headings, paragraphs, footnotes, captions and tables. "
                                  "Exact wording, numbers and symbols. The proof uses the Gauss sum over the character group.")
args = ap.parse_args()
REPOS = {"base": f"google/gemma-4-{args.size}-it", "qat": f"google/gemma-4-{args.size}-it-qat-q4_0-unquantized"}
fs = HfFileSystem()
LM = "model.language_model."


class Remote:
    """Read single tensors (or rows) of a remote single-file safetensors checkpoint with range reads."""
    def __init__(self, repo):
        self.f = fs.open(f"{repo}/model.safetensors", "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.hdr = json.loads(self.f.read(n)); self.base = 8 + n
        self.cfg = json.loads(fs.read_text(f"{repo}/config.json"))

    def _read(self, a, b):
        self.f.seek(self.base + a); return self.f.read(b - a)

    def get(self, name, rows=None):
        h = self.hdr[name]; assert h["dtype"] == "BF16", h["dtype"]
        s, e = h["data_offsets"]; shape = h["shape"]
        if rows is None:
            buf = self._read(s, e)
        else:
            w = int(np.prod(shape[1:])) * 2
            buf = b"".join(self._read(s + r * w, s + (r + 1) * w) for r in rows); shape = [len(rows)] + shape[1:]
        a = (np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        return torch.from_numpy(a.copy())


ck = {k: Remote(r) for k, r in REPOS.items()}
tc = ck["base"].cfg["text_config"]
H, P, L, EPS, PAD = tc["hidden_size"], tc["hidden_size_per_layer_input"], tc["num_hidden_layers"], tc["rms_norm_eps"], tc["pad_token_id"]
assert ck["qat"].cfg["text_config"]["rms_norm_eps"] == EPS


def ple_module(c, mul=1.0):
    """Stub `self` for Gemma4TextModel.project_per_layer_inputs holding the checkpoint's real tensors."""
    s = types.SimpleNamespace(hidden_size_per_layer_input=P, config=types.SimpleNamespace(num_hidden_layers=L),
                              per_layer_model_projection_scale=H ** -0.5, per_layer_input_scale=2.0 ** -0.5)
    s.per_layer_model_projection = nn.Linear(H, L * P, bias=False)
    s.per_layer_model_projection.weight.data = c.get(LM + "per_layer_model_projection.weight") * mul
    s.per_layer_projection_norm = Gemma4RMSNorm(P, eps=EPS)
    s.per_layer_projection_norm.weight.data = c.get(LM + "per_layer_projection_norm.weight")
    return s


def measure(s, x, tok_pl):
    with torch.no_grad():
        pre = (s.per_layer_model_projection(x) * s.per_layer_model_projection_scale).reshape(*x.shape[:-1], L, P)
        normed = s.per_layer_projection_norm(pre)                      # what Gemma4 feeds into every layer
        final = Gemma4TextModel.project_per_layer_inputs(s, x, tok_pl)  # Google's code, unchanged
    ms = pre.pow(2).mean(-1)
    return ms, normed, final


tok = AutoTokenizer.from_pretrained(REPOS["base"])
ids = tok(args.text, add_special_tokens=False)["input_ids"]
inputs = {}
for k, c in ck.items():
    te = c.get(LM + "embed_tokens.weight", rows=ids) * torch.tensor(H ** 0.5)  # Gemma4TextScaledWordEmbedding
    tpl = (c.get(LM + "embed_tokens_per_layer.weight", rows=ids) * P ** 0.5).reshape(len(ids), L, P)
    inputs[k] = {"text": (te, tpl)}

if args.image:
    from PIL import Image
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
    pc = json.loads(fs.read_text(f"{REPOS['base']}/processor_config.json"))["image_processor"]
    pc.pop("image_processor_type", None); pc["max_soft_tokens"] = 1120
    bf = Gemma4ImageProcessor(**pc)(images=[Image.open(args.image).convert("RGB")], return_tensors="pt")
    pv, pos = bf["pixel_values"][0], bf["image_position_ids"][0]
    keep = (pos != -1).all(-1); pv, pos = pv[keep], pos[keep]
    vcfg = Gemma4VisionConfig(**ck["base"].cfg["vision_config"]); vcfg._attn_implementation = "sdpa"
    for k, c in ck.items():
        vt = Gemma4VisionModel(vcfg).eval()
        vt.load_state_dict({n[len("model.vision_tower."):]: c.get(n) for n in c.hdr if n.startswith("model.vision_tower.")}, strict=True)
        emb = Gemma4MultimodalEmbedder(vcfg, Gemma4TextConfig(**tc)).eval()
        emb.embedding_projection.weight.data = c.get("model.embed_vision.embedding_projection.weight")
        with torch.no_grad():
            ie = emb(vt(pixel_values=pv[None], pixel_position_ids=pos[None]).last_hidden_state)
        # image positions: input_ids -> pad_token_id for the per-layer lookup (Gemma4Model.forward)
        ipl = (c.get(LM + "embed_tokens_per_layer.weight", rows=[PAD]) * P ** 0.5).reshape(1, L, P).expand(ie.shape[0], L, P)
        inputs[k]["image"] = (ie, ipl)
        del vt

q = lambda a: {k: float(f"{v:.4g}") for k, v in (("p10", np.percentile(a, 10)), ("median", np.median(a)), ("p90", np.percentile(a, 90)))}
mods = {"base": ple_module(ck["base"]), "qat": ple_module(ck["qat"]), "qat_x64": ple_module(ck["qat"], 64.0)}
for k in ("base", "qat"):
    w = mods[k].per_layer_model_projection.weight
    print(f"{k:5s} per_layer_model_projection.weight RMS = {w.pow(2).mean().sqrt():.6f}")
print(f"rms_norm_eps = {EPS}")
for kind in inputs["base"]:
    r = {}
    for k in ("base", "qat", "qat_x64"):
        x, tpl = inputs["qat" if k == "qat_x64" else k][kind]
        r[k] = measure(mods[k], x, tpl)
    nb, nq, nx = (r[k][1].norm(dim=-1).numpy() for k in ("base", "qat", "qat_x64"))
    fq, fx = r["qat"][2], r["qat_x64"][2]
    fcos = torch.nn.functional.cosine_similarity(fq, fx, dim=-1).numpy()
    print(f"\n[{kind} tokens: {inputs['base'][kind][0].shape[0]}]")
    print("  mean square before per_layer_projection_norm: base", q(r["base"][0].numpy()), " qat", q(r["qat"][0].numpy()))
    print("  norm gain sqrt(ms/(ms+eps)):                  base", q(np.sqrt(r["base"][0] / (r["base"][0] + EPS)).numpy()),
          " qat", q(np.sqrt(r["qat"][0] / (r["qat"][0] + EPS)).numpy()))
    print("  |normed| ratio qat / base:", q(nq / nb), "   qat / qat_x64:", q(nq / nx))
    print("  final per-layer input cosine, qat vs qat_x64:", q(fcos))
