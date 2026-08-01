#!/usr/bin/env python3
"""Checkpoint-derived sup-norm envelope for every finite FLAN-T5 sequence."""
import hashlib
import json
import math
import struct
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path("work/google_flan")
MODEL = ROOT / "model.safetensors"
GRAPH = Path("outputs/flan_full_graph.json")
W = load_file(str(MODEL), device="cpu")
graph = json.loads(GRAPH.read_text())
d = int(graph["config"]["d_model"])

raw = MODEL.read_bytes()
header_len = struct.unpack("<Q", raw[:8])[0]
header = json.loads(raw[8:8 + header_len])
data_start = 8 + header_len

def up(x: float) -> float:
    return math.nextafter(float(x), math.inf)

def maxabs(name: str) -> float:
    return up(torch.max(torch.abs(W[name].double())).item())

def row_l1(name: str) -> float:
    x = W[name].double().abs()
    s = torch.max(torch.sum(x, dim=1)).item()
    n = x.shape[1]
    gamma = (n * 2.0**-53) / (1.0 - n * 2.0**-53)
    return up(s * (1.0 + gamma))

needed = set()
for op in graph["ops"]:
    for key in ("weight", "q", "k", "v", "o", "wi_0", "wi_1", "wo"):
        if key in op:
            needed.add(op[key])

norms = {}
manifest = []
for name in sorted(needed):
    kind = "maxabs" if name == "shared.weight" or "layer_norm.weight" in name else "row_l1"
    bound = maxabs(name) if kind == "maxabs" else row_l1(name)
    norms[name] = bound
    meta = header[name]
    begin, end = meta["data_offsets"]
    shape = meta["shape"]
    rows = shape[0] if len(shape) > 1 else 1
    cols = math.prod(shape[1:]) if len(shape) > 1 else shape[0]
    manifest.append((name, kind, data_start + begin, data_start + end, rows, cols, bound))

sqrt_d = up(math.sqrt(d))
events = []

def rms_bound(weight: str) -> float:
    return up(sqrt_d * norms[weight])

def attention_bound(prefix: str, input_bound: float) -> float:
    # Softmax rows are probability vectors, so weighted values are convex
    # combinations. Q, K, masks and sequence length cannot enlarge this norm.
    return up(norms[prefix + ".o.weight"] * up(norms[prefix + ".v.weight"] * input_bound))

def mlp_bound(prefix: str, input_bound: float) -> float:
    u = up(norms[prefix + ".wi_0.weight"] * input_bound)
    v = up(norms[prefix + ".wi_1.weight"] * input_bound)
    # |GELU(u)| <= |u| for exact and tanh-approximate GELU.
    return up(norms[prefix + ".wo.weight"] * up(u * v))

enc = norms["shared.weight"]
events.append({"register": "enc_h.embed", "bound": enc})
for i in range(8):
    p = f"encoder.block.{i}"
    n = rms_bound(p + ".layer.0.layer_norm.weight")
    a = attention_bound(p + ".layer.0.SelfAttention", n)
    r = up(enc + a)
    mn = rms_bound(p + ".layer.1.layer_norm.weight")
    m = mlp_bound(p + ".layer.1.DenseReluDense", mn)
    enc = up(r + m)
    events.append({"register": f"enc_h.layer{i}", "normalized": n,
                   "attention": a, "mlp": m, "bound": enc})
memory = rms_bound("encoder.final_layer_norm.weight")
events.append({"register": "encoder_memory", "bound": memory})

dec = norms["shared.weight"]
events.append({"register": "dec_h.embed", "bound": dec})
for i in range(8):
    p = f"decoder.block.{i}"
    n = rms_bound(p + ".layer.0.layer_norm.weight")
    a = attention_bound(p + ".layer.0.SelfAttention", n)
    r = up(dec + a)
    cn = rms_bound(p + ".layer.1.layer_norm.weight")
    ca = attention_bound(p + ".layer.1.EncDecAttention", memory)
    cr = up(r + ca)
    mn = rms_bound(p + ".layer.2.layer_norm.weight")
    m = mlp_bound(p + ".layer.2.DenseReluDense", mn)
    dec = up(cr + m)
    events.append({"register": f"dec_h.layer{i}", "self_normalized": n,
                   "self_attention": a, "cross_query_normalized": cn,
                   "cross_attention": ca, "mlp": m, "bound": dec})

readout = rms_bound("decoder.final_layer_norm.weight")
logits = up(norms["lm_head.weight"] * readout)
events.append({"register": "readout_h", "bound": readout})
events.append({"register": "logits", "bound": logits})

manifest_path = Path("outputs/flan_reachable_norm_manifest.tsv")
manifest_path.write_text("name\tkind\tbegin\tend\trows\tcols\tbound\n" + "".join(
    f"{n}\t{k}\t{b}\t{e}\t{r}\t{c}\t{x:.17g}\n" for n, k, b, e, r, c, x in manifest))

out = {
    "language": "FLAN-REACHABLE-SUPNORM-BOUND-1",
    "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
    "full_graph_sha256": hashlib.sha256(GRAPH.read_bytes()).hexdigest(),
    "scope": "every backend-valid nonempty finite encoder sequence and every finite decoder continuation from the start token",
    "norm": "maximum absolute component over positions and channels",
    "lemmas": {
        "embedding": "finite lookup <= checkpoint maximum absolute entry",
        "rmsnorm": "|x_i|/sqrt(mean(x^2)+epsilon) <= sqrt(d_model)",
        "linear": "||Wx||_inf <= max_row_sum_abs(W) * ||x||_inf",
        "attention": "softmax weighted values are convex combinations; bound is independent of sequence length",
        "gelu": "|GELU(x)| <= |x|",
        "residual": "triangle inequality",
    },
    "d_model": d,
    "sqrt_d_model_up": sqrt_d,
    "tensor_norm_count": len(norms),
    "events": events,
    "final_bounds": {"encoder_hidden": enc, "encoder_memory": memory, "decoder_hidden": dec,
                     "readout_hidden": readout, "logit_abs": logits},
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "status": "checkpoint-specific universal real-arithmetic envelope; C++23 independently rechecks tensor norms",
}
path = Path("outputs/flan_reachable_state_bounds.json")
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"artifact": str(path), "tensor_norms": len(norms),
                  "final_bounds": out["final_bounds"]}, indent=2))
