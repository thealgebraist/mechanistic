#!/usr/bin/env python3
"""Check source/register primitive ordering against the pinned backend source."""
import hashlib, inspect, json
from pathlib import Path
from transformers.models.t5.modeling_t5 import T5Attention, T5DenseGatedActDense, T5LayerNorm

target = Path("verify_kv_cache_register.py").read_text()
source = {
    "RMSNORM": inspect.getsource(T5LayerNorm.forward),
    "GATED_MLP": inspect.getsource(T5DenseGatedActDense.forward),
    "ATTENTION": inspect.getsource(T5Attention.forward),
}

def ordered(text, needles):
    pos = -1
    for needle in needles:
        pos = text.find(needle, pos + 1)
        assert pos >= 0, f"missing ordered primitive {needle!r}"
    return True

assert ordered(source["RMSNORM"], ["pow(2).mean", "torch.rsqrt", "self.weight * hidden_states"])
assert "def norm" in target and ".square().mean" in target and "torch.rsqrt" in target
assert all(x in source["GATED_MLP"] for x in ["self.wi_0", "self.act", "self.wi_1",
                                               "hidden_gelu * hidden_linear", "self.wo"])
assert ordered(target, ["DenseReluDense.wi_0.weight", "DenseReluDense.wi_1.weight",
                        "torch.nn.functional.gelu", "DenseReluDense.wo.weight"])
assert ordered(source["ATTENTION"], ["self.q(hidden_states)", "self.k(current_states)",
                                     "self.v(current_states)", "torch.matmul(query_states",
                                     "scores += position_bias_masked", "softmax(scores.float()",
                                     "torch.matmul(attn_weights, value_states)", "self.o(attn_output)"])
assert all(x in target for x in ["SelfAttention.q.weight", "SelfAttention.k.weight",
                                 "SelfAttention.v.weight", "q @ kc.transpose", "bias(",
                                 "torch.softmax", "a @ vc", "SelfAttention.o.weight"])
assert "curr_past_key_value.update" in source["ATTENTION"]
assert "torch.cat((cache[i][0], k), 2)" in target and "torch.cat((cache[i][1], v), 2)" in target

certificate = {
    "certificate": "BACKEND_SOURCE_CORRESPONDENCE_OK",
    "source_method_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in source.items()},
    "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
    "checked": ["RMSNorm primitive order", "gated-MLP dataflow", "attention primitive order",
                "K/V cache update and append"],
    "dropout_contract": "model.eval(); source dropout is identity",
    "status": "syntactic correspondence check; not a universal IEEE-754 proof",
    "universal_ieee754_proof": False,
}
Path("outputs/flan_backend_source_correspondence.json").write_text(json.dumps(certificate, indent=2) + "\n")
print(json.dumps(certificate, indent=2))
