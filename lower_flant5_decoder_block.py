"""Compile one FLAN-T5 decoder block into a connected register trace.

The trace is intentionally one concrete example (one prompt, one decoder
position). It contains the block's actual RMS norms, self-attention,
cross-attention, residual additions, and gated MLP weights.
"""
import json, torch
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer

ROOT = Path("work/google_flan")
PROMPT = "question: What color is the sky? answer:"
tok = T5Tokenizer.from_pretrained(str(ROOT), local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(str(ROOT), local_files_only=True, dtype=torch.float32)
model.eval()
enc = tok(PROMPT, return_tensors="pt")
dec = torch.tensor([[model.config.decoder_start_token_id]])
captured = {}

def block_hook(module, args, out):
    captured["x0"] = args[0].detach()[0, 0].clone()
    captured["block_out"] = out[0].detach()[0, 0].clone()

def layer_hook(name):
    def hook(module, args, out):
        z = out[0] if isinstance(out, tuple) else out
        captured[name] = z.detach()[0, 0].clone()
    return hook

h = model.decoder.block[0].register_forward_hook(block_hook)
lh0 = model.decoder.block[0].layer[0].register_forward_hook(layer_hook("layer0_out"))
lh1 = model.decoder.block[0].layer[1].register_forward_hook(layer_hook("layer1_out"))
lh2 = model.decoder.block[0].layer[2].register_forward_hook(layer_hook("layer2_out"))
with torch.no_grad():
    result = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                   decoder_input_ids=dec, return_dict=True, output_hidden_states=True)
h.remove()
lh0.remove(); lh1.remove(); lh2.remove()

b = model.decoder.block[0]
self_layer, cross_layer, mlp_layer = b.layer
x0 = captured["x0"]
memory = result.encoder_last_hidden_state[0].detach()

def rms(norm, x):
    return norm.weight * x / torch.sqrt(torch.mean(x * x) + norm.variance_epsilon)

def self_attn(attn, x):
    nh, d = attn.n_heads, attn.key_value_proj_dim
    q = attn.q(x).reshape(nh, d); k = attn.k(x).reshape(nh, d); v = attn.v(x).reshape(nh, d)
    w = torch.ones(nh)
    return attn.o((w[:, None] * v).reshape(-1))

def cross_attn(attn, x, mem):
    nh, d = attn.n_heads, attn.key_value_proj_dim
    q = attn.q(x).reshape(nh, d)
    k = attn.k(mem).reshape(mem.shape[0], nh, d).permute(1, 0, 2)
    v = attn.v(mem).reshape(mem.shape[0], nh, d).permute(1, 0, 2)
    # T5's q projection already incorporates its trained scale; the HF
    # implementation forms the raw dot product here (no extra 1/sqrt(d)).
    score = torch.einsum("hd,hld->hl", q, k)
    w = torch.softmax(score, dim=-1)
    head = torch.einsum("hl,hld->hd", w, v)
    return attn.o(head.reshape(-1))

n0 = rms(self_layer.layer_norm, x0); a0 = self_attn(self_layer.SelfAttention, n0); x1 = x0 + a0
n1 = rms(cross_layer.layer_norm, x1); a1 = cross_attn(cross_layer.EncDecAttention, n1, memory); x2 = x1 + a1
n2 = rms(mlp_layer.layer_norm, x2)
u = mlp_layer.DenseReluDense.wi_0(n2); v = mlp_layer.DenseReluDense.wi_1(n2)
m = mlp_layer.DenseReluDense.wo(torch.nn.functional.gelu(u, approximate="tanh") * v); y = x2 + m

block_target = captured["block_out"]
reference_error = float(torch.max(torch.abs(y - block_target)))
print('component_errors', float((x1-captured['layer0_out']).abs().max()), float((x2-captured['layer1_out']).abs().max()), float((y-captured['layer2_out']).abs().max()))
print('reference_error_before_serialize', reference_error)

def mat(w, inp, out):
    return {"op": "MATMUL", "weights": w.detach().tolist(), "input": inp, "output": out}
def norm(mod, inp, out):
    return {"op": "RMSNORM", "weight": mod.weight.detach().tolist(), "epsilon": mod.variance_epsilon, "input": inp, "output": out}

sa = self_layer.SelfAttention; ca = cross_layer.EncDecAttention; dm = mlp_layer.DenseReluDense
ops = [
    {"op":"LOAD_VECTOR", "name":"x0", "values":x0.tolist()},
    {"op":"LOAD_MATRIX", "name":"memory", "values":memory.tolist()},
    norm(self_layer.layer_norm, "x0", "n0"),
    {"op":"SELF_ATTENTION_ONE", "input":"n0", "q":sa.q.weight.tolist(), "k":sa.k.weight.tolist(), "v":sa.v.weight.tolist(), "o":sa.o.weight.tolist(), "heads":sa.n_heads, "head_width":sa.key_value_proj_dim, "output":"a0"},
    {"op":"ADD", "left":"x0", "right":"a0", "output":"x1"},
    norm(cross_layer.layer_norm, "x1", "n1"),
    {"op":"CROSS_ATTENTION", "input":"n1", "memory":"memory", "q":ca.q.weight.tolist(), "k":ca.k.weight.tolist(), "v":ca.v.weight.tolist(), "o":ca.o.weight.tolist(), "heads":ca.n_heads, "head_width":ca.key_value_proj_dim, "output":"a1"},
    {"op":"ADD", "left":"x1", "right":"a1", "output":"x2"},
    norm(mlp_layer.layer_norm, "x2", "n2"),
    mat(dm.wi_0.weight, "n2", "u"), mat(dm.wi_1.weight, "n2", "v"),
    {"op":"GELU_GATE", "left":"u", "right":"v", "output":"gated"},
    mat(dm.wo.weight, "gated", "m"),
    {"op":"ADD", "left":"x2", "right":"m", "output":"y"}, {"op":"HALT"}
]
program = {"language":"NEURAL-ALGEBRA-1", "block":"decoder.block[0]", "prompt":PROMPT,
           "decoder_position":0, "encoder_length":int(memory.shape[0]), "ops":ops,
           "target":y.tolist()}
out = Path("outputs/flan_decoder_block_level1.json")
out.write_text(json.dumps(program, separators=(",", ":")) + "\n")
print(json.dumps({"block":program["block"], "encoder_length":int(memory.shape[0]),
                  "hidden_width":int(x0.numel()), "max_reference_error":reference_error,
                  "program_bytes":out.stat().st_size}, indent=2))
