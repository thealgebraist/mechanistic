"""Sequence-parametric FLAN-T5 register interpreter with native KV layout."""
import argparse, json, math, torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

ROOT = "work/google_flan"
torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision("highest")
ap = argparse.ArgumentParser()
ap.add_argument("--text", default="question: Who wrote Hamlet? answer:")
ap.add_argument("--decoder-text", default=" The play is a")
ap.add_argument("--decoder-ids", default=None,
                help="comma-separated token IDs; takes precedence over decoder-text")
ap.add_argument("--repeat", type=int, default=4)
args = ap.parse_args()
m = T5ForConditionalGeneration.from_pretrained(ROOT, local_files_only=True,
                                                dtype=torch.float32).eval()
t = T5Tokenizer.from_pretrained(ROOT, local_files_only=True)
e = t(args.text, return_tensors="pt")
start = m.config.decoder_start_token_id
continuation = ([int(x) for x in args.decoder_ids.split(",") if x.strip()]
                if args.decoder_ids is not None
                else t.encode(args.decoder_text, add_special_tokens=False))
assert continuation and args.repeat >= 1
ids = torch.tensor([[start] + continuation * args.repeat])
W = m.state_dict(); H = m.config.num_heads; D = m.config.d_kv

def norm(x, w):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * w

def bucket(r, n=32, maximum=128):
    x = -r
    half = n
    x = torch.maximum(x, torch.zeros_like(x))
    exact = half // 2
    small = x < exact
    large = exact + (torch.log(x.float().clamp_min(1) / exact) /
                     math.log(maximum / exact) * (half - exact)).long()
    return torch.where(small, x, large.clamp(max=half - 1)).long()

def bias(name, position, length):
    if name not in W:
        return torch.zeros(1, H, 1, length)
    table = W[name]
    rel = torch.arange(length)[None, :] - position
    return table[bucket(rel, table.shape[0], 128)].permute(2, 0, 1)

with torch.no_grad():
    memory = m.encoder(input_ids=e.input_ids,
                       attention_mask=e.attention_mask).last_hidden_state
    full = m(input_ids=e.input_ids, attention_mask=e.attention_mask,
             decoder_input_ids=ids).logits[0]
    cache = [None] * m.config.num_decoder_layers
    errors = []
    for pos in range(ids.shape[1]):
        y = W["shared.weight"][ids[0, pos]][None, None, :]
        for i in range(m.config.num_decoder_layers):
            p = f"decoder.block.{i}"
            n = norm(y, W[p + ".layer.0.layer_norm.weight"])
            q = (n @ W[p + ".layer.0.SelfAttention.q.weight"].T).view(1, 1, H, D).transpose(1, 2)
            k = (n @ W[p + ".layer.0.SelfAttention.k.weight"].T).view(1, 1, H, D).transpose(1, 2)
            v = (n @ W[p + ".layer.0.SelfAttention.v.weight"].T).view(1, 1, H, D).transpose(1, 2)
            if cache[i] is None:
                kc, vc = k, v
            else:
                kc, vc = torch.cat((cache[i][0], k), 2), torch.cat((cache[i][1], v), 2)
            cache[i] = (kc, vc)
            a = torch.softmax((q @ kc.transpose(-1, -2) + bias(
                "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                pos, pos + 1)).float(), -1).to(y.dtype)
            y = y + (a @ vc).transpose(1, 2).reshape(1, 1, -1) @ W[p + ".layer.0.SelfAttention.o.weight"].T
            n = norm(y, W[p + ".layer.1.layer_norm.weight"])
            q = (n @ W[p + ".layer.1.EncDecAttention.q.weight"].T).view(1, 1, H, D).transpose(1, 2)
            k = (memory @ W[p + ".layer.1.EncDecAttention.k.weight"].T).view(1, -1, H, D).transpose(1, 2)
            v = (memory @ W[p + ".layer.1.EncDecAttention.v.weight"].T).view(1, -1, H, D).transpose(1, 2)
            a = torch.softmax((q @ k.transpose(-1, -2)).float(), -1).to(y.dtype)
            y = y + (a @ v).transpose(1, 2).reshape(1, 1, -1) @ W[p + ".layer.1.EncDecAttention.o.weight"].T
            n = norm(y, W[p + ".layer.2.layer_norm.weight"])
            u = n @ W[p + ".layer.2.DenseReluDense.wi_0.weight"].T
            v = n @ W[p + ".layer.2.DenseReluDense.wi_1.weight"].T
            y = y + (torch.nn.functional.gelu(u, approximate="tanh") * v) @ W[p + ".layer.2.DenseReluDense.wo.weight"].T
        logits = norm(y, W["decoder.final_layer_norm.weight"]) @ W["lm_head.weight"].T
        errors.append(float((logits[0, 0] - full[pos]).abs().max()))
print(json.dumps({"certificate": "REGISTER_KV_CACHE_REPLAY", "position_errors": errors,
                  "max_logit_error": max(errors), "cache_layout": "B,H,L,D",
                  "encoder_tokens": int(e.input_ids.shape[1]),
                  "decoder_positions": int(ids.shape[1]),
                  "arbitrary_sequence_cli": True}))
assert max(errors) < 1e-3
