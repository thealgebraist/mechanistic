"""Numerically execute the complete FLAN-T5 graph for one decoder position."""
import math
import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import T5ForConditionalGeneration, T5Tokenizer

ROOT = Path('work/google_flan')
ap = argparse.ArgumentParser()
ap.add_argument('--text', default='question: What color is the sky? answer:')
ap.add_argument('--decoder-token', default=None)
args = ap.parse_args()
TEXT = args.text
W = load_file(str(ROOT / 'model.safetensors'), device='cpu')
tok = T5Tokenizer.from_pretrained(str(ROOT), local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(str(ROOT), local_files_only=True, dtype=torch.float32)
model.eval()
A = torch.ops.aten
enc = tok(TEXT, return_tensors='pt')
if args.decoder_token is None:
    dec = torch.tensor([[model.config.decoder_start_token_id]])
else:
    token = tok.encode(args.decoder_token, add_special_tokens=False)[0]
    dec = torch.tensor([[model.config.decoder_start_token_id, token]])

def add(x, y):
    if not isinstance(y, torch.Tensor): y = scalar(x, y)
    return A.add.Tensor(x, y)
def mul(x, y):
    if not isinstance(y, torch.Tensor): y = scalar(x, y)
    return A.mul.Tensor(x, y)
def scalar(x, value): return torch.full((), value, dtype=x.dtype, device=x.device)
def linear(x, weight):
    shape = x.shape[:-1] + (weight.shape[0],)
    return A.mm.default(x.reshape(-1, x.shape[-1]), A.t.default(weight)).reshape(shape)
def batch_matmul(x, y):
    lead = x.shape[:-2]; assert lead == y.shape[:-2]
    z = A.bmm.default(x.reshape(-1, x.shape[-2], x.shape[-1]), y.reshape(-1, y.shape[-2], y.shape[-1]))
    return z.reshape(lead + (x.shape[-2], y.shape[-1]))
def softmax(x): return A._softmax.default(x, -1, False)
def rms(x, weight):
    square = A.pow.Tensor_Scalar(x.float(), 2)
    mean = A.mean.dim(square, [-1], True)
    inv = A.rsqrt.default(add(mean, scalar(mean, 1e-6)))
    return mul(mul(x, inv), weight)

def relative_bucket(rel, bidirectional, buckets=32, maximum=128):
    n = -rel
    half = buckets
    result = torch.zeros_like(n)
    if bidirectional:
        half //= 2
        result += (n < 0).long() * half
        n = n.abs()
    else:
        n = torch.maximum(n, torch.zeros_like(n))
    exact = half // 2
    small = n < exact
    large = exact + (torch.log(n.float().clamp_min(1) / exact) /
                     math.log(maximum / exact) * (half - exact)).long()
    large = large.clamp(max=half - 1)
    return result + torch.where(small, n, large).long()

def relative_bias(name, queries, keys, bidirectional):
    table = W[name]
    rel = torch.arange(keys)[None, :] - torch.arange(queries)[:, None]
    buckets = relative_bucket(rel, bidirectional, table.shape[0], 128)
    return table[buckets].permute(2, 0, 1).unsqueeze(0)

def self_attention(x, prefix, bias, mask=0):
    h = model.config.num_heads
    q = linear(x, W[prefix + '.q.weight']).view(1, x.shape[0], h, -1).permute(0, 2, 1, 3)
    k = linear(x, W[prefix + '.k.weight']).view(1, x.shape[0], h, -1).permute(0, 2, 1, 3)
    v = linear(x, W[prefix + '.v.weight']).view(1, x.shape[0], h, -1).permute(0, 2, 1, 3)
    scores = add(add(batch_matmul(q, k.transpose(-1, -2)), bias), mask)
    p = softmax(scores.float()).to(x.dtype)
    z = batch_matmul(p, v).permute(0, 2, 1, 3).reshape(1, x.shape[0], -1)[0]
    return linear(z, W[prefix + '.o.weight'])

def cross_attention(x, memory, prefix):
    h = model.config.num_heads
    q = linear(x, W[prefix + '.q.weight']).view(1, x.shape[0], h, -1).permute(0, 2, 1, 3)
    k = linear(memory, W[prefix + '.k.weight']).view(1, memory.shape[0], h, -1).permute(0, 2, 1, 3)
    v = linear(memory, W[prefix + '.v.weight']).view(1, memory.shape[0], h, -1).permute(0, 2, 1, 3)
    p = softmax(batch_matmul(q, k.transpose(-1, -2)).float()).to(x.dtype)
    z = batch_matmul(p, v).permute(0, 2, 1, 3).reshape(1, x.shape[0], -1)[0]
    return linear(z, W[prefix + '.o.weight'])

def gated_mlp(x, prefix):
    u = linear(x, W[prefix + '.wi_0.weight'])
    v = linear(x, W[prefix + '.wi_1.weight'])
    cube = A.pow.Tensor_Scalar(u, 3.0)
    inner = add(u, mul(scalar(u, 0.044715), cube))
    t = A.tanh.default(mul(scalar(u, math.sqrt(2.0 / math.pi)), inner))
    gelu = mul(mul(scalar(u, 0.5), u), add(scalar(u, 1.0), t))
    return linear(mul(gelu, v), W[prefix + '.wo.weight'])

ids = enc.input_ids[0]
x = W['shared.weight'][ids]
enc_bias = relative_bias('encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight', x.shape[0], x.shape[0], True)
for i in range(model.config.num_layers):
    p = f'encoder.block.{i}'
    n = rms(x, W[p + '.layer.0.layer_norm.weight'])
    x = add(x, self_attention(n, p + '.layer.0.SelfAttention', enc_bias))
    n = rms(x, W[p + '.layer.1.layer_norm.weight'])
    x = add(x, gated_mlp(n, p + '.layer.1.DenseReluDense'))
memory = rms(x, W['encoder.final_layer_norm.weight'])
with torch.no_grad():
    ref_memory = model.encoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state[0]
print('encoder_error', float((memory-ref_memory).abs().max()))

y = W['shared.weight'][dec[0]]
dec_len = dec.shape[1]
dec_bias = relative_bias('decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight', dec_len, dec_len, False)
dec_mask = torch.triu(torch.full((1, 1, dec_len, dec_len), torch.finfo(torch.float32).min), diagonal=1)
for i in range(model.config.num_decoder_layers):
    p = f'decoder.block.{i}'
    n = rms(y, W[p + '.layer.0.layer_norm.weight'])
    y = add(y, self_attention(n, p + '.layer.0.SelfAttention', dec_bias, dec_mask))
    n = rms(y, W[p + '.layer.1.layer_norm.weight'])
    y = add(y, cross_attention(n, memory, p + '.layer.1.EncDecAttention'))
    n = rms(y, W[p + '.layer.2.layer_norm.weight'])
    y = add(y, gated_mlp(n, p + '.layer.2.DenseReluDense'))

y = rms(y, W['decoder.final_layer_norm.weight'])
with torch.no_grad():
    ref_dec = model.decoder(input_ids=dec, encoder_hidden_states=ref_memory.unsqueeze(0), encoder_attention_mask=enc.attention_mask).last_hidden_state[0, -1]
print('decoder_hidden_error', float((y[-1]-ref_dec).abs().max()))
# Match T5ForConditionalGeneration.forward exactly: lm_head is applied to the
# complete decoder sequence before selecting the final position.  On CPU the
# binary32 reduction kernel may round an M=1 matrix product differently from
# the corresponding row of the model's M=decoder_length product.
all_logits = linear(y, W['lm_head.weight'])
logits = all_logits[-1]
with torch.no_grad():
    reference = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                      decoder_input_ids=dec).logits[0, -1]
error = float((logits - reference).abs().max())
print({'certificate': 'ATEN_PRSL_FULL_GRAPH_REPLAY', 'max_logit_error': error,
       'top_token_equal': bool(logits.argmax() == reference.argmax()),
       'encoder_tokens': int(x.shape[0]), 'decoder_tokens': int(y.shape[0]),
       'readout_rows': int(all_logits.shape[0]),
       'decoder_layers': model.config.num_decoder_layers})
assert error < 2e-3
