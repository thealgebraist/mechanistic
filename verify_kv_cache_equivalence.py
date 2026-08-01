"""Compare sequential decoder execution with full-prefix FLAN execution."""
import math, torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import T5ForConditionalGeneration,T5Tokenizer
R=Path('work/google_flan'); W=load_file(str(R/'model.safetensors'),device='cpu'); tok=T5Tokenizer.from_pretrained(str(R),local_files_only=True); m=T5ForConditionalGeneration.from_pretrained(str(R),local_files_only=True,dtype=torch.float32); m.eval()
text='question: Who wrote Hamlet? answer:'; enc=tok(text,return_tensors='pt'); start=m.config.decoder_start_token_id; nxt=tok.encode(' The',add_special_tokens=False)[0]; dec=torch.tensor([[start,nxt]])
with torch.no_grad(): full=m(input_ids=enc.input_ids,attention_mask=enc.attention_mask,decoder_input_ids=dec).logits[0]
def rms(x,w): return x*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)*w
def bucket(rel,bidir,b=32,md=128):
 n=-rel; half=b; out=torch.zeros_like(n)
 if bidir: half//=2; out+=(n<0).long()*half; n=n.abs()
 else:n=torch.maximum(n,torch.zeros_like(n))
 e=half//2; small=n<e; large=e+(torch.log(n.float().clamp_min(1)/e)/math.log(md/e)*(half-e)).long(); return out+torch.where(small,n,large.clamp(max=half-1)).long()
def rb(name,qpos,klen,bidir):
 if name not in W: return torch.zeros(1,m.config.num_heads,klen)
 tab=W[name]; rel=torch.arange(klen)-qpos; return tab[bucket(rel,bidir,tab.shape[0],128)].T[None,:,:]
memory=m.encoder(input_ids=enc.input_ids,attention_mask=enc.attention_mask).last_hidden_state[0]
caches=[None]*m.config.num_decoder_layers; y=W['shared.weight'][dec[0,0]][None,:]; errors=[]; hidden_errors=[]
for pos in range(2):
 if pos: y=W['shared.weight'][dec[0,pos]][None,:]
 for i in range(m.config.num_decoder_layers):
  p=f'decoder.block.{i}'; n=rms(y,W[p+'.layer.0.layer_norm.weight']); h=m.config.num_heads; d=64
  q=(n@W[p+'.layer.0.SelfAttention.q.weight'].T).view(1,h,d)
  knew=(n@W[p+'.layer.0.SelfAttention.k.weight'].T).view(1,h,d); vnew=(n@W[p+'.layer.0.SelfAttention.v.weight'].T).view(1,h,d)
  if caches[i] is None: kcache,v_cache=knew,vnew
  else: kcache,v_cache=torch.cat((caches[i][0],knew),0),torch.cat((caches[i][1],vnew),0)
  caches[i]=(kcache,v_cache); score=torch.einsum('bhd,lhd->bhl',q,kcache)+rb(p+'.layer.0.SelfAttention.relative_attention_bias.weight',pos,pos+1,False); weights=torch.softmax(score.float(),-1).to(y.dtype); z=torch.einsum('bhl,lhd->bhd',weights,v_cache).reshape(1,-1)@W[p+'.layer.0.SelfAttention.o.weight'].T; y=y+z
  n=rms(y,W[p+'.layer.1.layer_norm.weight']); q=(n@W[p+'.layer.1.EncDecAttention.q.weight'].T).view(1,h,d); k=(memory@W[p+'.layer.1.EncDecAttention.k.weight'].T).view(memory.shape[0],h,d); v=(memory@W[p+'.layer.1.EncDecAttention.v.weight'].T).view(memory.shape[0],h,d); weights=torch.softmax(torch.einsum('bhd,lhd->bhl',q,k).float(),-1).to(y.dtype); y=y+torch.einsum('bhl,lhd->bhd',weights,v).reshape(1,-1)@W[p+'.layer.1.EncDecAttention.o.weight'].T; n=rms(y,W[p+'.layer.2.layer_norm.weight']); u=n@W[p+'.layer.2.DenseReluDense.wi_0.weight'].T; v=n@W[p+'.layer.2.DenseReluDense.wi_1.weight'].T; y=y+(torch.nn.functional.gelu(u,approximate='tanh')*v)@W[p+'.layer.2.DenseReluDense.wo.weight'].T
 y=rms(y,W['decoder.final_layer_norm.weight']); logits=y@W['lm_head.weight'].T; errors.append(float((logits[0]-full[pos]).abs().max()))
 with torch.no_grad(): ref_h=m.decoder(input_ids=dec[:,:pos+1],encoder_hidden_states=memory[None],encoder_attention_mask=enc.attention_mask).last_hidden_state[0,-1]
 hidden_errors.append(float((y[0]-ref_h).abs().max()))
print({'certificate':'KV_CACHE_SEQUENTIAL_EQUIVALENCE','position_errors':errors,'hidden_errors':hidden_errors,'max_logit_error':max(errors),'cache_layers':len(caches),'cache_length':2}); assert max(errors)<2e-3
