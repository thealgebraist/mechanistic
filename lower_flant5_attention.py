"""Lower a one-token FLAN-T5 decoder self-attention operation."""
import json,torch
from pathlib import Path
from transformers import T5ForConditionalGeneration,T5Tokenizer
ROOT=Path('work/google_flan'); text='question: What color is the sky? answer:'
tok=T5Tokenizer.from_pretrained(str(ROOT),local_files_only=True); model=T5ForConditionalGeneration.from_pretrained(str(ROOT),local_files_only=True,dtype=torch.float32); model.eval()
enc=tok(text,return_tensors='pt'); dec=torch.tensor([[model.config.decoder_start_token_id]]); captured={}
attn=model.decoder.block[0].layer[0].SelfAttention
def hook(module,args,out): captured['x']=args[0].detach()[0,0].clone(); captured['y']=out[0].detach()[0,0].clone()
handle=attn.register_forward_hook(hook)
with torch.no_grad(): model(input_ids=enc.input_ids,attention_mask=enc.attention_mask,decoder_input_ids=dec)
handle.remove(); x=captured['x']; target=captured['y']; nh=attn.n_heads; dk=attn.key_value_proj_dim
q=(attn.q(x).reshape(nh,dk)); k=(attn.k(x).reshape(nh,dk)); v=(attn.v(x).reshape(nh,dk)); score=(q*k).sum(-1)/(dk**.5); weights=torch.ones(nh); head=weights[:,None]*v; joined=head.reshape(-1); y=attn.o(joined)
program={'language':'NEURAL-ALGEBRA-1','block':'decoder.block[0].layer[0].SelfAttention','prompt':text,'prefix_length':1,'ops':[{'op':'LOAD_VECTOR','name':'x','values':x.tolist()},{'op':'MATMUL_HEADS','weights':{'q':attn.q.weight.tolist(),'k':attn.k.weight.tolist(),'v':attn.v.weight.tolist()},'heads':nh,'head_width':dk,'outputs':['q','k','v']},{'op':'SCALED_DOT_SELF','q':'q','k':'k','scale':dk**-.5,'output':'score'},{'op':'SOFTMAX','input':'score','output':'weights'},{'op':'WEIGHTED_SUM','weights':'weights','values':'v','output':'heads'},{'op':'CONCAT','input':'heads','output':'joined'},{'op':'MATMUL','matrix':'o','weights':attn.o.weight.tolist(),'input':'joined','output':'y'},{'op':'HALT'}]}
Path('outputs/flan_attention_level1.json').write_text(json.dumps(program,separators=(',',':'))+'\n'); err=float(torch.max(torch.abs(y-target))); print(json.dumps({'heads':nh,'head_width':dk,'max_error':err,'program_bytes':Path('outputs/flan_attention_level1.json').stat().st_size},indent=2)); assert err<1e-4
