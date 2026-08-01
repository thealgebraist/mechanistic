"""Lower a two-token-prefix FLAN-T5 decoder step with an explicit KV cache."""
import json, torch
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer

ROOT=Path("work/google_flan"); prompt="question: What color is the sky? answer:"
tok=T5Tokenizer.from_pretrained(str(ROOT),local_files_only=True)
model=T5ForConditionalGeneration.from_pretrained(str(ROOT),local_files_only=True,dtype=torch.float32); model.eval()
enc=tok(prompt,return_tensors="pt"); next_token=tok.encode(" The",add_special_tokens=False)[0]
dec=torch.tensor([[model.config.decoder_start_token_id,next_token]])
cap={}
def bh(mod,args,out): cap["xseq"]=args[0].detach()[0].clone(); cap["yseq"]=out[0].detach()[0].clone()
def lh(mod,args,out): cap["l0"]=(out[0] if isinstance(out,tuple) else out).detach()[0].clone()
def ah(mod,args,out):
 cap["sa_in"]=args[0].detach()[0].clone(); cap["sa_out"]=out[0].detach()[0].clone(); cap["sa_bias"]=out[1].detach()[0,:,1,:].clone(); cap["sa_w"]=out[2].detach()[0,:,1,:].clone()
def proj(name):
 def ph(mod,args,out): cap[name]=out.detach()[0].clone()
 return ph
h=model.decoder.block[0].register_forward_hook(bh)
h0=model.decoder.block[0].layer[0].register_forward_hook(lh)
h1=model.decoder.block[0].layer[0].SelfAttention.register_forward_hook(ah)
sa_mod=model.decoder.block[0].layer[0].SelfAttention
hq=sa_mod.q.register_forward_hook(proj('q_actual')); hk=sa_mod.k.register_forward_hook(proj('k_actual')); hv=sa_mod.v.register_forward_hook(proj('v_actual'))
with torch.no_grad(): result=model(input_ids=enc.input_ids,attention_mask=enc.attention_mask,decoder_input_ids=dec,return_dict=True,output_attentions=True)
h.remove(); h0.remove(); h1.remove(); hq.remove(); hk.remove(); hv.remove(); b=model.decoder.block[0]; sl,cl,ml=b.layer; sa=sl.SelfAttention; ca=cl.EncDecAttention; dm=ml.DenseReluDense
xseq=cap["xseq"]; x=xseq[-1]; mem=result.encoder_last_hidden_state[0]
def rms(n,z): return n.weight*z/torch.sqrt(torch.mean(z*z,dim=-1,keepdim=True)+n.variance_epsilon)
nseq=rms(sl.layer_norm,xseq); q=(sa.q(nseq[-1])).reshape(sa.n_heads,sa.key_value_proj_dim)
k=(sa.k(nseq)).reshape(2,sa.n_heads,sa.key_value_proj_dim).permute(1,0,2); v=(sa.v(nseq)).reshape(2,sa.n_heads,sa.key_value_proj_dim).permute(1,0,2)
bias=cap['sa_bias']; w=torch.softmax(torch.einsum("hd,hld->hl",q,k)+bias,dim=-1); a0=sa.o(torch.einsum("hl,hld->hd",w,v).reshape(-1)); x1=x+a0
n1=rms(cl.layer_norm,x1); q2=ca.q(n1).reshape(ca.n_heads,ca.key_value_proj_dim); k2=ca.k(mem).reshape(mem.shape[0],ca.n_heads,ca.key_value_proj_dim).permute(1,0,2); v2=ca.v(mem).reshape(mem.shape[0],ca.n_heads,ca.key_value_proj_dim).permute(1,0,2); w2=torch.softmax(torch.einsum("hd,hld->hl",q2,k2),dim=-1); a1=ca.o(torch.einsum("hl,hld->hd",w2,v2).reshape(-1)); x2=x1+a1
n2=rms(ml.layer_norm,x2); m=dm.wo(torch.nn.functional.gelu(dm.wi_0(n2),approximate="tanh")*dm.wi_1(n2)); y=x2+m
print('self_error',float((a0-cap['sa_out'][-1]).abs().max()),float((x+a0-cap['l0'][-1]).abs().max()),float((w-cap['sa_w']).abs().max()))
print('projection_errors',float(((sa.q(nseq[-1])-cap['q_actual'][-1])).abs().max()),float(((sa.k(nseq)-cap['k_actual'])).abs().max()),float(((sa.v(nseq)-cap['v_actual'])).abs().max()))
print('norm_input_error',float((nseq-cap['sa_in']).abs().max()))
err=float((y-cap["yseq"][-1]).abs().max()); print('cached_reference_error',err)
def norm(n,inp,out): return {"op":"RMSNORM","weight":n.weight.tolist(),"epsilon":n.variance_epsilon,"input":inp,"output":out}
def mat(w,inp,out): return {"op":"MATMUL","weights":w.tolist(),"input":inp,"output":out}
ops=[{"op":"INPUT_VECTOR","name":"x","type":"Vect[512] Float32"},{"op":"INPUT_MATRIX","name":"self_k_cache","type":"Matrix[6,2,64] Float32"},{"op":"INPUT_MATRIX","name":"self_v_cache","type":"Matrix[6,2,64] Float32"},{"op":"INPUT_MATRIX","name":"self_bias","type":"Matrix[6,2] Float32"},{"op":"INPUT_MATRIX","name":"memory","type":"Matrix[11,512] Float32"},norm(sl.layer_norm,"x","n0"),{"op":"SELF_ATTENTION_CACHE","input":"n0","k_cache":"self_k_cache","v_cache":"self_v_cache","bias":"self_bias","q":sa.q.weight.tolist(),"o":sa.o.weight.tolist(),"output":"a0"},{"op":"ADD","left":"x","right":"a0","output":"x1"},norm(cl.layer_norm,"x1","n1"),{"op":"CROSS_ATTENTION","input":"n1","memory":"memory","q":ca.q.weight.tolist(),"k":ca.k.weight.tolist(),"v":ca.v.weight.tolist(),"o":ca.o.weight.tolist(),"heads":ca.n_heads,"head_width":ca.key_value_proj_dim,"output":"a1"},{"op":"ADD","left":"x1","right":"a1","output":"x2"},norm(ml.layer_norm,"x2","n2"),mat(dm.wi_0.weight,"n2","u"),mat(dm.wi_1.weight,"n2","v"),{"op":"GELU_GATE","left":"u","right":"v","output":"gated"},mat(dm.wo.weight,"gated","m"),{"op":"ADD","left":"x2","right":"m","output":"y"},{"op":"HALT"}]
fixture={"x":x.tolist(),"self_k_cache":k.tolist(),"self_v_cache":v.tolist(),"self_bias":bias.tolist(),"memory":mem.tolist()}
program={"language":"NEURAL-ALGEBRA-1-CACHED","block":"decoder.block[0].step","prompt":prompt,"prefix_token_ids":dec[0].tolist(),"ops":ops,"target":y.tolist()}
Path("outputs/flan_cached_step.json").write_text(json.dumps(program,separators=(",",":"))+"\n"); Path("outputs/flan_cached_step_fixture.json").write_text(json.dumps(fixture,separators=(",",":"))+"\n")
print(json.dumps({"prefix_length":2,"cache_shapes":{"k":list(k.shape),"v":list(v.shape)},"max_reference_error":err,"program_bytes":Path("outputs/flan_cached_step.json").stat().st_size,"fixture_bytes":Path("outputs/flan_cached_step_fixture.json").stat().st_size},indent=2))
