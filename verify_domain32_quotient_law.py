import json,torch
from pathlib import Path
from transformers import T5ForConditionalGeneration,T5Tokenizer
Q=json.loads(Path('outputs/flan_domain32_quotient.json').read_text()); H=Q['horizon']; B=2; V=32128
ROOT=Path('work/google_flan'); tok=T5Tokenizer.from_pretrained(str(ROOT),local_files_only=True); model=T5ForConditionalGeneration.from_pretrained(str(ROOT),local_files_only=True,dtype=torch.float32); model.eval(); start=model.config.decoder_start_token_id
by={s['id']:s for s in Q['states']}
def exact(text,prefix):
 enc=tok(text,return_tensors='pt'); dec=torch.tensor([[start,*prefix]])
 with torch.no_grad(): p=torch.softmax(model(input_ids=enc.input_ids,attention_mask=enc.attention_mask,decoder_input_ids=dec).logits[0,-1],dim=-1).tolist()
 order=sorted(range(V),key=lambda i:p[i],reverse=True)[:B]; return [(i,p[i]) for i in order],1-sum(p[i] for i in order)
def quotient_law(root):
 out={}
 def rec(sid,prefix,prob,d):
  if d==H: out[tuple(prefix)]=out.get(tuple(prefix),0)+prob; return
  s=by[sid]; top=[(i,q/65535) for i,q in s['emit'][:B]]; other=1-sum(p for _,p in top)
  nxt=dict(s['next'])
  for i,p in top:
   if i in nxt: rec(nxt[i],prefix+[i],prob*p,d+1)
  out[tuple(prefix+['OTHER'])]=out.get(tuple(prefix+['OTHER']),0)+prob*other
 rec(Q['roots']['0'],[],1.,0)
 return out
errs=[]; local_errs=[]
for pid,text in enumerate(Q['prompts']):
 exactlaw={}
 def rec(prefix,prob,d):
  if d==H: exactlaw[tuple(prefix)]=exactlaw.get(tuple(prefix),0)+prob; return
  top,other=exact(text,prefix)
  for i,p in top: rec(prefix+[i],prob*p,d+1)
  exactlaw[tuple(prefix+['OTHER'])]=exactlaw.get(tuple(prefix+['OTHER']),0)+prob*other
 rec([],1.,0)
 # Rebase root for this prompt; the quotient transition structure is prompt-specific.
 root=Q['roots'][str(pid)]; approx={}
 def ar(sid,prefix,prob,d):
  if d==H: approx[tuple(prefix)]=approx.get(tuple(prefix),0)+prob; return
  s=by[sid]; top=[(i,q/65535) for i,q in s['emit'][:B]]; other=1-sum(p for _,p in top); nxt=dict(s['next'])
  true_top,true_other=exact(Q['prompts'][pid],prefix)
  tv_local=.5*sum(abs(dict(true_top).get(i,0)-dict(top).get(i,0)) for i in set(dict(true_top))|set(dict(top)))+.5*abs(true_other-other)
  local_errs.append(tv_local)
  for i,p in top:
   if d==H-1: approx[tuple(prefix+[i])]=approx.get(tuple(prefix+[i]),0)+prob*p
   elif i in nxt: ar(nxt[i],prefix+[i],prob*p,d+1)
  approx[tuple(prefix+['OTHER'])]=approx.get(tuple(prefix+['OTHER']),0)+prob*other
 ar(root,[],1.,0)
 keys=set(exactlaw)|set(approx); errs.append(.5*sum(abs(exactlaw.get(k,0)-approx.get(k,0)) for k in keys))
bound=min(1,H*max(local_errs))
print(json.dumps({'prompts':len(Q['prompts']),'quotient_states':len(Q['states']),'max_sequence_tv':max(errs),'mean_sequence_tv':sum(errs)/len(errs),'max_local_tv':max(local_errs),'bound':bound},indent=2))
assert max(errs)<=bound+1e-12
