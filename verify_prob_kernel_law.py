import json,torch
from pathlib import Path
from transformers import T5ForConditionalGeneration,T5Tokenizer
Q=json.loads(Path('outputs/flan_domain32_prob_kernel.json').read_text()); H=Q['horizon']; V=32128
ROOT=Path('work/google_flan'); tok=T5Tokenizer.from_pretrained(str(ROOT),local_files_only=True); model=T5ForConditionalGeneration.from_pretrained(str(ROOT),local_files_only=True,dtype=torch.float32); model.eval(); start=model.config.decoder_start_token_id; by={s['id']:s for s in Q['states']}
def exact(text,prefix):
 enc=tok(text,return_tensors='pt'); dec=torch.tensor([[start,*prefix]])
 with torch.no_grad(): p=torch.softmax(model(input_ids=enc.input_ids,attention_mask=enc.attention_mask,decoder_input_ids=dec).logits[0,-1],dim=-1).tolist()
 order=sorted(range(V),key=lambda i:p[i],reverse=True)[:2]; return [(i,p[i]) for i in order],1-sum(p[i] for i in order)
def exactlaw(text):
 out={}
 def rec(pre,pr,d):
  if d==H:out[tuple(pre)]=out.get(tuple(pre),0)+pr;return
  top,other=exact(text,pre)
  for i,p in top:rec(pre+[i],pr*p,d+1)
  out[tuple(pre+['OTHER'])]=out.get(tuple(pre+['OTHER']),0)+pr*other
 rec([],1.,0);return out
def approxlaw(root):
 out={}
 def rec(sid,pre,pr,d):
  s=by[sid]
  for token,p in s['emission'].items():
   if token=='OTHER':out[tuple(pre+['OTHER'])]=out.get(tuple(pre+['OTHER']),0)+pr*p;continue
   t=int(token) if isinstance(token,str) else token
   if d==H-1:out[tuple(pre+[t])]=out.get(tuple(pre+[t]),0)+pr*p;continue
   for dst,k in s['kernel'].get(str(t),{}).items():
    if dst=='OTHER': out[tuple(pre+["OTHER"])]=out.get(tuple(pre+["OTHER"]),0)+pr*p*k
    else: rec(int(dst),pre+[t],pr*p*k,d+1)
 rec(root,[],1.,0);return out
errs=[]
for pid,text in enumerate(Q['prompts']):
 a=exactlaw(text); b=approxlaw(Q['roots'][str(pid)]); keys=set(a)|set(b); errs.append(.5*sum(abs(a.get(k,0)-b.get(k,0)) for k in keys))
print(json.dumps({'prompts':len(Q['prompts']),'states':len(Q['states']),'max_sequence_tv':max(errs),'mean_sequence_tv':sum(errs)/len(errs)},indent=2))
