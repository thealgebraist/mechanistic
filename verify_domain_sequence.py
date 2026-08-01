import gzip,json,torch
from pathlib import Path
from transformers import T5ForConditionalGeneration,T5Tokenizer
ROOT=Path('work/google_flan'); P=json.loads(gzip.decompress(Path('outputs/flan_domain32_program.json.gz').read_bytes())); H=P['horizon']; B=P['branching_source']; V=P['vocab_size']
tok=T5Tokenizer.from_pretrained(str(ROOT),local_files_only=True); model=T5ForConditionalGeneration.from_pretrained(str(ROOT),local_files_only=True,dtype=torch.float32); model.eval(); start=model.config.decoder_start_token_id
by={(s['prompt_id'],tuple(s['stack'])):s for s in P['states']}
def exact(text,prefix):
 enc=tok(text,return_tensors='pt'); dec=torch.tensor([[start,*prefix]])
 with torch.no_grad(): p=torch.softmax(model(input_ids=enc.input_ids,attention_mask=enc.attention_mask,decoder_input_ids=dec).logits[0,-1],dim=-1).tolist()
 order=sorted(range(V),key=lambda i:p[i],reverse=True)[:B]; return [(i,p[i]) for i in order],1-sum(p[i] for i in order)
def law(pid,approx):
 out={}; text=P['prompts'][pid]
 def rec(prefix,prob,d):
  if d==H: out[tuple(prefix)]=out.get(tuple(prefix),0)+prob; return
  if approx:
   s=by[(pid,tuple(prefix))]; top=[(i,q/65535) for i,q in s['emit'][:B]]; other=1-sum(v for _,v in top)
  else: top,other=exact(text,prefix)
  for i,p in top: rec(prefix+[i],prob*p,d+1)
  out[tuple(prefix+['OTHER'])]=out.get(tuple(prefix+['OTHER']),0)+prob*other
 rec([],1.,0); return out
errs=[]
for pid in range(len(P['prompts'])):
 a,b=law(pid,False),law(pid,True); keys=set(a)|set(b); errs.append(.5*sum(abs(a.get(k,0)-b.get(k,0)) for k in keys))
print(json.dumps({'prompts':len(P['prompts']),'states':len(P['states']),'horizon':H,'max_sequence_tv':max(errs),'mean_sequence_tv':sum(errs)/len(errs),'note':'exact over projected top-2 plus OTHER laws'},indent=2))
