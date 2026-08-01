"""Verify a projected length-3 sequence law against FLAN-T5.

The projection alphabet is {selected top-2 token IDs, OTHER}; OTHER is
absorbing. This makes the comparison a proper probability law and matches the
branching semantics of PRSL-STACK-1.
"""
import gzip,json, torch
from pathlib import Path
from transformers import T5ForConditionalGeneration,T5Tokenizer

ROOT=Path('work/google_flan'); H=3; BRANCH=2
prompts=json.loads(Path('outputs/flan_stack_certificate.json').read_text())['prompts']
program=json.loads(gzip.decompress(Path('outputs/flan_stack_program_16k.json.gz').read_bytes()))
approx_by={(s['prompt_id'],tuple(s['stack'])):s for s in program['states']}
tok=T5Tokenizer.from_pretrained(str(ROOT),local_files_only=True)
model=T5ForConditionalGeneration.from_pretrained(str(ROOT),local_files_only=True,dtype=torch.float32); model.eval(); start=model.config.decoder_start_token_id

def dist(text,prefix):
 enc=tok(text,return_tensors='pt'); dec=torch.tensor([[start,*prefix]])
 with torch.no_grad(): p=torch.softmax(model(input_ids=enc.input_ids,attention_mask=enc.attention_mask,decoder_input_ids=dec).logits[0,-1],dim=-1).tolist()
 order=sorted(range(len(p)),key=lambda i:p[i],reverse=True)[:BRANCH]
 return [(i,p[i]) for i in order],1-sum(p[i] for i in order)

def law(text,approx=False,pid=0):
 out={}
 def rec(prefix,prob,depth):
  if depth==H: out[tuple(prefix)]=out.get(tuple(prefix),0)+prob; return
  if approx:
   s=approx_by[(pid,tuple(prefix))]; top=[(i,q/65535) for i,q in s['emit'][:BRANCH]]; other=1-sum(p for _,p in top)
  else: top,other=dist(text,prefix)
  for i,p in top: rec(prefix+[i],prob*p,depth+1)
  out[tuple(prefix+['OTHER'])]=out.get(tuple(prefix+['OTHER']),0)+prob*other
 rec([],1.,0); return out

errs=[]
for pid,text in enumerate(prompts):
 a=law(text,False,pid); b=law(text,True,pid)
 keys=set(a)|set(b); errs.append(.5*sum(abs(a.get(k,0)-b.get(k,0)) for k in keys))
max_local=0.02285801480125124
print('SEQUENCE_LAW_REFERENCE_ENUMERATION_OK')
print('prompts:',len(prompts),'horizon:',H,'projected_paths_per_prompt:',2**H+sum(2**d for d in range(H)))
print('measured_sequence_TV_max:',max(errs),'measured_sequence_TV_mean:',sum(errs)/len(errs))
print('certified_PRSL_sequence_TV_bound:',min(1,H*max_local))
assert max(errs)<=H*max_local+1e-12
