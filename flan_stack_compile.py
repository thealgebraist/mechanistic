"""Compile bounded FLAN-T5 behavior into a branching probabilistic stack program."""
import gzip,json
from pathlib import Path
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

ROOT=Path('work/google_flan'); OUT=Path('outputs'); OUT.mkdir(exist_ok=True)
PROMPTS=['translate English to German: The keys are on the table.','translate English to German: The cat is on the mat.','summarize: The small observatory was flooded after a storm.','question: What color is the sky? answer:','classify sentiment: I loved the careful and helpful explanation.','complete: The keys to the cabinet','translate English to French: The door is open.','question: Who wrote Hamlet? answer:']
H=3; BRANCH=2

def qdist(probs,k):
    order=sorted(range(len(probs)),key=lambda i:probs[i],reverse=True)[:k]
    q=[(i,int(round(probs[i]*65535))) for i in order]
    return q,max(0,65535-sum(v for _,v in q))
def tv(ref,q,V):
    x=[0.0]*V
    for i,v in q:x[i]=v/65535
    return .5*sum(abs(a-b) for a,b in zip(ref,x))

tok=T5Tokenizer.from_pretrained(str(ROOT),local_files_only=True)
model=T5ForConditionalGeneration.from_pretrained(str(ROOT),local_files_only=True,dtype=torch.float32); model.eval(); V=model.config.vocab_size; start=model.config.decoder_start_token_id
states=[]
with torch.no_grad():
  for pi,text in enumerate(PROMPTS):
    enc=tok(text,return_tensors='pt')
    frontier=[(())]
    seen=set()
    for depth in range(H):
      nxt=[]
      for prefix in frontier:
        if prefix in seen:continue
        seen.add(prefix)
        dec=torch.tensor([[start,*prefix]])
        logits=model(input_ids=enc.input_ids,attention_mask=enc.attention_mask,decoder_input_ids=dec).logits[0,-1]
        probs=torch.softmax(logits,dim=-1).tolist(); q,other=qdist(probs,BRANCH)
        states.append({'prompt_id':pi,'depth':depth,'stack':list(prefix),'emit':q,'other':other,'reference':probs})
        nxt.extend([(prefix+(i,)) for i,_ in q])
      frontier=nxt

report=[]
for budget,k in [(2048,1),(4096,2),(8192,4),(16384,8)]:
  program={'language':'PRSL-STACK-1','semantics':{'registers':['prompt_id','depth'],'stack':'decoder_prefix','instruction':'EMIT_TOPK; PUSH'},'vocab_size':V,'horizon':H,'branching_source':BRANCH,'states':[]}
  for s in states:
    q,other=qdist(s['reference'],k)
    program['states'].append({'prompt_id':s['prompt_id'],'depth':s['depth'],'stack':s['stack'],'emit':q,'other':other})
  blob=gzip.compress(json.dumps(program,separators=(',',':')).encode(),9)
  errors=[tv(s['reference'],p['emit'],V) for s,p in zip(states,program['states'])]
  report.append({'budget':budget,'actual_bytes':len(blob),'top_k':k,'states':len(states),'max_tv_error':max(errors),'mean_tv_error':sum(errors)/len(errors),'program':program,'errors':errors})
out={'model':'google/flan-t5-small','language':'PRSL-STACK-1','prompts':PROMPTS,'horizon':H,'branching':BRANCH,'state_count':len(states),'frontier':[{x:y for x,y in r.items() if x not in ('program','errors')} for r in report]}
(OUT/'flan_stack_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
(OUT/'flan_stack_program_16k.json.gz').write_bytes(gzip.compress(json.dumps(report[-1]['program'],separators=(',',':')).encode(),9))
print(json.dumps(out,indent=2))
