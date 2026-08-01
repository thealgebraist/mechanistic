import argparse,gzip,json,torch
from pathlib import Path
from transformers import T5ForConditionalGeneration,T5Tokenizer
ap=argparse.ArgumentParser(); ap.add_argument('--prompts',default='domain_prompts.json'); ap.add_argument('--output',default='outputs/flan_domain32_program.json.gz'); ap.add_argument('--horizon',type=int,default=3); a=ap.parse_args()
ROOT=Path('work/google_flan'); prompts=json.loads(Path(a.prompts).read_text()); H=a.horizon; B=2; K=8
tok=T5Tokenizer.from_pretrained(str(ROOT),local_files_only=True); model=T5ForConditionalGeneration.from_pretrained(str(ROOT),local_files_only=True,dtype=torch.float32); model.eval(); start=model.config.decoder_start_token_id; V=model.config.vocab_size
states=[]
with torch.no_grad():
 for pid,text in enumerate(prompts):
  enc=tok(text,return_tensors='pt'); frontier=[()]; seen=set()
  for depth in range(H):
   nxt=[]
   for prefix in frontier:
    if prefix in seen: continue
    seen.add(prefix); dec=torch.tensor([[start,*prefix]])
    p=torch.softmax(model(input_ids=enc.input_ids,attention_mask=enc.attention_mask,decoder_input_ids=dec).logits[0,-1],dim=-1).tolist()
    order=sorted(range(V),key=lambda i:p[i],reverse=True)[:K]; q=[(i,int(round(p[i]*65535))) for i in order]
    states.append({'prompt_id':pid,'depth':depth,'stack':list(prefix),'emit':q,'other':65535-sum(v for _,v in q),
                   'reference_projected':{'emit':[(i,p[i]) for i in order],
                                          'other':1.0-sum(p[i] for i in order)}})
    nxt += [prefix+(i,) for i in order[:B]]
   frontier=nxt
program={'language':'PRSL-STACK-1','prompts':prompts,'vocab_size':V,'horizon':H,'branching_source':B,
         'probability_encoding':{'units':65535,'explicit_tokens':K,'tail':'OTHER'},'states':states}
Path(a.output).write_bytes(gzip.compress(json.dumps(program,separators=(',',':')).encode(),9))
maxerr=max(.5*sum(abs((a if i==j else 0)-(b if i==j else 0)) for i in range(V) for j in []) for a,b in []) if False else 0
print(json.dumps({'prompts':len(prompts),'states':len(states),'bytes':Path(a.output).stat().st_size,'output':a.output},indent=2))
