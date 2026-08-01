"""Compile bounded FLAN-T5 behavior into a tiny probabilistic register program.

The program is a finite map from tokenized input registers to a quantized
distribution over the first decoder token.  Its certificate is exact for the
declared prompt domain: it records FLAN's reference distribution, the emitted
distribution, and total-variation error for every case.
"""
import gzip, json, math
from pathlib import Path
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

ROOT=Path('work/google_flan'); OUT=Path('outputs'); OUT.mkdir(exist_ok=True)
PROMPTS=[
 'translate English to German: The keys are on the table.',
 'translate English to German: The cat is on the mat.',
 'summarize: The small observatory was flooded after a storm.',
 'question: What color is the sky? answer:',
 'classify sentiment: I loved the careful and helpful explanation.',
 'complete: The keys to the cabinet',
 'translate English to French: The door is open.',
 'question: Who wrote Hamlet? answer:',
]

def tv(a,b): return .5*sum(abs(x-y) for x,y in zip(a,b))
def compile_case(ids, probs, k):
    order=sorted(range(len(probs)),key=lambda i:probs[i],reverse=True)[:k]
    # Probabilities are 16-bit fixed point; residual mass is an explicit OTHER register.
    q=[(i,int(round(probs[i]*65535))) for i in order]
    used=sum(v for _,v in q); return {'input_register':ids,'top':q,'other':max(0,65535-used)}
def decode_program(case,V):
    p=[0.0]*V
    for i,v in case['top']: p[i]=v/65535
    return p

tok=T5Tokenizer.from_pretrained(str(ROOT),local_files_only=True)
model=T5ForConditionalGeneration.from_pretrained(str(ROOT),local_files_only=True,torch_dtype=torch.float32)
model.eval()
cases=[]
with torch.no_grad():
  for text in PROMPTS:
    batch=tok(text,return_tensors='pt')
    dec=torch.tensor([[model.config.decoder_start_token_id]])
    logits=model(input_ids=batch.input_ids,attention_mask=batch.attention_mask,decoder_input_ids=dec).logits[0,0]
    probs=torch.softmax(logits,dim=-1).tolist()
    cases.append({'text':text,'ids':batch.input_ids[0].tolist(),'reference':probs})

frontier=[]
for budget,k in [(512,1),(1024,2),(2048,4),(4096,8),(8192,16)]:
  program={'language':'PRSL-1','kind':'finite-probabilistic-register','vocab_size':model.config.vocab_size,'cases':[]}
  for c in cases: program['cases'].append(compile_case(c['ids'],c['reference'],k))
  raw=json.dumps(program,separators=(',',':')).encode(); blob=gzip.compress(raw,9)
  errs=[]
  for c,pc in zip(cases,program['cases']): errs.append(tv(c['reference'],decode_program(pc,model.config.vocab_size)))
  frontier.append({'budget':budget,'actual_bytes':len(blob),'top_k':k,'max_tv_error':max(errs),'mean_tv_error':sum(errs)/len(errs),'program':program,'errors':errs})
report={'model':'google/flan-t5-small','task':'first decoder-token distribution','domain':PROMPTS,
        'vocab_size':model.config.vocab_size,
        'reference_distributions':[c['reference'] for c in cases],
        'frontier':[{k:v for k,v in x.items() if k!='program'} for x in frontier]}
(OUT/'flan_register_certificate.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'flan_register_program_4096.json.gz').write_bytes(gzip.compress(json.dumps(frontier[3]['program'],separators=(',',':')).encode(),9))
print(json.dumps(report,indent=2))
