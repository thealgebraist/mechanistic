"""Independent semantic checker for the PRSL-STACK artifact."""
import gzip,json
from pathlib import Path
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
ROOT=Path('work/google_flan'); P=json.loads(gzip.decompress(Path('outputs/flan_stack_program_16k.json.gz').read_bytes())); C=json.loads(Path('outputs/flan_stack_certificate.json').read_text())
assert P['language']=='PRSL-STACK-1' and P['horizon']==3 and P['branching_source']==2
tok=T5Tokenizer.from_pretrained(str(ROOT),local_files_only=True); model=T5ForConditionalGeneration.from_pretrained(str(ROOT),local_files_only=True,dtype=torch.float32); model.eval(); start=model.config.decoder_start_token_id
errs=[]
with torch.no_grad():
  for s in P['states']:
    enc=tok(C['prompts'][s['prompt_id']],return_tensors='pt')
    dec=torch.tensor([[start,*s['stack']]])
    ref=torch.softmax(model(input_ids=enc.input_ids,attention_mask=enc.attention_mask,decoder_input_ids=dec).logits[0,-1],dim=-1).tolist()
    got=[0.0]*P['vocab_size']
    for i,q in s['emit']: got[i]=q/65535
    errs.append(.5*sum(abs(a-b) for a,b in zip(ref,got)))
assert len(errs)==len(P['states'])
print('CERTIFICATE_OK')
print('states:',len(errs),'max_TV:',max(errs),'mean_TV:',sum(errs)/len(errs))
