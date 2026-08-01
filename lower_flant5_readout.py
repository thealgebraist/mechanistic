"""Lower one FLAN-T5 decoder readout into explicit Level-1 opcodes."""
import json,torch
from pathlib import Path
from transformers import T5ForConditionalGeneration,T5Tokenizer
ROOT=Path('work/google_flan'); text='question: What color is the sky? answer:'; topk=8
tok=T5Tokenizer.from_pretrained(str(ROOT),local_files_only=True)
model=T5ForConditionalGeneration.from_pretrained(str(ROOT),local_files_only=True,dtype=torch.float32); model.eval(); enc=tok(text,return_tensors='pt'); dec=torch.tensor([[model.config.decoder_start_token_id]])
with torch.no_grad():
 out=model(input_ids=enc.input_ids,attention_mask=enc.attention_mask,decoder_input_ids=dec,output_hidden_states=True)
 h=out.decoder_hidden_states[-1][0,-1].tolist(); logits=out.logits[0,-1].tolist(); order=sorted(range(len(logits)),key=lambda i:logits[i],reverse=True)[:topk]
 # Level-1 program stores only candidate vocabulary rows; OTHER aggregates the rest.
 ops=[{'op':'LOAD_HIDDEN','register':'h','values':h}]
 for i in order: ops.append({'op':'DOT_ROW','matrix':'lm_head','token_id':i,'weights':model.lm_head.weight[i].tolist(),'output':f'logit_{i}'})
 ops += [{'op':'SOFTMAX_TOPK_OTHER','inputs':[f'logit_{i}' for i in order],'other':'OTHER'},{'op':'HALT'}]
 program={'language':'NEURAL-ALGEBRA-1','source':'google/flan-t5-small','prompt':text,'decoder_prefix':[],'hidden_width':len(h),'candidate_tokens':order,'ops':ops}
Path('outputs/flan_readout_level1.json').write_text(json.dumps(program,separators=(',',':'))+'\n')
def interpret(p):
 vals={}; hidden=p['ops'][0]['values']
 for op in p['ops'][1:]:
  if op['op']=='DOT_ROW': vals[op['output']]=sum(a*b for a,b in zip(hidden,op['weights']))
 return vals
calc=interpret(program); err=max(abs(calc[f'logit_{i}']-logits[i]) for i in order)
print(json.dumps({'candidate_tokens':order,'max_logit_error':err,'program_bytes':Path('outputs/flan_readout_level1.json').stat().st_size},indent=2))
assert err<1e-4
