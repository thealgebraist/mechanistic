"""Select the largest certified PRSL emission support under a byte cap."""
import argparse,gzip,json,hashlib
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument('--bytes',type=int,required=True); ap.add_argument('--out',default='outputs/budget_prsl_program.json.gz'); args=ap.parse_args()
src=json.loads(gzip.decompress(Path('outputs/flan_stack_program_16k.json.gz').read_bytes()))
roots={str(s['prompt_id']):i for i,s in enumerate(src['states']) if s['depth']==0}
checkpoint=hashlib.sha256(Path('work/google_flan/model.safetensors').read_bytes()).hexdigest()
best=None
for k in range(1,9):
 p={'language':'PRSL-STACK-1','vocab_size':src['vocab_size'],'horizon':src['horizon'],'branching_source':src['branching_source'],'roots':roots,'source_checkpoint_sha256':checkpoint,'states':[]}
 for s in src['states']:
  emit=s['emit'][:k]; p['states'].append({'prompt_id':s['prompt_id'],'depth':s['depth'],'stack':s['stack'],'emit':emit,'other':65535-sum(v for _,v in emit)})
 blob=gzip.compress(json.dumps(p,separators=(',',':')).encode(),9)
 if len(blob)<=args.bytes: best=(k,p,blob)
if best is None:
 print(json.dumps({'status':'NO_PROGRAM_FITS','budget':args.bytes})); raise SystemExit(2)
k,p,blob=best; Path(args.out).write_bytes(blob)
print(json.dumps({'status':'OK','budget':args.bytes,'actual_bytes':len(blob),'top_k':k,'states':len(p['states']),'output':args.out},indent=2))
