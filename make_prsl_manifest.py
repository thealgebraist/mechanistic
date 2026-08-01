"""Create provenance for the finite-domain FLAN PRSL certificate."""
import gzip, hashlib, json
from pathlib import Path
files=['work/google_flan/config.json','work/google_flan/model.safetensors','work/google_flan/spiece.model','outputs/flan_domain32_program.json.gz','outputs/flan_domain32_quotient.json','outputs/flan_domain32.prslb']
def digest(path):
 h=hashlib.sha256(); size=0
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): size+=len(block); h.update(block)
 return {'bytes':size,'sha256':h.hexdigest()}
manifest={'checkpoint_root':'work/google_flan','artifacts':{p:digest(p) for p in files}}
q=json.loads(Path('outputs/flan_domain32_quotient.json').read_text()); src=json.loads(gzip.decompress(Path('outputs/flan_domain32_program.json.gz').read_bytes()))
manifest['domain']={'prompts':len(src['prompts']),'source_states':len(src['states']),'quotient_states':len(q['states']),'horizon':src['horizon'],'local_tv':q['max_local_tv'],'horizon_bound':q['horizon_bound']}
Path('outputs/flan_prsl_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))
