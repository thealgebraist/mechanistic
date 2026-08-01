import argparse,gzip,hashlib,json
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--quotient',required=True); ap.add_argument('--binary',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
def h(p):
 x=Path(p).read_bytes(); return {'bytes':len(x),'sha256':hashlib.sha256(x).hexdigest()}
src=json.loads(gzip.decompress(Path(a.source).read_bytes())); q=json.loads(Path(a.quotient).read_text()); out={'source':h(a.source),'quotient':h(a.quotient),'binary':h(a.binary),'domain':{'prompts':len(src['prompts']),'source_states':len(src['states']),'quotient_states':len(q['states']),'horizon':src['horizon'],'local_tv':q['max_local_tv'],'horizon_bound':q['horizon_bound']}}
Path(a.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
