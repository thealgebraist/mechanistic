"""Verify the symbolic full FLAN graph against the actual safetensors file."""
import hashlib,json
from pathlib import Path
from safetensors import safe_open
graph=json.loads(Path('outputs/flan_full_graph.json').read_text()); model=Path('work/google_flan/model.safetensors')
with safe_open(str(model),framework='np') as f: tensors=set(f.keys()); shapes={k:f.get_tensor(k).shape for k in f.keys()}
refs=[]
for op in graph['ops']:
 for key in ('weight','q','k','v','o','wi_0','wi_1','wo'):
  if key in op: refs.append((op[key],key))
missing=sorted({x for x,_ in refs if x not in tensors}); assert not missing, missing
assert graph['counts']['opcodes']==len(graph['ops'])
produced={'encoder_tokens','decoder_tokens'}
unknown=[]
for i,op in enumerate(graph['ops']):
 for key in ('input','left','right','memory','distribution'):
  if key in op and op[key] not in produced and op[key] not in ('encoder_tokens','decoder_tokens'): unknown.append((i,op['op'],key,op[key]))
 for key in ('output','next_cache'):
  if key in op: produced.add(op[key])
assert not unknown, unknown[:8]
h=hashlib.sha256(model.read_bytes()).hexdigest(); assert h==graph['checkpoint_sha256']
print(json.dumps({'certificate':'FULL_GRAPH_TENSOR_REFERENCES_OK','opcodes':len(graph['ops']),'weight_references':len(refs),'unique_tensors':len(set(x for x,_ in refs)),'checkpoint_sha256':h,'tensor_count':len(tensors)},indent=2))
