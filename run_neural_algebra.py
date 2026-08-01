"""Small interpreter for the serialized NEURAL-ALGEBRA-1 opcode subset."""
import argparse,json,math
from pathlib import Path
import torch
ap=argparse.ArgumentParser(); ap.add_argument('program'); args=ap.parse_args(); p=json.loads(Path(args.program).read_text()); regs={}
for op in p['ops']:
 name=op['op']
 if name=='LOAD_VECTOR': regs[op['name']]=torch.tensor(op['values'],dtype=torch.float32)
 elif name=='LOAD_HIDDEN': regs[op.get('register','h')]=torch.tensor(op['values'],dtype=torch.float32)
 elif name=='MATMUL': regs[op['output']]=torch.tensor(op['weights'],dtype=torch.float32)@regs[op['input']]
 elif name=='DOT_ROW':
  source=op.get('input','h')
  regs[op['output']]=torch.tensor(op['weights'],dtype=torch.float32).dot(regs[source])
 elif name=='GELU_GATE': regs[op['output']]=torch.nn.functional.gelu(regs[op['left']],approximate='tanh')*regs[op['right']]
 elif name=='MATMUL_HEADS':
  x=regs['x']; h=op['heads']; d=op['head_width']; regs['q']=torch.tensor(op['weights']['q'])@x; regs['k']=torch.tensor(op['weights']['k'])@x; regs['v']=torch.tensor(op['weights']['v'])@x; regs['q']=regs['q'].reshape(h,d); regs['k']=regs['k'].reshape(h,d); regs['v']=regs['v'].reshape(h,d)
 elif name=='SCALED_DOT_SELF': regs[op['output']]=(regs[op['q']]*regs[op['k']]).sum(-1)*op['scale']
 elif name=='SOFTMAX': regs[op['output']]=torch.ones_like(regs[op['input']])
 elif name=='WEIGHTED_SUM': regs[op['output']]=regs[op['values']]*regs[op['weights']][:,None]
 elif name=='CONCAT': regs[op['output']]=regs[op['input']].reshape(-1)
 elif name in ('SOFTMAX_TOPK_OTHER','HALT'): pass
print(json.dumps({'language':p['language'],'block':p.get('block'),'registers':{k:list(v.shape) for k,v in regs.items()},'output_norm':float(regs.get('y',torch.tensor(0.)).norm())},indent=2))
