"""Construct and check a conservative approximate PRSL quotient."""
import gzip,json,itertools
from pathlib import Path

src=json.loads(gzip.decompress(Path('outputs/flan_stack_program_16k.json.gz').read_bytes()))
states=src['states']; V=src['vocab_size']; delta=.05
def tv(a,b):
 d={i:q for i,q in a['emit']}; e={i:q for i,q in b['emit']}; keys=set(d)|set(e)
 explicit=.5*sum(abs(d.get(k,0)-e.get(k,0))/65535 for k in keys)
 # OTHER is an explicit normalized outcome, not discarded mass.
 otherA=a['other']; otherB=b['other']
 return explicit + .5*abs(otherA-otherB)/65535
by={(s['prompt_id'],tuple(s['stack'])):i for i,s in enumerate(states)}
children=[]
for s in states:
 c=[]
 for tok,_ in s['emit'][:src['branching_source']]:
  j=by.get((s['prompt_id'],tuple(s['stack'])+(tok,)))
  if j is not None:c.append((tok,j))
 children.append(tuple(c))
block=[None]*len(states); blocks=[]
for depth in range(src['horizon'],-1,-1):
 for i,s in enumerate(states):
  if s['depth']!=depth:continue
  placed=False
  for bid,g in enumerate(blocks):
   j=g[0]; t=states[j]
   if t['depth']!=depth or tv(s,t)>delta:continue
   if tuple(tok for tok,_ in children[i]) != tuple(tok for tok,_ in children[j]):continue
   if any(block[x]!=block[y] for (_,x),(_,y) in zip(children[i],children[j])):continue
   g.append(i);block[i]=bid;placed=True;break
  if not placed:
   block[i]=len(blocks);blocks.append([i])
quot=[]
for bid,g in enumerate(blocks):
 i=g[0]; s=states[i]
 quot.append({'id':bid,'depth':s['depth'],'representative':i,'emit':s['emit'],'other':s['other'],'next':[(tok,block[j]) for tok,j in children[i]]})
max_local=max(tv(states[i],states[g[0]]) for g in blocks for i in g)
out={'language':'PRSL-STACK-1-APPROX','source_states':len(states),'quotient_states':len(quot),'delta_fixed_point':delta,'max_local_tv':max_local,'horizon':src['horizon'],'states':quot}
out['roots']={str(s['prompt_id']):block[i] for i,s in enumerate(states) if s['depth']==0}
Path('outputs/flan_stack_approx_quotient.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'source_states':len(states),'quotient_states':len(quot),'merged':len(states)-len(quot),'max_local_tv':max_local,'horizon_bound':src['horizon']*max_local},indent=2))
