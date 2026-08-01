import argparse,gzip,json
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--source',default='outputs/flan_domain32_program.json.gz'); ap.add_argument('--output',default='outputs/flan_domain32_quotient.json'); a=ap.parse_args()
src=json.loads(gzip.decompress(Path(a.source).read_bytes())); states=src['states']; delta=.05
by={(s['prompt_id'],tuple(s['stack'])):i for i,s in enumerate(states)}
def tv(a,b):
 d={i:q for i,q in a['emit']}; e={i:q for i,q in b['emit']}; keys=set(d)|set(e)
 return .5*sum(abs(d.get(k,0)-e.get(k,0))/65535 for k in keys)+.5*abs(a['other']-b['other'])/65535
children=[]
for s in states:
 c=[]
 for tok,_ in s['emit'][:src['branching_source']]:
  j=by.get((s['prompt_id'],tuple(s['stack'])+(tok,)))
  if j is not None:c.append((tok,j))
 children.append(tuple(c))
block=[None]*len(states); groups=[]
for depth in range(src['horizon'],-1,-1):
 for i,s in enumerate(states):
  if s['depth']!=depth:continue
  for bid,g in enumerate(groups):
   j=g[0]; t=states[j]
   candidate=g+[i]
   avg={}
   for z in candidate:
    for token,value in states[z]['emit']: avg[token]=avg.get(token,0)+value/len(candidate)
   mixture_top=tuple(token for token,_ in sorted(avg.items(),key=lambda kv:kv[1],reverse=True)[:2])
   support_ok = depth == src['horizon']-1 or mixture_top == tuple(x for x,_ in children[j])
   if t['depth']==depth and tv(s,t)<=delta and tuple(x for x,_ in children[i])==tuple(x for x,_ in children[j]) and tuple(x for x,_ in s['emit'][:2])==tuple(x for x,_ in t['emit'][:2]) and support_ok and all(block[x]==block[y] for (_,x),(_,y) in zip(children[i],children[j])):
    g.append(i); block[i]=bid; break
  else: block[i]=len(groups); groups.append([i])
qstates=[]
for bid,g in enumerate(groups):
 i=g[0]; s=states[i]; sums={}; other=0
 for j in g:
  for token,value in states[j]['emit']: sums[token]=sums.get(token,0)+value/len(g)
  other += states[j]['other']/len(g)
 emit=[(token,int(round(value))) for token,value in sorted(sums.items(),key=lambda kv:kv[1],reverse=True)[:8]]
 used=sum(v for _,v in emit); other=65535-used
 qstates.append({'id':bid,'depth':s['depth'],'members':g,'representative':i,'belief_weights':[1/len(g)]*len(g),'emit':emit,'other':other,'next':[(tok,block[j]) for tok,j in children[i]]})
maxlocal=max(tv(states[i],qstates[bid]) for bid,g in enumerate(groups) for i in g)
out={'language':'PRSL-STACK-1-APPROX','prompts':src['prompts'],'source_states':len(states),'quotient_states':len(groups),'delta':delta,'max_local_tv':maxlocal,'horizon':src['horizon'],'horizon_bound':min(1.0,src['horizon']*maxlocal),'roots':{str(s['prompt_id']):block[i] for i,s in enumerate(states) if s['depth']==0},'states':qstates}
Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({**out,'horizon_bound':min(1,src['horizon']*maxlocal)},indent=2))
