import gzip,json
from pathlib import Path
src=json.loads(gzip.decompress(Path('outputs/flan_domain32_program.json.gz').read_bytes())); states=src['states']; H=src['horizon']; delta=.001
by={(s['prompt_id'],tuple(s['stack'])):i for i,s in enumerate(states)}
def dist(s): return {i:q/65535 for i,q in s['emit']}|{'OTHER':s['other']/65535}
def tv(a,b): return .5*sum(abs(a.get(k,0)-b.get(k,0)) for k in set(a)|set(b))
children=[]
for s in states:
 c={}
 for tok,_ in s['emit'][:2]:
  j=by.get((s['prompt_id'],tuple(s['stack'])+(tok,)))
  c[tok]=j
 children.append(c)
# Greedy emission clustering at each depth; transition stochasticity is handled
# by the kernel below rather than by requiring identical successor supports.
groups=[]; block=[None]*len(states)
for depth in range(H-1,-1,-1):
 for i,s in enumerate(states):
  if s['depth']!=depth: continue
  d=dist(s)
  for bid,g in enumerate(groups):
   candidate=g+[i]; avg={}
   for z in candidate:
    for t,p in dist(states[z]).items():
     if t!='OTHER': avg[t]=avg.get(t,0)+p/len(candidate)
   mix_support=set(t for t,_ in sorted(avg.items(),key=lambda kv:kv[1],reverse=True)[:2])
   same_support=all(set(t for t,_ in states[z]['emit'][:2])==set(t for t,_ in states[g[0]]['emit'][:2]) for z in candidate)
   if states[g[0]]['depth']==depth and tv(d,dist(states[g[0]]))<=delta and same_support and mix_support==set(t for t,_ in states[g[0]]['emit'][:2]):
    g.append(i); block[i]=bid; break
  else: block[i]=len(groups); groups.append([i])
qstates=[]
for bid,g in enumerate(groups):
 w=1/len(g); mix={}
 for i in g:
  for t,p in dist(states[i]).items(): mix[t]=mix.get(t,0)+w*p
 transitions={}
 top_tokens=[t for t,p in sorted(((t,p) for t,p in mix.items() if t!='OTHER'),key=lambda x:x[1],reverse=True)[:2]]
 projected={t:mix[t] for t in top_tokens}; projected['OTHER']=1-sum(projected.values())
 for t,p in projected.items():
  if t=='OTHER' or states[g[0]]['depth']==H-1: continue
  z={}
  for i in g:
   pi=dist(states[i]).get(t,0); j=children[i].get(t)
   if pi and j is not None: z[block[j]]=z.get(block[j],0)+w*pi
   elif pi: z['OTHER']=z.get('OTHER',0)+w*pi
  if p: transitions[str(t)]={str(k):v/p for k,v in z.items()}
 qstates.append({'id':bid,'depth':states[g[0]]['depth'],'members':g,'weights':[w]*len(g),'emission':projected,'kernel':transitions})
roots={str(s['prompt_id']):block[i] for i,s in enumerate(states) if s['depth']==0}
out={'language':'PRSL-PFA-1','prompts':src['prompts'],'source_states':len(states),'quotient_states':len(qstates),'delta':delta,'horizon':H,'roots':roots,'states':qstates}
Path('outputs/flan_domain32_prob_kernel.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'source_states':len(states),'quotient_states':len(qstates),'merged':len(states)-len(qstates),'max_local_tv':max(tv(dist(states[i]),qstates[bid]['emission']) for bid,g in enumerate(groups) for i in g)},indent=2))
