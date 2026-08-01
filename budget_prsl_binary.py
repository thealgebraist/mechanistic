"""Compile the largest prefix of the certified prompt domain fitting n bytes."""
import argparse, json, struct, hashlib
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--bytes',type=int,required=True); ap.add_argument('--quotient',default='outputs/flan_domain32_quotient.json'); ap.add_argument('--output',default=None); a=ap.parse_args()
q=json.loads(Path(a.quotient).read_text()); by={s['id']:s for s in q['states']}
def reach(root):
 seen=set(); todo=[root]
 while todo:
  i=todo.pop()
  if i in seen: continue
  seen.add(i)
  for _,j in by[i]['next'][:2]: todo.append(j)
 return seen
chosen=[]; states=set()
for pid in range(len(q['prompts'])):
 trial=states|reach(int(q['roots'][str(pid)])); size=6+2*(len(chosen)+1)+19*len(trial)
 if size<=a.bytes: chosen.append(pid); states=trial
roots=[int(q['roots'][str(i)]) for i in chosen]; remap={old:i for i,old in enumerate(sorted(states))}
if not chosen: raise SystemExit(f'no prompt domain fits {a.bytes} bytes')
out=bytearray(b'PRSL1\0'); out+=struct.pack('<HHBB',len(states),len(chosen),q['horizon'],2)
for r in roots: out+=struct.pack('<H',remap[r])
for old in sorted(states):
 s=by[old]; em=s['emit'][:2]; out+=struct.pack('<BH',s['depth'],65535-sum(m for _,m in em))
 for t,m in em: out+=struct.pack('<HH',t,m)
 for _ in range(2-len(em)): out+=struct.pack('<HH',0,0)
 for t,j in s['next'][:2]: out+=struct.pack('<HH',t,remap[j])
 for _ in range(2-len(s['next'][:2])): out+=struct.pack('<HH',0,0)
path=Path(a.output or f'outputs/flan_budget_{a.bytes}.prslb'); path.write_bytes(out)
cert={'artifact':str(path),'requested_bytes':a.bytes,'actual_bytes':len(out),'prompts':chosen,'states':len(states),'horizon':q['horizon'],'sha256':hashlib.sha256(out).hexdigest(),'source':a.quotient}
Path(str(path)+'.json').write_text(json.dumps(cert,indent=2)+'\n'); print(json.dumps(cert,indent=2)); assert len(out)<=a.bytes
