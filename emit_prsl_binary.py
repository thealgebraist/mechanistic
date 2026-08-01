"""Serialize the certified PRSL quotient in a compact fixed-width format."""
import argparse, json, struct, hashlib
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--quotient',default='outputs/flan_domain32_quotient.json'); ap.add_argument('--output',default='outputs/flan_domain32.prslb'); a=ap.parse_args()
q=json.loads(Path(a.quotient).read_text())
out=bytearray(); out += b'PRSL1\0'; out += struct.pack('<HHBB',len(q['states']),len(q['prompts']),q['horizon'],2)
for i in range(len(q['prompts'])): out += struct.pack('<H',int(q['roots'][str(i)]))
for s in q['states']:
 em=s['emit'][:2]; out += struct.pack('<BH',s['depth'],65535-sum(m for _,m in em))
 for t,m in s['emit'][:2]: out += struct.pack('<HH',t,m)
 for t,j in s['next'][:2]: out += struct.pack('<HH',t,j)
 for _ in range(2-len(s['next'][:2])): out += struct.pack('<HH',0,0)
Path(a.output).write_bytes(out)
print({'artifact':a.output,'bytes':len(out),'sha256':hashlib.sha256(out).hexdigest(),'states':len(q['states']),'prompts':len(q['prompts'])})
