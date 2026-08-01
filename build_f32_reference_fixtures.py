#!/usr/bin/env python3
import struct
from pathlib import Path
import numpy as np
rng=np.random.default_rng(20260731)
def bits(x):return struct.unpack("<I",np.float32(x).tobytes())[0]
def val(u):return np.frombuffer(struct.pack("<I",int(u)),dtype="<f4")[0]
us=[]
while len(us)<4096:
 u=int(rng.integers(0,2**32,dtype=np.uint32));x=val(u)
 if np.isfinite(x):us.append(u)
rows=[]
with np.errstate(all="ignore"):
 for a,b in zip(us[::2],us[1::2]):
  x,y=val(a),val(b)
  rows.append((a,b,bits(np.float32(x+y)),bits(np.float32(x-y)),bits(np.float32(x*y)),bits(np.float32(x/y))))
Path("outputs/f32_reference_pairs.tsv").write_text("a\tb\tadd\tsub\tmul\tdiv\n"+"".join("\t".join(f"{u:08x}" for u in r)+"\n" for r in rows))
dots=[]
for _ in range(64):
 a=rng.normal(0,2,64).astype(np.float32);b=rng.normal(0,2,64).astype(np.float32);s=np.float32(0)
 for x,y in zip(a,b):s=np.float32(s+np.float32(x*y))
 dots.append((",".join(f"{bits(x):08x}" for x in a),",".join(f"{bits(x):08x}" for x in b),f"{bits(s):08x}"))
Path("outputs/f32_reference_dots.tsv").write_text("a\tb\tdot\n"+"".join("\t".join(r)+"\n" for r in dots))
print({"pairs":len(rows),"dots":len(dots)})
