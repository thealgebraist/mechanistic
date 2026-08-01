#!/usr/bin/env python3
"""Bind every safetensors slice to exact binary32 bits and graph references."""
import hashlib,json,struct,zlib
from pathlib import Path
import numpy as np
model=Path("work/google_flan/model.safetensors");raw=model.read_bytes();hlen=struct.unpack("<Q",raw[:8])[0]
header=json.loads(raw[8:8+hlen]);base=8+hlen
graph=json.loads(Path("outputs/flan_full_graph.json").read_text());micro=json.loads(Path("outputs/flan_binary32_microcode.json").read_text())
refs=[]
for i,op in enumerate(graph["ops"]):
 for key in ("weight","q","k","v","o","wi_0","wi_1","wo"):
  if key in op:refs.append((op[key],i,key))
refmap={}
for name,i,key in refs:refmap.setdefault(name,[]).append({"macro_index":i,"argument":key})
records=[]
for name,meta in header.items():
 if name=="__metadata__":continue
 lo,hi=meta["data_offsets"];data=raw[base+lo:base+hi];shape=meta["shape"]
 arr=np.frombuffer(data,dtype="<f4") if meta["dtype"]=="F32" else None
 rec={"name":name,"dtype":meta["dtype"],"shape":shape,"elements":int(np.prod(shape,dtype=np.int64)),
  "relative_begin":lo,"relative_end":hi,"absolute_begin":base+lo,"absolute_end":base+hi,"byte_length":len(data),
  "slice_sha256":hashlib.sha256(data).hexdigest(),"slice_crc32":f"{zlib.crc32(data)&0xffffffff:08x}",
  "all_finite_binary32":bool(np.isfinite(arr).all()) if arr is not None else None,"graph_references":refmap.get(name,[])}
 records.append(rec)
records.sort(key=lambda r:r["relative_begin"])
assert all(r["dtype"]=="F32" and r["byte_length"]==4*r["elements"] and r["all_finite_binary32"] for r in records)
assert records[0]["relative_begin"]==0 and records[-1]["absolute_end"]==len(raw)
assert all(a["relative_end"]==b["relative_begin"] for a,b in zip(records,records[1:]))
assert set(refmap)<=set(r["name"] for r in records)
out={"language":"FLAN-SAFETENSORS-BINARY32-BITS-1","checkpoint_sha256":hashlib.sha256(raw).hexdigest(),"file_bytes":len(raw),
 "header_length":hlen,"data_base":base,"tensor_count":len(records),"all_tensors_f32":True,"all_values_finite":True,
 "graph_sha256":hashlib.sha256(Path("outputs/flan_full_graph.json").read_bytes()).hexdigest(),"microcode_sha256":hashlib.sha256(Path("outputs/flan_binary32_microcode.json").read_bytes()).hexdigest(),
 "graph_weight_reference_occurrences":len(refs),"graph_unique_weight_references":len(refmap),"tensors":records,
 "value_semantics":"each 4-byte little-endian word is an exact IEEE-754 binary32 bit pattern; finite values denote exact dyadic rationals",
 "coverage":"all checkpoint tensors and all graph weight references"}
path=Path("outputs/flan_checkpoint_bit_manifest.json");path.write_text(json.dumps(out,indent=2)+"\n")
tsv=Path("outputs/flan_checkpoint_bit_manifest.tsv");tsv.write_text("name\tdtype\tbegin\tend\telements\tcrc32\trefs\n"+"".join(f"{r['name']}\t{r['dtype']}\t{r['absolute_begin']}\t{r['absolute_end']}\t{r['elements']}\t{r['slice_crc32']}\t{len(r['graph_references'])}\n" for r in records))
print(json.dumps({"artifact":str(path),"tensors":len(records),"references":len(refs),"unique_references":len(refmap),"bytes":len(raw)},indent=2))
