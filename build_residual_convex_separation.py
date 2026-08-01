#!/usr/bin/env python3
"""Audit whether a coordinate-box convex domain separates the first residual from zero."""
import hashlib,json
from pathlib import Path
import torch
from safetensors.torch import load_file
model=Path("work/google_flan/model.safetensors");W=load_file(str(model),device="cpu")
e=W["shared.weight"].double();rw=W["encoder.block.0.layer.0.layer_norm.weight"].double()
v=W["encoder.block.0.layer.0.SelfAttention.v.weight"].double();o=W["encoder.block.0.layer.0.SelfAttention.o.weight"].double()
vs=(512.0**.5)*torch.linalg.vector_norm(v*rw.unsqueeze(0),2,1)
a=o.abs()@vs
distance=torch.linalg.vector_norm(torch.relu(e.abs()-a.unsqueeze(0)),2,1)
projection=(e.square().sum(1)-(e.abs()*a).sum(1))/torch.linalg.vector_norm(e,2,1)
out={"language":"FLAN-FIRST-RESIDUAL-CONVEX-SEPARATION-1","checkpoint_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),
 "residual":"shared[token] + encoder.block.0.self_attention(RMSNorm(shared[sequence]))",
 "relaxation":"attention output coordinate box from weighted-RMS V supports and absolute O transport",
 "token_count":e.shape[0],"attention_coordinate_abs_min":float(a.min()),"attention_coordinate_abs_max":float(a.max()),
 "minimum_box_distance_to_zero":float(distance.min()),"tokens_with_positive_box_distance":int((distance>0).sum()),
 "minimum_embedding_direction_witness":float(projection.min()),"tokens_with_positive_embedding_direction_witness":int((projection>0).sum()),
 "result":"UNKNOWN","reason":"the coordinate-box relaxation contains -embedding for every token; it cannot certify a positive residual norm",
 "next_domain":"multihead convex-hull support oracle or affine zonotope retaining shared attention coefficients",
 "not_a_counterexample":True}
path=Path("outputs/flan_first_residual_convex_separation.json");path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"artifact":str(path),"result":out["result"],"min_distance":out["minimum_box_distance_to_zero"],"positive_tokens":out["tokens_with_positive_box_distance"]},indent=2))
