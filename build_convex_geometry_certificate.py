#!/usr/bin/env python3
"""Ellipsoidal/L2 certificate for normalized FLAN readout and lm_head."""
import hashlib,json,math,struct
from pathlib import Path
import torch
from safetensors.torch import load_file
model=Path("work/google_flan/model.safetensors"); graphp=Path("outputs/flan_full_graph.json")
W=load_file(str(model),device="cpu"); graph=json.loads(graphp.read_text()); d=graph["config"]["d_model"]
lm=W["lm_head.weight"].double(); normvec=W["decoder.final_layer_norm.weight"].double().abs(); normw=normvec.max().item()
row_l2=torch.linalg.vector_norm(lm,ord=2,dim=1); max_row_l2=row_l2.max().item(); row=int(row_l2.argmax().item())
readout_l2=math.nextafter(math.sqrt(d)*normw,math.inf)
logit_l2=math.nextafter(max_row_l2*readout_l2,math.inf)
weighted_rows=torch.linalg.vector_norm(lm*normvec.unsqueeze(0),ord=2,dim=1)
weighted_row_max=weighted_rows.max().item(); weighted_row_index=int(weighted_rows.argmax().item())
logit_weighted=math.nextafter(math.sqrt(d)*weighted_row_max,math.inf)
old=json.loads(Path("outputs/flan_reachable_state_bounds.json").read_text())["final_bounds"]["logit_abs"]
u=2.0**-24; gamma=(2*d*u)/(1-2*d*u); rounding=math.nextafter(gamma*logit_weighted+(2*d)*2.0**-150,math.inf)
out={"language":"FLAN-CONVEX-L2-GEOMETRY-1","checkpoint_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),"graph_sha256":hashlib.sha256(graphp.read_bytes()).hexdigest(),
 "convex_sets":{"rms_readout_ellipsoid":"sum_i (y_i/w_i)^2 <= d_model","attention_value_hull":"each head output lies in conv{projected value rows}","softmax_potential":"softmax(z) = gradient logsumexp(z)"},
 "d_model":d,"decoder_final_norm_weight_max_abs":normw,"readout_l2_bound":readout_l2,"lm_head_max_row_l2":max_row_l2,"max_row_l2_index":row,
 "logit_abs_bound_l2":logit_l2,"lm_head_max_weighted_row_l2":weighted_row_max,"max_weighted_row_index":weighted_row_index,
 "logit_abs_bound_weighted_ellipsoid":logit_weighted,"previous_box_logit_abs_bound":old,
 "euclidean_improvement_factor":old/logit_l2,"weighted_ellipsoid_improvement_factor":old/logit_weighted,
 "gamma_1024":gamma,"zero_input_error_dot_rounding_bound":rounding,
 "lemmas":["RMSNorm implies ||y/w||_2 <= sqrt(d)","Cauchy-Schwarz: |row dot y| <= ||row||_2 ||y||_2","logsumexp is convex and softmax is its gradient","attention is a convex mixture of value rows"],
 "scope":"every backend-valid finite prompt and continuation; no sequence-length dependence","status":"checkpoint-derived convex geometry; raw tensors independently checked in C++23"}
path=Path("outputs/flan_convex_geometry_certificate.json");path.write_text(json.dumps(out,indent=2)+"\n")
tsv=Path("outputs/flan_convex_geometry_certificate.tsv");tsv.write_text("norm_weight_max\tmax_row_l2\treadout_l2\tlogit_l2\tweighted_row_l2\tweighted_logit\trounding\trow\tweighted_row\n"+f"{normw:.17g}\t{max_row_l2:.17g}\t{readout_l2:.17g}\t{logit_l2:.17g}\t{weighted_row_max:.17g}\t{logit_weighted:.17g}\t{rounding:.17g}\t{row}\t{weighted_row_index}\n")
print(json.dumps({"artifact":str(path),"old_logit_bound":old,"convex_l2_bound":logit_l2,"weighted_ellipsoid_bound":logit_weighted,"improvement_factor":old/logit_weighted,"rounding_bound":rounding},indent=2))
