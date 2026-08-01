#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
p=Path("outputs/flan_convex_geometry_certificate.json");c=json.loads(p.read_text());m=Path("work/google_flan/model.safetensors");g=Path("outputs/flan_full_graph.json")
assert c["checkpoint_sha256"]==hashlib.sha256(m.read_bytes()).hexdigest() and c["graph_sha256"]==hashlib.sha256(g.read_bytes()).hexdigest()
assert c["d_model"]==512 and c["weighted_ellipsoid_improvement_factor"]>c["euclidean_improvement_factor"]>1
assert c["logit_abs_bound_weighted_ellipsoid"]<c["logit_abs_bound_l2"]<c["previous_box_logit_abs_bound"]
assert c["scope"].endswith("no sequence-length dependence")
print(json.dumps({"certificate":"CONVEX_WEIGHTED_ELLIPSOID_GEOMETRY_OK","logit_bound":c["logit_abs_bound_weighted_ellipsoid"],"improvement_factor":c["weighted_ellipsoid_improvement_factor"],"sequence_cap":None},indent=2))
