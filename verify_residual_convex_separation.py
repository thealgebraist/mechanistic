#!/usr/bin/env python3
import json
from pathlib import Path
x=json.loads(Path("outputs/flan_first_residual_convex_separation.json").read_text())
assert x["token_count"]==32128 and x["minimum_box_distance_to_zero"]==0
assert x["tokens_with_positive_box_distance"]==0 and x["result"]=="UNKNOWN" and x["not_a_counterexample"]
print("FLAN_RESIDUAL_BOX_SEPARATION_UNKNOWN tokens=32128")
