#!/usr/bin/env python3
"""Conservative IEEE binary32 transfer for the final FLAN lm_head matmul."""
import hashlib, json
from decimal import Decimal, ROUND_CEILING, getcontext
from pathlib import Path

getcontext().prec = 100
graph_path = Path("outputs/flan_full_graph.json")
graph = json.loads(graph_path.read_text())
reachable_path = Path("outputs/flan_reachable_state_bounds.json")
reachable = json.loads(reachable_path.read_text())
add_path = Path("outputs/flan_ieee_add_transfers.json")
abstract = json.loads(add_path.read_text())
convex_path = Path("outputs/flan_convex_geometry_certificate.json")
convex = json.loads(convex_path.read_text())

norm = {}
shape = {}
for line in Path("outputs/flan_reachable_norm_manifest.tsv").read_text().splitlines()[1:]:
    f = line.split("\t"); norm[f[0]] = Decimal(f[6]); shape[f[0]] = (int(f[4]), int(f[5]))
env = {x["output"]: Decimal(x["abs_bound"]) for x in abstract["op_output_bounds"]}
matches = [(i,o) for i,o in enumerate(graph["ops"]) if o["op"] == "MATMUL"]
assert len(matches) == 1
index, op = matches[0]
weight = op["weight"]; rows, n = shape[weight]
B = env[op["input"]]; W = norm[weight]
u = Decimal(2) ** -24
k = 2 * n
gamma = Decimal(k) * u / (Decimal(1) - Decimal(k) * u)
eta2 = Decimal(2) ** -150
# Valid for any fixed reduction tree with no more than n products and n adds;
# the additive term conservatively charges one half-min-subnormal per operation.
rounding_product_bound = Decimal(str(convex["logit_abs_bound_weighted_ellipsoid"]))
rounding_real = gamma * rounding_product_bound + Decimal(k) * eta2
scale = int(abstract["error_units_per_real"])
gain = int(W.to_integral_value(rounding=ROUND_CEILING))
bias = int((rounding_real * scale).to_integral_value(rounding=ROUND_CEILING))
record = {
    "index": index, "opcode": "MATMUL", "weight": weight,
    "rows": rows, "dot_length": n, "operation_count_bound": k,
    "weight_max_row_l1": str(W), "input_abs_bound": str(B),
    "unit_roundoff": str(u), "gamma_k": str(gamma),
    "convex_rounding_product_bound": str(rounding_product_bound),
    "error_units_per_real": scale, "gain": gain, "bias_units": bias,
    "bias_real_upper": str(Decimal(bias)/scale),
    "theorem": "e_out <= ceil(||W||_inf)*e_in + ceil(scale*(gamma_(2n)*sqrt(d)*max_row_l2(W_row .* rms_weight) + 2n*2^-150))/scale",
    "source_contract": "binary32 RNE multiply/reduction, at most 2n rounded operations per output; no overflow",
    "target_contract": "exact-real checkpoint matrix multiplication",
    "status": "CERTIFIED_UNDER_IEEE754_RNE_REDUCTION_CONTRACT",
}
payload=json.dumps(record,sort_keys=True,separators=(",",":")).encode(); record["transfer_sha256"]=hashlib.sha256(payload).hexdigest()
out={"language":"FLAN-IEEE754-MATMUL-TRANSFER-1",
     "checkpoint_sha256":graph["checkpoint_sha256"],
     "graph_sha256":hashlib.sha256(graph_path.read_bytes()).hexdigest(),
     "reachable_bounds_sha256":hashlib.sha256(reachable_path.read_bytes()).hexdigest(),
     "convex_geometry_sha256":hashlib.sha256(convex_path.read_bytes()).hexdigest(),
     "transfer":record}
path=Path("outputs/flan_ieee_matmul_transfer.json"); path.write_text(json.dumps(out,indent=2)+"\n")
tsv=Path("outputs/flan_ieee_matmul_transfer.tsv")
tsv.write_text("index\tdot_length\tweight_norm\tinput_bound\trounding_product_bound\tscale\tgain\tbias_units\n"+
               f"{index}\t{n}\t{W}\t{B}\t{rounding_product_bound}\t{scale}\t{gain}\t{bias}\n")
print(json.dumps({"artifact":str(path),"index":index,"gain":gain,"bias_units":bias,"gamma":str(gamma)},indent=2))
