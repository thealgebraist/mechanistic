#!/usr/bin/env python3
"""Affine binary32-to-real transfer certificates for all FLAN residual ADDs."""
import hashlib
import json
import math
from decimal import Decimal, ROUND_CEILING, getcontext
from pathlib import Path

getcontext().prec = 80
graph_path = Path("outputs/flan_full_graph.json")
reachable_path = Path("outputs/flan_reachable_state_bounds.json")
manifest_path = Path("outputs/flan_reachable_norm_manifest.tsv")
graph = json.loads(graph_path.read_text())
reachable = json.loads(reachable_path.read_text())
convex_path = Path("outputs/flan_convex_reachable_bounds.json")
convex = json.loads(convex_path.read_text()); convex_by_index={x["index"]:x for x in convex["op_bounds"]}

norm = {}
for line in manifest_path.read_text().splitlines()[1:]:
    f = line.split("\t")
    norm[f[0]] = Decimal(f[6])

sqrt_d = Decimal(str(reachable["sqrt_d_model_up"]))
scale = 1 << 40
u = Decimal(2) ** -24
half_min_subnormal = Decimal(2) ** -150
env = {}
adds = []
op_bounds = []

def rms(weight): return sqrt_d * norm[weight]
def linear(weight, x): return norm[weight] * x
def attention(op, value_bound): return norm[op["o"]] * norm[op["v"]] * value_bound
def mlp(op, x): return norm[op["wo"]] * norm[op["wi_0"]] * x * norm[op["wi_1"]] * x

for index, op in enumerate(graph["ops"]):
    kind = op["op"]
    out = op.get("output")
    bound = None
    if kind == "EMBED": bound = norm[op["weight"]]
    elif kind == "RMSNORM": bound = rms(op["weight"])
    elif kind in {"SELF_ATTENTION_SEQUENCE", "SELF_ATTENTION_KV"}:
        bound = attention(op, env[op["input"]])
    elif kind == "CROSS_ATTENTION":
        bound = attention(op, env[op["memory"]])
    elif kind == "GATED_MLP": bound = mlp(op, env[op["input"]])
    elif kind == "MATMUL": bound = linear(op["weight"], env[op["input"]])
    elif kind == "SOFTMAX": bound = Decimal(1)
    elif kind == "ADD":
        local=convex_by_index[index]["input_bounds"]
        left, right = Decimal(str(local["left"])), Decimal(str(local["right"]))
        bound = left + right
        real_bias = u * bound + half_min_subnormal
        bias_units = int((real_bias * scale).to_integral_value(rounding=ROUND_CEILING))
        record = {
            "index": index, "opcode": "ADD", "left": op["left"], "right": op["right"],
            "left_abs_bound": str(left), "right_abs_bound": str(right),
            "error_scale": scale, "gain": 2, "bias_units": bias_units,
            "bias_real_upper": str(Decimal(bias_units) / scale),
            "theorem": "|RN32(a+b)-(a_prime+b_prime)| <= 2*e + 2^-24*(B_left+B_right) + 2^-150",
            "assumption": "IEEE-754 binary32 round-to-nearest-ties-to-even; exact-real target ADD",
            "status": "CERTIFIED_UNDER_IEEE754_RNE",
        }
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        record["transfer_sha256"] = hashlib.sha256(payload).hexdigest()
        adds.append(record)
    if out is not None and bound is not None:
        env[out] = bound
        op_bounds.append({"index": index, "output": out, "abs_bound": str(bound)})

assert len(adds) == 40
tsv_path = Path("outputs/flan_ieee_add_transfers.tsv")
tsv_path.write_text("index\tleft_bound\tright_bound\tscale\tgain\tbias_units\n" + "".join(
    f"{r['index']}\t{r['left_abs_bound']}\t{r['right_abs_bound']}\t{scale}\t2\t{r['bias_units']}\n"
    for r in adds))
out = {
    "language": "FLAN-IEEE754-ADD-TRANSFER-1",
    "checkpoint_sha256": graph["checkpoint_sha256"],
    "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
    "reachable_bounds_sha256": hashlib.sha256(reachable_path.read_bytes()).hexdigest(),
    "convex_reachable_bounds_sha256": hashlib.sha256(convex_path.read_bytes()).hexdigest(),
    "error_units_per_real": scale,
    "source": "IEEE-754 binary32 round-to-nearest-ties-to-even",
    "target": "exact-real probabilistic register machine",
    "transfers": adds,
    "op_output_bounds": convex["op_bounds"],
    "tsv_sha256": hashlib.sha256(tsv_path.read_bytes()).hexdigest(),
    "status": "40/40 ADD occurrences certified under the explicit IEEE-754 source contract",
}
path = Path("outputs/flan_ieee_add_transfers.json")
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"artifact": str(path), "add_transfers": len(adds), "scale": scale,
                  "max_bias_units": max(x["bias_units"] for x in adds)}, indent=2))
