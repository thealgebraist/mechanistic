#!/usr/bin/env python3
"""Propagate certified local errors through the actual FLAN register DAG.

This is an audit, not a new transfer theorem.  It combines the occurrence-level
affine certificates with the convex reachable diameters.  Each cap is valid for
the clipped target contracts: two points in [-B,B] are at sup distance <= 2B.
"""
import hashlib
import json
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 100
SCALE = Decimal(2**40)
graph_path = Path("outputs/flan_full_graph.json")
ledger_path = Path("outputs/flan_backend_error_obligations.json")
convex_path = Path("outputs/flan_convex_reachable_bounds.json")
fused_path = Path("outputs/flan_fused_rms_attention_transfers.json")
graph = json.loads(graph_path.read_text())
ledger = json.loads(ledger_path.read_text())
convex = json.loads(convex_path.read_text())
obligation = {x["index"]: x for x in ledger["obligations"]}
bounds = {x["index"]: x for x in convex["op_bounds"]}
fused = {x["index"]: x for x in json.loads(fused_path.read_text())["transfers"]} if fused_path.exists() else {}

errors = {}
rows = []
first_cap = None
for index, op in enumerate(graph["ops"]):
    kind = op["op"]
    cert = obligation[index]
    gain = Decimal(cert["local_gain"])
    bias = Decimal(cert["local_bias_units"])
    names = [op[k] for k in ("input", "left", "right", "memory", "distribution") if k in op]
    incoming = [errors.get(name, Decimal(0)) for name in names]

    # ADD's transfer is 2*max(e_left,e_right)+b in the generic theorem.  The
    # primitive proof also gives the sharper e_left+e_right+b DAG recurrence.
    fused_rec = fused.get(index)
    fused_applicable = bool(fused_rec and rows and rows[-1]["opcode"] == "RMSNORM" and
                            all(Decimal(x) == 0 for x in rows[-1]["input_error_units"]))
    if fused_applicable:
        pre = Decimal(fused_rec["bias_units"])
        recurrence = "RMSRadial_structured_zero_input_fusion"
    elif kind == "ADD":
        pre = sum(incoming, bias)
        recurrence = "sum_inputs_plus_bias"
    elif incoming:
        pre = gain * max(incoming) + bias
        recurrence = "gain_times_max_input_plus_bias"
    else:
        pre = bias
        recurrence = "bias_only"

    out_name = op.get("output")
    cap = None
    if index in bounds:
        cap = SCALE if kind == "SOFTMAX" else Decimal(str(2 * bounds[index]["abs_bound"])) * SCALE
    post = min(pre, cap) if cap is not None else pre
    capped = cap is not None and pre > cap
    if capped and first_cap is None:
        first_cap = index
    if out_name is not None:
        errors[out_name] = post
    rows.append({
        "index": index, "opcode": kind, "output": out_name,
        "input_registers": names, "input_error_units": [str(x) for x in incoming],
        "recurrence": recurrence, "gain": int(gain), "bias_units": int(bias),
        "pre_cap_error_units": str(pre), "diameter_cap_units": str(cap) if cap is not None else None,
        "post_cap_error_units": str(post), "cap_active": capped,
        "post_cap_real": str(post / SCALE), "structured_fusion_active": fused_applicable,
    })

softmax = rows[127]
out = {
    "language": "FLAN-CONVEX-DAG-ERROR-AUDIT-1",
    "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
    "obligations_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
    "convex_bounds_sha256": hashlib.sha256(convex_path.read_bytes()).hexdigest(),
    "fused_rms_attention_sha256": hashlib.sha256(fused_path.read_bytes()).hexdigest() if fused_path.exists() else None,
    "unit_scale": int(SCALE),
    "method": "occurrence affine transfers on register DAG, with convex reachable diameter caps",
    "scope": ledger["obligations"][0]["quantification"],
    "rows": rows,
    "summary": {
        "first_active_diameter_cap_index": first_cap,
        "first_active_diameter_cap_opcode": rows[first_cap]["opcode"] if first_cap is not None else None,
        "active_diameter_caps": sum(x["cap_active"] for x in rows),
        "structured_fusions_used": sum(x["structured_fusion_active"] for x in rows),
        "softmax_input_error_real": str(Decimal(softmax["input_error_units"][0]) / SCALE) if softmax["input_error_units"] else None,
        "softmax_pre_cap_tv": str(Decimal(softmax["pre_cap_error_units"]) / SCALE),
        "certified_one_step_tv": str(Decimal(softmax["post_cap_error_units"]) / SCALE),
        "nontrivial_tv": Decimal(softmax["post_cap_error_units"]) < SCALE,
    },
    "interpretation": "The audit localizes saturation; it does not strengthen any primitive transfer contract.",
}
path = Path("outputs/flan_convex_error_bottleneck.json")
path.write_text(json.dumps(out, indent=2) + "\n")
tsv = Path("outputs/flan_convex_error_bottleneck.tsv")
tsv.write_text("index\topcode\tpre\tcap\tpost\tactive\n" + "".join(
    f'{r["index"]}\t{r["opcode"]}\t{r["pre_cap_error_units"]}\t{r["diameter_cap_units"] or "-"}\t{r["post_cap_error_units"]}\t{int(r["cap_active"])}\n'
    for r in rows))
print(json.dumps({"artifact": str(path), **out["summary"]}, indent=2))
