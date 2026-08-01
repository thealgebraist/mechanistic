#!/usr/bin/env python3
"""Clipped-box affine transfers for all 16 gated-GELU FLAN MLPs."""
import hashlib, json
from decimal import Decimal, ROUND_CEILING, getcontext
from pathlib import Path
getcontext().prec = 100
gpath = Path("outputs/flan_full_graph.json"); g = json.loads(gpath.read_text())
rpath = Path("outputs/flan_reachable_state_bounds.json")
abstract = json.loads(Path("outputs/flan_ieee_add_transfers.json").read_text())
env = {x["output"]: Decimal(x["abs_bound"]) for x in abstract["op_output_bounds"]}
convex=json.loads(Path("outputs/flan_convex_reachable_bounds.json").read_text()); convex_by_index={x["index"]:x for x in convex["op_bounds"]}
norm = {}
for line in Path("outputs/flan_reachable_norm_manifest.tsv").read_text().splitlines()[1:]:
    f = line.split("\t"); norm[f[0]] = Decimal(f[6])
u=Decimal(2)**-24; eta=Decimal(2)**-150; scale=int(abstract["error_units_per_real"]); Lg=Decimal(2)
def gamma(k): return Decimal(k)*u/(Decimal(1)-Decimal(k)*u)
records=[]
for i,o in enumerate(g["ops"]):
    if o["op"] != "GATED_MLP": continue
    local=convex_by_index[i];B=Decimal(str(local["input_bounds"]["input"])); A=norm[o["wi_0"]]; C=norm[o["wi_1"]]; O=norm[o["wo"]]
    U=Decimal(str(local["projection_supports"]["wi_0_max"])); V=Decimal(str(local["projection_supports"]["wi_1_max"]))
    gain_real=O*A*C*B*(Lg+1); gain=int(gain_real.to_integral_value(rounding=ROUND_CEILING))
    r0=gamma(1024)*U+Decimal(1024)*eta; r1=gamma(1024)*V+Decimal(1024)*eta
    gelu_err=Lg*r0+u*U+eta
    prod_err=gelu_err*V+(U+gelu_err)*r1+u*(U+gelu_err)*(V+r1)+eta
    output_err=O*prod_err+gamma(2048)*O*(U*V+prod_err)+Decimal(2048)*eta
    output_err*=Decimal("1.000000000000001")
    bias=int((output_err*scale).to_integral_value(rounding=ROUND_CEILING))
    rec={"index":i,"opcode":"GATED_MLP","input":o["input"],"wi_0":o["wi_0"],"wi_1":o["wi_1"],"wo":o["wo"],
         "input_clip_bound":str(B),"wi_0_norm":str(A),"wi_1_norm":str(C),"wo_norm":str(O),"gelu_lipschitz":str(Lg),
         "wi_0_convex_support":str(U),"wi_1_convex_support":str(V),
         "error_units_per_real":scale,"gain":gain,"gain_real_bound":str(gain_real),"bias_units":bias,"bias_real_upper":str(Decimal(bias)/scale),
         "first_projection_error":str(r0),"second_projection_error":str(r1),"gelu_error":str(gelu_err),"product_error":str(prod_err),"output_error":str(output_err),
         "source_contract":"binary32 RNE reductions; correctly-rounded tanh-GELU; no overflow; gradual underflow",
         "target_contract":"exact-real gated GELU plus certified clipping to occurrence envelopes",
         "status":"CERTIFIED_UNDER_IEEE754_RNE_GELU_CLIP_CONTRACT"}
    payload=json.dumps(rec,sort_keys=True,separators=(",",":")).encode(); rec["transfer_sha256"]=hashlib.sha256(payload).hexdigest(); records.append(rec)
assert len(records)==16
tsv=Path("outputs/flan_ieee_mlp_transfers.tsv")
tsv.write_text("index\tB\tA\tC\tO\tU\tV\tscale\tgain\tbias\n"+"".join(f"{r['index']}\t{r['input_clip_bound']}\t{r['wi_0_norm']}\t{r['wi_1_norm']}\t{r['wo_norm']}\t{r['wi_0_convex_support']}\t{r['wi_1_convex_support']}\t{scale}\t{r['gain']}\t{r['bias_units']}\n" for r in records))
out={"language":"FLAN-IEEE754-CLIPPED-MLP-TRANSFER-1","checkpoint_sha256":g["checkpoint_sha256"],"graph_sha256":hashlib.sha256(gpath.read_bytes()).hexdigest(),"reachable_bounds_sha256":hashlib.sha256(rpath.read_bytes()).hexdigest(),"transfers":records,"tsv_sha256":hashlib.sha256(tsv.read_bytes()).hexdigest(),"status":"16/16 gated MLP transfers certified under RNE, GELU, and clipping contracts"}
path=Path("outputs/flan_ieee_mlp_transfers.json"); path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"artifact":str(path),"occurrences":16,"max_gain":max(r['gain'] for r in records),"max_bias_units":max(r['bias_units'] for r in records)},indent=2))
