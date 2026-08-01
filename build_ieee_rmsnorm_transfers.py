#!/usr/bin/env python3
"""Universal affine transfers for all 42 FLAN RMSNorm occurrences."""
import hashlib,json,struct
from decimal import Decimal,ROUND_CEILING,getcontext
from pathlib import Path
getcontext().prec=100
gpath=Path("outputs/flan_full_graph.json"); graph=json.loads(gpath.read_text())
rpath=Path("outputs/flan_reachable_state_bounds.json")
abstract=json.loads(Path("outputs/flan_ieee_add_transfers.json").read_text())
env={x["output"]:Decimal(x["abs_bound"]) for x in abstract["op_output_bounds"]}
convex=json.loads(Path("outputs/flan_convex_reachable_bounds.json").read_text()); convex_by_index={x["index"]:x for x in convex["op_bounds"]}
norm={}
for line in Path("outputs/flan_reachable_norm_manifest.tsv").read_text().splitlines()[1:]:
 f=line.split("\t");norm[f[0]]=Decimal(f[6])
d=Decimal(512); n=512; u=Decimal(2)**-24; eta=Decimal(2)**-150
eps_float=struct.unpack("<f",struct.pack("<f",1e-6))[0]
eps=Decimal.from_float(eps_float); sqrt_eps=eps.sqrt(); scale=int(abstract["error_units_per_real"])
k=2*n+1; gamma=Decimal(k)*u/(Decimal(1)-Decimal(k)*u)
records=[]
for index,op in enumerate(graph["ops"]):
 if op["op"]!="RMSNORM":continue
 B=Decimal(str(convex_by_index[index]["input_bounds"]["input"])); w=norm[op["weight"]]
 # Global l_inf Lipschitz bound for w*x/sqrt(mean(x^2)+eps).
 L=w*(Decimal(1)+d.sqrt())/sqrt_eps
 gain=int(L.to_integral_value(rounding=ROUND_CEILING))
 # Scale-relative forward error.  The sum of squares has no cancellation, so
 # its reduction error is gamma*m rather than gamma*B^2.  If t=m+eps, then
 # |t_hat-t|/t <= rho uniformly in m>=0.  Multiplication by x cancels the
 # 1/sqrt(t) factor because |x_i|/sqrt(m+eps) <= sqrt(d).
 alpha=gamma+u*(Decimal(1)+gamma)
 beta0=Decimal(k)*eta
 beta=beta0+u*(eps+beta0)+eta
 rho=alpha+beta/eps
 assert rho < 1
 one_minus=Decimal(1)-rho
 # Mean-value bound for z |-> z^-1/2 plus correctly-rounded rsqrt.
 rsqrt_relative_transport=rho/(Decimal(2)*one_minus**Decimal("1.5"))+u/one_minus.sqrt()
 normalized_exact_bound=d.sqrt()
 normalize_err=normalized_exact_bound*rsqrt_relative_transport + u*normalized_exact_bound*(Decimal(1)+u)/one_minus.sqrt()+eta
 output_err=w*normalize_err+u*w*(normalized_exact_bound+normalize_err)+eta
 # Decimal-to-C++ cross-check margin; it only enlarges the analytic bound.
 output_err *= Decimal("1.000000000000001")
 bias=int((output_err*scale).to_integral_value(rounding=ROUND_CEILING))
 rec={"index":index,"opcode":"RMSNORM","weight":op["weight"],"input":op["input"],
      "input_abs_bound":str(B),"weight_abs_bound":str(w),"d_model":n,
      "epsilon_binary32_exact":str(eps),"reduction_operation_bound":k,"gamma_k":str(gamma),
      "error_units_per_real":scale,"gain":gain,"bias_units":bias,"bias_real_upper":str(Decimal(bias)/scale),
      "lipschitz_bound":str(L),"relative_energy_coefficient":str(alpha),"absolute_energy_remainder":str(beta),
      "relative_shifted_mean_error":str(rho),"rsqrt_relative_transport":str(rsqrt_relative_transport),
      "normalized_exact_bound":str(normalized_exact_bound),"normalized_error_bound":str(normalize_err),
      "convex_argument":"nonnegative square reduction plus |x_i|/sqrt(mean(x^2)+eps) <= sqrt(d)",
      "source_contract":"binary32 RNE; nonnegative mean reduction; correctly-rounded rsqrt; no overflow; gradual underflow",
      "target_contract":"exact-real RMSNorm with the exact binary32 epsilon and checkpoint weight",
      "status":"CERTIFIED_UNDER_IEEE754_RNE_RSQRT_CONTRACT"}
 p=json.dumps(rec,sort_keys=True,separators=(",",":")).encode();rec["transfer_sha256"]=hashlib.sha256(p).hexdigest();records.append(rec)
assert len(records)==42
tsv=Path("outputs/flan_ieee_rmsnorm_transfers.tsv")
tsv.write_text("index\tB\tw\teps\td\tscale\tgain\tbias\n"+"".join(f"{r['index']}\t{r['input_abs_bound']}\t{r['weight_abs_bound']}\t{r['epsilon_binary32_exact']}\t512\t{scale}\t{r['gain']}\t{r['bias_units']}\n" for r in records))
out={"language":"FLAN-IEEE754-RMSNORM-TRANSFER-1","checkpoint_sha256":graph["checkpoint_sha256"],
 "graph_sha256":hashlib.sha256(gpath.read_bytes()).hexdigest(),"reachable_bounds_sha256":hashlib.sha256(rpath.read_bytes()).hexdigest(),
 "transfers":records,"tsv_sha256":hashlib.sha256(tsv.read_bytes()).hexdigest(),
 "method":"scale-relative nonnegative-energy transport; bias independent of reachable input radius",
 "status":"42/42 RMSNorm occurrences certified under explicit RNE and correctly-rounded-rsqrt contract"}
path=Path("outputs/flan_ieee_rmsnorm_transfers.json");path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"artifact":str(path),"occurrences":42,"max_gain":max(r['gain'] for r in records),"max_bias_units":max(r['bias_units'] for r in records)},indent=2))
