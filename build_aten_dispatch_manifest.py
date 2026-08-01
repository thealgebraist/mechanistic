#!/usr/bin/env python3
"""Capture the lower-level ATen ABI used by full and cached FLAN execution."""
import collections,hashlib,json,platform
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import T5ForConditionalGeneration
class Capture(TorchDispatchMode):
 def __init__(self):self.schemas=[]
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  self.schemas.append(str(func._schema));return func(*args,**(kwargs or {}))
def family(schema):return schema.split("(",1)[0]
torch.use_deterministic_algorithms(True)
root=Path("work/google_flan");model=T5ForConditionalGeneration.from_pretrained(str(root),local_files_only=True,dtype=torch.float32).eval()
enc=torch.tensor([[13959,1566,12,2968,10,3,9,422,794,1]],dtype=torch.long);dec=torch.tensor([[0,3]],dtype=torch.long)
full=Capture()
with torch.no_grad(),full: out=model(input_ids=enc,decoder_input_ids=dec,use_cache=True,return_dict=True)
cached=Capture()
with torch.no_grad(),cached: model(encoder_outputs=(out.encoder_last_hidden_state,),decoder_input_ids=torch.tensor([[7]],dtype=torch.long),past_key_values=out.past_key_values,use_cache=True,return_dict=True)
all_schemas=sorted(set(full.schemas)|set(cached.schemas));structural={"aten::_unsafe_view","aten::alias","aten::arange","aten::arange.start","aten::cat","aten::clone","aten::copy_","aten::embedding","aten::expand","aten::full","aten::full_like","aten::lift_fresh","aten::ones","aten::permute","aten::select.int","aten::slice.Tensor","aten::t","aten::transpose.int","aten::unsqueeze","aten::view","aten::zeros","aten::zeros_like","aten::_to_copy"}
control={"aten::abs","aten::eq.Scalar","aten::gt.Scalar","aten::gt.Tensor","aten::lt.Scalar","aten::masked_fill.Scalar","aten::minimum","aten::neg","aten::rsub.Scalar","aten::triu","aten::where.self"}
elementary={"aten::add.Tensor","aten::add_.Tensor","aten::div.Tensor","aten::mul.Tensor","aten::mul_.Tensor","aten::pow.Tensor_Scalar","aten::sub.Tensor"}
reduction={"aten::bmm","aten::mean.dim","aten::mm"};trans={"aten::_softmax","aten::log","aten::rsqrt","aten::tanh"}
rows=[]
for s in all_schemas:
 f=family(s)
 if f in structural:category="STRUCTURAL_ATEN"
 elif f in control:category="MASK_AND_CONTROL_ATEN"
 elif f in elementary:category="ELEMENTARY_F32_ATEN"
 elif f in reduction:category="REDUCTION_ATEN"
 elif f in trans:category="TRANSCENDENTAL_OR_SOFTMAX_ATEN"
 else:raise AssertionError(f)
 rows.append({"schema":s,"family":f,"schema_sha256":hashlib.sha256(s.encode()).hexdigest(),"full_calls":full.schemas.count(s),"cached_calls":cached.schemas.count(s),"category":category,"target_binding":"invoke the identical torch.ops ATen schema on the same device/dtype/arguments","semantic_status":"DEFINITIONAL_EQUALITY_UNDER_SHARED_ATEN_DISPATCH"})
schedule=Path("outputs/flan_t5_forward_schedule_certificate.json");backend=Path("outputs/flan_backend_contract.json")
outj={"language":"FLAN-SHARED-ATEN-DISPATCH-1","torch_version":torch.__version__,"platform":platform.platform(),"device":"cpu","dtype":"float32","deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"full_dispatch_calls":len(full.schemas),"cached_dispatch_calls":len(cached.schemas),"unique_schemas":len(rows),"schemas":rows,
 "required_path_coverage":["full encoder plus full decoder","one-token decoder with populated KV cache"],"shape_generality_argument":"operator schemas are shape-polymorphic; source schedule certificate proves fixed layer/control structure, while both cache branches are traced","forward_schedule_sha256":hashlib.sha256(schedule.read_bytes()).hexdigest(),"backend_contract_sha256":hashlib.sha256(backend.read_bytes()).hexdigest(),
 "exact_result":"source and PRSL target are definitionally equal when both dispatch these same schemas with bit-identical arguments","portable_backend_independent":False,"trust_boundary":"dispatcher kernel implementation and argument correspondence remain shared ABI assumptions"}
path=Path("outputs/flan_aten_dispatch_manifest.json");path.write_text(json.dumps(outj,indent=2)+"\n")
print(json.dumps({"artifact":str(path),"full_calls":len(full.schemas),"cached_calls":len(cached.schemas),"schemas":len(rows),"categories":dict(collections.Counter(r["category"] for r in rows))},indent=2))
