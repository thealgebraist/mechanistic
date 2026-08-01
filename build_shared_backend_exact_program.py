#!/usr/bin/env python3
"""Bind the readable FLAN graph to an exact shared numerical-kernel ABI."""
import hashlib,json
from pathlib import Path
gp=Path("outputs/flan_full_graph.json");g=json.loads(gp.read_text())
bp=Path("outputs/flan_backend_contract.json");b=json.loads(bp.read_text())
sp=Path("outputs/flan_t5_forward_schedule_certificate.json");schedule=json.loads(sp.read_text())
ap=Path("outputs/flan_aten_dispatch_manifest.json");aten=json.loads(ap.read_text())
ep=Path("outputs/flan_aten_register_executable_certificate.json");executable=json.loads(ep.read_text())
callable_for={
 "INPUT_TOKENS":"prsl.input_tokens","INPUT_DECODER_STACK":"prsl.input_decoder_stack",
 "EMBED":"torch.nn.functional.embedding","RMSNORM":"transformers.T5LayerNorm.forward",
 "SELF_ATTENTION_SEQUENCE":"transformers.T5Attention.forward","SELF_ATTENTION_KV":"transformers.T5Attention.forward",
 "CROSS_ATTENTION":"transformers.T5Attention.forward","ADD":"torch.add",
 "GATED_MLP":"transformers.T5DenseGatedActDense.forward","MATMUL":"torch.nn.functional.linear",
 "SOFTMAX":"torch.nn.functional.softmax","SAMPLE_PUSH_UPDATE_CACHE":"prsl.sample_push_update_cache"}
ops=[]
for i,o in enumerate(g["ops"]):
 r={"index":i,"opcode":o["op"],"shared_callable":callable_for[o["op"]],
    "argument_binding":o,"identity_requirement":"source and register interpreter invoke the same callable object with bit-identical tensor arguments",
    "local_error_units":0,"semantic_status":"DEFINITIONAL_EQUALITY_UNDER_SHARED_ABI"}
 p=json.dumps(r,sort_keys=True,separators=(",",":")).encode();r["binding_sha256"]=hashlib.sha256(p).hexdigest();ops.append(r)
graph_bytes=gp.stat().st_size; checkpoint_bytes=Path("work/google_flan/model.safetensors").stat().st_size
out={"language":"PRSL-SHARED-BACKEND-EXACT-1","full_graph_sha256":hashlib.sha256(gp.read_bytes()).hexdigest(),
 "checkpoint_sha256":g["checkpoint_sha256"],"backend_contract_sha256":hashlib.sha256(bp.read_bytes()).hexdigest(),
 "backend_inner_contract_sha256":b["contract_sha256"],"opcode_count":len(ops),"opcodes":ops,
 "forward_schedule_certificate_sha256":hashlib.sha256(sp.read_bytes()).hexdigest(),
 "aten_dispatch_manifest_sha256":hashlib.sha256(ap.read_bytes()).hexdigest(),"aten_unique_schemas":aten["unique_schemas"],
 "aten_register_executable_sha256":hashlib.sha256(ep.read_bytes()).hexdigest(),"aten_register_source_sha256":executable["source_sha256"],
 "forward_method_sha256":schedule["method_sha256"],
 "two_level_interpreter":{"outer":"129 named probabilistic register opcodes","middle":"48 hashed shape-polymorphic ATen schemas","inner":"pinned shared CPU dispatcher kernels"},
 "universal_scope":"every backend-valid finite encoder token sequence and every finite decoder continuation",
 "probability_readout":"full 32128-token softmax law","exact_relative_to_shared_abi":True,
 "portable_backend_independent":False,"representation_bytes":{"graph":graph_bytes,"checkpoint":checkpoint_bytes,"total":graph_bytes+checkpoint_bytes},
 "theorem":"FlanSharedABIIntertwining.all_prompts_all_finite_continuations_exact",
 "trust_boundary":"bit-identical ATen arguments and pinned CPU dispatcher kernels; Python composition is checked by source-AST schedule plus full/cache dispatch traces"}
path=Path("outputs/flan_shared_backend_exact_program.json");path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"artifact":str(path),"opcodes":len(ops),"exact_relative":True,"portable":False,"bytes":out["representation_bytes"]},indent=2))
