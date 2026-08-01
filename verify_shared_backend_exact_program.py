#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
gp=Path("outputs/flan_full_graph.json");g=json.loads(gp.read_text());bp=Path("outputs/flan_backend_contract.json");c=json.loads(Path("outputs/flan_shared_backend_exact_program.json").read_text())
sp=Path("outputs/flan_t5_forward_schedule_certificate.json");schedule=json.loads(sp.read_text())
ap=Path("outputs/flan_aten_dispatch_manifest.json");aten=json.loads(ap.read_text())
ep=Path("outputs/flan_aten_register_executable_certificate.json");executable=json.loads(ep.read_text())
assert c["full_graph_sha256"]==hashlib.sha256(gp.read_bytes()).hexdigest() and c["checkpoint_sha256"]==g["checkpoint_sha256"]
assert c["backend_contract_sha256"]==hashlib.sha256(bp.read_bytes()).hexdigest();assert c["opcode_count"]==len(c["opcodes"])==len(g["ops"])==129
assert c["forward_schedule_certificate_sha256"]==hashlib.sha256(sp.read_bytes()).hexdigest()
assert c["aten_dispatch_manifest_sha256"]==hashlib.sha256(ap.read_bytes()).hexdigest() and c["aten_unique_schemas"]==aten["unique_schemas"]==48
assert c["aten_register_executable_sha256"]==hashlib.sha256(ep.read_bytes()).hexdigest() and c["aten_register_source_sha256"]==executable["source_sha256"]
assert c["forward_method_sha256"]==schedule["method_sha256"] and len(c["forward_method_sha256"])==6
for i,(o,r) in enumerate(zip(g["ops"],c["opcodes"])):
 assert r["index"]==i and r["opcode"]==o["op"] and r["argument_binding"]==o and r["local_error_units"]==0
 p=dict(r);d=p.pop("binding_sha256");assert d==hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()
assert c["exact_relative_to_shared_abi"] and not c["portable_backend_independent"]
assert c["theorem"].startswith("FlanSharedABIIntertwining.")
assert c["representation_bytes"]["total"]==c["representation_bytes"]["graph"]+c["representation_bytes"]["checkpoint"]
print(json.dumps({"certificate":"SHARED_BACKEND_EXACT_PROGRAM_OK","opcodes":129,"all_sequence_scope":True,"exact_relative_to_shared_abi":True,"portable_backend_independent":False},indent=2))
