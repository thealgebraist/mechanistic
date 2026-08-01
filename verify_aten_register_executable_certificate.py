#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
x=json.loads(Path("outputs/flan_aten_register_executable_certificate.json").read_text());s=Path(x["source"])
assert x["language"]=="FLAN-EXECUTABLE-ATEN-PRSL-1" and x["source_sha256"]==hashlib.sha256(s.read_bytes()).hexdigest()
assert x["forbidden_implicit_matmul_nodes"]==0 and not x["forbidden_high_level_calls"] and len(x["required_explicit_aten_calls"])==9
assert x["semantic_status"]=="EXECUTABLE_CANDIDATE_ZERO_ERROR_ON_REGRESSION_MATRIX"
assert x["runtime_evidence"]=={"tested_cases":4,"maximum_observed_logit_error":0.0,"includes_two_token_decoder":True,"universal_proof":False}
assert "do not prove" in x["remaining_proof_limit"] and not x["portable_backend_independent"]
print("FLAN_EXECUTABLE_ATEN_PRSL_OK explicit_calls=9 implicit_matmul=0 observed_error=0")
