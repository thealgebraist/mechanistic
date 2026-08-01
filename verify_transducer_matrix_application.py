#!/usr/bin/env python3
import json
import hashlib
from pathlib import Path
x=json.loads(Path("outputs/flan_transducer_matrix_application.json").read_text())
assert x["language"]=="FLAN-PROBABILISTIC-INTERTWINING-AUDIT-1"
assert len(x["pdf_claim_audit"])==7
assert x["corrected_application"]["result"]=="EXACT_RELATIVE_TO_SHARED_ABI"
assert x["portable_application"]["result"]=="OPEN_REFINEMENT_OBLIGATION"
assert x["minimality_application"]["result"]=="MACHINE_CHECKED_LOWER_BOUND"
for name,h in x["formal_sources_sha256"].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
assert any(c["verdict"]=="NUMERICALLY_INCONSISTENT" for c in x["pdf_claim_audit"])
print("FLAN_PROBABILISTIC_INTERTWINING_AUDIT_OK claims=7")
