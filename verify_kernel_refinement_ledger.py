#!/usr/bin/env python3
import json
import hashlib
from pathlib import Path
x=json.loads(Path("outputs/flan_kernel_refinement_ledger.json").read_text())
assert x["language"]=="FLAN-KERNEL-REFINEMENT-LEDGER-2"
assert x["occurrences"]==947 and len(x["rows"])==947 and x["used_primitive_kinds"]==23 and x["primitive_schema_kinds"]==25
assert x["portable_base_primitive_schema_kinds"]==23 and x["used_portable_base_primitive_kinds"]==21
assert [r["pc"] for r in x["rows"]]==list(range(947)) and x["open_occurrences"]>0
assert sum(x["status_counts"].values())==947 and not x["universal_portable_equivalence_complete"]
assert x["portable_semantics_complete"] and not x["pinned_backend_refinement_complete"]
assert x["portable_layer_counts"]["DERIVED_ORDERED_SCALAR_KERNEL"]==260
assert x["ordered_kernel_lowering_sha256"]==hashlib.sha256(Path("OrderedKernelLowering.lean").read_bytes()).hexdigest()
print("FLAN_KERNEL_REFINEMENT_LEDGER_OK occurrences=947 open="+str(x["open_occurrences"]))
