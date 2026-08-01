#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
x=json.loads(Path("outputs/flan_intertwining_theorem_manifest.json").read_text())
kernel_path=Path("outputs/flan_kernel_refinement_ledger.json")
assert x["language"]=="FLAN-SHARED-ABI-INTERTWINING-THEOREM-1" and x["opcode_count"]==129
assert x["states_are_distinct_in_theorem"] and not x["portable_backend_independent"]
assert x["aten_unique_schemas"]==48
assert [b["index"] for b in x["per_opcode_bindings"]]==list(range(129))
for name,h in x["lean_source_sha256"].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
assert x["trace_theorem"].endswith("all_prompts_all_finite_continuations_exact")
assert x["probability_law_theorem"].endswith("same_categorical_law_same_finite_trace")
assert x["minimality_theorem"].endswith("distinguishable_states_require_distinct_graph_states")
assert x["kernel_refinement_ledger_sha256"]==hashlib.sha256(kernel_path.read_bytes()).hexdigest()
assert x["portable_ordered_reduction_theorem"].endswith("ordered_reduce_commutes")
assert x["portable_ordered_matmul_entry_theorem"].endswith("ordered_matmul_entry_commutes")
assert x["portable_semantics_complete"] and not x["pinned_backend_refinement_complete"]
assert x["open_backend_micro_occurrences"]==343
print("FLAN_INTERTWINING_THEOREM_MANIFEST_OK opcodes=129 distinct_states=true")
