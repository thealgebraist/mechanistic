#!/usr/bin/env python3
"""Bind the concrete FLAN shared-ABI manifest to the nontrivial Lean theorem."""
import hashlib,json
from pathlib import Path
shared_path=Path("outputs/flan_shared_backend_exact_program.json");shared=json.loads(shared_path.read_text())
graph_path=Path("outputs/flan_full_graph.json");micro_path=Path("outputs/flan_binary32_microcode.json");bits_path=Path("outputs/flan_checkpoint_bit_manifest.json");aten_path=Path("outputs/flan_aten_dispatch_manifest.json");aten=json.loads(aten_path.read_text())
kernel_path=Path("outputs/flan_kernel_refinement_ledger.json");kernel=json.loads(kernel_path.read_text())
lean_paths=[Path("FlanSharedABIIntertwining.lean"),Path("ProbabilisticIntertwining.lean"),Path("ProbabilityLawSemantics.lean"),Path("OrderedKernelLowering.lean"),Path("WholeModelEquivalence.lean"),Path("GeneratedFlanProgram.lean"),Path("CheckpointBitSemantics.lean")]
assert len(shared["opcodes"])==129
bindings=[]
for x in shared["opcodes"]:
 assert x["local_error_units"]==0 and x["semantic_status"]=="DEFINITIONAL_EQUALITY_UNDER_SHARED_ABI"
 bindings.append({"index":x["index"],"opcode":x["opcode"],"shared_callable":x["shared_callable"],"binding_sha256":x["binding_sha256"],"commuting_obligation":"registerPrimitive tag (encode state) = encode (sourcePrimitive tag state)","status":"ASSUMED_BY_HASHED_SHARED_CALLABLE_BINDING"})
out={"language":"FLAN-SHARED-ABI-INTERTWINING-THEOREM-1","graph_sha256":hashlib.sha256(graph_path.read_bytes()).hexdigest(),
 "checkpoint_sha256":shared["checkpoint_sha256"],"shared_program_sha256":hashlib.sha256(shared_path.read_bytes()).hexdigest(),
 "portable_microcode_sha256":hashlib.sha256(micro_path.read_bytes()).hexdigest(),"kernel_refinement_ledger_sha256":hashlib.sha256(kernel_path.read_bytes()).hexdigest(),"checkpoint_bit_manifest_sha256":hashlib.sha256(bits_path.read_bytes()).hexdigest(),"aten_dispatch_manifest_sha256":hashlib.sha256(aten_path.read_bytes()).hexdigest(),"aten_unique_schemas":aten["unique_schemas"],"lean_source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in lean_paths},
 "source_state_type":"pinned backend tensor/cache/token-stack state","target_state_type":"typed PRSL register state","states_are_distinct_in_theorem":True,
 "projection":"SharedABICertificate.encode","per_opcode_bindings":bindings,"opcode_count":len(bindings),
 "step_theorem":"FlanSharedABIIntertwining.generated_129_opcode_step_commutes",
 "trace_theorem":"FlanSharedABIIntertwining.all_prompts_all_finite_continuations_exact",
 "probability_law_theorem":"ProbabilityLawSemantics.same_categorical_law_same_finite_trace",
 "minimality_theorem":"ProbabilityLawSemantics.distinguishable_states_require_distinct_graph_states",
 "minimality_meaning":"any exact quotient encoding may merge only states with equal finite-continuation masses",
 "portable_ordered_reduction_theorem":"OrderedKernelLowering.ordered_reduce_commutes",
 "portable_ordered_matmul_entry_theorem":"OrderedKernelLowering.ordered_matmul_entry_commutes",
 "portable_semantics_complete":kernel["portable_semantics_complete"],
 "pinned_backend_refinement_complete":kernel["pinned_backend_refinement_complete"],
 "open_backend_micro_occurrences":kernel["open_occurrences"],
 "quantification":"every Prompt value and every finite continuation List Tok",
 "result":"MACHINE_CHECKED_FROM_PER_OPCODE_COMMUTING_ASSUMPTIONS",
 "trust_boundary":"hash matching and identical callable/argument claims are checked externally; Lean does not formalize Python/PyTorch operational semantics",
 "portable_backend_independent":False}
path=Path("outputs/flan_intertwining_theorem_manifest.json");path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"artifact":str(path),"opcodes":len(bindings),"distinct_states":True,"result":out["result"]},indent=2))
