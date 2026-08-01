#!/usr/bin/env python3
"""Audit and apply the valid core of transducer_matrix_math_proof.pdf."""
import hashlib,json,math
from pathlib import Path
pdf=Path("/Users/anders/Desktop/transducer_matrix_math_proof.pdf")
shared_path=Path("outputs/flan_shared_backend_exact_program.json");shared=json.loads(shared_path.read_text())
micro_path=Path("outputs/flan_binary32_microcode.json");micro=json.loads(micro_path.read_text())
assert len(shared["opcodes"])==129 and all(x["local_error_units"]==0 for x in shared["opcodes"])
claims=[
 {"claim":"commuting transition and observation operators imply finite-word trace exactness","verdict":"VALID_AFTER_REPAIR","reason":"formalized for arbitrary distribution-state carriers in ProbabilisticIntertwining.lean; induction quantifies over every source state"},
 {"claim":"the neural model reachable state set HR is finite","verdict":"UNPROVED_AND_UNNECESSARY","reason":"arbitrary-length KV caches and tensor-valued state are not given a finite-state bound"},
 {"claim":"binary deterministic matrices model FLAN token generation","verdict":"FALSE_FOR_SAMPLING","reason":"FLAN readout is a categorical kernel; stochastic operators or explicit random bits are required"},
 {"claim":"the construction yields a small quotient","verdict":"UNSUPPORTED","reason":"indicator-state construction has one basis vector per reachable state and may be infinite or exponentially large; reported quotient size equals source size"},
 {"claim":"10,000 prompts prove complete equivalence","verdict":"FALSE_AS_A_PROOF_METHOD","reason":"finite testing cannot establish all-sequence extensional equality"},
 {"claim":"reported entropy 15.6565 bits corresponds to 81,208 equiprobable states","verdict":"NUMERICALLY_INCONSISTENT","reason":f"log2(81208)={math.log2(81208):.12f}, not 15.6565"},
 {"claim":"SentencePiece Viterbi max-plus algebra proves FLAN decoder equivalence","verdict":"CATEGORY_MISMATCH","reason":"tokenizer path optimization and autoregressive neural probability kernels are separate transducers"}]
out={"language":"FLAN-PROBABILISTIC-INTERTWINING-AUDIT-1","source_pdf_sha256":hashlib.sha256(pdf.read_bytes()).hexdigest(),
 "shared_exact_program_sha256":hashlib.sha256(shared_path.read_bytes()).hexdigest(),"portable_microcode_sha256":hashlib.sha256(micro_path.read_bytes()).hexdigest(),
 "formal_sources_sha256":{"ProbabilisticIntertwining.lean":hashlib.sha256(Path("ProbabilisticIntertwining.lean").read_bytes()).hexdigest(),"ProbabilityLawSemantics.lean":hashlib.sha256(Path("ProbabilityLawSemantics.lean").read_bytes()).hexdigest()},
 "pdf_claim_audit":claims,
 "corrected_application":{"source_state":"probability measures over pinned FLAN tensor/cache/token-stack states","target_state":"probability measures over typed PRSL register states","project":"register extraction induced by the 129 argument bindings","transition_intertwining":"target.step token (Q mu) = Q (source.step token mu)","observation_intertwining":"target.softmax(Q mu) = source.softmax(mu)","evidence":"all 129 shared-callable bindings have zero local error and identical arguments under the pinned ABI","theorem":"ProbabilisticIntertwining.full_trace_exact","scope":shared["universal_scope"],"result":"EXACT_RELATIVE_TO_SHARED_ABI"},
 "portable_application":{"result":"OPEN_REFINEMENT_OBLIGATION","reason":"947-opcode portable expansion preserves macro semantics definitionally, but pinned PyTorch kernels are not proved to implement its exact binary32 operational rules"},
 "minimality_application":{"equivalence":"two source states agree on the mass of every finite token continuation","fiber_theorem":"ProbabilityLawSemantics.exact_quotient_cannot_merge_distinguishable_states","separation_theorem":"ProbabilityLawSemantics.distinguishable_states_require_distinct_graph_states","result":"MACHINE_CHECKED_LOWER_BOUND","implication":"a smaller exact graph exists only when its merged FLAN states are behaviorally equivalent; the PDF supplies no such nontrivial merges"},
 "matrix_interpretation":"on finite distribution carriers the operators are stochastic matrices and the corrected equations reduce to the PDF equations; no finite basis is required for the theorem"}
path=Path("outputs/flan_transducer_matrix_application.json");path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"artifact":str(path),"claims":len(claims),"shared_result":out["corrected_application"]["result"],"portable_result":out["portable_application"]["result"]},indent=2))
