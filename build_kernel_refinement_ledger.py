#!/usr/bin/env python3
"""Occurrence-complete refinement ledger for portable microcode vs pinned backend."""
import hashlib,json,collections
from pathlib import Path
mp=Path("outputs/flan_binary32_microcode.json");m=json.loads(mp.read_text())
cls={
 "TOKEN_BIND":("EXACT_STRUCTURAL","exact finite token binding"),"TENSOR_GATHER":("EXACT_STRUCTURAL","integer indexing into bit-bound tensor"),
 "RESHAPE":("EXACT_STRUCTURAL","index bijection"),"TRANSPOSE":("EXACT_STRUCTURAL","index permutation"),"RELATIVE_BIAS":("EXACT_STRUCTURAL","integer bucket function plus exact gather"),
 "CAUSAL_MASK":("EXACT_STRUCTURAL","typed mask construction"),"CONCAT":("EXACT_STRUCTURAL","index concatenation"),"CACHE_APPEND":("EXACT_STRUCTURAL","typed append"),"TOKEN_APPEND":("EXACT_STRUCTURAL","typed append"),"HALT":("EXACT_STRUCTURAL","identity termination"),
 "F32_REDUCE_MAX":("CONDITIONAL_EXACT","max is association-independent for finite non-NaN values; activation finiteness remains required"),
 "CATEGORICAL_INVERSE_CDF":("EXACT_AS_PROBABILITY_LAW","implementation coupling unnecessary; induced categorical mass must equal readout"),
 "F32_ADD":("CONTROLLED_REFERENCE_AVAILABLE","binary32 RNE scalar implementation; pinned vector kernel refinement open"),"F32_SUB":("CONTROLLED_REFERENCE_AVAILABLE","binary32 RNE scalar implementation; pinned vector kernel refinement open"),"F32_MUL":("CONTROLLED_REFERENCE_AVAILABLE","binary32 RNE scalar implementation; pinned vector kernel refinement open"),"F32_DIV":("CONTROLLED_REFERENCE_AVAILABLE","binary32 RNE scalar implementation; pinned vector kernel refinement open"),"F32_SQUARE":("CONTROLLED_REFERENCE_AVAILABLE","specialized binary32 multiplication"),"F32_MUL_CONST":("CONTROLLED_REFERENCE_AVAILABLE","binary32 multiplication by bit-bound constant"),"F32_ADD_CONST":("CONTROLLED_REFERENCE_AVAILABLE","binary32 addition of bit-bound constant"),"F32_CUBE":("CONTROLLED_REFERENCE_AVAILABLE","two ordered binary32 multiplications"),
 "F32_REDUCE_SUM":("OPEN_BACKEND_ORDER_REFINEMENT","portable operation is derived by ordered scalar addition; pinned backend reduction association/vectorization is not formally matched"),"F32_MATMUL":("OPEN_BACKEND_ORDER_REFINEMENT","portable operation is derived by lexicographic scalar multiply/add; pinned GEMM association/FMA behavior is not formally matched"),
 "F32_RSQRT":("OPEN_BACKEND_TRANSCENDENTAL_REFINEMENT","correct rounding and pinned backend algorithm not formally matched"),"F32_EXP":("OPEN_BACKEND_TRANSCENDENTAL_REFINEMENT","correct rounding and pinned backend algorithm not formally matched"),"F32_TANH":("OPEN_BACKEND_TRANSCENDENTAL_REFINEMENT","correct rounding and pinned backend algorithm not formally matched")}
derived={"F32_REDUCE_SUM","F32_MATMUL"}
assert set(cls)==set(m["scalar_semantics"])
rows=[]
for op in m["microcode"]:
 status,obligation=cls[op["micro_opcode"]]
 portable_status=("DERIVED_ORDERED_SCALAR_KERNEL" if op["micro_opcode"] in derived else
                  "DECLARED_PORTABLE_PRIMITIVE")
 rows.append({"pc":op["pc"],"macro_index":op["macro_index"],"micro_opcode":op["micro_opcode"],"portable_layer_status":portable_status,"backend_refinement_status":status,"status":status,"obligation":obligation,"portable_proof_artifact":"OrderedKernelLowering.lean" if op["micro_opcode"] in derived else None})
counts=collections.Counter(r["status"] for r in rows);kind_counts=collections.Counter(r["micro_opcode"] for r in rows)
portable_counts=collections.Counter(r["portable_layer_status"] for r in rows)
proof=Path("OrderedKernelLowering.lean")
out={"language":"FLAN-KERNEL-REFINEMENT-LEDGER-2","microcode_sha256":hashlib.sha256(mp.read_bytes()).hexdigest(),"ordered_kernel_lowering_sha256":hashlib.sha256(proof.read_bytes()).hexdigest(),"occurrences":len(rows),"primitive_schema_kinds":len(cls),"used_primitive_kinds":len(kind_counts),"portable_base_primitive_schema_kinds":len(set(cls)-derived),"used_portable_base_primitive_kinds":len(set(kind_counts)-derived),"status_counts":dict(sorted(counts.items())),"portable_layer_counts":dict(sorted(portable_counts.items())),"kind_counts":dict(sorted(kind_counts.items())),"rows":rows,
 "portable_semantics_complete":True,"pinned_backend_refinement_complete":False,
 "universal_portable_equivalence_complete":False,"open_occurrences":sum(v for k,v in counts.items() if k.startswith("OPEN_")),
 "layering_note":"ordered reduction and matmul are proved derived from scalar add/mul in the portable DSL; equality of pinned ATen kernels to that order remains a separate backend obligation",
 "next_exact_route":"prove or replace pinned F32_REDUCE_SUM/F32_MATMUL ordering and F32_RSQRT/F32_EXP/F32_TANH implementations"}
path=Path("outputs/flan_kernel_refinement_ledger.json");path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"artifact":str(path),"occurrences":len(rows),"status_counts":out["status_counts"],"open":out["open_occurrences"]},indent=2))
