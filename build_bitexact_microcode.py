#!/usr/bin/env python3
"""Expand the complete FLAN register graph into symbolic binary32 microcode."""
import hashlib,json
from pathlib import Path
gp=Path("outputs/flan_full_graph.json");g=json.loads(gp.read_text())
bp=Path("outputs/flan_backend_contract.json");backend=json.loads(bp.read_text())
ROUND="binary32_roundTiesToEven_after_each_scalar_result"
SEM={
 "TOKEN_BIND":"bind finite token vector", "TENSOR_GATHER":"exact integer-index tensor gather",
 "F32_SQUARE":"map x -> round32(x*x)","F32_REDUCE_SUM":"left-associated indexed round32 addition",
 "F32_MUL_CONST":"map x -> round32(x*c)","F32_ADD_CONST":"map x -> round32(x+c)",
 "F32_RSQRT":"map x -> correctlyRoundedBinary32(1/sqrt(x))","F32_MUL":"broadcast map round32(x*y)",
 "F32_ADD":"broadcast map round32(x+y)","F32_MATMUL":"lexicographic dot products with round32 multiply/add",
 "RESHAPE":"exact index bijection","TRANSPOSE":"exact index permutation","RELATIVE_BIAS":"exact T5 bucket lookup and tensor gather",
 "CAUSAL_MASK":"exact extended-real mask construction","F32_REDUCE_MAX":"ordered maximum; NaN policy pinned by ABI",
 "F32_SUB":"broadcast map round32(x-y)","F32_EXP":"map correctlyRoundedBinary32(exp(x))",
 "F32_DIV":"broadcast map round32(x/y)","F32_TANH":"map correctlyRoundedBinary32(tanh(x))",
 "F32_CUBE":"map round32(round32(x*x)*x)","CONCAT":"exact indexed tensor concatenation",
 "CATEGORICAL_INVERSE_CDF":"least token whose ordered cumulative mass exceeds supplied uniform bits",
 "CACHE_APPEND":"exact append to typed finite cache","TOKEN_APPEND":"exact append to token stack","HALT":"return register state"}
def seq(kind):
 if kind=="RMSNORM":return ["F32_SQUARE","F32_REDUCE_SUM","F32_MUL_CONST","F32_ADD_CONST","F32_RSQRT","F32_MUL","F32_MUL"]
 if kind in {"SELF_ATTENTION_SEQUENCE","SELF_ATTENTION_KV","CROSS_ATTENTION"}:return ["F32_MATMUL","F32_MATMUL","F32_MATMUL","RESHAPE","TRANSPOSE","F32_MATMUL","RELATIVE_BIAS","CAUSAL_MASK","F32_ADD","F32_REDUCE_MAX","F32_SUB","F32_EXP","F32_REDUCE_SUM","F32_DIV","F32_MATMUL","RESHAPE","F32_MATMUL"]
 if kind=="GATED_MLP":return ["F32_MATMUL","F32_MATMUL","F32_MUL_CONST","F32_CUBE","F32_MUL_CONST","F32_ADD","F32_MUL_CONST","F32_TANH","F32_ADD_CONST","F32_MUL","F32_MUL","F32_MATMUL"]
 if kind=="SOFTMAX":return ["F32_REDUCE_MAX","F32_SUB","F32_EXP","F32_REDUCE_SUM","F32_DIV"]
 return {"INPUT_TOKENS":["TOKEN_BIND"],"INPUT_DECODER_STACK":["TOKEN_BIND"],"EMBED":["TENSOR_GATHER"],"ADD":["F32_ADD"],"MATMUL":["F32_MATMUL"],"SAMPLE_PUSH_UPDATE_CACHE":["CATEGORICAL_INVERSE_CDF","TOKEN_APPEND","CACHE_APPEND"],"HALT":["HALT"]}.get(kind)
micro=[];spans=[]
for mi,op in enumerate(g["ops"]):
 kinds=seq(op["op"]);assert kinds is not None,(mi,op["op"])
 start=len(micro)
 for local,k in enumerate(kinds):
  payload={"pc":len(micro),"macro_index":mi,"macro_opcode":op["op"],"local_index":local,"micro_opcode":k,
   "semantics":SEM[k],"rounding":ROUND if k.startswith("F32_") else "exact_or_explicitly_pinned",
   "arguments":op if local==0 else {"macro_binding_sha256":hashlib.sha256(json.dumps(op,sort_keys=True,separators=(",",":")).encode()).hexdigest()},
   "loop_form":"symbolic_finite_index_loop; dimensions read from typed registers and checkpoint shapes"}
  payload["instruction_sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest();micro.append(payload)
 spans.append({"macro_index":mi,"macro_opcode":op["op"],"pc_begin":start,"pc_end_exclusive":len(micro),"micro_count":len(kinds)})
out={"language":"PRSL-BINARY32-MICROCODE-1","graph_sha256":hashlib.sha256(gp.read_bytes()).hexdigest(),
 "checkpoint_sha256":g["checkpoint_sha256"],"backend_contract_sha256":hashlib.sha256(bp.read_bytes()).hexdigest(),
 "macro_opcode_count":len(g["ops"]),"micro_opcode_count":len(micro),"scalar_semantics":SEM,"spans":spans,"microcode":micro,
 "probability_semantics":{"normalization":"max-shift binary32 exp/sum/div microcode","sampling":"inverse CDF driven by explicit uniform random bitstream","trace_weight":"product of emitted categorical masses"},
 "size_claim":"instruction count is linear in macro graph size; tensor dimensions are symbolic loops, not unrolled states",
 "universal_scope":"every backend-valid finite encoder token vector and every finite decoder continuation",
 "semantic_status":"complete portable reference semantics; refinement of pinned PyTorch kernels remains an external proof obligation"}
path=Path("outputs/flan_binary32_microcode.json");path.write_text(json.dumps(out,indent=2)+"\n")
tsv=Path("outputs/flan_binary32_microcode.tsv");tsv.write_text("pc\tmacro_index\tmicro_opcode\trounding\n"+"".join(f"{x['pc']}\t{x['macro_index']}\t{x['micro_opcode']}\t{x['rounding']}\n" for x in micro))
print(json.dumps({"artifact":str(path),"macros":len(spans),"micro_ops":len(micro),"kinds":len(set(x['micro_opcode'] for x in micro))},indent=2))
