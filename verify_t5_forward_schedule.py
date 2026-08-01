#!/usr/bin/env python3
"""Bind installed Transformers T5 forward control/dataflow to the 129-op graph."""
import ast, hashlib, inspect, json, textwrap
from pathlib import Path
from transformers.models.t5 import modeling_t5 as m

methods={
 "MODEL":m.T5ForConditionalGeneration.forward,"STACK":m.T5Stack.forward,"BLOCK":m.T5Block.forward,
 "SELF_LAYER":m.T5LayerSelfAttention.forward,"CROSS_LAYER":m.T5LayerCrossAttention.forward,"FF_LAYER":m.T5LayerFF.forward}
src={k:textwrap.dedent(inspect.getsource(v)) for k,v in methods.items()}
for s in src.values(): ast.parse(s)
def ordered(s,*parts):
 p=-1
 for x in parts:
  q=s.find(x,p+1); assert q>=0,(x,p); p=q
checks={
 "model_encoder_decoder_lm_head": ordered(src["MODEL"],"self.encoder(","self.decoder(","self.lm_head(sequence_output)"),
 "stack_embed_loop_final_norm": ordered(src["STACK"],"self.embed_tokens(input_ids)","for i, layer_module in enumerate(self.block):","layer_outputs = layer_module(","self.final_layer_norm(hidden_states)"),
 "block_self_cross_ff": ordered(src["BLOCK"],"self.layer[0](","self.layer[1](","self.layer[-1]("),
 "self_norm_attention_residual": ordered(src["SELF_LAYER"],"self.layer_norm(hidden_states)","self.SelfAttention(","hidden_states + self.dropout(attention_output[0])"),
 "cross_norm_attention_residual": ordered(src["CROSS_LAYER"],"self.layer_norm(hidden_states)","self.EncDecAttention(","hidden_states + self.dropout(attention_output[0])"),
 "ff_norm_dense_residual": ordered(src["FF_LAYER"],"self.layer_norm(hidden_states)","self.DenseReluDense(","hidden_states + self.dropout(forwarded_states)"),
}
assert all(v is None for v in checks.values())
graph_path=Path("outputs/flan_full_graph.json");g=json.loads(graph_path.read_text());tags=[o["op"] for o in g["ops"]]
enc=["RMSNORM","SELF_ATTENTION_SEQUENCE","ADD","RMSNORM","GATED_MLP","ADD"]
dec=["RMSNORM","SELF_ATTENTION_KV","ADD","RMSNORM","CROSS_ATTENTION","ADD","RMSNORM","GATED_MLP","ADD"]
expected=["INPUT_TOKENS","EMBED"]+enc*8+["RMSNORM","INPUT_DECODER_STACK","EMBED"]+dec*8+["RMSNORM","MATMUL","SOFTMAX","SAMPLE_PUSH_UPDATE_CACHE"]
assert tags==expected and len(tags)==129
out={"certificate":"T5_FORWARD_SCHEDULE_TO_129_OP_GRAPH_OK","method_sha256":{k:hashlib.sha256(v.encode()).hexdigest() for k,v in src.items()},
 "checks":["model encoder -> decoder -> lm_head","stack embedding -> block loop -> final norm","block self -> optional cross -> feed-forward","self/cross/FF pre-norm and residual order","exact 8-layer encoder/decoder opcode template"],
 "graph_sha256":hashlib.sha256(graph_path.read_bytes()).hexdigest(),"opcode_count":129,"encoder_layers":8,"decoder_layers":8,
 "inference_contract":{"dtype":"float32","training":False,"dropout":"identity","tie_word_embeddings":False},
 "probabilistic_extension":"SOFTMAX and SAMPLE_PUSH_UPDATE_CACHE are the explicit autoregressive wrapper after source lm_logits",
 "status":"source AST parses and required dataflow/order patterns match exact graph template; not a formal semantics proof of Python/PyTorch"}
path=Path("outputs/flan_t5_forward_schedule_certificate.json");path.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
