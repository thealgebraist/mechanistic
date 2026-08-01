#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
p=Path("outputs/flan_t5_forward_schedule_certificate.json");c=json.loads(p.read_text());g=Path("outputs/flan_full_graph.json")
assert c["certificate"]=="T5_FORWARD_SCHEDULE_TO_129_OP_GRAPH_OK" and c["opcode_count"]==129
assert c["graph_sha256"]==hashlib.sha256(g.read_bytes()).hexdigest();assert len(c["method_sha256"])==6 and all(len(x)==64 for x in c["method_sha256"].values())
assert c["encoder_layers"]==c["decoder_layers"]==8 and c["inference_contract"]["tie_word_embeddings"] is False
print(json.dumps({"certificate":"T5_FORWARD_SCHEDULE_CERTIFICATE_OK","methods":6,"opcodes":129,"template_exact":True,"formal_python_semantics":False},indent=2))
