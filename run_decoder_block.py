"""Replay the connected serialized FLAN-T5 decoder block."""
import json, torch
from pathlib import Path

def T(x): return torch.tensor(x, dtype=torch.float32)
p = json.loads(Path("outputs/flan_decoder_block_level1.json").read_text())
r = {}
for op in p["ops"]:
    name = op["op"]
    if name == "LOAD_VECTOR": r[op["name"]] = T(op["values"])
    elif name == "LOAD_MATRIX": r[op["name"]] = T(op["values"])
    elif name == "RMSNORM":
        x = r[op["input"]]; r[op["output"]] = T(op["weight"]) * x / torch.sqrt(torch.mean(x*x) + op["epsilon"])
    elif name == "ADD": r[op["output"]] = r[op["left"]] + r[op["right"]]
    elif name == "MATMUL": r[op["output"]] = T(op["weights"]) @ r[op["input"]]
    elif name == "SELF_ATTENTION_ONE":
        x = r[op["input"]]; h, d = op["heads"], op["head_width"]
        q = (T(op["q"]) @ x).reshape(h,d); k = (T(op["k"]) @ x).reshape(h,d); v = (T(op["v"]) @ x).reshape(h,d)
        r[op["output"]] = T(op["o"]) @ v.reshape(-1)
    elif name == "CROSS_ATTENTION":
        x, mem = r[op["input"]], r[op["memory"]]; h, d = op["heads"], op["head_width"]
        q = (T(op["q"]) @ x).reshape(h,d)
        k = (mem @ T(op["k"]).T).reshape(mem.shape[0],h,d).permute(1,0,2)
        v = (mem @ T(op["v"]).T).reshape(mem.shape[0],h,d).permute(1,0,2)
        weights = torch.softmax(torch.einsum("hd,hld->hl", q, k), dim=-1)
        head = torch.einsum("hl,hld->hd", weights, v)
        r[op["output"]] = T(op["o"]) @ head.reshape(-1)
    elif name == "GELU_GATE": r[op["output"]] = torch.nn.functional.gelu(r[op["left"]], approximate="tanh") * r[op["right"]]
    elif name == "HALT": pass
    else: raise ValueError(f"unknown opcode {name}")
err = float((r["y"] - T(p["target"])).abs().max())
print(json.dumps({"certificate":"DECODER_BLOCK_REGISTER_REPLAY_OK", "block":p["block"],
                  "registers":{k:list(v.shape) for k,v in r.items()}, "max_replay_error":err,
                  "program_bytes":Path("outputs/flan_decoder_block_level1.json").stat().st_size}, indent=2))
assert err < 2e-4
