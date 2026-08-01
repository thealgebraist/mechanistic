"""Replay NEURAL-ALGEBRA-1-PARAMETRIC with an explicit input fixture."""
import json, torch
from pathlib import Path

def T(x): return torch.tensor(x, dtype=torch.float32)
program = json.loads(Path("outputs/flan_decoder_block_parametric.json").read_text())
fixture = json.loads(Path("outputs/flan_decoder_block_fixture.json").read_text())
r = {}
for op in program["ops"]:
    n = op["op"]
    if n == "INPUT_VECTOR": r[op["name"]] = T(fixture[op["name"]])
    elif n == "INPUT_MATRIX": r[op["name"]] = T(fixture[op["name"]])
    elif n == "RMSNORM":
        x = r[op["input"]]; r[op["output"]] = T(op["weight"]) * x / torch.sqrt(torch.mean(x*x) + op["epsilon"])
    elif n == "ADD": r[op["output"]] = r[op["left"]] + r[op["right"]]
    elif n == "MATMUL": r[op["output"]] = T(op["weights"]) @ r[op["input"]]
    elif n == "SELF_ATTENTION_ONE":
        x = r[op["input"]]; h, d = op["heads"], op["head_width"]
        v = (T(op["v"]) @ x).reshape(h,d)
        r[op["output"]] = T(op["o"]) @ v.reshape(-1)
    elif n == "CROSS_ATTENTION":
        x, mem = r[op["input"]], r[op["memory"]]; h, d = op["heads"], op["head_width"]
        q = (T(op["q"]) @ x).reshape(h,d)
        k = (mem @ T(op["k"]).T).reshape(mem.shape[0],h,d).permute(1,0,2)
        v = (mem @ T(op["v"]).T).reshape(mem.shape[0],h,d).permute(1,0,2)
        w = torch.softmax(torch.einsum("hd,hld->hl", q, k), dim=-1)
        r[op["output"]] = T(op["o"]) @ torch.einsum("hl,hld->hd", w, v).reshape(-1)
    elif n == "GELU_GATE": r[op["output"]] = torch.nn.functional.gelu(r[op["left"]], approximate="tanh") * r[op["right"]]
    elif n == "HALT": pass
    else: raise ValueError(n)
err = float((r["y"] - T(program["target"])).abs().max())
print(json.dumps({"certificate":"PARAMETRIC_DECODER_BLOCK_REPLAY_OK", "max_replay_error":err,
                  "program_bytes":Path("outputs/flan_decoder_block_parametric.json").stat().st_size,
                  "fixture_bytes":Path("outputs/flan_decoder_block_fixture.json").stat().st_size}, indent=2))
assert err < 2e-4
