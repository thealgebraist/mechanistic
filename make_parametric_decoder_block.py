"""Turn the frozen decoder-block trace into a reusable input-parametric graph."""
import json
from pathlib import Path

src = json.loads(Path("outputs/flan_decoder_block_level1.json").read_text())
fixture = {}
ops = []
for op in src["ops"]:
    op = dict(op)
    if op["op"] == "LOAD_VECTOR" and op["name"] == "x0":
        fixture["x0"] = op.pop("values"); op["op"] = "INPUT_VECTOR"
    elif op["op"] == "LOAD_MATRIX" and op["name"] == "memory":
        fixture["memory"] = op.pop("values"); op["op"] = "INPUT_MATRIX"
    ops.append(op)
out = dict(src); out["language"] = "NEURAL-ALGEBRA-1-PARAMETRIC"; out["ops"] = ops
out["inputs"] = {"x0": "Vect[512] Float32", "memory": "Matrix[11,512] Float32"}
Path("outputs/flan_decoder_block_parametric.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
Path("outputs/flan_decoder_block_fixture.json").write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
print(json.dumps({"parametric_bytes":Path("outputs/flan_decoder_block_parametric.json").stat().st_size,
                  "fixture_bytes":Path("outputs/flan_decoder_block_fixture.json").stat().st_size,
                  "inputs":out["inputs"]}, indent=2))
