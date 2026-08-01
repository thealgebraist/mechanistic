#!/usr/bin/env python3
"""Verify that the generated Lean tag sequence exactly mirrors the JSON graph."""
import hashlib, json, re
from pathlib import Path

graph_path = Path("outputs/flan_full_graph.json")
graph = json.loads(graph_path.read_text()); lean = Path("GeneratedFlanProgram.lean").read_text()
sha = hashlib.sha256(graph_path.read_bytes()).hexdigest()
assert f'def fullGraphSha256 : String := "{sha}"' in lean
assert f'def checkpointSha256 : String := "{graph["checkpoint_sha256"]}"' in lean
ctor = lambda name: "op" + "".join(part.title() for part in name.lower().split("_"))
body = re.search(r"def programTags : List OpTag := \[\n(.*?)\n\]", lean, re.S).group(1)
actual = [x.strip().rstrip(",") for x in body.splitlines() if x.strip()]
expected = [ctor(op["op"]) for op in graph["ops"]]
assert actual == expected and len(actual) == 129
print(json.dumps({"certificate": "GENERATED_FLAN_LEAN_PROGRAM_OK", "opcodes": 129,
                  "order_exact": True, "graph_sha256": sha,
                  "checkpoint_sha256": graph["checkpoint_sha256"]}, indent=2))
