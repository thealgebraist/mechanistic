#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

graph_path = Path("outputs/flan_full_graph.json")
graph = json.loads(graph_path.read_text())
path = Path("outputs/flan_ieee_add_transfers.json")
cert = json.loads(path.read_text())
assert cert["graph_sha256"] == hashlib.sha256(graph_path.read_bytes()).hexdigest()
assert cert["checkpoint_sha256"] == graph["checkpoint_sha256"]
adds = [(i,o) for i,o in enumerate(graph["ops"]) if o["op"] == "ADD"]
assert len(adds) == len(cert["transfers"]) == 40
for (index, op), rec in zip(adds, cert["transfers"]):
    assert rec["index"] == index and rec["left"] == op["left"] and rec["right"] == op["right"]
    payload = dict(rec); digest = payload.pop("transfer_sha256")
    assert digest == hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert rec["gain"] == 2 and rec["bias_units"] > 0
    assert rec["status"] == "CERTIFIED_UNDER_IEEE754_RNE"
tsv = Path("outputs/flan_ieee_add_transfers.tsv")
assert cert["tsv_sha256"] == hashlib.sha256(tsv.read_bytes()).hexdigest()
print(json.dumps({"certificate":"IEEE_ADD_TRANSFERS_OK", "add_occurrences":40,
                  "coverage_exact":True, "source_contract":"IEEE754_RNE"}, indent=2))
