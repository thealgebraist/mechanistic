#!/usr/bin/env python3
"""Choose the finest certified quotient level fitting a whole-domain budget."""
import argparse, hashlib, json, struct
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bytes", type=int, required=True)
ap.add_argument("--tower", default="outputs/flan_domain32_refinement_tower.json")
ap.add_argument("--output", default=None)
a = ap.parse_args()
tower_raw = Path(a.tower).read_bytes(); tower = json.loads(tower_raw)
levels = tower["levels"]
prompts = len(levels[0]["roots"])

def binary_size(level):
    return 6 + 6 + 2 * prompts + 19 * level["states"]

eligible = [level for level in levels if binary_size(level) <= a.bytes]
if not eligible:
    minimum = binary_size(levels[0])
    raise SystemExit(f"whole certified domain requires at least {minimum} bytes; budget={a.bytes}")
level = eligible[-1]
out = bytearray(b"PRSL1\0")
out += struct.pack("<HHBB", level["states"], prompts, level["horizon"], 2)
for pid in range(prompts): out += struct.pack("<H", int(level["roots"][str(pid)]))
for s in level["blocks"]:
    emit = s["emit"][:2]
    # Dropping explicit tail labels into `other` is a deterministic measurable
    # map, so total variation cannot increase (data-processing inequality).
    other = 65535 - sum(mass for _, mass in emit)
    out += struct.pack("<BH", s["depth"], other)
    for token, mass in emit: out += struct.pack("<HH", token, mass)
    for _ in range(2 - len(emit)): out += struct.pack("<HH", 0, 0)
    for token, target in s["next"][:2]: out += struct.pack("<HH", token, target)
    for _ in range(2 - len(s["next"][:2])): out += struct.pack("<HH", 0, 0)
path = Path(a.output or f"outputs/flan_tower_budget_{a.bytes}.prslb")
path.write_bytes(out)
certificate = {
    "language": "PRSL-TOWER-BUDGET-1",
    "artifact": str(path), "requested_bytes": a.bytes, "actual_bytes": len(out),
    "selected_level": level["level"], "states": level["states"], "prompts": prompts,
    "epsilon": level["epsilon"], "horizon": level["horizon"],
    "certified_horizon_tv_bound": level["horizon_bound"],
    "certified_neural_horizon_tv_bound": level["neural_horizon_bound"],
    "direct_trace_max_tv": level["direct_trace_max_tv"],
    "direct_neural_trace_bound": level["direct_neural_trace_bound"],
    "status": level["status"],
    "tail_projection": "top-2 tokens plus OTHER; TV nonexpansive",
    "tower_sha256": hashlib.sha256(tower_raw).hexdigest(),
    "level_certificate_sha256": level["certificate_sha256"],
    "binary_sha256": hashlib.sha256(out).hexdigest(),
    "coverage": tower["coverage"],
}
Path(str(path) + ".json").write_text(json.dumps(certificate, indent=2) + "\n")
print(json.dumps(certificate, indent=2))
assert len(out) <= a.bytes
