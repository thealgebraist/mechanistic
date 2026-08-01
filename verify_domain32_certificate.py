"""Dependency-free checker for the finite-domain PRSL quotient certificate.

This checks the claims that can be checked from the serialized oracle table:
normalization, well-formed memberships, deterministic support-preserving
successors, local total-variation error, and the finite-horizon union bound.
It does not claim equivalence outside the supplied prompt domain.
"""
import argparse, gzip, json, math
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument('--source',default='outputs/flan_domain32_program.json.gz'); ap.add_argument('--quotient',default='outputs/flan_domain32_quotient.json'); a=ap.parse_args()
SRC_PATH = Path(a.source)
Q_PATH = Path(a.quotient)
src = json.loads(gzip.decompress(SRC_PATH.read_bytes()))
quo = json.loads(Q_PATH.read_text())
source = src["states"]
qstates = quo["states"]
assert quo["source_states"] == len(source)
assert quo["quotient_states"] == len(qstates)
assert quo["horizon"] == src["horizon"]

def mass(s):
    return sum(v for _, v in s["emit"]) + s["other"]

def dist(s):
    d = {int(t): v / 65535.0 for t, v in s["emit"]}
    d[-1] = s["other"] / 65535.0
    return d

def tv(a, b):
    da, db = dist(a), dist(b)
    return 0.5 * sum(abs(da.get(k, 0.0) - db.get(k, 0.0)) for k in set(da) | set(db))

assert all(mass(s) == 65535 for s in source), "source mass failure"
assert all(mass(s) == 65535 for s in qstates), "quotient mass failure"

membership = {}
for qi, q in enumerate(qstates):
    assert q["id"] == qi
    assert len(q["members"]) == len(q["belief_weights"])
    assert abs(sum(q["belief_weights"]) - 1.0) < 1e-12
    for si in q["members"]:
        assert 0 <= si < len(source)
        assert si not in membership, "overlapping quotient blocks"
        membership[si] = qi
        assert source[si]["depth"] == q["depth"]
assert len(membership) == len(source), "quotient does not cover source"

def child_signature(si):
    s = source[si]
    if s["depth"] >= src["horizon"]:
        return ()
    lookup = {(x["prompt_id"], tuple(x["stack"])): j for j, x in enumerate(source)}
    out = []
    for tok, _ in s["emit"][:src["branching_source"]]:
        j = lookup.get((s["prompt_id"], tuple(s["stack"]) + (tok,)))
        if j is not None:
            out.append((tok, membership[j]))
    return tuple(out)

max_local = 0.0
for qi, q in enumerate(qstates):
    sigs = {child_signature(si) for si in q["members"]}
    assert len(sigs) == 1, f"non-deterministic quotient successor block {qi}"
    assert tuple((int(t), int(b)) for t, b in q["next"]) == next(iter(sigs)), f"bad successor block {qi}"
    for si in q["members"]:
        max_local = max(max_local, tv(source[si], q))

assert abs(max_local - quo["max_local_tv"]) < 2e-12
bound = min(1.0, src["horizon"] * max_local)
assert abs(bound - quo["horizon_bound"]) < 2e-12
roots = {int(k): v for k, v in quo["roots"].items()}
assert len(roots) == len(src["prompts"])
for pid, qi in roots.items():
    assert qstates[qi]["depth"] == 0
    assert source[qstates[qi]["members"][0]]["prompt_id"] == pid

print(json.dumps({
    "certificate": "DOMAIN32_QUOTIENT_CERTIFICATE_OK",
    "prompts": len(src["prompts"]),
    "source_states": len(source),
    "quotient_states": len(qstates),
    "max_local_tv": max_local,
    "horizon": src["horizon"],
    "horizon_union_bound": bound,
    "scope": "finite supplied prompt domain and serialized oracle table",
}, indent=2))
