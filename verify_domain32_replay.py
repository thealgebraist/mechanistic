"""Replay the concrete and quotient PRSL laws from serialized tables only."""
import argparse, gzip, json
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument('--source',default='outputs/flan_domain32_program.json.gz'); ap.add_argument('--quotient',default='outputs/flan_domain32_quotient.json'); a=ap.parse_args()
src = json.loads(gzip.decompress(Path(a.source).read_bytes()))
quo = json.loads(Path(a.quotient).read_text())
H, B = src["horizon"], 2
source = src["states"]
qstates = {s["id"]: s for s in quo["states"]}
lookup = {(s["prompt_id"], tuple(s["stack"])): i for i, s in enumerate(source)}
membership = {i: q["id"] for q in quo["states"] for i in q["members"]}

def law(s):
    top = [(int(t), v / 65535.0) for t, v in s["emit"][:B]]
    return top, 1.0 - sum(p for _, p in top)

def tv(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)

def concrete_children(i):
    s = source[i]
    return {t: lookup.get((s["prompt_id"], tuple(s["stack"]) + (t,)))
            for t, _ in s["emit"][:src["branching_source"]]}

def replay_source(root):
    out = {}
    def rec(i, prefix, p, d):
        if d == H:
            out[tuple(prefix)] = out.get(tuple(prefix), 0.0) + p; return
        top, other = law(source[i]); children = concrete_children(i)
        for token, mass in top:
            j = children.get(token)
            if d == H - 1: out[tuple(prefix + [token])] = out.get(tuple(prefix + [token]), 0.0) + p * mass
            elif j is not None: rec(j, prefix + [token], p * mass, d + 1)
        out[tuple(prefix + ["OTHER"])] = out.get(tuple(prefix + ["OTHER"]), 0.0) + p * other
    rec(root, [], 1.0, 0); return out

def replay_quotient(root):
    out = {}
    def rec(qi, prefix, p, d):
        if d == H:
            out[tuple(prefix)] = out.get(tuple(prefix), 0.0) + p; return
        s = qstates[qi]; top, other = law(s); children = dict((int(t), int(j)) for t, j in s["next"])
        for token, mass in top:
            if d == H - 1: out[tuple(prefix + [token])] = out.get(tuple(prefix + [token]), 0.0) + p * mass
            elif token in children: rec(children[token], prefix + [token], p * mass, d + 1)
        out[tuple(prefix + ["OTHER"])] = out.get(tuple(prefix + ["OTHER"]), 0.0) + p * other
    rec(root, [], 1.0, 0); return out

errors = []
for pid in range(len(src["prompts"])):
    root = next(i for i, s in enumerate(source) if s["prompt_id"] == pid and s["depth"] == 0)
    a, b = replay_source(root), replay_quotient(membership[root])
    errors.append(tv(a, b))
bound = min(1.0, H * float(quo["max_local_tv"]))
assert max(errors) <= bound + 1e-12
print(json.dumps({"certificate": "DOMAIN32_REPLAY_LAW_OK", "prompts": len(errors),
                  "max_sequence_tv": max(errors), "mean_sequence_tv": sum(errors)/len(errors),
                  "horizon_union_bound": bound,
                  "scope": "serialized finite oracle table"}, indent=2))
