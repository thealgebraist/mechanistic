#!/usr/bin/env python3
"""Independently verify nested partitions and commuting quotient transitions."""
import argparse, gzip, hashlib, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="outputs/flan_domain32_program.json.gz")
ap.add_argument("--tower", default="outputs/flan_domain32_refinement_tower.json")
a = ap.parse_args()
raw = Path(a.source).read_bytes(); src = json.loads(gzip.decompress(raw))
tower = json.loads(Path(a.tower).read_text()); n = len(src["states"])
assert tower["source_sha256"] == hashlib.sha256(raw).hexdigest()
quant_tv = src["probability_encoding"]["explicit_tokens"] / (2 * src["probability_encoding"]["units"])
assert tower["source_quantization_tv_bound"] == quant_tv
states = src["states"]
by = {(s["prompt_id"], tuple(s["stack"])): i for i, s in enumerate(states)}
children = []
for s in states:
    row = []
    for token, _ in s["emit"][:src["branching_source"]]:
        j = by.get((s["prompt_id"], tuple(s["stack"]) + (token,)))
        if j is not None: row.append((int(token), j))
    children.append(row)

def dist(s):
    out = {int(k): int(v) for k, v in s["emit"]}; out[-1] = int(s["other"])
    return out

def tv(x, y):
    dx, dy = dist(x), dist(y)
    return 0.5 * sum(abs(dx.get(k, 0) - dy.get(k, 0))
                     for k in set(dx) | set(dy)) / 65535

def projected_law(s):
    top = [(int(t), int(v) / 65535.0) for t, v in s["emit"][:src["branching_source"]]]
    return top, 1.0 - sum(p for _, p in top)

def trace_tv(x, y):
    return 0.5 * sum(abs(x.get(k, 0.0) - y.get(k, 0.0)) for k in set(x) | set(y))

def replay_source(root):
    out = {}
    def rec(i, prefix, mass, depth):
        top, other = projected_law(states[i]); child = dict(children[i])
        for token, p in top:
            key = tuple(prefix + [token])
            if depth == src["horizon"] - 1: out[key] = out.get(key, 0.0) + mass * p
            elif token in child: rec(child[token], prefix + [token], mass * p, depth + 1)
        key = tuple(prefix + ["OTHER"]); out[key] = out.get(key, 0.0) + mass * other
    rec(root, [], 1.0, 0); return out

previous = None
for k, level in enumerate(tower["levels"]):
    blocks = level["blocks"]
    members = [i for b in blocks for i in b["members"]]
    assert sorted(members) == list(range(n)) and len(members) == len(set(members))
    assert level["states"] == len(blocks)
    source_to_block = {}
    recomputed_max = 0.0
    for b in blocks:
        for i in b["members"]: source_to_block[i] = b["id"]
        rep = b["representative"]
        assert rep in b["members"]
        assert b["emit"] == states[rep]["emit"] and b["other"] == states[rep]["other"]
        for i in b["members"]:
            value = tv(states[rep], states[i])
            recomputed_max = max(recomputed_max, value)
            assert value <= level["epsilon"] + 1e-15
        assert len(b["edge_certificates"]) == len(b["next"])
        for edge, (token, successor) in zip(b["edge_certificates"], b["next"]):
            proof = edge["proof"]; cert = proof.pop("certificate_sha256")
            encoded_edge = json.dumps(edge, sort_keys=True, separators=(",", ":")).encode()
            assert cert == hashlib.sha256(encoded_edge).hexdigest()
            proof["certificate_sha256"] = cert
            assert edge["source_type"] == b["id"]
            assert edge["token_predicate"] == {"equals": token}
            assert edge["destination_type"] == successor
    for b in blocks:
        rep = b["representative"]
        assert b["next"] == [[token, source_to_block[j]] for token, j in children[rep]]
    assert abs(recomputed_max - level["max_representative_tv"]) < 1e-15
    assert abs(min(1.0, src["horizon"] * recomputed_max) - level["horizon_bound"]) < 1e-15
    assert abs(min(1.0, src["horizon"] * (recomputed_max + quant_tv)) -
               level["neural_horizon_bound"]) < 1e-15
    qby = {b["id"]: b for b in blocks}
    def replay_graph(root):
        out = {}
        def rec(q, prefix, mass, depth):
            s = qby[q]; top, other = projected_law(s); child = dict(s["next"])
            for token, p in top:
                key = tuple(prefix + [token])
                if depth == src["horizon"] - 1: out[key] = out.get(key, 0.0) + mass * p
                elif token in child: rec(child[token], prefix + [token], mass * p, depth + 1)
            key = tuple(prefix + ["OTHER"]); out[key] = out.get(key, 0.0) + mass * other
        rec(root, [], 1.0, 0); return out
    trace_errors = []
    for pid in range(len(src["prompts"])):
        concrete = next(i for i, s in enumerate(states) if s["prompt_id"] == pid and s["depth"] == 0)
        trace_errors.append(trace_tv(replay_source(concrete), replay_graph(level["roots"][str(pid)])))
    direct_max = max(trace_errors)
    assert abs(direct_max - level["direct_trace_max_tv"]) < 1e-15
    assert abs(sum(trace_errors) / len(trace_errors) - level["direct_trace_mean_tv"]) < 1e-15
    assert abs(min(1.0, direct_max + src["horizon"] * quant_tv) -
               level["direct_neural_trace_bound"]) < 1e-15
    assert level["status"] == ("certified exact" if direct_max == 0.0 else "certified approximate")
    payload = dict(level); cert = payload.pop("certificate_sha256")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert cert == hashlib.sha256(encoded).hexdigest()
    if previous is not None:
        parent = level["forgetful_parent"]
        assert len(parent) == len(blocks)
        for b in blocks:
            p = parent[b["id"]]
            assert set(b["members"]) <= set(previous["blocks"][p]["members"])
            coarse_next = {tok: q for tok, q in previous["blocks"][p]["next"]}
            for tok, fine_q in b["next"]:
                assert tok in coarse_next
                assert parent[fine_q] == coarse_next[tok]
        for prompt, fine_root in level["roots"].items():
            assert parent[fine_root] == previous["roots"][prompt]
        assert level["epsilon"] <= previous["epsilon"]
        assert level["states"] >= previous["states"]
        for witness in level["split_witnesses"]:
            left, right = witness["left_state"], witness["right_state"]
            assert left in previous["blocks"][witness["parent"]]["members"]
            assert right in previous["blocks"][witness["parent"]]["members"]
            assert witness["left_child"] != witness["right_child"]
            assert left in blocks[witness["left_child"]]["members"]
            assert right in blocks[witness["right_child"]]["members"]
    previous = level
assert tower["levels"][-1]["states"] == n
assert all(len(b["members"]) == 1 for b in tower["levels"][-1]["blocks"])
print(json.dumps({"certificate": "REFINEMENT_TOWER_OK", "levels": len(tower["levels"]),
                  "state_counts": [x["states"] for x in tower["levels"]],
                  "forgetful_maps": len(tower["levels"]) - 1,
                  "source_states": n, "coverage": tower["coverage"]}, indent=2))
