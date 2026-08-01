#!/usr/bin/env python3
"""Build a nested coarse-to-fine FLAN quotient tower with proof metadata."""
import argparse, gzip, hashlib, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="outputs/flan_domain32_program.json.gz")
ap.add_argument("--output", default="outputs/flan_domain32_refinement_tower.json")
ap.add_argument("--deltas", default="0.10,0.05,0.02,0")
a = ap.parse_args()

raw = Path(a.source).read_bytes()
src = json.loads(gzip.decompress(raw))
states = src["states"]
deltas = [float(x) for x in a.deltas.split(",")]
quant_tv = src["probability_encoding"]["explicit_tokens"] / (2 * src["probability_encoding"]["units"])
assert all(deltas[i] >= deltas[i + 1] for i in range(len(deltas) - 1))
by = {(s["prompt_id"], tuple(s["stack"])): i for i, s in enumerate(states)}

def distribution(s):
    d = {int(k): int(v) for k, v in s["emit"]}
    d[-1] = int(s["other"])
    return d

def tv(x, y):
    dx, dy = distribution(x), distribution(y)
    return 0.5 * sum(abs(dx.get(k, 0) - dy.get(k, 0)) for k in set(dx) | set(dy)) / 65535

def projected_law(s):
    top = [(int(t), int(v) / 65535.0) for t, v in s["emit"][:src["branching_source"]]]
    return top, 1.0 - sum(p for _, p in top)

def law_tv(x, y):
    return 0.5 * sum(abs(x.get(k, 0.0) - y.get(k, 0.0)) for k in set(x) | set(y))

def direct_trace_errors(qstates, roots):
    qby = {s["id"]: s for s in qstates}
    def replay_source(root):
        out = {}
        def rec(i, prefix, mass, depth):
            top, other = projected_law(states[i])
            child = dict(children[i])
            for token, p in top:
                key = tuple(prefix + [token])
                if depth == src["horizon"] - 1: out[key] = out.get(key, 0.0) + mass * p
                elif token in child: rec(child[token], prefix + [token], mass * p, depth + 1)
            key = tuple(prefix + ["OTHER"]); out[key] = out.get(key, 0.0) + mass * other
        rec(root, [], 1.0, 0); return out
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
    errors = []
    for pid in range(len(src["prompts"])):
        concrete = next(i for i, s in enumerate(states) if s["prompt_id"] == pid and s["depth"] == 0)
        errors.append(law_tv(replay_source(concrete), replay_graph(roots[str(pid)])))
    return errors

children = []
for s in states:
    row = []
    for token, _ in s["emit"][:src["branching_source"]]:
        j = by.get((s["prompt_id"], tuple(s["stack"]) + (token,)))
        if j is not None:
            row.append((int(token), j))
    children.append(row)

levels = []
parent_partition = [0] * len(states)
for level_id, delta in enumerate(deltas):
    block = [-1] * len(states)
    groups = []
    # The bounded trace graph is acyclic by depth, so this bottom-up pass is
    # the fixed point of successor-signature refinement.
    for depth in range(src["horizon"] - 1, -1, -1):
        candidates = [i for i, s in enumerate(states) if s["depth"] == depth]
        for i in candidates:
            sig_i = tuple((tok, block[j]) for tok, j in children[i])
            top_i = tuple(tok for tok, _ in states[i]["emit"][:src["branching_source"]])
            placed = False
            for bid, g in enumerate(groups):
                j = g[0]
                if states[j]["depth"] != depth or parent_partition[j] != parent_partition[i]:
                    continue
                sig_j = tuple((tok, block[z]) for tok, z in children[j])
                top_j = tuple(tok for tok, _ in states[j]["emit"][:src["branching_source"]])
                if sig_i == sig_j and top_i == top_j and all(tv(states[i], states[z]) <= delta for z in g):
                    g.append(i); block[i] = bid; placed = True; break
            if not placed:
                block[i] = len(groups); groups.append([i])

    qstates = []
    max_local = 0.0
    for bid, members in enumerate(groups):
        rep = members[0]
        max_local = max(max_local, *(tv(states[rep], states[i]) for i in members))
        edge_certificates = []
        for token, successor in [[tok, block[j]] for tok, j in children[rep]]:
            record = {
                "source_type": bid, "token_predicate": {"equals": token},
                "destination_type": successor, "coverage": "exact-finite-domain",
                "error_contract": {"metric": "total_variation", "epsilon": delta,
                                   "horizon": src["horizon"]},
                "proof": {"verifier": "verify_refinement_tower.py",
                          "assumptions": ["serialized-oracle-complete", "depth-acyclic"]},
                "residual_route": "graph",
            }
            edge_payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            record["proof"]["certificate_sha256"] = hashlib.sha256(edge_payload).hexdigest()
            edge_certificates.append(record)
        qstates.append({
            "id": bid,
            "depth": states[rep]["depth"],
            "members": members,
            "representative": rep,
            "emit": states[rep]["emit"],
            "other": states[rep]["other"],
            "next": [[tok, block[j]] for tok, j in children[rep]],
            "edge_certificates": edge_certificates,
        })
    if level_id == 0:
        forgetful = None
    else:
        forgetful = []
        for members in groups:
            parents = {parent_partition[i] for i in members}
            assert len(parents) == 1
            forgetful.append(next(iter(parents)))
    roots = {str(s["prompt_id"]): block[i] for i, s in enumerate(states) if s["depth"] == 0}
    split_witnesses = []
    if level_id > 0:
        children_by_parent = {}
        for fine_id, members in enumerate(groups):
            children_by_parent.setdefault(parent_partition[members[0]], []).append(fine_id)
        for parent_id, fine_ids in children_by_parent.items():
            if len(fine_ids) < 2:
                continue
            x, y = groups[fine_ids[0]][0], groups[fine_ids[1]][0]
            sx = tuple((tok, block[j]) for tok, j in children[x])
            sy = tuple((tok, block[j]) for tok, j in children[y])
            differing_token = next((tx for (tx, qx), (ty, qy) in zip(sx, sy)
                                     if tx != ty or qx != qy), None)
            split_witnesses.append({"parent": parent_id, "left_state": x,
                                    "right_state": y, "output_tv": tv(states[x], states[y]),
                                    "distinguishing_token": differing_token,
                                    "left_child": fine_ids[0], "right_child": fine_ids[1]})
    level = {
        "level": level_id,
        "epsilon": delta,
        "horizon": src["horizon"],
        "states": len(groups),
        "max_representative_tv": max_local,
        "horizon_bound": min(1.0, src["horizon"] * max_local),
        "neural_horizon_bound": min(1.0, src["horizon"] * (max_local + quant_tv)),
        "roots": roots,
        "forgetful_parent": forgetful,
        "split_witnesses": split_witnesses,
        "blocks": qstates,
    }
    trace_errors = direct_trace_errors(qstates, roots)
    level["direct_trace_max_tv"] = max(trace_errors)
    level["direct_trace_mean_tv"] = sum(trace_errors) / len(trace_errors)
    level["direct_neural_trace_bound"] = min(1.0, max(trace_errors) + src["horizon"] * quant_tv)
    level["status"] = "certified exact" if max(trace_errors) == 0.0 else "certified approximate"
    payload = json.dumps(level, sort_keys=True, separators=(",", ":")).encode()
    level["certificate_sha256"] = hashlib.sha256(payload).hexdigest()
    levels.append(level)
    parent_partition = block

out = {
    "language": "PRSL-REFINEMENT-TOWER-1",
    "source": a.source,
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "checkpoint_sha256": "495fa51e204676f1a857a9fc13c4c89f3f5ba9f480b898cebca02add25e6d749",
    "coverage": "exact finite serialized oracle domain",
    "source_quantization_tv_bound": quant_tv,
    "residual_classification": {"graph_covered": len(states), "neural_residual": 0,
                                "unverified": 0, "unexplored": "outside finite domain"},
    "levels": levels,
}
Path(a.output).write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"levels": [{"epsilon": x["epsilon"], "states": x["states"],
                              "max_tv": x["max_representative_tv"],
                              "bound": x["horizon_bound"]} for x in levels]}, indent=2))
