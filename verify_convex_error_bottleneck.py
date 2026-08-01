#!/usr/bin/env python3
import json
from decimal import Decimal
from pathlib import Path

x = json.loads(Path("outputs/flan_convex_error_bottleneck.json").read_text())
assert x["language"] == "FLAN-CONVEX-DAG-ERROR-AUDIT-1"
assert len(x["rows"]) == 129
assert [r["index"] for r in x["rows"]] == list(range(129))
active = [r for r in x["rows"] if r["cap_active"]]
assert active and active[0]["index"] == x["summary"]["first_active_diameter_cap_index"]
for r in x["rows"]:
    pre, post = Decimal(r["pre_cap_error_units"]), Decimal(r["post_cap_error_units"])
    assert post <= pre
    if r["diameter_cap_units"] is not None:
        cap = Decimal(r["diameter_cap_units"])
        assert post == min(pre, cap)
        assert r["cap_active"] == (pre > cap)
assert Decimal(x["summary"]["certified_one_step_tv"]) <= 1
assert x["summary"]["structured_fusions_used"] >= 2
assert not x["summary"]["nontrivial_tv"]
print("FLAN_CONVEX_DAG_ERROR_AUDIT_OK", x["summary"])
