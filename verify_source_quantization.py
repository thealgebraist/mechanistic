#!/usr/bin/env python3
"""Certify projected softmax-to-uint16 mass quantization on the oracle table."""
import argparse, gzip, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="outputs/flan_domain32_program.json.gz")
a = ap.parse_args()
src = json.loads(gzip.decompress(Path(a.source).read_bytes()))
units = src["probability_encoding"]["units"]
k = src["probability_encoding"]["explicit_tokens"]
analytic = k / (2 * units)
maximum = 0.0
for s in src["states"]:
    ref = {int(t): float(p) for t, p in s["reference_projected"]["emit"]}
    quant = {int(t): int(q) / units for t, q in s["emit"]}
    assert ref.keys() == quant.keys()
    value = 0.5 * (sum(abs(ref[t] - quant[t]) for t in ref) +
                   abs(float(s["reference_projected"]["other"]) - int(s["other"]) / units))
    maximum = max(maximum, value)
    assert value <= analytic + 1e-12
print(json.dumps({"certificate": "SOURCE_UINT16_QUANTIZATION_OK",
                  "states": len(src["states"]), "explicit_tokens": k,
                  "units": units, "analytic_tv_bound": analytic,
                  "observed_max_projected_tv": maximum,
                  "proof": "nearest rounding gives <=1/(2U) per explicit mass; OTHER is the remainder"},
                 indent=2))
