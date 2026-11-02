#!/usr/bin/env python3
"""Regenerate lean/conformance_vectors.json from the frozen CC-Core-v1 Python reference.

The Lean theorems `conformance_*` in lean/WaveLockProofs/MultiBlock.lean state the same
literal values; tests/test_lean_conformance.py checks the committed JSON against this
generator so a drift on either side is detected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wavelock.curvature_capacity_v1 import commit, reference, spec  # noqa: E402

CASES = [("empty", b""), ("abc", b"abc"), ("len191", bytes(191)), ("len192", bytes([255]) * 192)]


def vectors() -> dict:
    out: dict = {"pad": {}}
    for name, m in CASES:
        p = reference.pad(m)
        out["pad"][name] = {
            "len": len(m),
            "padded_len": len(p),
            "blocks": len(p) // spec.BYTES_PER_BLOCK,
            "length_field": list(p[-192:-184]),
            "block0_cells_0_3": reference.pack_block(p, 0)[:4],
        }
    out["weight_t0_x0"] = commit.weight(0, 0)
    out["weight_t31_x255"] = commit.weight(31, 255)
    out["rho_0"] = spec.round_constant(0)
    out["rho_100"] = spec.round_constant(100)
    out["counter_0"] = list(spec.encode_block_counter(0))
    out["counter_p_minus_1"] = list(spec.encode_block_counter(spec.P - 1))
    out["injection_u5_v7"] = commit.injection(5, 7)
    out["v_star"] = spec.V_STAR
    return out


if __name__ == "__main__":
    print(json.dumps(vectors(), indent=2, sort_keys=True))
