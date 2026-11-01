"""Committed Lean conformance vectors agree with the frozen Python reference.

The same literals are kernel-checked in lean/WaveLockProofs/MultiBlock.lean
(`conformance_pad_lengths`, `conformance_length_field`, `conformance_abc_block0`,
`conformance_constants`). This covers only the listed vectors, not full cc_hash parity.
"""
import json
import re
from pathlib import Path

from scripts.gen_lean_conformance import vectors

ROOT = Path(__file__).resolve().parents[1]


def test_committed_vectors_match_reference():
    committed = json.loads((ROOT / "lean" / "conformance_vectors.json").read_text())
    assert committed == vectors()


def test_lean_literals_match_vectors():
    v = vectors()
    lean = (ROOT / "lean" / "WaveLockProofs" / "MultiBlock.lean").read_text()
    for key in ("weight_t0_x0", "weight_t31_x255", "rho_0", "rho_100", "injection_u5_v7"):
        assert re.search(rf"= {v[key]}\b", lean), key
    assert "= 6513249" in lean and v["pad"]["abc"]["block0_cells_0_3"][0] == 6513249
    for name in ("empty", "abc", "len191", "len192"):
        assert f"= {v['pad'][name]['padded_len']}" in lean
        field = ", ".join(str(b) for b in v["pad"][name]["length_field"])
        assert f"[{field}]" in lean, name
