"""Phase CC-3 -- singular-reachability, protocol, and document-hygiene tests.

Covers: exact v_star and constant preimage c; replay of the pinned round-1
witness; round-0 unreachability bound; singular-hit commitment sensitivity;
reference/optimized agreement; normative-protocol constants and encoding; all
byte strings valid; version/domain separation; and the stale-headline removal
in the top-level results document. Does not weaken existing tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from wavelock.curvature_capacity_v1 import (spec as bspec, optimized as bopt,
                                            reference as bref)
from wavelock.curvature_capacity import spec as aspec

P = bspec.P
N = bspec.N
GAMMA = bspec.GAMMA
V_STAR = bspec.V_STAR
RATE = bspec.RATE

# Pinned round-1 single-coordinate witness (Part V); psi_1[20] == v_star.
# 382 hex chars = 191 bytes; nonzero bytes at the cell-20 neighbourhood only.
WITNESS_HEX = "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000074e04fea20a000000000000000000000000000000000000000000000000000000000000000000000000000000000000da00ff0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
HIT_CELL = 20
C_CONST = 357959172


def test_exact_v_star():
    assert V_STAR == 195225786
    assert (1 + GAMMA * V_STAR) % P == 0
    assert V_STAR == (-pow(GAMMA, P - 2, P)) % P


def test_exact_constant_preimage():
    A, B, D = aspec.A, aspec.B, aspec.D
    assert (C_CONST + A * C_CONST * (B - C_CONST * C_CONST)) % P == V_STAR


def test_round0_unreachability_bound():
    """Every psi_0 coordinate is < v_star or a fixed constant != v_star."""
    iv = bopt.iv_psi().reshape(-1)
    # rate cells: max is iv + 2^24 - 1
    max_rate = int(iv[:RATE].max()) + (1 << 24) - 1
    assert max_rate < V_STAR
    # CAP1 with D_TAG
    cap1 = (int(iv[bspec.CAP1]) + bspec.D_TAG) % P
    assert cap1 != V_STAR
    # fixed cells 67..255
    for x in range(67, 256):
        assert int(iv[x]) != V_STAR


def test_witness_replays_to_vstar():
    msg = bytes.fromhex(WITNESS_HEX.replace("\n", ""))
    assert len(msg) == 191
    padded = bopt._pad(msg)
    arr = np.frombuffer(padded, dtype=np.uint8).astype(np.int64)
    nb = arr.size // bspec.BYTES_PER_BLOCK
    arr = arr.reshape(nb, RATE, 3)
    elems = arr[:, :, 0] + (arr[:, :, 1] << 8) + (arr[:, :, 2] << 16)
    psi = bopt.iv_psi().reshape(-1).copy()
    psi[:RATE] = (psi[:RATE] + elems[0]) % P
    psi1 = bopt._wave_round(psi.reshape(N, N)) % P
    assert int(psi1.reshape(-1)[HIT_CELL]) == V_STAR
    assert int(np.sum(psi1 == V_STAR)) == 1  # exactly one singular coordinate


def test_singular_hit_round0_injection_is_zero():
    msg = bytes.fromhex(WITNESS_HEX.replace("\n", ""))
    padded = bopt._pad(msg)
    arr = np.frombuffer(padded, dtype=np.uint8).astype(np.int64).reshape(-1, RATE, 3)
    elems = arr[:, :, 0] + (arr[:, :, 1] << 8) + (arr[:, :, 2] << 16)
    psi0 = bopt.iv_psi().reshape(-1).copy()
    psi0[:RATE] = (psi0[:RATE] + elems[0]) % P
    psi1 = bopt._wave_round(psi0.reshape(N, N)).reshape(-1) % P
    u = int(psi0[HIT_CELL]); v = int(psi1[HIT_CELL])
    assert v == V_STAR
    assert (u + GAMMA * (u * v % P)) % P == 0  # j_B zeroed at the hit cell


def test_singular_hit_commitment_sensitivity():
    """Changing the singular cell's bytes changes the digest (no erasure)."""
    msg = bytes.fromhex(WITNESS_HEX.replace("\n", ""))
    d0 = bopt.cc_hash(msg)
    m2 = bytearray(msg)
    pos = 3 * HIT_CELL
    m2[pos] = (m2[pos] + 1) & 0xFF
    assert bopt.cc_hash(bytes(m2)) != d0


def test_witness_reference_optimized_agree():
    msg = bytes.fromhex(WITNESS_HEX.replace("\n", ""))
    assert bref.cc_hash(msg) == bopt.cc_hash(msg)


def test_normative_protocol_constants():
    assert bspec.BYTES_PER_BLOCK == 192
    assert bspec.RATE == 64
    assert (bspec.CAP0, bspec.CAP1, bspec.CAP2) == (64, 65, 66)
    assert bspec.T == 32
    assert bspec.ETA == 0 and bspec.ZETA == 0
    assert bspec.VERSION == "WaveLock-CC-Core-v1"


def test_padding_block_multiple():
    for L in (0, 1, 50, 191, 192, 193):
        padded = bopt._pad(b"\x00" * L)
        assert len(padded) % bspec.BYTES_PER_BLOCK == 0
        assert len(padded) >= bspec.BYTES_PER_BLOCK


def test_all_byte_strings_valid():
    """Any byte string is a valid message (no byte-level invalid encodings)."""
    for m in (b"", b"\x00", bytes(range(256)), b"\xff" * 500):
        d = bopt.cc_hash(m)
        assert isinstance(d, (bytes, bytearray)) and len(d) == 32


def test_version_and_domain_separation():
    assert bspec.VERSION != aspec.VERSION
    assert bspec.D_TAG != aspec.D_TAG


def test_results_doc_leads_with_cc3():
    """Stale-headline removal: the results doc's first section is the CC-3 verdict,
    before any historical Phase-1 conclusion, and no stale 'z3 not installed'
    appears outside a historical section."""
    path = os.path.join(os.path.dirname(__file__), os.pardir, "docs",
                        "WAVELOCK_CURVATURE_CAPACITY_RESULTS.md")
    text = open(path, encoding="utf-8").read()
    cur = text.find("## Current Verdict — Phase CC-3")
    hist = text.find("## Historical results")
    phase1 = text.find("Phase 1 historical result")
    assert cur != -1, "results doc must contain the Current Verdict — Phase CC-3 section"
    assert hist != -1 and hist > cur, "historical section must follow the current verdict"
    assert phase1 > cur, "Phase 1 conclusion must be in the historical section (after CC-3)"
    # the stale 'not installed' note must only appear after the historical marker
    idx = text.find("not** installed")
    if idx == -1:
        idx = text.find("not installed")
    if idx != -1:
        assert idx > hist, "stale 'not installed' note must be inside historical section"
