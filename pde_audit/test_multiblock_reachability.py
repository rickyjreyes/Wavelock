"""Tests for Phase 8K multi-block reachability + reduced-model lifting.

Proves the audit trace reproduces the normative implementation exactly, pins the
first-block impossibility, and pins deterministic reduced-model lifting facts
(including an exact, verified message-preimage-of-zero in a small system).
"""

import numpy as np
import pytest

from wavelock.pde_hash import spec, optimized as opt, reference as ref
from pde_audit import multiblock_reachability as mb
from pde_audit import reduced_lifting as rl

P, N = spec.P, spec.N


# ---------------------------------------------------------------------------
# Part II: trace parity with the normative implementations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("msg", [b"", b"abc", b"\x00" * 200, b"\xff" * 193,
                                  bytes(range(64)), b"WaveLock-multiblock-test"])
def test_block_trace_matches_normative(msg):
    pre = mb.block_trace(msg)["pre_squeeze"].reshape(N, N)
    assert np.array_equal(pre, opt._absorb(msg) % P)
    assert np.array_equal(pre.reshape(-1), np.array(ref.absorb(msg).cells, dtype=np.int64) % P)


def test_block_trace_rounds_compose_to_block():
    tr = mb.block_trace(b"abc", capture_rounds=True)
    # last captured round of each block equals S_after
    for blk in tr["blocks"]:
        assert np.array_equal(blk["rounds"][-1], blk["S_after"])


# ---------------------------------------------------------------------------
# Part IV: first-block impossibility
# ---------------------------------------------------------------------------
def test_first_block_eigenmode_unreachable():
    fb = mb.first_block_impossibility()
    # every uncontrolled capacity cell mismatches every eigenmode variant
    assert fb["min_uncontrolled_mismatch_over_all_variants"] > 0
    assert fb["iv_values_at_uncontrolled_max"] < 1000  # IV cells are small


# ---------------------------------------------------------------------------
# modular linear algebra helpers
# ---------------------------------------------------------------------------
def test_matmul_and_solve_mod():
    from pde_audit._harness import matmul_mod, mod_solve
    g = np.random.default_rng(0)
    A = g.integers(0, P, size=(20, 12), dtype=np.int64)
    x = g.integers(0, P, size=12, dtype=np.int64)
    y = matmul_mod(A, x.reshape(12, 1), P).reshape(-1)
    xs = mod_solve(A, y, P)
    assert xs is not None
    assert np.array_equal(matmul_mod(A, xs.reshape(12, 1), P).reshape(-1), y % P)


# ---------------------------------------------------------------------------
# Part VII: reduced-model lifting facts (deterministic, exact)
# ---------------------------------------------------------------------------
def test_reduced_no_lift_with_too_few_blocks():
    # N=2,p=7,T=1: 1 and 2 blocks cannot reach zero (matches SMT UNSAT)
    for blk in (1, 2):
        M = rl.RModel(2, 7, 5, 3, 3, 1, blk)
        r = rl.exhaustive(M)
        assert r["messages_reaching_zero"] == 0


def test_reduced_lift_exists_at_three_blocks_and_verifies():
    # N=2,p=7,T=2,blocks=3: at least one message reaches zero; verify it exactly
    M = rl.RModel(2, 7, 5, 3, 3, 2, 3)
    r = rl.exhaustive(M)
    assert r["messages_reaching_zero"] >= 1
    witness = np.array(r["zero_message_examples"][0], dtype=np.int64)
    S = rl.absorb(M, witness)
    assert np.array_equal(S, np.zeros(M.ncells, dtype=np.int64))   # exact lift


def test_reduced_message_collisions_exist():
    M = rl.RModel(2, 7, 5, 3, 3, 2, 3)
    r = rl.exhaustive(M)
    assert r["message_collisions"] >= 1
    a, b = r["collision_examples"][0]
    Sa = rl.absorb(M, np.array(a, dtype=np.int64))
    Sb = rl.absorb(M, np.array(b, dtype=np.int64))
    assert np.array_equal(Sa, Sb) and a != b      # distinct messages, equal state
