"""Phase CC-2 -- Candidate B (CC-Core-v1-B) regression and binding tests.

Covers: reference/optimized parity, pinned vectors, the exact j_B equation, the
singular value v_star, injectivity in u off v_star, collapse at v_star, full
47-state separation, the restricted-binding theorems (oddness of F, sign-pair
separation = 2u), path-order/symmetry, block/round domain separation, frozen
Design A guard, and version/domain separation from Candidate A.
"""

from __future__ import annotations

import pytest
import numpy as np

from wavelock.curvature_capacity_v1 import (spec as bspec, optimized as bopt,
                                            reference as bref)
from wavelock.curvature_capacity import optimized as aopt, spec as aspec
from . import _common as C

P = bspec.P
N = bspec.N
GAMMA = bspec.GAMMA
V_STAR = bspec.V_STAR
_ZERO = np.zeros((N, N), dtype=np.int64)

PINNED_B = {
    b"": "eb6402bb517d4d2ef409b6a1a16093cdae06462a548d240fe6783f0aec2216bd",
    b"\x00": "0ef50d4fecc6b7696a8bc0d06bcd73a1341bf9c854b19269b5a3c69994b29b84",
    b"\x01": "93815bbb3df1c3827c92e30daa96b97bffe18cc2b8e7a63f53f391de3da0110a",
    b"\xff": "3d0a0267f4223179c0a18f9a0a8dd5fa40d0583344d3a5ebdbe2dc529c1509c2",
    b"abc": "223797841b2e201aa2c3bfa40623306fbee14e420d5741b5d1a04309a4936922",
    b"WaveLock": "9e14e1cf32e2285be0f1d402e697d79171aa91138c8700897b72e330f4a708b9",
    b"\x00" * 192: "116d4e011f2aaad4bf31561db7fd45635d3b4ba4667345a2a91ec1ce46cd8264",
    b"\x00" * 193: "31785af8b1f9edcbb603109d2937753fd6946ce83e57424d5d1328d17899c9ff",
}


# --- parity & vectors (Part IV / discipline item 4) ---------------------
@pytest.mark.parametrize("msg", [b"", b"\x00", b"abc", b"WaveLock", b"\x00" * 200,
                                 bytes(range(50)), b"the quick brown fox"])
def test_reference_optimized_parity(msg):
    assert bref.cc_hash(msg) == bopt.cc_hash(msg)


@pytest.mark.parametrize("msg,hexd", list(PINNED_B.items()))
def test_pinned_vectors(msg, hexd):
    assert bopt.cc_hash(msg).hex() == hexd
    assert bref.cc_hash(msg).hex() == hexd


def test_wave_round_is_design_a():
    import importlib
    ev = importlib.import_module("wavelock.pde_hash.evolve")
    from wavelock.pde_hash.state import PDEState
    g = C.rng(11)
    for _ in range(10):
        st = g.integers(0, P, size=(N, N), dtype=np.int64)
        a = np.array(ev.evolve(PDEState([int(x) for x in st.reshape(-1)]), 1).cells,
                     dtype=np.int64).reshape(N, N) % P
        b = bopt._wave_round(st.copy()) % P
        assert np.array_equal(a, b)


# --- exact j_B equation & singular value (Part III/IV) ------------------
def test_jb_is_linear_injection():
    g = C.rng(12)
    for _ in range(200):
        u = int(g.integers(0, P)); v = int(g.integers(0, P))
        assert bopt._injection(np.int64(u), np.int64(v)) % P == (u * (1 + GAMMA * v)) % P


def test_jb_has_no_eta_or_zeta():
    assert bspec.ETA == 0 and bspec.ZETA == 0


def test_v_star_value():
    assert V_STAR == 195225786
    assert (1 + GAMMA * V_STAR) % P == 0


def test_collapse_at_v_star():
    g = C.rng(13)
    for _ in range(500):
        u = int(g.integers(0, P))
        assert (u + GAMMA * (u * V_STAR % P)) % P == 0


def test_injective_in_u_off_hyperplane():
    """For fixed v != v_star, j_B(.,v) is a bijection (slope nonzero)."""
    g = C.rng(14)
    for _ in range(200):
        v = int(g.integers(0, P))
        if v == V_STAR:
            continue
        slope = (1 + GAMMA * v) % P
        assert slope != 0
        a, b = int(g.integers(0, P)), int(g.integers(0, P))
        ja = (a * slope) % P
        jb = (b * slope) % P
        if a != b:
            assert ja != jb  # injective


# --- restricted binding theorems (Part X) -------------------------------
def test_wave_round_is_odd():
    """Lemma: F(-psi) = -F(psi)."""
    g = C.rng(15)
    for _ in range(30):
        psi = g.integers(0, P, size=(N, N), dtype=np.int64)
        f1 = bopt._wave_round(psi) % P
        f2 = bopt._wave_round((P - psi) % P) % P
        assert np.array_equal(f2, (P - f1) % P)


def test_sign_pair_injection_separation_is_2u():
    """Theorem 2: j_B(u,v) - j_B(-u,-v) = 2u, nonzero for nonzero u."""
    g = C.rng(16)
    for _ in range(200):
        u = int(g.integers(1, P)); v = int(g.integers(0, P))
        jp = (u + GAMMA * (u * v % P)) % P
        um = (P - u) % P; vm = (P - v) % P
        jm = (um + GAMMA * (um * vm % P)) % P
        assert (jp - jm) % P == (2 * u) % P
        assert (2 * u) % P != 0


# --- full 47-state separation (Part V) ----------------------------------
class TestFullFamilySeparation:
    _digs = None

    @classmethod
    def _get(cls):
        if cls._digs is None:
            from .phase_cc1_family import enumerate_full_family
            states, _ = enumerate_full_family()
            cls._digs = [bopt.trajectory_digest(
                np.array(s["cells"], dtype=np.int64).reshape(N, N)).hex()
                for s in states]
        return cls._digs

    def test_all_47_distinct(self):
        digs = self._get()
        assert len(digs) == 47
        assert len(set(digs)) == 47

    def test_min_hamming_at_least_64(self):
        digs = self._get()
        m = 256
        for i in range(len(digs)):
            for j in range(i + 1, len(digs)):
                m = min(m, C.hamming_bytes(bytes.fromhex(digs[i]), bytes.fromhex(digs[j])))
        assert m >= 64, f"min HD {m} < 64"


# --- path-order & symmetry ----------------------------------------------
def test_path_order_sensitivity():
    g = C.rng(17)
    blk0 = bytes(int(x) for x in g.integers(0, 256, size=192))
    blk1 = bytes(int(x) for x in g.integers(0, 256, size=192))
    assert bopt.cc_hash(blk0 + blk1) != bopt.cc_hash(blk1 + blk0)


def test_symmetry_does_not_preserve_digest():
    g = C.rng(18)
    base = g.integers(0, P, size=(N, N), dtype=np.int64)
    d0 = bopt.trajectory_digest(base).hex()
    for v in (np.roll(base, 1, 0), base[::-1, :].copy(), base.T.copy(),
              (P - base) % P):
        assert bopt.trajectory_digest(v).hex() != d0


# --- domain separation from Candidate A ---------------------------------
def test_distinct_d_tag_and_version():
    assert bspec.D_TAG != aspec.D_TAG
    assert bspec.VERSION != aspec.VERSION


def test_b_digests_differ_from_a_on_nonzero_messages():
    for m in [b"abc", b"WaveLock", b"\x01", b"\xff"]:
        assert bopt.cc_hash(m) != aopt.cc_hash(m)


def test_zero_trajectory_matches_a():
    """When psi=0, j_A=j_B=0, so the zero trajectory coincides (documented)."""
    assert bopt.trajectory_digest(_ZERO) == aopt.trajectory_digest(_ZERO)


# --- frozen Design A guard ----------------------------------------------
def test_design_a_frozen():
    from wavelock.pde_hash import optimized as a_opt
    assert a_opt.pde_hash(b"").hex() == \
        "d12c29be1429775e6dcc9ff3e29d9bca96865c0179a99b9bcee58581bf118820"


# --- singular hyperplane not path-erasing (Part IV regression) ----------
def test_full_collapse_state_no_erasure():
    """Perturbing the unique full-lattice singular state changes the digest."""
    c = 357959172  # F(c) = v_star on all cells
    base = np.full((N, N), c, dtype=np.int64)
    # confirm full-lattice collapse
    assert np.all(bopt._wave_round(base) % P == V_STAR)
    d_base = bopt.trajectory_digest(base).hex()
    g = C.rng(19)
    for _ in range(10):
        r0, c0 = int(g.integers(N)), int(g.integers(N))
        pert = base.copy()
        pert[r0, c0] = (pert[r0, c0] + int(g.integers(1, P))) % P
        assert bopt.trajectory_digest(pert).hex() != d_base
