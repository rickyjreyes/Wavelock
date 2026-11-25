"""Phase 8E: symmetry and structured-collision attacks.

The 5-point toroidal Laplacian plus a pointwise reaction means the core
state-transformation evolve_T is *equivariant* under the lattice symmetry group
(translations, 90-degree rotations, reflections, transpose). This module:

  1. verifies that equivariance directly on random states (a real structural
     property of the transform);
  2. tests whether it survives end-to-end (it should be broken by the
     position-dependent IV, the fixed-cell counter/finalization injections, and
     the fixed squeeze-pair table) by searching for exact / internal-state /
     truncated / reduced-round collisions and shifted/reflected relations;
  3. tests message-level structure: block swaps, block repeats, and additive
     modular structure.

Collision results are separated into: full-256-bit, internal-state,
reduced-round, truncated-output, and distinguishable-but-non-colliding.
Operates on raw state / raw output only.

Run:
    python -m pde_audit.symmetry_attacks
"""

from __future__ import annotations

import time

import numpy as np

from . import _harness as H
from ._harness import PDEVariant, PDEParams

V = PDEVariant(PDEParams())


# ---------------------------------------------------------------------------
# 1. core equivariance of evolve_T
# ---------------------------------------------------------------------------
def _equivariance(seed: int, n: int = 20) -> dict:
    g = H.rng(seed)
    transforms = {
        "translate_(1,0)": lambda s: np.roll(s, 1, 0),
        "translate_(0,1)": lambda s: np.roll(s, 0 + 1, 1),
        "translate_(3,5)": lambda s: np.roll(np.roll(s, 3, 0), 5, 1),
        "rot90": lambda s: np.rot90(s, 1),
        "rot180": lambda s: np.rot90(s, 2),
        "flip_rows": lambda s: s[::-1, :].copy(),
        "flip_cols": lambda s: s[:, ::-1].copy(),
        "transpose": lambda s: s.T.copy(),
    }
    out = {}
    for name, tf in transforms.items():
        ok = True
        for _ in range(n):
            s = g.integers(0, V.P.p, size=(V.P.N, V.P.N), dtype=np.int64)
            lhs = V.evolve_T(tf(s))
            rhs = tf(V.evolve_T(s))
            if not np.array_equal(lhs, rhs):
                ok = False
                break
        out[name] = bool(ok)
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _digest(m: bytes) -> bytes:
    return V.hash(m)


def _trunc_hits(digests, nbits):
    """Count truncated-output collisions among a list of digests (first nbits)."""
    seen = {}
    hits = 0
    mask_bytes = (nbits + 7) // 8
    for d in digests:
        key = bytes(d[:mask_bytes])
        if nbits % 8:
            # mask the last partial byte
            last = key[-1] & (0xFF << (8 - nbits % 8) & 0xFF)
            key = key[:-1] + bytes([last])
        if key in seen:
            hits += 1
        else:
            seen[key] = True
    return hits


# ---------------------------------------------------------------------------
# 2 + 3. end-to-end structured tests
# ---------------------------------------------------------------------------
def _block_swap_repeat(seed: int, n: int = 400) -> dict:
    g = H.rng(seed)
    blk = 192
    swap_hd, repeat_hd = [], []
    full_collisions = 0
    state_collisions = 0
    for _ in range(n):
        b0 = g.integers(0, 256, size=blk, dtype=np.uint8).tobytes()
        b1 = g.integers(0, 256, size=blk, dtype=np.uint8).tobytes()
        d01 = _digest(b0 + b1)
        d10 = _digest(b1 + b0)
        swap_hd.append(H.hamming_bytes(d01, d10))
        if d01 == d10:
            full_collisions += 1
        s01 = V.absorb(b0 + b1).tobytes()
        s10 = V.absorb(b1 + b0).tobytes()
        if s01 == s10:
            state_collisions += 1
        # repeat a block vs single
        dsingle = _digest(b0)
        ddouble = _digest(b0 + b0)
        repeat_hd.append(H.hamming_bytes(dsingle, ddouble))
    return {
        "n": n,
        "swap_full_collisions": full_collisions,
        "swap_state_collisions": state_collisions,
        "swap_output_hd_mean": float(np.mean(swap_hd)),
        "swap_output_hd_min": int(np.min(swap_hd)),
        "repeat_output_hd_mean": float(np.mean(repeat_hd)),
        "repeat_output_hd_min": int(np.min(repeat_hd)),
    }


def _geometric_message_relations(seed: int, n: int = 300) -> dict:
    """Build messages whose 64 packed rate-elements of a single block are a
    cyclic shift / reversal of each other, and check for state/digest relations.

    A single 192-byte block packs to 64 rate elements; permuting those elements
    is a message-level operation. If absorption were symmetric, related messages
    would yield related states.
    """
    g = H.rng(seed)
    full_coll = 0
    state_coll = 0
    shifted_state_match = 0
    out_hd = []
    for _ in range(n):
        block = g.integers(0, 256, size=192, dtype=np.uint8)
        elems = block.reshape(64, 3)
        # cyclic shift the 64 element-triples by 1
        shifted = np.roll(elems, 1, axis=0).reshape(-1).astype(np.uint8).tobytes()
        base = block.tobytes()
        db = _digest(base)
        ds = _digest(shifted)
        out_hd.append(H.hamming_bytes(db, ds))
        if db == ds:
            full_coll += 1
        sb = V.absorb(base)
        ss = V.absorb(shifted)
        if np.array_equal(sb, ss):
            state_coll += 1
        # is the shifted-message state a lattice translate of the base state?
        if any(np.array_equal(ss, np.roll(np.roll(sb, di, 0), dj, 1))
               for di in range(V.P.N) for dj in range(0, V.P.N, 4)):
            shifted_state_match += 1
    return {
        "n": n,
        "full_collisions": full_coll,
        "state_collisions": state_coll,
        "shifted_state_matches": shifted_state_match,
        "output_hd_mean": float(np.mean(out_hd)),
        "output_hd_min": int(np.min(out_hd)),
    }


def _modular_additive(seed: int, n: int = 200) -> dict:
    """Packed elements are < 2**24 < p, so no two distinct packed values alias
    mod p within one block. Confirm additive cancellation is impossible at the
    injection step by checking that distinct single-byte changes never produce
    equal post-absorption states."""
    g = H.rng(seed)
    coll = 0
    for _ in range(n):
        m = g.integers(0, 256, size=64, dtype=np.uint8)
        s0 = V.absorb(m.tobytes())
        m2 = m.copy()
        pos = int(g.integers(0, 64))
        m2[pos] = (int(m2[pos]) + 1) % 256
        s1 = V.absorb(m2.tobytes())
        if np.array_equal(s0, s1):
            coll += 1
    return {"n": n, "single_byte_state_collisions": coll,
            "note": "packed elements < 2**24 < p; no modular aliasing at injection"}


def _symmetry_collision_corpus(seed: int, n: int = 2000) -> dict:
    """Apply lattice-motivated message transforms to a corpus and look for
    truncated-output collisions across the transformed set."""
    g = H.rng(seed)
    digests = []
    for _ in range(n):
        block = g.integers(0, 256, size=192, dtype=np.uint8)
        digests.append(_digest(block.tobytes()))
        digests.append(_digest(np.roll(block.reshape(64, 3), 1, 0).reshape(-1).astype(np.uint8).tobytes()))
        digests.append(_digest(block.reshape(64, 3)[::-1].reshape(-1).astype(np.uint8).tobytes()))
    return {
        "n_digests": len(digests),
        "trunc_collisions": {str(b): _trunc_hits(digests, b) for b in (8, 16, 24, 32)},
    }


def main(seed: int = 80050) -> dict:
    t0 = time.perf_counter()
    eq = _equivariance(seed)
    print("  evolve_T equivariance:", {k: v for k, v in eq.items()})
    swap = _block_swap_repeat(seed + 1)
    print(f"  block swap: full_coll={swap['swap_full_collisions']} "
          f"state_coll={swap['swap_state_collisions']} hd_mean={swap['swap_output_hd_mean']:.1f}")
    geo = _geometric_message_relations(seed + 2)
    print(f"  geometric: full_coll={geo['full_collisions']} state_coll={geo['state_collisions']} "
          f"shifted_match={geo['shifted_state_matches']} hd_mean={geo['output_hd_mean']:.1f}")
    moda = _modular_additive(seed + 3)
    corpus = _symmetry_collision_corpus(seed + 4)
    print(f"  modular: byte_state_coll={moda['single_byte_state_collisions']}  "
          f"corpus trunc_coll={corpus['trunc_collisions']}")

    results = {
        "phase": "8E_symmetry",
        "metadata": H.env_metadata(),
        "seed": seed,
        "evolve_T_equivariance": eq,
        "equivariance_note":
            "evolve_T commutes with torus translations/rotations/reflections/"
            "transpose. This symmetry is BROKEN end-to-end by the position-"
            "dependent IV, the fixed-cell counter/finalization injections, and "
            "the fixed squeeze-pair table; the tests below check that no usable "
            "collision or relation survives.",
        "block_swap_repeat": swap,
        "geometric_message_relations": geo,
        "modular_additive": moda,
        "symmetry_collision_corpus": corpus,
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    path = H.save_artifact("phase8e_symmetry.json", results)
    print("  saved", path, f"({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
