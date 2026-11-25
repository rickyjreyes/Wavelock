"""Phase CC-2 Part VI -- controlled Candidate A vs Candidate B comparison.

Both candidates use identical messages, wave trajectories, round counts, lattice
size, seeds, squeeze width, and attack budgets. Only the injection differs
(A: j_A = u + GAMMA*u*v + ETA*u^2 + ZETA*v ; B: j_B = u + GAMMA*u*v).

Compares structural, empirical, and reduced-model properties. The decision is NOT
made on minimum Hamming distance alone (Part XV).
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from wavelock.curvature_capacity import optimized as aopt, spec as aspec
from wavelock.curvature_capacity_v1 import optimized as bopt, spec as bspec
from . import _common as C

P = aopt._P
N = aopt._N
_ZERO = np.zeros((N, N), dtype=np.int64)


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------
def structural() -> dict:
    a_eta_inv = pow(aspec.ETA, P - 2, P)
    return {
        "injection_multiplicity_in_u": {
            "A": "2-to-1 generically (proved: u' = -(1+GAMMA*v)/ETA - u)",
            "B": "1-to-1 (injective) for all v with (1+GAMMA*v) != 0; collapse at v=v_star",
        },
        "injection_degree_in_u": {"A": 2, "B": 1},
        "injection_zero_set": {
            "A": "parabola eta*u^2 + (1+gamma*v)u + zeta*v = 0 (a curve, ~p points)",
            "B": "{u=0} U {v=v_star} (2p-1 points)",
        },
        "singular_hyperplane": {
            "A": "none distinguished (2-to-1 fold everywhere)",
            "B": "v = v_star = %d (measure 1/p), constructible but not path-erasing" % bspec.V_STAR,
        },
        "coordinate_erasure_condition": {
            "A": "none (every (u,v) injects; fold is 2-to-1 not erasing)",
            "B": "v == v_star erases u at that coordinate that round (measure 1/p)",
        },
        "a_eta_inverse": a_eta_inv,
    }


def fixed_points_and_cycles(budget: int = 3000, seed: int = 97001) -> dict:
    """Bounded accumulator fixed-point and two-cycle search, identical budget."""
    def search(opt, seed):
        g = C.rng(seed)
        fp = 0
        tc = 0
        for _ in range(budget):
            psi = g.integers(0, P, size=(N, N), dtype=np.int64)
            psin = opt._wave_round(psi)
            Cf = g.integers(0, P, size=(N, N), dtype=np.int64)
            nxt = opt._accumulator_step(Cf, psi, psin, 0)
            if np.array_equal(nxt % P, Cf % P):
                fp += 1
            psin2 = opt._wave_round(psin)
            C1 = opt._accumulator_step(Cf, psi, psin, 0)
            C2 = opt._accumulator_step(C1, psin, psin2, 1)
            if np.array_equal(C2 % P, Cf % P):
                tc += 1
        return {"fixed_points_found": fp, "two_cycles_found": tc, "budget": budget}
    return {"A": search(aopt, seed), "B": search(bopt, seed)}


# ---------------------------------------------------------------------------
# Empirical (identical seeds / messages)
# ---------------------------------------------------------------------------
def avalanche(n_msgs: int = 120, seed: int = 97002) -> dict:
    g = C.rng(seed)
    a_hd, b_hd = [], []
    for _ in range(n_msgs):
        psi = g.integers(0, P, size=(N, N), dtype=np.int64)
        r0, c0 = int(g.integers(N)), int(g.integers(N))
        psi2 = psi.copy()
        psi2[r0, c0] = (psi2[r0, c0] + 1) % P
        a_hd.append(C.hamming_bytes(aopt.trajectory_digest(psi),
                                    aopt.trajectory_digest(psi2)))
        b_hd.append(C.hamming_bytes(bopt.trajectory_digest(psi),
                                    bopt.trajectory_digest(psi2)))
    return {
        "n_msgs": n_msgs,
        "A": {"mean": round(float(np.mean(a_hd)), 2), "min": int(min(a_hd)), "max": int(max(a_hd))},
        "B": {"mean": round(float(np.mean(b_hd)), 2), "min": int(min(b_hd)), "max": int(max(b_hd))},
        "expected_random": 128,
    }


def monobit_bias(n_msgs: int = 1000, seed: int = 97003) -> dict:
    g = C.rng(seed)
    def bias(opt):
        ones = np.zeros(256, dtype=np.int64)
        msgs = C.random_messages(seed, n_msgs, 1, 64)
        for m in msgs:
            d = opt.cc_hash(m)
            bits = np.unpackbits(np.frombuffer(d, dtype=np.uint8))
            ones += bits
        exp = n_msgs / 2
        z = (ones - exp) / np.sqrt(n_msgs / 4)
        return {"max_abs_z": round(float(np.max(np.abs(z))), 3),
                "bits_z_gt_3": int(np.sum(np.abs(z) > 3))}
    return {"n_msgs": n_msgs, "A": bias(aopt), "B": bias(bopt)}


def truncated_collision(nbits: int = 24, budget: int = 4000, seed: int = 97004) -> dict:
    def search(opt):
        seen = {}
        g = C.rng(seed)
        for e in range(budget):
            m = int(e).to_bytes(6, "big")
            d = opt.cc_hash(m)
            key = int.from_bytes(d, "big") >> (256 - nbits)
            if key in seen:
                return {"collision_at_eval": e, "expected_birthday": round((2 ** (nbits / 2)))}
            seen[key] = m
        return {"collision_at_eval": None, "budget": budget,
                "expected_birthday": round((2 ** (nbits / 2)))}
    return {"nbits": nbits, "A": search(aopt), "B": search(bopt)}


def path_order_sensitivity(n_msgs: int = 80, seed: int = 97005) -> dict:
    """Order sensitivity: hash a 2-block message vs its block-swapped variant."""
    g = C.rng(seed)
    a_diff = b_diff = 0
    for _ in range(n_msgs):
        blk0 = bytes(int(x) for x in g.integers(0, 256, size=192))
        blk1 = bytes(int(x) for x in g.integers(0, 256, size=192))
        m = blk0 + blk1
        m_swap = blk1 + blk0
        if aopt.cc_hash(m) != aopt.cc_hash(m_swap):
            a_diff += 1
        if bopt.cc_hash(m) != bopt.cc_hash(m_swap):
            b_diff += 1
    return {
        "n_msgs": n_msgs,
        "A_order_sensitive": a_diff,
        "B_order_sensitive": b_diff,
        "note": "fraction of block-swapped pairs giving distinct digests; both "
                "should be ~all (order is bound by rho_t and W_t).",
    }


def runtime_memory(n: int = 20, seed: int = 97006) -> dict:
    import time as _t
    msg = bytes(192)
    ta = tb = 0.0
    for _ in range(n):
        t0 = _t.perf_counter(); aopt.cc_hash(msg); ta += _t.perf_counter() - t0
        t0 = _t.perf_counter(); bopt.cc_hash(msg); tb += _t.perf_counter() - t0
    return {
        "n": n,
        "A_mean_ms": round(ta / n * 1000, 3),
        "B_mean_ms": round(tb / n * 1000, 3),
        "memory_note": "both carry two 256-cell int64 fields (~4 KiB state); identical.",
    }


# ---------------------------------------------------------------------------
# Reduced-model (toy coupled round image / multiplicity)
# ---------------------------------------------------------------------------
def reduced_model(primes=(3, 5, 7)) -> dict:
    """Compare coupled-round image size and max preimage multiplicity on a 2x2
    toy for both injections (full joint (psi,C) at p=3; psi-only slice at larger)."""
    out = {}
    for p in primes:
        n = 4  # 2x2
        if p ** (2 * n) > 700_000:
            # psi-slice only (fix C)
            mode = "psi_slice_fixed_C"
            a_img, a_max = _toy_slice(p, n, "A")
            b_img, b_max = _toy_slice(p, n, "B")
        else:
            mode = "full_joint"
            a_img, a_max = _toy_joint(p, n, "A")
            b_img, b_max = _toy_joint(p, n, "B")
        out[str(p)] = {
            "mode": mode,
            "A": {"image_size": a_img, "max_preimage_multiplicity": a_max},
            "B": {"image_size": b_img, "max_preimage_multiplicity": b_max},
        }
    return out


def _toy_constants(p):
    return dict(pm4=(p - 4) % p, D=aspec.D % p, A=aspec.A % p, B=aspec.B % p,
                D_C=aspec.D_C % p, GAMMA=aspec.GAMMA % p, ETA=aspec.ETA % p,
                ZETA=aspec.ZETA % p, A_C=aspec.A_C % p, MU=aspec.MU % p,
                RHO0=aspec.RHO0 % p, RHO1=aspec.RHO1 % p,
                WA=aspec.WA % p, WB=aspec.WB % p, WC=aspec.WC % p)


def _toy_round(psi, Cf, p, k, cand):
    n0 = 2
    pm4 = (p - 4) % p
    def lap(x):
        return (np.roll(x, -1, 0) + np.roll(x, 1, 0) + np.roll(x, -1, 1)
                + np.roll(x, 1, 1) + pm4 * x) % p
    sq = (psi * psi) % p
    bm = (aspec.B % p + p - sq) % p
    psin = (psi + (aspec.D % p) * lap(psi) % p + (aspec.A % p) * ((psi * bm) % p) % p) % p
    u = psi.reshape(-1); v = psin.reshape(-1)
    if cand == "A":
        j = (u + (aspec.GAMMA % p) * ((u * v) % p) + (aspec.ETA % p) * ((u * u) % p)
             + (aspec.ZETA % p) * v) % p
    else:
        j = (u + (aspec.GAMMA % p) * ((u * v) % p)) % p
    cd = (Cf.reshape(-1) + (aspec.D_C % p) * lap(Cf).reshape(-1) % p) % p
    idx = np.arange(4)
    w = (1 + (k + 1) * (aspec.WA % p) + (idx + 1) * (aspec.WB % p)
         + (k + 1) * (idx + 1) * (aspec.WC % p)) % p
    rho = (aspec.RHO0 % p + (aspec.RHO1 % p) * k) % p
    Cn = ((aspec.MU % p) * cd + (aspec.A_C % p) * ((cd * cd) % p) % p
          + (w * j) % p + rho) % p
    return psin.reshape(2, 2), Cn.reshape(2, 2)


def _toy_joint(p, n, cand):
    images = {}
    for ptup in product(range(p), repeat=n):
        for ctup in product(range(p), repeat=n):
            psi = np.array(ptup, dtype=np.int64).reshape(2, 2)
            Cf = np.array(ctup, dtype=np.int64).reshape(2, 2)
            psin, Cn = _toy_round(psi, Cf, p, 0, cand)
            key = psin.tobytes() + Cn.tobytes()
            images[key] = images.get(key, 0) + 1
    return len(images), max(images.values())


def _toy_slice(p, n, cand):
    C0 = np.ones((2, 2), dtype=np.int64)
    images = {}
    for ptup in product(range(p), repeat=n):
        psi = np.array(ptup, dtype=np.int64).reshape(2, 2)
        psin, Cn = _toy_round(psi, C0, p, 0, cand)
        key = psin.tobytes() + Cn.tobytes()
        images[key] = images.get(key, 0) + 1
    return len(images), max(images.values())


def main(seed: int = 97000) -> dict:
    t0 = time.perf_counter()
    print("  structural ...")
    st = structural()
    print("  fixed points / cycles ...")
    fpc = fixed_points_and_cycles()
    print("  avalanche ...")
    av = avalanche()
    print(f"    A mean {av['A']['mean']}, B mean {av['B']['mean']}")
    print("  monobit bias ...")
    mb = monobit_bias()
    print("  truncated collision ...")
    tc = truncated_collision()
    print("  path-order sensitivity ...")
    po = path_order_sensitivity()
    print("  runtime/memory ...")
    rt = runtime_memory()
    print("  reduced model ...")
    rm = reduced_model()

    out = {
        "artifact": "candidate_a_vs_b",
        "description": "Controlled Candidate A vs Candidate B comparison (identical seeds/budgets)",
        "metadata": C.env_metadata(),
        "seed": seed,
        "structural": st,
        "fixed_points_and_cycles": fpc,
        "empirical": {
            "full_family_separation": {
                "A": {"distinct": 47, "min_hamming": 98},
                "B": {"distinct": 47, "min_hamming": 105},
                "note": "from phase8j family binding artifacts; both separate all 47.",
            },
            "avalanche": av,
            "monobit_bias": mb,
            "truncated_collision": tc,
            "path_order_sensitivity": po,
            "runtime_memory": rt,
        },
        "reduced_model": rm,
        "summary": {
            "B_removes_generic_2to1": True,
            "B_introduces_singular_hyperplane": True,
            "singular_hyperplane_reachable": False,
            "singular_hyperplane_path_erasing": False,
            "B_min_hamming_higher": True,
            "decision_note": "Decision is made in Part XV across all axes, not on "
                             "minimum Hamming distance alone.",
        },
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("candidate_a_vs_b.json", out)
    print(f"  saved candidate_a_vs_b.json ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
