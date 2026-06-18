"""Phase CC-1 Part VIII -- exhaustive reduced model: coupled (psi, C) analysis.

Extends the Phase 1 reduced_models.py with a full joint (psi, C) enumeration
at very small primes. Previous analysis fixed C (a slice); this module
enumerates BOTH psi and C exhaustively (feasible only at p=3, N0=2: state
space p^(2*N0^2) = 3^8 = 6561 for each field, total 6561^2 ~ 43M -- too large
for joint enumeration). Instead we use:

  1. Full (psi, C) enumeration at p=3, N0=1 (scalar: 3*3=9 states total --
     trivial but shows algebraic structure clearly).
  2. At p=5, N0=2: enumerate all psi (5^4=625); for each psi, enumerate all
     C (5^4=625); check coupled round injectivity on the FULL joint domain.
     Total: 625*625 = ~390K pairs -- feasible.
  3. At p=7, N0=2: enumerate all psi (7^4=2401) x C (2401) = 5.7M -- borderline.
     Only do psi enumeration with sampled C slices.

Reports:
  - Whether the wave round alone is injective (it is NOT for the toy Design A)
  - Whether the coupled round is injective on the full (psi, C) domain
  - How many coupled collisions exist
  - Whether any trajectory (multi-round) collision exists in the toy
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from wavelock.curvature_capacity import spec
from . import _common as C


class TinyCoupled:
    """A miniature coupled core over F_p on an N0 x N0 torus.

    Uses the exact same algebraic shape as CC-Core-v0 but with tiny parameters.
    """

    def __init__(self, N0: int = 2, p: int = 5):
        self.N0 = N0
        self.p = p
        self.n = N0 * N0
        pm4 = (p - 4) % p
        self.pm4 = pm4
        # Use spec constants reduced mod p
        self.D = spec.D % p
        self.a = spec.A % p
        self.b = spec.B % p
        self.D_C = spec.D_C % p
        self.GAMMA = spec.GAMMA % p
        self.ETA = spec.ETA % p
        self.ZETA = spec.ZETA % p
        self.A_C = spec.A_C % p
        self.MU = spec.MU % p
        self.RHO0 = spec.RHO0 % p
        self.RHO1 = spec.RHO1 % p
        self.WA = spec.WA % p
        self.WB = spec.WB % p
        self.WC = spec.WC % p

    def _lap(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        return (
            np.roll(x, -1, 0) + np.roll(x, 1, 0)
            + np.roll(x, -1, 1) + np.roll(x, 1, 1)
            + self.pm4 * x
        ) % p

    def wave_round(self, psi: np.ndarray) -> np.ndarray:
        p = self.p
        sq = (psi * psi) % p
        bm = (self.b + p - sq) % p
        react = (self.a * ((psi * bm) % p)) % p
        lap = self._lap(psi)
        return (psi + (self.D * lap) % p + react) % p

    def _weights(self, t: int) -> np.ndarray:
        p = self.p
        idx = np.arange(self.n, dtype=np.int64)
        return (1 + (t + 1) * self.WA + (idx + 1) * self.WB
                + (t + 1) * (idx + 1) * self.WC) % p

    def acc_round(self, Cf: np.ndarray, psi: np.ndarray,
                  psin: np.ndarray, t: int) -> np.ndarray:
        p = self.p
        u = psi.reshape(-1)
        v = psin.reshape(-1)
        j = (u + self.GAMMA * ((u * v) % p) + self.ETA * ((u * u) % p) + self.ZETA * v) % p
        Cd_lap = self._lap(Cf).reshape(-1)
        cd = (Cf.reshape(-1) + (self.D_C * Cd_lap) % p) % p
        rho = (self.RHO0 + self.RHO1 * t) % p
        w = self._weights(t)
        out = (self.MU * cd + (self.A_C * ((cd * cd) % p)) % p + (w * j) % p + rho) % p
        return out.reshape(self.N0, self.N0)

    def coupled_round(self, psi: np.ndarray, Cf: np.ndarray,
                      t: int = 0) -> tuple[np.ndarray, np.ndarray]:
        psin = self.wave_round(psi)
        Cn = self.acc_round(Cf, psi, psin, t)
        return psin, Cn


def _enumerate_all_states(p: int, n: int):
    """Yield all p^n states as (n,)-shaped int64 arrays."""
    for tup in product(range(p), repeat=n):
        yield np.array(tup, dtype=np.int64)


def full_joint_enumeration(toy: TinyCoupled) -> dict:
    """Fully enumerate the (psi, C) joint state space; check coupled injectivity."""
    p = toy.p
    n = toy.n
    domain_size = p ** n

    if domain_size > 700:
        return {
            "enumerated": False,
            "reason": f"Domain size {domain_size}^2 = {domain_size**2} too large for full enumeration",
            "domain_size_psi": domain_size,
        }

    wave_images: dict[bytes, int] = {}
    wave_collisions = 0
    coupled_images: dict[bytes, bytes] = {}
    coupled_collisions = 0

    for psi_tup in product(range(p), repeat=n):
        psi = np.array(psi_tup, dtype=np.int64).reshape(toy.N0, toy.N0)
        psin = toy.wave_round(psi)
        wkey = psin.tobytes()
        wave_images[wkey] = wave_images.get(wkey, 0) + 1
        if wave_images[wkey] == 2:
            wave_collisions += 1

        for C_tup in product(range(p), repeat=n):
            Cf = np.array(C_tup, dtype=np.int64).reshape(toy.N0, toy.N0)
            psin_c, Cn = toy.coupled_round(psi, Cf, 0)
            joint_key = (psi_tup, C_tup)
            img_key = psin_c.tobytes() + Cn.tobytes()
            if img_key in coupled_images and coupled_images[img_key] != joint_key:
                coupled_collisions += 1
            else:
                coupled_images[img_key] = joint_key

    return {
        "enumerated": True,
        "p": p,
        "N0": toy.N0,
        "domain_psi": domain_size,
        "domain_joint": domain_size ** 2,
        "wave_collisions": wave_collisions,
        "wave_image_size": len(wave_images),
        "coupled_collisions_on_full_domain": coupled_collisions,
        "coupled_image_size": len(coupled_images),
        "coupled_injective": coupled_collisions == 0,
        "note": (
            "Full joint (psi, C) enumeration at round t=0. "
            "Coupled injectivity means every (psi, C) pair maps to a distinct "
            "(psi', C') pair."
        ),
    }


def multi_round_trajectory_collisions(toy: TinyCoupled, n_rounds: int = 3) -> dict:
    """Check whether two distinct initial (psi, C) pairs produce the same
    trajectory (psi_t, C_t) at the end of n_rounds rounds."""
    p = toy.p
    n = toy.n
    domain = p ** n
    if domain > 150:
        return {"enumerated": False, "reason": "domain too large"}

    trajectories: dict[bytes, tuple] = {}
    trajectory_collisions = 0

    for psi_tup in product(range(p), repeat=n):
        for C_tup in product(range(p), repeat=n):
            psi = np.array(psi_tup, dtype=np.int64).reshape(toy.N0, toy.N0)
            Cf = np.array(C_tup, dtype=np.int64).reshape(toy.N0, toy.N0)
            for t in range(n_rounds):
                psi, Cf = toy.coupled_round(psi, Cf, t)
            key = psi.tobytes() + Cf.tobytes()
            init = (psi_tup, C_tup)
            if key in trajectories and trajectories[key] != init:
                trajectory_collisions += 1
            else:
                trajectories[key] = init

    return {
        "enumerated": True,
        "p": p,
        "N0": toy.N0,
        "n_rounds": n_rounds,
        "domain_joint": domain ** 2,
        "trajectory_collisions": trajectory_collisions,
        "trajectory_image_size": len(trajectories),
        "injective_after_rounds": trajectory_collisions == 0,
    }


def main(seed: int = 93000) -> dict:
    t0 = time.perf_counter()
    results = {}

    # p=3, N0=2: domain_psi = 3^4 = 81; joint = 81^2 = 6561 -- feasible
    print("  p=3, N0=2: full joint enumeration ...")
    toy = TinyCoupled(N0=2, p=3)
    r1 = full_joint_enumeration(toy)
    print(f"    wave_coll={r1.get('wave_collisions')}, "
          f"coupled_coll={r1.get('coupled_collisions_on_full_domain')}, "
          f"injective={r1.get('coupled_injective')}")
    results["p3_N2"] = {
        "full_joint_enumeration": r1,
        "multi_round": multi_round_trajectory_collisions(toy, n_rounds=3),
    }

    # p=5, N0=2: domain_psi = 5^4 = 625; joint = 625^2 = 390625 -- do only joint single round
    print("  p=5, N0=2: full joint enumeration ...")
    toy5 = TinyCoupled(N0=2, p=5)
    r2 = full_joint_enumeration(toy5)
    print(f"    wave_coll={r2.get('wave_collisions')}, "
          f"coupled_coll={r2.get('coupled_collisions_on_full_domain')}, "
          f"injective={r2.get('coupled_injective')}")
    results["p5_N2"] = {"full_joint_enumeration": r2}

    # p=7, N0=2: domain 7^4=2401, joint 2401^2 -- too large for full joint, skip
    toy7 = TinyCoupled(N0=2, p=7)
    r3 = full_joint_enumeration(toy7)
    print(f"  p=7, N0=2: enumerable={r3.get('enumerated')} ({r3.get('reason', '')})")
    results["p7_N2"] = {"full_joint_enumeration": r3}

    out = {
        "artifact": "reduced_exhaustive_cc1",
        "description": "Full joint (psi,C) exhaustive enumeration of toy CC-Core-v0 models",
        "metadata": C.env_metadata(),
        "seed": seed,
        "equations": {
            "coupled_round": "(psi', C') = (F(psi), Phi_t(C, psi, F(psi)))",
            "coupled_injective": "True iff all (psi,C) map to distinct (psi',C')",
        },
        "toy_results": results,
        "interpretation": (
            "At p=3, N0=2 the full joint (psi,C) state space is 3^4 x 3^4 = 6561. "
            "At p=5, N0=2 it is 5^4 x 5^4 = 390,625. "
            "If coupled_injective is True, the coupled round is injective on the "
            "full joint domain (not just a wave slice). "
            "N=2 degeneracy (up==down, left==right on a 2x2 torus) dominates and "
            "results are NOT extrapolated to N=16."
        ),
        "limitations": [
            "N0=2 torus has extreme neighbor degeneracy (every cell's up==down, left==right)",
            "Results at N=16 remain unresolved",
            "Coupled round is tested at t=0 only for the single-round result",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("reduced_exhaustive_cc1.json", out)
    print(f"  saved reduced_exhaustive_cc1.json ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
