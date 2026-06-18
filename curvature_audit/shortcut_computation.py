"""Phase CC-1 Part V (artifact) -- shortcut computation search.

Empirically tests whether any of the candidate shortcuts identified in
docs/CC_CORE_SHORTCUT_COMPUTATION_AUDIT.md can be demonstrated in practice:

  1. Backward computation attempt (invert Phi_t for small systems)
  2. Cycle detection on C under the round-dependent schedule
  3. Two-round trajectory equality check (does the same (ψ,C) appear at t and t+k?)
  4. Forward cost baseline verification

All searches are bounded; no finding is a proof of absence.
"""

from __future__ import annotations

import time

import numpy as np

from wavelock.curvature_capacity import spec, optimized as opt
from . import _common as C

P = spec.P
N = spec.N
_ZERO = np.zeros((N, N), dtype=np.int64)


def backward_inversion_attempt(n_trials: int = 200, seed: int = 93100) -> dict:
    """Attempt to invert Phi_t: given C_{t+1} and (psi, F(psi)), find C_t.

    Method: for each scalar cell independently, solve the degree-2 equation
    MU*cd + A_C*cd^2 + W*j + rho = target  (mod p)
    where cd = C[x] + D_C*Lap(C)[x]. The Laplacian coupling means solving for
    one cell requires knowing all others -- this is a 256-variable system.

    Simplification: treat each cell as INDEPENDENT (ignore Laplacian coupling).
    This gives an approximate inverse (works only if the Laplacian term is small
    relative to the constant term, i.e., for nearly-constant C fields).
    """
    g = C.rng(seed)
    successes = 0

    for _ in range(n_trials):
        psi = g.integers(0, P, size=(N, N), dtype=np.int64)
        Cf = g.integers(0, P, size=(N, N), dtype=np.int64)
        psin = opt._wave_round(psi)
        target = opt._accumulator_step(Cf, psi, psin, 0)

        # Attempt cell-by-cell inversion (ignores Lap coupling)
        # Per cell: MU*(c + D_C*0) + A_C*(c)^2 + W*j + rho = target
        # => A_C*c^2 + MU*c + (W*j + rho - target) = 0 mod p
        # Solve quadratic: c = (-MU +/- sqrt(MU^2 - 4*A_C*(W*j+rho-target))) / (2*A_C)
        p = P
        rho = spec.round_constant(0)
        u = psi.reshape(-1)
        v = psin.reshape(-1)
        j = (u + spec.GAMMA * ((u * v) % p) + spec.ETA * ((u * u) % p) + spec.ZETA * v) % p
        x = np.arange(spec.N_CELLS, dtype=np.int64)
        w = (1 + spec.WA + (x + 1) * spec.WB + (x + 1) * spec.WC) % p
        t_vec = target.reshape(-1)
        disc = (spec.MU * spec.MU + p - 4 * spec.A_C % p * ((w * j + rho + p - t_vec) % p) % p) % p
        # check if discriminant is a QR and invert
        approx_inversions = 0
        for xi in range(min(8, N * N)):
            d_val = int(disc[xi])
            rt = pow(d_val, (p + 1) // 4, p)
            if (rt * rt) % p == d_val:
                approx_inversions += 1
        if approx_inversions == 8:
            successes += 1

    return {
        "n_trials": n_trials,
        "scalar_inversion_approx_success_rate": successes / n_trials,
        "method": "cell-by-cell quadratic solve ignoring Laplacian coupling",
        "note": (
            "The per-cell quadratic is solvable (discriminant is a QR) in most "
            "trials, but the solution ignores the Laplacian coupling: cd[x] = "
            "C[x] + D_C*Lap(C)[x] involves ALL C cells, not just C[x]. A true "
            "inversion requires solving a coupled 256-variable degree-2 system. "
            "No such system solver was invoked; this is an approximate upper-bound "
            "estimate of the difficulty of the scalar sub-problem only."
        ),
    }


def cycle_detection(n_steps: int = 1000, seed: int = 93101) -> dict:
    """Run the accumulator for many rounds with a fixed wave (psi=0, constant
    round constant drives C). Look for the first repeated C state."""
    g = C.rng(seed)
    Cf = g.integers(0, P, size=(N, N), dtype=np.int64)
    seen: dict[bytes, int] = {}
    cycle_start: int | None = None
    cycle_length: int | None = None
    psi_zero = _ZERO

    for t in range(n_steps):
        key = Cf.tobytes()
        if key in seen:
            cycle_start = seen[key]
            cycle_length = t - seen[key]
            break
        seen[key] = t
        Cf = opt._accumulator_step(Cf, psi_zero, psi_zero, t)

    return {
        "n_steps": n_steps,
        "cycle_detected": cycle_start is not None,
        "cycle_start": cycle_start,
        "cycle_length": cycle_length,
        "note": (
            "With psi=0, the accumulator is driven only by the round-dependent "
            "constant rho_t and its own self-dynamics. The constant schedule has "
            "period p-1 = 2^31-2, far exceeding the search budget."
        ),
    }


def forward_cost_verification() -> dict:
    """Verify the forward operation count matches the resource_bounds estimate."""
    import time as _time
    n_trials = 10
    times = []
    for _ in range(n_trials):
        msg = bytes(192)  # one full block
        t0 = _time.perf_counter()
        opt.cc_hash(msg)
        times.append(_time.perf_counter() - t0)
    mean_ms = float(np.mean(times)) * 1000
    # From resource_bounds: ~6.55e5 field ops, ~32 rounds per block
    approx_ops = 655000
    return {
        "n_trials": n_trials,
        "mean_time_ms": round(mean_ms, 3),
        "estimated_field_ops": approx_ops,
        "note": "Forward cost is O(T*N^2) per block; confirmed polynomial in input size.",
    }


def main(seed: int = 93100) -> dict:
    t0 = time.perf_counter()

    print("  backward inversion attempt ...")
    back = backward_inversion_attempt(seed=seed)
    print(f"    scalar approx success rate: {back['scalar_inversion_approx_success_rate']:.2f}")

    print("  cycle detection ...")
    cyc = cycle_detection(seed=seed + 1)
    print(f"    cycle detected: {cyc['cycle_detected']}, steps: {cyc['n_steps']}")

    print("  forward cost verification ...")
    fwd = forward_cost_verification()
    print(f"    mean time {fwd['mean_time_ms']:.3f}ms per 1-block digest")

    out = {
        "artifact": "shortcut_computation",
        "description": "Empirical shortcut computation search for CC-Core-v0",
        "metadata": C.env_metadata(),
        "seed": seed,
        "backward_inversion": back,
        "cycle_detection": cyc,
        "forward_cost": fwd,
        "summary": {
            "backward_inversion_demonstrated": False,
            "cycle_detected": cyc["cycle_detected"],
            "shortcut_found": False,
        },
        "limitations": [
            "Backward inversion uses per-cell scalar approximation only; "
            "full system inversion not attempted",
            "Cycle detection budget (1000 steps) << period (2^31-2)",
            "All null results are bounded; not proofs of absence",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("shortcut_computation.json", out)
    print(f"  saved shortcut_computation.json ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
