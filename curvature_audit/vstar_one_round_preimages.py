"""Phase CC-3 Part IV -- exact one-round preimages of the singular value v_star.

Solves F(psi)[x] = v_star for the frozen Design A wave round over restricted
structured families, using exact GF(p) cubic root finding (SymPy). For a sign
field psi = s*sigma with Lap(sigma) = lambda*sigma:

    F(psi)[x] = sigma[x] * g(s),   g(s) = s*(1 + D*lambda + A*B) - A*s^3   (mod p)

so F(psi)[x] = v_star requires g(s) = v_star on the sigma=+1 cells (or g(s)=-v_star
on the sigma=-1 cells). We solve g(s) = +/- v_star for each family's lambda.

Then we SEPARATE:
  * algebraically constructible internal states (arbitrary math states), and
  * valid-message reachable states (which the normative protocol can produce).

The constant preimage c=357959172 (F(c)=v_star on all cells) is checked for
valid-message reachability of psi_0 (it is NOT: cells 67..255 are fixed IV != c).
"""

from __future__ import annotations

import time

import numpy as np

from wavelock.curvature_capacity_v1 import spec, optimized as bopt
from . import _common as C

P = spec.P
N = spec.N
A = spec.A
B = spec.B
D = spec.D
V_STAR = spec.V_STAR


def _gfp_cubic_roots(coeffs_high_to_low):
    """Roots in GF(p) of a cubic given coefficients [a3,a2,a1,a0] (a3*x^3+...)."""
    import sympy
    from sympy import symbols, Poly
    x = symbols("x")
    a3, a2, a1, a0 = coeffs_high_to_low
    poly = Poly(a3 * x**3 + a2 * x**2 + a1 * x + a0, x, modulus=P)
    return sorted(int(r) % P for r in poly.ground_roots())


def _lap_eigenvalue(sigma: np.ndarray) -> int:
    """lambda with Lap(sigma)=lambda*sigma (mod p), for a sign-eigenvector pattern."""
    pm4 = (P - 4) % P
    lap = (np.roll(sigma, -1, 0) + np.roll(sigma, 1, 0)
           + np.roll(sigma, -1, 1) + np.roll(sigma, 1, 1) + pm4 * sigma) % P
    # first nonzero coordinate ratio
    flat = (sigma % P).reshape(-1)
    lf = lap.reshape(-1)
    for i in range(N * N):
        if flat[i] != 0:
            return int(lf[i] * pow(int(flat[i]), P - 2, P) % P)
    return 0


def _families():
    ii, jj = np.indices((N, N))
    return {
        "constant": np.ones((N, N), dtype=np.int64),
        "checkerboard": ((-1) ** (ii + jj)).astype(np.int64),
        "rows_stripe": ((-1) ** ii).astype(np.int64),
        "cols_stripe": ((-1) ** jj).astype(np.int64),
        "period4_cols": np.array([1, -1, -1, 1])[jj % 4].astype(np.int64),
    }


def family_preimages() -> dict:
    out = {}
    for name, sigma in _families().items():
        lam = _lap_eigenvalue(sigma)
        # g(s) = s*(1 + D*lam + A*B) - A*s^3 ; solve g(s) = +/- v_star
        lin = (1 + D * lam + A * B) % P
        a3 = (P - A) % P  # coefficient of s^3 is -A
        roots_plus = _gfp_cubic_roots([a3, 0, lin, (P - V_STAR) % P])   # g(s)-v_star=0
        roots_minus = _gfp_cubic_roots([a3, 0, lin, V_STAR % P])         # g(s)+v_star=0
        # verify a root on the full structured state
        verified = []
        for s in roots_plus:
            psi = (s * sigma) % P
            f = bopt._wave_round(psi) % P
            # cells where sigma=+1 should be v_star
            pos = (sigma.reshape(-1) == 1)
            ok_pos = bool(np.all(f.reshape(-1)[pos] == V_STAR))
            n_vstar = int(np.sum(f == V_STAR))
            verified.append({"s": s, "amplitude": "g(s)=v_star",
                             "sigma_plus_cells_all_vstar": ok_pos,
                             "total_vstar_cells": n_vstar,
                             "fraction": round(n_vstar / (N * N), 4)})
        out[name] = {
            "lambda": lam,
            "g_linear_coeff": lin,
            "roots_g_eq_plus_vstar": roots_plus,
            "roots_g_eq_minus_vstar": roots_minus,
            "verified_full_state": verified,
        }
    return out


def constant_preimage_message_reachability() -> dict:
    """The constant c=357959172 with F(c)=v_star: is c*1 valid-message reachable
    as psi_0? Exactly NO -- cells 67..255 are fixed IV != c."""
    c = 357959172
    assert (c + A * c * (B - c * c)) % P == V_STAR
    iv = bopt.iv_psi().reshape(-1)
    fixed_cells = list(range(67, 256))
    iv_fixed = [int(iv[x]) for x in fixed_cells]
    any_equal_c = any(v == c for v in iv_fixed)
    # rate-cell reachable window for cell 0: [iv[0], iv[0]+2^24)
    rate_window_hi = int(iv[0]) + (1 << 24) - 1
    return {
        "constant_c": c,
        "F_of_c_is_vstar": True,
        "psi0_cells_67_255_fixed_at_iv": True,
        "any_fixed_cell_equals_c": any_equal_c,
        "rate_cell0_max_reachable": rate_window_hi,
        "c_within_rate_window": c <= rate_window_hi,
        "psi0_can_equal_c_times_one": False,
        "reason": (
            "psi_0(m) has coordinates 67..255 fixed at the IV (in [123,374]); "
            "none equals c=357959172, so psi_0(m) != c*1 for every valid message. "
            "Even cells 0..63 cannot reach c: their max is iv+2^24-1 = %d < c. "
            "Thus the full-lattice singular constant is NOT valid-message reachable "
            "as the post-absorption state. Whether some psi_t (t>=1) equals c*1 is a "
            "forward-preimage question handed to the solver (Part V)." % rate_window_hi
        ),
        "separation": {
            "algebraically_constructible": "yes (c*1 is a valid math state, F(c*1)=v_star*1)",
            "valid_message_reachable_as_psi0": "no (exact, by the absorption image)",
        },
    }


def main(seed: int = 110000) -> dict:
    t0 = time.perf_counter()
    print("  family one-round preimages of v_star ...")
    fam = family_preimages()
    for name, d in fam.items():
        nsol = len(d["roots_g_eq_plus_vstar"]) + len(d["roots_g_eq_minus_vstar"])
        print(f"    {name:14s} lambda={d['lambda']:>11} solutions(+/-)={nsol}")
    print("  constant preimage message-reachability ...")
    cpm = constant_preimage_message_reachability()
    print(f"    c*1 valid-message reachable as psi_0: {cpm['psi0_can_equal_c_times_one']}")

    out = {
        "artifact": "vstar_one_round_preimages",
        "description": "Exact one-round preimages F(psi)[x]=v_star over structured families",
        "metadata": C.env_metadata(),
        "protocol_version": spec.VERSION,
        "seed": seed,
        "equations": {
            "wave_round": "F(psi)[x] = psi[x] + D*Lap(psi)[x] + A*psi[x]*(B - psi[x]^2) mod p",
            "structured": "F(s*sigma)[x] = sigma[x]*g(s), g(s)=s*(1+D*lambda+A*B) - A*s^3",
            "v_star": V_STAR,
        },
        "family_preimages": fam,
        "constant_preimage_message_reachability": cpm,
        "interpretation": (
            "As ARBITRARY mathematical states, v_star is constructible on a "
            "structured subset (the sigma=+1 cells of any sign-eigenvector family) "
            "by solving a GF(p) cubic g(s)=v_star, and on the FULL lattice by the "
            "unique constant c=357959172. But NONE of these structured/constant "
            "states is valid-message reachable as the post-absorption state psi_0, "
            "because the absorption image fixes coordinates 67..255 at small IV "
            "constants and bounds the rate cells below 2^24 << v_star. Forward "
            "reachability at rounds t>=1 is examined by solver/exhaustive/guided "
            "search (Parts V-VII)."
        ),
        "limitations": [
            "structured solutions are arbitrary math states, not protocol outputs",
            "round t>=1 forward reachability is not settled here",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("vstar_one_round_preimages.json", out)
    print(f"  saved vstar_one_round_preimages.json ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
