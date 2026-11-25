"""Phase 8K (Parts VII-VIII): reduced-model lifting + SMT.

Builds small systems with the SAME architecture as WaveLock-PDE-256-v0 (toroidal
Laplacian + cubic reaction over F_p, rate/capacity split, additive block
injection, per-block counter, finalization) and asks, exhaustively or with z3,
whether a valid message can drive the pre-squeeze state to zero or produce a
message-level (pre-squeeze) collision -- i.e. whether internal non-injectivity
LIFTS to the message layer in the small case, and how many blocks it needs.

Reduced rate/capacity split: rate = first ceil(N*N/4) cells (mirrors the 64/256
= 1/4 ratio of the normative system); capacity = the rest, with cap0/cap1/cap2
the first three capacity cells (counter-lo / finalization / counter-hi). For
tiny p, byte packing is impractical, so message symbols are arbitrary F_p
elements (the faithful reduced analog of the relaxed control model); this is
stated explicitly and not conflated with the normative byte model.

Reduced-model success is NOT extrapolated to the normative system; it is used to
identify lifting mechanisms.

Run:
    python -m pde_audit.reduced_lifting
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product

import numpy as np

from . import _harness as H

try:
    import z3
    _Z3 = True
except Exception:  # pragma: no cover
    _Z3 = False


@dataclass
class RModel:
    N: int
    p: int
    D: int
    a: int
    b: int
    T: int
    blocks: int

    @property
    def ncells(self):
        return self.N * self.N

    @property
    def rate(self):
        return max(1, (self.ncells + 3) // 4)

    @property
    def cap0(self):
        return self.rate

    @property
    def cap1(self):
        return min(self.rate + 1, self.ncells - 1)

    @property
    def cap2(self):
        return min(self.rate + 2, self.ncells - 1)


def iv(M: RModel):
    return np.array([(1 + c) % M.p for c in range(M.ncells)], dtype=np.int64)


def one_round(M: RModel, psi_flat):
    N, p = M.N, M.p
    g = psi_flat.reshape(N, N)
    pm4 = (p - 4) % p
    out = np.empty_like(g)
    for i in range(N):
        for j in range(N):
            psi = int(g[i, j])
            sq = (psi * psi) % p
            react = (M.a * (psi * ((M.b + (p - sq)) % p) % p)) % p
            lap = (int(g[(i + 1) % N, j]) + int(g[(i - 1) % N, j])
                   + int(g[i, (j + 1) % N]) + int(g[i, (j - 1) % N])
                   + (pm4 * psi) % p) % p
            out[i, j] = (psi + (M.D * lap) % p + react) % p
    return out.reshape(-1)


def evolve_T(M: RModel, psi_flat):
    for _ in range(M.T):
        psi_flat = one_round(M, psi_flat)
    return psi_flat


def absorb(M: RModel, symbols):
    """symbols: array length blocks*rate of F_p message elements -> pre-squeeze."""
    S = iv(M)
    for k in range(M.blocks):
        U = S.copy()
        U[:M.rate] = (U[:M.rate] + symbols[k * M.rate:(k + 1) * M.rate]) % M.p
        U[M.cap0] = (U[M.cap0] + (k + 1) * 7) % M.p          # counter-lo (G=7)
        U[M.cap2] = (U[M.cap2] + 0) % M.p                    # counter-hi (q1=0 here)
        if k == M.blocks - 1:
            U[M.cap1] = (U[M.cap1] + 3) % M.p                # finalization const
        S = evolve_T(M, U)
    return S


def exhaustive(M: RModel, cap_messages=300_000) -> dict:
    """Enumerate all F_p message symbol vectors; find messages -> zero and
    message-level pre-squeeze collisions."""
    nvars = M.rate * M.blocks
    total = M.p ** nvars
    if total > cap_messages:
        return {"skipped": True, "reason": f"{total} messages > cap {cap_messages}"}
    zero = np.zeros(M.ncells, dtype=np.int64)
    seen = {}
    zero_msgs = []
    collisions = []
    for combo in product(range(M.p), repeat=nvars):
        sym = np.array(combo, dtype=np.int64)
        S = absorb(M, sym)
        if np.array_equal(S, zero):
            zero_msgs.append(combo)
        key = S.tobytes()
        if key in seen:
            collisions.append((seen[key], combo))
        else:
            seen[key] = combo
    return {
        "skipped": False,
        "messages_enumerated": total,
        "messages_reaching_zero": len(zero_msgs),
        "zero_message_examples": [list(x) for x in zero_msgs[:5]],
        "message_collisions": len(collisions),
        "collision_examples": [[list(a), list(b)] for a, b in collisions[:5]],
        "distinct_pre_squeeze_states": len(seen),
        "image_fraction": len(seen) / total,
    }


def z3_message_to_zero(M: RModel, timeout_ms=20000) -> dict:
    if not _Z3:
        return {"available": False}
    p = M.p
    syms = [z3.Int(f"m{i}") for i in range(M.rate * M.blocks)]
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    for v in syms:
        s.add(v >= 0, v < p)
    S = [z3.IntVal(int(x)) for x in iv(M)]
    for k in range(M.blocks):
        U = list(S)
        for c in range(M.rate):
            U[c] = (U[c] + syms[k * M.rate + c]) % p
        U[M.cap0] = (U[M.cap0] + (k + 1) * 7) % p
        if k == M.blocks - 1:
            U[M.cap1] = (U[M.cap1] + 3) % p
        for _ in range(M.T):
            U = _z3_round(M, U, p)
        S = U
    for c in range(M.ncells):
        s.add(S[c] % p == 0)
    t0 = time.perf_counter()
    res = s.check()
    dt = time.perf_counter() - t0
    out = {"available": True, "result": str(res), "solve_time_s": round(dt, 3),
           "vars": len(syms)}
    if str(res) == "sat":
        mdl = s.model()
        out["witness"] = [mdl[v].as_long() for v in syms]
        # verify witness through the exact numpy pipeline
        S2 = absorb(M, np.array(out["witness"], dtype=np.int64))
        out["verified_zero"] = bool(np.array_equal(S2, np.zeros(M.ncells, dtype=np.int64)))
    return out


def _z3_round(M: RModel, cells, p):
    N = M.N
    pm4 = (p - 4) % p
    out = [None] * (N * N)
    for i in range(N):
        for j in range(N):
            psi = cells[i * N + j]
            sq = (psi * psi) % p
            react = (M.a * (psi * ((M.b + (p - sq)) % p) % p)) % p
            lap = (cells[((i + 1) % N) * N + j] + cells[((i - 1) % N) * N + j]
                   + cells[i * N + (j + 1) % N] + cells[i * N + (j - 1) % N]
                   + (pm4 * psi) % p) % p
            out[i * N + j] = (psi + (M.D * lap) % p + react) % p
    return out


EXHAUSTIVE_GRID = [
    RModel(2, 5, 5, 3, 2, T, blk) for T in (1, 2, 3) for blk in (1, 2, 3)
] + [
    RModel(2, 7, 5, 3, 3, T, blk) for T in (1, 2) for blk in (1, 2, 3)
] + [
    RModel(2, 11, 5, 3, 5, 1, blk) for blk in (1, 2, 3)
] + [
    RModel(2, 13, 5, 3, 6, 1, blk) for blk in (1, 2)
]

SMT_GRID = [
    RModel(2, 7, 5, 3, 3, 1, 2), RModel(2, 7, 5, 3, 3, 2, 3),
    RModel(2, 11, 5, 3, 5, 1, 3), RModel(2, 13, 5, 3, 6, 2, 3),
    RModel(4, 5, 5, 3, 2, 1, 1), RModel(4, 7, 5, 3, 3, 1, 2),
]


def main(seed: int = 80130) -> dict:
    t0 = time.perf_counter()
    print("  Exhaustive reduced-model lifting ...")
    ex = []
    for M in EXHAUSTIVE_GRID:
        r = exhaustive(M)
        tag = f"N{M.N}_p{M.p}_T{M.T}_blk{M.blocks}"
        ex.append({"model": tag, "params": M.__dict__, **r})
        if not r.get("skipped"):
            print(f"    {tag}: msgs={r['messages_enumerated']} ->zero={r['messages_reaching_zero']} "
                  f"collisions={r['message_collisions']} img_frac={r['image_fraction']:.3f}")
    print("  SMT (z3) message->zero ...")
    smt = []
    for M in SMT_GRID:
        r = z3_message_to_zero(M)
        tag = f"N{M.N}_p{M.p}_T{M.T}_blk{M.blocks}"
        smt.append({"model": tag, "params": M.__dict__, **r})
        if r.get("available"):
            print(f"    {tag}: {r['result']} in {r['solve_time_s']}s"
                  + (f" witness_verified={r.get('verified_zero')}" if r['result'] == 'sat' else ""))

    results = {
        "phase": "8K_reduced_lifting",
        "metadata": H.env_metadata(),
        "z3_available": _Z3,
        "rate_capacity_note": "rate = ceil(N*N/4) cells (mirrors 64/256); message "
                              "symbols are arbitrary F_p (relaxed analog for tiny p).",
        "exhaustive": ex,
        "smt": smt,
        "interpretation": "Reduced-model lifting (or its failure) identifies "
                          "mechanisms only; NOT extrapolated to the normative system.",
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    path = H.save_artifact("phase8k_reduced_lifting.json", results)
    print("  saved", path, f"({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
