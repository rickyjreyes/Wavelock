"""Phase CC-3 Part V -- solver-assisted valid-message reachability of v_star.

Encodes ACTUAL message bytes (0..255) as variables under the normative
absorption map and asks z3 whether a valid message drives a wave coordinate to
v_star (or the full lattice to the constant c=357959172).

Tasks:
  A. round-0 rate coordinate == v_star  -> expected UNSAT (exact, matches theorem).
  B. round-0 rate coordinate == c=357959172 -> expected UNSAT (above the byte window).
  C. round-0 fixed coordinate (67..255) == v_star -> expected UNSAT (fixed at IV).
  D. round-1 rate coordinate == v_star (one wave round, bytes free) -> SAT/UNSAT/UNKNOWN.
  E. round-1 full constant c*1 -> expected UNKNOWN/UNSAT (256 eqns; bounded attempt).

A timeout is UNKNOWN, NOT evidence of unreachability. Any SAT witness is replayed
through the reference and optimized implementations and recorded.
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
PM4 = (P - 4) % P
C_CONST = 357959172


def _iv():
    return bopt.iv_psi().reshape(-1)


def _neighbors(x):
    i, j = divmod(x, N)
    up = ((i - 1) % N) * N + j
    dn = ((i + 1) % N) * N + j
    lf = i * N + (j - 1) % N
    rt = i * N + (j + 1) % N
    return up, dn, lf, rt


def _round0_cell_eq(target_value, cell, expect):
    """z3: can message bytes make psi_0[cell] == target_value? (rate cell)."""
    import z3
    iv = _iv()
    s = z3.Solver(); s.set("timeout", 15000)
    # 3 message bytes for this rate cell
    b0, b1, b2 = z3.Ints(f"b0_{cell} b1_{cell} b2_{cell}")
    for b in (b0, b1, b2):
        s.add(b >= 0, b < 256)
    elem = b0 + (b1 * 256) + (b2 * 65536)
    psi0 = (int(iv[cell]) + elem) % P
    s.add(psi0 == target_value % P)
    t0 = time.perf_counter()
    r = s.check()
    out = {"cell": cell, "target": int(target_value), "result": str(r),
           "runtime_s": round(time.perf_counter() - t0, 3), "expected": expect}
    return out


def _round0_fixed_cell(target_value, cell, expect):
    """Fixed coordinate 67..255 holds iv[cell] regardless of message."""
    iv = _iv()
    val = int(iv[cell])
    return {"cell": cell, "fixed_value": val, "target": int(target_value),
            "result": "unsat" if val != target_value % P else "sat",
            "expected": expect,
            "note": "coordinate is never written by absorption; holds the IV constant."}


def _round1_cell_vstar(cell=20, timeout_ms=60000):
    """z3: can message bytes make psi_1[cell] == v_star after ONE wave round?

    cell and its 4 neighbors are rate cells (fully message-controlled). One
    degree-3 equation over F_p in 15 byte variables.
    """
    import z3
    iv = _iv()
    up, dn, lf, rt = _neighbors(cell)
    cells = [cell, up, dn, lf, rt]
    assert all(c < 64 for c in cells), "need all-rate neighborhood"
    s = z3.Solver(); s.set("timeout", timeout_ms)
    psi0 = {}
    bytevars = {}
    for c in cells:
        b0, b1, b2 = z3.Ints(f"b0_{c} b1_{c} b2_{c}")
        for b in (b0, b1, b2):
            s.add(b >= 0, b < 256)
        bytevars[c] = (b0, b1, b2)
        psi0[c] = (int(iv[c]) + b0 + b1 * 256 + b2 * 65536) % P
    u = psi0[cell]
    lap = (psi0[up] + psi0[dn] + psi0[lf] + psi0[rt] + (PM4 * u) % P) % P
    react = (A * ((u * ((B + P - (u * u) % P) % P)) % P)) % P
    psi1 = (u + (D * lap) % P + react) % P
    s.add(psi1 == V_STAR)
    t0 = time.perf_counter()
    r = s.check()
    dt = round(time.perf_counter() - t0, 3)
    out = {"cell": cell, "neighbors": cells, "variables": 15, "equations": 1,
           "degree": 3, "result": str(r), "runtime_s": dt, "timeout_ms": timeout_ms}
    if str(r) == "sat":
        m = s.model()
        # build the full message bytes (only these cells set; rest 0), length 191
        msg = bytearray(191)
        for c in cells:
            b0, b1, b2 = bytevars[c]
            vals = [m[b0].as_long(), m[b1].as_long(), m[b2].as_long()]
            for off, vv in enumerate(vals):
                pos = 3 * c + off
                if pos < 191:
                    msg[pos] = vv
        out["witness_message_hex"] = bytes(msg).hex()
        # REPLAY through the real protocol: absorb block 0, one wave round, check cell.
        padded = bopt._pad(bytes(msg))
        arr = np.frombuffer(padded, dtype=np.uint8).astype(np.int64)
        nb = arr.size // spec.BYTES_PER_BLOCK
        arr = arr.reshape(nb, spec.RATE, 3)
        elems = arr[:, :, 0] + (arr[:, :, 1] << 8) + (arr[:, :, 2] << 16)
        psi = bopt.iv_psi().reshape(-1).copy()
        psi[:spec.RATE] = (psi[:spec.RATE] + elems[0]) % P
        psi1 = bopt._wave_round(psi.reshape(N, N)) % P
        out["replay_psi1_cell_value"] = int(psi1.reshape(-1)[cell])
        out["replay_matches_vstar"] = bool(int(psi1.reshape(-1)[cell]) == V_STAR)
        out["replay_total_vstar_cells_round1"] = int(np.sum(psi1 == V_STAR))
        out["replay_note"] = ("psi_1[cell] depends only on psi_0 at cell and its 4 "
                              "rate neighbors, so the isolated witness replays exactly "
                              "through the normative protocol: a VALID message reaches "
                              "v_star at one coordinate at round 1.")
    return out


def _round1_full_constant(timeout_ms=60000):
    """z3: one wave round to the full constant c*1 from a message-reachable psi_0.

    256 equations; cells 67..255 of psi_0 are fixed IV, so F(psi_0)=c*1 forces
    fixed-cell constraints that are generically unsatisfiable. Bounded attempt."""
    import z3
    iv = _iv()
    s = z3.Solver(); s.set("timeout", timeout_ms)
    # psi_0: rate cells free (via elem in [0,2^24)), others fixed at IV
    psi0 = []
    for x in range(N * N):
        if x < 64:
            e = z3.Int(f"e_{x}")
            s.add(e >= 0, e < (1 << 24))
            psi0.append((int(iv[x]) + e) % P)
        else:
            psi0.append(int(iv[x]) % P)  # fixed (CAP cells ignored for this bounded probe)
    # one wave round, require == c at every cell
    for x in range(N * N):
        up, dn, lf, rt = _neighbors(x)
        u = psi0[x]
        lap = (psi0[up] + psi0[dn] + psi0[lf] + psi0[rt] + (PM4 * u) % P) % P
        react = (A * ((u * ((B + P - (u * u) % P) % P)) % P)) % P
        psi1 = (u + (D * lap) % P + react) % P
        s.add(psi1 == C_CONST)
    t0 = time.perf_counter()
    r = s.check()
    return {"variables": 64, "equations": 256, "degree": 3,
            "result": str(r), "runtime_s": round(time.perf_counter() - t0, 3),
            "timeout_ms": timeout_ms,
            "note": "psi_1 = c*1 from a message-reachable psi_0 (rate cells free, "
                    "67..255 fixed at IV). UNSAT => not reachable in one round; "
                    "UNKNOWN => timeout (not a security result)."}


def main(seed: int = 111000) -> dict:
    t0 = time.perf_counter()
    print("  A: round-0 rate cell == v_star ...")
    a = _round0_cell_eq(V_STAR, cell=0, expect="unsat")
    print("    ", a["result"])
    print("  B: round-0 rate cell == c ...")
    b = _round0_cell_eq(C_CONST, cell=0, expect="unsat")
    print("    ", b["result"])
    print("  C: round-0 fixed cell == v_star ...")
    cc = _round0_fixed_cell(V_STAR, cell=100, expect="unsat")
    print("    ", cc["result"])
    print("  D: round-1 rate cell == v_star (z3, one round) ...")
    d = _round1_cell_vstar()
    print("    ", d["result"])
    print("  E: round-1 full constant c*1 (z3, bounded) ...")
    e = _round1_full_constant()
    print("    ", e["result"])

    out = {
        "artifact": "vstar_message_solver",
        "description": "Solver-assisted valid-message reachability of v_star (z3)",
        "metadata": C.env_metadata(),
        "protocol_version": spec.VERSION,
        "seed": seed,
        "equations": {
            "psi0_rate": "psi_0[c] = (iv[c] + b0 + 256*b1 + 65536*b2) mod p, b_i in [0,255]",
            "wave_round": "F(psi)[x] = psi[x] + D*Lap(psi)[x] + A*psi[x]*(B-psi[x]^2) mod p",
            "v_star": V_STAR, "c_constant": C_CONST,
        },
        "task_A_round0_rate_vstar": a,
        "task_B_round0_rate_c": b,
        "task_C_round0_fixed_vstar": cc,
        "task_D_round1_rate_vstar": d,
        "task_E_round1_full_constant": e,
        "summary": {
            "round0_vstar_reachable": a["result"] == "sat",
            "round0_c_reachable": b["result"] == "sat",
            "round1_single_coord_vstar": d["result"],
            "round1_full_constant": e["result"],
            "any_witness_found": a["result"] == "sat" or b["result"] == "sat"
                                 or d.get("result") == "sat",
        },
        "limitations": [
            "round-1 single-coordinate is one wave round only; full T=32 trajectory "
            "is beyond solver reach and is handled by exhaustive/guided search",
            "UNKNOWN results are timeouts, not unreachability proofs",
            "task E ignores CAP-cell injection (bounded probe of the rate-controlled slab)",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("vstar_message_solver.json", out)
    print(f"  saved vstar_message_solver.json ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
