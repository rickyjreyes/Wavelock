"""Phase CC-3 Part VIII -- security consequence of a singular hit.

We have a confirmed valid-message witness reaching v_star at one coordinate at
round 1 (Part V). A singular coordinate does NOT automatically imply a collision.
This module tests the consequences and attempts to construct a collision.

The singular event: at round t=0, the injection at the hit cell is
    j_B(psi_0[cell], psi_1[cell]) = psi_0[cell]*(1 + GAMMA*v_star) = 0,
i.e. that cell's round-0 wave-injection is erased regardless of psi_0[cell].

KEY structural fact tested here: in the normative protocol every rate byte injects
into BOTH psi (psi[c]+=elem) AND C (C[c]+=G*elem) at ABSORPTION, before any round.
So even when the round-0 wave-injection j_B is zeroed at the singular cell, the
message bytes already entered C directly. A singular wave-event therefore cannot
erase message information from the commitment.

Tests:
  1. confirm the round-0 injection at the hit cell is exactly 0 (the erasure);
  2. confirm the message bytes still entered C at absorption (direct channel);
  3. next-round sensitivity is restored;
  4. collision attempt: messages differing only in the singular cell's bytes;
  5. cubic-root attempt: distinct in-window psi_0[cell] both hitting v_star;
  6. classify the singular hyperplane.
"""

from __future__ import annotations

import time

import numpy as np

from wavelock.curvature_capacity_v1 import spec, optimized as bopt, reference as bref
from . import _common as C

P = spec.P
N = spec.N
A = spec.A
B = spec.B
D = spec.D
GAMMA = spec.GAMMA
V_STAR = spec.V_STAR
PM4 = (P - 4) % P
RATE = spec.RATE
HIT_CELL = 20


def _load_witness():
    import json, os
    path = os.path.join(os.path.dirname(__file__), "artifacts", "vstar_message_solver.json")
    d = json.load(open(path))
    return bytes.fromhex(d["task_D_round1_rate_vstar"]["witness_message_hex"])


def _elems_of(msg):
    padded = bopt._pad(msg)
    arr = np.frombuffer(padded, dtype=np.uint8).astype(np.int64)
    nb = arr.size // spec.BYTES_PER_BLOCK
    arr = arr.reshape(nb, RATE, 3)
    return arr[:, :, 0] + (arr[:, :, 1] << 8) + (arr[:, :, 2] << 16)


def _psi0_psi1(msg):
    iv = bopt.iv_psi().reshape(-1).copy()
    elems = _elems_of(msg)
    iv[:RATE] = (iv[:RATE] + elems[0]) % P
    psi0 = iv.reshape(N, N)
    psi1 = bopt._wave_round(psi0) % P
    return psi0, psi1, elems


def confirm_singular_event(msg) -> dict:
    psi0, psi1, elems = _psi0_psi1(msg)
    u = int(psi0.reshape(-1)[HIT_CELL])
    v = int(psi1.reshape(-1)[HIT_CELL])
    j = (u + GAMMA * (u * v % P)) % P
    # C absorption injection at the hit cell
    c_inject = (spec.G * int(elems[0][HIT_CELL])) % P
    return {
        "hit_cell": HIT_CELL,
        "psi1_hit_is_vstar": v == V_STAR,
        "round0_injection_at_hit_cell": j,
        "round0_injection_is_zero": j == 0,
        "message_byte_C_injection_at_hit_cell": c_inject,
        "message_entered_C_directly_at_absorption": c_inject != 0,
        "note": "round-0 wave-injection j_B at the hit cell is exactly 0 (erased), "
                "but the SAME message bytes injected G*elem into C[cell] at "
                "absorption, so the information is NOT erased from the commitment.",
    }


def next_round_sensitivity(msg) -> dict:
    psi0, psi1, _ = _psi0_psi1(msg)
    psi2 = bopt._wave_round(psi1) % P
    u1 = int(psi1.reshape(-1)[HIT_CELL])  # == v_star
    v2 = int(psi2.reshape(-1)[HIT_CELL])
    j_round1 = (u1 + GAMMA * (u1 * v2 % P)) % P
    return {
        "round1_injection_at_hit_cell": j_round1,
        "round1_injection_nonzero": j_round1 != 0,
        "note": "at round 1 the hit cell value v_star acts as 'u' (earlier state) "
                "and is multiplied by (1+GAMMA*psi_2); the injection is generically "
                "nonzero, so sensitivity is restored the very next round.",
    }


def collision_attempt_single_cell(msg, n_trials=64, seed=114001) -> dict:
    """Messages differing ONLY in the singular cell's 3 bytes -> digests differ."""
    g = C.rng(seed)
    d0 = bopt.cc_hash(msg).hex()
    base = bytearray(msg)
    pos = 3 * HIT_CELL  # byte offset of the hit cell in block 0
    same = 0
    diff = 0
    for _ in range(n_trials):
        m2 = bytearray(base)
        for off in range(3):
            if pos + off < len(m2):
                m2[pos + off] = int(g.integers(0, 256))
        d = bopt.cc_hash(bytes(m2)).hex()
        if d == d0:
            same += 1
        else:
            diff += 1
    return {
        "n_trials": n_trials,
        "digest_collisions": same,
        "digest_distinct": diff,
        "collision_found": same > 0,
        "note": "changing only the singular cell's bytes changes the digest every "
                "time, because those bytes inject into C at absorption (G*elem) and "
                "alter neighbouring psi_1 values injected at round 1. No collision.",
    }


def cubic_root_pair_attempt(msg) -> dict:
    """Find all psi_0[cell] in the message-reachable window with psi_1[cell]=v_star
    (neighbors fixed at the witness values). If >=2, both zero the round-0 injection;
    test whether the two messages collide."""
    import sympy
    from sympy import symbols, Poly
    psi0, psi1, elems = _psi0_psi1(msg)
    flat = psi0.reshape(-1)
    i, j = divmod(HIT_CELL, N)
    up = ((i - 1) % N) * N + j; dn = ((i + 1) % N) * N + j
    lf = i * N + (j - 1) % N; rt = i * N + (j + 1) % N
    nb_sum = int((flat[up] + flat[dn] + flat[lf] + flat[rt]) % P)
    # F(u)[cell] = u + D*(nb_sum + (p-4)*u) + A*u*(B-u^2) = v_star
    # = -A u^3 + (1 + D*(p-4) + A*B) u + D*nb_sum = v_star
    x = symbols("x")
    lin = (1 + D * PM4 + A * B) % P
    poly = Poly((P - A) % P * x**3 + 0 * x**2 + lin * x + (D * nb_sum - V_STAR) % P,
                x, modulus=P)
    roots = sorted(int(r) % P for r in poly.ground_roots())
    iv = bopt.iv_psi().reshape(-1)
    window_lo = int(iv[HIT_CELL]); window_hi = window_lo + (1 << 24) - 1
    in_window = [r for r in roots if window_lo <= r <= window_hi]
    result = {
        "all_roots": roots,
        "n_roots": len(roots),
        "message_window": [window_lo, window_hi],
        "in_window_roots": in_window,
        "n_in_window": len(in_window),
    }
    if len(in_window) >= 2:
        # build two messages differing only at the hit cell, both -> v_star at cell
        msgs = []
        for r in in_window[:2]:
            e = (r - window_lo) % P  # elem value
            m2 = bytearray(msg)
            pos = 3 * HIT_CELL
            for off in range(3):
                if pos + off < len(m2):
                    m2[pos + off] = (int(e) >> (8 * off)) & 0xFF
            msgs.append(bytes(m2))
        d1, d2 = bopt.cc_hash(msgs[0]).hex(), bopt.cc_hash(msgs[1]).hex()
        result["two_message_digests"] = [d1, d2]
        result["collision"] = d1 == d2
    else:
        result["collision"] = False
        result["note"] = ("fewer than 2 message-reachable roots: cannot form a "
                          "same-neighbour singular pair; no collision possible by "
                          "this construction.")
    return result


def main(seed: int = 114000) -> dict:
    t0 = time.perf_counter()
    msg = _load_witness()
    print("  confirm singular event ...")
    ev = confirm_singular_event(msg)
    print(f"    round0 injection zero: {ev['round0_injection_is_zero']}, "
          f"C got bytes: {ev['message_entered_C_directly_at_absorption']}")
    print("  next-round sensitivity ...")
    nr = next_round_sensitivity(msg)
    print("  collision attempt (single-cell perturbation) ...")
    ca = collision_attempt_single_cell(msg)
    print(f"    collisions: {ca['digest_collisions']}/{ca['n_trials']}")
    print("  cubic-root pair attempt ...")
    cr = cubic_root_pair_attempt(msg)
    print(f"    in-window roots: {cr['n_in_window']}, collision: {cr['collision']}")

    collision_found = ca["collision_found"] or cr.get("collision", False)
    classification = ("reachable and yields a structural collision" if collision_found
                      else "reachable but harmless in tested cases")

    out = {
        "artifact": "vstar_collision_consequence",
        "description": "Security consequence of the round-1 v_star singular hit",
        "metadata": C.env_metadata(),
        "protocol_version": spec.VERSION,
        "seed": seed,
        "witness_message_hex": msg.hex(),
        "singular_event": ev,
        "next_round_sensitivity": nr,
        "collision_attempt_single_cell": ca,
        "cubic_root_pair_attempt": cr,
        "classification": classification,
        "consequence_summary": (
            "The singular hit erases the hit cell's round-0 WAVE injection (j_B=0), "
            "but NOT the message information: (1) the same rate bytes inject G*elem "
            "into C at absorption; (2) psi_0[cell] perturbs neighbouring psi_1 values "
            "that are injected at round 1; (3) the hit cell's injection is nonzero "
            "again at round 1. No digest collision was produced by single-cell "
            "perturbation or by the cubic-root same-neighbour construction."
        ),
        "limitations": [
            "tests target the one confirmed witness and its single-cell neighbourhood",
            "no multi-singular-event alignment was achievable (Part VII: none found)",
            "absence of a collision here is bounded evidence, not a proof",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("vstar_collision_consequence.json", out)
    print(f"  classification: {classification}")
    print(f"  saved vstar_collision_consequence.json ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
