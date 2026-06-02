#!/usr/bin/env python3
"""
ATTACK 7 (+2) — Seed-space weakness / practical brute-force input recovery.

WaveLock's standalone API (CurvatureKeyPair / CurvatureKeyPairV3) commits to a
caller-supplied seed. The shipped CLI default is seed=42; demos use 123; the
legacy MT path is limited to a 32-bit seed. If a deployment commits to a
low-entropy integer seed, the commitment C leaks the seed by brute force,
because WaveLock(seed) is cheap and fully public.

We:
  1. Hide a "secret" small/medium integer seed, publish only C, and recover it
     by sweeping candidates -> measure wall-clock and seeds/sec.
  2. Extrapolate the cost to exhaust 32-bit and 64-bit seed spaces.
  3. Contrast with the OTS path (os.urandom, 256-bit) which is NOT brute-forceable.

Usage:  python audit/a4_seedspace.py
Writes: audit/artifacts/a4_seedspace.json
"""
from __future__ import annotations
import sys, os, json, time, hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audit._wl as H
from audit.a1_attractor_entropy import evolve_batch, psi0_batch  # batched kernel

# Use the SAME schema the real consensus keypair publishes (CurvatureKeyPairV3
# default is WLv3.1). The schema string is embedded in the serialized header,
# so attacker and victim must agree on it.
SCHEMA = H.wl.SCHEMA_V3_SHAKE


def commitment_for(seed, n=4):
    v = H.evolve(H.psi0_xof(int(seed), n))
    return hashlib.sha256(H.serialize(v, SCHEMA)).hexdigest()


def brute_force(target_C, lo, hi, n=4, batch=4000):
    """Sweep seeds [lo,hi) in batches; return (found_seed, tested, rate)."""
    side = H.side_for_n(n)
    t0 = time.time()
    tested = 0
    for start in range(lo, hi, batch):
        seeds = list(range(start, min(start + batch, hi)))
        ps = evolve_batch(psi0_batch(seeds, n))
        for i, s in enumerate(seeds):
            v = ps[i]
            if not np.all(np.isfinite(v)):
                tested += 1
                continue
            C = hashlib.sha256(H.serialize(v, SCHEMA)).hexdigest()
            tested += 1
            if C == target_C:
                return s, tested, tested / (time.time() - t0)
    return None, tested, tested / (time.time() - t0)


def main():
    out = {"params": {"n": 4, "path": "consensus/XOF (WLv3.1)"}}

    # 1) Recover the shipped CLI default seed (42) from C alone.
    target = commitment_for(42)
    found, tested, rate = brute_force(target, 0, 100_000)
    out["recover_cli_default"] = {
        "published_commitment": target,
        "search_range": "0..100000",
        "recovered_seed": found,
        "candidates_tested": tested,
        "rate_seeds_per_sec": round(rate, 1),
        "success": found == 42,
    }

    # 2) Recover a "random-looking" secret integer hidden in 0..2^20.
    secret = 0xABCDE  # 703710
    target2 = commitment_for(secret)
    t0 = time.time()
    found2, tested2, rate2 = brute_force(target2, 0, 1 << 20)
    out["recover_20bit_secret"] = {
        "secret_seed": secret,
        "published_commitment": target2,
        "search_space_bits": 20,
        "recovered_seed": found2,
        "candidates_tested": tested2,
        "wall_clock_sec": round(time.time() - t0, 2),
        "rate_seeds_per_sec": round(rate2, 1),
        "success": found2 == secret,
    }

    # 3) Extrapolate to full keyspaces using the measured rate.
    rate_eff = rate2
    out["extrapolation"] = {
        "measured_rate_seeds_per_sec": round(rate_eff, 1),
        "exhaust_32bit_seconds": round((1 << 32) / rate_eff, 1),
        "exhaust_32bit_days": round((1 << 32) / rate_eff / 86400, 2),
        "exhaust_40bit_days": round((1 << 40) / rate_eff / 86400, 1),
        "note": ("Single-core pure-numpy rate. Trivially parallelizable across "
                 "cores/GPUs; the 4x4 kernel is ~microseconds. A 32-bit seed "
                 "space (and the legacy MT path is capped at 32-bit) is "
                 "exhaustible; small-integer/CLI-default seeds fall in seconds."),
    }

    # 4) Contrast: OTS path uses os.urandom(256-bit) -> not brute-forceable.
    out["mitigation_high_entropy_seed"] = {
        "ots_default_entropy_bits": 256,
        "note": ("wavelock/crypto/wavelock_ots.py seeds with os.urandom(256-bit); "
                 "at 2^256 the brute force is infeasible. The weakness is the "
                 "STANDALONE WaveLock API accepting low-entropy integer seeds "
                 "and the CLI defaulting to seed=42, NOT a flaw when a full-"
                 "entropy seed is supplied."),
    }

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "artifacts", "a4_seedspace.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=== a4 seed-space brute force ===")
    print(f"CLI default (42) recovered from C: {out['recover_cli_default']['success']} "
          f"(tested {tested})")
    r = out["recover_20bit_secret"]
    print(f"20-bit secret {hex(secret)} recovered: {r['success']} in "
          f"{r['wall_clock_sec']}s ({r['rate_seeds_per_sec']}/s)")
    e = out["extrapolation"]
    print(f"Extrapolated 32-bit exhaustion: {e['exhaust_32bit_days']} days "
          f"single-core @ {e['measured_rate_seeds_per_sec']}/s "
          f"(parallelizable -> hours)")


if __name__ == "__main__":
    main()
