"""
forge_from_snapshot.py — the attack that kills legacy WaveLock SIGv2.

Thesis of the break
-------------------
Legacy WaveLock signs with

    signature = H("SIGv2" || message || header || ψ★)

and verifies by recomputing exactly that hash. Verification therefore needs
ψ★. But ψ★ is *also* everything required to sign. So any party that holds
enough material to verify (the "verifier material" that legacy strict mode
literally publishes as ``psi_star``) can forge a signature on ANY message.

This is the definition of a symmetric MAC keyed by a published secret — not an
asymmetric signature. There is no public/private asymmetry to break; it was
never there.

What this module provides
-------------------------
* :func:`forge_legacy_signature` — given only a ψ★ snapshot (the exact thing
  legacy strict verification distributes), forge a valid SIGv2 signature for an
  arbitrary attacker-chosen message. This SUCCEEDS — and is meant to.
* :func:`attempt_forge_ots_from_public` — the analogous attack against
  WaveLock-OTS: given only the *public* key (commitments/hashes), try to forge.
  This FAILS, because the public key reveals no unrevealed secret slice.

These functions are exercised as regression tests in
``tests/test_ots_security.py`` and ``tests/test_legacy_sigv2_broken.py``:
the legacy forge must keep working (proving the old design is dead), and the
OTS forge must keep failing (proving the new design restores asymmetry).
"""

from __future__ import annotations

import numpy as np

from wavelock.chain.WaveLock import CurvatureKeyPair

def _to_numpy_array(x, dtype=np.float64):
    if hasattr(x, "get"):  # CuPy array
        x = x.get()
    return np.asarray(x, dtype=dtype).copy()

def make_legacy_victim(n: int = 4, seed: int = 12):
    """Create a legacy SIGv2 keypair and return (keypair, psi_star_snapshot).

    The snapshot is exactly the ``psi_star`` array that legacy strict
    verification publishes/loads as "verifier material".
    """
    kp = CurvatureKeyPair(n=n, seed=seed, test_mode=True)
    # snapshot = np.asarray(kp.psi_star, dtype=np.float64).copy()
    snapshot = _to_numpy_array(kp.psi_star)
    return kp, snapshot


def forge_legacy_signature(psi_star_snapshot: np.ndarray, n: int,
                           attacker_message: str) -> str:
    """Forge a valid legacy SIGv2 signature using ONLY the ψ★ snapshot.

    The attacker never sees the original seed or any genuine signature — only
    the verifier material (ψ★). Yet the returned signature verifies. This is
    the whole break.
    """
    forger = CurvatureKeyPair(n=n, seed=1, test_mode=True)  # any seed; overwritten
    forger.psi_star = psi_star_snapshot  # set the leaked verifier material
    # sign() hashes ("SIGv2" || message || header || ψ★) — attacker has ψ★.
    return forger.sign(attacker_message)


def legacy_forgery_succeeds(n: int = 4, seed: int = 12,
                            attacker_message: str = "TRANSFER 1000000 TO ATTACKER"
                            ) -> bool:
    """End-to-end demonstration: snapshot -> forged sig -> verifier accepts."""
    victim, snapshot = make_legacy_victim(n=n, seed=seed)
    forged = forge_legacy_signature(snapshot, n, attacker_message)
    # The victim/verifier, holding ψ★, accepts the forged signature.
    return bool(victim.verify(attacker_message, forged))


def attempt_forge_ots_from_public(public_key: dict, target_message: str) -> dict:
    """Best-effort forgery of a WaveLock-OTS signature from the PUBLIC key only.

    The public key contains commitments ``pk[i][b] = H(... || sk[i][b])`` plus a
    Merkle root and a ψ-commitment — no secret slice. To forge, the attacker
    would need a preimage of ``pk[i][bit_i]`` (i.e. ``sk[i][bit_i]``) for every
    bit of ``H(message)``. With SHAKE256-256 that is a 256-bit preimage search
    per slice; infeasible.

    Here we make the strongest *cheap* attempt: build a fully canonical
    signature (correct fingerprint, key id, recomputed message_digest, all
    fields present) but submit the public commitments themselves as if they were
    the revealed secrets. This is the natural "I only have the public key, let
    me reuse it" forgery. It is structurally valid yet must FAIL, because each
    revealed slice must be a SHAKE256 preimage of pk[i][bit] — which the public
    commitment is not.
    """
    from wavelock.crypto.wavelock_ots import _message_digest, VERSION

    p_hash = bytes.fromhex(public_key["params_hash"])
    pk_commitments = public_key["pk_commitments"]
    n_bits = len(pk_commitments)
    digest = _message_digest(target_message, p_hash)
    # Naive forgery: claim the public commitment IS the secret slice.
    # pick bit 0 arbitrarily; verifier recomputes the real selected bit anyway.
    revealed = [pk_commitments[i][0] for i in range(n_bits)]
    return {
        "scheme": public_key["scheme"],
        "version": VERSION,
        "hash_alg": public_key["hash_alg"],
        "one_time_key_id": public_key["one_time_key_id"],
        "public_key_fingerprint": public_key["public_key_fingerprint"],
        "params_hash": public_key["params_hash"],
        "psi_commitment": public_key["psi_commitment"],
        "message_digest": digest.hex(),
        "revealed_slices": revealed,
    }


if __name__ == "__main__":
    ok = legacy_forgery_succeeds()
    print(f"[legacy SIGv2] forgery from ψ★ snapshot succeeded: {ok}")
    print("  (this is EXPECTED — legacy SIGv2 is broken by design)")
