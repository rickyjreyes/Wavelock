"""
ots_key_substitution.py — merkle_root / pk_commitments are not bound in verify.

Finding A (WAVELOCK_OTS_REDTEAM.md).

``generate_ots_keypair`` computes a ``merkle_root`` over all ``pk_commitments``
and publishes it in the public key. ``docs/WAVELOCK_OTS_DESIGN.md`` (§4) and the
deterministic-keypair test present ``merkle_root`` as part of the binding public
key. **But ``verify_ots`` never references ``merkle_root`` and never checks that
``pk_commitments`` are consistent with ``merkle_root`` or ``psi_commitment``.**

Consequences:

1. The published ``merkle_root`` provides ZERO verification value. You can
   overwrite it with garbage and signatures still verify.
2. ``pk_commitments`` are accepted with no integrity binding to the key's
   advertised fingerprint. Any deployment that identifies / pins a WaveLock-OTS
   key by its compact fingerprint (``merkle_root``, ``one_time_key_id``, or
   ``psi_commitment``) while accepting the ``pk_commitments`` array alongside is
   fully forgeable: the attacker swaps in their OWN ``pk_commitments`` + params
   (for which they hold the secret), keeps the victim's fingerprint fields, and
   signs anything.

Threat model: attacker has the victim's public key + their own keypair. The
attacker never learns the victim's seed or ψ★.
"""

from __future__ import annotations

import copy

from wavelock.crypto.wavelock_ots import (
    generate_ots_keypair,
    sign_ots,
    verify_ots,
)


def merkle_root_is_ignored() -> bool:
    """A public key whose merkle_root is garbage still verifies. (defect)"""
    kp = generate_ots_keypair(seed=bytes(range(32)))
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "legit message")
    mangled = copy.deepcopy(pub)
    mangled["merkle_root"] = "de" * 32  # totally inconsistent with pk_commitments
    return bool(verify_ots(mangled, "legit message", sig))


def key_substitution_forgery() -> bool:
    """Splice victim fingerprint fields onto attacker commitments → verifies.

    Models a deployment whose key identity is the merkle_root / one_time_key_id.
    The attacker presents a public key that carries the VICTIM's fingerprint but
    the ATTACKER's pk_commitments + params + psi_commitment, then signs an evil
    message with the attacker's own secret. verify_ots accepts it because it
    never ties the fingerprint to the commitments it actually checks.
    """
    victim = generate_ots_keypair(seed=bytes(range(32)))
    vpub = victim["public_key"]

    attacker = generate_ots_keypair(seed=bytes(range(100, 132)))
    apub, asec = attacker["public_key"], attacker["secret_key"]

    evil = "TRANSFER 1000000 TO ATTACKER"
    asig = sign_ots(asec, evil)

    forged_pub = copy.deepcopy(apub)
    # Keep the victim's compact fingerprint fields; everything verify_ots
    # actually checks (params, params_hash, psi_commitment, pk_commitments)
    # remains the attacker's.
    forged_pub["one_time_key_id"] = vpub["one_time_key_id"]
    forged_pub["merkle_root"] = vpub["merkle_root"]

    return bool(verify_ots(forged_pub, evil, asig))


if __name__ == "__main__":
    print(f"[A] garbage merkle_root still verifies:   {merkle_root_is_ignored()}")
    print(f"[A] fingerprint-keyed substitution forge: {key_substitution_forgery()}")
    print("  (both SUCCEED — merkle_root is published but never verified; "
          "pk_commitments are unbound to the key fingerprint.)")
