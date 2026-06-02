"""
ots_block_body_unbound.py — Mythos M1: OTS block signature does not bind body.

THREAT (pre-fix): block acceptance verified the OTS signature against the
arbitrary ``meta.ots_auth.message`` string, while the actually-accepted payload
is ``Block.messages``. Those two were never tied together. An attacker who holds
ANY public, valid OTS signature over a benign message (e.g. ``"hello world"``)
could wrap it into a block whose body says something completely different and
have it accepted — the signature "authorized" text nobody agreed to put in the
block.

POST-FIX: ``server._verify_ots_block`` recomputes a canonical block-signing
transcript (:func:`server.canonical_ots_block_digest`) from the *received* block
and (a) requires ``meta.ots_auth.message`` to equal that digest and (b) verifies
the signature against that digest. A benign signature over ``"hello world"`` no
longer authorizes any block body. Both attack functions below now return
``False`` (= attack BLOCKED).

Run directly to see the two breaks fail closed.
"""

from __future__ import annotations

import os
import tempfile

from wavelock.chain.Block import Block
from wavelock.crypto.ots_ledger import PersistentOTSReplayLedger
from wavelock.crypto.wavelock_ots import (
    generate_ots_keypair,
    sign_ots,
    verify_ots,
)
from wavelock.network import server


def _ledger(tmp: str) -> PersistentOTSReplayLedger:
    return PersistentOTSReplayLedger(path=os.path.join(tmp, "ots_replay.jsonl"))


def benign_signature_authorizes_malicious_body(tmp: str | None = None) -> bool:
    """Wrap a benign ``"hello world"`` signature into a malicious-body block.

    Returns True iff the block is ACCEPTED (the attack succeeds). Post-fix: False.
    """
    own_tmp = tmp is None
    tmp = tmp or tempfile.mkdtemp(prefix="m1-")
    try:
        kp = generate_ots_keypair(entropy_bits=256)
        pub, sec = kp["public_key"], kp["secret_key"]

        # Attacker holds a public, crypto-valid signature over a benign message.
        benign = "hello world"
        sig = sign_ots(sec, benign)
        assert verify_ots(pub, benign, sig) is True

        # Splice it into a block whose ACCEPTED body says something else, while
        # claiming the signed message is still the benign "hello world".
        meta = server.build_ots_block_meta(pub, benign, sig)
        b = Block(index=1, messages=["transfer 1000000 to attacker"],
                  previous_hash="0" * 64, difficulty=1, block_type="OTS", meta=meta)

        return bool(server._verify_ots_block(b, None, ledger=_ledger(tmp)))
    finally:
        if own_tmp:
            pass  # leave tmp dir for inspection in __main__


def body_tampering_after_signing(tmp: str | None = None) -> bool:
    """Sign a block correctly, then mutate ``Block.messages``.

    Returns True iff the tampered-body block is still ACCEPTED. Post-fix: False.
    """
    tmp = tmp or tempfile.mkdtemp(prefix="m1b-")
    kp = generate_ots_keypair(entropy_bits=256)
    # Properly body-bound block (signature signs the canonical transcript).
    good = server.build_signed_ots_block(kp["secret_key"], kp["public_key"],
                                         ["pay alice 5"])
    # Attacker rewrites the body but keeps the signed auth material.
    tampered = Block(index=good.index, messages=["pay attacker 5000000"],
                     previous_hash=good.previous_hash, difficulty=1,
                     block_type="OTS", meta=good.meta)
    return bool(server._verify_ots_block(tampered, None, ledger=_ledger(tmp)))


if __name__ == "__main__":
    a = benign_signature_authorizes_malicious_body()
    b = body_tampering_after_signing()
    print(f"[M1] benign 'hello world' signature authorizes malicious body: {a}")
    print(f"[M1] post-signing body tampering accepted:                     {b}")
    if not a and not b:
        print("  => BLOCKED. The OTS signature now binds the canonical block body.")
    else:
        print("  => BREAK STILL WORKS (M1 NOT fixed).")
