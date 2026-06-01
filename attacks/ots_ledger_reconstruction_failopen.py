"""
ots_ledger_reconstruction_failopen.py — Mythos M2: replay fail-open on restart.

THREAT (pre-fix): durable replay control depended on the side cache
``ots_replay.jsonl`` AND on ``block_requires_ots(b)`` being evaluated WITHOUT the
config during reconstruction. Two gaps:

  1. ``index_signature`` only ran for blocks detected as OTS by block_type /
     meta.auth_scheme — blocks accepted only because ``cfg.require_ots`` was set
     could be missed.
  2. The consumed set lived only in the JSONL file + memory. Deleting
     ``ots_replay.jsonl`` and restarting produced an EMPTY consumed set, so an
     attacker holding a copied key could replay an already-consumed OTS identity.

POST-FIX: ``server._reconstruct_consumed_ots`` (run by ``ChainState.load_from_disk``)
rebuilds the consumed set from ACCEPTED CHAIN STATE by STRUCTURE — every accepted
block carrying well-formed WaveLock-OTS auth is folded in via
``index_signature`` regardless of current config — and fails closed if an
OTS-claiming block has malformed auth. Deleting the side ledger no longer
resurrects consumed identities. The replay function below now returns ``False``
(= attack BLOCKED).
"""

from __future__ import annotations

import copy
import os
import tempfile

from wavelock.chain.Block import Block
from wavelock.crypto.ots_ledger import PersistentOTSReplayLedger
from wavelock.crypto.wavelock_ots import generate_ots_keypair
from wavelock.network import server


def replay_after_side_ledger_delete(tmp: str | None = None) -> bool:
    """Accept an OTS block, delete the side ledger, restart, replay.

    Returns True iff the replay is ACCEPTED after deleting ``ots_replay.jsonl``
    (the fail-open break). Post-fix: False — reconstruction from the accepted
    chain re-marks the identity consumed.
    """
    tmp = tmp or tempfile.mkdtemp(prefix="m2-")
    path = os.path.join(tmp, "ots_replay.jsonl")

    kp = generate_ots_keypair(entropy_bits=256)
    b1 = server.build_signed_ots_block(kp["secret_key"], kp["public_key"],
                                       ["pay alice 5"])

    led1 = PersistentOTSReplayLedger(path=path)
    assert server._verify_ots_block(b1, None, ledger=led1) is True  # consumed + written

    # Attacker deletes the side replay cache.
    os.remove(path)

    # Restart: a fresh ledger object on the (now-missing) file is EMPTY...
    led2 = PersistentOTSReplayLedger(path=path)
    # ...but the node reconstructs consumed identities from accepted chain state
    # (this is exactly what ChainState.load_from_disk does).
    server._reconstruct_consumed_ots([b1], led2)

    # Replay the same OTS identity in a new block body.
    b2 = Block(index=2, messages=list(b1.messages), previous_hash=b1.previous_hash,
               difficulty=1, block_type="OTS", meta=copy.deepcopy(b1.meta))
    return bool(server._verify_ots_block(b2, None, ledger=led2))


def reconstruction_indexes_cfg_only_block(tmp: str | None = None) -> bool:
    """Reconstruction must index OTS auth detected purely by STRUCTURE.

    Builds an accepted block that carries well-formed OTS auth but a GENERIC
    block_type and NO ``meta.auth_scheme`` (as if it were accepted only because
    ``cfg.require_ots`` was set). Returns True iff reconstruction correctly marks
    its identity consumed (independent of current config). Post-fix: True.
    """
    tmp = tmp or tempfile.mkdtemp(prefix="m2b-")
    path = os.path.join(tmp, "ots_replay.jsonl")
    kp = generate_ots_keypair(entropy_bits=256)
    signed = server.build_signed_ots_block(kp["secret_key"], kp["public_key"],
                                           ["pay alice 5"])
    # Strip the structural OTS markers, keep the auth material in meta.
    auth = signed.meta["ots_auth"]
    stripped = Block(index=1, messages=list(signed.messages),
                     previous_hash=signed.previous_hash, difficulty=1,
                     block_type="GENERIC", meta={"ots_auth": copy.deepcopy(auth)})

    led = PersistentOTSReplayLedger(path=path)
    server._reconstruct_consumed_ots([stripped], led)
    return bool(led.is_consumed(auth["signature"]))


if __name__ == "__main__":
    replay = replay_after_side_ledger_delete()
    indexed = reconstruction_indexes_cfg_only_block()
    print(f"[M2] replay accepted after deleting side ledger: {replay}")
    print(f"[M2] cfg-only OTS block indexed by structure:    {indexed}")
    if not replay and indexed:
        print("  => BLOCKED. Consumed OTS identities are reconstructed from chain state.")
    else:
        print("  => BREAK STILL WORKS (M2 NOT fixed).")
