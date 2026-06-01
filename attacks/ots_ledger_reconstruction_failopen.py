"""
ots_ledger_reconstruction_failopen.py — replay ledger reconstruction is narrower
than acceptance, so a deleted JSONL + restart re-opens replay (fail-OPEN).

Finding M2 (attacks/WAVELOCK_MYTHOS_BREAK_REPORT.md).

Thesis of the break
-------------------
Two predicates decide whether a block is "OTS-required", and they DISAGREE:

* Acceptance uses ``block_requires_ots(b, cfg)`` — returns True for
  ``block_type == "OTS"`` OR ``meta.auth_scheme == WaveLock-OTS-v1`` OR
  ``cfg.require_ots``.
* Ledger reconstruction (``ChainState.load_from_disk``) uses
  ``block_requires_ots(b)`` with **no cfg** — it CANNOT see ``cfg.require_ots``.

So a block that was OTS-verified and consumed *only because* the node runs with
``cfg.require_ots = True`` (a generic-typed block, no ``auth_scheme`` in meta,
just an ``ots_auth`` payload) is NOT recognised as OTS during reconstruction.
Its consumed ``one_time_key_id`` / leaf id are never re-folded into the rebuilt
ledger.

The durable JSONL cache normally papers over this. But the threat model grants
the attacker the ability to delete the local ledger file and restart the node.
After deletion + restart:

* the fresh ``PersistentOTSReplayLedger`` loads an empty consumed set;
* ``load_from_disk`` reconstructs from accepted blocks, but SKIPS the
  ``require_ots``-classified block (wrong predicate);
* the consumed set is now empty for that key id → the previously-accepted OTS
  signature REPLAYS and is accepted a second time.

This is a durable-replay fail-open: the control that is supposed to make
"one-time" an enforced invariant silently drops state for a whole class of
accepted blocks. ``index_signature`` is also memory-only, so reconstruction is
the ONLY recovery path once the JSONL is gone — and it has a hole.

Threat model: ``cfg.require_ots = True`` deployment (the config knob the project
ships to *enforce* OTS), attacker can delete ``ledger/ots_replay.jsonl`` and
restart (both in scope), and replays one already-accepted signature.

This SUCCEEDS today.
"""

from __future__ import annotations

import os
import tempfile
import types

from wavelock.chain.Block import Block
from wavelock.crypto.ots_ledger import PersistentOTSReplayLedger
from wavelock.crypto.wavelock_ots import generate_ots_keypair, sign_ots
from wavelock.network import server


def _reconstruct(ledger, accepted_blocks):
    """Mirror ChainState.load_from_disk's reconstruction loop exactly."""
    for blk in accepted_blocks:
        if server.block_requires_ots(blk):  # <-- no cfg: the bug
            auth = server._extract_ots_auth(blk)
            if auth and isinstance(auth.get("signature"), dict):
                ledger.index_signature(auth["signature"])


def replay_after_delete_and_restart() -> bool:
    """Return True if a consumed OTS sig replays after JSONL delete + restart."""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "ots_replay.jsonl")

    cfg = types.SimpleNamespace(require_ots=True)
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "pay alice 5")

    # OTS-required ONLY via cfg.require_ots: generic type, no auth_scheme in meta.
    meta = {"ots_auth": {"public_key": pub, "message": "pay alice 5", "signature": sig}}
    block = Block(index=1, messages=["ots"], previous_hash="0" * 64,
                  difficulty=1, block_type="GENERIC", meta=meta)

    led = PersistentOTSReplayLedger(path=path)
    assert server._verify_ots_block(block, cfg, ledger=led) is True       # 1st accept
    assert server._verify_ots_block(block, cfg, ledger=led) is False      # replay blocked

    # Attacker deletes the durable ledger; node restarts and reconstructs.
    os.remove(path)
    led2 = PersistentOTSReplayLedger(path=path)
    _reconstruct(led2, [block])  # block survives in agreed chain state

    # The reconstruction MISSED this block → replay is accepted again.
    return bool(server._verify_ots_block(block, cfg, ledger=led2))


if __name__ == "__main__":
    replayed = replay_after_delete_and_restart()
    print(f"[M2] OTS replay accepted after JSONL delete + restart: {replayed}")
    if replayed:
        print("  (BREAK — reconstruction uses block_requires_ots(b) WITHOUT cfg, "
              "so require_ots-classified OTS blocks are not re-folded into the "
              "rebuilt ledger; the replay control fails OPEN.)")
    else:
        print("  (defended — reconstruction recovered the consumed id.)")
