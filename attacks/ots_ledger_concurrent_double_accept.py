"""
ots_ledger_concurrent_double_accept.py — PersistentOTSReplayLedger has no
inter-process / inter-instance lock; two instances on the same file both accept
the same signature.

Finding M3 (attacks/WAVELOCK_MYTHOS_BREAK_REPORT.md).

Thesis of the break
-------------------
``PersistentOTSReplayLedger`` serialises ``accept`` with a per-INSTANCE
``threading.RLock``. That protects concurrent threads inside one process sharing
one ledger object. It does NOT protect:

* two processes (e.g. a restarted/forked worker, a second node) opening the SAME
  JSONL file — each has its own RLock and its own in-memory consumed set;
* the single-node reality that the server keeps TWO distinct ledgers anyway
  (``CONSENSUS_OTS_LEDGER`` for block acceptance vs ``OTS_LEDGER`` behind
  ``verify_ots_payload``), which do not share state.

The durable append uses ``open(path, "a")`` with ``fsync`` but no ``flock``/
``O_EXCL`` claim, so two instances can both pass their (empty) consumed-set check
and both append a consumption record for the same ``one_time_key_id`` — a
double-accept. The on-disk JSONL ends up with two records for the same id, which
is itself evidence the "one-time" invariant was violated.

Threat model: attacker submits the same OTS signature to two ledger instances
(two processes / two nodes / the two distinct in-process ledgers) before either
durably reflects the other's consumption. Both accept.

This SUCCEEDS today. The project's honest caveat already flags full cross-node
closure as future work; this PoC pins the concrete mechanism (no file-level
mutual exclusion) so a single-host multi-process deployment is not mistaken for
safe.
"""

from __future__ import annotations

import os
import tempfile

from wavelock.crypto.ots_ledger import PersistentOTSReplayLedger
from wavelock.crypto.wavelock_ots import generate_ots_keypair, sign_ots


def two_instances_double_accept() -> bool:
    """Return True if two ledger instances on one file both accept one sig."""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "ots_replay.jsonl")

    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "pay alice 5")

    # Two ledger objects (= two processes / nodes) opened on the same file.
    a = PersistentOTSReplayLedger(path=path)
    b = PersistentOTSReplayLedger(path=path)

    # Interleaved accept: both check their (empty) sets, then both consume.
    ra = a.accept(pub, "pay alice 5", sig)
    rb = b.accept(pub, "pay alice 5", sig)

    n_records = sum(1 for line in open(path) if line.strip())
    return bool(ra and rb and n_records >= 2)


if __name__ == "__main__":
    doubled = two_instances_double_accept()
    print(f"[M3] same signature double-accepted by two ledger instances: {doubled}")
    if doubled:
        print("  (BREAK — no flock/O_EXCL on the JSONL append; per-instance "
              "RLock does not serialise across processes/instances.)")
    else:
        print("  (defended — second accept was rejected.)")
