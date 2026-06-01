"""
ots_ledger_concurrent_double_accept.py — Mythos M3: no inter-process ledger lock.

THREAT (pre-fix): ``PersistentOTSReplayLedger`` guarded its accept critical
section with only a per-INSTANCE ``threading.RLock``. Two separate ledger
instances (or two processes) pointed at the SAME ledger file each kept their own
in-memory consumed set, loaded once at construction. Their
read-check-append-fsync-update sections could interleave, so both could accept
the SAME OTS identity — one-time becomes two-time.

POST-FIX: the whole critical section runs under an OS-level ``flock`` (POSIX),
and under that lock the ledger re-scans the file so a concurrent append by
another instance/process is seen before the duplicate check. At most one of N
competing instances accepts a given identity. ``concurrent_accept_count`` now
returns 1.

NOTE: inter-process exclusion relies on ``fcntl.flock`` (POSIX). On platforms
without ``fcntl`` only the per-instance lock applies — see
``ots_ledger.INTERPROCESS_LOCKING``.
"""

from __future__ import annotations

import copy
import os
import tempfile
import threading

from wavelock.crypto.ots_ledger import PersistentOTSReplayLedger, INTERPROCESS_LOCKING
from wavelock.crypto.wavelock_ots import generate_ots_keypair, sign_ots


def concurrent_accept_count(tmp: str | None = None, n: int = 8) -> int:
    """N separate ledger instances on one file race to accept ONE identity.

    Returns how many instances reported acceptance. The attack succeeds if this
    is > 1 (double accept). Post-fix on POSIX: exactly 1.
    """
    tmp = tmp or tempfile.mkdtemp(prefix="m3-")
    path = os.path.join(tmp, "ots_replay.jsonl")

    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "double-accept-me")

    # Each "process" is a distinct ledger instance constructed up front (so each
    # starts with an empty in-memory set, like a fresh process would).
    ledgers = [PersistentOTSReplayLedger(path=path) for _ in range(n)]
    results: list[bool] = [False] * n
    barrier = threading.Barrier(n)

    def worker(i: int) -> None:
        barrier.wait()
        results[i] = ledgers[i].accept(pub, "double-accept-me", copy.deepcopy(sig))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return sum(1 for r in results if r)


def two_instances_double_accept(tmp: str | None = None) -> bool:
    """Two distinct instances on the same file both accept the same identity?

    Returns True iff BOTH accept (the break). Post-fix on POSIX: False.
    """
    tmp = tmp or tempfile.mkdtemp(prefix="m3b-")
    path = os.path.join(tmp, "ots_replay.jsonl")
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "one-time")

    led1 = PersistentOTSReplayLedger(path=path)
    led2 = PersistentOTSReplayLedger(path=path)
    r1 = led1.accept(pub, "one-time", copy.deepcopy(sig))
    r2 = led2.accept(pub, "one-time", copy.deepcopy(sig))
    return bool(r1 and r2)


if __name__ == "__main__":
    print(f"[M3] inter-process locking available: {INTERPROCESS_LOCKING}")
    count = concurrent_accept_count()
    both = two_instances_double_accept()
    print(f"[M3] concurrent acceptances of one identity (want 1): {count}")
    print(f"[M3] two instances both accept same identity:         {both}")
    if count <= 1 and not both:
        print("  => BLOCKED. Inter-process flock serializes the accept critical section.")
    else:
        print("  => BREAK STILL WORKS (M3 NOT fixed).")
