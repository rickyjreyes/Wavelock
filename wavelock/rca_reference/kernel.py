"""Guarded execution of the PROPOSED RCA reference safety kernel (issues #42/#43).

STATUS: proposed reference mechanism (``docs/RCA_FORMALIZATION.md``); not preexisting production
behaviour, and not used by the chain, network or WaveLock-OTS code.

Architecture (each element answers one counterexample of ``RCACounterexamples.lean``):

1. **One critical section per decision.**  Every mutating call takes the memory lock, reloads the
   memory from disk, evaluates admissibility on that fresh view, appends the record, and (for an
   action) runs the effect -- all before releasing the lock.  No authorization outlives the call,
   so no stale or reused decision exists (counterexamples 4, 6, 9).
2. **Write-ahead intent.**  An ``execute`` record is durable (``fsync``) before the effect is
   invoked; a failed write raises and the effect never runs.  After a crash the record may name an
   effect that did not run (at-most-once), never the reverse (counterexample 10).
3. **Durable, fail-closed memory.**  Acknowledged records survive restarts; a missing, torn,
   rewritten or inadmissible memory stops the agent instead of resetting it (counterexample 1).
4. **Drift before action.**  Each observation's anchored deviation is evaluated in the same
   transaction that records it, with a sticky latch (counterexamples 2, 7).
5. **Monotone policy.**  Updates may only tighten (counterexample 3); releasing the latch needs an
   operator credential bound to the exact memory prefix (no replay).

Trust boundary (stated, not enforced): effects are reachable only through :meth:`request`
(Python cannot stop code that calls an effect directly, counterexample 5); the memory directory
is not rolled back, deleted or edited except as detected by the optional anchor; ``fsync`` is
honest; the operator key is not available to the agent; observations faithfully report behaviour
(counterexample 8).
"""
from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
from typing import Callable, Optional

from wavelock.storage.durability import atomic_write, exclusive_lock

from .drift import check_vector
from .memory import (RECORD_VERSION, MemoryCorrupt, Policy, View, Verifier, admissible,
                     canonical, check_record, digest, genesis_record, load, next_view)

LOG_NAME = "safety-memory.jsonl"
LOCK_NAME = "safety-memory.lock"
CREDENTIAL_PURPOSE = "rca-reset-v1"


def operator_credential(key: bytes, agent: str, index: int, head: str) -> str:
    """Operator credential that releases the latch at exactly memory position ``index`` whose
    previous record has digest ``head``; useless at any other position or after any rewrite."""
    message = canonical({"agent": agent, "index": index, "prev": head,
                         "purpose": CREDENTIAL_PURPOSE})
    return hmac.new(key, message, hashlib.sha256).hexdigest()


class HmacOperatorVerifier:
    """Reference verifier for operator credentials.

    HMAC is symmetric: whoever holds the verification key can also mint credentials, so a
    deployment must use an asymmetric verifier whose signing key the agent never holds.  The
    kernel only needs ``verifier(agent, index, head, credential) -> bool``.
    """

    def __init__(self, key: bytes):
        self._key = bytes(key)

    def __call__(self, agent: str, index: int, head: str, credential: str) -> bool:
        expected = operator_credential(self._key, agent, index, head)
        return isinstance(credential, str) and hmac.compare_digest(credential, expected)


class GuardedAgent:
    """One agent's safety memory and its guarded action path.

    Every public method returns ``True`` when its record was made durable (and, for
    :meth:`request`, the effect was invoked) and ``False`` when the guard refused (nothing was
    written).  ``OSError`` means the write failed: the record may or may not be durable and the
    effect did not run.  ``MemoryCorrupt`` means the memory cannot be trusted: the agent stops.
    Effects run while the memory lock is held; they must not call back into the kernel.
    """

    def __init__(self, directory, *, verifier: Verifier, anchor: Optional[tuple] = None):
        self.directory = Path(directory)
        self._verifier = verifier
        self._anchor = anchor
        with self._lock():
            self.view: View = self._load()

    @classmethod
    def provision(cls, directory, *, agent: str, baseline, policy: Policy, verifier: Verifier,
                  anchor: Optional[tuple] = None) -> "GuardedAgent":
        """Create the memory with its genesis record (atomically, never over an existing one)."""
        directory = Path(directory)
        record = genesis_record(agent, baseline, policy)
        with exclusive_lock(directory / LOCK_NAME):
            atomic_write(directory / LOG_NAME, canonical(record) + b"\n", exclusive=True)
        return cls(directory, verifier=verifier, anchor=anchor)

    # -- internals (also driven step by step by the conformance tests) -------------------------

    def _lock(self):
        return exclusive_lock(self.directory / LOCK_NAME)

    def _load(self) -> View:
        try:
            data = (self.directory / LOG_NAME).read_bytes()
        except FileNotFoundError as error:
            raise MemoryCorrupt("no safety memory (the agent is not provisioned)") from error
        return load(data, verifier=self._verifier, anchor=self._anchor)

    def _write_line(self, data: bytes) -> None:
        fd = os.open(self.directory / LOG_NAME, os.O_WRONLY | os.O_APPEND)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)

    def _submit(self, view: View, fields: dict, effect: Optional[Callable] = None) -> bool:
        """Guard, durable append and (for an action) the effect, on a view loaded under the lock."""
        record = check_record({"v": RECORD_VERSION, "i": view.length, "prev": view.head,
                               **fields})
        if not admissible(view, record, verifier=self._verifier):
            return False
        line = canonical(record)
        self._write_line(line + b"\n")
        self.view = next_view(view, record, digest(line))
        if effect is not None:
            effect(record["action"])
        return True

    def _transact(self, fields: dict, effect: Optional[Callable] = None) -> bool:
        with self._lock():
            return self._submit(self._load(), fields, effect)

    # -- the API -----------------------------------------------------------------------------

    def refresh(self) -> View:
        """Reload the view (read only)."""
        with self._lock():
            self.view = self._load()
        return self.view

    def observe(self, seq: int, values) -> bool:
        """Record one observation; the drift latch is evaluated in the same transaction."""
        return self._transact({"kind": "observe", "seq": seq, "values": list(check_vector(values))})

    def update_policy(self, policy: Policy) -> bool:
        """Install a strictly tighter policy (next version); anything else is refused."""
        return self._transact({"kind": "policy", "policy": policy.to_json()})

    def request(self, action: str, effect: Callable[[str], object]) -> bool:
        """Execute ``effect(action)`` iff the guard admits ``action`` on the fresh memory view.

        The ``execute`` record is durable before ``effect`` runs (at-most-once).
        """
        if not callable(effect):
            raise TypeError("effect must be callable")
        return self._transact({"kind": "execute", "action": action}, effect)

    def reset(self, credential: str) -> bool:
        """Release the latch with an operator credential for the current memory position."""
        return self._transact({"kind": "reset", "credential": credential})
