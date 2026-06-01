"""
Consensus-level WaveLock-OTS replay/usage ledger.

This module provides the **stateful** half of the clean boundary mandated by the
WaveLock-OTS design:

    pure cryptographic verification  ->  wavelock.crypto.wavelock_ots.verify_ots
    stateful acceptance / replay      ->  this module (and the block-acceptance
                                          layer that consumes it)

``verify_ots`` is and stays pure: same inputs, same answer, no I/O, no mutation.
It cannot, on its own, stop Finding C / Finding D — a Lamport-style key that
signs twice produces two *individually valid* signatures, and a copied secret
key on another host bypasses the host-local signing registry
(``wavelock_ots._claim_one_time_key``). The only thing that turns "one-time"
from advisory file state into an enforced invariant is a durable, shared ledger
that records which ``one_time_key_id`` / OTS leaf identifiers have already been
*accepted*, and rejects any later signature that reuses one.

:class:`PersistentOTSReplayLedger` is that ledger. It is:

* **Durable** — consumed identifiers are appended to a JSONL file and ``fsync``-ed
  before the in-memory set is updated, so a crash mid-accept never "loses" a
  consumption (fail-closed: if the durable write fails, the accept fails).
* **Canonical / reconstructable** — the consumed set is exactly the set of
  identifiers carried by already-accepted OTS blocks. A node can rebuild it by
  replaying accepted blocks (:meth:`index_signature`), so the ledger is a
  function of chain state rather than private host state.
* **Fail-closed** — any verification, parsing, hashing, fingerprint, Merkle, or
  replay error rejects (returns ``False``); nothing is consumed on rejection.
* **Inter-process safe (Mythos M3)** — the whole read-check-append-fsync-update
  critical section runs under an OS-level ``flock`` (POSIX), so two separate
  ledger instances/processes sharing the same file cannot both accept the same
  OTS identity. Under the lock the file is re-scanned so a concurrent append is
  seen before the duplicate check.

HONEST LIMITS. A single-file ledger with file locking is *consensus-shaped*, not
a finished distributed consensus. Cross-node/global enforcement is only truly
closed when this rejection runs at every accepting node against a ledger derived
from agreed chain state — that remains future work. File locking gives correct
exclusion for instances/processes on the SAME host/filesystem, not across hosts.
Until cross-node replication exists this is the load-bearing control on one host
plus the mechanism other nodes reuse — not a claim of production safety.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from typing import Optional

try:  # POSIX inter-process file locking (Mythos M3).
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows / non-POSIX
    _fcntl = None

#: True when OS-level (inter-process) ledger locking is available. On POSIX this
#: is ``flock``. On platforms without ``fcntl`` the ledger still serializes
#: within a process via its ``RLock``, but it CANNOT guarantee two separate
#: processes won't both accept the same identity — tests that assert
#: inter-process exclusion are POSIX-only (see tests/test_ots_mythos_break.py).
INTERPROCESS_LOCKING = _fcntl is not None

from wavelock.crypto.wavelock_ots import (
    SCHEME as OTS_SCHEME,
    verify_ots,
    signature_transcript,
)

#: Bumped if the on-disk record shape changes.
LEDGER_VERSION = 1


class OTSLedgerError(Exception):
    """Raised only for unrecoverable ledger I/O / corruption conditions.

    Replay *rejections* are not exceptions — :meth:`PersistentOTSReplayLedger.accept`
    returns ``False`` for them. This is reserved for a ledger file we cannot read
    or trust, where failing closed means refusing to operate rather than silently
    accepting possible reuse.
    """


def default_ledger_path() -> str:
    """Where the consensus replay ledger lives by default.

    Override with ``WAVELOCK_OTS_LEDGER``. Defaults to ``ledger/ots_replay.jsonl``
    alongside the block ledger so it travels with chain state.
    """
    p = os.environ.get("WAVELOCK_OTS_LEDGER")
    if p:
        return p
    return os.path.join("ledger", "ots_replay.jsonl")


def _ids_for(signature: dict) -> tuple[str, str]:
    """Return ``(one_time_key_id, leaf_id)`` for a signature.

    ``leaf_id`` is the signature's ``public_key_fingerprint`` — the canonical,
    collision-resistant identifier of the exact OTS public key (its full
    ``pk_commitments`` / Merkle root / params / ψ-commitment), recomputed and
    checked by ``verify_ots`` before this is ever trusted. Recording both means a
    replay is caught whether the attacker reuses the key id or splices a new id
    onto the same committed key material.
    """
    return str(signature.get("one_time_key_id")), str(signature.get("public_key_fingerprint"))


class PersistentOTSReplayLedger:
    """Durable, fail-closed ledger of consumed OTS ``one_time_key_id`` / leaf ids.

    Use :meth:`accept` from the acceptance layer: it verifies the signature with
    the pure :func:`verify_ots`, checks the durable consumed set, and (only on a
    first valid use) records the identifiers durably before returning ``True``.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or default_ledger_path()
        self._lock = threading.RLock()
        self._key_ids: set[str] = set()
        self._leaf_ids: set[str] = set()
        self._load()

    # -- inter-process locking (Mythos M3) -----------------------------------

    @contextlib.contextmanager
    def _interprocess_lock(self):
        """Hold an OS-level exclusive lock over the ledger for the whole
        read-check-append-fsync-update critical section.

        Uses ``flock`` on a sibling ``<path>.lock`` file so two separate
        :class:`PersistentOTSReplayLedger` instances (or processes) sharing the
        same ledger file cannot interleave their accept critical sections and
        both accept the same OTS identity. On non-POSIX platforms without
        ``fcntl`` this is a no-op and only the per-instance ``RLock`` applies
        (documented; inter-process exclusion is POSIX-only).
        """
        if _fcntl is None:  # pragma: no cover - non-POSIX
            yield
            return
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        lock_path = self.path + ".lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX)
            yield
        finally:
            try:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            finally:
                os.close(fd)

    # -- durable state -------------------------------------------------------

    def _scan_file_into_sets(self) -> None:
        """Union every consumed id currently on disk into the in-memory sets.

        Additive (never clears) so identifiers folded in from accepted chain
        state via :meth:`index_signature` survive a re-scan, while appends made
        by *other* processes since we last read are picked up. Corruption fails
        closed (raises :class:`OTSLedgerError`)."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception as e:  # noqa: BLE001 - rethrow as ledger error
                        raise OTSLedgerError(
                            f"corrupt OTS ledger line {lineno} in {self.path!r}: {e}"
                        ) from e
                    kid = rec.get("one_time_key_id")
                    leaf = rec.get("leaf_id")
                    if kid is not None:
                        self._key_ids.add(str(kid))
                    if leaf is not None:
                        self._leaf_ids.add(str(leaf))
        except OSError as e:
            raise OTSLedgerError(
                f"cannot read OTS ledger {self.path!r}: {e}"
            ) from e

    def _load(self) -> None:
        """Read the existing ledger into memory. Corruption fails closed."""
        with self._lock:
            self._scan_file_into_sets()

    def _append(self, kid: str, leaf: str, transcript: str) -> None:
        """Append one consumed record durably (flush + fsync). Raises on failure."""
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        rec = {
            "v": LEDGER_VERSION,
            "one_time_key_id": kid,
            "leaf_id": leaf,
            "transcript": transcript,
        }
        line = json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    # -- queries -------------------------------------------------------------

    def is_consumed(self, signature: dict) -> bool:
        """True if this signature's key id OR leaf id has already been accepted."""
        kid, leaf = _ids_for(signature)
        with self._lock:
            return kid in self._key_ids or leaf in self._leaf_ids

    # -- mutation ------------------------------------------------------------

    def accept(self, public_key: dict, message, signature: dict) -> bool:
        """Verify, then consume. Returns ``True`` only on a first valid use.

        Fail-closed. Returns ``False`` (consuming nothing) on:

        * a non-WaveLock-OTS scheme on either object (legacy SIGv2 is never
          accepted here);
        * any ``verify_ots`` failure (strict canonical / Merkle / fingerprint
          checks);
        * a replayed/duplicate ``one_time_key_id`` or leaf id;
        * a durable-write failure or any other exception.
        """
        try:
            if not isinstance(public_key, dict) or not isinstance(signature, dict):
                return False
            if public_key.get("scheme") != OTS_SCHEME:
                return False
            if signature.get("scheme") != OTS_SCHEME:
                return False
            # Pure cryptographic verification (no state) first (no lock needed).
            if not verify_ots(public_key, message, signature):
                return False
            kid, leaf = _ids_for(signature)
            transcript = signature_transcript(signature)
            # Entire read-check-append-fsync-update is one critical section,
            # guarded BOTH by the per-instance RLock (intra-process) and by an
            # OS-level flock (inter-process, Mythos M3). Under the flock we
            # re-scan the file so appends made by another process/instance since
            # construction are seen before the duplicate check — two separate
            # instances on the same file cannot both accept the same identity.
            with self._lock, self._interprocess_lock():
                self._scan_file_into_sets()
                if kid in self._key_ids or leaf in self._leaf_ids:
                    return False
                # Durable record BEFORE the in-memory mutation: if the fsync'd
                # append fails we raise out and consume nothing (fail-closed).
                self._append(kid, leaf, transcript)
                self._key_ids.add(kid)
                self._leaf_ids.add(leaf)
            return True
        except Exception:
            # Any parsing/hash/fingerprint/Merkle/replay/IO error rejects.
            return False

    def index_signature(self, signature: dict) -> None:
        """Memory-only fold-in of an already-accepted signature's identifiers.

        Used when reconstructing the consumed set from accepted chain blocks at
        startup, so the ledger stays canonical (= a function of chain state) even
        if the JSONL cache was deleted. Does not re-verify (the block was already
        accepted) and does not write to disk.
        """
        if not isinstance(signature, dict):
            return
        kid, leaf = _ids_for(signature)
        with self._lock:
            if kid != "None":
                self._key_ids.add(kid)
            if leaf != "None":
                self._leaf_ids.add(leaf)


__all__ = [
    "LEDGER_VERSION",
    "INTERPROCESS_LOCKING",
    "OTSLedgerError",
    "default_ledger_path",
    "PersistentOTSReplayLedger",
]
