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
* **Reconstructable accepted identities** — a node folds identifiers from
  verified accepted blocks into the consumed set (:meth:`index_signature`).
  The durable ledger may also contain conservatively burned identities when
  acceptance fails after recording consumption but before appending a block.
* **Fail-closed** — any verification, parsing, hashing, fingerprint, Merkle, or
  replay error rejects (returns ``False``). An uncertain durable write may
  consume an identity even when acceptance reports failure; never reuse it.
* **Inter-process safe (Mythos M3)** — the whole read-check-append-fsync-update
  critical section runs under ``flock`` (POSIX) or SQLite exclusion (Windows), so two separate
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

IDENTITIES. Each accepted signature consumes three identifiers:

* ``one_time_key_id`` — a label chosen at key generation;
* ``leaf_id`` — the ``public_key_fingerprint`` (covers the label);
* ``material_id`` — :func:`wavelock_ots.ots_material_id`, the Merkle root of
  the key's one-time signing commitments. It is the only one of the three that
  a relabelling (new key id, new ψ-commitment, other hex spelling) cannot
  change, so it is what makes each commitment table single use.

Records are ``LEDGER_VERSION`` 2. Version 1 records (written before material
identifiers existed) are still loaded; they consume their labels only.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
from typing import Optional
from wavelock.storage.durability import ensure_directory, fsync_directory

try:  # POSIX inter-process file locking (Mythos M3).
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows / non-POSIX
    _fcntl = None

#: Cross-process exclusion uses flock on POSIX and a SQLite exclusive
#: transaction on Windows/non-POSIX. RLock additionally serializes each instance.
INTERPROCESS_LOCKING = True  # flock on POSIX; SQLite lock elsewhere

from wavelock.crypto import wavelock_ots as _ots
from wavelock.crypto.wavelock_ots import (
    SCHEME as OTS_SCHEME,
    WaveLockOTSError,
    verify_ots,
    signature_transcript,
)

#: Bumped if the on-disk record shape changes. 2 adds ``material_id``.
LEDGER_VERSION = 2

#: Exact field set of each record version the loader accepts.
_RECORD_FIELDS = {
    1: frozenset({"v", "one_time_key_id", "leaf_id", "transcript"}),
    2: frozenset({"v", "one_time_key_id", "leaf_id", "material_id", "transcript"}),
}


def _is_hex64(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


class OTSLedgerError(Exception):
    """Raised only for unrecoverable ledger I/O / corruption conditions.

    Replay *rejections* are not exceptions — :meth:`PersistentOTSReplayLedger.accept`
    returns ``False`` for them. This is reserved for a ledger file we cannot read
    or trust, where failing closed means refusing to operate rather than silently
    accepting possible reuse.
    """


def default_ledger_path() -> str:
    """Where the consensus replay ledger lives by default.

    Override with ``WAVELOCK_OTS_LEDGER``. Defaults to ``<data-dir>/ledger/ots_replay.jsonl``
    alongside the block ledger so it travels with chain state.
    """
    p = os.environ.get("WAVELOCK_OTS_LEDGER")
    if p:
        return p
    from wavelock.storage.runtime import LEDGER_DIR
    return str(LEDGER_DIR / "ots_replay.jsonl")


def _ids_for(signature: dict) -> tuple[str, str]:
    """Return ``(one_time_key_id, leaf_id)`` for a signature.

    ``leaf_id`` is the signature's ``public_key_fingerprint``: the hash of the
    canonical public-key payload, recomputed and checked by ``verify_ots`` before
    this is ever trusted.

    Both are labels, not key *material*: the fingerprint covers
    ``one_time_key_id``, so the same commitment table under a new key id gets a
    new fingerprint. The ledger therefore also consumes
    :func:`wavelock_ots.ots_material_id`, which relabelling cannot change. See
    ``docs/WAVELOCK_OTS_FORMAL_VERIFICATION.md``.
    """
    return str(signature.get("one_time_key_id")), str(signature.get("public_key_fingerprint"))


def _identity_types_ok(signature: dict) -> bool:
    """True when both replay identities are strings this ledger can record.

    Replay records require a non-empty string key id (see
    ``_scan_file_into_sets``). ``verify_ots`` compares key ids with ``!=`` and
    does not type them, so without this check a ``""``, ``null`` or array key id
    was accepted once and then left the ledger or chain unloadable.
    """
    kid = signature.get("one_time_key_id")
    leaf = signature.get("public_key_fingerprint")
    return isinstance(kid, str) and kid != "" and isinstance(leaf, str)


class PersistentOTSReplayLedger:
    """Durable, fail-closed ledger of consumed key ids, leaf ids and material ids.

    Use :meth:`accept` from the acceptance layer: it verifies the signature with
    the pure :func:`verify_ots`, checks the durable consumed set, and (only on a
    first valid use) records the identifiers durably before returning ``True``.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or default_ledger_path()
        self._lock = threading.RLock()
        self._key_ids: set[str] = set()
        self._leaf_ids: set[str] = set()
        self._material_ids: set[str] = set()
        self._load()

    # -- inter-process locking (Mythos M3) -----------------------------------

    @contextlib.contextmanager
    def _interprocess_lock(self):
        """Hold an OS-level exclusive lock over the ledger for the whole
        read-check-append-fsync-update critical section.

        Uses ``flock`` on a sibling ``<path>.lock`` file so two separate
        :class:`PersistentOTSReplayLedger` instances (or processes) sharing the
        same ledger file cannot interleave their accept critical sections and
        both accept the same OTS identity. On platforms without fcntl, a SQLite
        exclusive transaction supplies cross-process exclusion.
        """
        directory = os.path.dirname(os.path.abspath(self.path))
        ensure_directory(directory)
        if _fcntl is None:
            # SQLite supplies an OS-backed cross-process lock on Windows too.
            # This database coordinates access; JSONL remains the replay data.
            connection = sqlite3.connect(self.path + ".mutex.sqlite3", timeout=30)
            try:
                connection.execute("BEGIN EXCLUSIVE")
                yield
            finally:
                connection.rollback()
                connection.close()
            return
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
                    version = rec.get("v") if isinstance(rec, dict) else None
                    fields = _RECORD_FIELDS.get(version) if type(version) is int else None
                    if (fields is None or set(rec) != fields
                            or not isinstance(rec["one_time_key_id"], str) or not rec["one_time_key_id"]
                            or not all(_is_hex64(rec[k]) for k in fields - {"v", "one_time_key_id"})):
                        raise OTSLedgerError(f"malformed OTS ledger record at line {lineno}")
                    self._key_ids.add(rec["one_time_key_id"])
                    self._leaf_ids.add(rec["leaf_id"])
                    if "material_id" in rec:
                        self._material_ids.add(rec["material_id"])
        except OSError as e:
            raise OTSLedgerError(
                f"cannot read OTS ledger {self.path!r}: {e}"
            ) from e

    def _load(self) -> None:
        """Read the existing ledger into memory. Corruption fails closed."""
        with self._lock, self._interprocess_lock():
            self._scan_file_into_sets()

    def _append(self, kid: str, leaf: str, material: str, transcript: str) -> None:
        """Append one consumed record durably (flush + fsync). Raises on failure."""
        directory = os.path.dirname(os.path.abspath(self.path))
        ensure_directory(directory)
        rec = {
            "v": LEDGER_VERSION,
            "one_time_key_id": kid,
            "leaf_id": leaf,
            "material_id": material,
            "transcript": transcript,
        }
        line = json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        fsync_directory(directory)

    # -- queries -------------------------------------------------------------

    def is_consumed(self, signature: dict, public_key: Optional[dict] = None) -> bool:
        """True if the key id or leaf id, or (given ``public_key``) the
        commitment table, has already been consumed."""
        kid, leaf = _ids_for(signature)
        material = None
        if public_key is not None:
            try:
                material = _ots.ots_material_id(public_key)
            except WaveLockOTSError:
                material = None  # an unidentifiable table cannot be accepted anyway
        with self._lock, self._interprocess_lock():
            self._scan_file_into_sets()
            return (kid in self._key_ids or leaf in self._leaf_ids
                    or (material is not None and material in self._material_ids))

    # -- mutation ------------------------------------------------------------

    def accept(self, public_key: dict, message, signature: dict) -> bool:
        """Verify, then consume. Returns ``True`` only on a first valid use.

        Fail-closed. Returns ``False`` on:

        * a non-WaveLock-OTS scheme on either object (legacy SIGv2 is never
          accepted here);
        * any ``verify_ots`` failure (strict canonical / Merkle / fingerprint
          checks);
        * a ``one_time_key_id`` that is not a non-empty string, or a
          fingerprint that is not a string (records must stay loadable);
        * a replayed/duplicate ``one_time_key_id``, leaf id, or commitment
          table (``ots_material_id``) — whatever labels it now carries;
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
            # Only identities this ledger can durably record and reload.
            if not _identity_types_ok(signature):
                return False
            kid, leaf = _ids_for(signature)
            material = _ots.ots_material_id(public_key)
            transcript = signature_transcript(signature)
            # Entire read-check-append-fsync-update is one critical section,
            # guarded BOTH by the per-instance RLock (intra-process) and by an
            # OS-level flock (inter-process, Mythos M3). Under the flock we
            # re-scan the file so appends made by another process/instance since
            # construction are seen before the duplicate check — two separate
            # instances on the same file cannot both accept the same identity.
            with self._lock, self._interprocess_lock():
                self._scan_file_into_sets()
                if (kid in self._key_ids or leaf in self._leaf_ids
                        or material in self._material_ids):
                    return False
                # Durable record BEFORE the in-memory mutation: if the fsync'd
                # append fails we reject. A partial/uncertain write can still
                # conservatively consume the key; it never permits reuse.
                self._append(kid, leaf, material, transcript)
                self._key_ids.add(kid)
                self._leaf_ids.add(leaf)
                self._material_ids.add(material)
            return True
        except Exception:
            # Any parsing/hash/fingerprint/Merkle/replay/IO error rejects.
            return False

    def index_signature(self, signature: dict, public_key: Optional[dict] = None) -> None:
        """Memory-only fold-in of an already-accepted signature's identifiers.

        Used when reconstructing the consumed set from accepted chain blocks at
        startup, so accepted identities remain consumed even if the JSONL cache
        was deleted. This cannot recover identities burned before chain append.
        Does not re-verify (the block was already verified) or write to disk.
        With ``public_key`` the commitment table is folded in too; a table that
        cannot be identified raises :class:`WaveLockOTSError`.
        """
        if not isinstance(signature, dict):
            return
        kid, leaf = _ids_for(signature)
        material = _ots.ots_material_id(public_key) if public_key is not None else None
        # Skip only absent fields. Comparing str(...) with "None" also dropped a
        # real key id spelled "None", so that identity was not reconstructed.
        with self._lock:
            if signature.get("one_time_key_id") is not None:
                self._key_ids.add(kid)
            if signature.get("public_key_fingerprint") is not None:
                self._leaf_ids.add(leaf)
            if material is not None:
                self._material_ids.add(material)


__all__ = [
    "LEDGER_VERSION",
    "INTERPROCESS_LOCKING",
    "OTSLedgerError",
    "default_ledger_path",
    "PersistentOTSReplayLedger",
]
