"""Persistent safety memory of the PROPOSED RCA reference safety kernel (issues #42/#43).

STATUS: proposed reference mechanism, introduced with ``docs/RCA_FORMALIZATION.md``.  Nothing in
this repository implemented RCA agent state or safety memory before it; the proofs in
``lean/WaveLockProofs/RCA*.lean`` are about this reference, not about preexisting behaviour.

The memory is one append-only JSON-lines file.  Every line is the canonical JSON encoding of one
record ``{"v": 1, "i": index, "prev": <sha256 of the previous line>, "kind": ..., ...}``:

========== ========================================= =============================================
kind       fields                                    admissible after view ``w`` when
========== ========================================= =============================================
genesis    ``agent``, ``baseline``, ``policy``       only as record 0
observe    ``seq`` (>= 1), ``values`` (int vector)   ``seq > w.last_seq`` and same dimension
policy     ``policy``                                strict successor: version + 1, threshold not
                                                     larger, prohibited set not smaller
execute    ``action``                                not locked and action not prohibited
reset      ``credential``                            locked, last deviation below the threshold,
                                                     and the operator credential is valid for
                                                     exactly this position and prefix
========== ========================================= =============================================

``load`` folds the records into a :class:`View` and fails closed (``MemoryCorrupt``) on anything
else: a missing file, a torn or non-canonical line, a broken hash chain, an inadmissible record,
or a log that does not extend the externally supplied anchor.  The kernel appends a record only
if it is admissible for the view loaded under the lock, so it never writes a log its own loader
rejects.  The latch is sticky: an observation whose anchored deviation reaches the threshold
(or a policy tightening below the last deviation) locks the agent until a valid operator reset.

The Lean counterparts are ``RCA.Policy``, ``RCA.Rec``, ``RCA.View``, ``RCA.admissible``,
``RCA.next`` and ``RCA.load`` in ``lean/WaveLockProofs/RCAAgentState.lean``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Callable, Optional

from .drift import check_vector, l1

RECORD_VERSION = 1
ZERO = "0" * 64
COMMON_FIELDS = ("v", "i", "prev", "kind")
KIND_FIELDS = {
    "genesis": ("agent", "baseline", "policy"),
    "observe": ("seq", "values"),
    "policy": ("policy",),
    "execute": ("action",),
    "reset": ("credential",),
}

#: ``verifier(agent, index, head, credential) -> bool``
Verifier = Callable[[str, int, str, str], bool]


class MemoryCorrupt(Exception):
    """The safety memory is missing, unreadable or inadmissible: the kernel refuses to act."""


def canonical(obj) -> bytes:
    """Canonical encoding of a record or credential message (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def digest(line: bytes) -> str:
    return hashlib.sha256(line).hexdigest()


def _is_hex64(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def _is_nat(x) -> bool:
    return type(x) is int and x >= 0


@dataclass(frozen=True)
class Policy:
    """Policy in force: ``threshold`` is the anchored-deviation latch threshold (>= 1);
    ``prohibited`` is the sorted set of action kinds that never execute."""
    version: int
    threshold: int
    prohibited: tuple

    @classmethod
    def make(cls, version: int, threshold: int, prohibited=()) -> "Policy":
        return cls.from_json({"version": version, "threshold": threshold,
                              "prohibited": sorted(set(prohibited))})

    @classmethod
    def from_json(cls, obj) -> "Policy":
        if not isinstance(obj, dict) or set(obj) != {"version", "threshold", "prohibited"}:
            raise ValueError("a policy has exactly version, threshold and prohibited")
        version, threshold, prohibited = obj["version"], obj["threshold"], obj["prohibited"]
        if not _is_nat(version):
            raise ValueError("policy version must be a non-negative integer")
        if type(threshold) is not int or threshold < 1:
            raise ValueError("policy threshold must be an integer >= 1")
        if (not isinstance(prohibited, list)
                or not all(isinstance(k, str) and k for k in prohibited)
                or prohibited != sorted(set(prohibited))):
            raise ValueError("prohibited must be a sorted list of distinct non-empty strings")
        return cls(version, threshold, tuple(prohibited))

    def to_json(self) -> dict:
        return {"version": self.version, "threshold": self.threshold,
                "prohibited": list(self.prohibited)}


def successor(new: Policy, old: Policy) -> bool:
    """Policy updates may only tighten: next version, threshold not larger, nothing unprohibited."""
    return (new.version == old.version + 1 and new.threshold <= old.threshold
            and set(old.prohibited) <= set(new.prohibited))


@dataclass(frozen=True)
class View:
    """The state derived from a memory prefix (``RCA.View`` plus the prefix identity)."""
    agent: str
    baseline: tuple
    policy: Policy
    last_seq: int
    last_dev: int
    locked: bool
    length: int
    head: str

    def authorizes(self, action: str) -> bool:
        """The pre-action guard: the agent is not locked and ``action`` is not prohibited."""
        return not self.locked and action not in self.policy.prohibited


def check_record(rec) -> dict:
    """Schema of one record (types and exact field set); raises ``ValueError``."""
    if not isinstance(rec, dict) or rec.get("kind") not in KIND_FIELDS:
        raise ValueError("unknown record kind")
    kind = rec["kind"]
    if set(rec) != set(COMMON_FIELDS) | set(KIND_FIELDS[kind]):
        raise ValueError(f"wrong field set for a {kind} record")
    if rec["v"] != RECORD_VERSION or type(rec["v"]) is not int:
        raise ValueError("unknown record version")
    if not _is_nat(rec["i"]) or not _is_hex64(rec["prev"]):
        raise ValueError("bad index or chain link")
    if kind == "genesis":
        if not isinstance(rec["agent"], str) or not rec["agent"]:
            raise ValueError("agent id must be a non-empty string")
        if not check_vector(rec["baseline"]):
            raise ValueError("the baseline must be a non-empty integer vector")
        Policy.from_json(rec["policy"])
    elif kind == "observe":
        if type(rec["seq"]) is not int or rec["seq"] < 1:
            raise ValueError("observation sequence numbers start at 1")
        check_vector(rec["values"])
    elif kind == "policy":
        Policy.from_json(rec["policy"])
    elif kind == "execute":
        if not isinstance(rec["action"], str) or not rec["action"]:
            raise ValueError("action kind must be a non-empty string")
    elif not _is_hex64(rec["credential"]):
        raise ValueError("an operator credential is 64 lowercase hex digits")
    return rec


def admissible(view: View, rec: dict, *, verifier: Verifier) -> bool:
    """Whether ``rec`` may follow the prefix summarised by ``view`` (``RCA.admissible``)."""
    kind = rec["kind"]
    if kind == "observe":
        return rec["seq"] > view.last_seq and len(rec["values"]) == len(view.baseline)
    if kind == "policy":
        return successor(Policy.from_json(rec["policy"]), view.policy)
    if kind == "execute":
        return view.authorizes(rec["action"])
    if kind == "reset":
        return (view.locked and view.last_dev < view.policy.threshold
                and bool(verifier(view.agent, view.length, view.head, rec["credential"])))
    return False


def next_view(view: View, rec: dict, line_digest: str) -> View:
    """The view after an admissible record (``RCA.next``)."""
    kind = rec["kind"]
    if kind == "observe":
        dev = l1(rec["values"], view.baseline)
        view = replace(view, last_seq=rec["seq"], last_dev=dev,
                       locked=view.locked or dev >= view.policy.threshold)
    elif kind == "policy":
        policy = Policy.from_json(rec["policy"])
        view = replace(view, policy=policy, locked=view.locked or policy.threshold <= view.last_dev)
    elif kind == "reset":
        view = replace(view, locked=False)
    return replace(view, length=view.length + 1, head=line_digest)


def genesis_record(agent: str, baseline, policy: Policy) -> dict:
    return check_record({"v": RECORD_VERSION, "i": 0, "prev": ZERO, "kind": "genesis",
                         "agent": agent, "baseline": list(check_vector(baseline)),
                         "policy": policy.to_json()})


def _reject_constant(name):
    raise ValueError(f"non-finite number {name}")


def _no_duplicates(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


def parse_line(line: bytes) -> dict:
    """Decode one canonical record line; raise ``MemoryCorrupt`` on anything else."""
    try:
        rec = json.loads(line.decode("ascii"), object_pairs_hook=_no_duplicates,
                         parse_constant=_reject_constant)
        check_record(rec)
        if canonical(rec) != line:
            raise ValueError("non-canonical encoding")
    except (UnicodeDecodeError, ValueError, TypeError, RecursionError) as error:
        raise MemoryCorrupt(f"unreadable safety-memory record: {error}") from error
    return rec


def load(data: bytes, *, verifier: Verifier, anchor: Optional[tuple] = None) -> View:
    """Fold a memory file into its view (``RCA.load``), failing closed.

    ``anchor = (index, digest)`` is an externally kept commitment to a past prefix; the log must
    still contain that exact record (rollback and history rewriting below it are detected).
    """
    if not data:
        raise MemoryCorrupt("no safety memory (the agent is not provisioned)")
    if not data.endswith(b"\n"):
        raise MemoryCorrupt("torn safety-memory tail")
    view = None
    prev = ZERO
    for i, line in enumerate(data[:-1].split(b"\n")):
        rec = parse_line(line)
        if rec["i"] != i or rec["prev"] != prev:
            raise MemoryCorrupt(f"broken hash chain at record {i}")
        line_digest = digest(line)
        if i == 0:
            if rec["kind"] != "genesis":
                raise MemoryCorrupt("the first record must be the genesis record")
            view = View(rec["agent"], tuple(rec["baseline"]), Policy.from_json(rec["policy"]),
                        0, 0, False, 1, line_digest)
        elif not admissible(view, rec, verifier=verifier):
            raise MemoryCorrupt(f"inadmissible {rec['kind']} record at {i}")
        else:
            view = next_view(view, rec, line_digest)
        if anchor is not None and anchor[0] == i and anchor[1] != line_digest:
            raise MemoryCorrupt("the safety memory does not extend its anchor")
        prev = line_digest
    if anchor is not None and anchor[0] >= view.length:
        raise MemoryCorrupt("the safety memory was rolled back below its anchor")
    return view
