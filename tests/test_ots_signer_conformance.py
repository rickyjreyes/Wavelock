"""Python/Lean conformance for the WaveLock-OTS signing guard.

``lean/ots_signer_traces.json`` runs the preexisting signing guard: real key files (one secret seed
per material code), real ``sign_ots`` calls and real registry directories, with claim failures and
a crash injected at the modelled points.  ``rev`` selects the guard: ``current`` is this code;
``c3cb0d1`` (key-id claim only) and ``flag_only`` (no registry) are emulated by disabling the
corresponding claims.  ``scripts/gen_ots_signer_conformance.py`` turns the same traces into
``OTSSignerConformance.lean``, where each expected outcome is checked by ``decide`` against
``OTSSignerGuard``.  Agreement on finite traces is evidence that the model describes the code, not a
refinement proof.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_ots_signer_conformance as gen  # noqa: E402

from wavelock.crypto import wavelock_ots as ots  # noqa: E402
from wavelock.storage import durability  # noqa: E402

SIGNER = {t["name"]: t for t in json.loads(
    (ROOT / "lean" / "ots_signer_traces.json").read_text(encoding="utf-8"))["traces"]}


class SimulatedCrash(BaseException):
    """Process death at an injected point (escapes every ``except Exception``)."""


def test_generated_lean_is_current():
    assert (ROOT / "lean" / "WaveLockProofs" / "OTSSignerConformance.lean").read_text(
        encoding="utf-8") == gen.render()


_KEYS: dict = {}


def material_key(mat: int) -> dict:
    """One real WaveLock-OTS keypair per material code (a fixed high-entropy seed)."""
    if mat not in _KEYS:
        seed = hashlib.sha256(b"ots-signer-conformance-material-%d" % mat).digest()
        _KEYS[mat] = ots.generate_ots_keypair(seed=seed)
    return _KEYS[mat]


def marker_name(marker) -> str:
    kind, value = marker
    if kind == "kid":
        return os.path.basename(ots._key_id_marker_path(f"kid-{value}"))
    return os.path.basename(ots._signing_material_marker_path(material_key(value)["secret_key"]))


def run_signer(trace: dict, base: Path, monkeypatch, rev_override=None) -> dict:
    files: dict = {}
    results, sigs = [], []
    for ev in trace["events"]:
        kind = ev[0]
        if kind == "key":
            _, f, kid, mat = ev
            secret = copy.deepcopy(material_key(mat)["secret_key"])
            secret["one_time_key_id"], secret["used"] = f"kid-{kid}", False
            files[f] = (secret, kid, mat)
            continue
        if kind == "wipe":
            shutil.rmtree(base / ev[1], ignore_errors=True)
            continue
        p, f, d, msg = ev[1:5]
        rev = ev[5] if kind == "sign" and len(ev) > 5 else trace["rev"]
        rev = rev_override or rev
        secret, kid, mat = files[f]
        with monkeypatch.context() as m:
            m.setenv("WAVELOCK_OTS_STATE_DIR", str(base / d))
            if rev in ("c3cb0d1", "flag_only"):
                m.setattr(ots, "_claim_signing_material", lambda path: True)
            if rev == "flag_only":
                m.setattr(ots, "_claim_one_time_key", lambda kid: True)
            if kind == "sign_claim_fail":
                index, outcome = ev[5], ev[6]
                real, calls = durability.atomic_write, [0]

                def faulty(path, data, *, exclusive=False, mode=0o600,
                           _index=index, _outcome=outcome, _real=real, _calls=calls):
                    n = _calls[0]
                    _calls[0] += 1
                    if n == _index:
                        if _outcome == "fail_after":
                            _real(path, data, exclusive=exclusive, mode=mode)
                        raise OSError("injected claim failure")
                    return _real(path, data, exclusive=exclusive, mode=mode)
                m.setattr(durability, "atomic_write", faulty)
            if kind == "sign_crash_after_claims":
                def crash(*args, **kwargs):
                    raise SimulatedCrash()
                m.setattr(ots, "_secret_slice", crash)
            try:
                ots.sign_ots(secret, f"msg-{msg}", allow_reuse=kind == "sign_allow_reuse")
            except ots.OTSKeyReuseError:
                results.append("refused")
            except ots.WaveLockOTSError:
                results.append("error")
            except SimulatedCrash:
                results.append("crash")
            else:
                results.append("signed")
                sigs.append([kid, mat, msg])
    markers = {d: sorted(os.listdir(base / d)) if (base / d).exists() else []
               for d in trace["expected"]["markers"]}
    return {"sigs": sigs, "results": results, "markers": markers}


def expected_signer(trace: dict) -> dict:
    exp = trace["expected"]
    return {"sigs": exp["sigs"], "results": exp["results"],
            "markers": {d: sorted(marker_name(m) for m in ms) for d, ms in exp["markers"].items()}}


@pytest.mark.parametrize("name", sorted(SIGNER))
def test_signer_trace_matches_lean(name, tmp_path, monkeypatch):
    trace = SIGNER[name]
    assert run_signer(trace, tmp_path, monkeypatch) == expected_signer(trace)


def test_signer_present_defect_and_correction(tmp_path, monkeypatch):
    """Negative control: the same two-keys-one-seed events give the c3cb0d1 outcome only under the
    emulated c3cb0d1 guard, and the corrected outcome only under the current guard."""
    old, new = SIGNER["signer_same_seed_two_key_ids_c3cb0d1"], SIGNER["signer_same_seed_two_key_ids_current"]
    got_new = run_signer(old, tmp_path / "a", monkeypatch, rev_override="current")
    assert got_new["sigs"] == new["expected"]["sigs"] != old["expected"]["sigs"]
    got_old = run_signer(new, tmp_path / "b", monkeypatch, rev_override="c3cb0d1")
    assert got_old["sigs"] == old["expected"]["sigs"]
    # Both halves of the positions where the two digests differ are now public.
    m1, m2 = (ots._digest_bits(ots._message_digest(f"msg-{x}", bytes.fromhex(
        material_key(7)["secret_key"]["params_hash"]))) for x in (100, 200))
    assert sum(a != b for a, b in zip(m1, m2)) > 64


def test_signer_registry_is_load_bearing(tmp_path, monkeypatch):
    """Negative control: without the registry (flag only), the refused copies of the current-guard
    traces sign."""
    for name in ("signer_copy_refused", "signer_relabelled_copy_current"):
        got = run_signer(SIGNER[name], tmp_path / name, monkeypatch, rev_override="flag_only")
        assert len(got["sigs"]) == 2 and SIGNER[name]["expected"]["results"][-1] == "refused"


def test_signer_mutated_expectation_is_detected(tmp_path, monkeypatch):
    trace = copy.deepcopy(SIGNER["signer_copy_refused"])
    trace["expected"]["results"][-1] = "signed"
    assert run_signer(trace, tmp_path, monkeypatch) != expected_signer(trace)
