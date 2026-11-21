"""Regression pins for the signing guard's material claim (issues #42/#43).

At ``c3cb0d1`` the host-local key-state registry claimed only the ``one_time_key_id``: two key
files holding one seed under two key ids (``generate_ots_keypair(seed=s)`` twice, or a relabelled
copy of a used secret key) both signed through the guarded path, revealing both secret halves of
every position where the two message digests differ.  ``sign_ots`` now also claims the signing
material ``(params_hash, seed)`` before any slice is revealed.  Lean:
``OTSSignerGuard.signer_single_use`` and ``OTSSignerGuard.c3cb0d1_material_reused``.
"""
import copy
import os

import pytest

from wavelock.crypto import wavelock_ots as ots

SEED = bytes(range(32))


@pytest.fixture(scope="module")
def two_keys_one_seed():
    return ots.generate_ots_keypair(seed=SEED), ots.generate_ots_keypair(seed=SEED)


def _both_halves(sig1, sig2):
    b1 = ots._digest_bits(bytes.fromhex(sig1["message_digest"]))
    b2 = ots._digest_bits(bytes.fromhex(sig2["message_digest"]))
    return sum(x != y for x, y in zip(b1, b2))


def test_same_seed_under_two_key_ids_signs_once(two_keys_one_seed):
    a, b = two_keys_one_seed
    assert a["public_key"]["one_time_key_id"] != b["public_key"]["one_time_key_id"]
    assert ots.ots_material_id(a["public_key"]) == ots.ots_material_id(b["public_key"])
    ots.sign_ots(copy.deepcopy(a["secret_key"]), "pay alice 5")
    with pytest.raises(ots.OTSKeyReuseError, match="signing material"):
        ots.sign_ots(copy.deepcopy(b["secret_key"]), "pay mallory 1000")


def test_relabelled_copy_of_used_key_is_refused(two_keys_one_seed):
    a, _ = two_keys_one_seed
    secret = copy.deepcopy(a["secret_key"])
    ots.sign_ots(secret, "first")
    relabelled = copy.deepcopy(a["secret_key"])
    relabelled["one_time_key_id"], relabelled["used"] = "relabelled-id", False
    with pytest.raises(ots.OTSKeyReuseError, match="signing material"):
        ots.sign_ots(relabelled, "second")


def test_plain_copy_still_refused_by_key_id(two_keys_one_seed):
    a, _ = two_keys_one_seed
    ots.sign_ots(copy.deepcopy(a["secret_key"]), "first")
    with pytest.raises(ots.OTSKeyReuseError, match="one_time_key_id"):
        ots.sign_ots(copy.deepcopy(a["secret_key"]), "second")


def test_c3cb0d1_guard_reproduces_the_leak(two_keys_one_seed, monkeypatch):
    """The defect, preserved: with only the key-id claim, one seed signs twice."""
    monkeypatch.setattr(ots, "_claim_signing_material", lambda path: True)
    a, b = two_keys_one_seed
    s1 = ots.sign_ots(copy.deepcopy(a["secret_key"]), "pay alice 5")
    s2 = ots.sign_ots(copy.deepcopy(b["secret_key"]), "pay mallory 1000")
    assert ots.verify_ots(a["public_key"], "pay alice 5", s1)
    assert ots.verify_ots(b["public_key"], "pay mallory 1000", s2)
    assert _both_halves(s1, s2) > 64


def test_distinct_params_are_distinct_material(two_keys_one_seed):
    """One seed under other parameters derives other slices: no shared secret values, no refusal."""
    a, _ = two_keys_one_seed
    other = ots.generate_ots_keypair(params=ots.default_params(n=6), seed=SEED)
    assert other["secret_key"]["params_hash"] != a["secret_key"]["params_hash"]
    ots.sign_ots(copy.deepcopy(a["secret_key"]), "m1")
    ots.sign_ots(copy.deepcopy(other["secret_key"]), "m2")


def test_material_marker_name_is_a_domain_separated_hash(two_keys_one_seed, tmp_path, monkeypatch):
    monkeypatch.setenv("WAVELOCK_OTS_STATE_DIR", str(tmp_path))
    a, _ = two_keys_one_seed
    path = ots._signing_material_marker_path(a["secret_key"])
    name = os.path.basename(path)
    assert name.startswith("material-") and name.endswith(".used")
    assert SEED.hex() not in name and a["secret_key"]["seed_hex"] not in name
    ots.sign_ots(copy.deepcopy(a["secret_key"]), "m")
    assert open(path, "rb").read() == b"signing-material\n"


def test_unusable_key_claims_nothing(tmp_path, monkeypatch):
    """A key without seed material fails before any marker is published (nothing is burned)."""
    monkeypatch.setenv("WAVELOCK_OTS_STATE_DIR", str(tmp_path))
    kp = ots.generate_ots_keypair()
    broken = dict(kp["secret_key"], seed_hex=None)
    with pytest.raises(ots.WaveLockOTSError, match="no seed material"):
        ots.sign_ots(broken, "m")
    assert not tmp_path.exists() or os.listdir(tmp_path) == []
    ots.sign_ots(kp["secret_key"], "m")


def test_boundaries_are_unchanged(two_keys_one_seed, tmp_path, monkeypatch):
    """Documented boundaries (Lean counterexamples): allow_reuse, a second registry directory and
    material that signed before material markers existed still bypass the host-local guard."""
    a, b = two_keys_one_seed
    ots.sign_ots(copy.deepcopy(a["secret_key"]), "m1", allow_reuse=True)
    ots.sign_ots(copy.deepcopy(a["secret_key"]), "m2", allow_reuse=True)
    monkeypatch.setenv("WAVELOCK_OTS_STATE_DIR", str(tmp_path / "other"))
    ots.sign_ots(copy.deepcopy(a["secret_key"]), "m3")
    monkeypatch.setenv("WAVELOCK_OTS_STATE_DIR", str(tmp_path / "legacy"))
    with monkeypatch.context() as m:
        m.setattr(ots, "_claim_signing_material", lambda path: True)
        ots.sign_ots(copy.deepcopy(a["secret_key"]), "signed before the upgrade")
    ots.sign_ots(copy.deepcopy(b["secret_key"]), "relabelled after the upgrade")


def _relabelled(secret, kid):
    out = copy.deepcopy(secret)
    out["one_time_key_id"], out["used"] = kid, False
    return out


def test_threads_race_on_one_material(two_keys_one_seed):
    """Eight threads, eight key ids, one seed: exactly one signature leaves sign_ots."""
    import threading
    a, _ = two_keys_one_seed
    barrier, results = threading.Barrier(8), []

    def worker(i):
        secret = _relabelled(a["secret_key"], f"thread-{i}")
        barrier.wait()
        try:
            ots.sign_ots(secret, f"m{i}")
            results.append("signed")
        except ots.OTSKeyReuseError:
            results.append("refused")
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert sorted(results) == ["refused"] * 7 + ["signed"]


RACER = """
import copy, json, sys
from wavelock.crypto import wavelock_ots as ots
secret = json.load(open(sys.argv[1]))
secret["one_time_key_id"] = sys.argv[2]
try:
    ots.sign_ots(secret, "race " + sys.argv[2])
    print("signed")
except ots.OTSKeyReuseError:
    print("refused")
"""


def test_processes_race_on_one_material(two_keys_one_seed, tmp_path):
    """Six processes on one registry, six key ids, one seed: exactly one signs."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    a, _ = two_keys_one_seed
    secret_path = tmp_path / "secret.json"
    secret_path.write_text(json.dumps(a["secret_key"]))
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root), "WAVELOCK_OTS_STATE_DIR": str(tmp_path / "reg"),
           "WAVELOCK_DATA_DIR": str(tmp_path / "data")}
    procs = [subprocess.Popen([sys.executable, "-c", RACER, str(secret_path), f"proc-{i}"],
                              env=env, stdout=subprocess.PIPE, text=True) for i in range(6)]
    outs = [p.communicate(timeout=180)[0].strip() for p in procs]
    assert sorted(outs) == ["refused"] * 5 + ["signed"]


def test_crash_between_the_two_claims_reveals_nothing(two_keys_one_seed, tmp_path):
    """A process killed after claiming the key id and before claiming the material reveals no
    slice; the burned key id stays refused, and the material can still sign exactly once."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    a, _ = two_keys_one_seed
    secret_path = tmp_path / "secret.json"
    secret_path.write_text(json.dumps(a["secret_key"]))
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root), "WAVELOCK_OTS_STATE_DIR": str(tmp_path / "reg"),
           "WAVELOCK_DATA_DIR": str(tmp_path / "data")}
    crash = subprocess.run([sys.executable, "-c", """
import json, os, sys
from wavelock.crypto import wavelock_ots as ots
ots._claim_signing_material = lambda path: os._exit(71)
ots.sign_ots(json.load(open(sys.argv[1])), "never")
""", str(secret_path)], env=env, capture_output=True, text=True, timeout=120)
    assert crash.returncode == 71
    os.environ["WAVELOCK_OTS_STATE_DIR"] = str(tmp_path / "reg")
    try:
        with pytest.raises(ots.OTSKeyReuseError, match="one_time_key_id"):
            ots.sign_ots(copy.deepcopy(a["secret_key"]), "same key id")
        ots.sign_ots(_relabelled(a["secret_key"], "after-crash"), "the one signature")
        with pytest.raises(ots.OTSKeyReuseError, match="signing material"):
            ots.sign_ots(_relabelled(a["secret_key"], "third"), "never")
    finally:
        del os.environ["WAVELOCK_OTS_STATE_DIR"]


def test_registry_entries_are_private_and_secret_free(two_keys_one_seed, tmp_path, monkeypatch):
    monkeypatch.setenv("WAVELOCK_OTS_STATE_DIR", str(tmp_path))
    a, _ = two_keys_one_seed
    ots.sign_ots(copy.deepcopy(a["secret_key"]), "m")
    seed_hex = a["secret_key"]["seed_hex"]
    for name in os.listdir(tmp_path):
        path = tmp_path / name
        data = path.read_bytes()
        assert seed_hex not in name and seed_hex.encode() not in data
        assert bytes.fromhex(seed_hex) not in data
        if os.name == "posix":
            assert (path.stat().st_mode & 0o077) == 0


def test_migration_helper_claims_legacy_material(two_keys_one_seed, monkeypatch):
    """A key that signed under the key-id-only guard: claim_signing_material records its material,
    after which a relabelled copy is refused."""
    a, _ = two_keys_one_seed
    with monkeypatch.context() as m:
        m.setattr(ots, "_claim_signing_material", lambda path: True)
        ots.sign_ots(copy.deepcopy(a["secret_key"]), "signed before the upgrade")
    assert ots.claim_signing_material(a["secret_key"]) is True
    assert ots.claim_signing_material(a["secret_key"]) is False
    with pytest.raises(ots.OTSKeyReuseError, match="signing material"):
        ots.sign_ots(_relabelled(a["secret_key"], "relabelled"), "refused")
