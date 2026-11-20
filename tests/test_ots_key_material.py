"""Single use of WaveLock-OTS key material (commitment tables), issue #41 Phase II.

Lean: ``OTSVerification`` (``materialId_eq_of_decode_eq``, ``relabelled_replays_keep_material``,
``psi_relabel_verifies``, ``distinct_material_distinct_tables``) and ``OTSKeyConsumption``
(``wavelock_ots_material_single_use``). Trace conformance: ``test_ots_formal_conformance.py``.
"""
from __future__ import annotations

import copy
import json

import pytest

from wavelock.chain.Block import Block
from wavelock.chain.ots_blocks import (
    build_ots_block_meta,
    build_signed_ots_block,
    canonical_ots_block_digest,
    verify_ots_block,
    verify_ots_chain,
)
from wavelock.crypto import wavelock_ots as ots
from wavelock.crypto.ots_ledger import LEDGER_VERSION, OTSLedgerError, PersistentOTSReplayLedger
from wavelock.network import server


@pytest.fixture(scope="module")
def keypair():
    return ots.generate_ots_keypair()


def _refingerprint(pk, sig=None):
    pk = dict(pk)
    pk["public_key_fingerprint"] = ots.public_key_fingerprint(pk)
    if sig is not None:
        sig = dict(sig, one_time_key_id=pk["one_time_key_id"],
                   psi_commitment=pk["psi_commitment"],
                   public_key_fingerprint=pk["public_key_fingerprint"])
    return pk, sig


def _holder_block(pk, sk, body, index, prev):
    """The seed holder signs a block for a relabelled public key (true derivation, ψ relabelled)."""
    timestamp = "1700000000.0"
    header = {"version": 2, "index": index, "timestamp": timestamp, "difficulty": 0}
    draft = Block(index=index, messages=body, previous_hash=prev, difficulty=0, block_type="OTS",
                  meta={"block_header": header, "auth_scheme": ots.SCHEME,
                        "ots_auth": {"public_key": pk}}, timestamp=timestamp)
    digest = canonical_ots_block_digest(draft)
    signer = dict(copy.deepcopy(sk), one_time_key_id=pk["one_time_key_id"],
                  public_key_fingerprint=pk["public_key_fingerprint"])
    signature = ots.sign_ots(signer, digest, allow_reuse=True)
    signature["psi_commitment"] = pk["psi_commitment"]
    meta = build_ots_block_meta(pk, digest, signature)
    meta["block_header"] = header
    return Block(index=index, messages=body, previous_hash=prev, difficulty=0, block_type="OTS",
                 meta=meta, timestamp=timestamp)


# ---------------------------------------------------------------------------
# The identifier
# ---------------------------------------------------------------------------

def test_material_id_is_the_recomputed_commitment_root(keypair):
    pk = keypair["public_key"]
    material = ots.ots_material_id(pk)
    assert material == pk["merkle_root"]
    assert material == ots._merkle_root_from_commitments(pk["pk_commitments"]).hex()
    assert len(material) == 64 and material == material.lower()


def test_material_id_ignores_every_label_the_verifier_admits(keypair):
    """Equality preservation: key id, ψ-commitment, fingerprint and hex spelling are labels."""
    pk, sk = keypair["public_key"], keypair["secret_key"]
    sig = ots.sign_ots(sk, "m", allow_reuse=True)
    bits = ots._digest_bits(bytes.fromhex(sig["message_digest"]))
    table = copy.deepcopy(pk["pk_commitments"])
    unselected = table[0][1 - bits[0]]
    table[0][1 - bits[0]] = " ".join(unselected[i:i + 2] for i in range(0, 64, 2)).upper()
    variants = [
        dict(pk, one_time_key_id="another-id"),
        dict(pk, psi_commitment="ab" * 32),
        dict(pk, pk_commitments=table, one_time_key_id="third-id"),
    ]
    for variant in variants:
        variant, signature = _refingerprint(variant, sig)
        assert ots.verify_ots(variant, "m", signature), "the relabelling verifies"
        assert variant["public_key_fingerprint"] != pk["public_key_fingerprint"]
        assert ots.ots_material_id(variant) == ots.ots_material_id(pk)


def test_material_id_differs_between_keys_and_from_block_root(keypair):
    other = ots.generate_ots_keypair()["public_key"]
    assert ots.ots_material_id(other) != ots.ots_material_id(keypair["public_key"])
    block = build_signed_ots_block(keypair["secret_key"], keypair["public_key"], ["x"],
                                   difficulty=0, allow_reuse=True)
    # Block.merkle_root is the tree over the block's messages, not the key's commitments.
    assert block.merkle_root != ots.ots_material_id(block.meta["ots_auth"]["public_key"])


@pytest.mark.parametrize("mutate", [
    lambda pk: pk.update(merkle_root="00" * 32),
    lambda pk: pk.update(pk_commitments=pk["pk_commitments"][:-1]),
    lambda pk: pk["pk_commitments"].__setitem__(0, ["zz", "00"]),
    lambda pk: pk.update(pk_commitments="not a table"),
])
def test_material_id_fails_closed(keypair, mutate):
    pk = copy.deepcopy(keypair["public_key"])
    mutate(pk)
    with pytest.raises(ots.WaveLockOTSError):
        ots.ots_material_id(pk)


# ---------------------------------------------------------------------------
# Acceptance paths
# ---------------------------------------------------------------------------

def test_ledger_rejects_relabelled_material(tmp_path, keypair):
    pk, sk = keypair["public_key"], keypair["secret_key"]
    sig = ots.sign_ots(sk, "pay alice 5", allow_reuse=True)
    ledger = PersistentOTSReplayLedger(path=str(tmp_path / "replay.jsonl"))
    assert ledger.accept(pk, "pay alice 5", sig)
    for fields in ({"one_time_key_id": "observer"},
                   {"one_time_key_id": "observer-2", "psi_commitment": "cd" * 32}):
        relabelled, signature = _refingerprint(dict(pk, **fields), sig)
        assert ots.verify_ots(relabelled, "pay alice 5", signature)
        assert not ledger.accept(relabelled, "pay alice 5", signature)
        assert ledger.is_consumed(signature, public_key=relabelled)
        assert not ledger.is_consumed(signature)  # its labels alone are fresh
    record = json.loads(open(tmp_path / "replay.jsonl", encoding="utf-8").read().splitlines()[0])
    assert record["v"] == LEDGER_VERSION == 2
    assert record["material_id"] == ots.ots_material_id(pk)


def test_in_memory_reference_ledger_consumes_material(keypair):
    pk, sk = keypair["public_key"], keypair["secret_key"]
    sig = ots.sign_ots(sk, "m", allow_reuse=True)
    ledger = ots.OTSReplayLedger()
    assert ledger.accept(pk, "m", sig)
    relabelled, signature = _refingerprint(dict(pk, one_time_key_id="observer"), sig)
    assert not ledger.accept(relabelled, "m", signature)


def test_version_1_records_still_load_and_new_records_are_version_2(tmp_path, keypair):
    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps({"v": 1, "one_time_key_id": "old-id", "leaf_id": "a" * 64,
                                "transcript": "b" * 64}) + "\n", encoding="utf-8")
    ledger = PersistentOTSReplayLedger(path=str(path))
    signature = {"one_time_key_id": "old-id", "public_key_fingerprint": "c" * 64}
    assert ledger.is_consumed(signature)
    sig = ots.sign_ots(keypair["secret_key"], "v2", allow_reuse=True)
    assert ledger.accept(keypair["public_key"], "v2", sig)
    versions = [json.loads(line)["v"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert versions == [1, 2]
    assert PersistentOTSReplayLedger(path=str(path)).is_consumed(sig, public_key=keypair["public_key"])


@pytest.mark.parametrize("record", [
    {"v": 2, "one_time_key_id": "x", "leaf_id": "a" * 64, "transcript": "b" * 64},
    {"v": 2, "one_time_key_id": "x", "leaf_id": "a" * 64, "material_id": "short", "transcript": "b" * 64},
    {"v": 1, "one_time_key_id": "x", "leaf_id": "a" * 64, "material_id": "c" * 64, "transcript": "b" * 64},
    {"v": 3, "one_time_key_id": "x", "leaf_id": "a" * 64, "material_id": "c" * 64, "transcript": "b" * 64},
])
def test_malformed_versioned_records_fail_closed(tmp_path, record):
    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(OTSLedgerError):
        PersistentOTSReplayLedger(path=str(path))


def _material_reuse_chain(keypair):
    """A chain PR #47 accepted: two blocks over one commitment table under different key ids."""
    pk, sk = keypair["public_key"], keypair["secret_key"]
    first = build_signed_ots_block(sk, pk, ["first"], difficulty=0, allow_reuse=True)
    relabelled, _ = _refingerprint(dict(pk, one_time_key_id="second-id"))
    second = _holder_block(relabelled, sk, ["second"], 2, first.hash)
    assert verify_ots_block(first) and verify_ots_block(second)
    return [first, second]


def test_chain_validator_rejects_material_reuse(keypair):
    blocks = _material_reuse_chain(keypair)
    assert verify_ots_chain(blocks[:1])
    assert not verify_ots_chain(blocks)


def test_reload_of_material_reuse_chain_fails_closed(tmp_path, monkeypatch, keypair):
    blocks = _material_reuse_chain(keypair)
    monkeypatch.setattr(server, "load_all_blocks", lambda: blocks)
    monkeypatch.setattr(server, "CONSENSUS_OTS_LEDGER",
                        PersistentOTSReplayLedger(path=str(tmp_path / "replay.jsonl")))
    with pytest.raises(OTSLedgerError):
        server.ChainState().load_from_disk(require_ots=True)


def test_reconstruction_recovers_material_without_side_ledger(tmp_path, keypair):
    first, second = _material_reuse_chain(keypair)
    ledger = PersistentOTSReplayLedger(path=str(tmp_path / "replay.jsonl"))
    server._reconstruct_consumed_ots([first], ledger)
    auth = second.meta["ots_auth"]
    assert not ledger.is_consumed(auth["signature"])  # fresh labels
    assert ledger.is_consumed(auth["signature"], public_key=auth["public_key"])
    assert not ledger.accept(auth["public_key"], auth["message"], auth["signature"])


def test_reconstruction_fails_closed_on_unidentifiable_table(tmp_path, keypair):
    block = build_signed_ots_block(keypair["secret_key"], keypair["public_key"], ["x"],
                                   difficulty=0, allow_reuse=True)
    meta = copy.deepcopy(block.meta)
    meta["ots_auth"]["public_key"]["merkle_root"] = "00" * 32
    broken = Block(index=1, messages=block.messages, previous_hash=block.previous_hash,
                   difficulty=0, block_type="OTS", meta=meta, timestamp=block.timestamp)
    ledger = PersistentOTSReplayLedger(path=str(tmp_path / "replay.jsonl"))
    with pytest.raises(OTSLedgerError):
        server._reconstruct_consumed_ots([broken], ledger)


# ---------------------------------------------------------------------------
# Boundary: the guarantee is per public commitment table, not per secret value
# ---------------------------------------------------------------------------

def _table_from_secret_values(keypair, params_hash_hex, kid):
    """A deliberately non-standard key: the *same* secret slice values committed under another
    params_hash. Honest key generation derives the slices from params_hash, so this needs custom
    code by the secret holder."""
    sk = keypair["secret_key"]
    seed = bytes.fromhex(sk["seed_hex"])
    psi = bytes.fromhex(sk["psi_commitment"])
    true_p = bytes.fromhex(sk["params_hash"])
    new_p = bytes.fromhex(params_hash_hex)
    secrets_ = [[ots._secret_slice(seed, psi, true_p, i, b) for b in (0, 1)] for i in range(256)]
    table = [[ots._public_slice(new_p, i, b, secrets_[i][b]).hex() for b in (0, 1)] for i in range(256)]
    pk = dict(keypair["public_key"], params_hash=params_hash_hex, one_time_key_id=kid,
              pk_commitments=table, merkle_root=ots._merkle_root_from_commitments(table).hex())
    pk["public_key_fingerprint"] = ots.public_key_fingerprint(pk)

    def sign(message):
        digest = ots._message_digest(message, new_p)
        bits = ots._digest_bits(digest)
        return {"scheme": ots.SCHEME, "version": ots.VERSION, "hash_alg": ots.HASH_ALG,
                "one_time_key_id": kid, "public_key_fingerprint": pk["public_key_fingerprint"],
                "params_hash": params_hash_hex, "psi_commitment": pk["psi_commitment"],
                "message_digest": digest.hex(),
                "revealed_slices": [secrets_[i][bit].hex() for i, bit in enumerate(bits)]}
    return pk, sign


def test_boundary_secret_values_under_a_new_table(tmp_path, keypair):
    """A holder who recommits the same secret values under another params_hash gets a new public
    table, hence a new material id: that second use is accepted and exposes secret values. It
    cannot authorise anything under the first table, which stays consumed."""
    pk, sk = keypair["public_key"], keypair["secret_key"]
    ledger = PersistentOTSReplayLedger(path=str(tmp_path / "replay.jsonl"))
    first = ots.sign_ots(sk, "first", allow_reuse=True)
    assert ledger.accept(pk, "first", first)
    pk2, sign2 = _table_from_secret_values(keypair, "ee" * 32, "recommitted")
    second = sign2("second")
    assert ots.verify_ots(pk2, "second", second)
    assert ots.ots_material_id(pk2) != ots.ots_material_id(pk)
    assert ledger.accept(pk2, "second", second)  # a different public table: not detectable
    shared = set(first["revealed_slices"]) & set(second["revealed_slices"])
    assert shared, "the two signatures reveal some of the same secret values"
    # The first table remains single use under any label.
    relabelled, again = _refingerprint(dict(pk, one_time_key_id="third"),
                                       ots.sign_ots(sk, "third", allow_reuse=True))
    assert not ledger.accept(relabelled, "third", again)
