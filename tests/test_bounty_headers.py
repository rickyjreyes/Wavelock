"""Versioned correction for a concrete v1 header concatenation ambiguity."""
import copy

import pytest

from wavelock.chain.Block import Block
from wavelock.chain.config import Config
from wavelock.chain.ots_blocks import (
    build_signed_ots_block, canonical_ots_block_digest, verify_ots_chain,
    OTS_BLOCK_TRANSCRIPT_VERSION,
)
from wavelock.crypto.wavelock_ots import generate_ots_keypair


def header(index, timestamp, version):
    return Block(index, ["same"], "0"*64, difficulty=0, timestamp=timestamp,
                 nonce=0, block_hash="0"*64, header_version=version)


def test_confirmed_legacy_ambiguity_and_v2_separation():
    a, b = header(1, "23", 1), header(12, "3", 1)
    assert a.calculate_hash(0) == b.calculate_hash(0)
    assert a.calculate_hash(0) == "4914a9e6406485270303bdc82222e8a2c57a74f99bfd4d5ea4df88ce4d2785e8"
    a.header_version = b.header_version = 2
    assert a.calculate_hash(0) != b.calculate_hash(0)


def test_reduced_exhaustive_header_tuples():
    legacy, canonical = set(), set()
    for index in range(1, 41):
        for time in range(1, 41):
            legacy.add(header(index, str(time), 1).calculate_hash(0))
            canonical.add(header(index, str(time), 2).calculate_hash(0))
    assert len(legacy) < 1600
    assert len(canonical) == 1600


def test_legacy_storage_is_explicitly_readable():
    block = header(1, "23", 1)
    block.hash = block.calculate_hash(0)
    wire = block.to_dict()
    del wire["header_version"]
    loaded = Block.from_dict(wire)
    assert loaded.header_version == 1
    assert loaded.calculate_hash(loaded.nonce) == block.hash


@pytest.mark.parametrize("version", [0, 3, True, "2", None])
def test_unknown_header_versions_rejected(version):
    wire = header(1, "23", 2).to_dict()
    wire["header_version"] = version
    with pytest.raises(ValueError):
        Block.from_dict(wire)


def test_ots_v1_unchanged_and_downgraded_header_not_accepted(monkeypatch):
    from wavelock.network import server
    kp = generate_ots_keypair()
    block = build_signed_ots_block(kp["secret_key"], kp["public_key"], ["same"], difficulty=0)
    assert OTS_BLOCK_TRANSCRIPT_VERSION == 1
    digest = canonical_ots_block_digest(block)
    legacy = copy.deepcopy(block)
    legacy.header_version = 1
    legacy.hash = legacy.calculate_hash(legacy.nonce)
    assert canonical_ots_block_digest(legacy) == digest
    assert not verify_ots_chain([legacy])  # header v2 context cannot be downgraded
    historical_key = generate_ots_keypair()
    historical = build_signed_ots_block(historical_key["secret_key"], historical_key["public_key"],
                                        ["historical"], difficulty=0, header_version=1)
    assert verify_ots_chain([historical])
    monkeypatch.setattr(server, "CHAIN", server.ChainState())
    monkeypatch.setattr(server, "_verify_ots_block", lambda *a: pytest.fail("downgrade reached key consumption"))
    assert not server.try_accept_block(legacy, Config(pow_target="f"*64))


def test_remining_header_cannot_mutate_signed_context():
    kp = generate_ots_keypair()
    block = build_signed_ots_block(kp["secret_key"], kp["public_key"], ["same"], difficulty=0)
    block.timestamp = "999"
    block.hash = block.calculate_hash(block.nonce)
    assert not verify_ots_chain([block])


def test_external_tip_detects_truncation_and_unanchored_limit_is_explicit():
    blocks = []
    for index in (1, 2):
        kp = generate_ots_keypair()
        blocks.append(build_signed_ots_block(kp["secret_key"], kp["public_key"], [str(index)],
                      index=index, previous_hash=blocks[-1].hash if blocks else "0"*64, difficulty=0))
    assert verify_ots_chain(blocks, expected_tip=blocks[-1].hash)
    assert not verify_ots_chain(blocks[:-1], expected_tip=blocks[-1].hash)
    assert verify_ots_chain(blocks[:-1])  # a prefix alone cannot prove completeness
