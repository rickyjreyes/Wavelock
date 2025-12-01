import cupy as cp

from wavelock.chain.CurvaChain import CurvaChain
from wavelock.chain.WaveLock import CurvatureKeyPair
from wavelock.chain.pov import (
    make_pov_record,
    add_pov_block,
    recompute_reward_from_meta,
    select_validator,
)


def test_pov_record_and_block_roundtrip():
    chain = CurvaChain(difficulty=2)
    kp = CurvatureKeyPair(n=4, seed=123)

    record = make_pov_record(
        validator_id="validator-1",
        target_block_index=0,
        ok=True,
        psi_star=kp.psi_star,
    )

    blk = add_pov_block(chain, record)

    assert blk.block_type == "VERIFICATION_TX"
    meta = blk.meta
    assert meta["validator_id"] == "validator-1"
    assert meta["target_block_index"] == 0
    assert meta["ok"] is True

    # reward in meta matches recomputed reward
    expected_reward = recompute_reward_from_meta(meta)
    assert meta["reward"] == expected_reward


def test_select_validator_deterministic():
    peers = ["a", "b", "c", "d"]
    seed = "somehashvalue"

    v1 = select_validator(peers, seed)
    v2 = select_validator(peers, seed)

    assert v1 in peers
    assert v1 == v2  # deterministic for same seed
