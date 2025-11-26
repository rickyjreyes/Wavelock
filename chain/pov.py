from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List

import hashlib

from chain.CurvaChain import CurvaChain
from chain.Block import Block
from chain.WaveLock import (
    classify_wcc_run,
    wcc_avg_curvature_budget,
    _steps,
)


@dataclass
class PoVRecord:
    """
    Proof-of-Verification record tied to a specific ψ* validation task.

    - validator_id: ID of the validator node / user
    - target_block_index: which block (or artifact) was being verified
    - ok: True if verification succeeded, False otherwise
    - wcc_class: "PWCC", "NPWCC", or "OUT_OF_WCC"
    - avg_c_theta: average curvature budget per lattice site
    - reward: integer reward units (WLC proto-units)
    """
    validator_id: str
    target_block_index: int
    ok: bool
    wcc_class: str
    avg_c_theta: float
    reward: int


def compute_pov_reward(wcc_class: str, avg_c_theta: float) -> int:
    """
    Simple model-relative reward rule.

    This is deliberately conservative and easily tweakable:

    - PWCC    → 10 units
    - NPWCC   → 5 units
    - other   → 0 units
    """
    if wcc_class == "PWCC":
        return 10
    if wcc_class == "NPWCC":
        return 5
    return 0


def make_pov_record(
    validator_id: str,
    target_block_index: int,
    ok: bool,
    psi_star,
) -> PoVRecord:
    """
    Build a PoVRecord from a ψ* configuration and WCC rails.
    """
    wcc_class = classify_wcc_run(_steps, psi_star)
    avg_c = wcc_avg_curvature_budget(psi_star)
    reward = compute_pov_reward(wcc_class, avg_c)
    return PoVRecord(
        validator_id=validator_id,
        target_block_index=target_block_index,
        ok=ok,
        wcc_class=wcc_class,
        avg_c_theta=avg_c,
        reward=reward,
    )


def add_pov_block(chain: CurvaChain, record: PoVRecord) -> Block:
    """
    Append a VERIFICATION_TX block carrying a PoVRecord.

    The message encodes the verification intent; meta encodes all PoV data.
    """
    msg = f"VERIFICATION_TX:{record.target_block_index}:{int(record.ok)}"
    meta = asdict(record)

    chain.add_block(
        messages=[msg],
        block_type="VERIFICATION_TX",
        meta=meta,
    )
    return chain.get_latest_block()


def recompute_reward_from_meta(meta: dict) -> int:
    """
    Recompute the expected reward from a block's meta and compare.

    This lets you audit that a VERIFICATION_TX block hasn't lied about reward.
    """
    wcc_class = meta.get("wcc_class", "OUT_OF_WCC")
    avg_c = float(meta.get("avg_c_theta", 0.0))
    return compute_pov_reward(wcc_class, avg_c)


def select_validator(peers: List[str], seed: str) -> str:
    """
    Deterministic validator selection from a peer list.

    - peers: list of peer IDs (e.g., user IDs, node IDs)
    - seed:  string seed (e.g., last block hash)

    Returns a single selected validator_id.
    """
    if not peers:
        raise ValueError("no peers available for validator selection")

    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    idx = int(h, 16) % len(peers)
    return peers[idx]
