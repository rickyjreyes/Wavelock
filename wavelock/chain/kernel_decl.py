from __future__ import annotations
from typing import Optional
import hashlib
import os

from wavelock.chain.CurvaChain import CurvaChain
from wavelock.chain.Block import Block
from wavelock.chain.WaveLock import KERNEL_VERSION, _kernel_hash


def spec_hash(path: str) -> str:
    """
    Compute a SHA-256 hash of the WaveLock spec file.

    Typical candidates:
    - MASTER_EQUATIONS.md
    - SPEC_WAVELOCK.md
    - README.md (WaveLock section)
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def add_kernel_decl_block(
    chain: CurvaChain,
    spec_path: str = "MASTER_EQUATIONS.md",
    declared_by: str = "local",
) -> Block:
    """
    Append a KERNEL_DECL block to the chain, binding:

    - kernel_version
    - kernel_hash   (from WaveLock)
    - spec_hash     (from the provided spec_path)
    - declared_by   (user or node id)
    """
    k_hash = _kernel_hash()
    s_hash = spec_hash(spec_path)

    meta = {
        "kernel_version": KERNEL_VERSION,
        "kernel_hash": k_hash,
        "spec_path": spec_path,
        "spec_hash": s_hash,
        "declared_by": declared_by,
    }

    chain.add_block(
        messages=["KERNEL_DECL"],
        block_type="KERNEL_DECL",
        meta=meta,
    )
    return chain.get_latest_block()
