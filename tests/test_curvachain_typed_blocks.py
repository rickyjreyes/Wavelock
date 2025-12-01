from wavelock.chain.CurvaChain import CurvaChain
from wavelock.chain.Block import Block


def test_typed_blocks_and_roundtrip():
    chain = CurvaChain(difficulty=3)

    # 1) add a generic block
    meta_research = {"artifact_id": "hash://abc123", "author": "ricky"}
    chain.add_block(
        messages=["first research note"],
        block_type="RESEARCH_TX",
        meta=meta_research,
    )

    # 2) add a kernel declaration block
    meta_kernel = {
        "kernel_version": "WL-psi-001",
        "kernel_hash": "deadbeef" * 8,
        "spec_hash": "cafebabe" * 8,
        "declared_by": "ricky",
    }
    chain.add_block(
        messages=["KERNEL_DECL"],
        block_type="KERNEL_DECL",
        meta=meta_kernel,
    )

    # 3) validate chain
    assert chain.is_chain_valid()

    # 4) typed block fields are correct
    b0 = chain.chain[0]
    b1 = chain.chain[1]
    b2 = chain.chain[2]

    assert b0.block_type in ("GENESIS", "GENERIC")  # depending on how you set genesis
    assert b1.block_type == "RESEARCH_TX"
    assert b1.meta == meta_research

    assert b2.block_type == "KERNEL_DECL"
    assert b2.meta == meta_kernel

    # 5) test to_dict / from_dict round-trip on a typed block
    original = b1
    data = original.to_dict()
    restored = Block.from_dict(data)

    assert restored.block_type == original.block_type
    assert restored.meta == original.meta
    assert restored.hash == original.hash
    assert restored.hash == restored.calculate_hash(restored.nonce)
