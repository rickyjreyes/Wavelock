import os
import tempfile

from chain.CurvaChain import CurvaChain
from chain.kernel_decl import add_kernel_decl_block, spec_hash
from chain.WaveLock import _kernel_hash, KERNEL_VERSION


def test_spec_hash_changes_with_content():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "spec.md")
        with open(path, "w") as f:
            f.write("v1")
        h1 = spec_hash(path)

        with open(path, "w") as f:
            f.write("v2")
        h2 = spec_hash(path)

        assert h1 != h2


def test_add_kernel_decl_block_sets_meta_fields():
    chain = CurvaChain(difficulty=2)

    # Use a small temp spec file for hashing
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = os.path.join(tmpdir, "spec.md")
        with open(spec_path, "w") as f:
            f.write("WaveLock spec")

        blk = add_kernel_decl_block(
            chain,
            spec_path=spec_path,
            declared_by="ricky",
        )

    assert blk.block_type == "KERNEL_DECL"
    assert blk.meta["kernel_version"] == KERNEL_VERSION
    assert blk.meta["kernel_hash"] == _kernel_hash()
    assert blk.meta["declared_by"] == "ricky"
    assert "spec_hash" in blk.meta
