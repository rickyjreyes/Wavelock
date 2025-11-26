import sys
import os

# Full path to the directory containing chain_utils.py
CHAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, CHAIN_PATH)

# Now safely import modules
from chain.chain_utils import load_all_blocks, verify_merkle_root

blocks = load_all_blocks()
for b in blocks:
    assert verify_merkle_root(b), f"❌ Merkle root invalid for Block {b.index}"
print("✅ All Merkle roots valid")
