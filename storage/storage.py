import os
import json
from typing import List
from Block import Block
from CurvaChain import CurvaChain


LEDGER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger")
BLOCK_FILE_TEMPLATE = "blk{:05d}.jsonl"
INDEX_FILE = "index.json"

def ensure_ledger_dir():
    if not os.path.exists(LEDGER_DIR):
        os.makedirs(LEDGER_DIR)

def get_block_file_path(file_number: int) -> str:
    return os.path.join(LEDGER_DIR, BLOCK_FILE_TEMPLATE.format(file_number))

def append_block(block: Block):
    ensure_ledger_dir()
    index_path = os.path.join(LEDGER_DIR, INDEX_FILE)

    # Load or create index
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
    else:
        index = {"file_number": 0, "block_count": 0}

    block_data = {
        "index": block.index,
        "timestamp": block.timestamp,
        "messages": block.messages,
        "previous_hash": block.previous_hash,
        "nonce": block.nonce,
        "hash": block.hash,
        "difficulty": block.difficulty,
        "merkle_root": block.merkle_root,
    }

    # Append to ledger file
    file_path = get_block_file_path(index["file_number"])
    with open(file_path, "a") as f:
        f.write(json.dumps(block_data) + "\n")

    # Rotate files every 100 blocks
    index["block_count"] += 1
    if index["block_count"] % 100 == 0:
        index["file_number"] += 1

    with open(index_path, "w") as f:
        json.dump(index, f)

    print(f"📦 Block {block.index} appended to {file_path}")

def load_all_blocks() -> List[Block]:
    ensure_ledger_dir()
    blocks: List[Block] = []
    for filename in sorted(os.listdir(LEDGER_DIR)):
        if filename.startswith("blk") and filename.endswith(".jsonl"):
            with open(os.path.join(LEDGER_DIR, filename), "r") as f:
                for line in f:
                    data = json.loads(line)
                    block = type("Block", (), {})()
                    block.__dict__.update(data)
                    blocks.append(block)
    print(f"📂 Loaded {len(blocks)} blocks from ledger/")
    return blocks

def load_chain() -> CurvaChain:
    blocks = load_all_blocks()
    if not blocks:
        return CurvaChain()
    chain = CurvaChain(difficulty=blocks[0].difficulty)
    chain.chain = blocks
    return chain
