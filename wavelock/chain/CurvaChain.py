from wavelock.chain.Block import Block
from typing import List, Optional, Dict, Any

class CurvaChain:
    def __init__(self, difficulty: int = 4):
        self.difficulty = difficulty
        self.chain: List[Block] = [self.create_genesis_block()]

    def create_genesis_block(self) -> Block:
        # Explicitly mark genesis
        return Block(
            index=0,
            messages=["Genesis Block"],
            previous_hash="0" * 64,
            difficulty=self.difficulty,
            block_type="GENESIS",
            meta={},
        )

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def add_block(
        self,
        messages: List[str],
        block_type: str = "GENERIC",
        meta: Optional[Dict[str, Any]] = None,
    ):
        previous_hash = self.get_latest_block().hash
        new_block = Block(
            index=len(self.chain),
            messages=messages,
            previous_hash=previous_hash,
            difficulty=self.difficulty,
            block_type=block_type,
            meta=meta,
        )
        self.chain.append(new_block)

    def is_chain_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i - 1]

            # Recalculate hash with the stored nonce and compare
            if current.hash != current.calculate_hash(current.nonce):
                return False

            # Check linkage
            if current.previous_hash != prev.hash:
                return False

            # Check difficulty
            if not current.hash.startswith('0' * self.difficulty):
                return False

        return True
