import hashlib
import time
from typing import List, Optional, Dict, Any
import json

class Block:
    def __init__(
        self,
        index: int,
        messages: List[str],
        previous_hash: str,
        difficulty: int = 4,
        timestamp: Optional[str] = None,      # 🔥 FIXED: always store as string
        nonce: Optional[int] = None,
        block_hash: Optional[str] = None,
        merkle_root: Optional[str] = None,
        block_type: str = "GENERIC",
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.index = index

        # 🔥 FIX #1 — timestamp must never change; never use "or" fallback
        if timestamp is None:
            self.timestamp = str(time.time())
        else:
            self.timestamp = str(timestamp)

        self.messages = messages
        self.previous_hash = previous_hash
        self.difficulty = difficulty
        self.block_type = block_type

        # 🔥 FIX #2 — meta must be JSON-stable
        self.meta = meta if meta is not None else {}
        self.meta = json.loads(json.dumps(self.meta, sort_keys=True))

        self.merkle_root = merkle_root or self.calculate_merkle_root()

        if nonce is not None and block_hash is not None:
            # Rehydrating from disk
            self.nonce = nonce
            self.hash = block_hash
        else:
            # New block — mine it
            self.nonce, self.hash = self.mine_block()

    def calculate_merkle_root(self) -> str:
        def hash_pair(a, b):
            return hashlib.sha256((a + b).encode()).hexdigest()

        hashes = [hashlib.sha256(msg.encode()).hexdigest() for msg in self.messages]
        if not hashes:
            return hashlib.sha256(b"").hexdigest()
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])
            hashes = [hash_pair(hashes[i], hashes[i+1]) for i in range(0, len(hashes), 2)]
        return hashes[0]

    def calculate_hash(self, nonce: int) -> str:

        # 🔥 FIX #3 — JSON stable meta
        stable_meta = json.dumps(self.meta, sort_keys=True)

        block_content = (
            f"{self.index}"
            f"{self.timestamp}"
            f"{self.previous_hash}"
            f"{self.merkle_root}"
            f"{self.difficulty}"
            f"{self.block_type}"
            f"{stable_meta}"
            f"{nonce}"
        )
        return hashlib.sha256(block_content.encode()).hexdigest()

    def mine_block(self):
        nonce = 0
        while True:
            hash_result = self.calculate_hash(nonce)
            if hash_result.startswith('0' * self.difficulty):
                return nonce, hash_result
            nonce += 1

    def to_dict(self):
        return {
            "index": self.index,
            "messages": self.messages,
            "previous_hash": self.previous_hash,
            "difficulty": self.difficulty,
            "timestamp": self.timestamp,               # stored as string
            "nonce": self.nonce,
            "hash": self.hash,
            "merkle_root": self.merkle_root,
            "block_type": self.block_type,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(data):
        return Block(
            index=data["index"],
            messages=data["messages"],
            previous_hash=data["previous_hash"],
            difficulty=data.get("difficulty", 4),
            timestamp=data["timestamp"],               # always string
            nonce=data["nonce"],
            block_hash=data["hash"],
            merkle_root=data.get("merkle_root"),
            block_type=data.get("block_type", "GENERIC"),
            meta=data.get("meta", {}),
        )
