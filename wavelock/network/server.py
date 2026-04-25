# network/server.py
from __future__ import annotations
import os, socket, threading, time, json, traceback
from typing import List, Tuple, Dict, Optional

import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np

# --- repo imports ---
from wavelock.chain.config import load_config
from wavelock.network.peer_utils import load_peers, save_peers, add_peer, random_peers
from wavelock.network.protocol import (
    encode_message, decode_message,
    INV, GET_BLOCK, SEND_BLOCKS, GET_PEERS, SEND_PEERS,
    GET_CHAIN, SEND_BLOCK,
)
from wavelock.chain.chain_utils import load_all_blocks, save_block_to_disk
from wavelock.chain.Block import Block
from wavelock.chain.WaveLock import (
    CurvatureKeyPair, SCHEMA_V2, _serialize_commitment_v2, _canonical_json,
)

HOST = "0.0.0.0"

###############################################################################
# In-memory chain index + helpers
###############################################################################

class ChainState:
    def __init__(self):
        self._lock = threading.RLock()
        self.blocks: List[Block] = []
        self.by_hash: Dict[str, Block] = {}

    def load_from_disk(self):
        with self._lock:
            self.blocks = load_all_blocks()
            self.by_hash = {b.hash: b for b in self.blocks}

    def tip(self) -> Optional[Block]:
        with self._lock:
            return self.blocks[-1] if self.blocks else None

    def has_hash(self, h: str) -> bool:
        with self._lock:
            return h in self.by_hash

    def get_by_hash(self, h: str) -> Optional[Block]:
        with self._lock:
            return self.by_hash.get(h)

    def append(self, b: Block):
        with self._lock:
            save_block_to_disk(b)
            self.blocks.append(b)
            self.by_hash[b.hash] = b

CHAIN = ChainState()

###############################################################################
# Trust / strict verification
###############################################################################

def _load_trusted_commitments(path: str = "trusted_commitments.json") -> List[str]:
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path, "r"))
    except Exception:
        return []

def _commitment_is_trusted(commitment: str, trusted: List[str]) -> bool:
    return str(commitment) in set(trusted)

def _load_published_psi(commitment: str):
    key = commitment.replace(":", "_").lower()
    npz_path = os.path.join("commitments", f"{key}.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=False)
    psi = data["psi_star"]
    return cp.asarray(psi)

def _extract_curvature_fields(b: Block) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse message/signature/commitment from b.messages[] lines."""
    msg = sig = com = None
    for line in getattr(b, "messages", []) or []:
        if isinstance(line, str):
            if line.startswith("message: "):
                msg = line[len("message: "):]
            elif line.startswith("signature: "):
                sig = line[len("signature: "):]
            elif line.startswith("commitment: "):
                com = line[len("commitment: "):].strip()
    return msg, sig, com

def _verify_pow_and_linkage(b: Block, cfg) -> bool:
    tip = CHAIN.tip()
    expected_prev = tip.hash if tip else "0" * 64
    if b.previous_hash != expected_prev:
        print(f"Reject: prev_hash mismatch (got {b.previous_hash[:12]}..., expected {expected_prev[:12]}...)")
        return False
    try:
        target_hex = cfg.pow_target
        ok = int(b.hash, 16) <= int(target_hex, 16)
    except Exception:
        zeros = len(cfg.pow_target) - len(cfg.pow_target.lstrip("0"))
        ok = b.hash.startswith("0" * max(1, zeros))
    if not ok:
        print(f"Reject: PoW target not met ({b.hash[:12]}... > target)")
    return ok

def _strict_verify_curvature(b: Block, cfg) -> bool:
    trusted = _load_trusted_commitments()
    msg, sig, com = _extract_curvature_fields(b)

    if not com or not _commitment_is_trusted(com, trusted):
        print("Reject: commitment missing or not trusted.")
        return False

    if not cfg.require_full_verify:
        return True

    psi = _load_published_psi(com)
    if psi is None:
        if bool(int(os.getenv("WAVELOCK_REJECT_IF_UNPUBLISHED", "0"))):
            print("Reject: psi* not published for strict mode.")
            return False
        else:
            print("Strict: psi* not published, allowing due to policy.")
            return True

    kp = CurvatureKeyPair(n=4, test_mode=True)
    kp.psi_star = psi
    kp.psi_0 = cp.zeros_like(psi)
    kp.commitment = com

    msg_ok = (msg is not None) and (sig is not None)
    if not msg_ok:
        print("Reject: missing curvature message/signature fields.")
        return False

    if not kp.verify(msg, sig):
        print("Reject: curvature signature invalid.")
        return False

    return True

###############################################################################
# P2P: dedupe + broadcast
###############################################################################

_seen_hashes: Dict[str, float] = {}

def _dedupe_seen(h: str, ttl=300) -> bool:
    now = time.time()
    for k, t in list(_seen_hashes.items()):
        if now - t > ttl:
            _seen_hashes.pop(k, None)
    if h in _seen_hashes:
        return True
    _seen_hashes[h] = now
    return False

def broadcast_inv(block_hash: str):
    if _dedupe_seen(block_hash):
        return
    for host, port in load_peers():
        try:
            with socket.socket() as s:
                s.settimeout(1.0)
                s.connect((host, int(port)))
                s.sendall(encode_message(INV, [block_hash]))
        except Exception:
            pass

###############################################################################
# Accept/validate incoming blocks
###############################################################################

def has_block(h: str) -> bool:
    return CHAIN.has_hash(h)

def get_block_by_hash(h: str) -> Optional[Block]:
    return CHAIN.get_by_hash(h)

def _block_from_dict(d: dict) -> Block:
    """Reconstruct Block from a dict payload, using canonical field names."""
    return Block.from_dict(d)

def try_accept_block_dict(d: dict, cfg) -> bool:
    b = _block_from_dict(d)
    return try_accept_block(b, cfg)

def try_accept_block(b: Block, cfg) -> bool:
    if not _verify_pow_and_linkage(b, cfg):
        return False
    if not _strict_verify_curvature(b, cfg):
        return False
    CHAIN.append(b)
    print(f"Accepted Block #{b.index} | {b.hash[:12]}...")
    broadcast_inv(b.hash)
    return True

###############################################################################
# Connection handler
###############################################################################

def handle_client(conn: socket.socket, addr: Tuple[str, int], cfg):
    try:
        buf = b""
        conn.settimeout(10.0)
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
            # Try to decode the buffer as a complete JSON message
            try:
                opcode, data = decode_message(buf)
            except Exception:
                continue
            if opcode is None:
                continue
            # Successfully decoded — reset buffer
            buf = b""

            if opcode == INV:
                want = [h for h in data if not has_block(h)]
                if want:
                    conn.sendall(encode_message(GET_BLOCK, want))

            elif opcode == GET_BLOCK:
                blocks = []
                for h in data:
                    b = get_block_by_hash(h)
                    if b:
                        blocks.append(b.to_dict())
                if blocks:
                    conn.sendall(encode_message(SEND_BLOCKS, blocks))

            elif opcode == SEND_BLOCKS:
                for bd in data:
                    try:
                        if not has_block(bd.get("hash", "")):
                            try_accept_block_dict(bd, cfg)
                    except Exception as e:
                        print("Failed to accept remote block:", e)

            elif opcode == GET_PEERS:
                plist = [f"{h}:{p}" for (h, p) in load_peers()]
                conn.sendall(encode_message(SEND_PEERS, plist))

            elif opcode == SEND_PEERS:
                for ent in data:
                    try:
                        h, p = ent.split(":")
                        add_peer(h, int(p))
                    except Exception:
                        pass

            elif opcode == GET_CHAIN:
                ser = [b.to_dict() for b in CHAIN.blocks]
                conn.sendall(encode_message(SEND_BLOCK, ser))

    except socket.timeout:
        pass
    except Exception as e:
        print("handler error:", e)
        traceback.print_exc()
    finally:
        try:
            conn.close()
        except Exception:
            pass

###############################################################################
# Server main
###############################################################################

def main():
    CHAIN.load_from_disk()
    cfg = load_config(os.getenv("WAVELOCK_CONFIG"))

    trusted = _load_trusted_commitments()
    print(f"Trusted commitments loaded: {len(trusted)}")
    print(f"WaveLock P2P server listening on port {cfg.port}")
    print(f"  Strict curvature verify: {'ON' if cfg.require_full_verify else 'OFF'}")
    reject_unpub = bool(int(os.getenv("WAVELOCK_REJECT_IF_UNPUBLISHED", "0")))
    print(f"  Reject if psi* unpublished: {'ON' if reject_unpub else 'OFF'}")

    for seed in cfg.seeds or []:
        try:
            host, port = seed.split(":")
            add_peer(host, int(port))
        except Exception:
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, cfg.port))
        srv.listen(64)
        while True:
            try:
                conn, addr = srv.accept()
                try:
                    add_peer(addr[0], cfg.port)
                except Exception:
                    pass
                threading.Thread(target=handle_client, args=(conn, addr, cfg), daemon=True).start()
            except KeyboardInterrupt:
                print("\nServer shutdown requested.")
                break
            except Exception as e:
                print("accept() error:", e)
                time.sleep(0.2)

if __name__ == "__main__":
    main()
