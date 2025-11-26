# network/server.py
from __future__ import annotations
import os, socket, threading, time, json, traceback
from typing import List, Tuple, Dict, Optional

# --- repo imports (existing) ---
from chain.config import load_config
from .peer_utils import load_peers, save_peers, add_peer, random_peers
from network.protocol import encode_message, decode_message, INV, GET_BLOCK, SEND_BLOCKS, GET_PEERS, SEND_PEERS, GET_CHAIN, SEND_BLOCK
from chain.chain_utils import load_all_blocks, save_block_to_disk
from chain.Block import Block  # your Block class (index, prev_hash, hash, nonce, messages, ...)

# --- curvature / trust imports ---
import numpy as np
try:
    import cupy as cp
except Exception:
    cp = None  # if GPU not available, strict verify will be disabled gracefully

from chain.WaveLock import CurvatureKeyPair, SCHEMA_V2, _serialize_commitment_v2, _commit_header, _canonical_json

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
            save_block_to_disk(b)               # persist
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

def _load_published_psi(commitment: str) -> Optional[cp.ndarray]:
    if cp is None:
        return None
    # commitments/wlv2_<hex>.npz
    key = commitment.replace(":", "_").lower()
    npz_path = os.path.join("commitments", f"{key}.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=False)
    psi = data["psi_star"]
    return cp.asarray(psi)

def _extract_curvature_fields(b: Block) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse message/signature/commitment from b.messages[] lines"""
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
    # prev linkage
    tip = CHAIN.tip()
    expected_prev = tip.hash if tip else "0"*64
    if b.prev_hash != expected_prev:
        print(f"❌ Reject: prev_hash mismatch (got {b.prev_hash[:12]}..., expected {expected_prev[:12]}...)")
        return False
    # simple PoW: numeric hash <= target
    try:
        target_hex = cfg.pow_target
        ok = int(b.hash, 16) <= int(target_hex, 16)
    except Exception:
        # fallback: leading zero-prefix check length
        zeros = len(cfg.pow_target) - len(cfg.pow_target.lstrip("0"))
        ok = b.hash.startswith("0"*max(1, zeros))
    if not ok:
        print(f"❌ Reject: PoW target not met ({b.hash[:12]}... > target)")
    return ok

def _strict_verify_curvature(b: Block, cfg) -> bool:
    trusted = _load_trusted_commitments()
    msg, sig, com = _extract_curvature_fields(b)

    # Commitment allow-list
    if not com or not _commitment_is_trusted(com, trusted):
        print("❌ Reject: commitment missing or not trusted.")
        return False

    # If strict verify disabled, allow after trust check
    if not cfg.require_full_verify:
        return True

    # Strict mode ON: must have published ψ*
    psi = _load_published_psi(com)
    if psi is None:
        if bool(int(os.getenv("WAVELOCK_REJECT_IF_UNPUBLISHED", "0"))):
            print("❌ Reject: ψ* not published for strict mode.")
            return False
        else:
            print("⚠️ Strict: ψ* not published, allowing due to policy (set WAVELOCK_REJECT_IF_UNPUBLISHED=1 to force reject).")
            return True

    if cp is None:
        print("⚠️ Strict requested but CuPy unavailable; allowing block after trust check.")
        return True

    # Build a verifier key from published ψ*
    kp = CurvatureKeyPair(n=4)  # n is irrelevant here; we overwrite fields
    kp.psi_star = psi
    kp.psi_0 = cp.zeros_like(psi)  # not used by verify
    kp.commitment = com

    msg_ok = (msg is not None) and (sig is not None)
    if not msg_ok:
        print("❌ Reject: missing curvature message/signature fields.")
        return False

    if not kp.verify(msg, sig):
        print("❌ Reject: curvature signature invalid (WLv2/SIGv2).")
        return False

    return True

###############################################################################
# P2P: dedupe + broadcast
###############################################################################

_seen_hashes: Dict[str, float] = {}

def _dedupe_seen(h: str, ttl=300) -> bool:
    now = time.time()
    # prune
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
            # silent network errors
            pass

###############################################################################
# Accept/validate incoming blocks
###############################################################################

def has_block(h: str) -> bool:
    return CHAIN.has_hash(h)

def get_block_by_hash(h: str) -> Optional[Block]:
    return CHAIN.get_by_hash(h)

def _block_from_dict(d: dict) -> Block:
    # Make a Block from a dict payload (attributes map 1:1)
    b = Block(
        index=d.get("index"),
        prev_hash=d.get("prev_hash"),
        merkle=d.get("merkle"),
        messages=d.get("messages"),
        nonce=d.get("nonce"),
        timestamp=d.get("timestamp"),
    )
    # Some Block implementations compute hash in ctor; preserve given hash if present
    if hasattr(b, "hash") and d.get("hash"):
        try:
            b.hash = d["hash"]
        except Exception:
            pass
    return b

def try_accept_block_dict(d: dict, cfg) -> bool:
    b = _block_from_dict(d)
    return try_accept_block(b, cfg)

def try_accept_block(b: Block, cfg) -> bool:
    # PoW + linkage
    if not _verify_pow_and_linkage(b, cfg):
        return False
    # Curvature trust/strict checks
    if not _strict_verify_curvature(b, cfg):
        return False
    # Accept & persist
    CHAIN.append(b)
    print(f"✅ Accepted Block #{b.index} | {b.hash[:12]}...")
    # Gossip
    broadcast_inv(b.hash)
    return True

###############################################################################
# Connection handler
###############################################################################

def handle_client(conn: socket.socket, addr: Tuple[str, int], cfg):
    peer_host, peer_port = addr[0], addr[1]
    try:
        # On connect, ask for peers (one-shot)
        conn.sendall(encode_message(GET_PEERS, []))

        # Read loop
        buf = b""
        conn.settimeout(10.0)
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
            # Try decode as many frames as present
            while True:
                try:
                    opcode, data, consumed = decode_message(buf)
                except Exception:
                    # incomplete; read more
                    break
                if consumed <= 0:
                    break
                buf = buf[consumed:]
                # Dispatch
                if opcode == INV:
                    want = []
                    for h in data:
                        if not has_block(h):
                            want.append(h)
                    if want:
                        conn.sendall(encode_message(GET_BLOCK, want))

                elif opcode == GET_BLOCK:
                    blocks = []
                    for h in data:
                        b = get_block_by_hash(h)
                        if b:
                            blocks.append({
                                "index": b.index,
                                "prev_hash": b.prev_hash,
                                "merkle": getattr(b, "merkle", None),
                                "messages": b.messages,
                                "nonce": b.nonce,
                                "timestamp": getattr(b, "timestamp", None),
                                "hash": b.hash,
                            })
                    if blocks:
                        conn.sendall(encode_message(SEND_BLOCKS, blocks))

                elif opcode == SEND_BLOCKS:
                    for bd in data:
                        try:
                            if not has_block(bd.get("hash", "")):
                                try_accept_block_dict(bd, cfg)
                        except Exception as e:
                            print("⚠️ Failed to accept remote block:", e)

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
                    # legacy client support: dump all blocks
                    ser = []
                    for b in CHAIN.blocks:
                        ser.append({
                            "index": b.index,
                            "prev_hash": b.prev_hash,
                            "merkle": getattr(b, "merkle", None),
                            "messages": b.messages,
                            "nonce": b.nonce,
                            "timestamp": getattr(b, "timestamp", None),
                            "hash": b.hash,
                        })
                    conn.sendall(encode_message(SEND_BLOCK, ser))

                else:
                    # Unknown opcode -> ignore
                    pass

    except socket.timeout:
        pass
    except Exception as e:
        print("⚠️ handler error:", e)
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
    # Load chain + config
    CHAIN.load_from_disk()
    cfg = load_config(os.getenv("WAVELOCK_CONFIG"))

    # Print policy banner
    trusted = _load_trusted_commitments()
    print(f"✅ Trusted commitments loaded: {len(trusted)}")
    print(f"🌐 WaveLock P2P server (WLv2-ready) listening on port {cfg.port}")
    print(f"   • Strict curvature verify: {'ON' if cfg.require_full_verify else 'OFF'}")
    reject_unpub = bool(int(os.getenv("WAVELOCK_REJECT_IF_UNPUBLISHED", "0")))
    print(f"   • Reject if ψ* unpublished: {'ON' if reject_unpub else 'OFF'}")

    # Seed peers from config
    for seed in cfg.seeds or []:
        try:
            host, port = seed.split(":")
            add_peer(host, int(port))
        except Exception:
            pass

    # TCP server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, cfg.port))
        srv.listen(64)
        while True:
            try:
                conn, addr = srv.accept()
                # optional: record peer
                try:
                    add_peer(addr[0], cfg.port)  # store by our port they reached
                except Exception:
                    pass
                threading.Thread(target=handle_client, args=(conn, addr, cfg), daemon=True).start()
            except KeyboardInterrupt:
                print("\n🛑 Server shutdown requested.")
                break
            except Exception as e:
                print("⚠️ accept() error:", e)
                time.sleep(0.2)

if __name__ == "__main__":
    main()
