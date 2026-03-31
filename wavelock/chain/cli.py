import json, argparse, socket, sys, os, time, shutil, hashlib
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np

from wavelock.chain.WaveLock import CurvatureKeyPair, _to_numpy
from wavelock.chain.UserRegistry import UserRegistry, sign_message_with_user, verify_signed_message
from wavelock.chain.CurvaChain import CurvaChain
from wavelock.chain.chain_utils import load_all_blocks, audit_ledger, save_block_to_disk, reset_ledger
from wavelock.chain.chain_utils import verify_chain as verify_chain_canonical
from wavelock.chain.Block import Block
from wavelock.network.protocol import (
    encode_message, decode_message,
    GET_CHAIN, SEND_BLOCK, GET_HASH, SEND_HASH,
    GET_PEERS, SEND_PEERS, VERIFY_SIGNATURE, SEND_VERIFICATION,
)

# Global chain instance
chain = CurvaChain(difficulty=3)

# Default peer list
known_peers = [
    ("localhost", 9001),
]

PEERS_FILE = "peers.json"


# ============================================================
# Utility helpers (canonical, non-duplicated)
# ============================================================

def compute_sha256(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def is_dev_mode():
    return os.getenv("WAVELOCK_MODE", "dev").lower() == "dev"


def save_hashlock():
    try:
        h = compute_sha256("ledger/blk00000.jsonl")
        with open("ledger/blk00000.hash", "w") as f:
            f.write(h)
    except Exception as e:
        print(f"Could not save hash lock: {e}")


def verify_hashlock():
    try:
        if not os.path.exists("ledger/blk00000.hash"):
            print("Hash lock file missing.")
            return False
        with open("ledger/blk00000.hash") as f:
            expected = f.read().strip()
        current = compute_sha256("ledger/blk00000.jsonl")
        if current != expected:
            print("Ledger hash mismatch. Possible tamper.")
            return False
        print("Hash lock verified.")
        return True
    except Exception as e:
        print(f"Error verifying hash: {e}")
        return False


def protect_ledger():
    try:
        os.chmod("ledger/blk00000.jsonl", 0o444)
        print("Ledger is now read-only.")
    except Exception as e:
        print(f"Could not lock ledger file: {e}")


def unprotect_ledger():
    if is_dev_mode() or os.getenv("ALLOW_UNLOCK") == "1":
        try:
            os.chmod("ledger/blk00000.jsonl", 0o644)
            print("Ledger write access restored.")
        except Exception as e:
            print(f"Could not unlock ledger file: {e}")
    else:
        print("Unlocking is disabled. Set WAVELOCK_MODE=dev or ALLOW_UNLOCK=1 to enable.")


# ============================================================
# Peer helpers
# ============================================================

def load_peers():
    if not os.path.exists(PEERS_FILE):
        return []
    with open(PEERS_FILE) as f:
        return json.load(f)


def save_peers(peers):
    with open(PEERS_FILE, "w") as f:
        json.dump(list(set(peers)), f, indent=2)


def add_peer(host, port):
    peers = load_peers()
    peer = f"{host}:{port}"
    if peer not in peers:
        peers.append(peer)
        save_peers(peers)
        print(f"Peer added: {peer}")
    else:
        print(f"Peer already exists: {peer}")


# ============================================================
# Core operations
# ============================================================

def generate_key(n, seed):
    keypair = CurvatureKeyPair(n=n, seed=seed, test_mode=True)
    with open("keypair.json", "w") as f:
        json.dump({
            "psi_0": _to_numpy(keypair.psi_0).tolist(),
            "psi_star": _to_numpy(keypair.psi_star).tolist(),
            "commitment": keypair.commitment,
            "n": n,
            "seed": seed
        }, f)
    print("Keypair saved to keypair.json")


def sign_message(message, keypair_path="keypair.json"):
    with open(keypair_path, "r") as f:
        data = json.load(f)

    keypair = CurvatureKeyPair(n=data["n"], test_mode=True)
    keypair.psi_0 = cp.asarray(data["psi_0"], dtype=cp.float64)
    keypair.psi_star = cp.asarray(data["psi_star"], dtype=cp.float64)
    keypair.commitment = data["commitment"]

    signature = keypair.sign(message)
    payload = {
        "message": message,
        "signature": signature,
        "commitment": keypair.commitment
    }
    with open("signed_message.json", "w") as f:
        json.dump(payload, f)
    print("Signed and saved to signed_message.json")


def mine_block_cli(signed_path="signed_message.json"):
    if not verify_signed_message(signed_path):
        print("Signature verification failed. Aborting.")
        return

    with open(signed_path, "r") as f:
        signed = json.load(f)

    with open("keypair.json", "r") as f:
        key_data = json.load(f)

    keypair = CurvatureKeyPair(n=key_data["n"], test_mode=True)
    keypair.psi_0 = cp.asarray(key_data["psi_0"], dtype=cp.float64)
    keypair.psi_star = cp.asarray(key_data["psi_star"], dtype=cp.float64)
    keypair.commitment = key_data["commitment"]

    if not keypair.verify(signed["message"], signed["signature"]):
        print("Signature verification failed. Aborting.")
        return

    chain_blocks = load_all_blocks()
    if not chain_blocks:
        previous_hash = "0" * 64
        index = 1
    else:
        last_block = chain_blocks[-1]
        previous_hash = last_block.hash
        index = last_block.index + 1

    messages = [
        f"message: {signed['message']}",
        f"signature: {signed['signature']}",
        f"commitment: {signed['commitment']}"
    ]

    new_block = Block(index=index, messages=messages, previous_hash=previous_hash)
    save_block_to_disk(new_block)
    broadcast_block_to_peers(new_block)
    print(f"Mined Block #{new_block.index} | Hash: {new_block.hash[:12]}... | Nonce: {new_block.nonce}")


def verify_chain():
    blocks = load_all_blocks()
    if not blocks:
        print("No blocks found.")
        return

    try:
        with open("keypair.json", "r") as f:
            key_data = json.load(f)
            keypair = CurvatureKeyPair(n=key_data["n"], test_mode=True)
            keypair.psi_0 = cp.asarray(key_data["psi_0"], dtype=cp.float64)
            keypair.psi_star = cp.asarray(key_data["psi_star"], dtype=cp.float64)
            keypair.commitment = key_data["commitment"]
    except Exception as e:
        print(f"Error loading keypair: {e}")
        return

    for i, block in enumerate(blocks):
        if block.hash != block.calculate_hash(block.nonce):
            print(f"Block #{block.index} hash mismatch.")
            return
        if not block.hash.startswith('0' * block.difficulty):
            print(f"Block #{block.index} does not meet difficulty.")
            return
        if i > 0 and block.previous_hash != blocks[i - 1].hash:
            print(f"Block #{block.index} previous_hash mismatch.")
            return

        message_line = next((m for m in block.messages if m.startswith("message: ")), None)
        signature_line = next((m for m in block.messages if m.startswith("signature: ")), None)
        commitment_line = next((m for m in block.messages if m.startswith("commitment: ")), None)

        if not message_line or not signature_line or not commitment_line:
            print(f"Block #{block.index} missing curvature metadata.")
            return

        msg = message_line[len("message: "):]
        sig = signature_line[len("signature: "):]
        commitment = commitment_line[len("commitment: "):]

        if commitment != keypair.commitment:
            print(f"Block #{block.index} commitment does not match stored keypair.")
            return

        if not keypair.verify(msg, sig):
            print(f"Block #{block.index} curvature signature invalid.")
            return

    print(f"Chain passed {len(blocks)} blocks with valid signatures and hashes.")


# ============================================================
# Networking
# ============================================================

def sync_chain_from_peer(host="localhost", port=9001):
    try:
        print(f"Connecting to {host}:{port} for chain sync...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(encode_message(GET_CHAIN))
            response = s.recv(65536)
            msg_type, data = decode_message(response)

            if msg_type != SEND_BLOCK:
                print("Invalid response from peer.")
                return

            print(f"Received {len(data)} blocks. Verifying...")

            peer_blocks = []
            for block_data in data:
                block = Block.from_dict(block_data)
                peer_blocks.append(block)

            if compare_and_reorg_chain(peer_blocks):
                print("Peer chain accepted and reorganized.")
            else:
                print("Peer chain rejected.")

            print("Sync complete. Use `verify` to confirm integrity.")
    except Exception as e:
        print(f"Sync failed: {e}")


def verify_block_list(blocks):
    try:
        with open("keypair.json", "r") as f:
            key_data = json.load(f)
        keypair = CurvatureKeyPair(n=key_data["n"], test_mode=True)
        keypair.psi_0 = cp.asarray(key_data["psi_0"], dtype=cp.float64)
        keypair.psi_star = cp.asarray(key_data["psi_star"], dtype=cp.float64)
        keypair.commitment = key_data["commitment"]
    except Exception as e:
        print(f"Error loading keypair for verification: {e}")
        return False

    for i, block in enumerate(blocks):
        if block.hash != block.calculate_hash(block.nonce):
            print(f"Block #{block.index} hash mismatch.")
            return False
        if not block.hash.startswith('0' * block.difficulty):
            print(f"Block #{block.index} does not meet difficulty.")
            return False
        if i > 0 and block.previous_hash != blocks[i - 1].hash:
            print(f"Block #{block.index} previous_hash mismatch.")
            return False

        msg = next((m for m in block.messages if m.startswith("message: ")), None)
        sig = next((m for m in block.messages if m.startswith("signature: ")), None)
        com = next((m for m in block.messages if m.startswith("commitment: ")), None)

        if not msg or not sig or not com:
            print(f"Block #{block.index} missing metadata.")
            return False

        if com[len("commitment: "):] != keypair.commitment:
            print(f"Block #{block.index} commitment mismatch.")
            return False

        if not keypair.verify(msg[len("message: "):], sig[len("signature: "):]):
            print(f"Block #{block.index} signature invalid.")
            return False

    return True


def compare_and_reorg_chain(peer_blocks):
    local_blocks = load_all_blocks()

    if len(peer_blocks) <= len(local_blocks):
        print("Peer chain is not longer. Ignoring.")
        return False

    if not verify_block_list(peer_blocks):
        print("Peer chain invalid.")
        return False

    try:
        shutil.copyfile("ledger/blk00000.jsonl", "ledger/backup_before_reorg.jsonl")
        print("Local chain backed up.")
    except Exception as e:
        print(f"Backup failed: {e}")
    reset_ledger(force=True)
    for block in peer_blocks:
        save_block_to_disk(block)
    print("Chain reorganization complete.")
    return True


def broadcast_block_to_peers(block):
    msg = encode_message(SEND_BLOCK, [block.to_dict()])
    live_peers = []
    for peer in load_peers():
        try:
            from wavelock.network.peer_utils import _parse_peer
            host, port = _parse_peer(peer)
            port = int(port)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect((host, port))
                s.sendall(msg)
                live_peers.append(peer)
                print(f"Broadcasted block #{block.index} to {host}:{port}")
        except Exception as e:
            print(f"Failed to broadcast to {peer}: {e}")
    if live_peers:
        save_peers(live_peers)


def discover_peers():
    known = set(load_peers())
    discovered = set()
    for peer in known:
        try:
            from wavelock.network.peer_utils import _parse_peer
            host, port = _parse_peer(peer)
            port = int(port)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect((host, port))
                s.sendall(encode_message(GET_PEERS))
                response = s.recv(65536)
                msg_type, data = decode_message(response)
                if msg_type == SEND_PEERS:
                    for new_peer in data:
                        if new_peer not in known:
                            discovered.add(new_peer)
        except Exception as e:
            print(f"Peer {peer} unreachable during discovery: {e}")
    if discovered:
        save_peers(list(known | discovered))
        print(f"Discovered new peers: {discovered}")


def check_peer_status():
    print("Peer Connectivity Test")
    for host, port in known_peers:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect((host, port))
                s.sendall(encode_message(GET_HASH))
                response = s.recv(1024)
                msg_type, data = decode_message(response)
                if msg_type == SEND_HASH:
                    print(f"Peer {host}:{port} online - Ledger SHA256: {data[:12]}...")
                else:
                    print(f"Peer {host}:{port} responded with unexpected data")
        except Exception as e:
            print(f"Peer {host}:{port} offline: {e}")


def precheck_peer_hash(host, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(encode_message(GET_HASH))
            response = s.recv(1024)
            msg_type, data = decode_message(response)
            if msg_type != SEND_HASH:
                print("Peer did not respond with a valid hash.")
                return False
            peer_hash = data.strip()
            local_hash = compute_sha256("ledger/blk00000.jsonl")
            if peer_hash == local_hash:
                print("Peer hash matches local ledger.")
            else:
                print(f"Peer hash mismatch:\n  Local: {local_hash}\n  Peer : {peer_hash}")
            return True
    except Exception as e:
        print(f"Could not fetch peer hash: {e}")
        return False


def sync_if_trusted(host="localhost", port=9001):
    if precheck_peer_hash(host, port):
        sync_chain_from_peer(host, port)
    else:
        print("Sync skipped due to hash mismatch.")


def tamper_block(live=False):
    if os.getenv("IS_MASTER") == "1":
        print("Tampering is disabled on master nodes.")
        return

    original = os.path.join("ledger", "blk00000.jsonl")
    if not os.path.exists(original):
        print("Ledger not found.")
        return

    with open(original, "r") as f:
        lines = f.readlines()

    target_idx = 1 if len(lines) > 1 else 0
    data = json.loads(lines[target_idx])
    data["messages"][0] += " [TAMPERED]"
    lines[target_idx] = json.dumps(data) + "\n"

    try:
        with open(original, "w") as f:
            f.writelines(lines)
    except PermissionError:
        print("Ledger is locked. Use --unlock or call unprotect_ledger() before tampering.")
        return

    print("Tampered copy saved. Run `verify` on original to check integrity.")


def bootstrap_master(force=False, port=9001):
    ledger_path = "ledger/blk00000.jsonl"

    if os.getenv("IS_MASTER") == "1" and os.path.exists(ledger_path) and not force:
        print("Master ledger already exists. Verifying hash lock...")
        if not verify_hashlock():
            print("WARNING: Ledger hash does not match. Use --force-rebuild to regenerate.")
            return
        print("Hash lock verified.")
        protect_ledger()
        verify_chain()
        print("Master is ready.")
        os.system(f"python -m wavelock.network.server --port {port}")
        return

    unprotect_ledger()
    reset_ledger(force=True)
    generate_key(n=4, seed=123)
    sign_message("genesis")
    mine_block_cli()
    verify_chain()
    save_hashlock()
    protect_ledger()
    print("Ledger hash locked and protected.")
    os.system(f"python -m wavelock.network.server --port {port}")


def remote_verify_signature(host, port, message, signature, commitment):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        payload = {
            "message": message,
            "signature": signature,
            "commitment": commitment
        }
        s.sendall(encode_message(VERIFY_SIGNATURE, payload))
        response = s.recv(4096)
        msg_type, data = decode_message(response)
        if msg_type == SEND_VERIFICATION:
            print(f"Remote verification: {data['valid']}")
        else:
            print("Remote verification failed")


def mine_daemon(args):
    host, port = args.peer.split(":")
    port = int(port)
    print(f"Miner daemon -> peer {host}:{port}")

    start_epoch = int(os.getenv("WAVELOCK_MINING_START_EPOCH", "0"))
    while time.time() < start_epoch:
        wait = int(start_epoch - time.time())
        print(f"Mining opens in {wait}s...", end="\r")
        time.sleep(1)

    i = 0
    while True:
        try:
            msg = args.message or f"mined at {int(time.time())} #{i}"
            tmp = f".miner_payload_{os.getpid()}.json"
            sign_message_with_user(args.user, msg, tmp)
            if not verify_signed_message(tmp):
                print("local verify failed; sleeping...")
                time.sleep(args.sleep)
                continue

            import subprocess
            r = subprocess.run(
                [sys.executable, "-m", "wavelock.chain.cli", "mine", "--signed_path", tmp],
                capture_output=True, text=True,
            )
            sys.stdout.write(r.stdout)
            sys.stderr.write(r.stderr)
            i += 1
            time.sleep(args.sleep)
        except KeyboardInterrupt:
            print("\nMiner stopped")
            break
        except Exception as e:
            print("Miner error:", e)
            time.sleep(2.0)


# ============================================================
# Main CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="WaveLock CLI")
    subparsers = parser.add_subparsers(dest="command")

    keygen = subparsers.add_parser("keygen")
    keygen.add_argument("--n", type=int, default=4)
    keygen.add_argument("--seed", type=int, default=42)

    sign = subparsers.add_parser("sign", help="Sign a message with a user")
    sign.add_argument("user_id")
    sign.add_argument("--message", required=True)
    sign.add_argument("--output", type=str, default="signed_message.json")

    verify = subparsers.add_parser("verify", help="Verify the entire chain")
    verify.add_argument("--path", default=None)

    mine = subparsers.add_parser("mine")
    mine.add_argument("--signed_path", type=str, default="signed_message.json")

    subparsers.add_parser("view")
    sync_p = subparsers.add_parser("sync")
    sync_p.add_argument("--host", type=str, default="localhost")
    sync_p.add_argument("--port", type=int, default=9001)
    subparsers.add_parser("tamper")
    subparsers.add_parser("restore")

    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--force-rebuild", action="store_true")
    bootstrap.add_argument("--port", type=int, default=9001)

    subparsers.add_parser("reset")
    subparsers.add_parser("unlock")
    subparsers.add_parser("audit", help="Audit all ledger files and locks")
    subparsers.add_parser("peers")
    peer_p = subparsers.add_parser("peer")
    peer_p.add_argument("host")
    peer_p.add_argument("port", type=int)
    subparsers.add_parser("discover", help="Trigger peer discovery via gossip")

    add = subparsers.add_parser("add", help="Add a new user")
    add.add_argument("user_id")
    add.add_argument("--n", type=int, default=4)
    add.add_argument("--seed", type=int, default=None)

    sp = subparsers.add_parser("mine-daemon", help="Run continuous miner")
    sp.add_argument("--peer", default="127.0.0.1:9001")
    sp.add_argument("--message", default=None)
    sp.add_argument("--user", default="ricky")
    sp.add_argument("--sleep", type=float, default=0.5)

    args = parser.parse_args()

    if args.command == "keygen":
        generate_key(args.n, args.seed)
    elif args.command == "add":
        registry = UserRegistry()
        registry.add_user(args.user_id, n=args.n, seed=args.seed)
    elif args.command == "verify":
        verify_chain_canonical()
    elif args.command == "sign":
        sign_message_with_user(args.user_id, args.message, args.output)
    elif args.command == "mine":
        mine_block_cli(args.signed_path)
    elif args.command == "view":
        blocks = load_all_blocks()
        print("Ledger Overview:")
        for block in blocks:
            print(f"  Block #{block.index} | Hash: {block.hash[:12]}... | "
                  f"Messages: {len(block.messages)} | Time: {block.timestamp}")
        if not blocks:
            print("  Ledger is empty.")
    elif args.command == "sync":
        sync_if_trusted(args.host, args.port)
    elif args.command == "tamper":
        tamper_block()
    elif args.command == "restore":
        sync_if_trusted("localhost", 9001)
        print("Verifying after sync...")
        verify_chain()
    elif args.command == "bootstrap":
        bootstrap_master(force=args.force_rebuild, port=args.port)
    elif args.command == "reset":
        reset_ledger()
    elif args.command == "unlock":
        unprotect_ledger()
    elif args.command == "audit":
        ok = audit_ledger()
        blocks = load_all_blocks()
        verify_chain_canonical(blocks)
        sys.exit(0 if ok else 1)
    elif args.command == "peers":
        check_peer_status()
    elif args.command == "peer":
        add_peer(args.host, args.port)
    elif args.command == "discover":
        discover_peers()
    elif args.command == "mine-daemon":
        mine_daemon(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
