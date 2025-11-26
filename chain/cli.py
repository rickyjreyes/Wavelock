# keep these (dedupe if repeated)
import json, argparse, socket, sys, os, time, shutil, hashlib
import cupy as cp

# at top of chain/cli.py
from .WaveLock import CurvatureKeyPair
from .UserRegistry import UserRegistry, sign_message_with_user, verify_signed_message
from .CurvaChain import CurvaChain
from .chain_utils import load_all_blocks, audit_ledger, save_block_to_disk, reset_ledger
from .chain_utils import verify_chain as verify_chain_canonical
from .Block import Block
# from .network.protocol import (
#     encode_message, decode_message, GET_CHAIN, SEND_BLOCK, GET_HASH, SEND_HASH,
#     GET_PEERS, SEND_PEERS, VERIFY_SIGNATURE, SEND_VERIFICATION
# )


# network
network_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network")
sys.path.append(network_path)
from network.protocol import encode_message, decode_message, GET_CHAIN, SEND_BLOCK, GET_HASH, SEND_HASH, GET_PEERS, SEND_PEERS, VERIFY_SIGNATURE, SEND_VERIFICATION

# Global chain instance
chain = CurvaChain(difficulty=3)



# Default peer list
known_peers = [
    ("localhost", 9001),
    # Add more (host, port) tuples as needed
]

def compute_sha256(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def save_hashlock():
    try:
        h = compute_sha256("ledger/blk00000.jsonl")
        with open("ledger/blk00000.hash", "w") as f:
            f.write(h)
    except Exception as e:
        print(f"⚠️ Could not save hash lock: {e}")

def verify_hashlock():
    try:
        if not os.path.exists("ledger/blk00000.hash"):
            print("❌ Hash lock file missing.")
            return False
        with open("ledger/blk00000.hash") as f:
            expected = f.read().strip()
        current = compute_sha256("ledger/blk00000.jsonl")
        if current != expected:
            print("🚨 Ledger hash mismatch. Possible tamper.")
            return False
        print("🔐 Hash lock verified.")
        return True
    except Exception as e:
        print(f"❌ Error verifying hash: {e}")
        return False

def protect_ledger():
    try:
        os.chmod("ledger/blk00000.jsonl", 0o444)
        print("🔒 Ledger is now read-only.")
    except Exception as e:
        print(f"⚠️ Could not lock ledger file: {e}")

def unprotect_ledger():
    if is_dev_mode() or os.getenv("ALLOW_UNLOCK") == "1":
        try:
            os.chmod("ledger/blk00000.jsonl", 0o644)
            print("🔓 Ledger write access restored.")
        except Exception as e:
            print(f"⚠️ Could not unlock ledger file: {e}")
    else:
        print("🚫 Unlocking is disabled. Set WAVELOCK_MODE=dev or ALLOW_UNLOCK=1 to enable.")

# def mine_block_cli():
#     chain = load_all_blocks()
#     if not chain:
#         previous_hash = "0" * 64
#         index = 1
#     else:
#         last_block = chain[-1]
#         previous_hash = last_block.hash
#         index = last_block.index + 1

#     messages = ["Hello from WaveLock"]
#     new_block = Block(index=index, messages=messages, previous_hash=previous_hash)
#     save_block_to_disk(new_block)
#     print(f"⛏️  Mined Block #{new_block.index} | Hash: {new_block.hash[:12]}... | Nonce: {new_block.nonce}")

def generate_key(n, seed):
    keypair = CurvatureKeyPair(n=n, seed=seed)
    with open("keypair.json", "w") as f:
        json.dump({
            "psi_0": keypair.psi_0.tolist(),
            "psi_star": keypair.psi_star.tolist(),  # 💾 save for exact match
            "commitment": keypair.commitment,
            "n": n,
            "seed": seed
        }, f)
    print("🔐 Keypair saved to keypair.json")


def sign_message(message, keypair_path="keypair.json"):
    with open(keypair_path, "r") as f:
        data = json.load(f)

    keypair = CurvatureKeyPair(n=data["n"])
    keypair.psi_0 = cp.asarray(data["psi_0"])
    keypair.psi_star = cp.asarray(data["psi_star"])  # exact match
    keypair.commitment = data["commitment"]

    signature = keypair.sign(message)
    payload = {
        "message": message,
        "signature": signature,
        "commitment": keypair.commitment
    }
    with open("signed_message.json", "w") as f:
        json.dump(payload, f)
    print("✍️ Signed and saved to signed_message.json")

def mine_block_cli(signed_path="signed_message.json"):
        # ✅ Verify curvature signature from user registry
    if not verify_signed_message(signed_path):
        print("❌ Signature verification failed. Aborting.")
        return


    # 🔐 Load signed message first
    with open(signed_path, "r") as f:
        signed = json.load(f)

    # ✅ Load keypair and verify signature using saved ψ*
    # ✅ Load keypair and verify signature using saved ψ*
    with open("keypair.json", "r") as f:
        key_data = json.load(f)

    keypair = CurvatureKeyPair(n=key_data["n"])
    keypair.psi_0 = cp.asarray(key_data["psi_0"])
    keypair.psi_star = cp.asarray(key_data["psi_star"])  # exact ψ*
    keypair.commitment = key_data["commitment"]

    if not keypair.verify(signed["message"], signed["signature"]):
        print("❌ Signature verification failed. Aborting.")
        return


    # 🧱 Load existing chain
    chain = load_all_blocks()
    if not chain:
        previous_hash = "0" * 64
        index = 1
    else:
        last_block = chain[-1]
        previous_hash = last_block.hash
        index = last_block.index + 1

    # 🧾 Prepare block messages
    messages = [
        f"message: {signed['message']}",
        f"signature: {signed['signature']}",
        f"commitment: {signed['commitment']}"
    ]

    # ⛏️ Mine and store the block
    new_block = Block(index=index, messages=messages, previous_hash=previous_hash)
    save_block_to_disk(new_block)
    broadcast_block_to_peers(new_block)
    print(f"⛏️  Mined Block #{new_block.index} | Hash: {new_block.hash[:12]}... | Nonce: {new_block.nonce}")


# def add_signed_block(signed_path="signed_message.json"):
#     with open(signed_path, "r") as f:
#         signed = json.load(f)
#     block_data = [
#         f"message: {signed['message']}",
#         f"signature: {signed['signature']}",
#         f"commitment: {signed['commitment']}"
#     ]
#     chain.add_block(block_data)
#     latest_block = chain.get_latest_block()
#     append_block(latest_block)

#     print("⛓ Block added with curvature-locked message.")

def verify_chain():
    blocks = load_all_blocks()

    if not blocks:
        print("⚠️ No blocks found.")
        return

    # ✅ Load ψ* from keypair.json
    try:
        with open("keypair.json", "r") as f:
            key_data = json.load(f)
            keypair = CurvatureKeyPair(n=key_data["n"])
            keypair.psi_0 = cp.asarray(key_data["psi_0"])
            keypair.psi_star = cp.asarray(key_data["psi_star"])
            keypair.commitment = key_data["commitment"]
    except Exception as e:
        print(f"❌ Error loading keypair: {e}")
        return

    for i, block in enumerate(blocks):
        # ✅ Verify PoW integrity
        if block.hash != block.calculate_hash(block.nonce):
            print(f"❌ Block #{block.index} hash mismatch.")
            return
        if not block.hash.startswith('0' * block.difficulty):
            print(f"❌ Block #{block.index} does not meet difficulty.")
            return

        # ✅ Verify linkage
        if i > 0 and block.previous_hash != blocks[i - 1].hash:
            print(f"❌ Block #{block.index} previous_hash mismatch.")
            return

        # ✅ Extract curvature fields
        message_line = next((m for m in block.messages if m.startswith("message: ")), None)
        signature_line = next((m for m in block.messages if m.startswith("signature: ")), None)
        commitment_line = next((m for m in block.messages if m.startswith("commitment: ")), None)

        if not message_line or not signature_line or not commitment_line:
            print(f"❌ Block #{block.index} missing curvature metadata.")
            return

        msg = message_line[len("message: "):]
        sig = signature_line[len("signature: "):]
        commitment = commitment_line[len("commitment: "):]

        # ✅ Check commitment matches
        if commitment != keypair.commitment:
            print(f"❌ Block #{block.index} commitment does not match stored ψ*.")
            return

        # ✅ Verify signature
        if not keypair.verify(msg, sig):
            print(f"❌ Block #{block.index} curvature signature invalid.")
            return

    print(f"✅ Chain passed {len(blocks)} blocks with valid signatures and hashes.")

def sync_chain_from_peer(host="localhost", port=9001):
    import socket
    from protocol import encode_message, decode_message, GET_CHAIN, SEND_BLOCK
    from Block import Block
    from chain_utils import save_block_to_disk

    try:
        print(f"🌐 Connecting to {host}:{port} for chain sync...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(encode_message(GET_CHAIN))
            response = s.recv(65536)
            msg_type, data = decode_message(response)

            if msg_type != SEND_BLOCK:
                print("❌ Invalid response from peer.")
                return

            print(f"📦 Received {len(data)} blocks. Verifying...")

            peer_blocks = []
            for block_data in data:
                block = Block(
                    index=block_data["index"],
                    messages=block_data["messages"],
                    previous_hash=block_data["previous_hash"],
                    difficulty=block_data.get("difficulty", 4),
                    timestamp=block_data["timestamp"],
                    nonce=block_data["nonce"],
                    block_hash=block_data["hash"],
                    merkle_root=block_data["merkle_root"]
                )
                peer_blocks.append(block)

            # ✅ Compare and reorg after full list is built
            if compare_and_reorg_chain(peer_blocks):
                print("✅ Peer chain accepted and reorganized.")
            else:
                print("❌ Peer chain rejected.")


            # if verify_block_list(peer_blocks):
            #     print("✅ Peer chain verified.")
            #     reset_ledger()
            #     for block in peer_blocks:
            #         save_block_to_disk(block)
            #     print("✅ Synced peer chain saved.")
            # else:
            #     print("❌ Rejected peer chain: verification failed.")

            print("✅ Sync complete. Use `verify` to confirm integrity.")
    except Exception as e:
        print(f"❌ Sync failed: {e}")

def verify_block_list(blocks):
    try:
        with open("keypair.json", "r") as f:
            key_data = json.load(f)
        keypair = CurvatureKeyPair(n=key_data["n"])
        keypair.psi_0 = cp.asarray(key_data["psi_0"])
        keypair.psi_star = cp.asarray(key_data["psi_star"])
        keypair.commitment = key_data["commitment"]
    except Exception as e:
        print(f"❌ Error loading keypair for verification: {e}")
        return False

    for i, block in enumerate(blocks):
        if block.hash != block.calculate_hash(block.nonce):
            print(f"❌ Block #{block.index} hash mismatch.")
            return False
        if not block.hash.startswith('0' * block.difficulty):
            print(f"❌ Block #{block.index} does not meet difficulty.")
            return False
        if i > 0 and block.previous_hash != blocks[i - 1].hash:
            print(f"❌ Block #{block.index} previous_hash mismatch.")
            return False

        msg = next((m for m in block.messages if m.startswith("message: ")), None)
        sig = next((m for m in block.messages if m.startswith("signature: ")), None)
        com = next((m for m in block.messages if m.startswith("commitment: ")), None)

        if not msg or not sig or not com:
            print(f"❌ Block #{block.index} missing metadata.")
            return False

        if com[len("commitment: "):] != keypair.commitment:
            print(f"❌ Block #{block.index} commitment mismatch.")
            return False

        if not keypair.verify(msg[len("message: "):], sig[len("signature: "):]):
            print(f"❌ Block #{block.index} signature invalid.")
            return False

    return True

def tamper_block(live=False):
    import shutil

    if os.getenv("IS_MASTER") == "1":
        print("🚫 Tampering is disabled on master nodes.")
        return


    original = os.path.join("ledger", "blk00000.jsonl")
    tampered = original

    if not os.path.exists(original):
        print("❌ Ledger not found.")
        return

    if original != tampered:
        shutil.copyfile(original, tampered)


    with open(tampered, "r") as f:
        lines = f.readlines()

    target_idx = 1 if len(lines) > 1 else 0
    data = json.loads(lines[target_idx])
    data["messages"][0] += " 🚨"
    lines[target_idx] = json.dumps(data) + "\n"

    try:
        with open(tampered, "w") as f:
            f.writelines(lines)
    except PermissionError:
        print("🚫 Ledger is locked. Use --unlock or call unprotect_ledger() before tampering.")
        return


    print("🔧 Tampered copy saved. Run `verify` on original to check integrity.")


def bootstrap_master(force=False, port=9001):
    ledger_path = "ledger/blk00000.jsonl"

    if os.getenv("IS_MASTER") == "1" and os.path.exists(ledger_path) and not force:
        print("🛡️ Master ledger already exists. Verifying hash lock...")
        if not verify_hashlock():
            print("🚨 WARNING: Ledger hash does not match. Use --force-rebuild to regenerate.")
            return
        print("✅ Hash lock verified.")
        protect_ledger()
        verify_chain()
        print("✅ Master is ready.")
        os.system(f"python network/server.py --port {port}")
        return

    unprotect_ledger()
    reset_ledger(force=True)
    generate_key(n=4, seed=123)
    sign_message("genesis")
    mine_block_cli()
    verify_chain()
    save_hashlock()
    protect_ledger()
    print("✅ Ledger hash locked and protected.")
    os.system(f"python network/server.py --port {port}")



def restore_from_peer():
    print("♻️ Attempting to restore from peer...")
    sync_chain_from_peer(host="localhost", port=9001)
    print("🔁 Verifying after sync...")
    verify_chain()

def is_dev_mode():
    return os.getenv("WAVELOCK_MODE", "dev").lower() == "dev"

def compute_sha256(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def protect_ledger():
    try:
        os.chmod("ledger/blk00000.jsonl", 0o444)
        print("🔒 Ledger is now read-only.")
    except Exception as e:
        print(f"⚠️ Could not lock ledger file: {e}")

def unprotect_ledger():
    if is_dev_mode() or os.getenv("ALLOW_UNLOCK") == "1":
        try:
            os.chmod("ledger/blk00000.jsonl", 0o644)
            print("🔓 Ledger write access restored.")
        except Exception as e:
            print(f"⚠️ Could not unlock ledger file: {e}")
    else:
        print("🚫 Unlocking is disabled. Set WAVELOCK_MODE=dev or ALLOW_UNLOCK=1 to enable.")

def save_hashlock():
    try:
        h = compute_sha256("ledger/blk00000.jsonl")
        with open("ledger/blk00000.hash", "w") as f:
            f.write(h)
    except Exception as e:
        print(f"⚠️ Could not save hash lock: {e}")

def verify_hashlock():
    try:
        if not os.path.exists("ledger/blk00000.hash"):
            print("❌ Hash lock file missing.")
            return False
        with open("ledger/blk00000.hash") as f:
            expected = f.read().strip()
        current = compute_sha256("ledger/blk00000.jsonl")
        if current != expected:
            print("🚨 Ledger hash mismatch. Possible tamper.")
            return False
        print("🔐 Hash lock verified.")
        return True
    except Exception as e:
        print(f"❌ Error verifying hash: {e}")
        return False


# def audit_ledger():
#     print("🔎 Ledger Audit Report")
#     print("------------------------")
#     ledgers = [f for f in os.listdir("ledger") if f.startswith("blk") and f.endswith(".jsonl")]
#     if not ledgers:
#         print("❌ No ledger files found.")
#         return

#     for ledger_file in sorted(ledgers):
#         path = os.path.join("ledger", ledger_file)
#         print(f"\n📄 Auditing {ledger_file}...")
#         print(f"📏 SHA256: {compute_sha256(path)}")
#         ts = time.ctime(os.path.getmtime(path))
#         print(f"⏱ Last modified: {ts}")

#         if ledger_file == "blk00000.jsonl":
#             print("🔍 Verifying hash lock...")
#             if verify_hashlock():
#                 print("✅ Hash matches stored lock.")
#             else:
#                 print("❌ Hash lock mismatch.")

#             print("🔐 Checking file permissions...")
#             st_mode = os.stat(path).st_mode
#             if oct(st_mode)[-3:] == "444":
#                 print("✅ File is read-only.")
#             else:
#                 print("⚠️ File is writable.")

#     try:
#         with open("keypair.json", "r") as f:
#             key_data = json.load(f)
#         keypair = CurvatureKeyPair(n=key_data["n"])
#         keypair.psi_0 = cp.asarray(key_data["psi_0"])
#         keypair.psi_star = cp.asarray(key_data["psi_star"])
#         keypair.commitment = key_data["commitment"]
#         print(f"🔗 Commitment: {keypair.commitment}")
#     except Exception as e:
#         print(f"❌ Error loading keypair for audit: {e}")
#         return

#     blocks = load_all_blocks()
#     print(f"📦 Total blocks in ledger: {len(blocks)}")
#     last_ts = 0
#     for block in blocks:
#         if block.timestamp < last_ts:
#             print(f"⚠️ Block #{block.index} has a timestamp earlier than previous.")
#         last_ts = block.timestamp

#         msg = next((m for m in block.messages if m.startswith("message: ")), None)
#         sig = next((m for m in block.messages if m.startswith("signature: ")), None)
#         com = next((m for m in block.messages if m.startswith("commitment: ")), None)

#         if not msg or not sig or not com:
#             print(f"❌ Block #{block.index} is missing curvature metadata.")
#             continue

#         if com[len("commitment: "):] != keypair.commitment:
#             print(f"❌ Block #{block.index} commitment mismatch.")
#             continue

#         if not keypair.verify(msg[len("message: "):], sig[len("signature: "):]):
#             print(f"❌ Block #{block.index} signature invalid.")
#         else:
#             print(f"✅ Block #{block.index} curvature signature valid.")





def precheck_peer_hash(host, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(encode_message(GET_HASH))
            response = s.recv(1024)
            msg_type, data = decode_message(response)
            if msg_type != SEND_HASH:
                print("❌ Peer did not respond with a valid hash.")
                return False
            peer_hash = data.strip()
            local_hash = compute_sha256("ledger/blk00000.jsonl")
            if peer_hash == local_hash:
                print("✅ Peer hash matches local ledger.")
            else:
                print(f"⚠️ Peer hash mismatch:\n  Local: {local_hash}\n  Peer : {peer_hash}")
            return True
    except Exception as e:
        print(f"❌ Could not fetch peer hash: {e}")
        return False
    

def sync_if_trusted(host="localhost", port=9001):
    if precheck_peer_hash(host, port):
        from cli import sync_chain_from_peer
        sync_chain_from_peer(host, port)
    else:
        print("❌ Sync skipped due to hash mismatch.")

def broadcast_block_to_peers(block):
    msg = encode_message(SEND_BLOCK, [block.to_dict()])
    live_peers = []
    for peer in load_peers():
        try:
            from network.peer_utils import _parse_peer
            host, port = _parse_peer(peer)
            port = int(port)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect((host, port))
                s.sendall(msg)
                live_peers.append(peer)
                print(f"📡 Broadcasted block #{block.index} to {host}:{port}")
        except Exception as e:
            print(f"⚠️ Failed to broadcast to {peer} — {e}")
    if live_peers:
        save_peers(live_peers)  # 🧠 Save only responsive peers

def discover_peers():
    known_peers = set(load_peers())
    discovered = set()
    for peer in known_peers:
        try:
            from network.peer_utils import _parse_peer
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
                        if new_peer not in known_peers:
                            discovered.add(new_peer)
        except Exception as e:
            print(f"⚠️ Peer {peer} unreachable during discovery: {e}")
    if discovered:
        save_peers(list(known_peers | discovered))
        print(f"🔎 Discovered new peers: {discovered}")

def check_peer_status():
    print("🌐 Peer Connectivity Test")
    for host, port in known_peers:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect((host, port))
                s.sendall(encode_message(GET_HASH))
                response = s.recv(1024)
                msg_type, data = decode_message(response)
                if msg_type == SEND_HASH:
                    print(f"✅ Peer {host}:{port} online — Ledger SHA256: {data[:12]}...")
                else:
                    print(f"⚠️ Peer {host}:{port} responded with unexpected data")
        except Exception as e:
            print(f"❌ Peer {host}:{port} offline — {e}")

def compare_and_reorg_chain(peer_blocks):
    local_blocks = load_all_blocks()
    
    # Only accept longer chains
    if len(peer_blocks) <= len(local_blocks):
        print("📉 Peer chain is not longer. Ignoring.")
        return False

    # Verify peer chain fully
    if not verify_block_list(peer_blocks):
        print("❌ Peer chain invalid.")
        return False

    # Reorg: overwrite local ledger

    try:
        shutil.copyfile("ledger/blk00000.jsonl", "ledger/backup_before_reorg.jsonl")
        print("🧾 Local chain backed up.")
    except Exception as e:
        print(f"⚠️ Backup failed: {e}")
    reset_ledger(force=True)
    for block in peer_blocks:
        save_block_to_disk(block)
    print("🔁 Chain reorganization complete.")
    return True

PEERS_FILE = "peers.json"

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
        print(f"✅ Peer added: {peer}")
    else:
        print(f"ℹ️ Peer already exists: {peer}")


def is_dev_mode():
    return os.getenv("WAVELOCK_MODE", "dev").lower() == "dev"



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
            print(f"✅ Remote verification: {data['valid']}")
        else:
            print("❌ Remote verification failed")

def mine_daemon(args):
    import time, os
    from chain.UserRegistry import sign_message_with_user, verify_signed_message
    host, port = args.peer.split(":")
    port = int(port)
    print(f"⛏️  Miner daemon -> peer {host}:{port}")

    # honor optional launch gate
    start_epoch = int(os.getenv("WAVELOCK_MINING_START_EPOCH","0"))
    while time.time() < start_epoch:
        wait = int(start_epoch - time.time())
        print(f"⏳ Mining opens in {wait}s...", end="\r")
        time.sleep(1)

    i = 0
    while True:
        try:
            # sign a message (either given or rotating counter)
            msg = args.message or f"mined at {int(time.time())} #{i}"
            tmp = f".miner_payload_{os.getpid()}.json"
            sign_message_with_user(args.user, msg, tmp)
            if not verify_signed_message(tmp):
                print("❌ local verify failed; sleeping…")
                time.sleep(args.sleep); continue

            # mine one block (calls your existing CLI path)
            import subprocess, sys
            r = subprocess.run([sys.executable, "-m", "chain.cli", "mine", "--signed_path", tmp],
                               capture_output=True, text=True)
            sys.stdout.write(r.stdout)
            sys.stderr.write(r.stderr)
            i += 1
            time.sleep(args.sleep)
        except KeyboardInterrupt:
            print("\n🛑 miner stopped")
            break
        except Exception as e:
            print("⚠️ miner error:", e)
            time.sleep(2.0)



def main():
    # ledger = LedgerManager()
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
    verify.add_argument("--path", default=None)   # optional; we won’t use it



    # sign = subparsers.add_parser("sign")
    # sign.add_argument("message")

    mine = subparsers.add_parser("mine")
    mine.add_argument("--signed_path", type=str, default="signed_message.json")
    # verify = subparsers.add_parser("verify")
    load = subparsers.add_parser("load")
    view = subparsers.add_parser("view")
    sync = subparsers.add_parser("sync")
    tamper = subparsers.add_parser("tamper")
    sync.add_argument("--host", type=str, default="localhost")
    sync.add_argument("--port", type=int, default=9001)
    restore = subparsers.add_parser("restore")


    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--force-rebuild", action="store_true", help="Force wipe and rebuild master ledger")
    bootstrap.add_argument("--port", type=int, default=9001, help="Port to launch P2P server on")

    reset = subparsers.add_parser("reset")
    unlock = subparsers.add_parser("unlock")
    audit = subparsers.add_parser("audit", help="Audit all ledger files and locks")
    peers = subparsers.add_parser("peers")
    peer = subparsers.add_parser("peer")
    peer.add_argument("host")
    peer.add_argument("port", type=int)
    discover = subparsers.add_parser("discover", help="Trigger peer discovery via gossip")

    add = subparsers.add_parser("add", help="Add a new user")
    add.add_argument("user_id")
    add.add_argument("--n", type=int, default=4)
    add.add_argument("--seed", type=int, default=None)

    # near argparse setup:
    sp = subparsers.add_parser("mine-daemon", help="Run continuous miner")
    sp.add_argument("--peer", default="127.0.0.1:9001")
    sp.add_argument("--message", default=None, help="Message to sign & include each block")
    sp.add_argument("--user", default="ricky", help="User id in users.json")
    sp.add_argument("--sleep", type=float, default=0.5, help="Sleep between attempts when idle")
    sp.set_defaults(cmd="mine_daemon")

  

    args = parser.parse_args()

    if args.command == "keygen":
        generate_key(args.n, args.seed)
    elif args.command == "add":
        registry = UserRegistry()
        registry.add_user(args.user_id, n=args.n, seed=args.seed)
    elif args.command == "verify":
        verify_chain_canonical()   # <- the one from chain_utils
    elif args.command == "sign":
        sign_message_with_user(args.user_id, args.message, args.output)
    elif args.command == "mine":
        mine_block_cli(args.signed_path)
    elif args.command == "verify":
        verify_chain()
    elif args.command == "load":
        loaded = load_chain()
        print(f"📂 Loaded chain with {len(loaded.chain)} blocks")
    elif args.command == "view":
        blocks = load_all_blocks()
        print("📖 Ledger Overview:")
        for block in blocks:
            print(f"📜 Block #{block.index} | Hash: {block.hash[:12]}... | "
                f"Messages: {len(block.messages)} | Time: {round(block.timestamp)}")
        if not blocks:
            print("⚠️ Ledger is empty.")
    elif args.command == "sync":
        sync_if_trusted(args.host, args.port)
    elif args.command == "tamper":
        tamper_block()
    elif args.command == "restore":
        sync_if_trusted("localhost", 9001)
        print("🔁 Verifying after sync...")
        verify_chain()
    elif args.command == "bootstrap":
        bootstrap_master(force=args.force_rebuild, port=args.port)
    elif args.command == "reset":
        reset_ledger()
    elif args.command == "unlock":
        unprotect_ledger()
    elif args.command == "audit":
        ok = audit_ledger()              # prints per-file lock + RO
        # optional: run chain-level verify summary too
        blocks = load_all_blocks()
        verify_chain_canonical(blocks)
        import sys
        sys.exit(0 if ok else 1)
    elif args.command == "peers":
        check_peer_status()
    elif args.command == "peer":
        add_peer(args.host, args.port)
    elif args.command == "discover":
        discover_peers()
    elif args.command == "load":
        graph = ledger.load_graph()
        print(f"✅ Loaded ψ-graph with {len(graph.nodes)} nodes.")

    elif args.command == "save":
        ledger.save_block()
        print("✅ Saved ψ-graph and curvature-locked block to ledger.")

    elif args.command == "verify":
        if ledger.verify_ledger():
            print("✅ Ledger integrity check passed.")
        else:
            print("❌ Ledger verification failed.")

    elif args.command == "reload":
        graph = ledger.load_latest_from_ledger()
        print(f"✅ Reloaded ψ-graph from last verified block: {len(graph.nodes)} nodes.")

    elif args.command == "drift":
        if ledger.collapse_required():
            print("⚠️ Entropy drift detected — collapse required.")
        else:
            print("✅ Coherence intact — agent survives.")
      # in main()'s dispatch:
    elif args.cmd == "mine_daemon":
        mine_daemon(args)

    else:
        parser.print_help()


    


if __name__ == "__main__":
    main()
