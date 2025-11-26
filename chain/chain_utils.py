import os, time
import json
import cupy as cp
import matplotlib.pyplot as plt
from chain.CurvaChain import CurvaChain
from chain.WaveLock import CurvatureKeyPair, symbolic_verifier, load_quantum_keys
from chain.Block import Block



LEDGER_DIR = "ledger"
LEDGER_FILE = os.path.join(LEDGER_DIR, "blk00000.jsonl")


# --- add near the top ---
from pathlib import Path
import hashlib, sys

def _write_lock(path: Path) -> Path:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lock = path.with_suffix(".hash")
    lock.write_text(digest)
    return lock


def _is_writable(p: Path) -> bool:
    try:
        return os.access(p, os.W_OK)
    except Exception:
        return False

def _next_ledger_path() -> Path:
    led = Path("ledger")
    led.mkdir(exist_ok=True)
    files = sorted(led.glob("blk*.jsonl"))
    if not files:
        return led / "blk00000.jsonl"
    # assume contiguous numbering; pick max+1
    last_idx = max(int(f.stem.replace("blk","")) for f in files)
    return led / f"blk{last_idx+1:05d}.jsonl"

def _set_readonly(p: Path):
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetFileAttributesW(str(p), 0x1)  # READONLY
        else:
            os.chmod(p, 0o444)
    except Exception:
        pass




# --- 1. Save / Load Chain ---

def save_chain(chain: CurvaChain, filename="chain.json"):
    serializable = []
    for block in chain.chain:
        serializable.append({
            "index": block.index,
            "timestamp": block.timestamp,
            "messages": block.messages,
            "previous_hash": block.previous_hash,
            "nonce": block.nonce,
            "hash": block.hash,
            "difficulty": block.difficulty,
            "merkle_root": block.merkle_root,
        })
    with open(filename, "w") as f:
        json.dump(serializable, f)
    print(f"💾 Chain saved to {filename}")

def load_chain(filename="chain.json") -> CurvaChain:
    with open(filename, "r") as f:
        data = json.load(f)
    chain = CurvaChain(difficulty=data[0]["difficulty"])
    chain.chain = []
    for block_data in data:
        block = type("Block", (), {})()
        block.__dict__.update(block_data)
        chain.chain.append(block)
    print(f"📂 Chain loaded from {filename}")
    return chain

# --- 2. Visualize ψ* + Entropy ---

def visualize_psi(psi_star):
    psi_np = cp.asnumpy(psi_star)
    entropy = -cp.sum((psi_star**2 / cp.sum(psi_star**2)) * cp.log(psi_star**2 / cp.sum(psi_star**2) + 1e-12)).get()

    plt.figure(figsize=(6, 5))
    plt.imshow(psi_np, cmap="viridis")
    plt.colorbar(label="ψ* magnitude")
    plt.title(f"ψ* Heatmap | Entropy: {entropy:.4f}")
    plt.tight_layout()
    plt.show()

# --- 3. Tamper Test (1-bit Perturbation) ---

def tamper_and_test(keypair: CurvatureKeyPair):
    tampered = keypair.psi_star.copy()
    tampered[0, 0] += 0.5  # or even try 1.0 for clear failure
    reference = keypair.evolve(cp.asarray(keypair.psi_0.copy()), keypair.n)
    result = symbolic_verifier(tampered, reference)
    print("🧪 Tamper test:")
    print(f"    Tampered accepted? {'✅' if result else '❌'}")
    return result


def save_block_to_disk(block):
    target = Path(LEDGER_FILE)
    target.parent.mkdir(exist_ok=True)

    try:
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(block.to_dict()) + "\n")
    except PermissionError:
        # read-only or locked → rotate
        target = _next_ledger_path()
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(block.to_dict()) + "\n")

    lock_path = _write_lock(target)
    if os.getenv("WL_LOCK_AFTER_WRITE") == "1":
        _set_readonly(target); _set_readonly(lock_path)


def load_all_blocks() -> list[Block]:
    """Load blocks from ALL rotated files: blk00000.jsonl, blk00001.jsonl, …"""
    blocks: list[Block] = []
    led = Path(LEDGER_DIR)
    if not led.exists():
        return blocks
    for p in sorted(led.glob("blk*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                blocks.append(Block(
                    index=data["index"],
                    messages=data["messages"],
                    previous_hash=data["previous_hash"],
                    difficulty=data.get("difficulty", 4),
                    timestamp=data["timestamp"],
                    nonce=data["nonce"],
                    block_hash=data["hash"],
                    merkle_root=data["merkle_root"],
                ))
    return blocks


def reset_ledger(force: bool = False):
    """Delete all ledger files and sidecars in ./ledger."""
    led = Path(LEDGER_DIR)
    if not force:
        ans = input("⚠️  This will delete your entire blockchain ledger. Type 'yes' to confirm: ").strip().lower()
        if ans != "yes":
            print("❌ Ledger reset cancelled.")
            return
    if led.exists():
        for p in led.glob("blk*.jsonl"):
            p.unlink(missing_ok=True)
        for p in led.glob("*.hash"):
            p.unlink(missing_ok=True)
        for p in led.glob("*.sha256"):
            p.unlink(missing_ok=True)
    print("✅ Ledger reset complete.")



def verify_chain(blocks=None, keypair=None):
    if blocks is None:
        blocks = load_all_blocks()
    if not blocks:
        print("⚠️ No blocks found.")
        return False

    if keypair is None:
        try:
            keypair = load_quantum_keys()  # expects psi_keypair.json
        except Exception:
            with open("keypair.json","r") as f:
                data = json.load(f)
            kp = CurvatureKeyPair(n=data["n"])
            kp.psi_0   = cp.asarray(data["psi_0"])
            kp.psi_star= cp.asarray(data["psi_star"])
            kp.commitment = data["commitment"]
            keypair = kp


    for i, block in enumerate(blocks):
        if block.hash != block.calculate_hash(block.nonce):
            print(f"❌ Block #{block.index} hash mismatch.")
            return False
        if not block.hash.startswith("0" * block.difficulty):
            print(f"❌ Block #{block.index} does not meet difficulty.")
            return False
        if i > 0 and block.previous_hash != blocks[i - 1].hash:
            print(f"❌ Block #{block.index} previous_hash mismatch.")
            return False

        msg_line = next((m for m in block.messages if m.startswith("message: ")), None)
        sig_line = next((m for m in block.messages if m.startswith("signature: ")), None)
        commit_line = next((m for m in block.messages if m.startswith("commitment: ")), None)

        if not msg_line or not sig_line or not commit_line:
            print(f"❌ Block #{block.index} missing curvature metadata.")
            return False

        msg = msg_line[len("message: "):]
        sig = sig_line[len("signature: "):]
        commit = commit_line[len("commitment: "):]

        if commit != keypair.commitment:
            print(f"❌ Block #{block.index} commitment mismatch.")
            return False
        if not keypair.verify(msg, sig):
            print(f"❌ Block #{block.index} signature invalid.")
            return False

        if not verify_merkle_root(block):
            print(f"❌ Block #{block.index} Merkle root invalid.")
            return False

    print(f"✅ Verified {len(blocks)} blocks: curvature, linkage, hash, Merkle.")
    return True

    
def get_last_signature(blocks):
    for block in reversed(blocks):
        for line in block.messages:
            if line.startswith("signature: "):
                return line[len("signature: "):]
    return None

def verify_merkle_root(block: Block) -> bool:
    return block.merkle_root == block.calculate_merkle_root()





def compute_sha256(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
    
def _is_readonly(path: str) -> bool:
    """True if file is read-only (Windows RO bit or POSIX no-write)."""
    try:
        import ctypes
        attrs = ctypes.windll.kernel32.GetFileAttributesW(path)
        return bool(attrs & 0x1)  # FILE_ATTRIBUTE_READONLY
    except Exception:
        return not os.access(path, os.W_OK)

def audit_ledger() -> bool:
    print("🔎 Ledger Audit Report")
    print("------------------------")
    led_dir = "ledger"
    if not os.path.isdir(led_dir):
        print("❌ No ledger directory found.")
        return False

    ledgers = [f for f in os.listdir(led_dir) if f.startswith("blk") and f.endswith(".jsonl")]
    if not ledgers:
        print("❌ No ledger files found.")
        return False

    overall_ok = True

    for ledger_file in sorted(ledgers):
        path = os.path.join(led_dir, ledger_file)
        print(f"\n📄 Auditing {ledger_file}...")
        sha = compute_sha256(path)  # reuse your existing helper
        print(f"📏 SHA256: {sha}")
        ts = time.ctime(os.path.getmtime(path))
        print(f"⏱ Last modified: {ts}")

        # per-file lock check (blk000NN.jsonl → blk000NN.hash)
        lock = os.path.splitext(path)[0] + ".hash"
        print("🔍 Verifying hash lock...")
        if os.path.exists(lock):
            stored = open(lock, "r", encoding="utf-8").read().strip().lower()
            if stored == sha:
                print("🔐 Hash lock verified.")
                print("✅ Hash matches stored lock.")
            else:
                print("❌ Hash lock mismatch.")
                overall_ok = False
        else:
            print("❌ Hash lock file missing.")
            overall_ok = False

        print("🔐 Checking file permissions...")
        print("✅ File is read-only." if _is_readonly(path) else "⚠️ File is writable.")

    # commitment + curvature signature audit (your existing block checks)
    try:
        with open("keypair.json", "r") as f:
            key_data = json.load(f)
        keypair = CurvatureKeyPair(n=key_data["n"])
        keypair.psi_0 = cp.asarray(key_data["psi_0"])
        keypair.psi_star = cp.asarray(key_data["psi_star"])
        keypair.commitment = key_data["commitment"]
        print(f"\n🔗 Commitment: {keypair.commitment}")
    except Exception as e:
        print(f"\n❌ Error loading keypair for audit: {e}")
        return False

    blocks = load_all_blocks()
    print(f"📦 Total blocks in ledger: {len(blocks)}")
    last_ts = 0
    for block in blocks:
        if block.timestamp < last_ts:
            print(f"⚠️ Block #{block.index} has a timestamp earlier than previous.")
        last_ts = block.timestamp

        msg = next((m for m in block.messages if m.startswith("message: ")), None)
        sig = next((m for m in block.messages if m.startswith("signature: ")), None)
        com = next((m for m in block.messages if m.startswith("commitment: ")), None)

        if not msg or not sig or not com:
            print(f"❌ Block #{block.index} is missing curvature metadata.")
            overall_ok = False
            continue

        if com[len("commitment: "):] != keypair.commitment:
            print(f"❌ Block #{block.index} commitment mismatch.")
            overall_ok = False
            continue

        if not keypair.verify(msg[len("message: "):], sig[len("signature: "):]):
            print(f"❌ Block #{block.index} signature invalid.")
            overall_ok = False
        else:
            print(f"✅ Block #{block.index} curvature signature valid.")

    return overall_ok