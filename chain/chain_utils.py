import os, time, json, hashlib, sys
from pathlib import Path
import cupy as cp
import matplotlib.pyplot as plt

from chain.CurvaChain import CurvaChain
from chain.WaveLock import CurvatureKeyPair, symbolic_verifier, load_quantum_keys
from chain.Block import Block

# ============================================================
# 0. Ledger Path (GLOBAL, ABSOLUTE, CANONICAL)
# ============================================================

# project root = folder ABOVE /chain
ROOT = Path(__file__).resolve().parents[1]

# The real global ledger directory used by ALL nodes, tests, etc.
LEDGER_DIR = ROOT / "ledger"
LEDGER_DIR.mkdir(exist_ok=True)

# Main rotating ledger file
LEDGER_FILE = LEDGER_DIR / "blk00000.jsonl"


# ============================================================
# 1. Internal helpers
# ============================================================

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
    """Return next blkNNNNN.jsonl path inside canonical ledger dir."""
    LEDGER_DIR.mkdir(exist_ok=True)
    files = sorted(LEDGER_DIR.glob("blk*.jsonl"))
    if not files:
        return LEDGER_DIR / "blk00000.jsonl"

    last_idx = max(int(f.stem.replace("blk", "")) for f in files)
    return LEDGER_DIR / f"blk{last_idx+1:05d}.jsonl"


def _set_readonly(p: Path):
    """Windows + POSIX compatible read-only helper."""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetFileAttributesW(str(p), 0x1)
        else:
            os.chmod(p, 0o444)
    except Exception:
        pass



# ============================================================
# 2. Save / Load Full Chains (chain.json)
# ============================================================

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
            "block_type": block.block_type,
            "meta": block.meta,
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



# ============================================================
# 3. Visualization Helpers
# ============================================================

def visualize_psi(psi_star):
    psi_np = cp.asnumpy(psi_star)
    entropy = -cp.sum((psi_star**2 / cp.sum(psi_star**2))
                      * cp.log(psi_star**2 / cp.sum(psi_star**2) + 1e-12)).get()

    plt.figure(figsize=(6, 5))
    plt.imshow(psi_np, cmap="viridis")
    plt.colorbar(label="ψ* magnitude")
    plt.title(f"ψ* Heatmap | Entropy: {entropy:.4f}")
    plt.tight_layout()
    plt.show()



# ============================================================
# 4. Tamper Test
# ============================================================

def tamper_and_test(keypair: CurvatureKeyPair):
    tampered = keypair.psi_star.copy()
    tampered[0, 0] += 0.5
    reference = keypair.evolve(cp.asarray(keypair.psi_0.copy()), keypair.n)
    result = symbolic_verifier(tampered, reference)
    print("🧪 Tamper test:")
    print(f"    Tampered accepted? {'✅' if result else '❌'}")
    return result



# ============================================================
# 5. Ledger I/O (block-level)
# ============================================================

def save_block_to_disk(block: Block):
    """Append block JSON to canonical rotating ledger file."""
    target = LEDGER_FILE
    target.parent.mkdir(exist_ok=True)

    try:
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(block.to_dict()) + "\n")
    except PermissionError:
        # target is read-only, rotate ledger
        target = _next_ledger_path()
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(block.to_dict()) + "\n")

    lock_path = _write_lock(target)
    if os.getenv("WL_LOCK_AFTER_WRITE") == "1":
        _set_readonly(target)
        _set_readonly(lock_path)


def load_all_blocks() -> list[Block]:
    """Load ALL blocks from ALL blkNNNNN.jsonl files."""
    blocks: list[Block] = []
    led = LEDGER_DIR

    if not led.exists():
        return blocks

    for p in sorted(led.glob("blk*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                block = Block(
                    index=data["index"],
                    messages=data["messages"],
                    previous_hash=data["previous_hash"],
                    difficulty=data.get("difficulty", 4),
                    timestamp=float(data["timestamp"]),
                    nonce=data["nonce"],
                    block_hash=data["hash"],
                    merkle_root=data.get("merkle_root"),
                    block_type=data.get("block_type", "GENERIC"),
                    meta=data.get("meta", {}),
                )
                block.hash = data["hash"]   # ⭐ REQUIRED ⭐
                blocks.append(block)

    return blocks


def reset_ledger(force: bool = False):
    """Delete all ledger files inside canonical ledger directory."""
    led = LEDGER_DIR
    if not force:
        ans = input("⚠️  Delete entire blockchain ledger? Type 'yes' to confirm: ").strip().lower()
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



# ============================================================
# 6. Verification + Integrity Checks
# ============================================================

def verify_merkle_root(block: Block) -> bool:
    return block.merkle_root == block.calculate_merkle_root()


def verify_chain(blocks=None, keypair=None):
    if blocks is None:
        blocks = load_all_blocks()

    if not blocks:
        print("⚠️ No blocks found.")
        return False

    # Keypair loading fallback
    if keypair is None:
        try:
            keypair = load_quantum_keys()
        except Exception:
            with open("keypair.json", "r") as f:
                data = json.load(f)
            kp = CurvatureKeyPair(n=data["n"])
            kp.psi_0 = cp.asarray(data["psi_0"])
            kp.psi_star = cp.asarray(data["psi_star"])
            kp.commitment = data["commitment"]
            keypair = kp

    for i, block in enumerate(blocks):

        # Hash check
        if block.hash != block.calculate_hash(block.nonce):
            print(f"❌ Block #{block.index} hash mismatch.")
            return False

        # Difficulty check
        if not block.hash.startswith("0" * block.difficulty):
            print(f"❌ Block #{block.index} does not meet difficulty.")
            return False

        # Previous hash linkage
        if i > 0 and block.previous_hash != blocks[i - 1].hash:
            print(f"❌ Block #{block.index} previous_hash mismatch.")
            return False

        # Extract message/signature/commitment
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



# ============================================================
# 7. Ledger Auditing
# ============================================================

def compute_sha256(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _is_readonly(path: str) -> bool:
    try:
        import ctypes
        attrs = ctypes.windll.kernel32.GetFileAttributesW(path)
        return bool(attrs & 0x1)
    except Exception:
        return not os.access(path, os.W_OK)


def audit_ledger() -> bool:
    print("🔎 Ledger Audit Report")
    print("------------------------")

    led = LEDGER_DIR
    if not led.exists():
        print("❌ No ledger directory found.")
        return False

    ledgers = list(sorted(led.glob("blk*.jsonl")))
    if not ledgers:
        print("❌ No ledger files found.")
        return False

    overall_ok = True

    for ledger_path in ledgers:
        print(f"\n📄 Auditing {ledger_path.name}...")
        sha = compute_sha256(ledger_path)
        print(f"📏 SHA256: {sha}")
        ts = time.ctime(os.path.getmtime(ledger_path))
        print(f"⏱ Last modified: {ts}")

        # hash lock check
        lock = ledger_path.with_suffix(".hash")
        print("🔍 Verifying hash lock...")
        if lock.exists():
            stored = lock.read_text().strip().lower()
            if stored == sha:
                print("🔐 Hash lock verified.")
            else:
                print("❌ Hash lock mismatch.")
                overall_ok = False
        else:
            print("❌ Hash lock file missing.")
            overall_ok = False

        print("🔐 Checking file permissions...")
        print("⚠️ File is writable." if not _is_readonly(str(ledger_path)) else "✅ File is read-only.")

    # curvature signature audit
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
            print(f"❌ Block #{block.index} missing curvature metadata.")
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
