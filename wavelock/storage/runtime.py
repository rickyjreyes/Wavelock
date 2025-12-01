from pathlib import Path
import os

def get_runtime_dir():
    xdg = os.getenv("XDG_DATA_HOME")
    if xdg:
        root = Path(xdg) / "wavelock"
    else:
        root = Path.home() / ".wavelock"
    root.mkdir(parents=True, exist_ok=True)
    return root

RUNTIME_DIR = get_runtime_dir()

LEDGER_DIR = RUNTIME_DIR / "ledger"
LEDGER_DIR.mkdir(exist_ok=True)

COMMITMENTS_DIR = RUNTIME_DIR / "commitments"
COMMITMENTS_DIR.mkdir(exist_ok=True)

KEYPAIR_FILE = RUNTIME_DIR / "keypair.json"
USERS_FILE = RUNTIME_DIR / "users.json"
PEERS_FILE = RUNTIME_DIR / "peers.json"
TRUST_FILE = RUNTIME_DIR / "trusted_commitments.json"
