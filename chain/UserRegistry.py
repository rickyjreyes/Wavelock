import json, os
import cupy as cp
try:
    # when used as package (recommended: python -m chain.cli ...)
    from chain.WaveLock import CurvatureKeyPair
except Exception:
    # fallback when importing top-level
    from WaveLock import CurvatureKeyPair

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REG_PATH = os.path.join(ROOT, "users.json")

def _load():
    if not os.path.exists(REG_PATH):
        return {}
    with open(REG_PATH, "r") as f:
        return json.load(f)

def _save(reg):
    with open(REG_PATH, "w") as f:
        json.dump(reg, f, indent=2)

class UserRegistry:
    def __init__(self):
        self.reg = _load()

    def add_user(self, user_id: str, n: int = 4, seed: int | None = None):
        if user_id in self.reg:
            raise ValueError(f"user '{user_id}' already exists")
        kp = CurvatureKeyPair(n=n, seed=seed)
        self.reg[user_id] = {
            "n": n,
            "commitment": kp.commitment,
            "psi_0": cp.asnumpy(kp.psi_0).tolist(),
            "psi_star": cp.asnumpy(kp.psi_star).tolist(),
        }
        _save(self.reg)
        return kp.commitment

    def get_keypair(self, user_id: str) -> CurvatureKeyPair:
        if user_id not in self.reg:
            raise ValueError(f"User '{user_id}' not found")
        meta = self.reg[user_id]
        n = int(meta.get("n", 4))
        kp = CurvatureKeyPair(n=n, seed=None)  # overwritten by restored state
        kp.psi_0 = cp.asarray(meta["psi_0"], dtype=cp.float64)
        kp.psi_star = cp.asarray(meta["psi_star"], dtype=cp.float64)
        kp.commitment = str(meta["commitment"])
        return kp

def sign_message_with_user(user_id: str, message: str, out_path: str):
    reg = UserRegistry()
    kp = reg.get_keypair(user_id)
    sig = kp.sign(message)
    payload = {
        "user_id": user_id,
        "message": message,
        "signature": sig,
        "commitment": kp.commitment,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path

def verify_signed_message(path: str) -> bool:
    with open(path, "r") as f:
        data = json.load(f)
    user_id   = data["user_id"]
    message   = data["message"]
    signature = data["signature"]
    commit_in = data.get("commitment", "")
    reg = UserRegistry()
    kp = reg.get_keypair(user_id)
    if commit_in and str(kp.commitment) != str(commit_in):
        return False
    return kp.verify(message, signature)
