import os, json, hashlib, numpy as np
try:
    import cupy as cp
except Exception:
    cp = None

from chain.UserRegistry import UserRegistry
from chain.WaveLock import _serialize_commitment_v2, _commit_header, _canonical_json, SCHEMA_V2, load_quantum_keys

ROOT = os.getcwd()
PUBDIR = os.path.join(ROOT, "commitments")
TRUST  = os.path.join(ROOT, "trusted_commitments.json")
os.makedirs(PUBDIR, exist_ok=True)

# load existing trust list
trust = []
if os.path.exists(TRUST):
    try: trust = json.load(open(TRUST))
    except: trust = []

def publish_commit(psi, label):
    hexv = hashlib.sha256(_serialize_commitment_v2(psi)).hexdigest()
    commit = f"{SCHEMA_V2}:{hexv}"
    key = commit.replace(":", "_").lower()
    path = os.path.join(PUBDIR, f"{key}.npz")
    if not os.path.exists(path):
        np.savez(path,
                 psi_star=psi.get() if hasattr(psi, 'get') else np.asarray(psi),
                 header_json=_canonical_json(_commit_header(psi)).decode("utf-8"))
        print(f"[publish] {label}: {path}")
    else:
        print(f"[publish] {label}: already exists")
    if commit not in trust:
        trust.append(commit)
        print(f"[trust] add {commit}")
    return commit

# (A) publish all users from users.json
try:
    reg = UserRegistry()
    for uid, meta in getattr(reg, "reg", {}).items():
        psi = cp.asarray(meta["psi_star"]) if cp is not None else np.asarray(meta["psi_star"])
        publish_commit(psi, f"user:{uid}")
except Exception as e:
    print("[users] skipped:", e)

# (B) legacy key (psi_keypair.json), if present
try:
    kp = load_quantum_keys("psi_keypair.json")
    publish_commit(kp.psi_star, "legacy")
except Exception as e:
    print("[legacy] skipped:", e)

# save trust list
json.dump(trust, open(TRUST, "w"), indent=2)
print(f"[done] trusted_commitments.json entries: {len(trust)}")
