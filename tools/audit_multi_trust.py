import os, json, hashlib, numpy as np
try:
    import cupy as cp
except Exception:
    cp = None

from chain.chain_utils import load_all_blocks
from chain.WaveLock import CurvatureKeyPair, SCHEMA_V2, _serialize_commitment_v2

ROOT = os.getcwd()
TRUST = os.path.join(ROOT, "trusted_commitments.json")
PUB   = os.path.join(ROOT, "commitments")

trust = set(json.load(open(TRUST))) if os.path.exists(TRUST) else set()
blocks = load_all_blocks()

def load_public_psi(commitment: str):
    if ":" in commitment:
        key = commitment.replace(":", "_").lower()
    else:
        key = f"{SCHEMA_V2}:{commitment}".replace(":", "_").lower()
    path = os.path.join(PUB, f"{key}.npz")
    if not os.path.exists(path) or cp is None:
        return None
    data = np.load(path, allow_pickle=False)
    psi = data["psi_star"]
    return cp.asarray(psi)

ok = 0
print("?? Multi-trust Audit")
print("---------------------")
print(f"Trusted commitments: {len(trust)}")
for b in blocks:
    msg = next((m for m in b.messages if m.startswith("message: ")), None)
    sig = next((m for m in b.messages if m.startswith("signature: ")), None)
    com = next((m for m in b.messages if m.startswith("commitment: ")), None)
    print(f"\n Block #{b.index} | {b.hash[:12]}...")
    if not (msg and sig and com):
        print("   Missing curvature metadata")
        continue
    msg = msg[len("message: "):]
    sig = sig[len("signature: "):]
    com = com[len("commitment: "):].strip()

    # commitment allow-list check
    if com not in trust:
        print("   Commitment not in trusted_commitments.json")
        continue
    print("   Commitment is trusted")

    # optional full verify if ?* published
    psi = load_public_psi(com)
    if psi is None:
        print("   No published ?* (skipping full verify)")
        ok += 1
        continue

    kp = CurvatureKeyPair(n=4)
    kp.commitment = com
    kp.psi_star = psi
    kp.psi_0 = cp.zeros_like(psi) if cp is not None else None
    if kp.verify(msg, sig):
        print("   Curvature signature valid (full WLv2/SIGv2)")
        ok += 1
    else:
        print("   Curvature signature INVALID")

print(f"\n Audited {len(blocks)} blocks; {ok} passed policy.")
