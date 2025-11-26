# tools/reissue_trust.py
import os, sys, json, hashlib, numpy as np

# --- make imports work whether you have top-level modules or a 'chain/' package ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    # top-level layout (current repo default)
    from WaveLock import (
        load_quantum_keys, _commit_header, _serialize_commitment_v2,
        SCHEMA_V2, _canonical_json
    )
except ImportError:
    # package layout fallback (if you later convert to a real package)
    from chain.WaveLock import (
        load_quantum_keys, _commit_header, _serialize_commitment_v2,
        SCHEMA_V2, _canonical_json
    )

PUBDIR = os.path.join(ROOT, "commitments")
TRUST  = os.path.join(ROOT, "trusted_commitments.json")

def main():
    # adjust if you keep keys in a different file
    keyfile = os.path.join(ROOT, "psi_keypair.json")
    kp = load_quantum_keys(keyfile)

    v2_hex = hashlib.sha256(_serialize_commitment_v2(kp.psi_star)).hexdigest()
    commitment = f"{SCHEMA_V2}:{v2_hex}"

    os.makedirs(os.path.dirname(TRUST), exist_ok=True)
    with open(TRUST, "w") as f:
        json.dump([commitment], f, indent=2)
    print("✅ trusted_commitments.json updated:", commitment)

    # publish ψ* snapshot so the server can fully verify curvature (optional but recommended)
    os.makedirs(PUBDIR, exist_ok=True)
    header_json = _canonical_json(_commit_header(kp.psi_star)).decode("utf-8")
    key = commitment.replace(":", "_").lower()
    np.savez(os.path.join(PUBDIR, f"{key}.npz"),
             psi_star=kp.psi_star.get(),
             header_json=np.array(header_json))
    print("📦 published commitments/", f"{key}.npz")

if __name__ == "__main__":
    main()
