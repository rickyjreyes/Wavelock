"""Read-only historical observations: run with cwd/PYTHONPATH at tag v0.2.0.

Example from a separate v0.2.0 checkout:
  PYTHONPATH=. /path/to/python /path/to/hardening/audit/reproduce_v020_contract.py
This intentionally uses the historical serializer, never the normative API.
"""
import json
import subprocess

import numpy as np
from wavelock.chain.Block import Block
from wavelock.chain.Wavelock_numpy import CurvatureKeyPairV3, _serialize_commitment

BASELINE = "9430442ae93e26c192d9be1f5ee9137369f5df37"


def main():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != BASELINE:
        raise SystemExit("run this probe from the detached v0.2.0 baseline checkout")
    evidence = {"baseline_commit": head, "historical_api": "Wavelock_numpy._serialize_commitment"}
    with np.errstate(all="ignore"):
        for name, value in (("nan", np.nan), ("pos_inf", np.inf), ("neg_inf", -np.inf)):
            state = np.ones((4, 4))
            state[0, 0] = value
            evidence[name] = {"emitted_bytes": len(_serialize_commitment(state))}
    positive = np.zeros((4, 4))
    negative = positive.copy()
    negative[0, 0] = -0.0
    evidence["signed_zero_diverges"] = _serialize_commitment(positive) != _serialize_commitment(negative)
    evidence["integer_seed_42"] = CurvatureKeyPairV3(n=4, seed=42).commitment
    def block(index, timestamp):
        return Block(index, ["same"], "0"*64, difficulty=0, timestamp=timestamp,
                     nonce=0, block_hash="0"*64)
    left, right = block(1, "23"), block(12, "3")
    evidence["header_ambiguity"] = {"left": [1, "23"], "right": [12, "3"],
        "same_hash": left.calculate_hash(0) == right.calculate_hash(0), "hash": left.calculate_hash(0)}
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
