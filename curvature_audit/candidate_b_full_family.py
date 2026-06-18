"""Phase CC-2 Part V -- full 47-state Phase 8J regression for Candidate B.

Reconstructs the complete Phase 8J zero-preimage family (47 states) and computes
Candidate B trajectory commitments for all of them. Acceptance condition:

    psi^(i) != psi^(j)  =>  C_{T,B}^(i) != C_{T,B}^(j)   for all 47 known states.

This is NOT general collision resistance; it is separation of the known
terminal-state collision family. Records full pairwise Hamming statistics and
clustering by orbit, sign, amplitude, and eigenvalue (r) class.
"""

from __future__ import annotations

import time

import numpy as np

from wavelock.curvature_capacity_v1 import optimized as bopt
from wavelock.curvature_capacity import optimized as aopt
from . import _common as C
from .phase_cc1_family import enumerate_full_family

P = bopt._P
N = bopt._N


def main(seed: int = 96000) -> dict:
    t0 = time.perf_counter()
    print("  enumerating 47-state family ...")
    states, stats = enumerate_full_family()
    assert stats["total_states"] == 47, stats

    print("  computing Candidate B trajectory digests ...")
    digs_b = []
    digs_a = []
    for s in states:
        psi = np.array(s["cells"], dtype=np.int64).reshape(N, N)
        digs_b.append(bopt.trajectory_digest(psi).hex())
        digs_a.append(aopt.trajectory_digest(psi).hex())

    n = len(states)
    uniq_b = len(set(digs_b))

    # pairwise Hamming statistics (Candidate B)
    min_hd = 256
    max_hd = 0
    total = 0
    npairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            hd = C.hamming_bytes(bytes.fromhex(digs_b[i]), bytes.fromhex(digs_b[j]))
            min_hd = min(min_hd, hd)
            max_hd = max(max_hd, hd)
            total += hd
            npairs += 1

    # clustering by r-class (eigenvalue), amplitude sign label, and verifying
    # within-cluster separation too
    by_r: dict[int, list[int]] = {}
    by_amp: dict[str, list[int]] = {}
    for i, s in enumerate(states):
        by_r.setdefault(s["r"], []).append(i)
        by_amp.setdefault(s["amplitude_label"], []).append(i)

    def cluster_min_hd(idxs):
        if len(idxs) < 2:
            return None
        m = 256
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                m = min(m, C.hamming_bytes(bytes.fromhex(digs_b[idxs[a]]),
                                           bytes.fromhex(digs_b[idxs[b]])))
        return m

    r_clusters = {str(r): {"size": len(ix), "min_intra_hamming": cluster_min_hd(ix)}
                  for r, ix in sorted(by_r.items())}
    amp_clusters = {k: {"size": len(ix), "min_intra_hamming": cluster_min_hd(ix)}
                    for k, ix in sorted(by_amp.items())}

    # cross-candidate: are A and B digests different for the same state? (domain sep)
    a_vs_b_same = sum(1 for i in range(n) if digs_a[i] == digs_b[i])

    acceptance = uniq_b == n and min_hd > 0

    out = {
        "artifact": "candidate_b_full_family_binding",
        "description": "Full 47-state Phase 8J trajectory-binding regression for Candidate B",
        "metadata": C.env_metadata(),
        "seed": seed,
        "family_stats": stats,
        "candidate_b": {
            "n_states": n,
            "n_distinct_digests": uniq_b,
            "all_distinct": uniq_b == n,
            "min_pairwise_hamming": min_hd,
            "max_pairwise_hamming": max_hd,
            "mean_pairwise_hamming": round(total / npairs, 2),
            "n_pairs": npairs,
        },
        "clustering_by_eigenvalue_r": r_clusters,
        "clustering_by_amplitude_label": amp_clusters,
        "cross_candidate": {
            "states_with_identical_A_and_B_digest": a_vs_b_same,
            "note": "the zero state gives identical A/B digests (j_A=j_B=0 when psi=0); "
                    "all nonzero states differ between A and B.",
        },
        "acceptance_condition_met": acceptance,
        "acceptance_statement": (
            "psi^(i) != psi^(j) => C_{T,B}^(i) != C_{T,B}^(j) for all 47 known "
            "Phase 8J zero-collapse states. This is separation of the known "
            "terminal-state collision family, NOT general collision resistance."
        ),
        "per_state": [
            {"id": states[i]["id"], "r": states[i]["r"],
             "amplitude_label": states[i]["amplitude_label"],
             "digest_b": digs_b[i]}
            for i in range(n)
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("candidate_b_full_family_binding.json", out)
    print(f"    distinct={uniq_b}/{n}, min_HD={min_hd}, max_HD={max_hd}, "
          f"mean_HD={out['candidate_b']['mean_pairwise_hamming']}")
    print(f"    acceptance met: {acceptance}")
    print(f"  saved candidate_b_full_family_binding.json ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
