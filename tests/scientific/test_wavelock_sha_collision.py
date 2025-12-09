# ultimate_attack_sha_collision_cluster.py
# ============================================================
# SHA256-lifted brute-force collision cluster:
#   Enumerate seeds in parallel, compute ψ*, hash serialize(ψ*),
#   and detect any collisions in the hash space.
#   This is the ultimate empirical collision test.
# ============================================================

import os
import math
import hashlib
import cupy as cp
import numpy as np
import multiprocessing as mp
from collections import defaultdict

from wavelock.chain.WaveLock import CurvatureKeyPair, _serialize_commitment_v2

def commit_cp(psi_cp):
    raw = _serialize_commitment_v2(psi_cp)
    return hashlib.sha256(raw).hexdigest()

def worker(seed_start, seed_end, n, result_queue, progress_every=1000):
    local_table = {}
    collisions = []

    for seed in range(seed_start, seed_end):
        kp = CurvatureKeyPair(n=n, seed=seed)
        psi_star = kp.psi_star
        h = commit_cp(psi_star)

        if h in local_table and local_table[h] != seed:
            collisions.append((h, local_table[h], seed))
        else:
            local_table[h] = seed

        if (seed - seed_start) % progress_every == 0:
            print(f"[Worker {seed_start}-{seed_end}] seed={seed}, local_size={len(local_table)}")

    result_queue.put((local_table, collisions))

def run_sha_collision_cluster(n=6, seed_max=100000, workers=4):
    print(f"=== SHA256 Collision Cluster: n={n}, seeds=[0,{seed_max}) with {workers} workers ===")

    chunk = seed_max // workers
    processes = []
    result_queue = mp.Queue()

    for w in range(workers):
        start = w * chunk
        end = seed_max if w == workers - 1 else (w + 1) * chunk
        p = mp.Process(target=worker, args=(start, end, n, result_queue))
        p.start()
        processes.append(p)

    global_table = {}
    global_collisions = []

    for _ in range(workers):
        local_table, collisions = result_queue.get()
        # merge local tables
        for h, seed in local_table.items():
            if h in global_table and global_table[h] != seed:
                global_collisions.append((h, global_table[h], seed))
            else:
                global_table[h] = seed
        global_collisions.extend(collisions)

    for p in processes:
        p.join()

    print("\n=== Collision Search Result ===")
    print(f"Checked {seed_max} seeds.")
    if global_collisions:
        print(f"!!! COLLISIONS FOUND: {len(global_collisions)}")
        for h, s1, s2 in global_collisions[:10]:
            print(f"hash={h}, seeds={s1},{s2}")
    else:
        print("No collisions found in searched range.")













if __name__ == "__main__":
    run_sha_collision_cluster(n=6, seed_max=100000, workers=4)
