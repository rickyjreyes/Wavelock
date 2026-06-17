"""Audit-only toy systems: exhaustive enumeration of small finite-field variants.

For tractable (N, p) the ENTIRE state space of the one-round state-transformation
is enumerated to measure injectivity, surjectivity, preimage multiplicity,
fixed points, and the functional-graph cycle/component structure.

This is diagnostic only. Toy degeneracies (especially duplicated neighbors on
N=1 and N=2 toroidal lattices, where roll(+1) == roll(-1)) are reported and must
NOT be extrapolated directly to the full N=16 system; the goal is to expose
structural mechanisms, not to predict full-system behavior.

The toy map operates purely on the state (no message layer), so it is
well-defined for any prime p and lattice side N.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ToyParams:
    N: int
    p: int
    D: int
    a: int
    b: int
    T: int = 1


def _one_round(state: Tuple[int, ...], P: ToyParams) -> Tuple[int, ...]:
    """One round of the F_p Allen-Cahn-derived map on an N*N toroidal lattice."""
    N, p = P.N, P.p
    grid = [list(state[i * N:(i + 1) * N]) for i in range(N)]
    out = [0] * (N * N)
    pm4 = (p - 4) % p
    for i in range(N):
        for j in range(N):
            psi = grid[i][j]
            sq = (psi * psi) % p
            bm = (P.b + (p - sq)) % p
            react = (P.a * ((psi * bm) % p)) % p
            lap = (
                grid[(i + 1) % N][j] + grid[(i - 1) % N][j]
                + grid[i][(j + 1) % N] + grid[i][(j - 1) % N]
                + (pm4 * psi) % p
            ) % p
            out[i * N + j] = (psi + (P.D * lap) % p + react) % p
    return tuple(out)


def _map(state, P: ToyParams):
    for _ in range(P.T):
        state = _one_round(state, P)
    return state


def neighbor_degeneracy(P: ToyParams) -> str:
    if P.N == 1:
        return "N=1: all 4 neighbors are the same cell; diffusion term vanishes (lap=0)."
    if P.N == 2:
        return "N=2: roll(+1)==roll(-1); each axis contributes the other cell twice."
    return "none"


def enumerate_toy(P: ToyParams) -> dict:
    """Exhaustively enumerate the toy state space and characterize the map."""
    N, p = P.N, P.p
    ncells = N * N
    nstates = p ** ncells
    if nstates > 200_000:
        raise ValueError(f"state space too large to enumerate: {nstates}")

    states = list(itertools.product(range(p), repeat=ncells))
    index = {s: i for i, s in enumerate(states)}
    succ = [0] * nstates           # functional graph: succ[i] = index of f(state_i)
    preimage_count = [0] * nstates
    fixed_points = []

    for i, s in enumerate(states):
        t = _map(s, P)
        j = index[t]
        succ[i] = j
        preimage_count[j] += 1
        if j == i:
            fixed_points.append(i)

    image = set(succ)
    image_size = len(image)
    unreachable = nstates - image_size
    max_preimage = max(preimage_count)
    # preimage-count distribution
    dist = {}
    for c in preimage_count:
        dist[c] = dist.get(c, 0) + 1

    injective = (image_size == nstates)
    surjective = (image_size == nstates)  # finite endo: injective iff surjective

    # cycle decomposition via functional-graph walk
    cycles = _find_cycles(succ)
    components = _count_components(succ)

    return {
        "params": {"N": N, "p": p, "D": P.D, "a": P.a, "b": P.b, "T": P.T},
        "n_states": nstates,
        "image_size": image_size,
        "unreachable_outputs": unreachable,
        "image_fraction": image_size / nstates,
        "max_preimage_multiplicity": max_preimage,
        "preimage_count_distribution": {str(k): v for k, v in sorted(dist.items())},
        "num_fixed_points": len(fixed_points),
        "num_cycles": len(cycles),
        "cycle_length_distribution": _hist([len(c) for c in cycles]),
        "num_functional_components": components,
        "injective": injective,
        "surjective": surjective,
        "one_round_is": ("bijection" if injective else "neither injective nor surjective"),
        "neighbor_degeneracy": neighbor_degeneracy(P),
    }


def _find_cycles(succ: List[int]) -> List[List[int]]:
    n = len(succ)
    color = [0] * n  # 0=unseen, 1=on-stack, 2=done
    cycles = []
    for start in range(n):
        if color[start] != 0:
            continue
        path = []
        pos = {}
        node = start
        while color[node] == 0:
            color[node] = 1
            pos[node] = len(path)
            path.append(node)
            node = succ[node]
        if color[node] == 1:  # found a fresh cycle
            cycles.append(path[pos[node]:])
        for x in path:
            color[x] = 2
    return cycles


def _count_components(succ: List[int]) -> int:
    # weakly-connected components of the functional graph (undirected union-find)
    n = len(succ)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j in enumerate(succ):
        union(i, j)
    return len({find(i) for i in range(n)})


def _hist(values):
    h = {}
    for v in values:
        h[v] = h.get(v, 0) + 1
    return {str(k): v for k, v in sorted(h.items())}


# default toy grid for the audit
DEFAULT_TOY_GRID = [
    ToyParams(N=1, p=5, D=5, a=3, b=2, T=1),
    ToyParams(N=1, p=7, D=5, a=3, b=4, T=1),
    ToyParams(N=1, p=11, D=5, a=3, b=7, T=1),
    ToyParams(N=1, p=13, D=5, a=3, b=8, T=1),
    ToyParams(N=2, p=5, D=5, a=3, b=2, T=1),
    ToyParams(N=2, p=7, D=5, a=3, b=4, T=1),
    ToyParams(N=2, p=11, D=5, a=3, b=7, T=1),
    ToyParams(N=2, p=13, D=5, a=3, b=8, T=1),
    # a couple of multi-round toy points
    ToyParams(N=2, p=7, D=5, a=3, b=4, T=4),
    ToyParams(N=2, p=11, D=5, a=3, b=7, T=4),
]
