"""
seed_bruteforce.py — brute-forcing legacy small-integer seeds.

Legacy WaveLock keypairs are routinely created with tiny integer seeds (the
examples use ``seed=12``, ``seed=42``, ``seed=123``). The commitment is a
deterministic function of the seed, so an attacker who sees a commitment can
simply enumerate seeds, regenerate ψ★, and recover the full secret state —
after which legacy forgery (see forge_from_snapshot.py) is trivial.

This module:
  * :func:`bruteforce_legacy_seed` — recovers a small integer seed from a
    published legacy commitment by enumeration. SUCCEEDS for tiny seeds.
  * :func:`ots_seed_search_is_infeasible` — quantifies why the same approach is
    hopeless against WaveLock-OTS (256-bit os.urandom seeds): there is no small
    space to enumerate.

Used as a regression test: tiny legacy seeds remain brute-forceable (proving
the old design's seed handling is unsafe), while WaveLock-OTS rejects sub-128-bit
seeds outright and uses 256-bit entropy by default.
"""

from __future__ import annotations

from typing import Optional

from wavelock.chain.WaveLock import CurvatureKeyPair


def make_legacy_commitment(n: int = 4, seed: int = 123) -> str:
    return CurvatureKeyPair(n=n, seed=seed, test_mode=True).commitment


def bruteforce_legacy_seed(target_commitment: str, n: int = 4,
                           max_seed: int = 100_000) -> Optional[int]:
    """Recover a legacy integer seed by enumerating [0, max_seed).

    Returns the recovered seed or None. For the documented example seeds this
    returns almost immediately.
    """
    for candidate in range(max_seed):
        kp = CurvatureKeyPair(n=n, seed=candidate, test_mode=True)
        if kp.commitment == target_commitment:
            return candidate
    return None


def ots_seed_search_is_infeasible(entropy_bits: int = 256) -> dict:
    """Return the (astronomical) search-space size for a WaveLock-OTS seed.

    WaveLock-OTS uses os.urandom(entropy_bits//8) with a hard 128-bit floor, so
    there is no small integer space to enumerate. We return the numbers rather
    than attempt the search.
    """
    return {
        "entropy_bits": entropy_bits,
        "search_space": 2 ** entropy_bits,
        "feasible": False,
    }


if __name__ == "__main__":
    commitment = make_legacy_commitment(seed=123)
    recovered = bruteforce_legacy_seed(commitment, max_seed=200)
    print(f"[legacy] recovered seed: {recovered} (expected 123)")
    info = ots_seed_search_is_infeasible()
    print(f"[OTS] seed search space: 2^{info['entropy_bits']} — feasible: {info['feasible']}")
