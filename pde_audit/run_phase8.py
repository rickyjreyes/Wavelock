"""Phase 8 orchestrator: run all adversarial analyses and write an artifact index.

Each sub-analysis is independently runnable (python -m pde_audit.<module>); this
driver runs them in sequence with fixed seeds and records a machine-readable
index under pde_audit/artifacts/INDEX.json.

Usage:
    python -m pde_audit.run_phase8 [--quick] [phase ...]

Phases: avalanche state_map squeeze distinguishers symmetry collision preimage
        algebraic parameter baselines
"""

from __future__ import annotations

import sys
import time

from . import _harness as H
from . import (avalanche, state_map, squeeze_analysis, distinguishers,
               symmetry_attacks, collision_scaling, preimage_attacks,
               algebraic_analysis, parameter_sweep, baselines,
               eigenmode_collisions)

PHASES = {
    "avalanche": avalanche.main,
    "state_map": state_map.main,
    "squeeze": squeeze_analysis.main,
    "distinguishers": distinguishers.main,
    "symmetry": symmetry_attacks.main,
    "collision": collision_scaling.main,
    "preimage": preimage_attacks.main,
    "algebraic": algebraic_analysis.main,
    "parameter": parameter_sweep.main,
    "baselines": baselines.main,
    "eigenmode": eigenmode_collisions.main,
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    selected = [a for a in argv if not a.startswith("--")]
    phases = selected or list(PHASES)
    H.self_test()  # confirm the audit variant matches the normative primitive

    index = {"metadata": H.env_metadata(), "phases": {}, "artifacts": []}
    t0 = time.perf_counter()
    for name in phases:
        print(f"\n=== {name} ===")
        tt = time.perf_counter()
        res = PHASES[name]()
        index["phases"][name] = {
            "runtime_s": round(time.perf_counter() - tt, 2),
            "verdict_inputs": _extract_summary(name, res),
        }
    index["total_runtime_s"] = round(time.perf_counter() - t0, 2)
    import os
    index["artifacts"] = sorted(
        f for f in os.listdir(H.ARTIFACT_DIR) if f.endswith(".json"))
    H.save_artifact("INDEX.json", index)
    print("\nWrote artifacts/INDEX.json; total", index["total_runtime_s"], "s")
    return index


def _extract_summary(name, res):
    """Pull a few headline numbers per phase for the index."""
    try:
        if name == "avalanche":
            return {"meanHD_T32": res["by_T"]["32"]["hd"]["mean"]}
        if name == "state_map":
            return {"jac_full_rank_frac": res["jacobian_rank"]["frac_full_rank"],
                    "all_zero_fixed": res["fixed_points"]["all_zero_is_fixed_T_round"]}
        if name == "squeeze":
            return {"max_abs_monobit_z": res["per_bit_one_freq"]["max_abs_monobit_z"]}
        if name == "distinguishers":
            return {"any_distinguisher": res["any_distinguisher"]}
        if name == "symmetry":
            return {"swap_full_collisions": res["block_swap_repeat"]["swap_full_collisions"]}
        if name == "collision":
            return {"T32_ratios": {k: v.get("ratio_observed_over_expected")
                                   for k, v in res["by_T"]["32"].items()
                                   if isinstance(v, dict)}}
        if name == "preimage":
            return {"local_vs_random": res["local_search_n32_T8"]}
        if name == "algebraic":
            return {"affine_relation": res["affine_relation"]["affine_relation_exists"]}
        if name == "parameter":
            return {"n_regimes": len(res["rows"])}
        if name == "baselines":
            return {"candidate": res["rows"]["pde_T32_candidate"]}
        if name == "eigenmode":
            return {
                "constructive_zero_preimages":
                    res["part2_tile_enumeration"]["constructive_zero_preimage_count_one_round"],
                "message_lift_found": res["part4_reachability"]["search"]["found"],
            }
    except Exception as e:  # pragma: no cover
        return {"summary_error": str(e)}
    return {}


if __name__ == "__main__":
    main()
