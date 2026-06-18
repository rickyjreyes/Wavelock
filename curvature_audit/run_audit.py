"""Run the full curvature-capacity audit suite and write artifacts/INDEX.json.

This is the MEDIUM/heavy runner; it is NOT the fast CI path. CI runs the small
regression tests in curvature_audit/test_*.py instead.

    python -m curvature_audit.run_audit
"""

from __future__ import annotations

import time

from . import _common as C
from . import (curvature_metrics, eigenmode_attacks, path_collision,
               resource_bounds, scaling, reduced_models)

MODULES = [
    ("curvature_metrics", curvature_metrics, "curvature_metrics_demo.json"),
    ("eigenmode_attacks", eigenmode_attacks, "eigenmode_attacks.json"),
    ("path_collision", path_collision, "path_commitment_attacks.json"),
    ("resource_bounds", resource_bounds, "resource_bound_analysis.json"),
    ("scaling", scaling, "curvature_scaling.json"),
    ("reduced_models", reduced_models, "reduced_models.json"),
]


def main():
    t0 = time.perf_counter()
    index = {"suite": "curvature_capacity_audit", "metadata": C.env_metadata(),
             "modules": {}}
    for name, mod, artifact in MODULES:
        print(f"== {name} ==", flush=True)
        ts = time.perf_counter()
        res = mod.main()
        index["modules"][name] = {
            "artifact": artifact,
            "runtime_s": round(time.perf_counter() - ts, 2),
            "phase": res.get("phase"),
        }
    index["total_runtime_s"] = round(time.perf_counter() - t0, 2)
    path = C.save_artifact("INDEX.json", index)
    print("wrote", path, f"total {index['total_runtime_s']}s", flush=True)
    return index


if __name__ == "__main__":
    main()
