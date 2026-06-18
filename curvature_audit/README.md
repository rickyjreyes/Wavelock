# curvature_audit

Adversarial audit suite for the **WaveLock Curvature-Capacity Core**
(`wavelock/curvature_capacity/`, CC-Core-v0). Operates on raw coupled state /
raw 256-bit digest only; no conventional cryptographic primitive is routed over
candidate output (enforced by `test_forbidden_imports.py`).

**No security claim.** See `docs/WAVELOCK_CURVATURE_CAPACITY_RESULTS.md` for the
verdict (**Experimental**) and the precise allowed/forbidden statements.

## Modules

| module | task part | what it does |
|---|---|---|
| `curvature_metrics.py` | III | curvature/signature functionals + lifting-sensitivity |
| `eigenmode_attacks.py` | VII, XI | Design A collapse + trajectory separation, symmetry, fixed points |
| `path_collision.py` | X, XIII | pebble property, transcript/digest collisions, avalanche, bias, differential, multicollision, length-extension |
| `resource_bounds.py` | IV | R_comp / R_phys / R_machine numbers |
| `scaling.py` | VIII | curvature growth vs rounds / length / Hamming |
| `reduced_models.py` | XIII | toy coupled-core injectivity / lifting |
| `run_audit.py` | XV | runs all, writes `artifacts/INDEX.json` |

## Run

```bash
python -m curvature_audit.run_audit                 # full suite (~6 min), all artifacts
python -m curvature_audit.eigenmode_attacks         # central separation result
python -m pytest curvature_audit/ -c curvature_audit/pytest.ini -m "not slow" -q   # fast CI
```

Artifacts land in `curvature_audit/artifacts/`; each records branch, commit,
equations, seed, environment, runtime, budget, raw result, interpretation,
limitations. Heavy searches are NOT in fast CI.
