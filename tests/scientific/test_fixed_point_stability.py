#!/usr/bin/env python3
"""
Fixed-point, perturbation, and grid-risk diagnostics for the current
WaveLock NumPy reference evolution.

This test does not assume that the 50-step terminal field is a fixed point.
It measures whether the field is:
  1. a numerically resolved nonzero fixed point,
  2. a periodic/evolving transient,
  3. a nearly uniform field undergoing pure damping,
  4. non-finite or singularity-sensitive.

Run directly:
    python tests/scientific/test_fixed_point_stability.py

Select a profile:
    WAVELOCK_FIXED_POINT_PROFILE=standard \
        python tests/scientific/test_fixed_point_stability.py

The pytest smoke test uses the fast profile.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from wavelock.chain import Wavelock_numpy as wl


PROFILE = os.environ.get("WAVELOCK_FIXED_POINT_PROFILE", "fast").strip().lower()

PROFILE_CONFIG = {
    "fast": {
        "n_values": [6],          # side = 8
        "seeds": [42],
        "max_steps": 10_000,
        "checkpoints": [0, 1, 50, 500, 5_000, 10_000],
        "jacobian_n": 6,
        "perturb_steps": 500,
    },
    "standard": {
        "n_values": [6, 8, 10],   # sides = 8, 16, 32
        "seeds": [123, 7, 42, 99, 2025],
        "max_steps": 5_000,
        "checkpoints": [0, 1, 50, 500, 5_000],
        "jacobian_n": 6,
        "perturb_steps": 500,
    },
    "deep": {
        "n_values": [6, 8, 10, 12],  # sides = 8, 16, 32, 64
        "seeds": [123, 7, 42, 99, 2025],
        "max_steps": 20_000,
        "checkpoints": [0, 1, 50, 500, 5_000, 10_000, 20_000],
        "jacobian_n": 8,
        "perturb_steps": 2_000,
    },
}

FIXED_POINT_TOL = 1.0e-10
UNIFORM_REL_STD_TOL = 1.0e-3
DAMPING_MATCH_RTOL = 5.0e-2
DENOMINATOR_WARNING = 1.0e-6
AMPLITUDE_WARNING = 1.0e3


@dataclass
class FieldMetrics:
    n: int
    side: int
    seed: int
    step: int
    residual: float
    norm_l2: float
    mean: float
    std: float
    relative_std: float
    min_value: float
    max_value: float
    max_abs: float
    denominator_min_abs: float
    finite: bool
    classification: str


def side_from_n(n: int) -> int:
    return 2 ** max(1, int(n) // 2)


def step_map(psi: np.ndarray) -> np.ndarray:
    """One exact NumPy-reference WaveLock evolution step."""
    psi = np.asarray(psi, dtype=np.float64)
    lap = wl.laplacian(psi)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        denominator = psi + wl.epsilon * np.exp(-wl.beta * psi ** 2)
        feedback = wl.alpha * lap / denominator
        entropy = wl.theta * (
            psi * wl.laplacian(np.log(psi ** 2 + wl.delta))
        )
        next_psi = (
            psi
            + wl._dt * (feedback - entropy)
            - wl._damping * psi
        )

    return np.asarray(next_psi, dtype=np.float64)


def fixed_point_residual(psi: np.ndarray) -> float:
    next_psi = step_map(psi)
    denominator = float(np.linalg.norm(psi)) + np.finfo(np.float64).tiny
    return float(np.linalg.norm(next_psi - psi) / denominator)


def denominator_min_abs(psi: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        denominator = psi + wl.epsilon * np.exp(-wl.beta * psi ** 2)
    return float(np.nanmin(np.abs(denominator)))


def classify_state(
    residual: float,
    relative_std: float,
    finite: bool,
) -> str:
    if not finite:
        return "nonfinite"
    if residual <= FIXED_POINT_TOL:
        return "numerical_fixed_point"

    damping = abs(float(wl._damping))
    damping_error = abs(residual - damping)
    damping_match = damping_error <= max(
        1.0e-12,
        DAMPING_MATCH_RTOL * max(damping, np.finfo(np.float64).eps),
    )

    if relative_std <= UNIFORM_REL_STD_TOL and damping_match:
        return "uniform_damped_transient"

    return "evolving_transient"


def field_metrics(
    psi: np.ndarray,
    *,
    n: int,
    seed: int,
    step: int,
) -> FieldMetrics:
    norm_l2 = float(np.linalg.norm(psi))
    mean = float(np.mean(psi))
    std = float(np.std(psi))
    relative_std = std / (abs(mean) + np.finfo(np.float64).tiny)
    residual = fixed_point_residual(psi)
    finite = bool(np.isfinite(psi).all() and np.isfinite(residual))

    return FieldMetrics(
        n=int(n),
        side=int(psi.shape[0]),
        seed=int(seed),
        step=int(step),
        residual=residual,
        norm_l2=norm_l2,
        mean=mean,
        std=std,
        relative_std=float(relative_std),
        min_value=float(np.min(psi)),
        max_value=float(np.max(psi)),
        max_abs=float(np.max(np.abs(psi))),
        denominator_min_abs=denominator_min_abs(psi),
        finite=finite,
        classification=classify_state(residual, relative_std, finite),
    )


def initial_field(n: int, seed: int) -> np.ndarray:
    side = side_from_n(n)
    return np.asarray(
        wl.derive_psi_zero(seed, (side, side)),
        dtype=np.float64,
    )


def evolve_with_checkpoints(
    n: int,
    seed: int,
    max_steps: int,
    checkpoints: Iterable[int],
) -> Tuple[np.ndarray, List[FieldMetrics]]:
    psi = initial_field(n, seed)
    wanted = set(int(k) for k in checkpoints)
    rows: List[FieldMetrics] = []

    for k in range(int(max_steps) + 1):
        if k in wanted:
            rows.append(field_metrics(psi, n=n, seed=seed, step=k))

        if k == max_steps:
            break

        psi = step_map(psi)
        if not np.isfinite(psi).all():
            rows.append(field_metrics(psi, n=n, seed=seed, step=k + 1))
            break

    return psi, rows


def jacobian_action(psi: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Exact analytic action of DT(psi) on a perturbation vector.

    T(psi) = (1-d) psi + dt * G(psi)
    """
    psi = np.asarray(psi, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        exp_term = np.exp(-wl.beta * psi ** 2)
        q = psi + wl.epsilon * exp_term
        q_prime = 1.0 - 2.0 * wl.beta * wl.epsilon * psi * exp_term

        ell = np.log(psi ** 2 + wl.delta)
        ell_prime = 2.0 * psi / (psi ** 2 + wl.delta)

        lap_psi = wl.laplacian(psi)
        lap_ell = wl.laplacian(ell)

        derivative_feedback = wl.alpha * (
            wl.laplacian(vector) / q
            - lap_psi * (q_prime * vector) / (q ** 2)
        )

        derivative_entropy = wl.theta * (
            vector * lap_ell
            + psi * wl.laplacian(ell_prime * vector)
        )

        result = (
            (1.0 - wl._damping) * vector
            + wl._dt * (derivative_feedback - derivative_entropy)
        )

    return np.asarray(result, dtype=np.float64)


def dense_jacobian_spectral_radius(psi: np.ndarray) -> Dict[str, object]:
    """
    Construct the exact dense Jacobian by analytic matrix-vector products.

    Restricted to small grids because the matrix has (side^2)^2 entries.
    """
    shape = psi.shape
    dimension = int(psi.size)
    identity = np.eye(dimension, dtype=np.float64)
    jacobian = np.empty((dimension, dimension), dtype=np.float64)

    for column in range(dimension):
        basis = identity[:, column].reshape(shape)
        jacobian[:, column] = jacobian_action(psi, basis).ravel()

    eigenvalues = np.linalg.eigvals(jacobian)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    leading = eigenvalues[order[: min(8, eigenvalues.size)]]

    return {
        "dimension": dimension,
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "leading_eigenvalues": [
            {
                "real": float(np.real(value)),
                "imag": float(np.imag(value)),
                "magnitude": float(abs(value)),
            }
            for value in leading
        ],
    }


def perturbation_recovery(
    psi: np.ndarray,
    *,
    amplitude: float = 1.0e-6,
    steps: int = 500,
    seed: int = 12345,
) -> Dict[str, float]:
    """
    Compare a perturbed trajectory against the unperturbed continuation.

    This measures local trajectory contraction. It is not called fixed-point
    recovery unless the reference state itself satisfies the residual test.
    """
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(psi.shape)
    direction /= float(np.linalg.norm(direction))

    scale = max(float(np.linalg.norm(psi)), 1.0)
    perturbation = amplitude * scale * direction

    reference = psi.copy()
    perturbed = psi + perturbation
    initial_distance = float(np.linalg.norm(perturbed - reference))
    max_gain = 1.0
    final_gain = 1.0

    for _ in range(int(steps)):
        reference = step_map(reference)
        perturbed = step_map(perturbed)

        distance = float(np.linalg.norm(perturbed - reference))
        final_gain = distance / (
            initial_distance + np.finfo(np.float64).tiny
        )
        max_gain = max(max_gain, final_gain)

        if not np.isfinite(final_gain):
            break

    return {
        "amplitude": float(amplitude),
        "steps": int(steps),
        "initial_distance": initial_distance,
        "final_gain": float(final_gain),
        "max_gain": float(max_gain),
    }


def run_profile(profile: str = PROFILE) -> Dict[str, object]:
    if profile not in PROFILE_CONFIG:
        raise ValueError(
            f"unknown profile {profile!r}; choose "
            f"{sorted(PROFILE_CONFIG)}"
        )

    config = PROFILE_CONFIG[profile]
    metric_rows: List[FieldMetrics] = []
    terminal_states: Dict[Tuple[int, int], np.ndarray] = {}

    for n in config["n_values"]:
        for seed in config["seeds"]:
            terminal, rows = evolve_with_checkpoints(
                n=n,
                seed=seed,
                max_steps=config["max_steps"],
                checkpoints=config["checkpoints"],
            )
            metric_rows.extend(rows)
            terminal_states[(n, seed)] = terminal

    jacobian_n = int(config["jacobian_n"])
    jacobian_seed = int(config["seeds"][0])
    jacobian_state = terminal_states[(jacobian_n, jacobian_seed)]

    spectrum = dense_jacobian_spectral_radius(jacobian_state)
    recovery = perturbation_recovery(
        jacobian_state,
        steps=int(config["perturb_steps"]),
    )

    initial_risk_rows = []
    for n in config["n_values"]:
        for seed in config["seeds"]:
            psi0 = initial_field(n, seed)
            psi1 = step_map(psi0)
            initial_risk_rows.append(
                {
                    "n": int(n),
                    "side": side_from_n(n),
                    "seed": int(seed),
                    "initial_denominator_min_abs": denominator_min_abs(psi0),
                    "initial_max_abs": float(np.max(np.abs(psi0))),
                    "one_step_max_abs": float(np.max(np.abs(psi1))),
                    "one_step_norm_gain": float(
                        np.linalg.norm(psi1)
                        / (
                            np.linalg.norm(psi0)
                            + np.finfo(np.float64).tiny
                        )
                    ),
                    "finite_after_one_step": bool(np.isfinite(psi1).all()),
                }
            )

    warnings = []
    for row in metric_rows:
        if row.denominator_min_abs < DENOMINATOR_WARNING:
            warnings.append(
                f"near-zero denominator: n={row.n}, seed={row.seed}, "
                f"step={row.step}, min|q|={row.denominator_min_abs:.3e}"
            )
        if row.max_abs > AMPLITUDE_WARNING:
            warnings.append(
                f"large amplitude: n={row.n}, seed={row.seed}, "
                f"step={row.step}, max|psi|={row.max_abs:.3e}"
            )

    for row in initial_risk_rows:
        if row["initial_denominator_min_abs"] < DENOMINATOR_WARNING:
            warnings.append(
                "initial denominator risk: "
                f"n={row['n']}, seed={row['seed']}, "
                f"min|q|={row['initial_denominator_min_abs']:.3e}"
            )
        if row["one_step_max_abs"] > AMPLITUDE_WARNING:
            warnings.append(
                "one-step amplification: "
                f"n={row['n']}, seed={row['seed']}, "
                f"max|psi_1|={row['one_step_max_abs']:.3e}"
            )

    return {
        "profile": profile,
        "parameters": {
            "alpha": float(wl.alpha),
            "beta": float(wl.beta),
            "theta": float(wl.theta),
            "epsilon": float(wl.epsilon),
            "delta": float(wl.delta),
            "dt": float(wl._dt),
            "damping_per_step": float(wl._damping),
        },
        "metrics": [asdict(row) for row in metric_rows],
        "initial_grid_risk": initial_risk_rows,
        "jacobian": {
            "n": jacobian_n,
            "side": side_from_n(jacobian_n),
            "seed": jacobian_seed,
            **spectrum,
        },
        "perturbation_recovery": recovery,
        "warnings": sorted(set(warnings)),
    }


def _validate_report(report: Dict[str, object]) -> None:
    metrics = report["metrics"]
    assert metrics, "diagnostic produced no metric rows"

    for row in metrics:
        assert row["finite"], (
            "non-finite field encountered: "
            f"n={row['n']} seed={row['seed']} step={row['step']}"
        )
        assert np.isfinite(row["residual"])
        assert row["residual"] >= 0.0
        assert row["denominator_min_abs"] >= 0.0

    radius = report["jacobian"]["spectral_radius"]
    assert np.isfinite(radius)
    assert radius >= 0.0

    recovery = report["perturbation_recovery"]
    assert np.isfinite(recovery["final_gain"])
    assert np.isfinite(recovery["max_gain"])


def test_fixed_point_stability_smoke():
    report = run_profile("fast")
    _validate_report(report)

    terminal = [
        row
        for row in report["metrics"]
        if row["step"] == PROFILE_CONFIG["fast"]["max_steps"]
    ]
    assert len(terminal) == 1

    # This assertion prevents a slowly changing state from being silently
    # described as a fixed point. It does not require the current kernel to
    # produce a nonzero fixed point.
    row = terminal[0]
    if row["classification"] == "numerical_fixed_point":
        assert row["residual"] <= FIXED_POINT_TOL
    else:
        assert row["residual"] > FIXED_POINT_TOL


def main() -> int:
    report = run_profile(PROFILE)
    _validate_report(report)

    print("RISK_METRICS_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("RISK_METRICS_END")

    terminal_step = PROFILE_CONFIG[PROFILE]["max_steps"]
    terminal_rows = [
        row
        for row in report["metrics"]
        if row["step"] == terminal_step
    ]

    print("\nTERMINAL CLASSIFICATION")
    for row in terminal_rows:
        print(
            f"n={row['n']:2d} side={row['side']:3d} seed={row['seed']:7d} "
            f"R={row['residual']:.6e} "
            f"rel_std={row['relative_std']:.6e} "
            f"class={row['classification']}"
        )

    print(
        "\nJacobian spectral radius:",
        f"{report['jacobian']['spectral_radius']:.12g}",
    )
    print(
        "Perturbation final gain:",
        f"{report['perturbation_recovery']['final_gain']:.12g}",
    )

    if report["warnings"]:
        print("\nWARNINGS")
        for warning in report["warnings"]:
            print("-", warning)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
