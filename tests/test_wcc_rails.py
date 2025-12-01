import cupy as cp
import pytest

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    enforce_dimensional_lock,
    classify_wcc_run,
    wcc_avg_curvature_budget,
    check_quantum_classical_bound,
    _steps,
)


def test_dimensional_lock_accepts_valid_power_of_two_square():
    psi = cp.zeros((4, 4), dtype=cp.float64)
    # should NOT raise
    enforce_dimensional_lock(psi)


def test_dimensional_lock_rejects_non_square():
    psi = cp.zeros((4, 3), dtype=cp.float64)
    with pytest.raises(ValueError):
        enforce_dimensional_lock(psi)


def test_dimensional_lock_rejects_wrong_dimensionality():
    psi_1d = cp.zeros((4,), dtype=cp.float64)
    with pytest.raises(ValueError):
        enforce_dimensional_lock(psi_1d)


def test_wcc_classification_for_evolved_state():
    kp = CurvatureKeyPair(n=4, seed=123)
    psi = kp.psi_star

    wcc_class = classify_wcc_run(_steps, psi)
    avg_c = wcc_avg_curvature_budget(psi)

    assert wcc_class in {"PWCC", "NPWCC", "OUT_OF_WCC"}
    assert avg_c > 0.0  # curvature budget should be positive


def test_quantum_classical_bound_returns_boolean():
    kp = CurvatureKeyPair(n=4, seed=456)
    psi = kp.psi_star

    qc_flag = check_quantum_classical_bound(psi)
    # numpy.bool_ or Python bool both satisfy this
    assert qc_flag is True or qc_flag is False
