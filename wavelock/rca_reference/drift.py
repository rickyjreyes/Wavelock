"""Integer drift metrics of the PROPOSED RCA reference safety kernel (issue #43).

STATUS: proposed research model, introduced with ``docs/RCA_FORMALIZATION.md``.  No drift
metric existed in this repository before it (``audit/run_bounty_contract.py``, Critical 7:
"No behavior-wide drift oracle, observation contract, or detection verifier exists").  Being
mathematically defined does not make a metric scientifically validated: what these functions
detect is fixed by what the observation channel reports (see ``docs/RCA_FORMALIZATION.md``,
"Drift").

Observations are vectors of integers in fixed, caller-defined units (no floats, so the Lean model
``lean/WaveLockProofs/RCADriftMetrics.lean`` computes exactly the same numbers).

* ``anchored_deviation(o, baseline)`` -- L1 distance of an observation from the committed
  baseline (the kernel's latch criterion);
* ``step_deviations(trace)`` -- L1 distance between consecutive observations;
* ``path_drift(trace)`` -- accumulated step deviation (path length);
* ``predict_breach(dev, rate, horizon, threshold)`` -- conservative breach predictor, sound only
  when every future step deviation is at most ``rate``.
"""
from __future__ import annotations

from typing import Iterable, Sequence


def check_vector(values) -> tuple:
    """Return ``values`` as a tuple of Python ints; reject bools, floats and other types."""
    if not isinstance(values, (list, tuple)):
        raise ValueError("an observation vector must be a list of integers")
    out = tuple(values)
    if not all(type(x) is int for x in out):
        raise ValueError("observation coordinates must be integers (no bool or float)")
    return out


def l1(u: Sequence[int], v: Sequence[int]) -> int:
    """Sum of absolute coordinate differences of two vectors of equal length."""
    u, v = check_vector(u), check_vector(v)
    if len(u) != len(v):
        raise ValueError("vectors of different dimension")
    return sum(abs(a - b) for a, b in zip(u, v))


def anchored_deviation(values: Sequence[int], baseline: Sequence[int]) -> int:
    """Deviation of an observation from the committed baseline."""
    return l1(values, baseline)


def step_deviations(trace: Iterable[Sequence[int]]) -> list:
    """L1 distance between each observation and the previous one."""
    trace = [check_vector(t) for t in trace]
    return [l1(trace[i], trace[i - 1]) for i in range(1, len(trace))]


def path_drift(trace: Iterable[Sequence[int]]) -> int:
    """Accumulated step deviation (non-negative, non-decreasing along a trace)."""
    return sum(step_deviations(trace))


def predict_breach(deviation: int, rate: int, horizon: int, threshold: int) -> bool:
    """Whether the anchored deviation could reach ``threshold`` within ``horizon`` steps.

    Sound (no missed breach) under the stated assumption that every future step deviation is at
    most ``rate`` (``RCADriftMetrics.predict_sound``); conservative, so it raises alarms for
    breaches that need not happen (``predict_not_complete``); unsound without the rate bound
    (``predict_unsound_without_rate_bound``).
    """
    for name, x in (("deviation", deviation), ("rate", rate), ("horizon", horizon),
                    ("threshold", threshold)):
        if type(x) is not int or x < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    return deviation + rate * horizon >= threshold
