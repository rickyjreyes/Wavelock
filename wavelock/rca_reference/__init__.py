"""PROPOSED reference safety kernel for RCA agents (issues #42 and #43).

STATUS: ``PROPOSED_REFERENCE``.  At ``c3cb0d1`` this repository contained no RCA agent state,
safety memory, drift measurement, action-authorization guard or intervention mechanism for
agents (``docs/RCA_FORMALIZATION.md``, Phase 0).  This package is the minimal architecture
proposed to make pre-action intervention provable; its Lean model and proofs
(``lean/WaveLockProofs/RCA*.lean``) verify this reference, not preexisting production behaviour.
It is not wired into the chain, network or WaveLock-OTS code and changes none of their behaviour.
"""
from .drift import anchored_deviation, l1, path_drift, predict_breach, step_deviations
from .kernel import GuardedAgent, HmacOperatorVerifier, operator_credential
from .memory import MemoryCorrupt, Policy, View, load, successor

STATUS = "PROPOSED_REFERENCE"

__all__ = [
    "STATUS", "GuardedAgent", "HmacOperatorVerifier", "MemoryCorrupt", "Policy", "View",
    "anchored_deviation", "l1", "load", "operator_credential", "path_drift", "predict_breach",
    "step_deviations", "successor",
]
