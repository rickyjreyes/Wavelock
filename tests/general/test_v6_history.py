"""
WaveLock v6 History Integrity Tests
===================================

Tests to ensure SCHEMA_V6 properly depends on the full evolution history,
not just the terminal state ψ★.

ATTACK SCENARIO:
An attacker who observes intermediate evolution states might attempt to:
1. Truncate the history at step T-k
2. Evolve from there with a different ψ₀
3. Produce a different commitment with valid-looking history

These tests verify that such attacks produce detectably different commitments.
"""

import pytest
import numpy as np


# =============================================================================
# Test: v6 uses history, not just terminal state
# =============================================================================

def test_v6_commitment_depends_on_history():
    """
    v6 commitment must depend on evolution history, not just ψ★.
    
    Identical ψ★ with different histories must produce different commitments.
    """
    pytest.importorskip("cupy")
    
    from wavelock.chain.WaveLock import (
        CurvatureKeyPair,
        _serialize_commitment_v6,
        SCHEMA_V6,
    )
    
    # Create a v6 keypair
    kp = CurvatureKeyPair(n=4, seed=42, use_v6=True, test_mode=True)
    
    # Verify history was captured
    assert len(kp.psi_history) > 1, (
        f"v6 should capture evolution history, got {len(kp.psi_history)} frames"
    )
    
    # Full history serialization
    full_serial = _serialize_commitment_v6(kp.psi_history)
    
    # Terminal-only serialization (simulating an attacker who only has ψ★)
    terminal_only = _serialize_commitment_v6([kp.psi_history[-1]])
    
    assert full_serial != terminal_only, (
        "v6 SECURITY FAILURE: History serialization equals terminal-only serialization!\n"
        "This means v6 does not actually depend on evolution history."
    )


def test_v6_history_truncation_attack():
    """
    Truncated history must produce different commitment.
    
    Simulates an attacker who has partial history and tries to
    reconstruct a valid commitment.
    """
    pytest.importorskip("cupy")
    
    from wavelock.chain.WaveLock import (
        CurvatureKeyPair,
        _serialize_commitment_v6,
    )
    
    kp = CurvatureKeyPair(n=4, seed=42, use_v6=True, test_mode=True)
    
    full_history = kp.psi_history
    assert len(full_history) >= 10, "Need sufficient history for truncation test"
    
    # Various truncation attacks
    truncations = [
        ("first_half", full_history[:len(full_history)//2]),
        ("second_half", full_history[len(full_history)//2:]),
        ("every_other", full_history[::2]),
        ("every_third", full_history[::3]),
        ("last_10", full_history[-10:]),
        ("first_10", full_history[:10]),
    ]
    
    full_serial = _serialize_commitment_v6(full_history)
    
    for name, truncated in truncations:
        if len(truncated) == 0:
            continue
            
        truncated_serial = _serialize_commitment_v6(truncated)
        
        assert truncated_serial != full_serial, (
            f"v6 TRUNCATION ATTACK SUCCEEDED: '{name}' produced same serialization!\n"
            f"Full history: {len(full_history)} frames\n"
            f"Truncated: {len(truncated)} frames"
        )


def test_v6_history_reorder_attack():
    """
    Reordered history must produce different commitment.
    
    Simulates an attacker who has the full history but tries to
    reorder frames.
    """
    pytest.importorskip("cupy")
    
    from wavelock.chain.WaveLock import (
        CurvatureKeyPair,
        _serialize_commitment_v6,
    )
    
    kp = CurvatureKeyPair(n=4, seed=42, use_v6=True, test_mode=True)
    
    full_history = kp.psi_history
    full_serial = _serialize_commitment_v6(full_history)
    
    # Reversed history
    reversed_history = list(reversed(full_history))
    reversed_serial = _serialize_commitment_v6(reversed_history)
    
    assert reversed_serial != full_serial, (
        "v6 REORDER ATTACK SUCCEEDED: Reversed history produced same serialization!"
    )
    
    # Shuffled history (deterministic shuffle)
    import random
    rng = random.Random(12345)
    shuffled_history = full_history.copy()
    rng.shuffle(shuffled_history)
    shuffled_serial = _serialize_commitment_v6(shuffled_history)
    
    assert shuffled_serial != full_serial, (
        "v6 SHUFFLE ATTACK SUCCEEDED: Shuffled history produced same serialization!"
    )


def test_v6_history_injection_attack():
    """
    Injecting extra frames must produce different commitment.
    
    Simulates an attacker who tries to inject crafted intermediate states.
    """
    pytest.importorskip("cupy")
    import cupy as cp
    
    from wavelock.chain.WaveLock import (
        CurvatureKeyPair,
        _serialize_commitment_v6,
    )
    
    kp = CurvatureKeyPair(n=4, seed=42, use_v6=True, test_mode=True)
    
    full_history = kp.psi_history
    full_serial = _serialize_commitment_v6(full_history)
    
    # Inject a zero frame
    injected = full_history.copy()
    zero_frame = cp.zeros_like(full_history[0])
    injected.insert(len(injected)//2, zero_frame)
    injected_serial = _serialize_commitment_v6(injected)
    
    assert injected_serial != full_serial, (
        "v6 INJECTION ATTACK SUCCEEDED: Injected frame produced same serialization!"
    )
    
    # Inject a duplicate frame
    duplicated = full_history.copy()
    duplicated.insert(5, full_history[5].copy())
    duplicated_serial = _serialize_commitment_v6(duplicated)
    
    assert duplicated_serial != full_serial, (
        "v6 DUPLICATE ATTACK SUCCEEDED: Duplicated frame produced same serialization!"
    )


def test_v6_different_seeds_different_history():
    """
    Different seeds must produce different history serializations.
    """
    pytest.importorskip("cupy")
    
    from wavelock.chain.WaveLock import (
        CurvatureKeyPair,
        _serialize_commitment_v6,
    )
    
    kp1 = CurvatureKeyPair(n=4, seed=42, use_v6=True, test_mode=True)
    kp2 = CurvatureKeyPair(n=4, seed=99, use_v6=True, test_mode=True)
    
    serial1 = _serialize_commitment_v6(kp1.psi_history)
    serial2 = _serialize_commitment_v6(kp2.psi_history)
    
    assert serial1 != serial2, (
        "v6 COLLISION: Different seeds produced same history serialization!"
    )


# =============================================================================
# Test: v6 schema is correctly identified
# =============================================================================

def test_v6_schema_marker():
    """
    v6 keypair must have correct schema marker.
    """
    pytest.importorskip("cupy")
    
    from wavelock.chain.WaveLock import CurvatureKeyPair, SCHEMA_V6
    
    kp = CurvatureKeyPair(n=4, seed=42, use_v6=True, test_mode=True)
    
    assert kp.schema == SCHEMA_V6, (
        f"v6 keypair has wrong schema: {kp.schema} (expected {SCHEMA_V6})"
    )
    
    assert kp.commitment.startswith("WLv6:"), (
        f"v6 commitment has wrong prefix: {kp.commitment[:20]}..."
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
