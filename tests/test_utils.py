import pytest

from wavelock.chain.WaveLock import CurvatureKeyPair
from wavelock.chain.CurvaChain import CurvaChain
from wavelock.chain.chain_utils import (
    save_chain,
    load_chain,
    visualize_psi,
    tamper_and_test,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def keypair():
    """
    Test-only CurvatureKeyPair with explicit ψ★ access enabled.
    """
    return CurvatureKeyPair(n=4, seed=123, test_mode=True)


@pytest.fixture
def chain():
    """
    Fresh CurvaChain instance for tests.
    """
    return CurvaChain(difficulty=3)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_visualize_psi(keypair):
    """
    Ensure ψ★ visualization runs without error.
    """
    visualize_psi(keypair.psi_star)


def test_tamper_detection(keypair):
    """
    Ensure tampering is detected correctly.
    """
    tamper_and_test(keypair)


def test_chain_save_and_load(tmp_path, chain):
    """
    Ensure chain serialization round-trips correctly.
    """
    chain.add_block(["Signed curvature message"])

    path = tmp_path / "curva_chain.json"
    save_chain(chain, str(path))

    loaded_chain = load_chain(str(path))
    assert len(loaded_chain.chain) == len(chain.chain)
