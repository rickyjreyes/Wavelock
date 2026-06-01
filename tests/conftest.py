# tests/conftest.py

import os
import sys

import pytest

# Absolute path to the project root (the folder that contains `chain/` and `tests/`)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Make sure it's on sys.path so `import chain` works
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(autouse=True)
def _ots_state_dir(tmp_path, monkeypatch):
    """Isolate the WaveLock-OTS host-local key-state registry per test.

    The registry (Finding D mitigation) atomically claims one_time_key_ids on
    disk. Pointing it at a per-test tmp dir keeps tests hermetic and prevents
    cross-test/host-state pollution.
    """
    monkeypatch.setenv("WAVELOCK_OTS_STATE_DIR", str(tmp_path / "ots-state"))
