"""Pytest setup for the pde_audit suite.

Ensures the repository root is importable so ``import wavelock.pde_hash`` works
when running ``pytest pde_audit/`` from the project root.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
