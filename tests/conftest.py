# tests/conftest.py

import os
import sys

# Absolute path to the project root (the folder that contains `chain/` and `tests/`)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Make sure it's on sys.path so `import chain` works
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
