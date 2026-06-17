# pde_audit — WaveLock-PDE-256-v0 test & (forthcoming) cryptanalysis suite

Tests for the experimental hash-free PDE-native digest in `wavelock/pde_hash/`.
This suite runs **on the raw PDE output** — it never routes the digest through a
conventional hash before analysis.

Run from the repository root:

```bash
python -m pytest pde_audit/ -c pde_audit/pytest.ini -m "not slow"   # fast (~20s)
python -m pytest pde_audit/ -c pde_audit/pytest.ini                  # incl. 10k parity (~10min)
```

## Current contents (Design A implementation + parity phase)

- `test_forbidden_imports.py` — AST guard: fails if any module under
  `wavelock/pde_hash/` imports a forbidden module (`hashlib`, `blake3`,
  `cryptography`, `Crypto`, `hmac`, …) or references a forbidden digest/cipher
  symbol (`sha*`, `shake`, `blake*`, `md5`, `hmac`, `hkdf`, `aes`, `chacha`, …).
  Uses `ast`, so prose naming the primitives in docstrings does not trip it.
- `test_vectors.py` — pins the deterministic vectors in `vectors.json`
  (IV cells, final digests, round-granular intermediate snapshots) and checks
  both implementations reproduce them.
- `test_parity.py` — byte-for-byte agreement between the pure-Python reference
  and the independent NumPy implementation over the empty message, all 256
  one-byte messages, padding boundaries, structured inputs, repeated/reordered
  blocks, ≥10,000 deterministic random messages (`slow`), and round-granular
  intermediate snapshots.
- `vectors.json` — frozen test vectors.

## Not yet present

The adversarial cryptanalysis (differential/avalanche, collision, preimage,
structural distinguishers, dynamical analysis) and weak-baseline comparisons are
**Phase 8/9**, intentionally deferred pending review of this implementation
phase. No security verdict has been formed.
