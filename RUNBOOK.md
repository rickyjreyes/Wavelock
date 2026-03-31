# WaveLock RUNBOOK

Exact commands to exercise the core WaveLock workflows.
All commands assume you are in the repository root directory.

## Prerequisites

```bash
pip install numpy matplotlib pytest
# Optional GPU acceleration:
# pip install cupy-cuda12x
```

## 1. Run the demo

```bash
python hello_wavelock.py
```

This generates a keypair, signs a message, mines a block, verifies the chain,
and runs the runaway-drift test. Output goes to `./wavelock/ledger/`.

## 2. Generate a keypair

```bash
python -m wavelock.chain.cli keygen --n 4 --seed 42
```

Writes `keypair.json` to current directory.

## 3. Add a user and sign a message

```bash
python -m wavelock.chain.cli add ricky --n 4 --seed 42
python -m wavelock.chain.cli sign ricky --message "hello wavelock" --output signed.json
```

## 4. Mine a block

```bash
python -m wavelock.chain.cli mine --signed_path signed.json
```

## 5. Verify the chain

```bash
python -m wavelock.chain.cli verify
```

## 6. Audit the ledger

```bash
python -m wavelock.chain.cli audit
```

## 7. Run the test suite

```bash
python -m pytest tests/ -v
```

## 8. Run the attack battery (numpy-only, no GPU required)

```bash
python wavelock_full_battery.py
```

Produces `data/wavelock_data/wavelock_full_attack_battery.png` and
`data/wavelock_data/wavelock_attack_results.txt`.

## 9. Verify soliton data reproducibility

```bash
python scripts/verify_soliton_seed12.py
```

Regenerates the PDE output for seed=12 and confirms it matches the stored
data files in `data/wavelock_data/`.

## 10. Run the NumPy reference self-test

```bash
python wavelock/chain/Wavelock_numpy.py
```

## 11. Reset the ledger

```bash
python -m wavelock.chain.cli reset
```

## Notes

- CuPy (GPU) is optional. All core operations fall back to NumPy.
- The attack battery uses numpy-only implementations. No PyTorch required.
- The strongest defensible claim for WaveLock is:
  "deterministic nonlinear PDE map with empirical non-invertibility under tested attacks"
- Failure of tested attacks does NOT prove cryptographic security.
