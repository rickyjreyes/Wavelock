#!/usr/bin/env bash
# Reproduce every WaveLock audit finding. NumPy-only; no GPU required.
#
#   bash audit/run_all.sh            # default sweep sizes
#   A1_N=1000000 bash audit/run_all.sh
#
# Artifacts land in audit/artifacts/*.json. Each script prints a summary and
# writes machine-readable evidence consumed by audit/REPORT.md.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

A1_N="${A1_N:-1000000}"          # entropy/collision sweep size
A7_M="${A7_M:-20000}"            # distinguishability sample
A8_NTRAIN="${A8_NTRAIN:-40000}"  # surrogate training set

echo "### 0. Harness fidelity (must reproduce repo commitment byte-for-byte)"
python audit/_wl.py

echo "### a6 determinism (Critical candidate)";        python audit/a6_determinism.py
echo "### a2 jacobian / lyapunov";                     python audit/a2_jacobian.py
echo "### a5 serialization ambiguity";                 python audit/a5_serialization.py
echo "### a7 distinguishability (psi* vs C)";          python audit/a7_distinguishability.py "$A7_M"
echo "### a9 parameter regimes";                       python audit/a9_parameters.py
echo "### a8 surrogate inversion";                     python audit/a8_surrogate.py "$A8_NTRAIN"
echo "### a1 attractor/entropy sweep (N=$A1_N, slow)"; python audit/a1_attractor_entropy.py "$A1_N"
echo "### a3 second-preimage (reads a1/a5 artifacts)"; python audit/a3_second_preimage.py
echo "### a4 seed-space brute force (slow)";           python audit/a4_seedspace.py

echo "ALL AUDIT SCRIPTS COMPLETE — see audit/artifacts/ and audit/REPORT.md"
