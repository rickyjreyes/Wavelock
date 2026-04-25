# Changelog

All notable, consensus-affecting changes to the WaveLock canonical
reference implementation are recorded here. Schema-bumping changes are
deliberate protocol upgrades; pre-upgrade and post-upgrade commitments
are not interchangeable.

## [Unreleased] — WLv3.1 SHAKE-256 ψ₀ derivation upgrade

### Consensus break

The canonical seed → ψ₀ derivation on the consensus path is now
SHAKE-256 (NIST FIPS 202 XOF) with the `WL-PSI-INIT-v1` domain
separation tag, replacing `numpy.random.seed(s); numpy.random.rand(...)`
(Mersenne Twister).

This matches the patent's Best Mode and Claim 9, which specify an
extendable-output function for ψ₀ derivation. The pre-upgrade Mersenne
Twister path produced backend- and library-version-bound bytes that
were not byte-stable across implementations and therefore not a valid
basis for cross-implementation consensus.

A new schema label `SCHEMA_V3_SHAKE = "WLv3.1"` distinguishes commitments
produced under the SHAKE-256 regime from legacy `WLv2` commitments.
Commitments are not interchangeable across the two regimes; the schema
prefix in the commitment string is the authoritative discriminator.

### Behavior changes

- `wavelock.chain.Wavelock_numpy.CurvatureKeyPairV3.__init__`:
  default `use_xof_init` is now `True` (was `False`). Callers that
  explicitly want the legacy Mersenne Twister path must pass
  `use_xof_init=False`. Commitments produced with `use_xof_init=True`
  carry the `WLv3.1` schema label; legacy ones retain `WLv2`.

- `wavelock.chain.WaveLock.CurvatureKeyPair.__init__`: gains a
  `use_xof_init: Optional[bool] = None` parameter. When `None`
  (default), it auto-enables whenever any of `use_v3..use_v7` is set,
  which is exactly the consensus-emitting subset of the GPU class. The
  consensus guard at the same call site continues to forbid the GPU
  class from emitting consensus commitments outside `test_mode`; the
  XOF wiring is so that test-mode consensus runs on GPU produce
  byte-identical ψ₀ to the NumPy reference path.

- `_serialize_commitment_v2(psi)` is now a thin wrapper over a new
  `_serialize_commitment(psi, schema)` that writes the schema label
  into the binary header. The legacy V2 wrapper is preserved so
  external callers that imported it directly continue to work.

### Required follow-up

- Golden vectors in `tests/general/test_golden_vectors.py` have been
  reset to `PLACEHOLDER_GENERATE_ME` so the parametrized regression
  tests skip instead of asserting against stale Mersenne-Twister
  hashes. Repopulate with:

      python tests/general/test_golden_vectors.py generate

  and paste the printed dict over the placeholder one. Every entry
  will then carry `"schema": "WLv3.1"` — that's the loud signal that
  the canonical commitment format has flipped.

- Any persisted ledger or registry state produced before this change
  remains parseable but will fail re-derivation from seed under the
  new default. Such state is historical-format-only and must be
  re-issued through the new canonical path before being treated as
  consensus-binding.

### Patent-enablement effect

- Claim 9 / Best Mode (SHAKE-256 ψ₀ derivation) is now exercised on
  the actual canonical commitment path, not as an opt-in side branch.

- Claim 8 Markush group (SHA-256, SHA3-256, BLAKE3 as disjoint hash
  families) is now exercised end-to-end for BLAKE3 via two new tests
  in `tests/test_blake3_strict.py`:
  `test_blake3_end_to_end_through_keypair_commitment` and
  `test_blake3_dual_signature_through_keypair`. Both construct a
  `CurvatureKeyPairV3` with `secondary_family=HashFamily.BLAKE3`,
  generate a commitment, and round-trip it through `verify_commitment`
  and `verify_strict`.
