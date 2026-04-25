# Patent Scope and Filing Decisions

This document records the **deliberate scoping decisions** made for the
WaveLock provisional and any continuation/CIP filings. It is the audit
trail that pairs each patent claim with the implementation that supports
it (or does not), so an examiner, licensee, or future-litigation expert
witness can see what we knew, when we knew it, and what we chose to file.

If you are tempted to commercialize a feature whose claim language is
broader than what the implementation reliably does, read this document
first.

---

## Group I — Curvature-Regulated Evolution and Commitment

**Status: supported and filed.**

The PDE operator F, dual-hash commitment, canonical serialization with
kernel binding, and SHAKE-256 ψ₀ derivation are all implemented in this
repo (see `docs/CANONICAL.md`). Empirical attack-suite tests exercise
chaotic divergence, Lyapunov bounds, and absence of an algebraic
inverse.

### Caveats logged

- **BLAKE3.** The previous Python implementation silently fell back to
  BLAKE2b when the `blake3` PyPI package was missing. This has been
  hardened to raise `RuntimeError` instead of producing a non-BLAKE3
  digest under the BLAKE3 label. See `wavelock/chain/hash_families.py`.
  Any C-layer BLAKE3 implementation in a separate repo must vendor the
  official BLAKE3 reference from <https://github.com/BLAKE3-team/BLAKE3>
  (CC0/Apache-2.0) before its claim language is filed.

- **SHAKE-256 ψ₀ derivation (Claim 9).** Implemented in
  `wavelock/chain/xof_init.py` and exposed via `use_xof_init=True` on
  `CurvatureKeyPairV3`. The legacy `np.random.seed(int)` path is retained
  for backward compatibility with the existing test corpus; new
  consensus-grade commitments should use the XOF path. Claim 9 may be
  filed naming SHAKE-256 specifically OR generically as "an extendable-
  output function" — both are now supportable. The generic phrasing is
  preferred for downstream agility.

---

## Group II — Commitment-and-Replay Ledger

**Status: supported with a clarification.**

Claim 15 specifies "a Merkle-root field computed over fields (i)–(v) and
over a hash of the prior record." This repo implements that exact
binding in `wavelock/chain/ledger_merkle.py` (`compute_record_merkle_root`).

Note the layering: a ledger record's Merkle root is **distinct** from the
chain-block Merkle root in `Block.calculate_merkle_root()`. The chain
block Merkles its ordered `messages`; a record's Merkle binds the
record's commitment + operator + kernel + invariants + timestamp +
prior-record-hash. A record can be stored as one message inside a chain
block, in which case the block's Merkle covers the record's Merkle as a
leaf. This layering is intentional and worth disclosing in the spec so
the examiner does not conflate the two.

---

## Group III — Drift Detection Apparatus

**Status: NOT implemented in this repo. See decision below.**

### The known gap

The published validation runner (in a separate repository) computed a
diagonal-Mahalanobis-style drift metric over `/proc/stat`, `/proc/loadavg`,
and `/proc/meminfo`. In containerized environments, five of seven
observable channels flatlined at zero, collapsing the metric to a
one-dimensional signal on `cpu_pct` only. This caused inter-attack
distinguishability to fail and missed ~90% of attack samples at a
4σ threshold.

Bare-metal validation with `perf_event_open` against a richer observable
set has not yet been completed.

### The decision

We will NOT file Group III claims that imply inter-attack
distinguishability until bare-metal validation has either confirmed or
refuted that property. Two acceptable framings:

**Option A — narrow on filing (preferred).** Limit Claim 21(d) to
baseline-vs-runtime deviation detection ("produce a halt signal when
the distance metric exceeds a calibrated threshold"). Inter-attack
distinguishability becomes an aspiration of certain embodiments
described in the spec, never a claim limitation.

**Option B — defer.** Hold Group III back as a continuation or CIP and
file it only after bare-metal validation. If validation succeeds, file
the broader claim language. If it fails, file the narrowed Option A
claims.

In either case: **do not commercialize Group III** (raise money against
it, sell licenses citing it, or use it in marketing) until bare-metal
validation results are documented. Selling a feature whose own kill
criteria we know to fail in containerized environments creates written
evidence of inequitable conduct.

### Bare-metal validation gating criteria

Before Group III is unblocked for filing or commercialization, we must:

1. Run the validation suite on bare metal with `perf_event_open` enabled.
2. Confirm that all seven (or replacement) observable channels carry
   non-zero variance during baseline.
3. Recompute the drift metric (preferably full-covariance Mahalanobis,
   not the diagonal approximation) and confirm:
   - Baseline-to-attack separation >= configured threshold (this is the
     narrow claim).
   - Pairwise inter-attack separation, if claimed.
4. Publish the results — favorable or not — in `docs/inevitability/` so
   the gating decision is visible to anyone evaluating the filing.

---

## Cross-cutting issues

### Observable-injection function Φ

If/when Group III is unblocked, the production code must contain a
documented Φ implementation that matches the spec's "fixed deterministic
spatial-distribution function partitioning the lattice into K
sub-regions." A README pointer to this implementation should land in
the same commit that re-enables Group III filing.

### SHA3-256 in a C kernel

Any C-layer dual-hash implementation must include a real SHA3-256
implementation. The Python layer here uses `hashlib.sha3_256` (CPython
stdlib, FIPS 202). Cross-language interop should be tested with a
golden-vector test before shipping.

---

## How to update this document

This file is part of the patent audit trail. Every time a scope decision
is made or a gap is closed:

1. Update the relevant section above with the new status.
2. Reference the commit that closed the gap.
3. Do NOT delete the historical text — strike through or add a "Resolved"
   note, so the audit trail remains intact.

The point is not to maintain a pristine document. The point is to be
able to show, under oath, exactly what we knew and exactly what we
filed.
