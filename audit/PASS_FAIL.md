# WaveLock Audit — Pass / Fail / Constraint Matrix

A compact summary of the audit in [`REPORT.md`](./REPORT.md), read through the
M2M interpretation in [`README.md`](./README.md). This is a scoping aid, not a
verdict that "WaveLock failed."

---

## Works / passes

| Item | Status | Note |
|------|--------|------|
| 1M-seed exact collision search | **pass** | 0 exact `ψ*` collisions, 0 commitment collisions (N-1, N-2). |
| No attractor collapse | **pass** | 1e6/1e6 distinct commitments, all fields finite (N-2). |
| No Newton inversion shortcut | **pass** | Chaotic + ill-conditioned Jacobian; Newton/LM does not converge (N-3). |
| No surrogate inversion shortcut | **pass** | `ψ*→ψ0` surrogate R² ≈ 0.03, no better than predict-0.5 (N-4). |
| SHA-256 commitment randomness | **pass** | But attributed to **SHA-256**, not the PDE. |

## Fails / constraints

| Item | Status | Note |
|------|--------|------|
| Direct payload encryption claim | **fail / out of scope** | WaveLock is not a cipher; secrecy belongs to KEM/AEAD. |
| Standalone SHA replacement claim | **not established** | Pre-hash structure is distinguishable (H-2); one-wayness is SHA-256's. |
| Low-entropy seed mode | **fail** | Default `42` / 20-bit secret brute-forceable (C-2). |
| Arbitrary floating-point consensus | **fail** | Reassociation changes `ψ*`/`C` for the same seed (C-1). |
| Noncanonical serialization | **fail** | Signed zero, NaN payloads, dual schema/endian (H-1). |
| Raw `ψ*` indistinguishability | **fail** | Byte χ² ≈ 5.5×10⁶, skew 195, neighbor corr 0.57 (H-2). |

## M2M interpretation

| Layer | Status | Condition |
|-------|--------|-----------|
| Commitment layer | **conditional pass** | Holds once canonical serialization + high-entropy input + single consensus backend are enforced. |
| Replay / attestation layer | **conditional pass** | Not exercised by the original audit; depends on metadata binding (see BOUNTY_SCOPE Critical 4, 6). |
| Drift-detection layer | **conditional pass** | Not exercised by the original audit; depends on evasion resistance (BOUNTY_SCOPE Critical 7). |
| Production bounty readiness | **not ready** | Blocked until the hardening in [`HARDENING_RECOMMENDATIONS.md`](./HARDENING_RECOMMENDATIONS.md) is implemented. |

---

See [`BOUNTY_SCOPE.md`](./BOUNTY_SCOPE.md) for what a valid break looks like, and
[`HARDENING_RECOMMENDATIONS.md`](./HARDENING_RECOMMENDATIONS.md) for the changes
that move the conditional passes to unconditional.
