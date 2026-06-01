# WaveLock Theory-Break Audit (legacy SIGv2)

> **Status:** legacy WaveLock SIGv2 is **broken by design**. This document and
> the two companion scripts (`forge_from_snapshot.py`, `seed_bruteforce.py`)
> are preserved as *regression evidence*. They demonstrate that the old scheme
> is not an asymmetric signature, and they are wired into the test suite so the
> break can never silently "come back" as if it were secure.
>
> The fix is **WaveLock-OTS** (`wavelock/crypto/wavelock_ots.py`). See
> `docs/WAVELOCK_OTS_DESIGN.md`.

## 1. The core defect

Legacy WaveLock signs a message as:

```
signature = H("SIGv2" || message || header || ψ★)
```

and `verify(message, signature)` recomputes **the same hash** and compares.

That means:

- To **verify**, you must possess ψ★.
- To **sign / forge**, you must possess ψ★.

The verify-capability set and the forge-capability set are **identical**. This
is a symmetric MAC whose key (ψ★) is *published as verifier material* in strict
mode. It is not a public-key signature. There is no asymmetry to attack — it
was never present.

## 2. Attack A — forge from a ψ★ snapshot

`forge_from_snapshot.py`:

1. A victim creates a legacy keypair; strict verification distributes ψ★ as
   "verifier material".
2. The attacker takes only that ψ★ snapshot.
3. The attacker instantiates any keypair, overwrites `psi_star` with the
   snapshot, and calls `.sign(attacker_message)`.
4. The victim's `verify(attacker_message, forged_sig)` returns **True**.

The attacker never needed the seed or any genuine signature. **Anyone who can
verify can forge.**

## 3. Attack B — brute-force tiny seeds

`seed_bruteforce.py`:

- Legacy examples use `seed=12`, `seed=42`, `seed=123`.
- The commitment is a deterministic function of the seed.
- Enumerating small integers recovers the seed, regenerates ψ★, and unlocks
  Attack A.

128–256 bits of entropy are required to resist enumeration. Legacy used ~7 bits.

## 4. Secondary defects

- `keygen` writes `seed`, `psi_0`, and `psi_star` to `keypair.json` **in
  cleartext**.
- Server "non-strict" mode (`require_full_verify=False`) accepts a block on
  **trust-list membership alone**, performing no signature check.
- Even in "strict" mode, if ψ★ was not published the server **allowed** the
  block by default (`WAVELOCK_REJECT_IF_UNPUBLISHED=0`).

All three are fail-open behaviors.

## 5. What WaveLock-OTS changes

| Property                              | Legacy SIGv2 | WaveLock-OTS |
|---------------------------------------|--------------|--------------|
| Verifier needs ψ★                     | **Yes**      | No           |
| Verifier can forge                    | **Yes**      | No           |
| Public key = commitments only         | No           | Yes          |
| Seed entropy                          | ~7 bits      | ≥128, default 256 |
| Cleartext ψ★/seed in public artifact  | **Yes**      | No           |
| Non-strict verification bypass        | **Yes**      | Removed      |
| One-time-use enforced                 | No           | Yes          |

## 6. Regression guarantees

The scripts here are imported by:

- `tests/test_legacy_sigv2_broken.py` — asserts the legacy forge **succeeds**
  and tiny seeds are **recoverable** (the break is real and documented).
- `tests/test_ots_security.py` — asserts the same attacks **fail** against
  WaveLock-OTS.

If a future change accidentally makes the legacy break "look" fixed without
actually moving to OTS, or weakens OTS to re-introduce the break, these tests
fail loudly.
