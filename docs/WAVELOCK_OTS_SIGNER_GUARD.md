# WaveLock-OTS signing guard: material single use

Baseline: `master` at `c3cb0d1` (after PR #47). Related: issues #41, #42, #43 (the signing
guard is the only preexisting mechanism in this repository with persistent safety memory and a
pre-action gate; the RCA formalization is a separate pull request).

## Summary

`sign_ots` refuses to reveal secret slices twice on one host. It does so through a host-local
registry of marker files (`WAVELOCK_OTS_STATE_DIR`), each published atomically with
`atomic_write(exclusive=True)` before any slice is revealed.

At `c3cb0d1` the registry recorded only the `one_time_key_id`, which is a label. The secret
slices depend on the seed and the parameters, not the label. Two key files holding one seed under
two key ids therefore both signed through the guarded path. This happens with:

- `generate_ots_keypair(seed=s)` called twice, which assigns a fresh UUID each time, or
- a relabelled copy of a used secret file.

Reproduced, 122 of 256 positions then have both secret halves public, and
`attacks/ots_reuse_to_total_forgery.py` turns such pairs into forgeries. PR #47's verifier-side
material consumption rejects the second *acceptance* of a table, but it cannot stop the
*publication* of the second signature. A forger who learns both halves can race the honest
signature to the ledger.

`sign_ots` now also claims the **signing material** before revealing anything:
`material-<_h("WL-OTS-SIGNER-MATERIAL-v1", params_hash, seed)>.used`, next to the key-id marker.

## What determines the secret slices

```
sk[i][b] = _h("WL-OTS-SK-v1", seed, psi_commitment(evolve_psi_star(seed, params)), params_hash, i, b)
```

`_h` is SHAKE-256 over domain-separated, length-prefixed fields, so the inputs are framed
unambiguously.

- The slices of a key file are a function of `(seed, params, params_hash)`.
- The material marker is a function of `(params_hash, seed)`.

The consequences:

- **Equal derivation inputs give equal markers.** This needs no hash assumption. The second
  `sign_ots` over the same inputs is refused.
- **Files that differ only in `params`** share a marker but derive different slices. The result is
  a refusal (liveness), never a missed reuse.
- **A collision between the markers of two different materials** can likewise only cause a
  refusal.
- **Distinct markers derive equal slices** only through a SHAKE-256 collision.

## Exact guarantees

| Reuse | Guarantee | Status |
|---|---|---|
| Same derivation inputs `(params_hash, seed)`, any key ids, files, copies | No second signature on one registry (`OTSSigner.signer_single_use`) | `LEAN_PROVED` (model); conformance-tested against `sign_ots` |
| Same key id | No second signature (`signer_single_use`) | `LEAN_PROVED`; unchanged behaviour |
| Same secret values through `sign_ots` | Distinct, given an injective slice derivation (`signer_values_distinct`) | `CONDITIONAL` on SHAKE-256 collision resistance |
| Same public commitment table | Signer: as the previous row (the table is a function of `params_hash` and the slices). Verifier: PR #47, `wavelock_ots_material_single_use` | `CONDITIONAL` / `LEAN_PROVED` |
| Same secret values recommitted by custom code that does not call `sign_ots` | None: no guard runs on that path | `OPEN` (needs value-level consumption, PR #47 obligation 1) |

The model covers any interleaving of any number of processes, claim failures before and after
publication, crashes between claims, restarts, and power loss with POSIX directory `fsync`.

## Boundaries

Each boundary below is a kernel-checked counterexample in `OTSSignerGuard.lean`. None was
corrected, because each would need a different architecture.

| Boundary | Lean |
|---|---|
| `allow_reuse=True` / `--unsafe-allow-reuse` (documented as tests only) | `allow_reuse_bypasses_guard` |
| A second registry directory (another `WAVELOCK_OTS_STATE_DIR`, another host) | `second_registry_bypasses_guard` |
| A deleted registry | `wiped_registry_forgets` |
| Windows: no directory `fsync`, so power loss can drop a claimed marker (#26) | `windows_power_loss_forgets` (POSIX: `posix_power_loss_keeps`) |
| Material that signed before this change has only a key-id marker | `legacy_material_signs_once_more` |
| The advisory `used` flag alone (historical, Finding D) | `flag_only_copy_signs_twice` |

Failures are fail-closed:

- A crash after the claims burns the key (`crash_after_claims_burns_key`).
- A claim whose directory sync failed is published and blocks a retry
  (`failed_dir_sync_blocks_retry`).
- A failure after the key-id claim and before the material claim burns the key id and reveals
  nothing. The material can still sign once under another id (tested with a real process exit).

## Compatibility and migration

- **No wire format, signature, public-key, ledger or consensus change.** The verifier side,
  including PR #47's material consumption, is untouched.
- **The registry is local signer state.** The new marker type carries a versioned domain tag. Old
  markers keep working. Older code ignores the new markers, because it only ever claims key-id
  markers.
- **Signer behaviour is stricter.** A second key file holding the same `(params_hash, seed)` is
  refused on the host that signed the first. Honest key generation with random seeds never
  produces such files.
- **Migration.** Key files that signed before this change have no material marker.
  `wavelock_ots.claim_signing_material(secret_key)` records the material of such a file so a
  relabelled copy is refused. Material whose key file is gone cannot be recovered from the
  registry: it names only key-id hashes.
- **Privacy.** Marker names are hashes; contents are constant strings. The files are created with
  mode `0o600` and contain no seed bytes (tested).

## Evidence

| Evidence | Where |
|---|---|
| Lean model and theorems | `lean/WaveLockProofs/OTSSignerGuard.lean`: guard revisions `flagOnly`, `c3cb0d1`, `current`; `signer_single_use`, `signer_values_distinct`, `durable_markers_monotone`, counterexamples |
| Conformance (14 traces) | `lean/ots_signer_traces.json`: each checked by `decide` in `OTSSignerConformance.lean` and replayed against real `sign_ots` calls, keys and registries by `tests/test_ots_signer_conformance.py`. Injected claim failures (before and after publication) and a crash after the claims; the `c3cb0d1` and pre-registry guards are emulated. Negative controls show the old outcome only under the old guard, and that the registry is load-bearing. |
| Regression tests | `tests/test_ots_signer_material.py`: same seed under two key ids, relabelled and plain copies, the `c3cb0d1` leak preserved, distinct parameters, marker privacy and mode, unusable keys claim nothing, 8 threads and 6 processes racing on one material (exactly one signs), a crash between the two claims, the migration helper |
| Strict audit | `lean/scripts/AxiomAudit.lean` (receipt `lean/axiom_audit_receipt.txt`): every constant of the audited modules depends only on `propext`, `Classical.choice`, `Quot.sound` |

## Open obligations

1. Secret-value single use across custom recommitments (value-level consumption).
2. Cross-host and cross-registry coordination (only the verifier ledger and chain are shared).
3. Durable registries on Windows (#26).
4. A refinement proof from Python to the model; the evidence is trace conformance.
