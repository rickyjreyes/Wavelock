# WaveLock — Public Research & Reference Implementation

WaveLock is a public research and reference implementation for deterministic
curvature-regulated commitments, one-time public signatures, authenticated
records, replay enforcement, and adversarial reproducibility.

The supported public commitment boundary is **`WL-Consensus-Commitment-v1`**.
WaveLock-OTS provides the current public-verification signing path, and
CurvaChain provides the local authenticated-record and replay layer used by the
reference workflow.

> **Status:** active experimental research and hardened public reference
> implementation. [Public Bounty v1](audit/PUBLIC_BOUNTY_V1.md) defines the
> security properties currently offered for external attack. WaveLock is not a
> formal cryptographic standard and should not be treated as a replacement for
> established production cryptography.

---

## Current public architecture

WaveLock separates the supported reference path from experimental research and
historical compatibility code.

1. **`WL-Consensus-Commitment-v1` — normative public commitment profile.**
   The reference producer fixes serialization, byte order, dtype, metadata,
   kernel parameters, transcendental evaluation rules, reduction order, finite
   value handling, signed-zero normalization, production input requirements,
   and the supported NumPy reference backend. The matching verifier replays the
   declared computation and rejects profile, metadata, state, invariant, and
   backend mismatches.
2. **WaveLock-OTS — asymmetric one-time signatures.** The supported signing
   layer provides public verification without revealing `ψ★`. Every key is
   one-time and replay state is enforced at authenticated record acceptance.
3. **CurvaChain — authenticated records and replay enforcement.** Canonical
   block bodies, OTS identities, commitments, chain linkage, persistent replay
   state, and same-host writer coordination are bound into the reference node
   workflow.
4. **CC-Core-v1-B — curvature/path commitment research.** This remains an
   experimental research core for trajectory-dependent commitment studies. It
   co-evolves an accumulator with the wavefield so the research commitment
   depends on the ordered trajectory rather than only on the terminal state.

### Research status of CC-Core-v1-B

Candidate B uses the linear injection `j_B(u,v) = u(1 + γv) mod p`, removes
Candidate A's generic 2-to-1 injection weakness, and separates all 47 known
Phase 8J terminal-collapse states (minimum pairwise Hamming distance 105/256).
The Phase CC-3 audit found:

- all 47 known terminal-state collapse cases remain distinct under the path
  commitment;
- the singular value `v_star = -γ^-1 mod p = 195225786` is unreachable at round
  0 under the normative message protocol;
- a replay-verified 191-byte message reaches `v_star` at one coordinate at round
  1, but the tested singular event did not erase the path commitment or produce
  a structural collision;
- 205 fast tests pass and GitHub Actions completed successfully on the research
  branch;
- multi-coordinate singular reachability, general binding, collision hardness,
  second-preimage hardness, and any Layer-3 lower bound remain unresolved.

The curvature-capacity research code is isolated in:

```text
wavelock/curvature_capacity/       # frozen CC-Core-v0-A baseline
wavelock/curvature_capacity_v1/    # current CC-Core-v1-B candidate
curvature_audit/                   # adversarial tests, artifacts, and reports
```

Primary research documents:

- `docs/CC_CORE_V1_SPEC.md`
- `docs/CC_CORE_V1_NORMATIVE_PROTOCOL.md`
- `docs/CC_CORE_V1_VSTAR_REACHABILITY.md`
- `docs/WAVELOCK_CURVATURE_CAPACITY_RESULTS.md`

### Use classification

| Component | Current status | Intended use |
|---|---|---|
| `WL-Consensus-Commitment-v1` | Normative public reference profile | Deterministic commitment and replay verification |
| WaveLock-OTS | Supported experimental OTS layer | Public one-time signing and authenticated records |
| CurvaChain | Reference ledger/replay layer | Local replay, persistence, linkage, and acceptance experiments |
| `CC-Core-v1-B` | Experimental research core | Path/trajectory commitment studies |
| WLv2 / SIGv2 | Historical legacy design | Reproduction and migration testing only |
| Ed25519 / SLH-DSA / LMS / XMSS | Established alternatives | Production security |

No general theorem currently establishes that WaveLock is provably secure,
collision-resistant, one-way, 256-bit secure, or that it forces full sequential
execution. Public Bounty v1 intentionally separates passing implementation
regressions from cryptographic proof.

---

## Security boundary

### Supported path

Normal `keygen`, `sign`, `mine`, and `verify` commands use **WaveLock-OTS** and
the authenticated-record path. Historical SIGv2 tools are isolated under:

```text
wavelock-cli legacy ...
```

Legacy SIGv2 is retained only so earlier artifacts, attack reports, and
migration behavior remain reproducible. It is **not part of the supported
WaveLock security profile**. The historical design requires the verifier to
possess `ψ★`, which makes it unsuitable as a public-key signature construction.
See `attacks/WAVELOCK_THEORY_BREAK_AUDIT.md` and
`docs/MIGRATION_FROM_SIGV2.md`.

### WaveLock-OTS

WaveLock-OTS is a Lamport/WOTS-style one-time signature with WaveLock state
binding underneath (`wavelock/crypto/wavelock_ots.py`, CLI `wavelock-ots`).
The supported implementation provides:

- **Public-only verification.** Public verification does not load `ψ★`, a seed,
  or a secret key.
- **Canonical public keys and signatures.** `verify_ots` and `load_public_key`
  enforce exact field sets, recompute the Merkle root, recompute the public-key
  fingerprint, and bind signatures to that fingerprint.
- **One-time identity enforcement.** A key is consumed once. Stateful block
  acceptance rejects reuse through the persistent replay ledger.
- **Production input boundary.** The normative commitment producer requires raw
  `bytes` of at least 16 bytes; 32 bytes is the default/recommended size. Byte
  length is an input rule, not a statistical entropy measurement.
- **No public `ψ★` or production input export.** Public commitment and OTS
  artifacts do not expose the private wave state or producer input.

One-time use is fundamental: reusing a Lamport-style OTS key can enable forgery.
The corresponding proof-of-concept is preserved as a regression test, while the
normal acceptance path rejects consumed identities.

WaveLock-OTS remains experimental and does not carry a formal security proof.
For high-value production signing, use established independently reviewed
cryptography such as Ed25519, SLH-DSA, LMS, or XMSS as appropriate to the use
case.

### Authenticated-record hardening

The public reference path binds the canonical block body, OTS identity,
authenticated header context, chain linkage, and replay state. The current
implementation includes:

- canonical block-body signing;
- consumed OTS identity reconstruction from accepted chain state;
- fail-closed handling of malformed authentication data;
- persistent replay records;
- inter-process replay acceptance locking (`flock` on POSIX, SQLite on Windows);
- same-host writer coordination and chain-tip reload before append;
- versioned block-header encoding;
- deterministic consensus commitment encoding and replay verification.

Cross-node/global consensus, hostile-host rollback resistance, remote runtime
attestation, and behavior-wide drift verification are not offered properties of
this public repository.

See:

- [Public Bounty v1](audit/PUBLIC_BOUNTY_V1.md)
- [Bounty reproduction guide](audit/PUBLIC_BOUNTY_REPRODUCTION_GUIDE.md)
- [Normative security profile](docs/BOUNTY_SECURITY_PROFILE.md)
- `attacks/WAVELOCK_OTS_REDTEAM.md`
- `attacks/WAVELOCK_MYTHOS_BREAK_REPORT.md`

---

## Public Bounty v1

Public Bounty v1 exposes only security properties implemented and reproducible
from this public repository.

**Offered targets:**

- Critical 1–5: commitment collision, preimage recovery, supported consensus
  nondeterminism, computation-replay bypass, and deeper semantic binding failure;
- High 1–5: noncanonical acceptance, NaN/Inf acceptance, signed-zero divergence,
  production input boundary bypass, and unsupported backend admission;
- AR-1–AR-6: implemented authenticated-record and OTS integrity/replay checks.

**Withheld from the public offering:**

- remote runtime/machine/kernel/configuration attestation;
- behavior-wide drift detection;
- broad hostile rollback, arbitrary valid-format cache tampering, and trusted
  external-anchor guarantees.

Those withheld areas are not offered security properties of Public Bounty v1.
Private/internal WaveLock architecture is outside this repository and is not
required to reproduce the public bounty surface.

Passing regression tests means the documented implementation boundary behaves
as expected on the tested cases. It does **not** constitute a cryptographic
security proof.

---

## WaveLock-OTS quick start

```bash
wavelock-ots ots-keygen  --out keys/
wavelock-ots ots-sign    --secret keys/wl_ots_secret.json --message "pay alice 5" --sig sig.json
wavelock-ots ots-verify  --public keys/wl_ots_public.json --message "pay alice 5" --sig sig.json
wavelock-ots ots-inspect --public keys/wl_ots_public.json
```

Each key signs **once**. Generate a fresh key per message.

## WaveLock-Encrypt quick start (experimental)

`WaveLock-Encrypt v1` (`wavelock/crypto/wavelock_encrypt.py`, CLI
`wavelock-encrypt`) is an experimental hybrid public-key encryption wrapper.
Confidentiality and integrity come from X25519 (ephemeral-static), HKDF-SHA256,
and ChaCha20-Poly1305. The WaveLock contribution is canonical transcript/context
binding: decryption fails closed if authenticated context such as purpose,
ψ-commitment, block digest, or OTS fingerprint changes.

See `docs/WAVELOCK_ENCRYPT_SECURITY_NOTE.md`. This wrapper has not received a
production security audit.

```bash
wavelock-encrypt keygen  --private wlenc_private.pem --public wlenc_public.pem
wavelock-encrypt encrypt --public wlenc_public.pem --input msg.bin --output env.json \
    --purpose "transport/demo" --psi-commitment <hex>
wavelock-encrypt decrypt --private wlenc_private.pem --input env.json --output out.bin \
    --purpose "transport/demo" --psi-commitment <hex>
```

---

## Install and run

Python 3.9+ is supported. The normative reference workflow uses NumPy and
requires no GPU.

```bash
python -m pip install -e ".[blake3]" pytest
python hello_wavelock.py
```

The demo generates a fresh OTS key, signs a canonical block body, mines and
accepts it, deletes the secret key, verifies in a new process using public
material, and checks replay rejection. It uses disposable state and leaves
existing ledgers alone.

## Supported block workflow

Normal commands use **WaveLock-OTS**. A signature authenticates the canonical
block body and its parent. Mining uses that existing signature; it does not sign
a second message with the same key.

```bash
wavelock-cli --data-dir demo-node keygen --out keys/demo-1
wavelock-cli --data-dir demo-node sign --secret keys/demo-1/wl_ots_secret.json --message "research artifact abc" --output signed-block.json
wavelock-cli --data-dir demo-node verify --signed-path signed-block.json
wavelock-cli --data-dir demo-node mine --signed-path signed-block.json
wavelock-cli --data-dir demo-node verify
```

`add ricky` is a convenience for generating a fresh pair in `keys/ricky/`;
`sign ricky --message "..."` uses that pair. Labels are local aliases, not
long-lived authenticated identities. Each pair signs once. Generate another
pair in a new directory for the next block. Existing key files are never
silently overwritten. A stale signed parent requires a fresh key and signature.

The signed artifact contains the public key and selected OTS slices only.
Verification never loads a secret key, integer seed, registry, or ψ snapshot.
The standalone `wavelock-ots` commands remain available for detached messages;
those detached signatures are not block authorizations.

## Node configuration and persistence

CLI and node use the same `WAVELOCK_DATA_DIR` (default: `$XDG_DATA_HOME/wavelock`
or `~/.wavelock`). The CLI also accepts `--data-dir` before its subcommand. Set
`WAVELOCK_DATA_DIR` for the node.

Accepted blocks and the reconstructable OTS replay cache live under `ledger/`.
Signer-use markers live under `ots-state/`. Keep backups of signing state and
never reuse an OTS key across hosts or restored copies.

CLI mining and node acceptance share a same-host writer lock and reload the tip
before appending. A draft signed against an old parent is rejected: use a fresh
key to sign for the new tip. This coordinates local writers; it does not provide
distributed consensus or authenticated recovery from complete host rollback.

```bash
wavelockd --port 9001
```

The node defaults to `require_ots=true`. It verifies stored OTS history on
startup and rejects historical SIGv2 on the normal acceptance path. Configure
via `WAVELOCK_CONFIG` or `wavelockd --config node.json`:

```json
{"port": 9001, "require_ots": true}
```

`WAVELOCK_REQUIRE_OTS=1` also enables this policy. Explicit `false` / `0` is
reserved for historical compatibility experiments. Unknown JSON configuration
keys are errors. File settings override environment defaults.

New OTS ledgers do not silently import or rewrite an old SIGv2 ledger.
Historical package-local data can be inspected with the migration tools in
`docs/MIGRATION_FROM_SIGV2.md`.

Replay acceptance is serialized across processes on one filesystem (POSIX
`flock` or SQLite locking), and tip validation plus append are serialized within
one node. The reference node is not a distributed transactional store; crash
gaps can conservatively consume a key without a completed block. Cross-node
consensus and Merkle many-key signing remain separate research/roadmap work.

---

## Verification

Full validation used for Public Bounty v1 includes the core/OTS, curvature, PDE,
bounty-contract, durability, header, dry-run, CPU-dispatch, Linux, Windows, and
container checks described in `audit/PUBLIC_BOUNTY_V1_READINESS.md`.

For local verification:

```bash
python -m pytest tests/ -q
python -m pytest curvature_audit/ -c curvature_audit/pytest.ini -q
python -m pytest pde_audit/ -c pde_audit/pytest.ini -q
python audit/run_bounty_contract.py --check
python hello_wavelock.py
```

`tests/test_supported_workflow.py` exercises the real CLI across fresh processes,
public-only verification, key reuse, malformed artifacts, startup replay
reconstruction, configuration, and concurrent acceptance.

---

## Research and historical tools

- `wavelock/curvature_capacity_v1/`: current CC-Core-v1-B research candidate.
- `wavelock/curvature_capacity/`: frozen Candidate A baseline.
- `wavelock/pde_hash/`: historical hash-free PDE core and regression target.
- `wavelock/crypto/`: OTS, replay protection, and encryption wrappers.
- `wavelock/chain/ots_blocks.py`: shared OTS transcripts and pure public checks.
- `wavelock-cli legacy ...`: historical SIGv2 reproduction/migration tools.

Historical SIGv2 findings remain documented and reproducible without being part
of the supported security profile.

## License and patent notice

Copyright © 2025–2026 Ricky Reyes. All rights reserved.
See `LICENSE` and `PATENT_NOTICE.md`.
