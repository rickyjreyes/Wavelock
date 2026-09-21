# WaveLock bounty security profile

Status: implemented by merged PR #22. The algorithm, encoding and reference
vectors below are unchanged by Public Bounty v1 finalization; the stable
`v0.2.0` tag remains `9430442ae93e26c192d9be1f5ee9137369f5df37`.
[Public Bounty v1](../audit/PUBLIC_BOUNTY_V1.md) defines current public target
eligibility. [The earlier enforcement matrix](../audit/artifacts/bounty_contract_matrix.json)
is a historical full-research inventory; its broader attestation, drift and
hostile-storage entries are not offered Public Bounty v1 properties.

## Normative API and inputs

`wavelock.chain.consensus_commitment.commit_consensus_state` is the sole
supported producer of `WL-Consensus-Commitment-v1` artifacts. Example:

```python
import secrets
from wavelock.chain.consensus_commitment import (
    commit_consensus_state, verify_consensus_commitment, write_commitment_artifact,
)

secret_input = secrets.token_bytes(32)
artifact = commit_consensus_state(secret_input)
assert verify_consensus_commitment(artifact, secret_input)
write_commitment_artifact("commitment.json", artifact)  # refuses overwrite
```

Input must be exactly Python `bytes`, at least 16 bytes. Use 32 freshly random
bytes by default. Integers (including large integers), strings, bytearrays,
empty/short bytes and demo modes are rejected. This enforces representation
and length, **not statistical entropy**: 16 predictable bytes are not a
128-bit secret. Public test vectors below are test inputs only.

The producer validates input, selects the reference implementation, evolves
the state, rejects nonfinite values, normalizes zero, recomputes invariants,
serializes, hashes and labels the artifact. Callers do not assemble those steps.
`canonical_serialize` is a lower-level encoding function: it does not assert
that an arbitrary state was produced from a qualifying input.

## Exact profile descriptor and backend

[bounty_profile_v1.json](../audit/artifacts/bounty_profile_v1.json) is the exact
wire metadata object. Its bytes are sorted-key, compact, ASCII JSON with no
newline; the formatted file is a readable representation. All its fields are
bound in the preimage. Unknown/missing/extra/modified fields or profile versions
are rejected, including a changed kernel hash, dimensions, steps, parameters,
XOF, backend, normalization or array order. Metadata key insertion order has
no significance.

Only `backend="numpy-reference-v1"` executes here. The implementation always
uses NumPy CPU binary64; it does not dispatch to a caller backend. CuPy/GPU,
FFT, `numpy-fast`, fast-math and reassociation requests are rejected, as are
unregistered aliases. A state supplied to the serializer must be exactly a
NumPy ndarray with a binary64 floating dtype and shape `(4, 4)`; no implicit
conversion of GPU arrays, object arrays, integers, complex values or float32.
Big/little-endian internal arrays, Fortran order and noncontiguous layouts are
copied to the same logical C-order binary64 state.

The reference keeps the NumPy equation and binary64 array operations, with
fixed `exp`/`log` rounding and explicit reduction order. Raw NumPy's
CPU-dispatched transcendentals demonstrably differ; they are not used by this
producer. [Before/after evidence](../audit/artifacts/bounty_dispatch_after.json)
and subprocess regressions cover the observed mismatch.

The CI conformance matrix is Ubuntu 24.04/Python 3.9 and 3.12, and
Windows/Python 3.12, using the package's NumPy dependency. All use the **same**
three exact commitment vectors; platform-specific expectations are forbidden.
This is a tested reference implementation, not a proof that arbitrary NumPy
builds, CPUs or floating-point environments agree for all possible inputs.
New implementations must match the reference bytes and pass conformance before
being registered; merely selecting a matching label does not attest a runtime.

## Initialization, evolution and invariants

1. Compute 128 XOF bytes with SHAKE-256 over
   `b"WL-PSI-INIT-v1" || uint64_be(len(input)) || input`.
2. Split into sixteen 8-byte big-endian unsigned integers. For each, retain
   the low 53 bits and divide by `2**53` as binary64. Reshape C-order to `(4,4)`.
3. Execute 50 updates using the existing reference equation. State and array
   intermediates are binary64; exp/log use the explicit rounding rule below.
   Each NumPy operation evaluates separately in the order below;
   do not fuse or reassociate arithmetic. Periodic `roll(x,+1,axis)` and
   `roll(x,-1,axis)` supply the four neighbors:

```python
L(x) = ((((-4.0*x + roll(x,1,0)) + roll(x,-1,0))
          + roll(x,1,1)) + roll(x,-1,1))
fb   = alpha*L(psi) / (psi + epsilon*exp(-beta*psi**2))
ent  = theta * (psi * L(log(psi**2 + delta)))
dpsi = dt*(fb-ent) - damping*psi
psi  = psi + dpsi
```

| Parameter | Value |
|---|---:|
| alpha | 1.5 |
| beta | 0.0026 |
| theta | 0.00001 |
| epsilon | 0.000000000001 |
| delta | 0.000000000001 |
| dt | 0.1 |
| damping | 0.00002 |

The descriptor's `binary64` hex strings specify these constants exactly.
`exp` and `log` have the bound rule
`decimal80-half-even-to-binary64;exp-below-minus1000-is-zero-v1`: convert the
binary64 operand exactly with `Decimal.from_float`, evaluate Decimal exp/ln
rounded to 80 decimal digits with round-half-even, then convert the result to
binary64. The fixed context has `Emin=-999999`, `Emax=999999`, `clamp=0`, and
traps invalid operations, division by zero and overflow; it is independent of
the caller's Decimal context. For exp inputs strictly below -1000 return
positive zero (strictly below the binary64 underflow rounding threshold).
Log requires a positive finite operand. This specifies a decimal-then-binary
rounding rule, not an assertion of correctly rounded binary transcendental
functions for every possible input. Python documents the decimal exp/ln
rounding semantics in its [Decimal reference](https://docs.python.org/3/library/decimal.html).

The kernel identifier is `WL-psi-001-reference-v1`. The kernel hash is SHA-256
of the canonical JSON wire representation of the descriptor's `kernel` object.
It binds the declared equation and parameters; it is not a measured executable
hash or remote attestation measurement.

4. Compute `gx, gy = numpy.gradient(psi)` with unit spacing, default axes and
   edge order 1. Compute `fb` and `ent` again on the final state. Invariants are
   `E_grad = float(sum(gx*gx) + sum(gy*gy))`,
   `E_fb = float(sum(fb*fb))`, `E_ent = float(sum(ent*ent))`, and
   `E_tot = float((E_grad + E_fb) + E_ent)`. Each `sum` uses the fixed
   `pairwise16-v1` rule: flatten C-order to `a[0:16]`, set each of eight lanes
   `r[i] = float(a[i]) + float(a[i+8])`, then evaluate
   `((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]))` in binary64,
   preserving every indicated grouping. This pins the 16-element reference
   reduction without allowing runtime SIMD dispatch to choose another order.

Overflow, invalid arithmetic and division by zero raise before emission.
NaN and both infinities are rejected in every state element, floating metadata
value, supplied invariant and recomputed invariant. Underflow is allowed.
Normalize each floating `-0.0` to `+0.0` before serialization; ordinary negative
numbers are preserved. Supplied invariants must equal recomputation on the
normalized state. There is no tolerance or rounding of nonzero state values.

## Normative preimage and output

| Order | Encoding |
|---|---|
| 1 | Six magic/version bytes `57 4c 43 43 00 01` (`WLCC`, 0, 1) |
| 2 | Four-byte unsigned big-endian metadata byte length |
| 3 | Exact canonical ASCII JSON metadata described above |
| 4 | Sixteen IEEE-754 binary64 state values, big-endian, C order (128 bytes) |
| 5 | `E_grad`, `E_fb`, `E_ent`, `E_tot`, each binary64 big-endian (32 bytes) |

Metadata floats have one wire form: `{"binary64":"<16 lowercase hex digits>"}`
containing their big-endian binary64 bytes. Integers are JSON integers;
booleans are JSON booleans. Keys are sorted lexicographically; separators are
`,` and `:`; ASCII escaping follows Python `json.dumps(ensure_ascii=True)`;
NaN/Inf JSON extensions are forbidden. The registered descriptor contains no
arbitrary user strings or free-form fields. Wire header bytes must match it
exactly, including whitespace and scalar types.

The preimage is 1409 bytes in v1. The output is the literal profile identifier,
`:`, and the lowercase 64-character hex SHA-256 digest of the entire preimage.
There is no trailing newline in either hashed preimage or commitment string.
`validate_canonical_bytes` checks magic, exact header bytes, body length,
finite values, invariants and byte-for-byte reserialization; trailing bytes,
alternate JSON, a little-endian body and raw negative-zero bytes fail closed.

| Input | SHA-256 portion of commitment |
|---|---|
| `bytes(range(16))` | `6983241d9dee059dd82de32b996cf68fafa451a2bbe3775a675b3e791ce992eb` |
| `bytes(range(24))` | `0bf16e7bb05d2f3c3d572c60922465302cb2d8c30eecade6c8f2bc61e56b39d3` |
| `bytes(range(32))` | `f0d67235e3d0e712c31598f22a51fd303ac0334ecfa9fc0a7aec16e59fed3e46` |

Preliminary PR commits used an incomplete math descriptor and a different
header length/hash. Those unpublished vectors are retained in
[bounty_dispatch_before.json](../audit/artifacts/bounty_dispatch_before.json).
The final descriptor binds the corrected math rules; body hashes for all three
original reference vectors are preserved exactly. No released profile changed.

## Artifact, replay and authentication relationship

`CommitmentArtifact` contains the commitment and canonical preimage in memory.
Its public `to_dict()`/JSON file contains exactly `profile`, `metadata` and
`commitment`; no secret input or wave array. Treat the in-memory preimage as
sensitive. `write_commitment_artifact` checks encoding/hash consistency and
publishes the public JSON atomically, exclusively, after file flush.

`verify_consensus_commitment(artifact, secret_input, state=..., invariants=...)`
repeats the declared evolution and compares the entire commitment/metadata and
any supplied state/invariants. This is computation replay with an input the
verifier possesses. It is not public proof of secret knowledge, machine
attestation, a drift oracle, or a general signature. Encoding/hash consistency
alone does not prove the originating input, software or machine.

WaveLock-OTS remains the public signature/authentication layer, with its v1
signature format, fingerprint/Merkle checks and canonical OTS transcript v1.
This new commitment profile does not silently change OTS's existing internal
wave commitment. A deployment can place the public commitment artifact in
`extra_meta` of a signed block: the OTS signature then binds that exact object.
The node authenticates those bytes; applications must explicitly call the
commitment replay verifier if they require proof of the declared computation.
No new asymmetric construction is introduced.

## Block-header version and history

Historical header v1 concatenated unframed fields. `(index=1,timestamp="23")`
and `(index=12,timestamp="3")` have identical preimages when the other fields
match. New header v2 hashes `b"WL-BLOCK-HEADER-v2\x00"` followed by compact,
sorted, ASCII JSON containing `header_version,index,timestamp,previous_hash,
merkle_root,difficulty,block_type,meta,nonce`. Nonfinite JSON is rejected.
This format is separate from the floating commitment encoding above.

New OTS blocks also sign `meta.block_header` containing exactly
`version,index,timestamp,difficulty`. The existing v1 OTS transcript already
binds arbitrary metadata; its shape and domain remain unchanged. Only mining
nonce/hash may change after signing. Historical v1 OTS blocks remain readable
and verifiable with their original bytes. Absent `header_version` means v1;
unknown versions fail. The default OTS-required node accepts only new v2
blocks. A stored chain may have a v1 prefix, but cannot downgrade after v2.

`verify_ots_chain(blocks, expected_tip=trusted_hash)` can detect truncation
relative to an independently trusted tip. Without such an anchor, a valid
prefix alone cannot prove history completeness. The default node does not
obtain a trusted remote checkpoint automatically.

## Durability and remaining limits

Signer claims and immutable artifacts use same-directory temporary files,
file `fsync`, atomic exclusive hard-link publication, and POSIX parent-directory
`fsync`. Replacement of mutable key files uses atomic rename with the same
flush ordering. Windows uses flushed files and atomic publication, but Python's
portable implementation supplies no directory `fsync` power-loss guarantee.
The filesystem must support atomic links/renames and the selected lock method.

Replay acceptance holds same-host interprocess exclusion, rereads the ledger,
appends a validated record, flushes file and parent, then changes in-memory
state. Chain writers lock and reload the current tip before replay consumption
and chain append. POSIX uses `flock`; Windows uses SQLite exclusion. Invalid
or partial JSONL records stop loading; they are never silently skipped.

The replay ledger and block ledger are two files, not one atomic transaction.
A crash after consumption but before chain append can burn a key without
accepting its block. Retain both signing state and replay ledger; do not retry
with that key. Accepted-chain reconstruction recovers accepted identities but
cannot recover an unrecorded/burned identity after its only durable record is
deleted. Complete rollback of all local state cannot be solved by local locks.
Valid-format hostile edits to the unsigned replay cache are not fully
authenticated. These limitations keep the broad Critical 8 claim unresolved.

Same-host locking is not distributed consensus. Remote runtime attestation and
behavior-wide drift detection still require their own verifier and threat
model. The full bounty must not launch on the strength of thirteen passing
dry runs alone.

## Historical modes and versioning

Historical WLv1/WLv2/WLv3.1 and experimental CPU/GPU classes retain old encodings
for old artifacts and research. They expose `consensus_valid=False` and
`operating_mode="historical/research"`; their output cannot pass this profile's
verifier, and the historical serializer refuses the normative profile label.
Integer/demo seeds remain confined to those paths. SIGv2 remains deprecated.

Any consensus-relevant algorithm, parameter, shape, normalization or encoding
change requires a new registered profile/version and new vectors; never
reinterpret v1 or silently relabel WLv2. New block headers have their independent
version. This branch neither rewrites old artifacts nor moves `v0.2.0`.
