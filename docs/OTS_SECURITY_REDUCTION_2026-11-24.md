# One-time security reduction: exact obligations (WaveLock #52)

Base: master at 4e14ffbf204e1622551a2ee701b9797677285504.

## Existing byte semantics

The v1 implementation uses 256 digest bits and 256-bit honest secret slices.
For fixed public parameter hash p, its domain-separated functions are:

- D_p(m) = SHAKE256(WL-OTS-MSG-v1, p, m).
- L(s) = psi_commitment(evolve_psi_star(s, params)).
- K_{i,b}(s) = SHAKE256(WL-OTS-SK-v1, s, L(s), p, i, b).
- H_{i,b}(x) = SHAKE256(WL-OTS-PK-v1, p, i, b, x).

The implementation length-prefixes each argument in _h. The verifier permits
arbitrary decoded byte-string lengths for x; inversion must allow any accepted
preimage, not only a 32-byte candidate. Messages are normalized bytes
(UTF-8 for strings). Different Python objects representing identical bytes
are not distinct cryptographic messages.

The game fixes the authenticated public key; it does not let the adversary
substitute a fresh key. Key consumption and ledger safety are separately
proved in existing modules and are not secrecy assumptions.

## Unconditional mathematical core

With at most one signing query, consider an accepted forgery on a distinct
message. Equal digests extract a collision; unequal digests differ in at least
one coordinate. The accepted opening of the opposite half at that coordinate
is a preimage of its public commitment. The argument does not require the
secret halves to be independent or unique.

OTSSecurityReduction.lean encodes this event cover and finite uniform random
coin count. If each of the 2n tagged unopened events has probability <= e,
then forgery probability is at most collision-event probability + 2n e.
This is an event bound on the same experiment, not yet a proof that each
tagged event has the standard preimage-game distribution.

## Computational reduction (conditional, mathematical argument)

For a versioned security-parameter family with n(lambda)>0, polynomial n,
efficient key generation, and uniform accepted seed sampling, consider:

1. Real experiment: publish L(s) and commitments derived from K(s).
2. Ideal-table experiment: sample L(s) from its real marginal, replace the
   full table K(s) by mutually independent uniform secret strings, and
   recompute all public metadata.

A sufficient assumption is joint computational indistinguishability of
(L(s), K(s)) and (L(s), U), at the chosen resource bound, with advantage
epsilon_joint. Standard PRF security without the seed-dependent leakage L(s)
does not imply this. For example, a uniformly random secret table is an
ideal random function on the finite domain, yet publishing that same table
as leakage reveals every opening. Marginal pseudorandomness alone is
insufficient. This example refutes the inference, not WaveLock's particular
SHAKE/PDE construction.

In the ideal-table experiment a collision extractor runs the forger and
outputs its two messages if the digests agree. For the inversion extractor,
sample a uniform tag J=(i,b) among the 2n halves. Given an honestly distributed
challenge commitment H_J(U_J), sample all other independent secret halves,
generate L(s) independently with its real marginal, and recompute the full
public commitment table, Merkle root, key identifier and fingerprint. If the
signing query requests the embedded half, abort; otherwise answer it using
the known halves. On a distinct-message accepted forgery opening J without
disclosing J in the signing query, return its bytes. This is a valid
preimage, even if it differs from U_J. The simulated transcript before abort
has the ideal distribution because all the independent halves have their
correct joint distribution. For every ideal successful forgery with unequal
digests, at least one of the 2n guesses is an unopened half it opens.
Thus its probability is bounded by 2n times the inverter's success.

For any at-most-one-query adversary, the zero-query case can be handled by
first running that adversary, making a dummy query on a different message
after it produces its candidate, ignoring the answer, and returning the
candidate (provided the message domain contains two efficiently selectable
distinct messages). The one-query theorem suffices for the resulting game.

Consequently, with resources charged for sampling, the public PDE computation,
table generation, hashes, the forger, the signing query and verification,

    Adv_EUF-1 <= epsilon_joint + Adv_collision + 2n * Adv_tagged_preimage.

This requires tagged average-case preimage hardness for uniform honestly
generated slice challenges, collision resistance of the normalized message
digest, the joint leakage assumption, and an exact adapter from the parsed
Python verifier to the typed opening predicate. The inverter's challenge
distribution samples a uniform tag and uniform honest slice, and its output
domain includes all verifier-admissible strings. No independence of different
SHAKE domains follows just from their tags; assumptions must hold in the
actual joint oracle environment or as an explicit random-oracle model.

For n=256 the loss factor is 512. The fixed v1 parameters themselves do not
define an asymptotic PPT/negligible family. No numerical security-bit claim
follows without quantified advantages and resource limits. SHAKE security
and the real joint leakage assumption remain unproved; public recomputation
of the psi commitment is not itself proof of seed recovery or of hardness.

## Verification and scope

The Lean module covers only the unconditional event extraction and the
finite count inequality. The hybrid and efficient adversary simulations
above are mathematical conditional arguments, not yet mechanized. In
particular OTSVerification.OTSCrypto.slicesOK is opaque and needs the
explicit decoding/indexing adapter for verify_ots before claiming full
implementation-level correspondence. The historical replay/relabel and
reuse counterexamples are unchanged. No claim of P != NP or PDE hardness
is used. Issue #52 remains open.

Kernel build and strict module audit must succeed before merge. The new
audit is independent of the existing historical receipt, so the original
audit is still executed unchanged. A fresh consolidated receipt is a later
verification artifact.

## Hosted verification receipt (2026-11-24)

Lean source commit `3c096218155356389ab76777cec138a1a453d183`; GitHub Actions run [45445082370](https://github.com/rickyjreyes/Wavelock/actions/runs/45445082370), job 138626512055. Full library build: 7243 jobs, success. Targeted reduction build: 7222 jobs, success. Historical strict audit: no unexpected axioms or failed declarations. New reduction audit: 8 constants, all passed; all four theorem roots use only `propext`, `Quot.sound`, `Classical.choice`. Core/container CI also passed. This verifies the event and count theorems, not the unmechanized hybrid, SHAKE security or Python correspondence. Keep #52 open and await maintainer authorization before merge.
