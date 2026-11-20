# WaveLock bounty scope index

**[PUBLIC_BOUNTY_V1.md](PUBLIC_BOUNTY_V1.md) is the authoritative scope for the
public reference implementation.** Its [target matrix](artifacts/public_bounty_v1.json)
separates implementation, regression, eligibility and proof status. Use the
[submission template](PUBLIC_BOUNTY_SUBMISSION_TEMPLATE.md) and
[reproduction guide](PUBLIC_BOUNTY_REPRODUCTION_GUIDE.md) for public reports.

## Public Bounty v1

| Offered surface | Target IDs |
|---|---|
| Canonical collision, preimage recovery, supported consensus nondeterminism, computation replay bypass, deeper semantic binding failure | Critical 1–5 |
| Noncanonical acceptance, NaN/Inf acceptance, signed-zero divergence, production input boundary bypass, unsupported backend leakage | High 1–5 |
| Implemented one-time identity, signed body/transcript, key/fingerprint/Merkle, header/linkage, replay restart/parsing and same-host process-race checks | AR-1–AR-6 |

The normative profile is `WL-Consensus-Commitment-v1`, defined in
[BOUNTY_SECURITY_PROFILE.md](../docs/BOUNTY_SECURITY_PROFILE.md). Its producer
is `wavelock.chain.consensus_commitment.commit_consensus_state`; its computation
replay verifier is `verify_consensus_commitment`. Historical WLv* serializers
are non-consensus for this offering. High 1–3 do not automatically escalate to
Critical 5. Critical 1/2 remain open cryptanalytic targets with no claimed
collision/preimage proof or formal PDE hardness reduction.

## Withheld surfaces

Critical 6 (remote runtime/machine attestation), Critical 7 (behavior-wide drift
detection), and broad Critical 8 (hostile rollback, arbitrary valid-format cache
tampering and trusted-anchor guarantees) are **not offered security properties
of Public Bounty v1**. Their absence is not a vulnerability in this scoped
offering. Only the implemented public record/OTS checks listed above are
eligible. Private/internal architecture is outside this public scope and is
neither described nor required here.

## Historical full research scope

Earlier documents explored a broader commitment/attestation/replay/drift
research boundary. They remain audit history, not current Public Bounty v1
promises or an eligibility list:

- [Original audit](REPORT.md) and [interpretation](README.md);
- [Hardening recommendations](HARDENING_RECOMMENDATIONS.md);
- [PR #22 hardening report](BOUNTY_HARDENING_REPORT.md) and
  [historical contract matrix](artifacts/bounty_contract_matrix.json);
- [Full pre-finalization scope at the merged PR #22 baseline](https://github.com/rickyjreyes/Wavelock/blob/c4b1df112c93db8017b788d7f8f58b4f0e2087f9/audit/BOUNTY_SCOPE.md).

The historical matrix's `FAIL` and `full_bounty_ready: false` fields describe
that broader inventory. They are preserved rather than rewritten as passes.
Public v1 records the three broader surfaces in `withheld_targets` and uses
separate `implementation_status`, `regression_status`, `bounty_status` and
`security_proof` fields. A passing defensive regression is not a proof that an
open attack target cannot be broken.
