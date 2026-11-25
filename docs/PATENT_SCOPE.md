# Patent / Repository Relationship

This document explains the relationship between the public WaveLock repository
and the inventor's patent filings. It is an engineering notice, not a claim
chart, prosecution statement, legal opinion, disclaimer, or admission.

## Controlling patent record

WaveLock-related subject matter is described in the pending U.S. nonprovisional
application under docket **REYES-WAVELOCK-2026-NP**, titled:

**Curvature-Regulated Wavefield Evolution Methods, Protocol-Binding Commitment
Systems, and Drift Detection Apparatus**

The application was filed June 7, 2026 and claims the benefit of U.S.
Provisional Patent Application No. **63/826,336**, filed June 18, 2025.

The filed application describes, among other subject matter:

- curvature-regulated wavefield evolution and commitments;
- canonical serialization and kernel-bound state representations;
- commitment-and-replay ledger systems;
- curvature-derived invariants;
- computational-system drift detection and attestation;
- protocol-binding material;
- public-verifier embodiments;
- one-time-use and consumed-use authorization;
- replay prevention tied to accepted ledger or attestation state;
- authenticated-encryption context binding and access-control embodiments; and
- computational-agent and distributed-system embodiments.

The patent application itself, any later prosecution record, and any issued
claims control the legal patent analysis. Repository documentation does not
amend the patent application.

---

## Repository purpose

This repository is a changing research and reference implementation. It
contains prototypes, test harnesses, deprecated code, experimental branches,
security audits, attack reproductions, partial implementations, and engineering
roadmaps.

A technical statement about this repository is not a statement about patent
scope. In particular:

- an implementation being absent from this repository does not disclaim or
  abandon patent subject matter;
- an implementation being experimental or incomplete does not establish lack
  of written description, enablement, utility, validity, or enforceability;
- a failed test, security weakness, deprecated implementation, or redesign does
  not by itself narrow or surrender any patent claim;
- a repository implementation being present does not establish that a patent
  claim covers it; and
- later repository changes do not retroactively alter what was disclosed in a
  filed patent application.

Technical results should continue to be reported accurately. Negative results
and known limitations are engineering evidence and should not be hidden or
rewritten as legal conclusions.

---

## Current engineering map

### Curvature-regulated evolution and commitment

The Python repository contains implementations of the curvature-regulated PDE
operator, deterministic state evolution, canonical serialization, hash-family
binding, and related commitment experiments. See `docs/CANONICAL.md` for the
current reference-implementation map.

### Commitment and replay ledger

The repository contains ledger, Merkle-binding, one-time-signature,
consumed-identifier, and replay-rejection experiments. These implementation
notes are intended to document software behavior, not to define patent claim
boundaries.

### Drift detection and attestation

The filed patent application includes drift-detection and attestation subject
matter. The public Python repository does not need to contain every embodiment
or every production implementation described in the patent filing.

Any current or future drift-detection experiments may be documented here as
engineering work without characterizing them as filed, unfiled, supported,
unsupported, valid, invalid, enabled, disabled, patentable, or unpatentable.

### Protocol binding and public verification

The filed application includes protocol-binding, public-verifier,
replay-prevention, consumed-use, context-authentication, authenticated-encryption
context, access-control, and computational-agent embodiments. Repository code
may implement some, all, alternate, or later-developed versions of those ideas.
The implementation status is not a legal scope statement.

---

## Safe documentation rule

Repository documentation should describe **what the code does and what testing
shows**. It should avoid making unnecessary legal conclusions such as:

- "this claim is unsupported";
- "this feature is outside the patent";
- "we will not file this claim";
- "this implementation creates an enablement gap";
- "this test failure invalidates the patent";
- "this code is required for the patent to be valid"; or
- similar statements purporting to decide patent scope, validity, priority,
  enablement, enforceability, or infringement.

When a technical limitation exists, state the limitation directly and precisely.
For example: "the current containerized telemetry experiment produced low
variance in five channels" is an engineering result. Whether that result has any
legal patent consequence is a separate question for the patent record and legal
analysis.

---

## Licensing and reserved rights

Public availability of this repository does not grant a patent license and does
not dedicate WaveLock inventions to the public. Patent and copyright rights are
expressly reserved as stated in the repository's [`LICENSE`](../LICENSE) and
[`PATENT_NOTICE.md`](../PATENT_NOTICE.md).

No repository file should be read as granting commercial, deployment,
manufacturing, derivative-work, or patent rights unless a separate written
license expressly grants those rights.

---

## Maintenance

Keep engineering documentation current, reproducible, and candid. Keep legal
scope conclusions out of engineering audit files unless they reproduce an
actual filed statement or are added by qualified counsel for that purpose.
