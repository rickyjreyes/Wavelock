# GitHub Discussions Guide

Use GitHub Discussions for questions, protocol proposals, independent reproductions, defensive security evaluations, proof attempts, benchmark results, regression findings, and interpretation of WaveLock research artifacts.

## Recommended categories

- **Announcements**: Releases, deprecations, protocol-version changes, audit summaries, and maintainer notices.
- **General**: Broad discussion about the research architecture and repository direction.
- **Q&A**: Focused questions tied to an exact component, specification section, function, test, commit, or artifact.
- **Ideas**: Proposed protocol changes, proof obligations, defensive evaluations, misuse-resistance improvements, and comparisons with established cryptography.
- **Show and tell**: Independent builds, authorized security evaluations, regression results, fuzzing, benchmarks, formal statements, and reviews.
- **Polls**: Community priorities only. Polls are not security evidence.

## Current status boundaries

WaveLock is experimental research software. Its current components include an experimental path-commitment core, an experimental one-time signature layer, and a prototype ledger with replay enforcement. Legacy SIGv2 and WLv2 behavior is deprecated and insecure. None of the experimental components should be described as provably secure, production-ready, generally collision-resistant, generally one-way, or as having a proved sequential-work lower bound.

For production use, rely on established, independently reviewed primitives and standards such as Ed25519, SLH-DSA, LMS, XMSS, X25519, HKDF, and ChaCha20-Poly1305 as appropriate.

## Required evidence separation

Every substantive post should distinguish among:

1. **Implementation behavior**: what the current code accepts, rejects, computes, or preserves.
2. **Test evidence**: which finite suite, search budget, fixtures, environments, and parameters were evaluated.
3. **Formal status**: what precise theorem, reduction, or proof obligation is checked, incomplete, or absent.
4. **Security interpretation**: which attacker model and property are supported, contradicted, or unresolved.

A passing test suite proves neither general cryptographic security nor absence of undiscovered weaknesses. A failed authorized evaluation establishes only that the tested method did not violate the target property within its declared scope and budget.

## Minimum standard for security claims

Include the component and protocol version, repository commit, specification reference, exact property under evaluation, attacker model, authorized scope, environment, hardware, parameters, seeds, resource budget, stopping rule, commands, fixtures, machine-readable outputs, quantitative findings, comparisons with established baselines, and limitations.

Never post private keys, seeds, credentials, live endpoints, sensitive third-party information, or instructions targeting systems you are not authorized to test. Move bounded implementation tasks to issues while keeping protocol design, evidence interpretation, and research questions in Discussions.
