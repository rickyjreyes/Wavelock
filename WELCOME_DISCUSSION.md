# Welcome to the WaveLock Security Research Discussions

This forum is for inspectable discussion of the experimental path-commitment core, WaveLock one-time signatures, CurvaChain replay enforcement, canonical transcript binding, encryption context binding, tests, audits, and formal security questions.

WaveLock is research software for local development and controlled evaluation. It is not approved for production funds, secrets, identity, or critical infrastructure.

## Good contributions include

- exact test-suite reproductions;
- independent implementations;
- authorized defensive security evaluations;
- replay-prevention and canonicalization regression tests;
- fuzzing and property-based testing;
- formal theorem statements and proof attempts;
- resource and performance benchmarks;
- misuse-resistance improvements;
- comparisons with established cryptographic primitives;
- clear identification of unsupported or overstated security claims.

## Reporting standard

Identify the component, protocol version, specification section, repository commit, property under evaluation, attacker model, authorized test scope, environment, hardware, parameters, random seeds, resource limits, stopping rule, commands, fixtures, quantitative results, and machine-readable artifacts.

Separate four statements:

1. the implementation passed a particular test;
2. a property held under a finite evaluation scope;
3. a precise theorem or reduction was proved;
4. the system is suitable for production security.

These are not equivalent. Passing tests does not establish general one-wayness, collision resistance, unforgeability, sequential hardness, or production readiness.

Do not post private keys, credentials, live infrastructure details, sensitive third-party data, or material from systems you are not authorized to evaluate. For production applications, use established and independently reviewed cryptography.

Negative results, blocked proofs, deprecated designs, and identified weaknesses are valuable research outputs and should remain visible.
