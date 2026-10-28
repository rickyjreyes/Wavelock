# R22 preliminary formal verification

This Lean file proves a conditional injectivity lemma for an **abstract**
one-step commitment, and exhibits a simple noninjective polynomial evolution.

It is **not** an implementation model of the historical WaveLock PDE,
CC-Core-v1-B, CurvaChain or WaveLock-OTS. There is no proven OWF/security
reduction, collision resistance, or unavoidable sequential work here.

Formalization of the actual published protocol and modern versioned protocol
remains open in issue #37, with the attack/binding obligations of issue #32.
Build status: unverified at authoring time.
Run `cd lean && lake update && lake exe cache get && lake build`.
