# WaveLock CC-Core-v0 Prover/Verifier Protocol

**Status:** Experimental research protocol definition — Phase CC-1, Part VI.
**Constraint:** No conventional cryptographic primitive. No security claim.

This document defines three protocol variants for using the CC-Core-v0 trajectory
digest as a path-commitment primitive. No claim of interactive zero-knowledge,
unconditional binding, or computationally sound collision resistance is made.
These are definitional sketches for research purposes only.

---

## 1. Background

CC-Core-v0 produces a 256-bit digest `D = cc_hash(M)` that commits to:
- The terminal wave state ψ_T (inherited from Design A)
- The complete trajectory (ψ_0, ψ_1, …, ψ_T) via the accumulator field C

A prover who knows M can produce D. A verifier who knows D wants to check a claim
about M (e.g., that M has a certain structure, or that two messages share a prefix)
without necessarily seeing M in full.

---

## 2. Protocol Variant A: Plain Digest Commitment

**Primitive:** `D = cc_hash(M)`

**Prover step:**
1. Compute `D = cc_hash(M)` and publish D.

**Verifier step:**
1. Receive M (or a claimed M').
2. Recompute `D' = cc_hash(M')`.
3. Accept iff D == D'.

**Claim this makes:**
- "I know a message M such that cc_hash(M) = D."

**What is NOT claimed:**
- Collision resistance: we do not claim no M' ≠ M has cc_hash(M') = D.
- Binding: we do not claim a computationally bounded prover cannot find a second preimage.
- Zero-knowledge: the digest may leak information about M's structure.

**Use-case:** Integrity check (does M hash to D?). NOT a cryptographic commitment.

---

## 3. Protocol Variant B: Trajectory-State Commitment (Non-Interactive)

**Primitive:** `D = trajectory_digest(ψ_0)`

Here ψ_0 is a 16×16 field state (not a message), and the digest commits to the
ordered sequence (ψ_0, ψ_1, …, ψ_T).

**Prover step:**
1. Choose initial wave state ψ_0 ∈ F_p^256.
2. Compute D = trajectory_digest(ψ_0).
3. Publish D and optionally the intermediate states (ψ_1, …, ψ_T, C_1, …, C_T).

**Verifier step:**
1. Given (ψ_0, D): recompute trajectory and check D.
   - OR: given (ψ_0, ψ_1, …, ψ_T, C_0, …, C_T) and D: verify each coupled round
     and the squeeze output.

**Claim this makes:**
- "I know a wave trajectory (ψ_0, …, ψ_T) that produces digest D."
- Design A eigenmode states (≥47 verified) produce DISTINCT D values (see Parts I–II),
  so the trajectory digest does distinguish them.

**What is NOT claimed:**
- Uniqueness: we do not prove that no second trajectory produces the same D.
- Hardness: we do not prove that finding a second trajectory is computationally hard.

**Verification cost:** O(T × N²) field operations per round-step check.

---

## 4. Protocol Variant C: Prefix-Binding Verification

**Primitive:** Multi-block cc_hash with absorb state inspection.

This variant allows verifying that two messages M and M' share a common prefix
of exactly k blocks (k × BYTES_PER_BLOCK = 192k bytes).

**Setup:**
- Block size: 192 bytes (BYTES_PER_BLOCK).
- Internal state after k blocks: (ψ_k, C_k).

**Prover step:**
1. Absorb message M block by block. After block k, record (ψ_k, C_k).
2. Publish commitment `D_prefix = trajectory_digest(ψ_k, C_k_as_initial)`.
   (Or simply publish (ψ_k, C_k) if the verifier trusts the prover to run
   the correct accumulation.)

**Verifier step:**
1. Given the claimed prefix M[0..k-1] and the state (ψ_k, C_k):
   - Absorb the prefix from the standard IV and verify the state matches.
2. Verify that M has the same prefix by re-running the absorb on M[0..k-1].

**Claim this makes:**
- "Message M has the same first k blocks as claimed if the prover-provided (ψ_k, C_k)
  matches the honest re-computation."

**What is NOT claimed:**
- Hiding: the internal state (ψ_k, C_k) may be derivable from the prefix (it IS
  derivable by anyone who runs the absorb — it is NOT a commitment to the prefix
  in a cryptographic sense).
- Binding against a computationally unlimited prover: if the absorb has collisions
  at the state level, a dishonest prover could produce a different prefix with the
  same internal state.

---

## 5. Protocol Variant D: Interactive Trajectory-Equality Proof (Research Sketch)

**Goal:** Prover convinces verifier that two distinct initial states ψ_0 and ψ'_0
produce different trajectory digests, WITHOUT revealing ψ_0 or ψ'_0.

**Protocol (sketch):**
1. Prover commits: publish D = trajectory_digest(ψ_0), D' = trajectory_digest(ψ'_0).
2. Verifier sends challenge round index t_c ∈ {0, …, T−1}.
3. Prover opens: reveal (ψ_{t_c}, C_{t_c}) for both trajectories.
4. Verifier checks:
   - The coupled round (ψ_{t_c}, C_{t_c}) → (ψ_{t_c+1}, C_{t_c+1}) is consistent
     with the prover's claimed state at t_c+1.
   - The two states at t_c differ.

**Status:** This is a definitional sketch only. Soundness requires that the prover
cannot compute (ψ_{t_c}, C_{t_c}) consistent with D after seeing t_c — which would
require the squeezing to be a one-way function, which is NOT proved.

**What is NOT claimed:**
- Soundness: the protocol is not proved sound.
- Zero-knowledge: the opened state at t_c reveals information about the trajectory.
- Completeness: assumed (an honest prover can always open the correct state).

---

## 6. Claim Separation (Part VII)

The following three claims are distinct and have different support levels:

| Claim | Status | Evidence |
|---|---|---|
| **Trajectory uniqueness**: each ordered execution trace produces a unique digest | Unresolved | Toy-scale collisions exist (Parts VIII); N=16 unknown |
| **Path binding for the eigenmode family**: the 47 Design A zero-collapse states yield distinct digests | Confirmed (Part II) | 47/47 distinct, min HD 98 |
| **Hardness of inversion**: finding a second preimage requires ≥ 2^Ω(n) operations | Unproved | No lower bound; no demonstrated inversion either |

These three claims are **not equivalent** and MUST NOT be conflated:
- Path binding for the eigenmode family (confirmed) does NOT imply general trajectory uniqueness.
- General trajectory uniqueness (if true) does NOT imply hardness of inversion.
- Absence of a demonstrated attack (negative search result) is NOT a proof of hardness.

---

## 7. Protocol Security Summary

| Protocol | Completeness | Soundness | Zero-Knowledge |
|---|---|---|---|
| A: Plain digest | ✓ (trivially) | Not proved | Not claimed |
| B: Trajectory-state | ✓ (trivially) | Not proved | Not claimed |
| C: Prefix-binding | ✓ (trivially) | Not proved | Not claimed |
| D: Interactive sketch | Assumed | Not proved | Not claimed |

All four protocols are defined for **research purposes only**. None is recommended
for production use without a formal security proof.
