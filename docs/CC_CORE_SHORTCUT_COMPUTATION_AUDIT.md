# CC-Core-v0 Shortcut Computation Audit

**Status:** Research analysis — Phase CC-1, Part V.
**Constraint:** No forbidden claim.

A shortcut computation is any algorithm that computes the trajectory digest in
fewer than the nominal ~6.55×10⁵ field operations (forward cost, as established
by resource_bounds.py). This document audits candidate shortcuts and records
whether any is known to succeed or is ruled out.

---

## 1. Candidate Shortcut: Skip Wave Computation

**Idea:** Given the message, can the attacker compute the final C_T without
computing all ψ_t (t = 0 to T)?

**Analysis:**

The accumulator update at each round requires:

```
C_{t+1}[x] = f(C_t, ψ_t[x], ψ_{t+1}[x])
```

where ψ_{t+1} = F(ψ_t). The injection j depends explicitly on both ψ_t[x] AND
ψ_{t+1}[x]. To evaluate j without running F would require knowing the output
of the wave round without computing it.

**Ruling out mechanism:** The wave round F is a nonlinear polynomial of degree 3
in ψ. There is no known shortcut to evaluate F on an arbitrary input faster than
direct computation. Any shortcut to skip F must either invert F (at least as hard
as preimage finding for Design A) or predict F(ψ) without computing it (no known
method). **This shortcut is not demonstrated.**

---

## 2. Candidate Shortcut: Backward Computation

**Idea:** Given the final state (ψ_T, C_T), can the attacker recover C_{T-k}
by inverting Φ_t?

**Analysis:**

Inverting Φ_t requires: given C_{t+1} and (ψ_t, ψ_{t+1}), find C_t such that

```
μ·cd + A_C·cd²  +  W_t·j  +  ρ_t = C_{t+1}[x]  mod p
```

where cd = C[x] + D_C·Lap(C)[x].

Because of the Laplacian coupling, cd is a linear function of ALL C cells, not just
C[x]. Inverting the system for C requires solving a system of 256 quadratic
equations over F_p (from the A_C·cd² term). This is a degree-2 polynomial system
inversion problem (NP-hard in general over a generic field, but not provably hard
over F_p for this particular structure).

**Feasibility note:** For degree-2 systems over F_p with 256 variables and 256
equations, Gröbner basis methods have complexity roughly O(p^(n·(D−1))) where n=256
and D=2. In practice this is infeasible for p=2³¹−1. However, **no hardness proof
exists**; this is a heuristic infeasibility argument based on known algebraic geometry.

The round-dependent constant ρ_t and the position weights W_t(x) further complicate
the backward problem: even if one round were inverted, the accumulator state at the
PREVIOUS round is needed to invert the NEXT earlier round.

**Verdict:** No backward shortcut is demonstrated. The infeasibility is heuristic, not proved.

---

## 3. Candidate Shortcut: Meet-in-the-Middle on C

**Idea:** Can the attacker split the trajectory at round T/2 and search from both
ends to find a matching interior state?

**Analysis:**

A meet-in-the-middle attack on the trajectory requires:
- Forward computation for rounds 0..T/2: produces a set of (ψ_{T/2}, C_{T/2}) pairs.
- Backward computation for rounds T/2..T: requires inverting Φ_t (see §2 above).

Since backward computation is not demonstrated, the meet-in-the-middle attack does
not reduce the cost below the forward cost. If backward computation WERE feasible
at cost B per round, the meet-in-the-middle attack would cost O(p^(256/2) · B)
which is not obviously smaller than O(p^256).

**Verdict:** MITM attack not demonstrated as a shortcut.

---

## 4. Candidate Shortcut: Cycle Detection on C

**Idea:** If the accumulator has a short cycle (C_t = C_{t+k} for small k),
the trajectory would be periodic, enabling a shortcut by detecting the cycle.

**Analysis:**

The accumulator Φ_t is **round-dependent** (the constant ρ_t = RHO0 + RHO1·t varies
with t). A cycle of the trajectory would require not just C_t = C_{t+k} but also
the periodic recurrence to lock to a sub-period of the round-constant schedule.

Because ρ_t is linear in t with slope RHO1 = 507468114 (mod p, coprime to p-1),
the schedule has period exactly p − 1 (the multiplicative group order). No short
cycle of the round-dependent accumulator was found in the two-cycle search (Part IV,
budget 3000).

**Verdict:** Short cycles not observed within search budget. Not proved absent.

---

## 5. Candidate Shortcut: Algebraic Solve via Low-Degree Structure

**Idea:** If C_T is a low-degree polynomial in the initial ψ_0, can an attacker
solve for ψ_0 by linear algebra?

**Analysis:**

The degree-growth trace (Part IV) shows the heuristic degree of C_T in ψ_0
grows as roughly 2·3^(T−1). For T=32 rounds this exceeds p−1 = 2³¹−2 by many
orders of magnitude; Fermat reduction makes the true degree unpredictable but
at most p−1. A polynomial over F_p of degree p−1 with 256 variables would require
a coefficient table of size p^256, far beyond any feasible representation.

**Verdict:** No low-degree shortcut is known. The degree upper bound is heuristic
(not a Gröbner result); the true degree over F_p is unresolved.

---

## 6. Candidate Shortcut: Exploit the 2-to-1 u-Injection

**Idea:** Since j(u,v) = j(u',v) for u' = −(1+γv)/η − u, can this be used to
find two distinct wave trajectories yielding the same C trajectory?

**Analysis:**

The pairing maps u → u' = −(1+γv)/η − u. However, v = F(ψ_t)[x] depends on ψ_t
at ALL cells through the wave round; changing ψ_t[x] from u to u' changes
ψ_{t+1}[x] = F(ψ_t)[x] as well (since F is nonlinear and globally coupled).
The new v' = F(ψ')[x] ≠ v in general, breaking the cancellation.

Empirical check (Part IV, η-pairing attack, 200 trials): no trajectory digest
collision found by this method. Wave round equalities (F(ψ) = F(ψ')) were also
not found, confirming the pairing does not commute with F.

**Verdict:** 2-to-1 pairing does not yield a practical digest collision.

---

## 7. Shortcut Computation Summary

| Shortcut | Status | Reason |
|---|---|---|
| Skip wave computation | Not demonstrated | Requires evaluating F without computing it |
| Backward computation (Φ_t inversion) | Not demonstrated | Requires solving degree-2 system of 256 equations |
| Meet-in-the-middle on C | Not demonstrated | Depends on backward computation |
| Cycle detection | Not observed (bounded search) | Round-dependent constant prevents short cycles |
| Low-degree algebraic solve | Not demonstrated | Degree grows too rapidly; Gröbner unavailable |
| 2-to-1 injection exploit | Not demonstrated | Pairing breaks when applied to coupled (ψ,F(ψ)) |

No shortcut is demonstrated. None of the infeasibility arguments is a formal proof.
The absence of a demonstrated shortcut is **not** a claim of hardness.

---

## 8. Forward Cost Reference

From resource_bounds.py (Phase 1):

```
Forward operation count per digest: ~6.55 × 10^5 field ops
Field: F_p with p = 2^31 − 1
State: 512 cells (256 ψ + 256 C)
Rounds: T = 32 per block
```

Any shortcut must undercut this. None of the candidates above does so.
