# CC-Core-v0 Accumulator Algebra

**Status:** Research analysis — Phase CC-1, Part III.
**Constraint:** No conventional cryptographic primitive. No forbidden claim.

---

## 1. Notation

| Symbol | Value | Meaning |
|---|---|---|
| p | 2³¹ − 1 | Mersenne prime (field modulus) |
| N | 16 | Lattice side (N×N = 256 cells) |
| D_C | 3 | Accumulator self-diffusion coefficient |
| GAMMA (γ) | 11 | Cross-term coefficient u·v |
| ETA (η) | 13 | Self-square coefficient u² |
| ZETA (ζ) | 17 | Linear-in-v coefficient |
| A_C | 2 | Accumulator self-square coefficient |
| MU (μ) | 5 | Accumulator carry multiplier |
| RHO0 | 0x57434330 = 1463898160 | Round-constant base |
| RHO1 | 2654435761 mod p = 507468114 | Round-constant slope |
| WA, WB, WC | 40503, 50021, 60013 | Position-weight coefficients |

All arithmetic is modulo p.

---

## 2. The Accumulator Step Φ_t

The coupled round at time t maps (ψ_t, C_t) → (ψ_{t+1}, C_{t+1}):

```
ψ_{t+1} = F(ψ_t)          -- Design A wave round (frozen)
C_{t+1} = Φ_t(C_t, ψ_t, ψ_{t+1})
```

The accumulator update Φ_t is, **per cell x ∈ {0, …, 255}**:

```
u   := ψ_t[x]
v   := ψ_{t+1}[x]

j(u, v) := u  +  γ·u·v  +  η·u²  +  ζ·v        mod p   [injection]

Lap(C)[x] := C[x-N] + C[x+N] + C[x-1] + C[x+1] + (p-4)·C[x]  mod p
             (indices wrap toroidally)

cd[x]  := C[x] + D_C · Lap(C)[x]                mod p   [diffused C]

W_t(x) := 1  +  (t+1)·WA  +  (x+1)·WB  +  (t+1)·(x+1)·WC    mod p

ρ_t    := RHO0 + RHO1·t                          mod p

C_{t+1}[x] := μ·cd[x]  +  A_C·cd[x]²  +  W_t(x)·j(u, v)  +  ρ_t   mod p
```

---

## 3. Algebraic Degree Analysis

### 3.1 Injection term j(u, v)

j is a polynomial over F_p in (u, v) with total degree **2** (the highest-degree term is η·u²).

**Explicit expansion:**

```
j(u, v) = ζ·v  +  u·(1 + ζ)  +  γ·u·v  +  η·u²    [wrong expansion -- corrected below]
         = u  +  γ·u·v  +  η·u²  +  ζ·v             [correct]
```

Partial degrees: deg_u(j) = 2, deg_v(j) = 1.

### 3.2 The 2-to-1 structure of j in u

For fixed v, j(u, v) is a degree-2 polynomial in u. Its leading term η·u² is symmetric:
η·u² = η·(−u)². Therefore j(u, v) = j(u', v) if and only if:

```
u = u'    OR    u + u' = −(1 + γ·v) / η   mod p
```

(Proof: j(u,v) − j(u',v) = (u − u')(1 + γ·v + η(u + u')) = 0 mod p; since p is prime,
either the first factor is zero or the second is zero.)

This gives a **provable 2-to-1 map**: every j value has at most 2 preimages in u (for fixed v).
The pairing partner of u is:

```
u' = −(1 + γ·v)/η − u   mod p
```

**Known limitation:** This 2-to-1 structure is documented in the spec. It means the injection
term j is not injective in u. However, since ψ_t → ψ_{t+1} is determined by the wave round F
(which the attacker does not freely choose), the pairing does not yield a practical bypass.

### 3.3 Self-diffused accumulator cd

cd[x] = C[x] + D_C · Lap(C)[x] is a **linear** function of all C cells (degree 1 in each
C[y]), because the Laplacian is a linear operator.

### 3.4 Accumulator output C_{t+1}[x]

C_{t+1}[x] is a polynomial in (C[y] for all y, ψ_t[x], ψ_{t+1}[x]) of degree:

| Input | Degree |
|---|---|
| Each C[y] (through cd) | 2 (from A_C·cd²) |
| ψ_t[x] | 2 (from η·u²) |
| ψ_{t+1}[x] | 1 |
| Cross (C[y], ψ_t[x]) | 2 (cd is linear in C, multiplied by W·j which is linear in C only through the w term -- but W_t is fixed, so no cross in C) |

More precisely:
- The A_C·cd² term is quadratic in {C[y]} and independent of ψ.
- The W_t·j term is linear in {C[y]} × degree-2 in (ψ_t[x], ψ_{t+1}[x]) — wait, j does NOT depend on C, so the cross-degree in C from W·j is 0.

Corrected degree table for C_{t+1}[x] as a polynomial:

| Variables | Degree |
|---|---|
| C[y] for any y | 2 (from A_C·cd²) |
| ψ_t[x] alone | 2 (from j) |
| ψ_{t+1}[x] alone | 1 (from j) |
| joint (C, ψ) | degree 2 in C, degree 2 in ψ, no mixed terms (W·j and A_C·cd² are separate sums) |

### 3.5 Degree growth over T rounds

Heuristic upper bound (treating each cell independently, ignoring Fermat reduction):

| Round t | deg(ψ_t in ψ_0) | deg(j in ψ_0) | deg(C_t in ψ_0) |
|---|---|---|---|
| 0 | 1 | 2 | 0 |
| 1 | 3 | 6 | 2 |
| 2 | 9 | 18 | max(2·2, 18) = 18 |
| 3 | 27 | 54 | max(2·18, 54) = 54 |
| t | 3^t | 2·3^t | 2·3^(t−1) growing |

By round 4 the heuristic degree exceeds p−1 = 2³¹ − 2; Fermat reduction caps the
representable degree at p−1. **The true Gröbner-computed degree over F_p may be much lower.**
This upper bound is informational; no claim about cryptographic hardness follows from it.

---

## 4. Injectivity Properties

### 4.1 Wave round F (not modified)

F is the Design A Allen–Cahn step. Phase 8J proved: ≥47 distinct states map to 0 under F.
F is demonstrably **not injective** on the full domain.

### 4.2 Accumulator step Φ_t (per round)

Φ_t is a polynomial map F_p^256 → F_p^256 in (C, ψ_t, ψ_{t+1}).

For **fixed ψ_t, ψ_{t+1}** (hence fixed j per cell), Φ_t is a degree-2 polynomial in C
(from the A_C·cd² term). A degree-2 polynomial map over F_p is NOT guaranteed injective.

The toy-scale full joint enumeration (Part VIII) finds many coupled collisions at p=3, N0=2
and p=5, N0=2, confirming non-injectivity of the coupled round on the full joint domain in
small cases. Results are NOT extrapolated to N=16.

### 4.3 Coupled round (ψ, C) → (ψ', C')

The coupled round is a map F_p^512 → F_p^512. It is **not proved injective**. At toy scale
(N0=2, small p) the reduced exhaustive analysis finds collisions on the full joint domain.

### 4.4 Trajectory uniqueness vs path binding vs hardness

These are three distinct claims (see also Part VII):

1. **Trajectory uniqueness**: do distinct message prefixes ever produce identical (ψ, C)
   trajectories? — Unresolved at N=16. Toy enumeration finds collisions.

2. **Path binding**: do the 47 Design A zero-collapse states yield distinct CC-Core-v0
   trajectory digests? — **Confirmed** (47/47 distinct, min Hamming 98) by Parts I and II.

3. **Hardness of inversion**: does computing a second preimage of the trajectory digest
   require ≥ 2^Ω(n) operations? — **Unresolved.** No lower bound is proved.

---

## 5. Known Structural Weaknesses

| Weakness | Severity | Status |
|---|---|---|
| j is 2-to-1 in u (η·u² term) | Structural (provable) | Documented in spec; no bypass demonstrated |
| Coupled round non-injective (toy scale) | Demonstrated | NOT extrapolated to N=16 |
| No lower bound on inversion cost | Theoretical gap | Unresolved |
| Curvature magnitude is convention-dependent | Interpretive | Documented |
| Landauer/heat argument is interpretive | Not a cryptographic bound | Documented |

---

## 6. Degree Comparison: Candidate A vs Candidate B

| Property | Candidate A (η ≠ 0) | Candidate B (η = 0) |
|---|---|---|
| Degree of j in u | 2 | 1 |
| j injective in u? | No (2-to-1) | Yes (for most v) |
| Family separation | 47/47 (min HD 98) | 47/47 (min HD 105) |
| Avalanche (mean HD) | 127.2 | 128.0 |

Candidate B avoids the provable 2-to-1 weakness while achieving equal or better family
separation. Adopting Candidate B would require a spec revision and re-audit.

---

## 7. Summary

The accumulator step Φ_t is a degree-2 polynomial map over F_p in (C, ψ_t, ψ_{t+1}).
The injection term j is degree-2 in (u, v) with a provable 2-to-1 structure in u.
The coupled round is demonstrably non-injective at toy scale.
No attack has converted these structural properties into a digest collision.
No security lower bound has been proved.
