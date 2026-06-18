# WaveLock Curvature-Capacity Core — Adversarial Audit Results & Final Report

**Branch:** `research/curvature-capacity-wavelock` · **Date:** 2026-06-18
**Construction:** CC-Core-v0 (`wavelock/curvature_capacity/`, see
`docs/WAVELOCK_CURVATURE_CAPACITY_SPEC.md`).
**Artifacts:** `curvature_audit/artifacts/` (each records branch, commit,
equations, parameters, seed, environment, runtime, budget, raw result,
interpretation, limitations). Environment: Python 3.11, NumPy 2.4.6, Linux
x86_64. z3 / sympy / Sage **not** installed (recorded limitation).

> **Observed attack cost is not a lower bound and is not a proof of one-wayness,
> collision resistance, or cryptographic security. Negative results below carry
> their search budgets and are not proofs.**

---

## FINAL CONCLUSION

### Experimental: mechanism partially supported, no proof

The path-commitment idea is **partially supported**: the explicit Design A
eigenmode collisions, which collapse the *terminal wave state* to zero, are
**no longer digest collisions** under the trajectory commitment (9/9 distinct,
min pairwise Hamming 116/256), known low-complexity bypasses (eigenmode,
symmetry, fixed-point, additive/sign cancellation) were removed or not found
within budget, and no practical attack succeeded within stated budgets. **But no
lower-bound theorem exists**, the coupled round is **not** proven injective
(toy models show residual collisions), curvature magnitude is a
convention-dependent diagnostic with no link to attack cost, and the heat
argument is interpretive. The construction therefore qualifies as *Experimental*
under the task's rubric, **not** "conditionally supported" (which would require a
restricted-model theorem) and **not** "rejected" (no decisive break was found).

---

## 20-point final report

**1. Exact statement recovered from the P-vs-NP and curvature work.**
The repository's "inevitability" material asserts a *constraint claim*: no
verifier-accepted commitment exists unless a curvature-bounded evolution to a
reproducible ψ★ is executed, and producing ψ★ "cannot be cheap" (E_min > 0).
This is an **acceptance definition plus a cluster of structural arguments**, not
a complexity theorem. Full classification: `docs/CURVATURE_CAPACITY_CLAIM_AUDIT.md`.

**2. What is proved vs conjectured.**
*Proved (class 2):* the Laplacian-eigenmode collision theorem `F(s·σ)=0 iff
s²≡b−(2rD−1)/a`, its ≥47-state enumeration, the constructive non-injectivity of
the Design A state map, the refutation of "full-rank Jacobian ⇒ injective", and
first-block unreachability. *Conjecture / physical intuition (class 4):* every
"cannot be cheap" / Lyapunov / entropy / heat lower bound, and all five
attack-bounding arguments (which only defeat *approximate* attacks). No attack
cost or heat lower bound is proved.

**3. Formal curvature measure chosen.**
Diagnostic functionals on the centered integer lift: spatial curvature
K2 = Σ(Δψ)², gradient energy, temporal curvature, trajectory separation D_Γ,
spectral effective rank, digest entropy (`curvature_audit/curvature_metrics.py`).
**Finding:** curvature *magnitude* is lifting-convention dependent (K2 = 80
centered vs ≈9.2×10¹⁹ naive for one structured state — 18 orders of magnitude).
No curvature functional is a cryptographic invariant; all are diagnostics.

**4. Formal physical / computational resource model chosen.**
Three models in `docs/CURVATURE_HEAT_RESOURCE_MODEL.md`: R_comp (field ops;
forward = O(blocks·T·N²) ≈ 6.55×10⁵ ops/digest, polynomial), R_phys (Landauer
2.87×10⁻²¹ J/bit, Margolus–Levitin/Bremermann, Bekenstein — each bounds an
*implementation*, not the problem), R_machine (commodity → exascale budgets).

**5. Whether any exponential lower bound was actually proved.**
**No.** Neither 𝒞_attack(n) ≥ 2^{Ω(n)} nor Q_heat(n) ≥ 2^{Ω(n)} is established.
Forward cost and forward heat are polynomial; attack cost is not bounded below
at all. Curvature growth saturates in T and is ≈ constant in message length
(scaling artifact: exponential-fit R² ≈ 5×10⁻⁵, i.e. no exponential trend).

**6. Whether the heat argument is mathematically necessary or interpretive.**
**Interpretive only.** Landauer bounds irreversible-erasure energy of a
particular implementation; the WaveLock map is reversible-circuit-embeddable, so
its evaluation has no nonzero Landauer floor in principle. The energy figure for
a hypothetical 2²⁵⁶-operation attack (3.32×10⁵⁶ J) is meaningful *only if* such
an operation lower bound were proved — it is not. "Heat makes inversion
impossible" is unsupported and forbidden.

**7. Path-commitment candidates implemented.**
Candidate A (coupled accumulator field) — **primary, fully implemented and
analyzed** (reference + optimized, byte-parity). Candidate B (irreversible
trajectory injection) is subsumed by A. Candidate C (two-way ψ↔C coupling) is
described and reserved (kept out of the primary so Design A eigenmodes apply
cleanly to ψ). Candidate D (delayed state) is future work.

**8. Whether known Design A eigenmode collisions remain digest collisions.**
**No** — this is the central positive result. All representative eigenmode states
(checkerboard ±s, row/col stripes ±s, period-4 ±s, and the zero state) collapse
to terminal wave 0 under the unchanged Design A round (verified), yet produce
**9 distinct trajectory digests**, min pairwise Hamming 116. The accumulator's
odd-in-u injection (j=u when v=0) preserves the sign that Design A's
terminal-state map discarded. Pinned in
`test_eigenmode_regression.py::test_trajectory_commitment_separates_eigenmodes`.

**9. New eigenmode or symmetry bypasses found.**
**None within budget.** No sign-eigenmode / structured / random C mapped to zero
under the accumulator step (530,544 candidates checked). No toroidal symmetry
(translation, rotation, reflection, transpose, global sign) preserved the
trajectory digest. The zero-wave digest is non-degenerate (not all-zero bytes),
unlike Design A's zero state.

**10. Trajectory uniqueness results (pebble property).**
Over 1,200 random message pairs, **100%** gave distinct pre-squeeze coupled
states *and* distinct 256-bit digests; 0 full collisions. The basic pebble
condition (m≠m′ ⇒ Γ(m)≠Γ(m′)) holds across the sample; bounded-domain transcript
injectivity (no two messages share C_T) held over 2,000 samples. **Not** a proof
of global injectivity.

**11. Path-commitment collision results.**
Truncated 24-bit collision found at 3,291 evals vs birthday-expected 5,134
(ratio 0.64 — generic). Max 16-bit multicollision = 3 (generic). No structural
collision weakness within budget; full 128/256-bit behaviour extrapolated, not
observed.

**12. Curvature-growth scaling.**
Wave curvature reaches a stationary band quickly (saturates, like avalanche by
T≈8) — **no exponential growth in T**. Accumulator curvature vs message length
is essentially flat (no exponential, no meaningful power-law growth). The digest
description size is fixed at 256 bits regardless of n. Conclusion: **no measured
quantity grows exponentially.**

**13. Forward-cost scaling.**
Polynomial: O(blocks·T·N²); ≈6.55×10⁵ field ops per single-block digest, linear
in #blocks. Measured throughput 49.3 digests/s (unoptimized; ~2× Design A's
cost because two fields co-evolve). Forward execution does **not** require
exponential work (a rejection criterion that is satisfied/avoided).

**14. Attack-cost scaling.**
Only measured on truncations (≤24-bit collisions, generic rate). No exponential
attack-cost lower bound is proved or even strongly evidenced beyond small
truncations. **The asymmetry C_forward∈poly vs C_attack≥2^{Ω(n)} is assumed, not
demonstrated.**

**15. Reduced-model behavior.**
Toy coupled cores (N=2; p=5,7,11) enumerated. The bare wave round is heavily
non-injective (Design A finding reproduced: hundreds–thousands of colliding
buckets). The coupled (ψ,C) round **reduces** collisions by a large factor but
does **not** eliminate them (51, 51, 180 residual), dominated by N=2 neighbour
degeneracy. So the accumulator demonstrably helps but does **not** restore
injectivity even at toy scale; lift to N=16 is **unresolved**, not extrapolated.
z3/Gröbner unavailable — algebraic inversion at scale untested (limitation).

**16. Whether any attack bypasses curvature resolution.**
No attack that bypasses the path commitment was found *within budget*. However,
because no link between curvature resolution and attack cost is established, the
phrase "bypass curvature resolution" is not well-defined as a hardness statement
— there is no proved cost to bypass. This is the core conceptual gap.

**17. Exact artifacts and tests.**
Artifacts: `curvature_audit/artifacts/{INDEX,curvature_scaling,eigenmode_attacks,
path_commitment_attacks,resource_bound_analysis,curvature_metrics_demo,
reduced_models}.json`. Tests (70, all passing, <3 s): `test_parity.py`
(wave round = Design A, ref/opt parity, pinned vectors), `test_design_a_preserved.py`
(Design A digests + eigenmode theorem frozen), `test_eigenmode_regression.py`
(collapse + separation + non-degeneracy + symmetry), `test_forbidden_imports.py`
(no SHA/BLAKE/AES/etc. in package or audit). CI: `.github/workflows/curvature_audit.yml`.

**18. Rejected / experimental / conditionally supported.**
**Experimental.** Known shortcuts removed, exact trajectory commitment
implemented, no practical attack within budgets, curvature metrics defined and
measured — but **no lower-bound theorem**, and curvature growth is statistical/
diagnostic, not a proven hardness driver.

**19. Precise security claims that ARE allowed.**
- "Every representative Design A eigenmode collision (terminal wave state 0) maps
  to a *distinct* CC-Core-v0 trajectory digest (verified, enumerated family)."
- "The wave round inside CC-Core-v0 is byte-identical to the frozen Design A
  round (tested)."
- "Reference and optimized implementations agree byte-for-byte (tested)."
- "Forward evaluation is polynomial: O(blocks·T·N²)."
- "Within the stated budgets, the digest shows full avalanche (mean HD 127.7),
  low monobit bias (max |z| 2.7, 0 bits |z|>3), generic truncated collision/
  multicollision behaviour, and no symmetry/eigenmode/fixed-point bypass."
- "Curvature magnitude is convention-dependent and is a diagnostic, not an
  invariant."

**20. Precise claims that remain FORBIDDEN.**
- "provably secure" / "collision-resistant" / "one-way" (no proof, no lower bound).
- "P versus NP proves security" (no such result; the inequality is unproved).
- "heat makes inversion impossible" (interpretive only; reversible embedding).
- "exponential curvature" (curvature saturates; no exponential growth measured).
- "the accumulator makes the map injective" (toy models show residual collisions;
  unresolved at scale).
- any claim of 256-bit security (only ≤24-bit truncations were searched).

---

## Decision-criteria checklist (Part XIV)

| Reject if… | Observed? |
|---|---|
| simple eigenmodes collapse the *digest* | **No** — separated (9/9). |
| accumulator permits additive / sign cancellation | **No** found (odd-in-u + position weights); but η·u² gives a per-cell ≤2-to-1 in u (documented; not a digest break). |
| trajectories merge via a low-degree relation | none found at digest level; toy coupled round has residual (N=2-degenerate) collisions. |
| curvature growth only visual/statistical | **Yes (diagnostic only)** — this keeps it out of "supported", into "experimental". |
| no curvature↔attack link | **Yes** — none established. |
| thermal argument = Landauer only | **Yes** — interpretive; explicitly not used as a proof. |
| forward execution exponential | **No** — polynomial. |
| reduced models reveal easy lifting / broad collapse | partial — toy non-injectivity persists; not extrapolated. |
| security relies on unproved P-vs-NP | would-be, so **no such claim is made**. |

Two reject-triggers (curvature is statistical-only; no curvature↔attack link)
are *present*, which is exactly why the verdict is **Experimental** rather than
conditionally supported — the construction is retained for further study, with no
security claim.

## Limitations

- All searches budgeted (≤ a few thousand digests); collisions only on ≤24-bit
  truncations; preimage/lower-bound behaviour at 128/256 bits **extrapolated, not
  observed**.
- No z3 / Gröbner / SAT solver installed → algebraic inversion at N=16 untested.
- Coupled-round injectivity unresolved; toy evidence is N=2-degenerate.
- Curvature functionals are lifting-convention dependent.
- Unoptimized implementation (49 h/s).
- **No proof. Nothing here bounds any attack from below.**

## Reproduce

```bash
python -m curvature_audit.run_audit          # all artifacts + INDEX.json (~6 min)
python -m curvature_audit.eigenmode_attacks  # the central separation result
python -m pytest curvature_audit/ -c curvature_audit/pytest.ini -m "not slow" -q
```
