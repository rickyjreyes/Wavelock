# WaveLock-PDE-256-v0 — Multi-block Reachability & Collision-Lifting (Phase 8K)

**Status:** experimental. Branch `research/pde-eigenmode-collision-audit`.
Artifacts: `pde_audit/artifacts/phase8k_multiblock_reachability.json`,
`pde_audit/artifacts/phase8k_reduced_lifting.json`. Tests:
`pde_audit/test_multiblock_reachability.py`.

> **Observed attack cost is not a lower bound and is not a proof of one-wayness,
> collision resistance, or cryptographic security.**

## 0. Question

Phase 8J proved the internal one-round map is constructively non-injective (≥47
explicit preimages of zero). This phase asks whether a **valid message** can
steer the absorbed state onto a structured collision state, onto zero, or into
any internally colliding pair — i.e. whether the internal non-injectivity
**lifts** to the message layer. The normative primitive was **not modified**.

## 1. Exact multi-block model (Part II)

Block transition (`U_k` = pre-`evolve_T`, `S_{k+1}` = post):
```
U_k      = S_k + I(B_k, k, final)         (mod p)
S_{k+1}  = F^T(U_k) = evolve_T(U_k)
```
where `I` adds the 64 packed elements into cells `0..63`, the counter digits into
`cap0=64` and `cap2=66`, finalization into `cap1=65`, and **cells `67..255` are
not touched by the current block**. `S_0` = public IV.

`pde_audit/multiblock_reachability.block_trace` exposes `S_before`,
`U_after_injection`, every round state, and `S_after` per block. It reproduces
the normative reference **and** optimized implementations **byte-for-byte**
(parity-tested on empty/abc/zero/ff/structured messages).

## 2. First-block reachability — IMPOSSIBLE (proven) (Part IV)

For a one-block message, `U_0` has cells `67..255` equal to fixed IV values
`IV[c] = (1 + c + tag[c mod 19]) mod p`, all `≤ ~300`. Every structured collision
state requires those cells to be `±s` with `s ∈ {151946369, 1217065103,
1395627816, …} ~10⁸–10⁹`. Checking **all 46 enumerated eigenmodes** under **all
symmetry and sign variants**, the minimum number of mismatching uncontrolled
cells is **189 (all of them)**. Therefore **no structured collision state can
equal `U_0` for any one-block message** — an exact impossibility proof, not a
failed search. (This bound is specific to the first block.)

## 3. Differential controllability of the 189 capacity coordinates (Part VI)

For `k>0`, hitting an eigenmode `U_k` requires the prior blocks to steer
`S_k[67..255]` (189 coordinates) onto the eigenmode capacity pattern. Exact
modular Jacobian rank of the map *(prior-block rate inputs → `S_k[67..255]`)*:

| prior blocks | input vars | capacity rank / 189 | deficiency |
|---|---|---|---|
| 1 | 64 | **64** | 125 |
| 2 | 128 | **128** | 61 |
| 3 | 192 | **189** | **0** |

So with **≤2 blocks the capacity target is dimensionally unreachable** (rank <
189), and at **3 blocks the dimensional obstruction vanishes** (rank saturates to
189). This is *local, linearized* controllability — **full differential rank
does not equal global reachability.**

## 4. Linearized (modular-Newton) steering — does NOT converge (Part VI.2)

Exploiting the rank-189 map, a modular linear solve (`mod_solve`) was used to
compute a rate-input correction driving `S_3[67..255]` toward the checkerboard
capacity pattern, then re-evaluated through the **exact nonlinear** pipeline and
iterated. The capacity-mismatch residual stayed pinned at **189** across all
iterations (equal to the random baseline). **Reason:** `𝔽_p` has no
valuation/Hensel descent, so a linearized step solves the *linear* system but
need not reduce the *nonlinear* residual. Local solvability did not yield a lift.

## 5. Heuristic & relaxed searches — no lift (Parts III, V, IX)

- **Model B (byte-constrained), 6 000 messages, 1–3 blocks:** best capacity-coord
  match to checkerboard **0/189**; min pre-squeeze nonzero coords **256/256**;
  min digest Hamming distance to `00…00` **98** (random ≈128). **No exact lift.**
- **Model A (relaxed F_p rate), one block, 6 000 trials:** never reached zero;
  capacity cells `67..255` stay at IV, so the 64-dim rate slice generically
  misses the finite preimage set `F^{-T}(0)`.

## 6. Reduced-model lifting — mechanism CONFIRMED in the small case (Parts VII–VIII)

Same architecture (toroidal Laplacian + cubic reaction, rate/capacity split,
counter, finalization), `N=2` (rate=1 cell, capacity=3 cells), exhaustive over
all `𝔽_p` message symbols:

| model | blocks | msgs→zero | msg collisions | image frac |
|---|---|---|---|---|
| N=2, p=7, T=1 | 1 | 0 | 0 | 1.000 |
| N=2, p=7, T=1 | 2 | 0 | 0 | 1.000 |
| N=2, p=7, T=1 | 3 | **1** | 32 | 0.907 |
| N=2, p=7, T=2 | 3 | **6** | 80 | 0.767 |
| N=2, p=5, T=any | 3 | 0 | 122 | 0.024 |
| N=2, p=11, T=1 | 3 | 0 | 84 | 0.937 |

**SMT (z3):** `N=2,p=7,T=1,2 blocks` → **UNSAT** (no message reaches zero);
`N=4,p=5,1 block` and `N=4,p=7,2 blocks` → **UNSAT**; larger reduced cases →
`unknown` (z3 nonlinear-integer limit).

**Mechanism:** a message-preimage of zero is **provably absent below
≈capacity/rate blocks** (SMT UNSAT / exhaustive 0) and **appears once block
count reaches capacity/rate** (N=2 capacity=3, rate=1 → 3 blocks; verified exact
witnesses). Message-level pre-squeeze collisions are abundant in every reduced
model (internal non-injectivity **does** lift to message-level collisions in the
small case). This mirrors the normative differential-rank curve (rank saturates
at 3 blocks).

**Why it may or may not scale to the normative system:** the reduced "lift at
capacity/rate blocks" relies on solving the nonlinear system once the dimension
suffices. In the reduced case the field is tiny and the system is solvable by
exhaustion. In the normative system the dimension suffices at 3 blocks (rank
189), **but** (a) modular-Newton does not converge, (b) exhaustive/SMT search is
infeasible at `p=2³¹−1, T=32`, and (c) the byte model additionally forces the
eigenmode rate cells (`±s ≈10⁸`) outside the valid packed range `[0,2²⁴)`, so an
*eigenmode* `U_k` is unreachable under Model B even if its capacity were steered.
Whether some **non-eigenmode** message preimage of zero exists at ≥3 blocks is
**unresolved**.

## 7. Distinguisher (Part X)

Correlation between the checkerboard-mode projection of the pre-squeeze state and
each output bit: **max |corr| = 0.07** over 1 500 messages. No structural leak.

## 8. Layered verdict

| Property | Result |
|---|---|
| Global injectivity of state transformation | **False** (8J) |
| Internal state collision resistance | **Broken** (8J) |
| Internal preimage resistance for target zero | **Broken** (8J) |
| Generic arbitrary-target state inversion | **Unresolved** |
| Direct first-block eigenmode reachability | **Impossible** (proven, §2) |
| Multi-block eigenmode reachability (≤2 blocks) | **Impossible** (rank < 189, §3) |
| Multi-block eigenmode reachability (≥3 blocks) | **Unresolved** (rank saturates; no lift found) |
| Lifting in reduced models | **Demonstrated** (message→zero & collisions, §6) |
| Full message collision resistance (normative) | **Unresolved** |
| Full message preimage resistance (normative) | **Unresolved** |

- **State transformation:** constructively non-injective; preimage of the
  specific target zero is easy; generic inversion unresolved.
- **Complete message digest:** **Unresolved experimental candidate.** Lifting
  was demonstrated only in reduced models; in the normative system no message
  collision or message preimage of zero was found within the algebraic,
  differential, and heuristic budgets, and it is **not proven impossible** for
  ≥3 blocks. The phrase "the collision does not lift" is **not** used (no
  impossibility proof exists for the ≥3-block regime).

## 9. Reproduce

```bash
python -m pde_audit.multiblock_reachability   # seed 80120
python -m pde_audit.reduced_lifting           # seed 80130
python -m pytest pde_audit/test_multiblock_reachability.py -c pde_audit/pytest.ini -q
```
