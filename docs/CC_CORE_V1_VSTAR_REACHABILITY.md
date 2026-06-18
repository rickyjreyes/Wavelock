# CC-Core-v1 / Candidate B — Singular-Value Reachability Results

**Status:** Phase CC-3, Part IX. Results labeled by rigor. No security claim.
Singular value `v_star = −γ⁻¹ mod p = 195225786`, `j_B(u, v_star) = 0 ∀u`.

---

## Theorem 1 (round-0 coordinate unreachability) — [theorem]

**Statement.** For every valid message `m`, no coordinate of the post-absorption,
pre-first-wave-round state `ψ_0(m)` equals `v_star`:
```
∀ valid m, ∀ x ∈ {0,…,255}:  ψ_0(m)[x] ≠ v_star = 195225786.
```

**Proof.** By the normative absorption rule:
- Rate cells `x ∈ {0,…,63}`: `ψ_0(m)[x] = (ψ_IV[x] + elem[x]) mod p` with
  `elem[x] ∈ [0, 2²⁴)` and `ψ_IV[x] ∈ [54, 374]`. Hence `ψ_0(m)[x] ≤ 374 + 2²⁴ − 1
  = 16777589 < p` (no wraparound) and `≤ 16777589 < 195225786 = v_star`.
- `x = 65` (CAP1): `ψ_0(m)[65] = (ψ_IV[65] + D_TAG) mod p = 1464026030 ≠ v_star`.
- `x ∈ {64, 66}` (CAP0, CAP2): `ψ_IV[x] + (counter)·G`; for the counter values these
  are small fixed residues `≠ v_star`.
- `x ∈ {67,…,255}`: never written by absorption; `ψ_0(m)[x] = ψ_IV[x] ∈ [123, 374]
  ≠ v_star`.

Every coordinate is therefore `≠ v_star`. ∎

**Corollary.** The full-lattice singular constant `c·1` (`c = 357959172`, the
unique state with `F(c·1) = v_star·1`) is **not** valid-message reachable as
`ψ_0(m)`, since `c > 16777589` and `c ∉ {fixed CAP/IV constants}`.

**Solver corroboration.** z3 returns **UNSAT** for "message bytes such that
`ψ_0(m)[x] = v_star`" (rate cell) and for `ψ_0(m)[x] = c`
(`vstar_message_solver.json`, tasks A–C).

---

## Theorem 2 (round-1 single-coordinate reachability witness) — [computer-assisted theorem]

**Statement.** There exists a valid message `m★` (191 bytes) such that the
round-1 wave state has exactly one singular coordinate:
```
ψ_1(m★)[20] = v_star,   and  ψ_1(m★)[x] ≠ v_star for x ≠ 20.
```

**Proof (computer-assisted, replay-verified).** z3 solved the one-round equation
`F(ψ_0)[20] = v_star` with the cell-20 neighbourhood (rate cells 4,19,20,21,36)
encoded as message bytes; the witness was then **replayed through the reference
absorption + one Design A wave round**, confirming `ψ_1(m★)[20] = 195225786`
exactly and a singular-coordinate count of 1
(`vstar_message_solver.json` task D, `vstar_collision_consequence.json`). Because
`ψ_1[20]` depends only on `ψ_0` at cell 20 and its four rate neighbours, the
isolated witness replays exactly under the full protocol. ∎

So **coordinate reachability (question A) is YES**, at round 1, with an explicit
pinned witness.

---

## Theorem 3 (the singular hit is harmless) — [computer-assisted theorem / bounded evidence]

**Statement (consequence of Theorem 2).** The singular event at `m★` does not
erase message information and yields no digest collision in the tested
constructions.

**Argument.**
1. *Erasure of the wave injection only.* At round 0 the wave-injection at the hit
   cell is `j_B(ψ_0[20], v_star) = ψ_0[20]·(1+γ·v_star) = 0` (exact).
2. *But the bytes already entered C.* The normative absorption injects
   `C[20] += G·elem[20]` **before any round**, so the hit cell's message bytes are
   committed to C independently of the zeroed wave-injection. [theorem-level: this
   is exact from the absorption rule.]
3. *Sensitivity restored next round.* At round 1 the hit value `v_star` enters as
   the *earlier* state `u`, giving injection `v_star·(1+γ·ψ_2[20]) ≠ 0` generically.
4. *No collision.* (i) 0/64 messages differing only in the hit cell's bytes
   collide; (ii) the cell-20 cubic `F(u)[20]=v_star` has only **one**
   message-reachable in-window root, so no same-neighbour singular pair (which
   would share the zeroed round-0 injection) exists. [bounded evidence.]

**Classification:** the singular hyperplane is **reachable (one coordinate, round
1) but harmless in tested cases.**

---

## Exhaustive bounded verification — [exhaustive bounded verification]

Over all 1-byte messages, 1024 two-byte messages, all binary-alphabet `{0,255}`
messages of length 3–8, and a structured-message battery (all-zero, all-0xFF,
alternating, repeated, counters, periodic, mirrored, low-Hamming), the wave
trajectory produced **0 incidental `v_star` hits** (closest centered distance 46).
Incidental hits occur at the `≈1/p` rate; the round-1 witness is a *constructed*
state, not a typical message. (Bounded absence is not a global proof.)

---

## Structured / full-lattice / persistent reachability — [unresolved]

- **Structured-subset (question B):** as arbitrary math states, `v_star` lives on
  a half-lattice for each sign-eigenvector family (Part IV cubics), but those
  states are not message-reachable as `ψ_0`. A *message-reachable* multi-coordinate
  singular state was **not found**: simulated annealing reached count 0; z3 for
  2–4 simultaneous round-1 hits returned **UNKNOWN (timeout)**. Unresolved.
- **Full-lattice (question C):** requires controlling the 192 never-injected
  coordinates; impossible at round 0 (Theorem 1), z3 one-round full-constant UNSAT,
  no witness at round 1. Not found.
- **Persistent (question D):** `v_star` is not a wave fixed point
  (`F(v_star) ≠ v_star`); no persistence observed. Not found.

---

## Summary of rigor labels

| Result | Label |
|---|---|
| Round-0 coordinate unreachability (∀ valid m) | **theorem** (+ z3 UNSAT) |
| `c·1` not reachable as `ψ_0` | **theorem** (corollary) |
| Round-1 single-coordinate reachable | **computer-assisted theorem** (replay-verified witness) |
| Singular hit erases wave-injection but not C-commitment | **theorem** (from absorption rule) |
| No collision from the singular hit | **bounded evidence** |
| 0 incidental hits in bounded message sets | **exhaustive bounded verification** |
| Multi-coordinate / full-lattice / persistent reachability | **unresolved** (timeouts; no witness) |
