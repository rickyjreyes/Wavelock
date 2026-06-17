# WaveLock-PDE-256-v0 — Formal Specification

**Status:** experimental, versioned, provisional parameters. This document
specifies a *candidate* hash-free, PDE-native compression function. It makes
**no** security claim; see `docs/PDE_HASH_THREAT_MODEL.md` and (after testing)
`docs/PDE_HASH_RESULTS.md`.

**Branch:** `research/hash-free-pde-core`. **Date:** 2026-06-17.

This specification covers **Design A**: a discrete nonlinear *reaction–diffusion
PDE* over a finite field, used in a sponge construction. Design B (the exact
fixed-point translation of WaveLock's original curvature-feedback dynamics) is
specified separately once Design A is implemented and audited; both are tested
independently and neither is presumed to succeed or fail in advance.

The defining requirement, restated: the map must be

```
H_PDE : {0,1}* → {0,1}^256 ,    H_PDE(m) = Q( Φ_P^T ( A(m) ) )
```

with **no** SHA, SHAKE, BLAKE, MD5, RIPEMD, HMAC, HKDF, AES, ChaCha, SipHash,
Argon2/scrypt/PBKDF, or any library digest/XOF in `A`, `Φ`, the round
constants, or `Q`.

---

## 1. The governing dynamical system (Design A)

WaveLock-PDE-256-v0 is a **finite-field polynomial dynamical system derived
from the algebraic form of the Allen–Cahn reaction–diffusion equation**. The
*source* continuous PDE is the Allen–Cahn / Nagumo equation:

```
∂ψ/∂t = D ∇²ψ + a·ψ·(b − ψ²)
        └─diffusion─┘ └──cubic reaction──┘
```

- `D ∇²ψ` is the diffusion (Laplacian) term.
- `a·ψ·(b − ψ²)` is the bistable cubic reaction term (the Allen–Cahn nonlinearity).

**Scope and honesty of the connection.** The finite-field discretization below
preserves the *algebraic structure* of this PDE — a discrete Laplacian plus a
cubic reaction — exactly, so the construction is genuinely PDE-derived and is
not an arbitrary cellular automaton or generic mixing function. **However:** the
system operates over 𝔽_p, not over real-valued Allen–Cahn space and time. The
words "diffusion", "reaction", and "bistability" describe the *source* structure
only. Standard Allen–Cahn analytical results — energy decay, smoothing,
stability, convergence to ±√b — do **not** automatically transfer to arithmetic
over 𝔽_p and must not be treated as proven finite-field properties. In
particular, the T-round map is **not** known to be bijective; **injectivity,
preimage multiplicity, state collapse, and short cycles are unresolved
properties under test** (Phase 8).

### 1.1 Finite-field discretization (exact, bit-reproducible)

All arithmetic is in the prime field **𝔽_p with p = 2³¹ − 1** (a Mersenne
prime). Field elements are integers in `[0, p)`. Forward-Euler discretization
on an `N×N` toroidal lattice:

```
L(ψ)[i,j] = ( ψ[i+1,j] + ψ[i−1,j] + ψ[i,j+1] + ψ[i,j−1] + (p−4)·ψ[i,j] ) mod p      (Laplacian)
R(ψ)[i,j] = ( a · ψ[i,j] · ( b − ψ[i,j]² mod p ) ) mod p                            (cubic reaction)
ψ'[i,j]   = ( ψ[i,j] + D·L(ψ)[i,j] + R(ψ)[i,j] ) mod p                              (one round Φ)
```

Indices wrap modulo `N` (periodic / toroidal boundary). `−4 ≡ p−4 (mod p)` and
`b − ψ²` is computed as `(b + (p − (ψ² mod p))) mod p` to stay non-negative.
One application of `Φ` is **one round of the state transformation**; the
`T`-round transformation `evolve_T` (written `Φ^T` below) applies it `T` times.
`evolve_T` is **not** asserted to be a permutation; "transform" / "evolve" are
used throughout in place of "permute".

**Why p = 2³¹ − 1.** A product of two elements `< 2³¹` is `< 2⁶²`, which fits
in a signed/unsigned 64-bit integer with room to spare. Reduction mod a Mersenne
prime is exact via `x = (x & p) + (x >> 31); if x ≥ p: x −= p`. This makes the
entire primitive computable with `int64` in NumPy **and** with Python ints,
byte-for-byte identically, with no 128-bit intermediates and no floating point.
Cubing `ψ³` is done as two reductions: `t = ψ·ψ mod p`, then `t·ψ mod p`.

---

## 2. State

| Symbol | Meaning | v0 default |
|---|---|---|
| `N` | lattice side | **16** (256 cells) |
| `p` | field modulus | **2³¹ − 1** |
| `T` | rounds per state-transformation call | **32** (provisional; tested in Phase 8) |
| rate cells | indices absorbed into / squeezed from | **first 64 cells (row-major)** |
| capacity cells | cells not directly written by absorb | remaining **192 cells** |

State `S ∈ 𝔽_p^{N×N}`, stored row-major (C order). The construction is
**sponge-*like*** (absorb / evolve / squeeze), but it is **not** a proven
cryptographic sponge: `rate` and `capacity` are state-region *names only*. The
192 capacity cells do **not** automatically provide ~(192·31) ≈ 5952 bits of
generic sponge security, because that heuristic assumes a bijective
permutation and the T-round transformation here is not known to be bijective.
Whether the capacity does anything useful, and whether distinct messages
collapse into shared states, is exactly what Phase 8 measures.

`PDEState` (implementation type) wraps the `N×N` `int64` array plus the fixed
parameters `(N, p, T, D, a, b, rate)`.

### 2.1 Initial value (IV) — no hash

The IV is a fully specified public constant, generated by **direct base-256
injection** of the domain string, never by a hash:

```
tag = b"WaveLock-PDE-256-v0"                       # 19 bytes
IV[i,j] = ( 1 + i*N + j + tag[(i*N + j) mod len(tag)] ) mod p
```

Every IV cell is a small, explicitly computable constant. `tag[k]` is the
unsigned byte value (0–255) at position `k` of the ASCII bytes
`b"WaveLock-PDE-256-v0"`. (Test vectors in §9 pin `IV[0,0]`, `IV[0,1]`,
`IV[N-1,N-1]`.)

### 2.2 Cell regions and named cells (normative)

The flat (row-major) index of lattice cell `(i,j)` is `idx = i·N + j`, with
`i,j ∈ {0,…,N−1}` and `N = 16`, so flat indices run `0…255`. The regions are
fixed by flat index:

| Name | Flat indices | Purpose |
|---|---|---|
| **rate** `R` | `0 … 63` | message absorption (§3.4) |
| **capacity** `C` | `64 … 255` | not written by message absorption |
| `cap0` | `64` | block-counter low digit `q0` (§3.5) |
| `cap1` | `65` | finalization domain injection (§3.5) |
| `cap2` | `66` | block-counter high digit `q1` (§3.5) |

These are the only named cells, all in the capacity region. The absorption
write of §3.4 touches only rate cells `0…63`, so the counter (`cap0`/`cap2`)
and finalization (`cap1`) injections never collide with a message write.
(`rate`/`capacity` are region names only — see §2; no generic sponge security
is implied.)

---

## 3. Message encoding, padding, and absorption `A`

### 3.1 Accepted input and size

Input is an arbitrary byte string `m ∈ {0,1}*` (bit strings are handled at byte
granularity; sub-byte inputs are out of scope for v0). The 64-bit length field
(below) supports inputs up to `MAX_INPUT_BITS = 2⁶⁴ − 1` bits; **messages whose
bit length exceeds this bound MUST be rejected** (the reference and optimized
implementations raise on oversize input).

### 3.2 Byte → field-element packing (injective)

Message bytes are packed **3 bytes per field element** (24 bits `< 31` bits, so
no modular wraparound and the packing is injective):

```
elem(b0,b1,b2) = b0 + 256·b1 + 65536·b2        ∈ [0, 2²⁴)
```

The rate is 64 cells, so one **block = 64 field elements = 192 message bytes**.

### 3.3 Padding and length encoding (injective; kills zero-ambiguity)

Let `Lbits` = bit length of `m`. Padding rule (a `10*` rule plus an explicit
length block, so distinct messages never share a padded image):

1. Append a single `0x01` byte to `m`.
2. Append `0x00` bytes until the length is a multiple of 192 bytes (one block).
3. Append one **final dedicated length block**: a 192-byte block whose first 8
   bytes are `Lbits` big-endian and whose remaining 184 bytes are `0x00`. This
   final block is packed and absorbed like any other, and additionally triggers
   the domain-separation injection of §3.5.

Because (a) the original bit length is bound in a dedicated trailing block, and
(b) the `0x01` marker is always present, leading and trailing zero bytes of `m`
produce distinct padded streams, and the empty message has a valid, defined
padded image (one `0x01` byte → pad to a block → length block). Empty-input
output is therefore well defined.

### 3.4 Block construction and absorption schedule (sponge-like)

After padding (§3.3), the padded byte stream has length `192·B` for some
`B ≥ 2`. It is cut into `B` consecutive **byte-blocks** of 192 bytes each, in
order. Block `k` (`k = 0…B−1`) is converted to 64 field elements by §3.2
packing: element `c` of block `k` (`c = 0…63`) is
`elem(P[192k+3c], P[192k+3c+1], P[192k+3c+2])`, where `P` is the padded stream
and `P[·]` is an unsigned byte. The **last block** (`k = B−1`) is exactly the
dedicated length block of §3.3 step 3.

```
S ← IV                                          # §2.1
for k in 0 … B-1:
    block ← pack_block(P, k)                    # 64 field elements, §3.2
    for c in 0 … 63:                            # add into the 64 rate cells 0..63
        S[c] ← ( S[c] + block[c] ) mod p
    inject_counter(S, k)                        # §3.5 (mutates S[cap0], S[cap2])
    if k == B-1:                                # final/length block only
        inject_finalize(S)                      # §3.5 (mutates S[cap1])
    S ← Φ^T(S)                                  # T-round transform, §1.1 + §5
return S
```

Every message bit influences `S`: each bit sits in a distinct 24-bit field
element added into a rate cell, then `Φ` spreads it across the lattice over
`T` rounds before the next block. Absorption is **additive into existing
cells**, not a copy-and-truncate of the first bytes — all `B` blocks are
processed, and the final `Φ^T` runs after the length block so the length and
domain constant fully propagate before squeezing.

### 3.5 Counter and finalization injection (order/repeat/domain)

The block counter uses an **injective two-digit base-`p` encoding** of the
0-based block index `k`, so it does not repeat after `p` blocks (the
single-digit `(k+1) mod p` would). Let `q = k + 1`, `q0 = q mod p`,
`q1 = ⌊q / p⌋`. All injections are exact modular additions into named capacity
cells (§2.2):

```
inject_counter(S, k):   S[cap0] ← ( S[cap0] + q0·g ) mod p          # cap0 = 64,  g = 7
                        S[cap2] ← ( S[cap2] + q1·g ) mod p          # cap2 = 66,  g = 7
inject_finalize(S):     S[cap1] ← ( S[cap1] + d ) mod p             # cap1 = 65,  d = D_TAG
```

**Why two digits, and coverage proof.** The maximum block index is bounded by
the number of 192-byte padded blocks of the largest permitted input. With
`MAX_INPUT_BITS = 2⁶⁴ − 1`, the message is `< 2⁶⁴ / 8 = 2⁶¹` bytes, padded to
`< 2⁶¹/192 + 2 < 2⁵⁴` blocks, so `q = k + 1 < 2⁵⁴`. Since
`p² = (2³¹ − 1)² > 2⁶¹ ≫ 2⁵⁴`, we have `q < p²`, hence `q1 = ⌊q/p⌋ < p` and the
pair `(q0, q1)` is the unique base-`p` representation of `q` — it **never
aliases** within the declared range. The same generator `g = 7` is used in both
cells; the two cells live at distinct positions (`cap0`, `cap2`), so equal
`q0·g` and `q1·g` contributions land in different state coordinates and do not
collapse. `g` is shared because no asymmetry between the digits is required for
injectivity (the positions already separate them).

For ordinary messages `q < p`, so `q1 = 0`, `inject_counter` reduces to the
original single-digit injection into `cap0` only, and `cap2` is unchanged — all
pre-existing test vectors are preserved.

- `inject_counter` runs once per block, with the 0-based block index `k`, so an
  identical byte-block at a different position injects a different value into
  `cap0`/`cap2`; a repeated block therefore cannot additively cancel and block
  order is significant.
- `inject_finalize` runs **only** on the last block (`k = B−1`, the length
  block), adding the fixed domain constant `D_TAG` (§7) into `cap1`. This is the
  sole domain-separation mechanism and uses no conventional hash.
- `q0·g`, `q1·g`, and `d` are reduced mod `p`.

---

## 4. Squeeze `Q` (PDE-native, fixed 256-bit output)

WaveLock-PDE-256-v0 is **fixed-output**: `H_PDE : {0,1}* → {0,1}^256` always
produces exactly 32 bytes. It is **not** an XOF. The squeeze reads bits
**directly from the evolved state** by canonical cell-pair comparison ("sign of
a field difference"), interleaving further `T`-round transforms so the squeeze
is itself part of the dynamics. The bit budget below is fixed at 256 for the
public digest; an internal `output_bits` knob exists only for audit experiments
(e.g. truncated-collision tests) and is not part of the public API.

```
out_bits = []                                        # list of 0/1 ints, append order = bit order
while len(out_bits) < output_bits:                   # output_bits = 256 (fixed for the digest)
    for t in 0 … 63:                                 # 64 bits per squeeze round
        a_cell, b_cell = SQUEEZE_PAIRS[t]            # fixed disjoint cell-index pair
        out_bits.append( 1 if S[a_cell] > S[b_cell] else 0 )   # strict >; tie ⇒ 0
    if len(out_bits) < output_bits:
        S ← Φ^T(S)                                   # re-evolve between squeeze rounds
return pack_msb_first(out_bits[:output_bits])
```

**`SQUEEZE_PAIRS` (normative).** A fixed list of 64 disjoint index pairs:

```
SQUEEZE_PAIRS[t] = (t, t + 128)        for t = 0 … 63
```

i.e. cell `t` (rate region, flat indices `0…63`) is compared against cell
`t+128` (capacity region, flat indices `128…191`). All 128 referenced cells are
distinct, so each squeeze round reads 64 disjoint comparisons → 64 bits. The
comparison is the integer order of the two residues in `[0, p)`; a tie
(`S[a]==S[b]`) yields `0`.

**Bit count and rounds.** With 64 bits per squeeze round and the fixed
`output_bits = 256`, exactly **4** squeeze rounds run, with **3** intermediate
`Φ^T` re-evolutions (the loop re-evolves only when more bits are still needed,
so no `Φ` runs after the final round's bits are produced). The audit-only
`output_bits` knob must be a positive multiple of 64.

**`pack_msb_first` (normative).** The `output_bits` bits are packed into
`output_bits/8` bytes, MSB-first: `out_bits[0]` is bit 7 (the most significant
bit) of byte 0, `out_bits[7]` is bit 0 of byte 0, `out_bits[8]` is bit 7 of
byte 1, and so on:

```
output_bytes[i] = Σ_{r=0}^{7}  out_bits[8·i + r] · 2^(7−r)
```

For `output_bits = 256` this yields the 32-byte digest.

**Why comparison, not parity/truncation.** Comparing two field elements in
`[0, p)` yields a near-balanced bit when values are well-mixed and avoids the
low-bit bias that raw parity of a non-uniform field element can introduce.
v0 deliberately does **not** serialize-and-truncate the state; bit balance,
bias, correlation, and cell-selection sensitivity of this squeeze are tested in
Phase 5 / `pde_audit/`.

---

## 5. Modular arithmetic semantics and exact update order (normative)

**Modular semantics.** Every value held in any cell, and every intermediate
named below, is the unique mathematical residue in `[0, p)`. "`x mod p`" means
that residue. Subtraction is computed as `(x + (p − (y mod p))) mod p` so no
intermediate is ever negative. Multiplication of two residues is `< 2⁶²` and is
reduced immediately; cubing is two reduce-after-multiply steps
(`t ← ψ·ψ mod p`, `R ← a·(ψ·((b + (p − t)) mod p) mod p) mod p`). An
implementation MAY use Mersenne folding (`x = (x & p) + (x >> 31)`, repeated
until `< 2³¹`, then one conditional subtract) **iff** it yields this exact
residue; the residue, not the fold, is normative.

**Per-cell round (normative form).** For every cell `(i,j)`, using only
pre-update values `ψ`:

```
t       = ψ[i,j]·ψ[i,j]                     mod p
react   = a · ( ψ[i,j] · ((b + (p − t)) mod p) mod p )   mod p
lap     = ( ψ[(i+1)%N, j] + ψ[(i−1)%N, j]
          + ψ[i, (j+1)%N] + ψ[i, (j−1)%N]
          + (p − 4)·ψ[i,j] )                mod p
ψ'[i,j] = ( ψ[i,j] + (D·lap mod p) + react ) mod p
```

**Update order.** The order is fixed and MUST be followed by every
implementation:

1. For all cells, compute `react` and `lap` from the **pre-update** `ψ` only
   (Jacobi, not Gauss–Seidel — no cell sees another cell's new value).
2. Form `ψ'` for all cells simultaneously, then replace `ψ ← ψ'`.
3. One execution of steps 1–2 is one round; `Φ^T` (`evolve_T`) applies it `T` times.
4. Within a `T`-round transform call, rate cells are written by absorption (§3.4)
   *before* the first round; counter/finalization injection (§3.5) happens
   between the absorption write and the first round.

Neighbor indexing: `(i±1) mod N`, `(j±1) mod N`. Row-major flat index of cell
`(i,j)` is `i·N + j`.

---

## 6. Numerical / representation contract (Phase 3)

| Property | Specification |
|---|---|
| Field | 𝔽_p, `p = 2³¹ − 1` |
| Word width | 64-bit integers for all intermediates (products `< 2⁶²`) |
| Signedness | non-negative residues in `[0, p)`; reductions keep values non-negative |
| Overflow | impossible by construction (products `< 2⁶² < 2⁶³`); reduce after every multiply |
| Rounding | none (exact integer arithmetic) |
| Division | none in the canonical primitive |
| Shift | logical right shift in Mersenne reduction `x = (x & p) + (x >> 31)` |
| Endianness | length field big-endian; output bits packed MSB-first into bytes |
| Update ordering | Jacobi, fixed order per §5 |
| Boundary indexing | toroidal, `mod N` |

This contract guarantees byte-for-byte identical output on CPython, across CPU
architectures, between the pure-Python reference and the NumPy implementation,
and between optimized and unoptimized builds. Portability is **claimed only
after** the Phase 7 parity tests pass (empty message, all 256 one-byte messages,
padding boundaries, ≥10,000 random messages, with intermediate-round state
comparison).

---

## 7. Round constants (no hash)

All constants are fixed small integers chosen for the field, **not** derived
from any digest:

```
p     = 2147483647   # 2^31 − 1, the field modulus
N     = 16           # lattice side (256 cells)
D     = 5            # diffusion coefficient
a     = 3            # reaction gain
b     = 1431655765   # fixed mid-field bistable offset, in [0, p)
g     = 7            # counter generator (inject_counter)
D_TAG = 1464619076   # 0x574C5044 = ASCII "WLPD"; already < p, no reduction needed
T     = 32           # rounds per T-round transform Φ^T = evolve_T (provisional)
```

`D_TAG` is the constant `d` referenced in §3.5. `D, a, b, g, D_TAG, T` are
provisional v0 values. The audit (`pde_audit/`) will
sweep them and record every regime tried, including failures (Phase 8/10). Any
change to these constants bumps the version (`v0.1`, …) and regenerates test
vectors.

---

## 8. Multi-block compression and domain separation

Multi-block messages are compressed by the sponge-like loop of §3.4 (absorb-evolve
per block). Domain separation between different uses is provided **without a
hash** by (a) the IV tag injection (§2.1), (b) the per-block counter (§3.5), and
(c) the trailing length block + `"WLPD"` constant (§3.5). No keyed mode is
defined in v0.

---

## 9. Deterministic test vectors

The reference implementation pins exact values for all of the following into
`pde_audit/vectors.json` (and §13 of this document is updated with the same
values once `wavelock/pde_hash/reference.py` lands). Every vector is a fixed
function of this spec alone and MUST match byte-for-byte between the pure-Python
reference and the NumPy implementation.

**State-snapshot encoding.** A lattice-state snapshot is recorded as the 256
flat-ordered residues (`idx = i·N + j`, `idx = 0…255`) packed big-endian as 256
× 4-byte unsigned integers (1024 bytes), then hex-encoded **for the vector file
only**. (This hex is a faithful serialization of the integer state for test
bookkeeping; it is *not* part of the primitive and is never fed back into it.)

**Final-output vectors** (`H_PDE(m)` → 32 bytes, hex):
- `H_PDE(b"")` — empty message.
- `H_PDE(b"\x00")`, `H_PDE(b"\x01")`, `H_PDE(b"\xff")` — single bytes.
- `H_PDE(b"abc")`, `H_PDE(b"WaveLock")` — structured inputs.
- `H_PDE(192·b"\x00")` (exactly one byte-block of message) and
  `H_PDE(193·b"\x00")` (one byte into a second block) — padding boundary.

**Intermediate-state vectors** (for `m = b"abc"`, fully reproducible):
- `IV[0,0]`, `IV[0,1]`, `IV[15,15]` — scalar IV cells.
- `S_iv` — snapshot of `S` immediately after `S ← IV` (before any absorption).
- `S_absorb0` — snapshot after block 0's rate write + `inject_counter(·,0)`,
  **before** the first `Φ` round.
- `S_perm0` — snapshot after the first `Φ^T` = `evolve_T` (post block 0). (The
  key name `S_perm0` is a frozen identifier and does not imply the map is a
  permutation.)
- `S_final` — snapshot of `S` after the full absorption loop, **before** squeeze.
- `S_squeeze1` — snapshot after the first intermediate `Φ^T` inside squeeze.

These intermediate snapshots let Phase 7 compare the two implementations at
round granularity (not only final output), catching shared-bug masking.

---

## 10. Complexity

- **Time:** `O(blocks · T · N²)` field operations to absorb, plus
  `O(4 · T · N²)` to squeeze. For `N=16, T=32`: ≈ `32·256 = 8192` cell-updates
  per block transform, each a constant number of `int64` mul/add/reduce ops.
- **Memory:** `O(N²)` field elements = 256 `int64` (the optimized path may hold
  a few `N×N` temporaries). Constant in message length (streaming, sponge-like).

---

## 11. Required API (Phase 6)

```python
def pde_hash(message: bytes) -> bytes: ...          # fixed 256-bit digest (NOT an XOF)

def absorb(message: bytes) -> PDEState: ...
def evolve(state: PDEState, rounds: int) -> PDEState: ...
def evolve_T(state: PDEState) -> PDEState: ...      # the T-round transform (not a permutation)
def squeeze(state: PDEState, output_bits: int = 256) -> bytes: ...   # output_bits is audit-only
```

Package layout: `wavelock/pde_hash/{__init__,spec,absorb,state,evolve,squeeze,
reference,optimized,cli}.py`. The package MUST NOT import `hashlib`,
`cryptography`, `blake3`, PyCryptodome, or any conventional cryptographic
package; a forbidden-import test (`pde_audit/`) enforces this.

---

## 12. Explicit non-claims (Phase 10)

WaveLock-PDE-256-v0 is an **experimental nonlinear PDE digest candidate** and a
**candidate one-way transformation** with **no formal security proof** and **no
claim of production suitability**. Any positive statement will be phrased as
*empirical resistance under the attacks tested* and will distinguish observed
attack cost from a lower bound and from a proof of one-wayness.

---

## 13. Pinned test vectors (generated)

The exact values below are emitted by `wavelock/pde_hash/reference.py`, verified
identical from `wavelock/pde_hash/optimized.py`, and mirrored in
`pde_audit/vectors.json`. They are filled in by the implementation commit and
are normative for v0; any spec/constant change that alters them is a version
bump.

```
IV cells:      IV[0,0]=88   IV[0,1]=99   IV[15,15]=301

Final digests (32 bytes, hex):
  H_PDE("")                = d12c29be1429775e6dcc9ff3e29d9bca96865c0179a99b9bcee58581bf118820
  H_PDE(0x00)              = 170c1a577110c752fd581d3e1dc025397bf69d3a545f6d7f11a8626776511f1d
  H_PDE(0x01)              = 372d58536ab6d0ee1c032fcef516114ae735acce0dc9fa992aeeb6febb758d3b
  H_PDE(0xff)              = d3ed7292fc678f5fbd28099e3125166352d9d4aa04f7b655ffaaa536bbed2db3
  H_PDE("abc")             = e6231beb61a76e304a5292473a955a970b74b25f55027ca6f0cc34a1cd21985d
  H_PDE("WaveLock")        = 5109e4c0d3effe338c4b1b35555aac8db35f2754753afea961cd768a04937cb2
  H_PDE(0x00 * 192)        = 0479a893bc5a2be6c7e0afcae70f57f48f2fa75cde6a51fdef612b83b27b9a51
  H_PDE(0x00 * 193)        = 718b717932b9e4b4d320198001afd4907c6abeaedc50a9b24e32623bfeb0fba3

Intermediate snapshots for m="abc" (256 BE-uint32 = 1024 bytes; first 16 bytes shown):
  S_iv[:16]       = 00000058 00000063 00000079 00000069
  S_absorb0[:16]  = 006362b9 00000064 00000079 00000069
  S_perm0[:16]    = 594910ca 2154623a 6a67148e 2b42d20c
  S_final[:16]    = 7846bc94 6ce90edf 4d73aa4b 25b5ac8c
  S_squeeze1[:16] = 0d6becc3 7c5d2227 1a3109df 2f5c9023
```

Full 1024-byte intermediate snapshots and all final vectors are pinned in
`pde_audit/vectors.json`. Both the pure-Python reference and the NumPy
implementation reproduce every value above byte-for-byte.
