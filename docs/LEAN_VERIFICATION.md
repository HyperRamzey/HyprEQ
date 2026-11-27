# Lean verification (Tier 1 + Tier 2)

The `lean/` Lake package formalises the exact discrete and finite-dimensional
algebraic core of HyprEQ's DSP code.  It is deliberately not a model of IEEE-754
floating point: every theorem is about `Nat`, `Int`, or an abstract linearly
ordered field, and the floating-point layer stays in Rust where the tests live.

## What is proved

| Tier | Subject | Module |
|---|---|---|
| 1 | Convolver FFT plan sizing, overlap-add coverage | `HyprEQ/Tier1/FftPlan.lean` |
| 1 | WAV PCM byte packing / signed round trips / writer clamps | `HyprEQ/Tier1/WavQuant.lean` |
| 1 | Limiter lookahead, ramp, latency and ring-slot arithmetic | `HyprEQ/Tier1/LimiterIndex.lean` |
| 2 | PCHIP tangents: positivity, Fritsch–Carlson `|d| ≤ 3δ` bound, endpoint estimate | `HyprEQ/Tier2/PchipTangent.lean` |
| 2 | PCHIP cubic: derivative nonnegativity and interval monotonicity | `HyprEQ/Tier2/PchipMonotone.lean` |
| 2 | Limiter gain: `required ∈ (0,1)`, release follower stays in `[0,1]` | `HyprEQ/Tier2/LimiterGain.lean` |
| — | Ordered-field toolbox used by the Tier 2 proofs | `HyprEQ/OrderLemmas.lean` |
| — | Class non-vacuity: `OrderedField` derivable from standard order classes | `HyprEQ/Nonvacuity.lean` |

Tier 3 (continuous analysis and IEEE-754 semantics) is out of scope by design.

## Axiom policy

Every shipped theorem is audited by `AxiomAudit.lean` and must depend on a
subset of `[propext, Quot.sound]`.  The gate refuses, literally:

* `Classical.choice`
* `sorryAx`
* `Lean.ofReduceBool`
* `Lean.trustCompiler`

It additionally elaborates the non-vacuity witness and pins the footprint of
the ℚ-specialised term (see below).

Run it with:

```bash
scripts/lean-verify.sh
```

Expected final line:

```
PASS: 101 theorems audited; axioms subset of [propext, Quot.sound]
```

### Why the Tier 2 proofs use a local `OrderedField` class

Mathlib's ordered-field lemmas are not choice-free in practice.  In Lean
4.36/Mathlib, `#print axioms mul_nonneg` alone reports no axioms, but using it
over `[Field K] [LinearOrder K] [IsStrictOrderedRing K]` pulls
`Classical.choice` through derived-instance definitions (`PosMulMono`,
`IsOrderedAddMonoid`, `GroupWithZero`/`DivisionRing` projections); the same
happens to `mul_le_mul_of_nonneg_right`, `add_le_add_left`,
`lt_or_eq_of_le`, `inv_mul_cancel₀`, `Field.mul_inv_cancel`,
`Field.div_eq_mul_inv`, and to `linarith`/`nlinarith` goals over a generic
field.

`HyprEQ.OrderedField` restates the needed facts as `Prop` class fields
(hypotheses of a variable, not axioms), proved directly from the projections.
Division and inverse were removed from the class because their mere presence in
a field type taints the class; all DSP statements are therefore division-free
and cross-multiplied, e.g. `d * pchipDen = pchipNum` instead of
`d = pchipNum / pchipDen`.  Every theorem on top of the class audits clean.

### Non-vacuity of the class

`OrderedField` is a local `Prop` class, so a contradictory field would make
every Tier 2 theorem vacuous.  It is not contradictory:

* `HyprEQ/Nonvacuity.lean` proves `orderedField_of_classes`: `OrderedField K`
  follows, by direct projection, from `[Field K] [LinearOrder K]
  [ZeroLEOneClass K] [IsOrderedCancelAddMonoid K] [PosMulStrictMono K]
  [MulPosStrictMono K]`.  The audited axioms are `[propext, Quot.sound]` —
  zero `Classical.choice` — and `#synth OrderedField ℚ` resolves to
  `instOrderedFieldOfClasses`, so ℚ (and ℝ) inhabit the class.
* A ℚ-specialised *term* cannot audit zero-choice in Lean 4.36: core `Rat.add`
  and `Rat.mul` call `Rat.normalize`, whose `normalize.reduced` proof uses
  `Int.natAbs_ediv_of_dvd`, and that lemma depends on `Classical.choice`.
  Consequently every proof term that mentions `a + b` or `a * b` on ℚ — even
  `rfl : a + b = a + b` — carries it.  The root `Nonvacuity.lean` witnesses
  `Nonempty (OrderedField ℚ)` and records the footprint
  `[propext, Classical.choice, Quot.sound]`.

The gate audits the clean generic theorem and elaborates the root witness,
pinning both footprints; the ℚ witness is not one of the 101 shipped
theorems.  The `Classical.choice` in the specialised term is ℚ's, not the
class's.

## Theorem → source map

Line numbers refer to the current working tree.

### `Tier1/FftPlan.lean` — `src/fft.rs`, `src/convolve.rs`

| Theorem | Source | Statement |
|---|---|---|
| `bitLength_succ_gt` | `fft.rs:229` (`bit_length`) | `0 < n → n < 1 << bit_length n` |
| `shlBitLength_le_of_lt` | `fft.rs:240` (`shl_bit_length`) | `n < 2^k → 1 << bit_length n ≤ 2^k` |
| `nFft_pow2` | `convolve.rs:202` (`plan`) | transform size is a power of two |
| `nFft_covers_linear` | `convolve.rs:191,202` | `block_len + n_ir - 1 ≤ n_fft` (no circular wrap) |
| `blockLen_ge` | `convolve.rs:191` (`block_bound`) | `16384 ≤ block_len` |
| `blockLen_le` | `convolve.rs:191` | `block_len ≤ 65536` |

### `Tier1/WavQuant.lean` — `src/wav.rs`, `src/dither.rs`

| Theorem | Source | Statement |
|---|---|---|
| `packBytes_length` | `wav.rs:420` (`write_wav_pcm_or_float`) | packed list has exactly `n` bytes |
| `packBytes_mem_range` | `wav.rs:420` | every byte is in `[0,255]` |
| `unpackBytes_nonneg` / `unpackBytes_lt_full` | `wav.rs:229` (`decode`) | fold stays in `[0, 256^n)` |
| `byteFull_eq_two_mul_byteHalf` | `wav.rs:229` | `256^n = 2 * 128 * 256^(n-1)` |
| `toSigned_bounds` | `wav.rs:229` | signed interpretation in `[-128·256^(n-1), 128·256^(n-1))` |
| `unpackBytes_packBytes` | `wav.rs:229,420` | unsigned round trip on `[0,256^n)` |
| `toSigned_unpackBytes_packBytes` | `wav.rs:229,420` | signed round trip |
| `pcm_s16_roundtrip` | `wav.rs:286` | 16-bit round trip on `[-32768,32767]` |
| `pcm_s24_roundtrip` | `wav.rs:304` | 24-bit round trip on `[-8388608,8388607]` |
| `pcm_u8_roundtrip` | `wav.rs:229` (`decode`) | unsigned 8-bit interpretation |
| `clamp16_range` / `clamp24_range` | `wav.rs:474` | writer output stays in range |
| `clamp16_low` / `clamp16_high` | `wav.rs:474` | saturation at both rails |
| `clamp16_idem` / `clamp24_idem` | `wav.rs:474` | clamping is idempotent |

### `Tier1/LimiterIndex.lean` — `src/limiter.rs`

| Theorem | Source | Statement |
|---|---|---|
| `minLookahead_eq`, `minLookahead_le_lookahead` | `limiter.rs:334` | `report_delay + 2 + 1` floor |
| `requested_le_lookahead`, `one_le_lookahead` | `limiter.rs:340` | `lookahead = max(requested, min, 1)` |
| `reportDelay_add_three_le_lookahead` | `limiter.rs:334,340` | lookahead ≥ delay + 3 |
| `ramp_ge_minRampFrames`, `ramp_le_room`, `ramp_le_preferred` | `limiter.rs:377` | ramp clamps |
| `ramp_timeliness`, `ramp_plus_delay_lt_latency` | `limiter.rs:377,413` | ramp + delay + 1 ≤ lookahead |
| `latency_eq` | `limiter.rs:413` | `latency = lookahead + report_delay` |
| `slot_lt`, `slot_add_ring`, `slot_eq_of_lt`, `slot_inj` | `limiter.rs:745` | ring slot `j % ring_len` |

### `Tier2/PchipTangent.lean` — `src/pchip.rs`

| Theorem | Source | Statement |
|---|---|---|
| `pchipW1_pos`, `pchipW2_pos` | `pchip.rs:146-147` | interior weights are positive |
| `pchipW2_le_two_w1` | `pchip.rs:146-147` | `w2 ≤ 2 w1` |
| `pchipDen_pos`, `pchipNum_pos` | `pchip.rs:148` | cross-multiplied denominator/numerator positive |
| `pchip_tangent_pos` | `pchip.rs:148` | `d * den = num → 0 < d` |
| `pchip_tangent_le_three_mul` | `pchip.rs:148` | `d ≤ 3δ₀` |
| `edgeDen_pos`, `edge_le_three_mul` | `pchip.rs:68-78` | endpoint estimate ≤ `3·m0` |

### `Tier2/PchipMonotone.lean` — `src/pchip.rs`

The cubic is normalised to `h(0)=0`, `h(1)=1` with endpoint slopes `a`, `b`;
`pchip.rs:212-219` evaluates the same cubic in Hermite-basis form, which is
algebraically identical after that normalisation.

| Theorem | Statement |
|---|---|
| `hermiteD_bern` | derivative quadratic in Bernstein form |
| `hermiteD_nonneg_bern` | `a+b ≤ 3` branch: `q ≥ 0` |
| `shape_factor_nonpos` | `a²+b²+ab-6a-6b+9 ≤ 0` on the box above the diagonal |
| `hermiteD_nonneg` | Fritsch–Carlson: `0≤a,b≤3 → q(t) ≥ 0` on `[0,1]` |
| `hermiteCubic_diff` | `h(t2)-h(t1) = (t2-t1)·G` |
| `hermiteD_simpson` | `6G = q(t1)+4q(m)+q(t2)` (division-free) |
| `hermiteCubic_monotone` | `0≤a,b≤3 → t1≤t2 → h(t1) ≤ h(t2)` |

### `Tier2/LimiterGain.lean` — `src/limiter.rs`

| Theorem | Source | Statement |
|---|---|---|
| `required_gain_bounds` | `limiter.rs:735` (`required`) | `g·after = ceiling`, `0<ceiling<after → 0<g<1` |
| `kmin_le_left`, `kmin_le_right`, `le_kmin`, `kmin_bounds` | `limiter.rs:847` | order facts for `min` |
| `followerNext_bounds` | `limiter.rs:847` (`advance_gain`) | release follower maps `[0,1]` into `[0,1]` |

## What is not covered

* Floating-point rounding and IEEE-754 behaviour (Tier 3).  The Rust unit tests
  cover numerical values; the Lean proofs cover the exact integer and algebraic
  skeleton underneath them.
* `exp`/`ln` in `blend` and the dB-domain curves (`limiter.rs:581`), which need
  real analysis.
* The Kaiser/Bessel window (`src/window.rs`), for the same reason.
* The full streaming process loop as a state machine; `LimiterIndex` proves the
  arithmetic facts that loop's safety depends on (ramp fits in the lookahead,
  every slot is live), not the loop itself.
* `searchsorted_right` (`pchip.rs:35`): the binary search is the next Tier 1
  candidate, stated over a sorted `Int` list.
