# HyprEQ Lean 4 formalization

A formalization of the exact discrete and finite-dimensional algebraic core of the
HyprEQ DSP in `../src/`.  It is deliberately **not** a model of IEEE-754: Tier 1
covers `Nat`/`Int` bit and index arithmetic, Tier 2 covers finite-dimensional algebra
over a linearly ordered field.  Tier 3 — continuous analysis, `exp`/`ln`, and
floating-point semantics — is out of scope and stays pinned by the Rust tests.

- Toolchain: Lean 4.36.0 (pinned by `lean-toolchain`), Lake, Mathlib `v4.36.0`.
- First checkout: `lake exe cache get` (downloads the Mathlib oleans), then
  `lake build`.
- Purity gate: `./check_axioms.sh` (thin wrapper around
  `../scripts/lean-verify.sh`).  It builds the library, scans `HyprEQ/` for
  `sorry`/`admit`, audits **every** theorem with `#print axioms`, and fails unless
  each dependency is a subset of `[propext, Quot.sound]`.  It also verifies that
  `AxiomAudit.lean` covers exactly the theorems declared in the sources.
- Current result: **101 theorems**, axioms limited to `[propext, Quot.sound]`
  (several depend on nothing at all).  Literally zero `Classical.choice`,
  `sorryAx`, `Lean.ofReduceBool` and `Lean.trustCompiler`.  The gate also
  elaborates the ℚ non-vacuity witness and pins both of its measured
  footprints (see below).

The Rust tree is untouched; everything here lives under `lean/`.

## Why there is a local `OrderedField` class

Mathlib's ordered-field lemmas are not choice-free in practice.  In Lean 4.36 /
Mathlib, `#print axioms mul_nonneg` alone reports no axioms, but *using* it over
`[Field K] [LinearOrder K] [IsStrictOrderedRing K]` pulls `Classical.choice`
through derived-instance definitions (`PosMulMono`, `IsOrderedAddMonoid`,
`GroupWithZero`/`DivisionRing` projections).  The same happens to
`mul_le_mul_of_nonneg_right`, `add_le_add_left`, `lt_or_eq_of_le`,
`inv_mul_cancel₀`, `Field.mul_inv_cancel`, `Field.div_eq_mul_inv`, and to
`linarith`/`nlinarith` goals over a generic field.

`HyprEQ.OrderedField` therefore restates the needed facts as `Prop` class fields —
hypotheses of a variable, not axioms — and every lemma in `OrderLemmas.lean` is
proved directly from the projections.  Inverse and division were removed from the
class because their mere presence in a field type taints the class; all DSP
statements are stated division-free and cross-multiplied, e.g.
`d * pchipDen = pchipNum` instead of `d = pchipNum / pchipDen`.  The `ring`,
`ring_nf`, `linear_combination`, `omega` and `decide` tactics audit clean here.

## Non-vacuity of the class

A `Prop`-valued class with contradictory fields would make the Tier 2 theorems
vacuously true.  The class is inhabited, and the derivation is audited:

- `HyprEQ/Nonvacuity.lean` derives `OrderedField K` from the standard Mathlib
  order classes (`ZeroLEOneClass`, `IsOrderedCancelAddMonoid`,
  `PosMulStrictMono`, `MulPosStrictMono`) by direct projection.
  `orderedField_of_classes` audits to `[propext, Quot.sound]` — zero
  `Classical.choice` — and the instance fires at ℚ (`#synth OrderedField ℚ`
  prints `instOrderedFieldOfClasses`).
- A ℚ-specialised *term* inevitably carries `Classical.choice`: core `Rat.add`
  and `Rat.mul` call `Rat.normalize`, whose `normalize.reduced` proof uses
  `Int.natAbs_ediv_of_dvd`.  `lean/Nonvacuity.lean` measures this
  (`orderedField_rat_nonempty` audits to
  `[propext, Classical.choice, Quot.sound]`) and documents it.  That witness
  sits outside the audited library; the gate pins both footprints.  Zero
  choice for a ℚ-touching proof is impossible without replacing the carrier,
  but the class derivation itself is clean.

## Layout

| File | Rust source | Content |
|---|---|---|
| `HyprEQ/Basic.lean` | — | shared imports and tactic setup |
| `HyprEQ/OrderLemmas.lean` | — | the local `OrderedField` class and the 39-lemma ordered-field toolbox |
| `HyprEQ/Nonvacuity.lean` | — | standard-class derivation of `OrderedField`; audited non-vacuity theorem |
| `Nonvacuity.lean` | — | ℚ witness and the measured `Classical.choice` footprint of core `Rat` |
| `HyprEQ/Tier1/FftPlan.lean` | `fft.rs:229,240`, `convolve.rs:191,202` | `bit_length`/`shl_bit_length`, the overlap-add plan and no-circular-wrap bound |
| `HyprEQ/Tier1/WavQuant.lean` | `wav.rs:229,286,304,420,474`, `dither.rs` | little-endian packing, signed decoding, PCM round trips, writer clamps |
| `HyprEQ/Tier1/LimiterIndex.lean` | `limiter.rs:334,340,377,413,745` | lookahead/ramp/latency arithmetic and ring-slot indexing |
| `HyprEQ/Tier2/PchipTangent.lean` | `pchip.rs:68-78,146-148` | interior tangent and endpoint-estimate bounds |
| `HyprEQ/Tier2/PchipMonotone.lean` | `pchip.rs:212-219` | Hermite cubic monotonicity under Fritsch–Carlson |
| `HyprEQ/Tier2/LimiterGain.lean` | `limiter.rs:735,847` | required-gain bounds and the release follower |
| `HyprEQ.lean` | — | root module importing the whole proof surface |
| `AxiomAudit.lean` | — | `#print axioms` for all 101 theorems; consumed by the gate |

Detailed theorem-by-theorem source line mappings, and the statement of what is not
covered, live in [`../docs/LEAN_VERIFICATION.md`](../docs/LEAN_VERIFICATION.md).

## Theorem inventory

### `HyprEQ.OrderLemmas` — ordered-field toolbox (39)

Bridges: `lt_of_le_not_ge`, `lt_of_le_of_ne`, `ne_of_gt`, `zero_lt_one`.
Addition and negation: `add_nonneg`, `add_le_add`, `add_lt_add_left`, `add_pos`,
`add_nonpos`, `neg_nonneg`, `neg_pos`, `neg_eq_zero`, `le_add_of_nonneg_right`.
Multiplication: `mul_pos`, `mul_nonneg`, `mul_le_mul_of_nonneg_right`,
`mul_le_mul_of_nonneg_left`, `mul_nonpos_of_nonneg_of_nonpos`,
`mul_nonpos_of_nonpos_of_nonneg`, `mul_le_mul`, `sq_nonneg`,
`sq_pos_of_ne_zero`, `sq_nonneg_pow`.  Cancellation (division-free):
`le_of_mul_le_mul_right`, `lt_of_mul_lt_mul_right`, `eq_of_mul_eq_mul_right`.
Numerals and subtraction: `two_eq`, `zero_lt_two`, `three_eq`, `zero_lt_three`,
`four_eq`, `zero_lt_four`, `two_lt_three`, `six_eq`, `zero_lt_six`,
`sub_nonneg`, `le_of_sub_nonneg`, `sub_pos`, `sub_nonpos`.

### `HyprEQ.Nonvacuity` (1)

`orderedField_of_classes` — `OrderedField K` follows choice-free from the
standard order classes, so the six fields are mutually consistent (ℚ and ℝ
inhabit them).

### `HyprEQ.Tier1.FftPlan` (6)

`bitLength_succ_gt`, `shlBitLength_le_of_lt`, `nFft_pow2`,
`nFft_covers_linear` (the transport size covers the linear convolution, so no
circular wrap), `blockLen_ge`, `blockLen_le`.

### `HyprEQ.Tier1.WavQuant` (17)

`packBytes_length`, `packBytes_mem_range`, `unpackBytes_nonneg`,
`unpackBytes_lt_full`, `byteFull_eq_two_mul_byteHalf`, `toSigned_bounds`,
`unpackBytes_packBytes`, `toSigned_unpackBytes_packBytes`, `pcm_s16_roundtrip`,
`pcm_s24_roundtrip`, `pcm_u8_roundtrip`, `clamp16_range`, `clamp24_range`,
`clamp16_low`, `clamp16_high`, `clamp16_idem`, `clamp24_idem`.

### `HyprEQ.Tier1.LimiterIndex` (15)

`minLookahead_eq`, `requested_le_lookahead`, `minLookahead_le_lookahead`,
`one_le_lookahead`, `reportDelay_add_three_le_lookahead`,
`ramp_ge_minRampFrames`, `ramp_le_room`, `ramp_le_preferred`, `ramp_timeliness`,
`ramp_plus_delay_lt_latency`, `latency_eq`, `slot_lt`, `slot_add_ring`,
`slot_eq_of_lt`, `slot_inj`.

### `HyprEQ.Tier2.PchipTangent` (9)

`pchipW1_pos`, `pchipW2_pos`, `pchipW2_le_two_w1`, `pchipDen_pos`,
`pchipNum_pos`, `pchip_tangent_pos`, `pchip_tangent_le_three_mul`,
`edgeDen_pos`, `edge_le_three_mul`.

### `HyprEQ.Tier2.PchipMonotone` (7)

`hermiteD_bern`, `hermiteD_nonneg_bern`, `shape_factor_nonpos`,
`hermiteD_nonneg`, `hermiteCubic_diff`, `hermiteD_simpson`,
`hermiteCubic_monotone`.

### `HyprEQ.Tier2.LimiterGain` (7)

`kmin_le_left`, `kmin_le_right`, `le_kmin`, `kmin_bounds`, `neg_nonpos`,
`required_gain_bounds` (a required gain is in `(0,1)` when the post-stage peak is
above the ceiling), `followerNext_bounds` (the release follower maps `[0,1]` into
itself).

## What is not proved

- IEEE-754 rounding, FMA behaviour, and the f64 pipeline (Tier 3).
- `exp`/`ln` in the dB-domain `blend` and curves (`limiter.rs:581`).
- The Kaiser/Bessel window (`window.rs`), continuous analysis.
- The streaming process loop as a state machine: `LimiterIndex` proves the
  arithmetic its safety depends on, not the loop itself.
- `searchsorted_right` (`pchip.rs:35`) — the next Tier 1 candidate, to be stated
  over a sorted `Int` list.
