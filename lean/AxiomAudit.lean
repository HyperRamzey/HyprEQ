/-
  Axiom audit for every shipped theorem.

  This file is not part of the `HyprEQ` library; `scripts/lean-verify.sh`
  elaborates it and validates every line of output.  A theorem passes only
  when `#print axioms` reports a subset of `[propext, Quot.sound]`.  In
  particular the audit refuses `Classical.choice` (pulled in by Mathlib's
  derived ordered-field instances), `sorryAx`, `Lean.ofReduceBool` and
  `Lean.trustCompiler`.
-/
import HyprEQ

/-! ### `HyprEQ.OrderLemmas` -/
#print axioms HyprEQ.lt_of_le_not_ge
#print axioms HyprEQ.lt_of_le_of_ne
#print axioms HyprEQ.ne_of_gt
#print axioms HyprEQ.zero_lt_one
#print axioms HyprEQ.add_nonneg
#print axioms HyprEQ.add_le_add
#print axioms HyprEQ.add_lt_add_left
#print axioms HyprEQ.add_pos
#print axioms HyprEQ.neg_nonneg
#print axioms HyprEQ.neg_pos
#print axioms HyprEQ.neg_eq_zero
#print axioms HyprEQ.mul_pos
#print axioms HyprEQ.mul_nonneg
#print axioms HyprEQ.mul_le_mul_of_nonneg_right
#print axioms HyprEQ.mul_le_mul_of_nonneg_left
#print axioms HyprEQ.mul_nonpos_of_nonneg_of_nonpos
#print axioms HyprEQ.mul_nonpos_of_nonpos_of_nonneg
#print axioms HyprEQ.mul_le_mul
#print axioms HyprEQ.sq_nonneg
#print axioms HyprEQ.sq_pos_of_ne_zero
#print axioms HyprEQ.le_of_mul_le_mul_right
#print axioms HyprEQ.lt_of_mul_lt_mul_right
#print axioms HyprEQ.eq_of_mul_eq_mul_right
#print axioms HyprEQ.two_eq
#print axioms HyprEQ.zero_lt_two
#print axioms HyprEQ.three_eq
#print axioms HyprEQ.zero_lt_three
#print axioms HyprEQ.le_add_of_nonneg_right
#print axioms HyprEQ.sub_nonneg
#print axioms HyprEQ.le_of_sub_nonneg
#print axioms HyprEQ.sub_pos
#print axioms HyprEQ.sub_nonpos
#print axioms HyprEQ.sq_nonneg_pow
#print axioms HyprEQ.add_nonpos
#print axioms HyprEQ.four_eq
#print axioms HyprEQ.zero_lt_four
#print axioms HyprEQ.two_lt_three
#print axioms HyprEQ.six_eq
#print axioms HyprEQ.zero_lt_six

/-! ### `HyprEQ.Tier1.FftPlan` -/
#print axioms HyprEQ.bitLength_succ_gt
#print axioms HyprEQ.shlBitLength_le_of_lt
#print axioms HyprEQ.nFft_pow2
#print axioms HyprEQ.nFft_covers_linear
#print axioms HyprEQ.blockLen_ge
#print axioms HyprEQ.blockLen_le

/-! ### `HyprEQ.Tier1.WavQuant` -/
#print axioms HyprEQ.packBytes_length
#print axioms HyprEQ.packBytes_mem_range
#print axioms HyprEQ.unpackBytes_nonneg
#print axioms HyprEQ.unpackBytes_lt_full
#print axioms HyprEQ.byteFull_eq_two_mul_byteHalf
#print axioms HyprEQ.toSigned_bounds
#print axioms HyprEQ.unpackBytes_packBytes
#print axioms HyprEQ.toSigned_unpackBytes_packBytes
#print axioms HyprEQ.pcm_s16_roundtrip
#print axioms HyprEQ.pcm_s24_roundtrip
#print axioms HyprEQ.pcm_u8_roundtrip
#print axioms HyprEQ.clamp16_range
#print axioms HyprEQ.clamp24_range
#print axioms HyprEQ.clamp16_low
#print axioms HyprEQ.clamp16_high
#print axioms HyprEQ.clamp16_idem
#print axioms HyprEQ.clamp24_idem

/-! ### `HyprEQ.Tier1.LimiterIndex` -/
#print axioms HyprEQ.minLookahead_eq
#print axioms HyprEQ.requested_le_lookahead
#print axioms HyprEQ.minLookahead_le_lookahead
#print axioms HyprEQ.one_le_lookahead
#print axioms HyprEQ.reportDelay_add_three_le_lookahead
#print axioms HyprEQ.ramp_ge_minRampFrames
#print axioms HyprEQ.ramp_le_room
#print axioms HyprEQ.ramp_le_preferred
#print axioms HyprEQ.ramp_timeliness
#print axioms HyprEQ.ramp_plus_delay_lt_latency
#print axioms HyprEQ.latency_eq
#print axioms HyprEQ.slot_lt
#print axioms HyprEQ.slot_add_ring
#print axioms HyprEQ.slot_eq_of_lt
#print axioms HyprEQ.slot_inj

/-! ### `HyprEQ.Tier2.PchipTangent` -/
#print axioms HyprEQ.pchipW1_pos
#print axioms HyprEQ.pchipW2_pos
#print axioms HyprEQ.pchipW2_le_two_w1
#print axioms HyprEQ.pchipDen_pos
#print axioms HyprEQ.pchipNum_pos
#print axioms HyprEQ.pchip_tangent_pos
#print axioms HyprEQ.pchip_tangent_le_three_mul
#print axioms HyprEQ.edgeDen_pos
#print axioms HyprEQ.edge_le_three_mul

/-! ### `HyprEQ.Tier2.PchipMonotone` -/
#print axioms HyprEQ.hermiteD_bern
#print axioms HyprEQ.hermiteD_nonneg_bern
#print axioms HyprEQ.shape_factor_nonpos
#print axioms HyprEQ.hermiteD_nonneg
#print axioms HyprEQ.hermiteCubic_diff
#print axioms HyprEQ.hermiteD_simpson
#print axioms HyprEQ.hermiteCubic_monotone

/-! ### `HyprEQ.Tier2.LimiterGain` -/
#print axioms HyprEQ.kmin_le_left
#print axioms HyprEQ.kmin_le_right
#print axioms HyprEQ.le_kmin
#print axioms HyprEQ.kmin_bounds
#print axioms HyprEQ.neg_nonpos
#print axioms HyprEQ.required_gain_bounds
#print axioms HyprEQ.followerNext_bounds
