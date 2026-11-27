/-
  Tier 2: the PCHIP knot tangents of `src/pchip.rs`.

  The code computes the interior tangent as the weighted harmonic mean

      w1 = 2*h[k] + h[k-1],  w2 = h[k] + 2*h[k-1]
      d  = (w1 + w2) / (w1/δ[k-1] + w2/δ[k])

  when the two secants have the same sign, and 0 otherwise.  Fritsch–Carlson's
  shape-preserving condition is |d| ≤ 3*min(|δ[k-1]|, |δ[k]|); that bound is what
  makes the Hermite cubic monotone on the interval.

  The kernel never sees a division here: the defining equation is stated
  cross-multiplied as `d * den = num`, and every inequality is derived by
  multiplication, so no `field_simp` (which is choice-tainted in this Mathlib)
  and no division-by-zero side condition appears.  The positive-secant case is
  below; the negative case follows by the sign symmetry of the same equation.
-/
import HyprEQ.OrderLemmas

namespace HyprEQ

variable {K : Type*} [Field K] [LinearOrder K] [OrderedField K]

/-- `w1 = 2*h1 + h0` (`h0 = h[k-1]`, `h1 = h[k]`). -/
def pchipW1 (h0 h1 : K) : K := 2 * h1 + h0

/-- `w2 = h1 + 2*h0`. -/
def pchipW2 (h0 h1 : K) : K := h1 + 2 * h0

/-- Numerator of the cross-multiplied harmonic mean: `(w1 + w2) * δ0 * δ1`. -/
def pchipNum (h0 h1 d0 d1 : K) : K := (pchipW1 h0 h1 + pchipW2 h0 h1) * (d0 * d1)

/-- Denominator of the cross-multiplied harmonic mean: `w1*δ1 + w2*δ0`. -/
def pchipDen (h0 h1 d0 d1 : K) : K := pchipW1 h0 h1 * d1 + pchipW2 h0 h1 * d0

theorem pchipW1_pos {h0 h1 : K} (hh0 : 0 < h0) (hh1 : 0 < h1) : 0 < pchipW1 h0 h1 := by
  unfold pchipW1
  exact add_pos (mul_pos zero_lt_two hh1) hh0

theorem pchipW2_pos {h0 h1 : K} (hh0 : 0 < h0) (hh1 : 0 < h1) : 0 < pchipW2 h0 h1 := by
  unfold pchipW2
  exact add_pos hh1 (mul_pos zero_lt_two hh0)

/-- `w2 ≤ 2*w1`, the fact that makes the harmonic mean bounded by `3*δ0`. -/
theorem pchipW2_le_two_w1 (h0 h1 : K) (hh1 : 0 < h1) :
    pchipW2 h0 h1 ≤ 2 * pchipW1 h0 h1 := by
  apply le_of_sub_nonneg
  have hident : 2 * pchipW1 h0 h1 - pchipW2 h0 h1 = 3 * h1 := by
    unfold pchipW1 pchipW2
    ring
  rw [hident]
  exact le_of_lt (mul_pos zero_lt_three hh1)

theorem pchipDen_pos {h0 h1 d0 d1 : K}
    (hh0 : 0 < h0) (hh1 : 0 < h1) (hd0 : 0 < d0) (hd1 : 0 < d1) :
    0 < pchipDen h0 h1 d0 d1 := by
  unfold pchipDen
  exact add_pos (mul_pos (pchipW1_pos hh0 hh1) hd1)
    (mul_pos (pchipW2_pos hh0 hh1) hd0)

theorem pchipNum_pos {h0 h1 d0 d1 : K}
    (hh0 : 0 < h0) (hh1 : 0 < h1) (hd0 : 0 < d0) (hd1 : 0 < d1) :
    0 < pchipNum h0 h1 d0 d1 := by
  unfold pchipNum
  exact mul_pos (add_pos (pchipW1_pos hh0 hh1) (pchipW2_pos hh0 hh1))
    (mul_pos hd0 hd1)

/-- The code's tangent, given by its cross-multiplied defining equation, is
    strictly positive when both secants are (the same-sign branch of `pchip.rs`). -/
theorem pchip_tangent_pos {h0 h1 d0 d1 d : K}
    (hh0 : 0 < h0) (hh1 : 0 < h1) (hd0 : 0 < d0) (hd1 : 0 < d1)
    (hd : d * pchipDen h0 h1 d0 d1 = pchipNum h0 h1 d0 d1) : 0 < d := by
  have hden := pchipDen_pos hh0 hh1 hd0 hd1
  have hnum := pchipNum_pos hh0 hh1 hd0 hd1
  rcases le_total 0 d with h | h
  · by_cases h0d : d = 0
    · subst h0d
      rw [zero_mul] at hd
      rw [← hd] at hnum
      exact absurd hnum (lt_irrefl (0 : K))
    · exact lt_of_le_of_ne h (Ne.symm h0d)
  · have hle : d * pchipDen h0 h1 d0 d1 ≤ 0 :=
      mul_nonpos_of_nonpos_of_nonneg h (le_of_lt hden)
    rw [hd] at hle
    exact absurd hle (lt_iff_le_not_ge.mp hnum).2

/-- Fritsch–Carlson's upper bound at the smaller secant: `d ≤ 3*δ0` when `δ0 ≤ δ1`. -/
theorem pchip_tangent_le_three_mul {h0 h1 d0 d1 d : K}
    (hh0 : 0 < h0) (hh1 : 0 < h1) (hd0 : 0 < d0) (hd1 : 0 < d1)
    (hd : d * pchipDen h0 h1 d0 d1 = pchipNum h0 h1 d0 d1) :
    d ≤ 3 * d0 := by
  refine le_of_mul_le_mul_right (pchipDen_pos hh0 hh1 hd0 hd1) ?_
  rw [hd]
  have h2w : 0 ≤ 2 * pchipW1 h0 h1 - pchipW2 h0 h1 := by
    have hident : 2 * pchipW1 h0 h1 - pchipW2 h0 h1 = 3 * h1 := by
      unfold pchipW1 pchipW2
      ring
    rw [hident]
    exact le_of_lt (mul_pos zero_lt_three hh1)
  have hw2 := pchipW2_pos hh0 hh1
  have hbracket : 0 ≤ d0 * d1 * (2 * pchipW1 h0 h1 - pchipW2 h0 h1)
      + 3 * (d0 * d0) * pchipW2 h0 h1 := by
    apply add_nonneg
    · exact mul_nonneg (mul_nonneg (le_of_lt hd0) (le_of_lt hd1)) h2w
    · exact le_of_lt (mul_pos (mul_pos zero_lt_three (mul_pos hd0 hd0)) hw2)
  have hident : 3 * d0 * pchipDen h0 h1 d0 d1 =
      pchipNum h0 h1 d0 d1 +
        (d0 * d1 * (2 * pchipW1 h0 h1 - pchipW2 h0 h1)
          + 3 * (d0 * d0) * pchipW2 h0 h1) := by
    unfold pchipNum pchipDen
    ring
  rw [hident]
  exact le_add_of_nonneg_right hbracket

/-! ### The endpoint estimate (Fritsch & Carlson 1980, Eq. 8) -/

/-- Numerator `(2*h0 + h1)*m0 - h0*m1` of the one-sided three-point endpoint slope. -/
def edgeNum (h0 h1 m0 m1 : K) : K := (2 * h0 + h1) * m0 - h0 * m1

/-- Denominator `h0 + h1` of the endpoint slope. -/
def edgeDen (h0 h1 : K) : K := h0 + h1

theorem edgeDen_pos {h0 h1 : K} (hh0 : 0 < h0) (hh1 : 0 < h1) : 0 < edgeDen h0 h1 := by
  unfold edgeDen
  exact add_pos hh0 hh1

/-- The raw endpoint estimate is bounded by `3*m0` whenever `m1 ≥ 0`; this is the
    bound the `pchip_edge_case` clamp enforces with `3.0*m0` in the code. -/
theorem edge_le_three_mul {h0 h1 m0 m1 d : K}
    (hh0 : 0 < h0) (hh1 : 0 < h1) (hm0 : 0 < m0) (hm1 : 0 ≤ m1)
    (hd : d * edgeDen h0 h1 = edgeNum h0 h1 m0 m1) : d ≤ 3 * m0 := by
  refine le_of_mul_le_mul_right (edgeDen_pos hh0 hh1) ?_
  rw [hd]
  have hbracket : 0 ≤ (h0 + 2 * h1) * m0 + h0 * m1 := by
    apply add_nonneg
    · exact mul_nonneg (add_nonneg (le_of_lt hh0) (le_of_lt (mul_pos zero_lt_two hh1)))
        (le_of_lt hm0)
    · exact mul_nonneg (le_of_lt hh0) hm1
  have hident : 3 * m0 * edgeDen h0 h1 = edgeNum h0 h1 m0 m1 + ((h0 + 2 * h1) * m0 + h0 * m1) := by
    unfold edgeNum edgeDen
    ring
  rw [hident]
  exact le_add_of_nonneg_right hbracket

end HyprEQ
