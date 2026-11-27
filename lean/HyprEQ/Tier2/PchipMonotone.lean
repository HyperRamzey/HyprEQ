/-
  Tier 2: monotonicity of the PCHIP Hermite cubic (`src/pchip.rs`).

  On an interval normalized to `h(0) = 0`, `h(1) = 1`, the Hermite cubic with
  endpoint slopes `a = d[k-1]`, `b = d[k]` is

      h(t) = (a + b - 2) t^3 + (3 - 2a - b) t^2 + a t

  and its derivative is the quadratic

      q(t) = 3(a + b - 2) t^2 + (-4a - 2b + 6) t + a.

  Fritsch–Carlson's sufficient condition is `0 ≤ a, b ≤ 3`.  The proof splits on
  `a + b`:

  * `a + b ≤ 3`: the Bernstein form
    `q = a(1-t)^2 + 2(3-a-b)t(1-t) + b t^2` has nonnegative coefficients;
  * `a + b ≥ 3`: `q` is convex, and the completed-square identity
    `4A q = (2At + B)^2 + (4AC - B^2)` reduces the claim to `4AC - B^2 ≥ 0`,
    i.e. `f(a,b) ≤ 0` for `f = a² + b² + ab - 6a - 6b + 9`, which follows from
    the factorization `f = (a-3)(a-(a+b)+3) + ((a+b)-3)((a+b)-6)` and the box
    constraints.

  Monotonicity on `[0,1]` then follows from `q ≥ 0` through the exact Simpson
  identity for a quadratic: with `2m = t1 + t2`,

      6 * (h(t2) - h(t1))/(t2 - t1) = q(t1) + 4 q(m) + q(t2),

  so `h(t2) - h(t1) ≥ 0`.  The midpoint is passed explicitly with the defining
  equation `2*m = t1 + t2` so that the kernel stays division-free.
-/
import HyprEQ.OrderLemmas
import Mathlib.Tactic.LinearCombination

namespace HyprEQ

variable {K : Type*} [Field K] [LinearOrder K] [OrderedField K]

/-- Leading coefficient `3*(a+b-2)` of the derivative quadratic. -/
def hermiteA (a b : K) : K := 3 * (a + b - 2)

/-- Linear coefficient `6 - 4a - 2b` of the derivative quadratic. -/
def hermiteB (a b : K) : K := 6 - 4 * a - 2 * b

/-- Derivative `q(t)` of the normalized Hermite cubic. -/
def hermiteD (a b t : K) : K := hermiteA a b * t ^ 2 + hermiteB a b * t + a

/-- The normalized Hermite cubic itself. -/
def hermiteCubic (a b t : K) : K :=
  (a + b - 2) * t ^ 3 + (3 - 2 * a - b) * t ^ 2 + a * t

/-- Bernstein form of the derivative quadratic. -/
theorem hermiteD_bern (a b t : K) :
    hermiteD a b t = a * (1 - t) ^ 2 + 2 * (3 - a - b) * t * (1 - t) + b * t ^ 2 := by
  unfold hermiteD hermiteA hermiteB
  ring

/-- The `a + b ≤ 3` branch: all Bernstein coefficients are nonnegative. -/
theorem hermiteD_nonneg_bern {a b t : K}
    (ha0 : 0 ≤ a) (hb0 : 0 ≤ b) (hab : a + b ≤ 3) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    0 ≤ hermiteD a b t := by
  rw [hermiteD_bern]
  have h1 : 0 ≤ a * (1 - t) ^ 2 := mul_nonneg ha0 (sq_nonneg_pow (1 - t))
  have h2 : 0 ≤ 2 * (3 - a - b) * t * (1 - t) := by
    have h3ab : 0 ≤ 3 - a - b := by
      have hid : 3 - a - b = 3 - (a + b) := by ring
      rw [hid]
      exact sub_nonneg hab
    have h2a : 0 ≤ 2 * (3 - a - b) := mul_nonneg (le_of_lt zero_lt_two) h3ab
    exact mul_nonneg (mul_nonneg h2a ht0) (sub_nonneg ht1)
  have h3 : 0 ≤ b * t ^ 2 := mul_nonneg hb0 (sq_nonneg_pow t)
  exact add_nonneg (add_nonneg h1 h2) h3

/-- The shape factor `f(a,b) = a² + b² + ab - 6a - 6b + 9` is nonpositive on the
    box `[0,3]²` above the diagonal `a + b ≥ 3`. -/
theorem shape_factor_nonpos {a b : K}
    (ha3 : a ≤ 3) (hb3 : b ≤ 3) (hsum : 3 ≤ a + b) :
    a * a + b * b + a * b - 6 * a - 6 * b + 9 ≤ 0 := by
  have hfac : a * a + b * b + a * b - 6 * a - 6 * b + 9
      = (a - 3) * (a - (a + b) + 3) + ((a + b) - 3) * ((a + b) - 6) := by
    ring
  rw [hfac]
  apply add_nonpos
  · apply mul_nonpos_of_nonpos_of_nonneg
    · exact sub_nonpos ha3
    · have hid : a - (a + b) + 3 = 3 - b := by ring
      rw [hid]
      exact sub_nonneg hb3
  · apply mul_nonpos_of_nonneg_of_nonpos
    · exact sub_nonneg hsum
    · apply sub_nonpos
      have h6 : a + b ≤ 6 := by
        rw [six_eq]
        exact add_le_add ha3 hb3
      exact h6

/-- Fritsch–Carlson: the derivative is nonnegative on `[0,1]` for `0 ≤ a,b ≤ 3`. -/
theorem hermiteD_nonneg {a b t : K}
    (ha0 : 0 ≤ a) (ha3 : a ≤ 3) (hb0 : 0 ≤ b) (hb3 : b ≤ 3)
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1) : 0 ≤ hermiteD a b t := by
  rcases le_total (a + b) 3 with hsum | hsum
  · exact hermiteD_nonneg_bern ha0 hb0 hsum ht0 ht1
  · have hA : 0 < hermiteA a b := by
      unfold hermiteA
      exact mul_pos zero_lt_three (sub_pos (lt_of_lt_of_le two_lt_three hsum))
    have h4A : 0 < 4 * hermiteA a b := mul_pos zero_lt_four hA
    have hf := shape_factor_nonpos ha3 hb3 hsum
    have hnegf : 0 ≤ -(a * a + b * b + a * b - 6 * a - 6 * b + 9) := neg_nonneg hf
    have hdisc : 0 ≤ 4 * hermiteA a b * a - hermiteB a b ^ 2 := by
      have hid : 4 * hermiteA a b * a - hermiteB a b ^ 2
          = 4 * (-(a * a + b * b + a * b - 6 * a - 6 * b + 9)) := by
        unfold hermiteA hermiteB
        ring
      rw [hid]
      exact mul_nonneg (le_of_lt zero_lt_four) hnegf
    have hmain : 4 * hermiteA a b * hermiteD a b t
        = (2 * hermiteA a b * t + hermiteB a b) ^ 2
          + (4 * hermiteA a b * a - hermiteB a b ^ 2) := by
      unfold hermiteD
      ring
    have hq : 0 ≤ 4 * hermiteA a b * hermiteD a b t := by
      rw [hmain]
      exact add_nonneg (sq_nonneg_pow _) hdisc
    apply le_of_mul_le_mul_right h4A
    rw [zero_mul, mul_comm (hermiteD a b t) (4 * hermiteA a b)]
    exact hq

/-- Difference factorization: `h(t2) - h(t1) = (t2-t1) * G(t1,t2)`. -/
theorem hermiteCubic_diff (a b t1 t2 : K) :
    hermiteCubic a b t2 - hermiteCubic a b t1
      = (t2 - t1) * ((a + b - 2) * (t1 * t1 + t1 * t2 + t2 * t2)
          + (3 - 2 * a - b) * (t1 + t2) + a) := by
  unfold hermiteCubic
  ring

/-- Simpson's identity for the derivative quadratic, division-free: with
    `2*m = t1 + t2`, the secant slope `G` satisfies `6G = q(t1)+4q(m)+q(t2)`. -/
theorem hermiteD_simpson (a b t1 t2 m : K) (hm : 2 * m = t1 + t2) :
    6 * ((a + b - 2) * (t1 * t1 + t1 * t2 + t2 * t2) + (3 - 2 * a - b) * (t1 + t2) + a)
      = hermiteD a b t1 + 4 * hermiteD a b m + hermiteD a b t2 := by
  unfold hermiteD hermiteA hermiteB
  linear_combination
    (-(6 * a * m + 3 * a * t1 + 3 * a * t2 - 8 * a + 6 * b * m + 3 * b * t1 + 3 * b * t2
      - 4 * b - 12 * m - 6 * t1 - 6 * t2 + 12)) * hm

/-- Monotonicity of the PCHIP Hermite cubic on `[0,1]` under the
    Fritsch–Carlson slope bounds `0 ≤ a,b ≤ 3`. -/
theorem hermiteCubic_monotone {a b t1 t2 m : K}
    (ha0 : 0 ≤ a) (ha3 : a ≤ 3) (hb0 : 0 ≤ b) (hb3 : b ≤ 3)
    (ht10 : 0 ≤ t1) (ht21 : t2 ≤ 1) (ht12 : t1 ≤ t2) (hm : 2 * m = t1 + t2) :
    hermiteCubic a b t1 ≤ hermiteCubic a b t2 := by
  have ht11 : t1 ≤ 1 := le_trans ht12 ht21
  have ht20 : 0 ≤ t2 := le_trans ht10 ht12
  have hm0 : 0 ≤ m := by
    apply le_of_mul_le_mul_right zero_lt_two
    rw [zero_mul, mul_comm m 2, hm]
    exact add_nonneg ht10 ht20
  have hm1 : m ≤ 1 := by
    apply le_of_mul_le_mul_right zero_lt_two
    rw [mul_comm m 2, hm, one_mul, two_eq]
    exact add_le_add ht11 ht21
  have hq1 := hermiteD_nonneg ha0 ha3 hb0 hb3 ht10 ht11
  have hqm := hermiteD_nonneg ha0 ha3 hb0 hb3 hm0 hm1
  have hq2 := hermiteD_nonneg ha0 ha3 hb0 hb3 ht20 ht21
  have hRHS : 0 ≤ hermiteD a b t1 + 4 * hermiteD a b m + hermiteD a b t2 := by
    apply add_nonneg
    · exact add_nonneg hq1 (mul_nonneg (le_of_lt zero_lt_four) hqm)
    · exact hq2
  have hG6 := hermiteD_simpson a b t1 t2 m hm
  have hG0 : 0 ≤ (a + b - 2) * (t1 * t1 + t1 * t2 + t2 * t2)
      + (3 - 2 * a - b) * (t1 + t2) + a := by
    apply le_of_mul_le_mul_right zero_lt_six
    show (0 : K) * 6 ≤ ((a + b - 2) * (t1 * t1 + t1 * t2 + t2 * t2)
      + (3 - 2 * a - b) * (t1 + t2) + a) * 6
    rw [zero_mul]
    have h6G : 0 ≤ 6 * ((a + b - 2) * (t1 * t1 + t1 * t2 + t2 * t2)
        + (3 - 2 * a - b) * (t1 + t2) + a) := by
      rw [hG6]
      exact hRHS
    rwa [mul_comm _ 6]
  have hdiff : 0 ≤ hermiteCubic a b t2 - hermiteCubic a b t1 := by
    rw [hermiteCubic_diff]
    exact mul_nonneg (sub_nonneg ht12) hG0
  exact le_of_sub_nonneg hdiff

end HyprEQ
