/-
  Tier 2: the limiter's gain arithmetic (`src/limiter.rs`).

  Two facts the processing code relies on, stated over the division-free
  `OrderedField` toolbox:

  * `required` (limiter.rs:735) returns `ceiling / after` when the post-stage
    peak `after` is above the ceiling, else 1.  Stated as its defining equation
    `g * after = ceiling`, any such gain is in `(0, 1)` when
    `0 < ceiling < after` — a limiter never boosts and never inverts.
  * `advance_gain` (limiter.rs:847) either drops straight onto a lower target
    or moves by a one-pole step `g + (1-g)*(1-c)` toward a target at or above
    `g`, then clamps with `min(target)`.  Both candidates are in `[0,1]` when
    `g`, `target` and the coefficient are, so the follower cannot leave the
    unit interval.
-/
import HyprEQ.OrderLemmas

namespace HyprEQ

variable {K : Type*} [Field K] [LinearOrder K] [OrderedField K]

/-- `min` on a linear order, spelled with the order's decidability. -/
def kmin (a b : K) : K := if a ≤ b then a else b

theorem kmin_le_left (a b : K) : kmin a b ≤ a := by
  unfold kmin
  split_ifs with h
  · exact le_refl a
  · rcases le_total a b with h' | h'
    · exact absurd h' h
    · exact h'

theorem kmin_le_right (a b : K) : kmin a b ≤ b := by
  unfold kmin
  split_ifs with h
  · exact h
  · exact le_refl b

theorem le_kmin {c a b : K} (h1 : c ≤ a) (h2 : c ≤ b) : c ≤ kmin a b := by
  unfold kmin
  split_ifs with h
  · exact h1
  · exact h2

theorem kmin_bounds {a b : K} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hb0 : 0 ≤ b) (_hb1 : b ≤ 1) :
    0 ≤ kmin a b ∧ kmin a b ≤ 1 := by
  constructor
  · exact le_kmin ha0 hb0
  · exact le_trans (kmin_le_left a b) ha1

/-- `neg_nonpos`: `-a ≤ 0` when `a ≥ 0`. -/
theorem neg_nonpos {a : K} (h : 0 ≤ a) : -a ≤ 0 := by
  have h' := OrderedField.add_le_add_left (0 : K) a h (-a)
  rw [zero_add, add_neg_cancel] at h'
  exact h'

/-- `required`'s gain is strictly between 0 and 1 when the post-stage peak is
    above the ceiling (`g * after = ceiling`, `0 < ceiling < after`). -/
theorem required_gain_bounds {ceiling after g : K}
    (hc : 0 < ceiling) (ha : ceiling < after) (hg : g * after = ceiling) :
    0 < g ∧ g < 1 := by
  have ha0 : 0 < after := lt_trans hc ha
  constructor
  · rcases le_total 0 g with h | h
    · by_cases hg0 : g = 0
      · subst hg0
        rw [zero_mul] at hg
        rw [← hg] at hc
        exact absurd hc (lt_irrefl (0 : K))
      · exact lt_of_le_of_ne h (Ne.symm hg0)
    · have hle : g * after ≤ 0 := mul_nonpos_of_nonpos_of_nonneg h (le_of_lt ha0)
      rw [hg] at hle
      exact absurd hle (lt_iff_le_not_ge.mp hc).2
  · rcases le_total 1 g with h | h
    · exfalso
      by_cases h1 : g = 1
      · subst h1
        rw [one_mul] at hg
        rw [hg] at ha
        exact absurd ha (lt_irrefl ceiling)
      · have hlt : 1 < g := lt_of_le_of_ne h (Ne.symm h1)
        have hle : (1 : K) * after ≤ g * after :=
          mul_le_mul_of_nonneg_right (le_of_lt hlt) (le_of_lt ha0)
        rw [one_mul, hg] at hle
        exact absurd hle (lt_iff_le_not_ge.mp ha).2
    · by_cases h1 : g = 1
      · exfalso
        subst h1
        rw [one_mul] at hg
        rw [hg] at ha
        exact absurd ha (lt_irrefl ceiling)
      · exact lt_of_le_of_ne h h1

/-- One step of the release follower (`advance_gain`). -/
def followerNext (g target c : K) : K :=
  if target < g then target else kmin (g + (1 - g) * (1 - c)) target

/-- The follower maps the unit interval into itself. -/
theorem followerNext_bounds {g target c : K}
    (hg0 : 0 ≤ g) (hg1 : g ≤ 1) (ht0 : 0 ≤ target) (ht1 : target ≤ 1)
    (hc0 : 0 ≤ c) (hc1 : c ≤ 1) :
    0 ≤ followerNext g target c ∧ followerNext g target c ≤ 1 := by
  unfold followerNext
  split_ifs with h
  · exact ⟨ht0, ht1⟩
  · have h1g : 0 ≤ 1 - g := sub_nonneg hg1
    have h1c : 0 ≤ 1 - c := sub_nonneg hc1
    have hc_le : 1 - c ≤ 1 := by
      have h' : (1 : K) + (-c) ≤ 1 + 0 :=
        add_le_add (le_refl 1) (neg_nonpos hc0)
      rw [add_zero] at h'
      have hid : (1 : K) - c = 1 + (-c) := by ring
      rw [hid]
      exact h'
    have ha0 : 0 ≤ g + (1 - g) * (1 - c) :=
      add_nonneg hg0 (mul_nonneg h1g h1c)
    have ha1 : g + (1 - g) * (1 - c) ≤ 1 := by
      have hm : (1 - g) * (1 - c) ≤ (1 - g) * 1 :=
        mul_le_mul_of_nonneg_left hc_le h1g
      rw [mul_one] at hm
      have hsum : g + (1 - g) = 1 := by ring
      calc g + (1 - g) * (1 - c) ≤ g + (1 - g) := add_le_add (le_refl g) hm
        _ = 1 := hsum
    exact kmin_bounds ha0 ha1 ht0 ht1

end HyprEQ
