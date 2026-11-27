/-
  Non-vacuity of `HyprEQ.OrderedField` at ℚ — the measured witness.

  `HyprEQ/Nonvacuity.lean` proves the audited, choice-free part: every field
  of `OrderedField` is a direct projection of a standard Mathlib order class
  (`ZeroLEOneClass`, `IsOrderedCancelAddMonoid`, `PosMulStrictMono`,
  `MulPosStrictMono`), so the class is derivable for any linearly ordered
  field carrying those classes.  At ℚ the instance fires:

      #synth OrderedField ℚ
      instOrderedFieldOfClasses

  What cannot be done in Lean 4.36 is to audit a ℚ-specialised proof with zero
  `Classical.choice`.  Core `Rat` arithmetic is itself choice-tainted:
  `Rat.add` and `Rat.mul` call `Rat.normalize`, whose `normalize.reduced`
  proof uses `Int.natAbs_ediv_of_dvd`; `#print axioms` reports
  `Classical.choice` for both, and consequently for every proof whose term
  mentions `a + b` or `a * b` on ℚ — even `rfl : a + b = a + b`.  The
  footprint below is therefore the toolchain's, not this file's: the class
  derivation is clean; the ℚ carrier instances it is applied to are not.

  This file is outside `HyprEQ/`, so it is not part of the audited library and
  `scripts/lean-verify.sh` does not count `orderedField_rat_nonempty` among
  the shipped theorems.  The gate audits the clean generic theorem
  `HyprEQ.orderedField_of_classes` and pins the footprint measured here.
-/
import HyprEQ

namespace HyprEQ

#synth OrderedField ℚ

theorem orderedField_rat_nonempty : Nonempty (OrderedField ℚ) := ⟨inferInstance⟩

#print axioms orderedField_of_classes
#print axioms orderedField_rat_nonempty

end HyprEQ
