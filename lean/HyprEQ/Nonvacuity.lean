/-
  Non-vacuity of the local `OrderedField` class.

  A `Prop`-valued class with contradictory fields would make every Tier 2
  theorem vacuously true.  This module shows the class is derivable from
  Mathlib's standard order classes without `Classical.choice`: every field is
  a direct projection of `ZeroLEOneClass`, `IsOrderedCancelAddMonoid`,
  `PosMulStrictMono`, or `MulPosStrictMono`.

  The instance fires at ℚ (and at any linearly ordered field carrying those
  standard instances), so `OrderedField ℚ` is inhabited.  The generic theorem
  `orderedField_of_classes` is the audited witness, with axioms
  `[propext, Quot.sound]`.  A ℚ-specialised *term* necessarily inherits
  `Classical.choice` from ℚ's own algebra instances — core `Rat.add`/`Rat.mul`
  call `Rat.normalize`, whose `normalize.reduced` proof uses
  `Int.natAbs_ediv_of_dvd` — so the specialised footprint is measured
  separately in the root `Nonvacuity.lean`, outside the audited library.
-/
import HyprEQ.OrderLemmas

namespace HyprEQ

/-- `OrderedField` follows from the standard Mathlib order classes, with no
    classical reasoning in the derivation. -/
instance instOrderedFieldOfClasses {K : Type*} [Field K] [LinearOrder K]
    [ZeroLEOneClass K] [IsOrderedCancelAddMonoid K]
    [PosMulStrictMono K] [MulPosStrictMono K] :
    OrderedField K where
  zero_le_one := ZeroLEOneClass.zero_le_one
  add_le_add_left := IsOrderedCancelAddMonoid.toIsOrderedAddMonoid.add_le_add_left
  le_of_add_le_add_left := IsOrderedCancelAddMonoid.le_of_add_le_add_left
  mul_lt_mul_of_pos_left := PosMulStrictMono.mul_lt_mul_of_pos_left
  mul_lt_mul_of_pos_right := MulPosStrictMono.mul_lt_mul_of_pos_right
  exists_pair_ne := Nontrivial.exists_pair_ne

/-- The class is inhabited whenever the standard order classes are.  `ℚ` and
    `ℝ` satisfy the hypotheses, so the six fields are mutually consistent. -/
theorem orderedField_of_classes {K : Type*} [Field K] [LinearOrder K]
    [ZeroLEOneClass K] [IsOrderedCancelAddMonoid K]
    [PosMulStrictMono K] [MulPosStrictMono K] :
    OrderedField K := instOrderedFieldOfClasses

end HyprEQ
