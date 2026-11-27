/-
Shared imports and conventions for the HyprEQ verification suite.

Nothing under `HyprEQ/` is stated over Mathlib's `ℚ`: in Lean 4.36 the `Rat`
instances themselves depend on `Classical.choice`, so a statement about `Rat`
would carry it through the typeclass.  Tier 1 is stated over `Nat`/`Int`
(where `omega`/`ring` produce proofs depending only on `propext` and
`Quot.sound`), and Tier 2 is stated over an arbitrary linear ordered field
`K` with the local `OrderedField` class.  That class is not vacuous:
`HyprEQ/Nonvacuity.lean` derives it choice-free from the standard Mathlib
order classes, and `../Nonvacuity.lean` (outside the audited tree) witnesses
`OrderedField ℚ` together with the `Classical.choice` footprint that ℚ's own
core arithmetic unavoidably carries.
-/
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Data.Finset.Card
import Mathlib.Order.Interval.Finset.Fin
import Mathlib.Algebra.Order.Field.Rat
import Mathlib.Algebra.Polynomial.Coeff

open Finset
open scoped BigOperators

namespace HyprEQ

end HyprEQ
