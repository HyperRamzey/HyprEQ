/-
Shared imports and conventions for the HyprEQ verification suite.

Nothing in this tree is stated over Mathlib's `ℚ`: in Lean 4.36 the `Rat`
instances themselves depend on `Classical.choice`, so a statement about `Rat`
would carry it through the typeclass.  Tier 1 is stated over `Nat`/`Int`
(where `omega`/`ring` produce proofs depending only on `propext` and
`Quot.sound`), and Tier 2 is stated over an arbitrary linear ordered field
`K` with `[Field K] [LinearOrder K] [IsStrictOrderedRing K]`.  Instantiating
`K := ℚ` is a mathematical instantiation outside the kernel; the audited
theorems themselves stay free of `Classical.choice`.
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
