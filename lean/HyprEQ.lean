/-
  Root module for the HyprEQ Lean proofs.

  Importing this file builds the entire shipped proof surface.  The axiom
  audit (`AxiomAudit.lean`) prints the dependencies of every theorem declared
  in these modules, and `scripts/lean-verify.sh` refuses any theorem whose
  axioms are not a subset of `[propext, Quot.sound]`.
-/
import HyprEQ.Basic
import HyprEQ.OrderLemmas
import HyprEQ.Nonvacuity
import HyprEQ.Tier1.FftPlan
import HyprEQ.Tier1.WavQuant
import HyprEQ.Tier1.LimiterIndex
import HyprEQ.Tier2.PchipTangent
import HyprEQ.Tier2.PchipMonotone
import HyprEQ.Tier2.LimiterGain
