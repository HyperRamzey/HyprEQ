/-
  Tier 1: the limiter's index arithmetic (`src/limiter.rs`, `Cfg`).

  All quantities are `Nat` counts of samples; the proofs are exact integer
  arithmetic with `omega` and the core `min`/`max`/`%` lemmas, so no
  `Classical.choice` enters.  The theorems are the invariants the processing
  loop relies on:

  * the minimum lookahead is `report_delay + MIN_RAMP_FRAMES + 1`, so the
    configured lookahead always covers the attack ramp plus the delay line;
  * the ramp is never shorter than `MIN_RAMP_FRAMES`, and never longer than
    either the room left by the lookahead or the preferred attack length;
  * every scheduled slot `j` addresses a live ring cell `j % ring_len`.
-/
import HyprEQ.Basic

namespace HyprEQ

/-- `MIN_RAMP_FRAMES` from `limiter.rs`. -/
def minRampFrames : Nat := 2

/-- `min_lookahead_samples()`. -/
def minLookahead (reportDelay : Nat) : Nat := reportDelay + minRampFrames + 1

/-- `lookahead_samples(rate)`. -/
def lookaheadSamples (requestedLookahead reportDelay : Nat) : Nat :=
  max (max requestedLookahead (minLookahead reportDelay)) 1

/-- `ramp_frames(rate)`: the attack ramp clamped to the lookahead room and the
    preferred attack length.  `la` is `lookahead_samples(rate)`. -/
def rampFrames (la reportDelay preferredRamp : Nat) : Nat :=
  min (max (la - reportDelay - 1) minRampFrames)
      (max preferredRamp minRampFrames)

/-- `latency_samples(rate)`. -/
def latencySamples (la reportDelay : Nat) : Nat := la + reportDelay

/-- `slot(j) = j % ring_len`. -/
def slot (j ringLen : Nat) : Nat := j % ringLen

theorem minLookahead_eq (reportDelay : Nat) :
    minLookahead reportDelay = reportDelay + 3 := by
  unfold minLookahead minRampFrames
  omega

theorem requested_le_lookahead (requestedLookahead reportDelay : Nat) :
    requestedLookahead ≤ lookaheadSamples requestedLookahead reportDelay := by
  unfold lookaheadSamples
  exact le_trans (le_max_left _ _) (le_max_left _ _)

theorem minLookahead_le_lookahead (requestedLookahead reportDelay : Nat) :
    minLookahead reportDelay ≤ lookaheadSamples requestedLookahead reportDelay := by
  unfold lookaheadSamples
  exact le_trans (le_max_right _ _) (le_max_left _ _)

theorem one_le_lookahead (requestedLookahead reportDelay : Nat) :
    1 ≤ lookaheadSamples requestedLookahead reportDelay := by
  unfold lookaheadSamples
  exact le_max_right _ _

theorem reportDelay_add_three_le_lookahead (requestedLookahead reportDelay : Nat) :
    reportDelay + 3 ≤ lookaheadSamples requestedLookahead reportDelay := by
  have h := minLookahead_le_lookahead requestedLookahead reportDelay
  rwa [minLookahead_eq] at h

theorem ramp_ge_minRampFrames (la reportDelay preferredRamp : Nat) :
    minRampFrames ≤ rampFrames la reportDelay preferredRamp := by
  unfold rampFrames
  exact le_min (le_max_right _ _) (le_max_right _ _)

theorem ramp_le_room (la reportDelay preferredRamp : Nat) :
    rampFrames la reportDelay preferredRamp
      ≤ max (la - reportDelay - 1) minRampFrames := by
  unfold rampFrames
  exact min_le_left _ _

theorem ramp_le_preferred (la reportDelay preferredRamp : Nat) :
    rampFrames la reportDelay preferredRamp ≤ max preferredRamp minRampFrames := by
  unfold rampFrames
  exact min_le_right _ _

/-- The configured ramp always fits in the lookahead: an attack never needs
    more frames than the scheduler has already committed. -/
theorem ramp_timeliness (requestedLookahead reportDelay preferredRamp : Nat) :
    rampFrames (lookaheadSamples requestedLookahead reportDelay) reportDelay preferredRamp
        + reportDelay + 1
      ≤ lookaheadSamples requestedLookahead reportDelay := by
  have hfloor := reportDelay_add_three_le_lookahead requestedLookahead reportDelay
  have hroom : minRampFrames ≤
      lookaheadSamples requestedLookahead reportDelay - reportDelay - 1 := by
    unfold minRampFrames
    omega
  have hmax : max (lookaheadSamples requestedLookahead reportDelay - reportDelay - 1)
        minRampFrames
      = lookaheadSamples requestedLookahead reportDelay - reportDelay - 1 :=
    le_antisymm (max_le (le_refl _) hroom) (le_max_left _ _)
  have hle := ramp_le_room (lookaheadSamples requestedLookahead reportDelay)
    reportDelay preferredRamp
  rw [hmax] at hle
  omega

/-- The ramp plus the delay line is strictly inside the lookahead, so the
    schedule for the last frame of a ramp is committed before the frame that
    needs it is emitted. -/
theorem ramp_plus_delay_lt_latency (requestedLookahead reportDelay preferredRamp : Nat) :
    rampFrames (lookaheadSamples requestedLookahead reportDelay) reportDelay preferredRamp
        + reportDelay
      < latencySamples (lookaheadSamples requestedLookahead reportDelay) reportDelay := by
  unfold latencySamples
  have h := ramp_timeliness requestedLookahead reportDelay preferredRamp
  omega

theorem latency_eq (la reportDelay : Nat) :
    latencySamples la reportDelay = la + reportDelay := rfl

theorem slot_lt {j ringLen : Nat} (hring : 0 < ringLen) :
    slot j ringLen < ringLen := by
  unfold slot
  exact Nat.mod_lt j hring

theorem slot_add_ring (j ringLen : Nat) :
    slot (j + ringLen) ringLen = slot j ringLen := by
  unfold slot
  exact Nat.add_mod_right j ringLen

theorem slot_eq_of_lt {j ringLen : Nat} (h : j < ringLen) :
    slot j ringLen = j := by
  unfold slot
  exact Nat.mod_eq_of_lt h

theorem slot_inj {j k ringLen : Nat} (hj : j < ringLen) (hk : k < ringLen)
    (h : slot j ringLen = slot k ringLen) : j = k := by
  rw [slot_eq_of_lt hj, slot_eq_of_lt hk] at h
  exact h

end HyprEQ
