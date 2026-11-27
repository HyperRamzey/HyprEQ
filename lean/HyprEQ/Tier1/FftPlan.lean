/-
Tier 1 — the convolver's FFT plan is big enough for the *linear* convolution.

Source: `src/fft.rs` (`bit_length`, `shl_bit_length`) and
`src/convolve.rs` (`block_bound`, `plan`).

`shl_bit_length v` is `1 << v.bit_length()`: the smallest power of two that is
*strictly greater* than `v` (for `v > 0`).  The convolver pads each block to
that size; `nFft_covers_linear` is the reason no circular wrap can occur: the
output length `block_len + n_ir - 1` fits inside the transform.
-/
import HyprEQ.Basic

namespace HyprEQ

/-- `usize::bit_length`: number of bits needed to represent `n`, `0` at zero. -/
def bitLength (n : Nat) : Nat :=
  if n = 0 then 0 else Nat.log2 n + 1

/-- `1 << v.bit_length()` from `src/fft.rs`.  In `Nat` the Rust saturation at
`usize::MAX` is not needed; the arithmetic is exact. -/
def shlBitLength (n : Nat) : Nat := 2 ^ bitLength n

/-- Python's `1 << v.bit_length()` is strictly above `v` whenever `v > 0`. -/
theorem bitLength_succ_gt {n : Nat} (hn : 0 < n) : n < shlBitLength n := by
  unfold shlBitLength bitLength
  simp only [if_neg (Nat.ne_of_gt hn)]
  have hlt : n.log2 < n.log2 + 1 := Nat.lt_succ_self _
  exact (Nat.log2_lt (Nat.ne_of_gt hn)).mp hlt

/-- The shift is the *least* power of two above `n`: if `n < 2^k` then the
chosen power is at most `2^k`. -/
theorem shlBitLength_le_of_lt {n k : Nat} (hn : 0 < n) (hk : n < 2 ^ k) :
    shlBitLength n ≤ 2 ^ k := by
  unfold shlBitLength bitLength
  simp only [if_neg (Nat.ne_of_gt hn)]
  have hlog : n.log2 < k := (Nat.log2_lt (Nat.ne_of_gt hn)).mpr hk
  have hle : n.log2 + 1 ≤ k := by omega
  exact Nat.pow_le_pow_right (by decide) hle

/-- `min(65536, n_in)` from `src/convolve.rs`. -/
def blockBound (n : Nat) : Nat := if n < 65536 then n else 65536

/-- `max(16384, min(65536, n_in))`. -/
def blockLen (n : Nat) : Nat := max 16384 (blockBound n)

/-- `shl_bit_length(block_len + n_ir - 2)` — the overlap-add transform size. -/
def nFft (n m : Nat) : Nat := shlBitLength (blockLen n + m - 2)

/-- The transform size is a power of two (the complex FFT planner's requirement). -/
theorem nFft_pow2 (n m : Nat) : ∃ k, nFft n m = 2 ^ k :=
  ⟨bitLength (blockLen n + m - 2), rfl⟩

/-- **No circular wrap.**  For a non-empty block and a non-empty FIR, the
linear convolution of the two fits inside the zero-padded transform:
`block_len + n_ir - 1 ≤ n_fft`.  This is the discrete fact behind the
overlap-add path being an exact linear convolution rather than a circular one. -/
theorem nFft_covers_linear {n m : Nat} (hn : 0 < n) (hm : 0 < m) :
    blockLen n + m - 1 ≤ nFft n m := by
  unfold nFft
  have hbk : 16384 ≤ blockLen n := by
    unfold blockLen
    exact Nat.le_max_left _ _
  have hv : 0 < blockLen n + m - 2 := by omega
  have hgt := bitLength_succ_gt (n := blockLen n + m - 2) hv
  omega

/-- The plan is at least one full block, as `max 16384` requires. -/
theorem blockLen_ge (n : Nat) : 16384 ≤ blockLen n := by
  unfold blockLen
  exact Nat.le_max_left _ _

/-- The block length never exceeds 65536, so the block loop's stride is bounded. -/
theorem blockLen_le (n : Nat) : blockLen n ≤ 65536 := by
  unfold blockLen blockBound
  split
  · exact Nat.max_le.mpr ⟨by omega, by omega⟩
  · exact Nat.max_le.mpr ⟨by omega, le_rfl⟩

end HyprEQ
