/-
Tier 1 — WAV integer packing is a bijection on its declared range.

Sources: `src/wav.rs` (`decode`, `write_wav_pcm_or_float`) and `src/dither.rs`.

A WAV PCM sample is a little-endian byte string.  `packBytes n q` writes the
low `n` bytes of `q` (Euclidean remainder, which is exactly two's-complement
low bytes); `unpackBytes` folds them back as an unsigned integer.  The signed
decoder subtracts `2^(8n)` exactly when the top bit is set.  The theorems below
prove the round trip on the signed range and the writer's clamp behaviour.
-/
import HyprEQ.Basic

namespace HyprEQ

/-- The unsigned value range of an `n`-byte sample: `256^n`. -/
def byteFull (n : Nat) : Int := (256 : Int) ^ n

/-- Half the unsigned range, i.e. the magnitude boundary of the signed range:
`128 * 256^(n-1)`.  For `n = 2` this is 32768, for `n = 3` it is 8388608. -/
def byteHalf (n : Nat) : Int := 128 * (256 : Int) ^ (n - 1)

/-- Little-endian byte extraction: `q % 256`, then recurse on `q / 256`.
This is the integer content of `to_le_bytes` for an `n`-byte PCM sample. -/
def packBytes : Nat → Int → List Int
  | 0, _ => []
  | n + 1, q => q % 256 :: packBytes n (q / 256)

/-- Fold bytes back as `b0 + 256*b1 + 256^2*b2 + ...` — `from_le_bytes`. -/
def unpackBytes : List Int → Int
  | [] => 0
  | b :: bs => b + 256 * unpackBytes bs

/-- Signed interpretation of an unsigned bit pattern: subtract the full range
when the sign bit is set (the `(v << 8) as i32 >> 8` trick in `wav.rs`). -/
def toSigned (n : Nat) (u : Int) : Int :=
  if byteHalf n ≤ u then u - byteFull n else u

/-! ## Byte-level facts -/

theorem packBytes_length (n : Nat) (q : Int) : (packBytes n q).length = n := by
  induction n generalizing q with
  | zero => rfl
  | succ n ih => simp [packBytes, ih]

theorem packBytes_mem_range :
    ∀ (n : Nat) (q b : Int), b ∈ packBytes n q → 0 ≤ b ∧ b < 256
  | 0, _, _, hb => by cases hb
  | n + 1, q, b, hb => by
      simp only [packBytes, List.mem_cons] at hb
      rcases hb with rfl | hb
      · exact ⟨Int.emod_nonneg q (by decide : (256 : Int) ≠ 0), Int.emod_lt_of_pos q (by decide : (0 : Int) < 256)⟩
      · exact packBytes_mem_range n (q / 256) b hb

theorem unpackBytes_nonneg {bs : List Int} (h : ∀ b ∈ bs, 0 ≤ b) :
    0 ≤ unpackBytes bs := by
  induction bs with
  | nil => simp [unpackBytes]
  | cons b bs ih =>
      simp only [unpackBytes]
      have hb : 0 ≤ b := h b (by simp)
      have hbs : 0 ≤ unpackBytes bs := ih (fun x hx => h x (by simp [hx]))
      omega

theorem unpackBytes_lt_full {n : Nat} {bs : List Int}
    (hlen : bs.length = n) (h : ∀ b ∈ bs, 0 ≤ b ∧ b < 256) :
    unpackBytes bs < byteFull n := by
  induction bs generalizing n with
  | nil =>
      cases n with
      | zero =>
          change (0 : Int) < byteFull 0
          unfold byteFull
          decide
      | succ n =>
          change 0 = n + 1 at hlen
          omega
  | cons b bs ih =>
      cases n with
      | zero =>
          change bs.length + 1 = 0 at hlen
          omega
      | succ n =>
          have hb := h b List.mem_cons_self
          have hbs : ∀ x ∈ bs, 0 ≤ x ∧ x < 256 := fun x hx => h x (List.Mem.tail b hx)
          have hlen' : bs.length = n := by
            change bs.length + 1 = n + 1 at hlen
            omega
          have ih' := ih hlen' hbs
          change b + 256 * unpackBytes bs < byteFull (n + 1)
          rw [byteFull] at ih' ⊢
          rw [pow_succ]
          omega

/-- The unsigned range is exactly twice the signed half-range. -/
theorem byteFull_eq_two_mul_byteHalf {n : Nat} (hn : 1 ≤ n) :
    byteFull n = 2 * byteHalf n := by
  unfold byteFull byteHalf
  have h : n - 1 + 1 = n := Nat.sub_add_cancel hn
  calc
    (256 : Int) ^ n = 256 ^ (n - 1 + 1) := by rw [h]
    _ = 256 ^ (n - 1) * 256 := by rw [pow_succ]
    _ = 2 * (128 * 256 ^ (n - 1)) := by ring

/-- The signed decoder is bounded by the half-range it came from. -/
theorem toSigned_bounds {n : Nat} (hn : 1 ≤ n) {u : Int}
    (h0 : 0 ≤ u) (hfull : u < byteFull n) :
    -byteHalf n ≤ toSigned n u ∧ toSigned n u < byteHalf n := by
  unfold toSigned
  have hfull_eq : byteFull n = 2 * byteHalf n := byteFull_eq_two_mul_byteHalf hn
  split
  · constructor <;> omega
  · constructor <;> omega

/-! ## The round trip -/

/-- `unpackBytes` of the packed bytes recovers `q` for nonnegative `q`, and
`q + 256^n` for negative `q` — i.e. the two's-complement bit pattern. -/
theorem unpackBytes_packBytes {n : Nat} (hn : 1 ≤ n) {q : Int}
    (hlo : -byteHalf n ≤ q) (hhi : q < byteHalf n) :
    unpackBytes (packBytes n q) = if q < 0 then q + byteFull n else q := by
  revert q
  induction n, hn using Nat.le_induction with
  | base =>
      intro q hlo hhi
      simp only [packBytes, unpackBytes]
      have hq := Int.mul_ediv_add_emod q 256
      have hr0 := Int.emod_nonneg q (by decide : (256 : Int) ≠ 0)
      have hr1 := Int.emod_lt_of_pos q (by decide : (0 : Int) < 256)
      have hfull1 : byteFull 1 = 256 := by decide
      have hhalf1 : byteHalf 1 = 128 := by decide
      rw [hhalf1] at hlo hhi
      rw [hfull1]
      have hc : q / 256 = -1 ∨ q / 256 = 0 := by omega
      rcases hc with hc | hc <;> split <;> omega
  | succ n hn ih =>
      intro q hlo hhi
      simp only [packBytes, unpackBytes]
      have hq := Int.mul_ediv_add_emod q 256
      have hr0 := Int.emod_nonneg q (by decide : (256 : Int) ≠ 0)
      have hr1 := Int.emod_lt_of_pos q (by decide : (0 : Int) < 256)
      have hhalf : byteHalf (n + 1) = 256 * byteHalf n := by
        unfold byteHalf
        have hpow : (256 : Int) ^ (n + 1 - 1) = 256 ^ (n - 1) * 256 := by
          rw [show n + 1 - 1 = (n - 1) + 1 by omega, pow_succ]
        rw [hpow]
        ring
      have hfull : byteFull (n + 1) = 256 * byteFull n := by
        unfold byteFull
        rw [pow_succ]
        ring
      have hlo' : -byteHalf n ≤ q / 256 := by
        rw [hhalf] at hlo
        omega
      have hhi' : q / 256 < byteHalf n := by
        rw [hhalf] at hhi
        omega
      rw [ih hlo' hhi', hfull]
      split <;> split <;> omega

/-- The signed decoder of the packed bytes is exactly the original sample on
the declared signed range.  This is `unpack16_pack16` and `unpack24_pack24`
from `wav.rs`, at every byte width. -/
theorem toSigned_unpackBytes_packBytes {n : Nat} (hn : 1 ≤ n) {q : Int}
    (hlo : -byteHalf n ≤ q) (hhi : q < byteHalf n) :
    toSigned n (unpackBytes (packBytes n q)) = q := by
  have hbytes := unpackBytes_packBytes hn hlo hhi
  have hbytes_range : ∀ b ∈ packBytes n q, 0 ≤ b ∧ b < 256 :=
    fun b hb => packBytes_mem_range n q b hb
  have hlen : (packBytes n q).length = n := packBytes_length n q
  have hnonneg : 0 ≤ unpackBytes (packBytes n q) :=
    unpackBytes_nonneg (fun b hb => (hbytes_range b hb).1)
  have hlt : unpackBytes (packBytes n q) < byteFull n :=
    unpackBytes_lt_full hlen hbytes_range
  unfold toSigned
  have hfull_eq : byteFull n = 2 * byteHalf n := byteFull_eq_two_mul_byteHalf hn
  split
  · -- sign bit set: the packed value is `q + full`
    rw [hbytes]
    split <;> omega
  · -- sign bit clear: the packed value is `q`
    rw [hbytes]
    split <;> omega

/-- 16-bit PCM: `src/wav.rs` `decode`/`write_wav_pcm_or_float` (bits = 16). -/
theorem pcm_s16_roundtrip {q : Int} (hlo : -32768 ≤ q) (hhi : q ≤ 32767) :
    toSigned 2 (unpackBytes (packBytes 2 q)) = q := by
  apply toSigned_unpackBytes_packBytes (by decide)
  · unfold byteHalf; norm_num; omega
  · unfold byteHalf; norm_num; omega

/-- 24-bit PCM: `src/wav.rs` (bits = 24), where the three bytes are sign-extended
by shifting into the top of an `i32` and back. -/
theorem pcm_s24_roundtrip {q : Int} (hlo : -8388608 ≤ q) (hhi : q ≤ 8388607) :
    toSigned 3 (unpackBytes (packBytes 3 q)) = q := by
  apply toSigned_unpackBytes_packBytes (by decide)
  · unfold byteHalf; norm_num; omega
  · unfold byteHalf; norm_num; omega

/-- 8-bit PCM is unsigned with a 128 offset (`wav.rs`: `(b - 128)/128`). -/
def pcm_u8_decode (b : Int) : Int := b - 128

theorem pcm_u8_roundtrip (b : Int) : pcm_u8_decode b = b - 128 := rfl

/-! ## The writer's clamp -/

/-- `q.clamp(-32768, 32767)` from the 16-bit writer. -/
def clamp16 (q : Int) : Int := max (-32768) (min 32767 q)

/-- `q.clamp(-8388608, 8388607)` from the 24-bit writer. -/
def clamp24 (q : Int) : Int := max (-8388608) (min 8388607 q)

theorem clamp16_range (q : Int) : -32768 ≤ clamp16 q ∧ clamp16 q ≤ 32767 := by
  unfold clamp16; constructor <;> omega

theorem clamp24_range (q : Int) : -8388608 ≤ clamp24 q ∧ clamp24 q ≤ 8388607 := by
  unfold clamp24; constructor <;> omega

theorem clamp16_low {q : Int} (h : q < -32768) : clamp16 q = -32768 := by
  unfold clamp16; omega

theorem clamp16_high {q : Int} (h : 32767 < q) : clamp16 q = 32767 := by
  unfold clamp16; omega

theorem clamp16_idem (q : Int) : clamp16 (clamp16 q) = clamp16 q := by
  unfold clamp16; omega

theorem clamp24_idem (q : Int) : clamp24 (clamp24 q) = clamp24 q := by
  unfold clamp24; omega

end HyprEQ
