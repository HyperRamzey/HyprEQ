//! Bit-exact port of the dither noise source `hypreq.py` uses.
//!
//! The Python writes 16/24-bit PCM with TPDF dither drawn from
//! `numpy.random.default_rng(0x66726D65)` — that is PCG64, one
//! [`numpy.random.Generator.uniform`] call per `u1` and `u2` sample. To land on
//! the same PCM bytes the Rust side has to reproduce that stream exactly, so
//! this is the LCG and output function from numpy's own
//! `numpy/random/src/pcg64/pcg64.h` (Apache-2.0 / MIT, Melissa O'Neill &
//! Robert Kern):
//!
//! ```c
//! pcg_setseq_128_step_r:   state = state * MULT + inc
//! pcg_output_xsl_rr_128_64: rotr64(hi ^ lo, state >> 122)
//! next_double:            (double)(r >> 11) * 0x1.0p-53
//! ```
//!
//! numpy steps the LCG *first* and applies the output function to the **new**
//! state; the rotation is the top six bits of the 128-bit state (`state >> 122`),
//! not the top five — the difference is visible in the first sample, which is why
//! [`tests::matches_numpy_oracle`] pins the whole first window by bit pattern.
//!
//! The seeded state is not derived here: it is the state numpy's
//! `default_rng(0x66726D65)` reaches through `SeedSequence`, read out of
//! `bit_generator.state` once and recorded as a constant.
//! [`tests::matches_numpy_oracle`] is the guard — if a future numpy changes
//! `SeedSequence`, that test fails loudly instead of quietly shifting the dither.

/// numpy's `PCG_DEFAULT_MULTIPLIER_128`.
pub const PCG64_MULTIPLIER: u128 = 0x2360ed051fc65da44385df649fccf645;

/// High half of the PCG64 state for `default_rng(0x66726D65)`.
pub const HYPREQ_DITHER_STATE_HI: u64 = 0xa6bc5a5f16892465;
/// Low half of that state.
pub const HYPREQ_DITHER_STATE_LO: u64 = 0xe72cdf103e6bff76;
/// High half of the corresponding increment.
pub const HYPREQ_DITHER_INC_HI: u64 = 0x98101544001ed01f;
/// Low half of that increment.
pub const HYPREQ_DITHER_INC_LO: u64 = 0x57110600c6939819;

/// The full 128-bit state numpy holds after seeding with `0x66726D65`.
pub const HYPREQ_DITHER_STATE: u128 =
    ((HYPREQ_DITHER_STATE_HI as u128) << 64) | HYPREQ_DITHER_STATE_LO as u128;
/// The full 128-bit increment that goes with it.
pub const HYPREQ_DITHER_INC: u128 =
    ((HYPREQ_DITHER_INC_HI as u128) << 64) | HYPREQ_DITHER_INC_LO as u128;

fn rotr64(v: u64, rot: u32) -> u64 {
    let rot = rot & 63;
    if rot == 0 {
        v
    } else {
        v.rotate_right(rot)
    }
}

/// `pcg_output_xsl_rr_128_64`.
#[inline]
fn xsl_rr(state: u128) -> u64 {
    let hi = (state >> 64) as u64;
    let lo = state as u64;
    rotr64(hi ^ lo, (state >> 122) as u32)
}

/// numpy's PCG64 bit generator, positioned at an arbitrary state.
#[derive(Clone, Copy, Debug)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    /// The stream `default_rng(0x66726D65)` produces — the dither source.
    pub fn hypreq_dither() -> Self {
        Self {
            state: HYPREQ_DITHER_STATE,
            inc: HYPREQ_DITHER_INC,
        }
    }

    /// Explicit state, for tests and for continuing a saved stream.
    pub fn from_state_inc(state: u128, inc: u128) -> Self {
        Self { state, inc }
    }

    /// `pcg64_next64`: step, then permute the new state.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(PCG64_MULTIPLIER)
            .wrapping_add(self.inc);
        xsl_rr(self.state)
    }

    /// numpy's `next_double`: the top 53 bits over 2^53, in `[0, 1)`.
    #[inline]
    pub fn next_double(&mut self) -> f64 {
        // 2^-53 as an exact power of two; a multiply rather than a divide so the
        // rounding is the same single operation numpy's C does.
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `uniform(-0.5, 0.5)`: `low + (high - low) * next_double` with
    /// `high - low == 1.0` exactly.
    #[inline]
    pub fn next_uniform_half(&mut self) -> f64 {
        -0.5 + self.next_double()
    }

    /// Burn `n` draws without producing them — used to line up with a Python
    /// array that was drawn in full and then sliced.
    pub fn advance_draws(&mut self, n: usize) {
        for _ in 0..n {
            self.next_u64();
        }
    }
}

/// TPDF dither matrix for `frames` frames × `channels` channels, laid out the
/// way `hypreq.py` draws it:
///
/// ```python
/// tpdf = rng.uniform(-0.5, 0.5, size=mat.shape) + rng.uniform(-0.5, 0.5, size=mat.shape)
/// ```
///
/// `mat` is the interleaved `(frames, channels)` buffer, so numpy fills it
/// row-major: the weight for frame `f`, channel `c` is flat index
/// `f * channels + c`, and the second array continues the same stream after the
/// first array's `frames * channels` draws. Shared across channels only by
/// position, not by frame index — writing it any other way shifts the whole
/// dither by one sample and changes every PCM byte.
pub fn tpdf_matrix(frames: usize, channels: usize) -> Vec<f64> {
    let n = frames * channels;
    let mut rng = Pcg64::hypreq_dither();
    let mut t = vec![0.0f64; n];
    for v in t.iter_mut() {
        *v = rng.next_uniform_half();
    }
    // The stream now sits exactly where numpy starts the second array.
    for v in t.iter_mut() {
        *v += rng.next_uniform_half();
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First twelve doubles of `default_rng(0x66726D65)`, as raw IEEE-754 bit
    /// patterns, captured from numpy 2.x: the first six are one
    /// `uniform(-0.5, 0.5, 6)` call and the next six the call that follows it.
    /// Bit patterns rather than decimal text so the comparison is exact.
    const ORACLE_UNIFORM_BITS: [u64; 12] = [
        0x3fda85f4b7d4d20a,
        0xbf97173fda408840,
        0xbfdc38ab8855c806,
        0xbfd8033ba50a65a4,
        0xbfd16b3f96a65884,
        0x3fc757780b5bc72c,
        0x3fd276ac4790d9f6,
        0xbfd8d65d2b671c92,
        0x3f979fc6ae901220,
        0xbfd60bee19ee74ce,
        0x3fc943bf32713ef4,
        0x3fb76d7af6c75140,
    ];

    /// And the raw uint64 stream, first eight of `random_raw(8)`.
    const ORACLE_RAW: [u64; 8] = [
        0xea17d2df53482c17,
        0x7a3a30096fddf241,
        0x0f1d51dea8dfe8b2,
        0x1ff3116bd669777f,
        0x3a5301a5669df0a2,
        0xaeaef016b78e5beb,
        0xc9dab11e4367decb,
        0x1ca68b52638dbc7c,
    ];

    #[test]
    fn matches_numpy_oracle() {
        let mut rng = Pcg64::hypreq_dither();
        for (i, want) in ORACLE_RAW.iter().enumerate() {
            assert_eq!(rng.next_u64(), *want, "raw draw {i}");
        }
        let mut rng = Pcg64::hypreq_dither();
        for (i, want) in ORACLE_UNIFORM_BITS.iter().enumerate() {
            let got = rng.next_uniform_half();
            assert_eq!(got.to_bits(), *want, "uniform draw {i}");
        }
    }

    /// `rng.uniform(-0.5,0.5,size=(4,2)) + rng.uniform(-0.5,0.5,size=(4,2))` from
    /// the oracle, flattened row-major, as bit patterns.
    const ORACLE_TPDF_BITS: [u64; 8] = [
        0x3fdbfff122bdd32c,
        0xbfd77d6217927d52,
        0xbfcf2d97de3a5118,
        0xbfd227dce7589154,
        0xbfe594588a2ae6d6,
        0x3fd844cab99fd702,
        0x3fe8d07313e25333,
        0xbfc66ab533a44fb4,
    ];

    #[test]
    fn tpdf_matrix_matches_numpy_layout() {
        let t = tpdf_matrix(4, 2);
        assert_eq!(t.len(), 8);
        for (i, (v, want)) in t.iter().zip(ORACLE_TPDF_BITS.iter()).enumerate() {
            assert_eq!(v.to_bits(), *want, "tpdf flat index {i}");
        }
    }

    #[test]
    fn tpdf_is_in_range_and_triangular() {
        let t = tpdf_matrix(2048, 2);
        assert!(t.iter().all(|v| *v >= -1.0 && *v <= 1.0));
        let mean = t.iter().sum::<f64>() / t.len() as f64;
        assert!(mean.abs() < 0.02, "mean {mean}");
        let centre = t.iter().filter(|v| v.abs() < 0.1).count();
        let edge = t.iter().filter(|v| **v > 0.8 || **v < -0.8).count();
        assert!(
            centre > 4 * edge,
            "triangular shape expected, centre {centre} edge {edge}"
        );
    }
}
