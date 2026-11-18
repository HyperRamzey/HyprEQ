//! Peak meters on the classic mixer-console scale: instant attack, a 0.30 s
//! peak-hold release, and a 0…1 parameter curve that puts -60 dBFS at 0 and 0 dBFS
//! at 1 — the convention a VST2 host reads, kept so that a level quoted from a log
//! line means the same thing as one read off a meter plug-in.
//!
//! ```c
//! decay_for(n)   = exp(-n / (0.30 * sample_rate))     /* 0.30 s peak-hold release */
//! hold_level(p)  = peak > prev*decay ? peak : prev*decay  /* instant attack */
//! level_to_param = (clamp(20*log10(v), -60, 0) + 60) / 60
//! ```
//!
//! The only deliberate deviation from that curve is at the -60 dBFS floor, and it is
//! documented on [`level_to_param`].
//!
//! HyprEQ meters a planar buffer of arbitrary channel count, so the meter count is a
//! constructor argument ([`Meters::with_channels`]) and samples arrive as `f64`
//! ([`Meters::observe_planar`]). One writer, many readers, so each meter is a single
//! `AtomicU32` holding `f32` bits — no lock, and a torn read is impossible.
//!
//! The f32 storage width is deliberate and load-bearing: every stored level is rounded
//! through `f32`, which is where the ≈1.2e-7 m^-1 slope of `param_to_db` near the top
//! of the range comes from. Samples are `f64` everywhere else in this crate; the meter
//! is a *display* quantity, and matching the reference behaviour matters more here than
//! the extra 29 bits of dynamic range would.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering as AtomicOrd};

/// Peak-hold release time constant, seconds. Matches `METER_FALL_S`.
pub const METER_FALL_S: f64 = 0.30;

/// Live meter state, shared between the audio/callback thread and the UI.
#[derive(Debug, Default)]
pub struct Meters {
    values: Vec<AtomicU32>,
    /// `f32` sample rate and last block size, packed as bits, for `tick()`.
    rate_bits: AtomicU32,
    block: AtomicU64,
    frames: AtomicU64,
}

impl Meters {
    /// `channels` meters, for a caller that is metering a planar buffer
    /// directly instead of the plugin's slot table.
    pub fn with_channels(channels: usize) -> Self {
        Self {
            values: (0..channels).map(|_| AtomicU32::new(0)).collect(),
            rate_bits: AtomicU32::new(0),
            block: AtomicU64::new(0),
            frames: AtomicU64::new(0),
        }
    }

    /// Number of meters this instance holds.
    #[cfg(test)]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.values.len()
    }

    /// Tell the meters the working rate and the block size they are fed, which
    /// is what sets the decay per call.
    pub fn set_format(&self, sample_rate: f32, block: usize) {
        self.rate_bits
            .store(sample_rate.to_bits(), AtomicOrd::Relaxed);
        self.block.store(block as u64, AtomicOrd::Relaxed);
    }

    fn decay(&self) -> f32 {
        let sr = f32::from_bits(self.rate_bits.load(AtomicOrd::Relaxed));
        let sr = if sr > 0.0 { sr as f64 } else { 48000.0 };
        let n = self.block.load(AtomicOrd::Relaxed).max(1) as f64;
        (-(n / (METER_FALL_S * sr))).exp() as f32
    }

    fn hold(&self, index: usize, peak: f32) {
        let Some(slot) = self.values.get(index) else {
            return;
        };
        let prev = f32::from_bits(slot.load(AtomicOrd::Relaxed));
        // A non-finite previous value is decayed to silence rather than multiplied: inf ×
        // 0.99 is inf, so one bad sample used to latch the bar at full scale for the rest
        // of the session, and only reset() would free it.
        //
        // The reference writes `float(np.float32(prev * self.decay()))`, which multiplies
        // in f64 and then rounds to f32. That is not a wider arithmetic than the `f32`
        // multiply here: the product of two binary32 significands needs at most 48 bits,
        // which a binary64 holds exactly, so the f64 path rounds the same exact value
        // once — the identical result, including into subnormals. `hold` staying in `f32`
        // is therefore parity, not a narrowing; the test `the_wider_multiply_is_a_no_op`
        // pins it against the f64 form bit for bit.
        let fall = if prev.is_finite() {
            prev * self.decay()
        } else {
            0.0
        };
        let peak = if peak.is_finite() { peak } else { f32::MAX };
        let v = if peak > fall { peak } else { fall };
        slot.store(v.to_bits(), AtomicOrd::Relaxed);
    }

    /// Peak of one buffer, in the `f32` domain exactly as the plugin measured it
    /// (`(float)fabs((double)x[i])`).
    fn peak_of(buf: &[f64], n: usize) -> f32 {
        let mut peak = 0.0f32;
        for &x in buf.iter().take(n) {
            // Clamped, not cast. The plugin this reproduces stored the peak in a float too,
            // but its samples came from a float buffer, so it could not meet an f64 above
            // f32::MAX; a bare `as f32` of one yields +inf, which [`Self::hold`] would then
            // latch. A non-finite sample is skipped, not clamped: `f64::min` returns the
            // other operand when one is NaN, so clamping turned a NaN into f32::MAX.
            let a = x.abs();
            if !a.is_finite() {
                continue;
            }
            let v = a.min(f64::from(f32::MAX)) as f32;
            if v > peak {
                peak = v;
            }
        }
        peak
    }

    /// Feed one block straight from a planar buffer, one meter per channel.
    ///
    /// This is the path HyprEQ's `measure_signal` uses: no slot table, no
    /// output pair, just the channels of the file being measured.
    pub fn observe_planar(&self, planar: &[&[f64]], n: usize) {
        for (c, buf) in planar.iter().enumerate() {
            if c >= self.values.len() {
                break;
            }
            self.hold(c, Self::peak_of(buf, n));
        }
        self.frames.fetch_add(n as u64, AtomicOrd::Relaxed);
    }

    /// Linear peak, 0.0 .. >1.0 (above 1.0 means the encoder will clip).
    pub fn peak(&self, index: usize) -> f32 {
        self.values
            .get(index)
            .map(|v| f32::from_bits(v.load(AtomicOrd::Relaxed)))
            .unwrap_or(0.0)
    }

    pub fn peaks(&self) -> Vec<f32> {
        (0..self.values.len()).map(|i| self.peak(i)).collect()
    }

    /// Blocks observed so far. Public for the same reason as [`Self::channels`]: a reading is
    /// only interpretable next to the integration window that produced it, and a test that sets
    /// up a window has to be able to say how many frames went through it.
    #[cfg(test)]
    pub fn frames(&self) -> u64 {
        self.frames.load(AtomicOrd::Relaxed)
    }

    /// Drop the latched peaks and the block count, keeping the format.
    ///
    /// Public so a caller can reuse one `Meters` across files — and so the tests below can prove
    /// that a silent block and a reset behave differently, which is the difference between a
    /// peak-hold meter and a decaying one.
    #[cfg(test)]
    pub fn reset(&self) {
        for v in &self.values {
            v.store(0, AtomicOrd::Relaxed);
        }
        self.frames.store(0, AtomicOrd::Relaxed);
    }
}

/// Meter value -> normalised dB parameter: `-60..0 dB` mapped onto `0..1`.
///
/// The VST plugin computed this as `20*log10(v)` and clamped the result to
/// `[-60, 0]`. That has a defect worth not inheriting: `log10(0.001)` is
/// `-2.9999999999999996`, so a signal sitting exactly at the -60 dB floor read
/// `6.9e-9` instead of `0`, and the floor was therefore unreachable — a meter
/// that never quite rests. The floor and ceiling are defined here by what they
/// *mean* (`-60 dBFS` is exactly `0.001`, `0 dBFS` is exactly `1.0`), which
/// reaches the same endpoints exactly and is monotone between them.
///
/// Only the meter scale changed. Nothing in the DSP is adjusted this way:
/// reproducing the module's transfer function is the point of the port.
#[must_use]
pub fn level_to_param(v: f32) -> f32 {
    // NaN has no ordering, and `partial_cmp` says so instead of quietly answering `false`
    // twice (`v <= 0.001` would also be false for NaN, and would fall through to the log
    // below). A meter has no honest reading for NaN; the floor is the safe one, and the
    // value is still visible in the numeric report.
    if v.partial_cmp(&0.001).is_none_or(|o| o.is_le()) {
        return 0.0; // at or below the -60 dBFS floor, and anything unmeasurable
    }
    if v >= 1.0 {
        return 1.0; // full scale and above
    }
    let db = 20.0 * (v as f64).log10();
    ((db + 60.0) / 60.0) as f32
}

/// Inverse, for axis labels.
#[cfg(test)]
#[must_use]
pub fn param_to_db(p: f32) -> f64 {
    (p as f64) * 60.0 - 60.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_matches_the_vst_formula() {
        let m = Meters::with_channels(2);
        m.set_format(48000.0, 1024);
        // exp(-1024 / (0.30 * 48000))
        let want = (-(1024.0f64 / (METER_FALL_S * 48000.0))).exp() as f32;
        assert_eq!(m.decay().to_bits(), want.to_bits());
    }

    /// The reference multiplies in `f64` and rounds once to `f32`; this port multiplies
    /// in `f32`. Those are the same operation, because a binary32 × binary32 product fits
    /// a binary64 significand exactly — so the `f64` step rounds nothing. Pinned because
    /// "widen the multiply to match the Python" would be a no-op at best and a double
    /// rounding bug at worst if someone changed the accumulation later.
    #[test]
    fn the_wider_multiply_is_a_no_op() {
        let mut state = 0x243F_6A88_85A3_08D3u64; // splitmix64, fixed seed
        let mut next = || {
            state = state
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(0x4B7F_1F0D_DD1B_4C3B);
            let u = ((state >> ((state >> 59) as u32 & 5)) | 1) >> 1;
            f32::from_bits(u as u32 >> 8)
        };
        let mut checked = 0usize;
        for _ in 0..100_000 {
            let a = next().abs();
            let b = next().abs();
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            let narrow = a * b;
            let wide = (f64::from(a) * f64::from(b)) as f32;
            assert_eq!(narrow.to_bits(), wide.to_bits(), "{a:e} * {b:e}");
            checked += 1;
        }
        assert!(checked > 50_000, "the sweep degenerated: {checked}");

        // The decay factors the meter actually uses, at the rates it sees them.
        for (rate, block) in [(48_000.0, 1024.0), (44_100.0, 512.0), (384_000.0, 64.0)] {
            let d = (-(block / (METER_FALL_S * rate))).exp();
            for prev in [1.0f32, 0.5, f32::MIN_POSITIVE, 1e-38] {
                assert_eq!(
                    (prev * (d as f32)).to_bits(),
                    ((f64::from(prev) * f64::from(d as f32)) as f32).to_bits(),
                    "decay at {rate} Hz / {block} frames"
                );
            }
        }
    }

    #[test]
    fn attack_is_instant_release_is_exponential() {
        let m = Meters::with_channels(2);
        m.set_format(48000.0, 480);
        m.hold(0, 0.5);
        assert_eq!(m.peak(0), 0.5);
        // A quieter block must not pull the meter down instantly: it falls by
        // exactly exp(-n / (0.30 * sr)) per block, which at 480 frames is 3.3 %.
        m.hold(0, 0.1);
        let one_step = 0.5 * (-(480.0f64 / (METER_FALL_S * 48000.0))).exp() as f32;
        assert_eq!(
            m.peak(0).to_bits(),
            one_step.to_bits(),
            "release must follow the VST curve"
        );
        m.hold(0, 0.9); // above -> instant
        assert_eq!(m.peak(0), 0.9);
    }

    #[test]
    fn param_scale_matches_plugin() {
        assert_eq!(level_to_param(1.0), 1.0);
        assert_eq!(level_to_param(0.0), 0.0);
        // level_to_param(v) = (clamp(20*log10(v), -60, 0) + 60) / 60, so half
        // scale (-6.0206 dB) sits at 0.8997, not near zero.
        assert!(
            (level_to_param(0.5) - 0.8997).abs() < 0.0002,
            "{}",
            level_to_param(0.5)
        );
        // The floor is reachable and exact, which is the behaviour the plugin
        // intended and could not express.
        assert_eq!(level_to_param(0.001), 0.0, "-60 dBFS reads zero");
        assert_eq!(level_to_param(0.0000001), 0.0);
        assert_eq!(level_to_param(0.0), 0.0);
        // overshoot past full scale clamps, like the plugin's parameter
        assert_eq!(level_to_param(2.0), 1.0);
        assert!(level_to_param(0.9) < level_to_param(0.95), "monotone");
        // the inverse of the -60..0 dB scale
        assert_eq!(param_to_db(0.5), -30.0);
        assert_eq!(param_to_db(0.0), -60.0);
        assert_eq!(param_to_db(1.0), 0.0);
    }

    /// A non-finite sample is not a level: it must neither become the peak nor hide the
    /// finite samples around it, and an `f64` beyond `f32` range is clamped, not skipped.
    #[test]
    fn non_finite_and_out_of_range_samples_are_handled() {
        let m = Meters::with_channels(2);
        m.set_format(48000.0, 4);
        let l = [f64::NAN, 0.25, 0.0, 0.0];
        let r = [0.5f64, f64::INFINITY, f64::NEG_INFINITY, 0.0];
        m.observe_planar(&[&l, &r], 4);
        assert_eq!(m.peak(0), 0.25, "a NaN must not become the peak");
        assert_eq!(
            m.peak(1),
            0.5,
            "an infinity must not hide the finite sample"
        );

        m.reset();
        let big = [1e300f64, 0.0, 0.0, 0.0];
        m.observe_planar(&[&big, &big], 4);
        assert_eq!(m.peak(0), f32::MAX, "beyond f32 range clamps to full scale");
    }

    #[test]
    fn a_latched_non_finite_meter_releases_instead_of_sticking() {
        let m = Meters::with_channels(1);
        m.set_format(48000.0, 4);
        // A sample can no longer put a non-finite value into a slot: `peak_of` skips them
        // and `hold` clamps the request. This state is set directly because it is what a
        // latched bar used to sit at for the rest of the session — inf × decay is inf, and
        // only reset() freed it. The guard in `hold` is what lets a silent block clear it.
        m.values[0].store(f32::INFINITY.to_bits(), AtomicOrd::Relaxed);
        m.hold(0, 0.0);
        assert_eq!(m.peak(0), 0.0, "a silent block clears a latched bar");

        // And an infinite peak request lands on full scale, which is finite and decays.
        m.hold(0, f32::INFINITY);
        assert_eq!(m.peak(0), f32::MAX, "an infinite request clamps");
        m.hold(0, 0.0);
        assert!(
            m.peak(0) < f32::MAX,
            "and the bar then falls: {}",
            m.peak(0)
        );
    }

    #[test]
    fn planar_observation_meters_one_channel_each() {
        let m = Meters::with_channels(2);
        m.set_format(48000.0, 4);
        let l = [0.25f64, -0.5, 0.0, 0.0];
        let r = [-0.125f64, 0.0, 0.0, 0.0];
        m.observe_planar(&[&l, &r], 4);
        assert_eq!(m.peak(0), 0.5);
        assert_eq!(m.peak(1), 0.125);
        assert_eq!(m.frames(), 4);
        assert_eq!(m.channels(), 2);
    }
}
