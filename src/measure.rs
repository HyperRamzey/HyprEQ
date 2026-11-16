//! Signal measurements reported for every buffer the tool touches, plus the
//! deterministic convolver probe that exercises them.
//!
//! [`measure_signal`] runs four instruments over the same planar `f64` buffer and
//! reports them side by side, because the point of the report is the *spread*
//! between them:
//!
//! * the requested over-sampling ratio ([`crate::tp::TruePeak`], x16 by default) —
//!   the number the limiter's ceiling is judged against;
//! * the ITU-R BS.1770-5 x4 meter, which is what a compliance document would
//!   quote, and which reads lower on material with a crest sitting between grid
//!   points;
//! * the plain sample peak and RMS, i.e. what a wave editor shows;
//! * the console-scale 0.30 s peak-hold meter ([`crate::meter::Meters`]), which is
//!   what a plug-in GUI would have painted.
//!
//! Two details are load-bearing for parity with `hypreq.py`:
//!
//! * the meters are fed in **1024-frame blocks** with `set_format` called per
//!   block. The VST release curve is evaluated per observation call, so the block
//!   size is part of the measurement, not an optimisation;
//! * RMS is computed per channel and then combined as
//!   `sqrt(mean(rms_c²))`, which for equal channels is the total power — not the
//!   mean of the dB values.
//!
//! An empty buffer measures as "no signal": peaks 0.0 and levels `-inf`, with the
//! crest terms forced to 0.0 rather than left as `inf - inf`.

use crate::tp::{db_from_peak, Detector, Ratio, TruePeak};

/// The measurement block size for the peak-hold meters.
pub const METER_BLOCK: usize = 1024;

/// Everything [`measure_signal`] learns about one buffer.
#[derive(Clone, Debug)]
pub struct SignalMeasurements {
    pub channels: usize,
    pub frames: usize,
    pub fs: u32,
    pub duration_s: f64,
    pub ratio: Ratio,
    pub sample_peak_lin: f64,
    pub sample_peak_dbfs: f64,
    pub per_channel_sample_dbfs: Vec<f64>,
    pub true_peak_lin: f64,
    pub true_peak_dbtp: f64,
    pub per_channel_dbtp: Vec<f64>,
    pub itu_x4_dbtp: f64,
    pub itu_x4_per_channel_dbtp: Vec<f64>,
    pub rms_dbfs: f64,
    pub per_channel_rms_dbfs: Vec<f64>,
    pub crest_tp_db: f64,
    pub crest_sample_db: f64,
    pub meter_peak_hold_lin: f64,
    pub meter_peak_hold_dbfs: f64,
    pub meter_param: f64,
}

impl SignalMeasurements {
    /// The "no signal" record: peaks 0, levels -inf, crests 0.
    fn empty(channels: usize, fs: u32, ratio: Ratio) -> Self {
        Self {
            channels,
            frames: 0,
            fs,
            duration_s: 0.0,
            ratio,
            sample_peak_lin: 0.0,
            sample_peak_dbfs: f64::NEG_INFINITY,
            per_channel_sample_dbfs: Vec::new(),
            true_peak_lin: 0.0,
            true_peak_dbtp: f64::NEG_INFINITY,
            per_channel_dbtp: Vec::new(),
            itu_x4_dbtp: f64::NEG_INFINITY,
            itu_x4_per_channel_dbtp: Vec::new(),
            rms_dbfs: f64::NEG_INFINITY,
            per_channel_rms_dbfs: Vec::new(),
            crest_tp_db: 0.0,
            crest_sample_db: 0.0,
            meter_peak_hold_lin: 0.0,
            meter_peak_hold_dbfs: f64::NEG_INFINITY,
            meter_param: 0.0,
        }
    }
}

/// Measure planar audio: true peak at `ratio` and at the ITU x4 bank, sample peak,
/// RMS, and the 0.30 s peak-hold meters.
pub fn measure_signal(planar: &[Vec<f64>], fs: u32, ratio: Ratio) -> SignalMeasurements {
    measure_signal_block(planar, fs, ratio, METER_BLOCK)
}

/// [`measure_signal`] with the meter observation block size exposed; the default
/// is [`METER_BLOCK`], and the Python uses the same value.
pub fn measure_signal_block(
    planar: &[Vec<f64>],
    fs: u32,
    ratio: Ratio,
    block_size: usize,
) -> SignalMeasurements {
    let nch = planar.len();
    // Metering walks the channels in lockstep, so it runs over the length they all share.
    // Per-channel peak and RMS below still see each channel's own samples.
    let frames = planar.iter().map(|c| c.len()).min().unwrap_or(0);
    if nch == 0 || frames == 0 {
        return SignalMeasurements::empty(nch, fs, ratio);
    }

    let mut tp_main = TruePeak::new(ratio, nch);
    let mut tp_itu = TruePeak::new(Ratio::X4, nch);
    let meters = crate::meter::Meters::with_channels(nch);

    let mut sp_ch = Vec::with_capacity(nch);
    let mut rms_ch_lin = Vec::with_capacity(nch);
    for (c, ch) in planar.iter().enumerate() {
        sp_ch.push(ch.iter().fold(0.0f64, |a, &v| a.max(v.abs())));
        // Sum of squares in f64; a 24-bit/48 kHz minute of audio is ~2.9e9 terms,
        // and the running sum stays well inside the range where f64 is exact for
        // the magnitudes involved.
        let mean_sq = ch.iter().fold(0.0f64, |a, &v| a + v * v) / ch.len() as f64;
        rms_ch_lin.push(mean_sq.sqrt());
        tp_main.push_ch(c, ch);
        tp_itu.push_ch(c, ch);
    }
    tp_main.drain();
    tp_itu.drain();

    // The meter model converts sample counts to time, so it needs a rate. Without one the
    // integration window is a division by zero; the peak/RMS/TP figures below are pure
    // sample statistics and stay valid, so only the meter block is skipped.
    if fs > 0 {
        for start in (0..frames).step_by(block_size.max(1)) {
            let end = (start + block_size).min(frames);
            let blk = end - start;
            let views: Vec<&[f64]> = planar.iter().map(|c| &c[start..end]).collect();
            meters.set_format(fs as f32, blk);
            meters.observe_planar(&views, blk);
        }
    }

    let sp_lin = sp_ch.iter().copied().fold(0.0f64, f64::max);
    let sp_db = db_from_peak(sp_lin);
    let tp_lin = tp_main.peak();
    let tp_db = tp_main.dbtp();
    let itu_db = tp_itu.dbtp();
    let rms_all = (rms_ch_lin.iter().map(|r| r * r).sum::<f64>() / nch as f64).sqrt();
    let rms_db = db_from_peak(rms_all);
    let hold_lin = meters.peaks().iter().copied().fold(0.0f32, f32::max) as f64;
    let hold_db = db_from_peak(hold_lin);
    let hold_param = crate::meter::level_to_param(hold_lin as f32) as f64;

    SignalMeasurements {
        channels: nch,
        frames,
        fs,
        duration_s: frames as f64 / fs as f64,
        ratio,
        sample_peak_lin: sp_lin,
        sample_peak_dbfs: sp_db,
        per_channel_sample_dbfs: sp_ch.iter().map(|&v| db_from_peak(v)).collect(),
        true_peak_lin: tp_lin,
        true_peak_dbtp: tp_db,
        per_channel_dbtp: tp_main.per_channel_dbtp(),
        itu_x4_dbtp: itu_db,
        itu_x4_per_channel_dbtp: tp_itu.per_channel_dbtp(),
        rms_dbfs: rms_db,
        per_channel_rms_dbfs: rms_ch_lin.iter().map(|&v| db_from_peak(v)).collect(),
        crest_tp_db: if tp_db.is_finite() && rms_db.is_finite() {
            tp_db - rms_db
        } else {
            0.0
        },
        crest_sample_db: if sp_db.is_finite() && rms_db.is_finite() {
            sp_db - rms_db
        } else {
            0.0
        },
        meter_peak_hold_lin: hold_lin,
        meter_peak_hold_dbfs: hold_db,
        meter_param: hold_param,
    }
}

/// `+0.42` / `-3.10` at a fixed number of decimals — the Python's `{:+.Nf}`.
///
/// `-inf` is a real measurement here — silence measures as `-inf dBFS` — and
/// NumPy formats an infinite level as the bare word with its sign, no decimals:
/// `f"{float('-inf'):+.3f}"` is `-inf`, not `-inf.000`. Rust prints `-inf` for
/// `{:+.3}` of a non-finite value too, but only by coincidence of its own float
/// formatting, so the infinities are named here rather than leaned on.
pub fn sgn(v: f64, dp: usize) -> String {
    if v == f64::NEG_INFINITY {
        return "-inf".to_string();
    }
    if v == f64::INFINITY {
        return "+inf".to_string();
    }
    format!("{v:+.*}", dp)
}

/// Print the four-instrument block the report shows for each buffer.
pub fn print_signal_measurements(label: &str, m: &SignalMeasurements) {
    let ch_tp = join_signed(&m.per_channel_dbtp, 2);
    let ch_itu = join_signed(&m.itu_x4_per_channel_dbtp, 2);
    println!("  {label}:");
    println!(
        "    True Peak ({})  : {} dBTP{} | ITU BS.1770-5 (x4): {} dBTP{}",
        m.ratio.label(),
        sgn(m.true_peak_dbtp, 3),
        if m.channels > 1 {
            format!(" [{ch_tp}]")
        } else {
            String::new()
        },
        sgn(m.itu_x4_dbtp, 3),
        if m.channels > 1 {
            format!(" [{ch_itu}]")
        } else {
            String::new()
        },
    );
    println!(
        "    Sample Peak / RMS : {} dBFS | {} dBFS RMS | Crest: {:.2} dB (TP) / {:.2} dB (sample)",
        sgn(m.sample_peak_dbfs, 3),
        sgn(m.rms_dbfs, 2),
        m.crest_tp_db,
        m.crest_sample_db,
    );
    println!(
        "    Meter (0.30s Hold): {} dBFS (param {:.4} on -60..0 dB scale | {} frames, {:.3} s)",
        sgn(m.meter_peak_hold_dbfs, 3),
        m.meter_param,
        crate::wav::comma_sep(m.frames as u64),
        m.duration_s,
    );
}

/// `", ".join(f"{v:+.2f}" for v in xs)`; an empty list joins to nothing.
fn join_signed(xs: &[f64], dp: usize) -> String {
    xs.iter()
        .map(|&v| sgn(v, dp))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The deterministic convolver probe: band-limited multi-tone plus a sinc burst,
/// normalised to 0.00 dBTP at x16.
///
/// Deterministic by construction — fixed frequencies, fixed phases, no RNG — so
/// the two implementations can be compared sample for sample and any difference
/// in the convolution or the limiter is arithmetic, not a different signal.
///
/// The content is chosen to be adversarial for a convolver-and-limiter chain
/// rather than pleasant to listen to:
///
/// * three tones, one mid (`min(4060, 0.35 fs)`), one high (`min(12 000, 0.43 fs)`
///   — close to the ITU band edge, where an interpolating meter's reading is worst)
///   and one at 150 Hz to carry energy through a bass-shelf EQ;
/// * a *zero-phase* sinc burst (windowed by a symmetric Hann) planted at the
///   midpoint, whose peak is already fully formed before the detector has seen the
///   samples on either side of it — the hardest thing you can ask a lookahead
///   limiter to survive;
/// * everything normalised so the x16 detector reads exactly 0 dBTP, which puts the
///   `-3 dBTP` ceiling to work immediately.
#[must_use]
pub fn build_convolver_probe(fs: u32, duration_s: f64) -> Vec<f64> {
    let n = ((fs as f64 * duration_s).round_ties_even() as usize).max(4096);
    let nyq_safe = 0.43 * fs as f64;
    let f_hi = 12000.0f64.min(nyq_safe);
    let f_mid = 4060.0f64.min(0.35 * fs as f64);

    let mut probe = vec![0.0f64; n];
    for (i, p) in probe.iter_mut().enumerate() {
        let t = i as f64 / fs as f64;
        *p = 0.55 * (2.0 * std::f64::consts::PI * f_mid * t + 0.30).sin()
            + 0.45 * (2.0 * std::f64::consts::PI * f_hi * t + std::f64::consts::PI / 4.0).sin()
            + 0.30 * (2.0 * std::f64::consts::PI * 150.0 * t).sin();
    }

    let burst_len = 1024.min(n / 4);
    let burst_start = n / 2;
    let fc = 0.42f64;
    let mut burst = vec![0.0f64; burst_len];
    for (i, b) in burst.iter_mut().enumerate() {
        let b_t = i as f64 - burst_len as f64 / 2.0;
        // np.where(b_t == 0.0, 2*fc, sin(2*pi*fc*b_t)/(pi*b_t)) — the sinc's DC
        // value, evaluated only at the exact centre sample.
        *b = if b_t == 0.0 {
            2.0 * fc
        } else {
            (2.0 * std::f64::consts::PI * fc * b_t).sin() / (std::f64::consts::PI * b_t)
        };
        // np.hanning: symmetric Hann, 0.5 - 0.5*cos(2*pi*i/(M-1)), with M-1 == 0
        // guarded (a one-sample window is 0.0 in NumPy).
        let w = if burst_len <= 1 {
            0.0
        } else {
            0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (burst_len - 1) as f64).cos()
        };
        *b *= w;
    }
    let bmax = burst.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-12);
    for (i, b) in burst.iter().enumerate() {
        probe[burst_start + i] += 1.25 * b / bmax;
    }

    // Normalise on the x16 detector's reading, after draining its filter.
    let mut d = Detector::new(Ratio::X16);
    d.push_block(&probe);
    let drain = Ratio::X16.group_delay().ceil() as usize + 1;
    d.push_block(&vec![0.0f64; drain]);
    let pk = d.peak().max(1e-12);
    for p in &mut probe {
        *p /= pk;
    }
    probe
}

/// Print the probe's own measurement block (used by `--measure-probe`).
pub fn print_probe_measurements(probe: &[f64], fs: u32, ratio: Ratio) {
    let m = measure_signal(&[probe.to_vec()], fs, ratio);
    print_signal_measurements("Convolver Probe (0 dBTP @ x16)", &m);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(fs: u32, secs: f64, f: f64, amp: f64) -> Vec<f64> {
        let n = (fs as f64 * secs) as usize;
        (0..n)
            .map(|i| amp * (2.0 * std::f64::consts::PI * f * i as f64 / fs as f64).sin())
            .collect()
    }

    #[test]
    fn empty_buffer_measures_as_no_signal() {
        let m = measure_signal(&[], 48000, Ratio::X16);
        assert_eq!(m.channels, 0);
        assert_eq!(m.frames, 0);
        assert_eq!(m.sample_peak_dbfs, f64::NEG_INFINITY);
        assert_eq!(m.true_peak_dbtp, f64::NEG_INFINITY);
        assert_eq!(m.crest_tp_db, 0.0);
        assert_eq!(m.meter_param, 0.0);

        let m = measure_signal(&[vec![], vec![]], 48000, Ratio::X16);
        assert_eq!(m.channels, 2);
        assert_eq!(m.frames, 0);
    }

    #[test]
    fn full_scale_sine_reads_zero_dbfs_and_a_positive_crest() {
        let s = sig(48000, 0.5, 997.0, 1.0);
        let m = measure_signal(&[s], 48000, Ratio::X16);
        assert!(
            (m.sample_peak_dbfs - 0.0).abs() < 1e-9,
            "{}",
            m.sample_peak_dbfs
        );
        assert!(m.true_peak_dbtp > -0.05, "tp {}", m.true_peak_dbtp);
        // A sine's RMS is -3.01 dBFS, and the crest is the difference.
        assert!((m.rms_dbfs + 3.0103).abs() < 0.01, "rms {}", m.rms_dbfs);
        assert!((m.crest_tp_db - (m.true_peak_dbtp - m.rms_dbfs)).abs() < 1e-12);
        assert!((m.crest_sample_db - 3.0103).abs() < 0.02);
        assert!(m.meter_peak_hold_dbfs > -0.05);
        assert!(m.meter_param > 0.999, "param {}", m.meter_param);
    }

    #[test]
    fn per_channel_columns_are_reported_for_stereo() {
        let l = sig(48000, 0.2, 500.0, 0.9);
        let r = sig(48000, 0.2, 500.0, 0.1);
        let m = measure_signal(&[l, r], 48000, Ratio::X16);
        assert_eq!(m.channels, 2);
        assert_eq!(m.per_channel_dbtp.len(), 2);
        assert_eq!(m.itu_x4_per_channel_dbtp.len(), 2);
        assert_eq!(m.per_channel_sample_dbfs.len(), 2);
        assert_eq!(m.per_channel_rms_dbfs.len(), 2);
        // Left is 20 dB above right, so the linked peak is the left one.
        assert!(m.per_channel_dbtp[0] > m.per_channel_dbtp[1] + 15.0);
        assert_eq!(m.sample_peak_dbfs, m.per_channel_sample_dbfs[0]);
        // Combined RMS is the mean *power*, not the mean level: 3 dB below the loud
        // channel here (the quiet one still contributes half the power's worth).
        let lin = |db: f64| 10f64.powf(db / 10.0);
        let want = 10f64
            * ((lin(m.per_channel_rms_dbfs[0]) + lin(m.per_channel_rms_dbfs[1])) / 2.0).log10();
        assert!(
            (m.rms_dbfs - want).abs() < 1e-9,
            "combined {} vs power mean {}",
            m.rms_dbfs,
            want
        );
        assert!(m.rms_dbfs < m.per_channel_rms_dbfs[0] - 2.0);
    }

    #[test]
    fn itu_x4_and_x16_agree_on_band_limited_content() {
        // 1 kHz stays deep inside both banks' pass-bands, so the two ratios must
        // read the same inter-sample peak to within a few hundredths of a dB.
        let s = sig(48000, 0.25, 1000.0, 0.5);
        let m = measure_signal(&[s.clone()], 48000, Ratio::X16);
        assert!(
            (m.true_peak_dbtp - m.itu_x4_dbtp).abs() < 0.05,
            "x16 {} vs x4 {}",
            m.true_peak_dbtp,
            m.itu_x4_dbtp
        );
        // And both sit at or above the sample peak: the whole point of TP metering.
        assert!(m.true_peak_dbtp >= m.sample_peak_dbfs - 1e-9);
    }

    #[test]
    fn meter_hold_follows_the_loudest_channel() {
        let l = sig(48000, 1.0, 400.0, 0.5);
        let r = sig(48000, 1.0, 400.0, 0.05);
        let m = measure_signal(&[l, r], 48000, Ratio::X16);
        assert!((m.meter_peak_hold_dbfs - m.per_channel_sample_dbfs[0]).abs() < 0.02);
        // The console scale is not linear in dB: the reference `level_to_param(0.5)`
        // for a -6 dBFS hold is 0.8996566534042358, nowhere near the 0.5 a naive
        // -60..0 dB mapping would give.
        assert!(
            (m.meter_param - 0.899_656_653_404_235_8).abs() < 1e-6,
            "param {}",
            m.meter_param
        );
    }

    #[test]
    fn probe_is_deterministic_and_zero_dbtp_at_x16() {
        for fs in [44100u32, 48000, 96000] {
            let a = build_convolver_probe(fs, 0.5);
            let b = build_convolver_probe(fs, 0.5);
            assert_eq!(a, b, "probe must be reproducible at {fs}");
            let m = measure_signal(&[a.clone()], fs, Ratio::X16);
            assert!(
                (m.true_peak_dbtp - 0.0).abs() < 1e-9,
                "fs {fs}: {} dBTP",
                m.true_peak_dbtp
            );
            // Band-limited to <= 0.45 fs, so the meter's band caveat does not apply.
            assert!(m.sample_peak_dbfs <= 0.001);
            assert!(m.frames >= 4096);
        }
    }

    #[test]
    fn probe_peaks_at_the_burst_centre() {
        let fs = 48000u32;
        let p = build_convolver_probe(fs, 0.5);
        let n = p.len();
        let mid = n / 2;
        // Reference: the NumPy probe for 0.5 s at 48 kHz is 24000 samples long and
        // its largest sample is 0.9936752385603075, sitting 512 samples past the
        // midpoint where the sinc burst crosses zero offset. The three tone bed
        // sums near full scale, so amplitude alone says little — the *location* of
        // the peak is what proves the burst is present and where the convolver's
        // group delay will be read from.
        assert_eq!(n, 24000);
        let (argmax, peak) = p.iter().enumerate().fold((0usize, 0.0f64), |acc, (i, &v)| {
            if v.abs() > acc.1 {
                (i, v.abs())
            } else {
                acc
            }
        });
        assert_eq!(argmax, mid + 512, "burst centre");
        assert!(
            (peak - 0.993_675_238_560_307_5).abs() < 1e-9,
            "burst peak {peak}"
        );
        assert!(p.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn signed_formatting_matches_python() {
        assert_eq!(sgn(0.42, 3), "+0.420");
        assert_eq!(sgn(-0.42, 3), "-0.420");
        assert_eq!(sgn(0.0, 2), "+0.00");
        // NumPy: f"{float('-inf'):+.3f}" == "-inf" — no decimals on an infinity.
        assert_eq!(sgn(f64::NEG_INFINITY, 3), "-inf");
        assert_eq!(sgn(f64::INFINITY, 3), "+inf");
        assert_eq!(sgn(f64::NEG_INFINITY, 2), "-inf");
        assert_eq!(join_signed(&[1.0, -2.0], 2), "+1.00, -2.00");
        assert_eq!(join_signed(&[], 2), "");
    }

    #[test]
    fn meter_hold_releases_over_the_tail_of_the_file() {
        // The meters are an instant-attack, 0.30 s exponential-release peak hold, and
        // `measure_signal` feeds them in blocks, so the column reports where the hold
        // has fallen to by the end of the file — not the maximum of the file.
        let mut early = vec![0.0f64; 48000];
        for i in 20_000..20_100 {
            early[i] = 0.999;
        }
        let m = measure_signal(&[early.clone()], 48000, Ratio::X16);
        assert!(
            m.true_peak_dbtp > -0.05,
            "the burst is there: {}",
            m.true_peak_dbtp
        );
        assert!(
            m.meter_peak_hold_dbfs < -10.0,
            "a burst 0.58 s from the end must have released, got {}",
            m.meter_peak_hold_dbfs
        );

        // The same burst at the very end is still held at full scale.
        let mut late = vec![0.0f64; 48000];
        let n = late.len();
        for i in n - 100..n {
            late[i] = 0.999;
        }
        let m2 = measure_signal(&[late], 48000, Ratio::X16);
        assert!(
            m2.meter_peak_hold_dbfs > -0.02,
            "held: {}",
            m2.meter_peak_hold_dbfs
        );

        // Block size changes the granularity of the release, not the peak it saw:
        // both must have caught the burst, and both must release similarly.
        let a = measure_signal_block(&[early.clone()], 48000, Ratio::X16, 1024);
        let b = measure_signal_block(&[early], 48000, Ratio::X16, 2048);
        assert!(
            (a.meter_peak_hold_dbfs - b.meter_peak_hold_dbfs).abs() < 6.0,
            "{} vs {}",
            a.meter_peak_hold_dbfs,
            b.meter_peak_hold_dbfs
        );
    }

    #[test]
    fn ragged_channels_and_a_zero_rate_do_not_panic() {
        // Channel 0 is not a promise the other channels made. Metering walks the common
        // prefix; the sample statistics still see everything.
        let ragged = vec![vec![0.5; 100], vec![0.25; 40]];
        let m = measure_signal(&ragged, 48_000, Ratio::X16);
        assert!(m.sample_peak_dbfs.is_finite(), "{m:?}");

        // A zero rate cannot be metered (its integration window is a division by zero),
        // but the peak/TP figures are pure sample statistics and stay valid.
        let m = measure_signal(&[vec![0.5; 64]], 0, Ratio::X16);
        assert!(
            m.sample_peak_dbfs.is_finite(),
            "sample peak must survive: {m:?}"
        );
        assert!(
            m.true_peak_dbtp.is_finite(),
            "true peak must survive: {m:?}"
        );
        // Observed on this build: metering at a zero rate produces NaN, so the guard is
        // what keeps the meter fields at their "nothing metered" values instead.
        assert!(
            !m.meter_param.is_nan() && !m.meter_peak_hold_lin.is_nan(),
            "a zero rate must not put NaN in the meter fields: {m:?}"
        );
        assert_eq!(
            m.meter_param, 0.0,
            "nothing was metered, so nothing to report"
        );
    }
}
