//! Minimum-phase FIR synthesis, concha-notch policy, and the audit export.
//!
//! [`synthesize_minimum_phase_ir`] is the heart of the tool: it turns control knots
//! `(freq_hz, gain_dB)` into a truncated **minimum-phase** impulse response by
//! Hilbert–cepstral folding — take the log-magnitude, extend it to a full Hermitian
//! spectrum, transform to the cepstrum, keep and double the causal half, transform
//! back, exponentiate. The result has the requested magnitude and the *smallest*
//! achievable phase lag, so the impulse peak sits at sample 0 with no pre-ringing:
//! a filter that fixes a resonance without smearing transients backwards.
//!
//! Three things in here are policy rather than maths, and they are the reason the
//! output is not simply `scipy.signal.minimum_phase(...)`:
//!
//! * **preamp** — the curve is normalised so its largest boost lands at
//!   `headroom_db`, i.e. the filter can never make the signal louder than
//!   `headroom_db` above full scale, whatever the measurement said.
//! * **the HF wire** — above `hf_bypass_hz` the control knots are `0.00 dB` by
//!   construction (see [`crate::model::derive_ctrl_pts_from_raw`]), so the coupler's
//!   half-wave cylinder null is left alone instead of being boosted into a
//!   sibilance peak. `gauss_apodize` adds a Gaussian roll-off above 20.2 kHz for
//!   callers that want the legacy brickwall behaviour.
//! * **half-Kaiser tail** — truncating an infinitely-supported minimum-phase IR is a
//!   step discontinuity, which is a broadband error. Fading the last `n_tail` taps
//!   turns that step into a taper; both floors are reported so the taper's effect is
//!   auditable rather than implied.

use crate::fft::{self, ifft, irfft, rfftfreq};
use crate::model::{round4, Point};
use crate::pchip::Pchip;
use crate::window::half_kaiser_tail_apply;
use rustfft::num_complex::Complex64;
use std::fmt;
use std::path::{Path, PathBuf};

/// Concha / driver-reflection notch policy inside `notch_range_hz`.
///
/// A coupler null at 3–4 kHz is largely *non-minimum-phase* (a reflection zero
/// outside the unit circle, plus L/R and head-to-head variation), so the boost a
/// straight inversion asks for there is not a boost the ear will hear consistently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotchMode {
    /// Cap positive boosts smoothly at `safe_cap_db` via `tanh` — the default.
    Safe,
    /// Leave the regularised fill exactly as the control knots say.
    Original,
    /// Clamp positive boosts in the range to `0.00 dB`.
    Zero,
}

impl NotchMode {
    pub const CHOICES: [&'static str; 3] = ["safe", "original", "zero"];

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Original => "original",
            Self::Zero => "zero",
        }
    }

    /// Parse, with the same vocabulary as `--notch-mode`.
    pub fn parse(s: &str) -> Result<Self, IrError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "safe" => Ok(Self::Safe),
            "original" => Ok(Self::Original),
            "zero" => Ok(Self::Zero),
            other => Err(IrError(format!(
                "Unknown notch mode: {other} (choose {})",
                Self::CHOICES.join(", ")
            ))),
        }
    }
}

impl std::str::FromStr for NotchMode {
    type Err = IrError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl fmt::Display for NotchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Anything that stops a synthesis or a policy step: a bad notch mode, a control
/// curve too short to interpolate, an FFT size too small to fold.
#[derive(Debug, Clone)]
pub struct IrError(pub String);

impl fmt::Display for IrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for IrError {}

/// What one synthesis produced, including the numbers the report prints.
#[derive(Debug, Clone)]
pub struct IrResult {
    /// The truncated, tail-faded minimum-phase FIR (`n_taps` samples, `f64`).
    pub ir: Vec<f64>,
    /// Normalised target magnitude on the FFT grid, `n_fft/2 + 1` bins. Kept because
    /// the verification plot needs the *linear-phase* counterpart impulse response,
    /// which is `irfft(mag_lin)` — the pre-ringing this method removes.
    pub mag_lin: Vec<f64>,
    /// Largest boost in the curve, before normalisation, dB.
    pub max_boost_db: f64,
    /// Applied preamp, dB (`headroom_db - max_boost_db`).
    pub preamp_db: f64,
    /// Share of the full-length filter's energy that survives truncation, %.
    pub energy_pct: f64,
    /// Level of the last 128 taps before the fade, dBFS.
    pub unwindowed_floor_dbfs: f64,
    /// Level of the last 100 taps after the fade, dBFS.
    pub windowed_floor_dbfs: f64,
}

/// `20·log10(Σ|h|)` — the worst-case (all-phase-aligned) gain of the filter, which is
/// the bound a user should compare a ceiling against.
#[must_use]
pub fn l1_gain_db(ir: &[f64]) -> f64 {
    crate::tp::db_from_peak(ir.iter().map(|h| h.abs()).sum())
}

/// Synthesise a minimum-phase FIR from control knots.
///
/// * `n_fft` — complex FFT length; sets the frequency resolution `fs/n_fft` of the
///   target grid. Must be even and at least 4.
/// * `n_taps` — output length. The filter is truncated, not windowed over its whole
///   length, so the transition band is whatever `n_taps` gives.
/// * `n_tail` — half-Kaiser fade length (`0` disables the fade).
/// * `kaiser_beta` — shape of that fade (`18.0` offline, `16.0` for the real-time IRs).
// The signature mirrors the reference's `synthesize_minimum_phase_ir` one parameter at a
// time, which is what makes the two auditable against each other line by line; bundling the
// numbers into a struct would trade that for a second naming of the same values.
#[allow(clippy::too_many_arguments)]
pub fn synthesize_minimum_phase_ir(
    ctrl_pts: &[(f64, f64)],
    fs: u32,
    n_fft: usize,
    n_taps: usize,
    n_tail: usize,
    headroom_db: f64,
    gauss_apodize: bool,
    kaiser_beta: f64,
) -> Result<IrResult, IrError> {
    if fs == 0 {
        // The reference has one rate guard and this is its message. Every time constant,
        // bin width and millisecond figure below divides by the rate.
        return Err(IrError("sample rate must be positive (got 0)".to_string()));
    }
    if ctrl_pts.len() < 2 {
        return Err(IrError(
            "control curve needs at least two knots to interpolate".to_string(),
        ));
    }
    if n_fft < 4 || !n_fft.is_multiple_of(2) {
        return Err(IrError(format!(
            "n_fft must be even and >= 4 (got {n_fft})"
        )));
    }
    if n_taps == 0 || n_taps > n_fft {
        return Err(IrError(format!(
            "n_taps must be in 1..=n_fft (got {n_taps} taps, n_fft {n_fft})"
        )));
    }

    let f_nodes: Vec<f64> = ctrl_pts.iter().map(|p| p.0).collect();
    let g_nodes: Vec<f64> = ctrl_pts.iter().map(|p| p.1).collect();
    let curve = Pchip::new(&f_nodes, &g_nodes);

    let fs_f = fs as f64;
    let freqs = rfftfreq(n_fft, fs_f);
    let mut target_db: Vec<f64> = freqs.iter().map(|&f| curve.eval(f)).collect();
    // DC is the first knot's gain, not the log-interpolant's clamp: there is no
    // log10(0), so the interpolator cannot answer, and "same as the lowest measured
    // frequency" is the only defensible answer.
    target_db[0] = g_nodes[0];

    if gauss_apodize {
        for (f, db) in freqs.iter().zip(target_db.iter_mut()) {
            if *f > 20200.0 {
                let x = (f - 20200.0) / 1650.0;
                *db += -3.0103 * x * x;
            }
        }
    }

    let max_boost_db = target_db.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let preamp_db = headroom_db - max_boost_db;

    let mag_lin: Vec<f64> = target_db
        .iter()
        .map(|db| 10f64.powf((db + preamp_db) / 20.0))
        .collect();
    let log_mag: Vec<f64> = mag_lin.iter().map(|m| m.max(1e-18).ln()).collect();

    // Hermitian extension of a *real, even* log-magnitude: log_mag[-2:0:-1] mirrors
    // the interior bins, so the cepstrum comes out real and the fold is exact.
    let bins = log_mag.len();
    let mut log_spec_full = Vec::with_capacity(n_fft);
    log_spec_full.extend_from_slice(&log_mag);
    for k in (1..bins - 1).rev() {
        log_spec_full.push(log_mag[k]);
    }
    if log_spec_full.len() != n_fft {
        return Err(IrError(format!(
            "internal: spectrum length {} != n_fft {n_fft}",
            log_spec_full.len()
        )));
    }

    let cepstrum: Vec<f64> = ifft(
        &log_spec_full
            .iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect::<Vec<_>>(),
    )
    .iter()
    .map(|c| c.re)
    .collect();

    // Cepstral lifter: keep n=0 and n=n/2 once, the causal half twice.
    let half = n_fft / 2;
    // The one-sided lift: DC and Nyquist keep their weight, the lower half doubles so the
    // inverse transform comes out real, and the upper half is dropped.
    let lifted: Vec<Complex64> = cepstrum
        .iter()
        .enumerate()
        .map(|(n, &c)| {
            let w = if n == 0 || n == half {
                1.0
            } else if n < half {
                2.0
            } else {
                0.0
            };
            Complex64::new(c * w, 0.0)
        })
        .collect();

    // Minimum-phase spectrum = exp(FFT(lifted cepstrum)), then an inverse transform.
    // The order is the algorithm: `exp` of the cepstrum is not a spectrum at all, and
    // skipping this forward transform costs exactly the preamp (the filter comes out
    // a near-unity delta instead of the headroom-scaled one).
    let min_phase_spec: Vec<Complex64> = crate::fft::fft(&lifted).iter().map(|c| c.exp()).collect();
    let ir_full: Vec<f64> = ifft(&min_phase_spec).iter().map(|c| c.re).collect();

    let mut ir = ir_full[..n_taps].to_vec();
    let unwindowed_floor_dbfs = db_floor(&ir, 128);

    half_kaiser_tail_apply(&mut ir, n_tail, kaiser_beta);

    let windowed_floor_dbfs = db_floor(&ir, 100);

    let energy_in: f64 = ir.iter().map(|v| v * v).sum();
    let energy_full: f64 = ir_full.iter().map(|v| v * v).sum();
    let energy_pct = energy_in / energy_full * 100.0;

    Ok(IrResult {
        ir,
        mag_lin,
        max_boost_db,
        preamp_db,
        energy_pct,
        unwindowed_floor_dbfs,
        windowed_floor_dbfs,
    })
}

/// `20·log10(max|last n taps| + eps)` — the decay at the truncation point. The `eps`
/// is what keeps a perfectly-zero tail at a finite (very negative) number instead of
/// `-inf`, exactly as the Python's `+ 1e-25` / `+ 1e-25` does.
#[must_use]
fn db_floor(ir: &[f64], n: usize) -> f64 {
    let k = n.min(ir.len());
    let peak = ir[ir.len() - k..]
        .iter()
        .cloned()
        .fold(0.0f64, |a, v| a.max(v.abs()));
    20.0 * (peak + 1e-25).log10()
}

/// The linear-phase counterpart of a synthesis, for the plot's pre-ringing trace.
///
/// This is the *centred* transform, `fftshift(irfft(mag_lin, n_fft))`, matching
/// `ir_lin_centered` in `hypreq.py`: the energy sits at `n_fft / 2`, so the plot's time
/// axis (which is zero at that index) shows the pre-ringing on the left. Callers must not
/// shift it again.
#[must_use]
pub fn linear_phase_ir_centered(mag_lin: &[f64], n_fft: usize) -> Vec<f64> {
    // NumPy accepts a real array here and reads it as a spectrum whose imaginary
    // part is zero everywhere — a linear phase. Handing `irfft` the same thing is
    // the whole translation; the magnitude itself is untouched.
    let spec: Vec<crate::fft::Complex64> = mag_lin
        .iter()
        .map(|&m| crate::fft::Complex64::new(m, 0.0))
        .collect();
    let t = irfft(&spec, n_fft);
    let half = n_fft / 2;
    let mut shifted = Vec::with_capacity(n_fft);
    shifted.extend_from_slice(&t[half..]);
    shifted.extend_from_slice(&t[..half]);
    shifted
}

/// Apply the concha-notch policy to a control curve.
///
/// Only knots *inside* `notch_range_hz` with a positive gain are touched; everything
/// else passes through bit-for-bit. `Safe` rounds to 0.0001 dB because the `tanh`
/// result is otherwise a different float on every platform, and a control curve that
/// is not reproducible is not auditable.
pub fn apply_notch_mode(
    ctrl_pts: &[(f64, f64)],
    mode: NotchMode,
    safe_cap_db: f64,
    notch_range_hz: Option<(f64, f64)>,
) -> Result<Vec<(f64, f64)>, IrError> {
    if mode == NotchMode::Original || notch_range_hz.is_none() {
        return Ok(ctrl_pts.to_vec());
    }
    let (f_lo, f_hi) = notch_range_hz.unwrap();
    let mut out = Vec::with_capacity(ctrl_pts.len());
    for &(f, g) in ctrl_pts {
        if f_lo <= f && f <= f_hi && g > 0.0 {
            match mode {
                NotchMode::Zero => out.push((f, 0.0)),
                NotchMode::Safe => {
                    if safe_cap_db <= 0.0 {
                        return Err(IrError(format!(
                            "safe_cap_db must be positive for notch-mode=safe (got {safe_cap_db})"
                        )));
                    }
                    out.push((f, round4(safe_cap_db * (g / safe_cap_db).tanh())));
                }
                NotchMode::Original => out.push((f, g)),
            }
        } else {
            out.push((f, g));
        }
    }
    Ok(out)
}

/// Write the auditable table: frequency, raw measurement, target, raw error, applied
/// EQ — all sampled at the *control* knots, so the rows are the numbers the filter was
/// actually built from.
///
/// CRLF-terminated and formatted to the same decimal places as the Python's
/// `csv.writer`, so the two files diff empty.
pub fn export_audit_csv(
    model_key: &str,
    raw_pts: &[Point],
    harm_pts: &[(f64, f64)],
    harm_offset: f64,
    ctrl_pts: &[(f64, f64)],
) -> Result<PathBuf, std::io::Error> {
    export_audit_csv_in(
        Path::new(""),
        model_key,
        raw_pts,
        harm_pts,
        harm_offset,
        ctrl_pts,
    )
}

/// [`export_audit_csv`] with the table written inside `dir` instead of the process's
/// working directory. `cargo test` runs tests in parallel in one process, and the working
/// directory is process-global state: a test that changes it can move every other test's
/// relative paths out from under it.
pub fn export_audit_csv_in(
    dir: &Path,
    model_key: &str,
    raw_pts: &[Point],
    harm_pts: &[(f64, f64)],
    harm_offset: f64,
    ctrl_pts: &[(f64, f64)],
) -> Result<PathBuf, std::io::Error> {
    let file_name = format!("{model_key}_audit_data.csv");
    let f_ctrl: Vec<f64> = ctrl_pts.iter().map(|p| p.0).collect();
    let g_ctrl: Vec<f64> = ctrl_pts.iter().map(|p| p.1).collect();

    let f_raw: Vec<f64> = raw_pts.iter().map(|p| p[0]).collect();
    let g_raw: Vec<f64> = raw_pts.iter().map(|p| *p.last().unwrap()).collect();
    let raw_at_ctrl = crate::pchip::pchip_log_interp(&f_raw, &g_raw, &f_ctrl);

    let f_h: Vec<f64> = harm_pts.iter().map(|p| p.0).collect();
    let g_h: Vec<f64> = harm_pts.iter().map(|p| p.1 + harm_offset).collect();
    let harm_at_ctrl = crate::pchip::pchip_log_interp(&f_h, &g_h, &f_ctrl);

    let mut s =
        String::from("frequency_hz,raw_dBr,harman_target_dBr,raw_error_dB,applied_eq_dB\r\n");
    for (((freq, r_val), h_val), eq_val) in f_ctrl
        .iter()
        .zip(&raw_at_ctrl)
        .zip(&harm_at_ctrl)
        .zip(&g_ctrl)
    {
        s.push_str(&format!(
            "{:.1},{:.2},{:.2},{:.2},{:.2}\r\n",
            freq,
            r_val,
            h_val,
            h_val - r_val,
            eq_val
        ));
    }
    let path = dir.join(file_name);
    std::fs::write(&path, s)?;
    Ok(path)
}

/// `max(65536, 1 << (taps*4 - 1).bit_length())` — the offline/mpv synthesis FFT size.
#[must_use]
pub fn mpv_n_fft(taps: usize) -> usize {
    // `taps * 4 - 1` is computed on a usize, so a tap count of 0 used to wrap to
    // usize::MAX; the reference's `1 << (taps * 4 - 1).bit_length()` has no such problem
    // and lands on the 65536 floor. Saturating gets the same answer without the panic.
    65536usize.max(fft::shl_bit_length(
        taps.saturating_mul(4).saturating_sub(1),
    ))
}

/// Knot gain at `notch_ref_hz` — the number the concha callout prints. A knot at
/// exactly that frequency wins over the interpolant, matching the Python.
#[must_use]
pub fn notch_db_at(ctrl_pts: &[(f64, f64)], notch_ref_hz: Option<f64>) -> f64 {
    let Some(ref_hz) = notch_ref_hz else {
        return 0.0;
    };
    if let Some((_, g)) = ctrl_pts.iter().find(|(f, _)| *f == ref_hz) {
        return *g;
    }
    let f: Vec<f64> = ctrl_pts.iter().map(|p| p.0).collect();
    let g: Vec<f64> = ctrl_pts.iter().map(|p| p.1).collect();
    crate::pchip::pchip_eval1(&f, &g, ref_hz)
}

/// Convenience for callers that only have a path to write the CSV next to.
#[must_use]
pub fn audit_csv_path(dir: &Path, model_key: &str) -> PathBuf {
    dir.join(format!("{model_key}_audit_data.csv"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat() -> Vec<(f64, f64)> {
        vec![
            (20.0, 0.0),
            (100.0, 0.0),
            (1000.0, 0.0),
            (10000.0, 0.0),
            (20000.0, 0.0),
        ]
    }

    #[test]
    fn a_flat_curve_is_an_impulse() {
        // 0 dB everywhere, -1.5 dB headroom ⇒ the filter is 10^(-1.5/20) at sample 0
        // and nothing anywhere else: minimum-phase of a constant is a scaled delta.
        let r = synthesize_minimum_phase_ir(&flat(), 48000, 4096, 1024, 128, -1.5, false, 18.0)
            .unwrap();
        // Reference values from the NumPy implementation on the same inputs:
        // ir[0] = 0.8413951416451951, every other tap exactly 0.0, and the whole
        // ||h||_1 sits at the headroom (-1.5 dB) because a flat curve is a gain.
        let expect = 0.841_395_141_645_195_1f64;
        assert_eq!(r.ir[0], expect);
        let tail: f64 = r.ir[1..]
            .iter()
            .cloned()
            .fold(0.0f64, |a, v| a.max(v.abs()));
        assert_eq!(tail, 0.0, "off-peak energy {tail}");
        assert_eq!(r.max_boost_db, 0.0);
        assert_eq!(r.preamp_db, -1.5);
        assert_eq!(r.energy_pct, 100.0);
        // A tail of exact zeros floors at 20·log10(1e-25) = -500 dBFS, not -inf.
        assert_eq!(r.unwindowed_floor_dbfs, -500.0);
        assert_eq!(r.windowed_floor_dbfs, -500.0);
        assert!(
            (l1_gain_db(&r.ir) + 1.5).abs() < 1e-12,
            "L1 {}",
            l1_gain_db(&r.ir)
        );
    }

    #[test]
    fn impulse_peak_is_at_sample_zero_with_no_pre_ringing() {
        let mut c = flat();
        for (_, g) in c.iter_mut() {
            *g = 0.0;
        }
        c[2].1 = 6.0; // a broad 1 kHz shelf
        let r = synthesize_minimum_phase_ir(&c, 48000, 8192, 2048, 256, -1.5, false, 18.0).unwrap();
        let peak =
            r.ir.iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap();
        assert!(peak.0 <= 1, "peak landed at {}", peak.0);
        let pre: f64 = r.ir[2..]
            .iter()
            .cloned()
            .fold(0.0f64, |a, v| a.max(v.abs()));
        assert!(pre < r.ir[0].abs(), "nothing before the peak may rival it");
    }

    #[test]
    fn preamp_normalises_the_largest_boost_to_headroom() {
        let mut c = flat();
        c[1].1 = 4.25;
        c[3].1 = -7.5;
        let r = synthesize_minimum_phase_ir(&c, 48000, 4096, 512, 64, -1.5, false, 18.0).unwrap();
        // The curve's peak is not the knot's 4.25 dB: `max_boost_db` is the maximum
        // of the PCHIP curve sampled on the rfft grid, whose spacing here is
        // 48000/4096 = 11.72 Hz, so the grid straddles the 100 Hz knot and lands a
        // hair under it. Reference: 4.246121547481881.
        assert_eq!(r.max_boost_db, 4.246_121_547_481_881);
        // The invariant is what the preamp is for: peak gain + preamp == headroom.
        assert_eq!(r.preamp_db, -1.5 - r.max_boost_db);
        // And ||h||_1 is *not* bounded by the headroom — a minimum-phase filter with
        // a boost rings up in the time domain. Reference: +1.52412933569098 dB.
        assert!(
            (l1_gain_db(&r.ir) - 1.524_129_335_690_98).abs() < 1e-9,
            "L1 {}",
            l1_gain_db(&r.ir)
        );
    }

    #[test]
    fn tail_fade_lowers_the_windowed_floor() {
        let mut c = flat();
        c[2].1 = -6.0;
        let faded =
            synthesize_minimum_phase_ir(&c, 48000, 4096, 1024, 256, -1.5, false, 18.0).unwrap();
        let raw = synthesize_minimum_phase_ir(&c, 48000, 4096, 1024, 0, -1.5, false, 18.0).unwrap();
        // Both report the same unwindowed floor; only the faded one gets a deeper
        // measured floor, which is the fade's entire job.
        assert!(
            (faded.unwindowed_floor_dbfs - raw.unwindowed_floor_dbfs).abs() < 1e-9,
            "the reported unwindowed floor is a property of the truncation"
        );
        assert!(
            faded.windowed_floor_dbfs < raw.windowed_floor_dbfs - 1.0,
            "{} vs {}",
            faded.windowed_floor_dbfs,
            raw.windowed_floor_dbfs
        );
    }

    #[test]
    fn gauss_apodizer_pulls_only_above_20200_hz() {
        let c = flat();
        let no = synthesize_minimum_phase_ir(&c, 48000, 4096, 512, 0, 0.0, false, 18.0).unwrap();
        let yes = synthesize_minimum_phase_ir(&c, 48000, 4096, 512, 0, 0.0, true, 18.0).unwrap();
        let freqs = rfftfreq(4096, 48000.0);
        let below = freqs.iter().position(|&f| f > 20200.0).unwrap();
        for (i, (&a, &b)) in no.mag_lin[1..below]
            .iter()
            .zip(&yes.mag_lin[1..below])
            .enumerate()
        {
            assert!((a - b).abs() < 1e-15, "bin {} changed", freqs[i + 1]);
        }
        assert!(
            yes.mag_lin[yes.mag_lin.len() - 1] < no.mag_lin[no.mag_lin.len() - 1],
            "Nyquist must come down"
        );
    }

    #[test]
    fn notch_modes_only_touch_the_window_and_the_positive_knots() {
        let c = vec![
            (1000.0, 3.0),
            (3000.0, 4.0),
            (3703.0, 8.0),
            (4300.0, 2.0),
            (4400.0, -5.0),
            (5000.0, 6.0),
        ];
        let range = Some((2800.0, 4400.0));
        let orig = apply_notch_mode(&c, NotchMode::Original, 1.5, range).unwrap();
        assert_eq!(orig, c, "original is a pass-through");
        let safe = apply_notch_mode(&c, NotchMode::Safe, 1.5, range).unwrap();
        assert_eq!(safe[0], (1000.0, 3.0), "below the range");
        assert_eq!(safe[5], (5000.0, 6.0), "above the range");
        assert_eq!(safe[4], (4400.0, -5.0), "a negative knot inside the range");
        // Reference (NumPy `round(1.5*tanh(g/1.5), 4)`): g=8 → 1.4999, g=4 → 1.4856,
        // g=2 → 1.3051. The 4-decimal grid is what makes the curve reproducible, so
        // the assert is exact, and the saturating shape is checked by the ordering.
        assert_eq!(safe[2].1, 1.499_9, "g=8 capped");
        assert_eq!(safe[1].1, 1.485_6, "g=4 capped");
        assert_eq!(safe[3].1, 1.305_1, "g=2 capped");
        assert!(
            safe[1].1 > safe[3].1 && safe[3].1 > 1.3,
            "the cap is monotonic in the requested boost: {:?} {:?}",
            safe[1].1,
            safe[3].1
        );
        let zero = apply_notch_mode(&c, NotchMode::Zero, 1.5, range).unwrap();
        assert_eq!(zero[1], (3000.0, 0.0));
        assert_eq!(zero[2], (3703.0, 0.0));
        let none_range = apply_notch_mode(&c, NotchMode::Safe, 1.5, None).unwrap();
        assert_eq!(
            none_range, c,
            "a model with no notch range is never touched"
        );
    }

    #[test]
    fn notch_mode_round_trips_names() {
        use std::str::FromStr;
        for name in NotchMode::CHOICES {
            let m = NotchMode::parse(name).unwrap();
            assert_eq!(m.as_str(), name);
            assert_eq!(m.to_string(), name);
            // The trait form is what `"safe".parse()` uses; both must agree.
            assert_eq!(NotchMode::from_str(name).unwrap(), m);
            assert_eq!(name.parse::<NotchMode>().unwrap(), m);
        }
        assert!(
            NotchMode::parse("  safe ").is_ok(),
            "surrounding space is trimmed"
        );
        assert!(
            NotchMode::parse("SAFE").is_ok(),
            "case-folded like argparse"
        );
        assert!(NotchMode::parse("aggressive").is_err());
    }

    #[test]
    fn notch_db_lookup_prefers_an_exact_knot() {
        let c = vec![(3700.0, 1.0), (3703.0, 2.5), (3800.0, 0.0)];
        assert_eq!(notch_db_at(&c, Some(3703.0)), 2.5);
        assert_eq!(notch_db_at(&c, None), 0.0);
        let interp = notch_db_at(&c, Some(3751.5));
        assert!(interp > 1.0 && interp < 2.5, "{interp}");
    }

    #[test]
    fn mpv_fft_size_matches_the_python_expression() {
        // Reference: max(65536, 1 << (taps*4 - 1).bit_length()) → 65536 taps need
        // 262144, 16384 taps land exactly on the 65536 floor.
        assert_eq!(mpv_n_fft(65536), 262144);
        assert_eq!(mpv_n_fft(16384), 65536);
        assert_eq!(
            mpv_n_fft(256),
            65536,
            "the 65536 floor dominates for small taps"
        );
    }

    #[test]
    fn rejects_degenerate_curves_and_sizes() {
        assert!(synthesize_minimum_phase_ir(
            &[(1000.0, 0.0)],
            48000,
            4096,
            512,
            0,
            -1.5,
            false,
            18.0
        )
        .is_err());
        assert!(
            synthesize_minimum_phase_ir(&flat(), 48000, 4095, 512, 0, -1.5, false, 18.0).is_err()
        );
        assert!(
            synthesize_minimum_phase_ir(&flat(), 48000, 4096, 5000, 0, -1.5, false, 18.0).is_err()
        );
    }

    #[test]
    fn audit_csv_is_crlf_and_fixed_width() {
        let dir = std::env::temp_dir().join(format!("hypreq_audit_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let raw = vec![
            Point::from(vec![20.0, -1.26]),
            Point::from(vec![1000.0, 0.0]),
        ];
        let target = vec![(20.0, 1.41), (1000.0, 0.0)];
        let ctrl = vec![(20.0, 0.12), (1000.0, -0.34)];
        // Written by path, never by moving the process's working directory: that is
        // process-global state and `cargo test` runs this test beside every other one.
        let p = export_audit_csv_in(&dir, "unitmodel", &raw, &target, 0.0, &ctrl).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text
            .starts_with("frequency_hz,raw_dBr,harman_target_dBr,raw_error_dB,applied_eq_dB\r\n"));
        assert!(text.ends_with("\r\n"));
        let row1 = text.lines().nth(1).unwrap();
        assert!(row1.starts_with("20.0,"), "{row1}");
        assert_eq!(row1.split(',').count(), 5);
        assert_eq!(p.file_name().unwrap(), "unitmodel_audit_data.csv");
        assert_eq!(
            p.parent().unwrap(),
            dir,
            "the table lands where it was asked to"
        );
        // The relative-path form still writes beside the caller, which is what the command
        // line relies on to keep filenames next to the convolvers.
        let rel = export_audit_csv("unitmodel_rel", &raw, &target, 0.0, &ctrl).unwrap();
        assert_eq!(rel, PathBuf::from("unitmodel_rel_audit_data.csv"));
        std::fs::remove_file(&rel).ok();
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod linear_probe_tests {
    use super::*;

    #[test]
    fn linear_phase_ir_is_centred() {
        // A flat magnitude with zero phase is the identity filter: `irfft` puts a single
        // sample at index 0, and the centring moves it to `n_fft / 2`. Off by one shift in
        // either direction and this lands at 0 or at the far end instead.
        let n = 4096usize;
        let ir = linear_phase_ir_centered(&vec![1.0; n / 2 + 1], n);
        let peak = ir.iter().enumerate().fold((0usize, 0.0f64), |a, (i, &v)| {
            if v.abs() > a.1 {
                (i, v.abs())
            } else {
                a
            }
        });
        assert_eq!(peak.0, n / 2, "energy must sit at the centre sample");
        assert!((peak.1 - 1.0).abs() < 1e-9, "peak magnitude {}", peak.1);
        assert!(ir[..n / 2 - 1].iter().all(|&v| v.abs() < 1e-9));
        assert!(ir[n / 2 + 1..].iter().all(|&v| v.abs() < 1e-9));
    }

    #[test]
    fn a_zero_tap_count_and_a_zero_rate_are_answered_not_wrapped() {
        // `taps * 4 - 1` on a usize wraps to usize::MAX at taps = 0, which then asks
        // shl_bit_length for a shift nobody can serve. The reference's formula has no such
        // trouble and floors at 65536.
        assert_eq!(mpv_n_fft(0), 65_536);
        assert_eq!(mpv_n_fft(1), 65_536);
        assert_eq!(mpv_n_fft(2), 65_536);
        assert_eq!(mpv_n_fft(16_384), 65_536);
        assert_eq!(mpv_n_fft(65_536), 262_144);

        // A zero sample rate is the one rate error the reference names, and it names it
        // the same way here.
        let e = synthesize_minimum_phase_ir(
            &[(20.0, 0.0), (1000.0, 0.0)],
            0,
            65_536,
            1024,
            128,
            -1.5,
            false,
            18.0,
        )
        .unwrap_err()
        .to_string();
        assert_eq!(e, "sample rate must be positive (got 0)");
    }
}
