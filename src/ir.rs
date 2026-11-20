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

    /// The mode a choice string names, or `None` for anything else.
    ///
    /// It answers a yes/no question and carries no wording, because the only caller that can fail
    /// is the CLI, and there the message has to be argparse's: `--notch-mode` is rejected by
    /// the CLI's own choice check with `invalid choice: 'x' (choose from 'safe', 'original',
    /// 'zero')` and exit code 2. A second, library-worded rejection next to it could only ever be
    /// dead — and it was: the one production caller threw its `Err` away and rebuilt argparse's
    /// sentence by hand.
    ///
    /// Space and case are folded here, which is a courtesy to library callers. It is *not* what
    /// the CLI does: argparse compares choices exactly, so `--notch-mode SAFE` is a usage error,
    /// and `validate_choice` is what enforces that.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "safe" => Some(Self::Safe),
            "original" => Some(Self::Original),
            "zero" => Some(Self::Zero),
            _ => None,
        }
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
    // DC is the lowest sorted knot's gain, evaluated at 0 Hz (which Pchip clamps to its
    // lowest sorted frequency knot f_min).
    target_db[0] = curve.eval(0.0);

    if gauss_apodize {
        let (f_start, width) = gauss_apodize_params(fs);
        for (f, db) in freqs.iter().zip(target_db.iter_mut()) {
            if *f > f_start {
                let x = (f - f_start) / width;
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

    // Compensated, and it is not decoration: the ratio is printed with twelve significant
    // digits, and on the ft1pro curve at the default tap count the true value is
    // 99.9999999999 while a left-to-right walk reports exactly 100.0000000000 — a
    // one-sided difference in the printed report, on identical filters.
    let energy_in = crate::accum::sum_of_squares(&ir);
    let energy_full = crate::accum::sum_of_squares(&ir_full);
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

/// Synthesizes minimum-phase IRs for a stereo pair under a single shared preamp.
///
/// Normalizes the hotter ear to `headroom_db` and applies that exact same preamp to both ears,
/// so the interaural level offset (balance) is preserved rather than erased by independent normalization.
#[allow(clippy::too_many_arguments)]
pub fn synthesize_stereo_minimum_phase_ir(
    left_ctrl: &[(f64, f64)],
    right_ctrl: &[(f64, f64)],
    fs: u32,
    n_fft: usize,
    n_taps: usize,
    n_tail: usize,
    headroom_db: f64,
    gauss_apodize: bool,
    kaiser_beta: f64,
) -> Result<(IrResult, IrResult), IrError> {
    let mut left = synthesize_minimum_phase_ir(
        left_ctrl,
        fs,
        n_fft,
        n_taps,
        n_tail,
        headroom_db,
        gauss_apodize,
        kaiser_beta,
    )?;
    let mut right = synthesize_minimum_phase_ir(
        right_ctrl,
        fs,
        n_fft,
        n_taps,
        n_tail,
        headroom_db,
        gauss_apodize,
        kaiser_beta,
    )?;
    let shared_max_boost = left.max_boost_db.max(right.max_boost_db);
    let shared_preamp = headroom_db - shared_max_boost;

    let scale_channel = |ch: &mut IrResult| {
        let diff = shared_preamp - ch.preamp_db;
        if diff.abs() > 1e-9 {
            let scale = 10.0f64.powf(diff / 20.0);
            for x in &mut ch.ir {
                *x *= scale;
            }
            for m in &mut ch.mag_lin {
                *m *= scale;
            }
            ch.preamp_db = shared_preamp;
            ch.unwindowed_floor_dbfs += diff;
            ch.windowed_floor_dbfs += diff;
        }
    };
    scale_channel(&mut left);
    scale_channel(&mut right);
    Ok((left, right))
}

/// `20·log10(max|last n taps| + eps)` — the decay at the truncation point. The `eps`
/// is what keeps a perfectly-zero tail at a finite (very negative) number instead of
/// `-inf`, exactly as the Python's `+ 1e-25` / `+ 1e-25` does.
#[must_use]
pub fn db_floor(ir: &[f64], n: usize) -> f64 {
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

/// Raised-cosine blending weight `w(f) ∈ [0, 1]` for the concha notch band `[f_lo, f_hi]`.
/// Transition skirt has width `skirt_oct` octaves (centered on `f_lo` and `f_hi`, with `±skirt_oct/2`).
#[must_use]
pub fn notch_blend_weight(f: f64, f_lo: f64, f_hi: f64, skirt_oct: f64) -> f64 {
    if !f.is_finite() || !f_lo.is_finite() || !f_hi.is_finite() || !skirt_oct.is_finite() {
        return 0.0;
    }
    if f <= 0.0 || f_lo <= 0.0 || f_hi <= f_lo || skirt_oct <= 0.0 {
        return if f >= f_lo && f <= f_hi { 1.0 } else { 0.0 };
    }
    let band_oct = (f_hi / f_lo).log2();
    let effective_skirt = skirt_oct.min(band_oct);
    let half_skirt = effective_skirt * 0.5;
    let lo_start = f_lo * 2.0f64.powf(-half_skirt);
    let lo_end = f_lo * 2.0f64.powf(half_skirt);
    let hi_start = f_hi * 2.0f64.powf(-half_skirt);
    let hi_end = f_hi * 2.0f64.powf(half_skirt);

    let w_lo = if f <= lo_start {
        0.0
    } else if f >= lo_end {
        1.0
    } else {
        let t = (f / lo_start).log2() / effective_skirt;
        0.5 * (1.0 - (std::f64::consts::PI * t).cos())
    };

    let w_hi = if f <= hi_start {
        1.0
    } else if f >= hi_end {
        0.0
    } else {
        let t = (hi_end / f).log2() / effective_skirt;
        0.5 * (1.0 - (std::f64::consts::PI * t).cos())
    };

    w_lo * w_hi
}

/// Apply the concha-notch policy to a control curve.
///
/// Boosts inside the notch band `notch_range_hz` are capped with smooth raised-cosine
/// blending skirts at the band edges to prevent discontinuous resonance cliffs.
/// `Safe` rounds to 0.0001 dB because the `tanh` result is otherwise a different float
/// on every platform, and a control curve that is not reproducible is not auditable.
pub fn apply_notch_mode(
    ctrl_pts: &[(f64, f64)],
    mode: NotchMode,
    safe_cap_db: f64,
    notch_range_hz: Option<(f64, f64)>,
) -> Result<Vec<(f64, f64)>, IrError> {
    if mode == NotchMode::Original || notch_range_hz.is_none() {
        return Ok(ctrl_pts.to_vec());
    }
    if mode == NotchMode::Safe && !(safe_cap_db.is_finite() && safe_cap_db > 0.0) {
        return Err(IrError(format!(
            "safe_cap_db must be positive for notch-mode=safe (got {safe_cap_db})"
        )));
    }
    let (f_lo, f_hi) = notch_range_hz.unwrap();
    let skirt_oct = 0.20;
    let mut out = Vec::with_capacity(ctrl_pts.len());
    for &(f, g) in ctrl_pts {
        let w = notch_blend_weight(f, f_lo, f_hi, skirt_oct);
        if w > 0.0 && g > 0.0 {
            match mode {
                NotchMode::Zero => {
                    let blended = (1.0 - w) * g;
                    out.push((f, round4(blended)));
                }
                NotchMode::Safe => {
                    let capped = safe_cap_db * (g / safe_cap_db).tanh();
                    let blended = (1.0 - w) * g + w * capped;
                    out.push((f, round4(blended)));
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
    // The ear-mean accessor, not `p.last()`: a two-ear audit table would otherwise be written out
    // as its right channel and the audit would disagree with the filter it claims to describe.
    let g_raw: Vec<f64> = raw_pts
        .iter()
        .map(|p| crate::model::point_gain(p))
        .collect();
    let raw_at_ctrl = crate::pchip::pchip_log_interp(&f_raw, &g_raw, &f_ctrl);

    let f_h: Vec<f64> = harm_pts.iter().map(|p| p.0).collect();
    let g_h: Vec<f64> = harm_pts.iter().map(|p| p.1 + harm_offset).collect();
    let harm_at_ctrl = crate::pchip::pchip_log_interp(&f_h, &g_h, &f_ctrl);

    let mut s = String::from("frequency_hz,raw_dBr,target_dBr,raw_error_dB,applied_eq_dB\r\n");
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

/// Scale-clamped Gaussian apodizer `(f_start_hz, width_hz)` for sample rate `fs`.
///
/// Roll-off is `w(f) = 2^(-0.5 * x^2)` where `x = (f - f_start) / width` (approx.
/// `-3.0103 * x^2` dB, i.e. `-3.0103` dB at `x = 1`). The 24 kHz Nyquist floor keeps
/// 44.1 kHz runs from moving `f_start` down to 18.559 kHz, which would attenuate audible
/// 19–20.2 kHz treble by up to 3.521 dB. At 48 kHz and above the original arithmetic is
/// preserved verbatim so shipped 48 kHz IRs remain bit-identical.
#[must_use]
pub fn gauss_apodize_params(fs: u32) -> (f64, f64) {
    let f_nyq = f64::from(fs) / 2.0;
    if f_nyq >= 24_000.0 {
        let f_start = f_nyq * (20_200.0 / 24_000.0);
        (f_start, (f_nyq - f_start) / (3_800.0 / 1_650.0))
    } else {
        (20_200.0, 1_650.0)
    }
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
        // a boost rings up in the time domain. Reference (Fritsch-Carlson PCHIP): +1.5099703362312815 dB.
        assert!(
            (l1_gain_db(&r.ir) - 1.509_970_336_231_281_5).abs() < 1e-9,
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
    fn stereo_synthesis_preserves_interaural_level_difference() {
        let l_curve = vec![(20.0, 1.0), (1000.0, 1.0), (20000.0, 1.0)];
        let r_curve = vec![(20.0, -1.0), (1000.0, -1.0), (20000.0, -1.0)];
        let (l, r) = synthesize_stereo_minimum_phase_ir(
            &l_curve, &r_curve, 48000, 4096, 512, 0, -1.5, false, 18.0,
        )
        .unwrap();
        assert_eq!(
            l.preamp_db, r.preamp_db,
            "preamp must be identical on both channels"
        );
        assert!((l.preamp_db - -2.5).abs() < 1e-9);
        let peak_l = l.ir.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        let peak_r = r.ir.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        let diff_db = 20.0 * (peak_l / peak_r).log10();
        assert!(
            (diff_db - 2.0).abs() < 0.05,
            "expected 2.0 dB L/R difference, got {diff_db:.3} dB"
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
    fn gauss_apodizer_keeps_44_1khz_audible_band_flat() {
        let c = flat();
        let no = synthesize_minimum_phase_ir(&c, 44100, 4096, 512, 0, 0.0, false, 18.0).unwrap();
        let yes = synthesize_minimum_phase_ir(&c, 44100, 4096, 512, 0, 0.0, true, 18.0).unwrap();
        let freqs = rfftfreq(4096, 44100.0);
        let below = freqs.iter().position(|&f| f > 20_200.0).unwrap();
        for (i, (&a, &b)) in no.mag_lin[1..below]
            .iter()
            .zip(&yes.mag_lin[1..below])
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-15,
                "44.1 kHz bin {} Hz changed by the apodizer",
                freqs[i + 1]
            );
        }
        assert!(
            yes.mag_lin[yes.mag_lin.len() - 1] < no.mag_lin[no.mag_lin.len() - 1],
            "the remaining ultrasonic band to Nyquist must still come down"
        );
    }

    #[test]
    fn gauss_apodizer_keeps_48khz_bit_identical_to_the_shipped_assets() {
        let c = flat();
        let no = synthesize_minimum_phase_ir(&c, 48000, 4096, 512, 0, 0.0, false, 18.0).unwrap();
        let yes = synthesize_minimum_phase_ir(&c, 48000, 4096, 512, 0, 0.0, true, 18.0).unwrap();
        let freqs = rfftfreq(4096, 48000.0);
        // The pre-clamp formula, retained verbatim for rates at or above 48 kHz.
        let f_nyq = 24000.0;
        let old_start = f_nyq * (20_200.0 / 24_000.0);
        let old_width = (f_nyq - old_start) / (3_800.0 / 1_650.0);
        for (i, &f) in freqs.iter().enumerate() {
            let x = (f - old_start) / old_width;
            let apod_db = if f > old_start { -3.0103 * x * x } else { 0.0 };
            let want = no.mag_lin[i] * 10f64.powf(apod_db / 20.0);
            assert_eq!(
                yes.mag_lin[i], want,
                "48 kHz bin {f} Hz: the scale clamp must not perturb shipped 48 kHz IRs"
            );
        }
    }

    #[test]
    fn gauss_apodize_params_clamps_below_48khz_and_scales_above() {
        let (s44, w44) = gauss_apodize_params(44_100);
        assert_eq!(s44, 20_200.0);
        assert_eq!(w44, 1_650.0);

        let (s48, w48) = gauss_apodize_params(48_000);
        assert!((s48 - 20_200.0).abs() < 1e-12);
        assert!((w48 - 1_650.0).abs() < 1e-12);

        let (s96, w96) = gauss_apodize_params(96_000);
        assert!((s96 - 40_400.0).abs() < 1e-12);
        assert!((w96 - 3_300.0).abs() < 1e-12);
    }

    #[test]
    fn notch_blend_weight_reaches_full_weight_on_narrow_bands_and_guards_non_finite() {
        // Guard against non-finite values leaking NaN
        assert_eq!(notch_blend_weight(f64::NAN, 3000.0, 5000.0, 0.20), 0.0);
        assert_eq!(notch_blend_weight(4000.0, f64::NAN, 5000.0, 0.20), 0.0);
        assert_eq!(notch_blend_weight(4000.0, 3000.0, f64::NAN, 0.20), 0.0);
        assert_eq!(notch_blend_weight(4000.0, 3000.0, 5000.0, f64::NAN), 0.0);

        // Narrow band [3900, 4100] (0.072 octaves < 0.20 octaves) must reach full w = 1.0 at geometric center
        let f_center = (3900.0 * 4100.0_f64).sqrt();
        let w = notch_blend_weight(f_center, 3900.0, 4100.0, 0.20);
        assert!(
            (w - 1.0).abs() < 1e-12,
            "narrow band peak weight must be 1.0, got {w}"
        );

        // Outside band must taper to 0.0
        assert_eq!(notch_blend_weight(3500.0, 3900.0, 4100.0, 0.20), 0.0);
        assert_eq!(notch_blend_weight(4500.0, 3900.0, 4100.0, 0.20), 0.0);
    }

    #[test]
    fn dc_evaluates_from_lowest_frequency_knot_regardless_of_knot_order() {
        let descending = vec![(20000.0, 10.0), (1000.0, 5.0), (20.0, -3.0)];
        let ascending = vec![(20.0, -3.0), (1000.0, 5.0), (20000.0, 10.0)];
        let res_desc =
            synthesize_minimum_phase_ir(&descending, 48000, 4096, 512, 0, 0.0, false, 18.0)
                .unwrap();
        let res_asc =
            synthesize_minimum_phase_ir(&ascending, 48000, 4096, 512, 0, 0.0, false, 18.0).unwrap();
        assert!(
            (res_desc.mag_lin[0] - res_asc.mag_lin[0]).abs() < 1e-12,
            "DC bin must match between sorted and unsorted knots: desc={}, asc={}",
            res_desc.mag_lin[0],
            res_asc.mag_lin[0]
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
        assert_eq!(safe[2].1, 1.499_9, "g=8 capped in core band");
        assert_eq!(safe[1].1, 1.485_6, "g=4 capped in core band");
        assert_eq!(safe[3].1, 1.479_6, "g=2 smoothly blended on skirt");
        assert!(
            safe[1].1 > 1.48 && safe[3].1 < 2.0,
            "the cap is monotonic in the requested boost: {:?} {:?}",
            safe[1].1,
            safe[3].1
        );
        let zero = apply_notch_mode(&c, NotchMode::Zero, 1.5, range).unwrap();
        assert!(zero[1].1 <= 0.001);
        assert_eq!(zero[2], (3703.0, 0.0));
        let none_range = apply_notch_mode(&c, NotchMode::Safe, 1.5, None).unwrap();
        assert_eq!(
            none_range, c,
            "a model with no notch range is never touched"
        );
    }

    #[test]
    fn notch_blend_weight_is_smooth_and_avoids_discontinuities() {
        let f_lo = 4300.0;
        let f_hi = 6300.0;
        let skirt = 0.20;
        // Outside the skirts
        assert_eq!(notch_blend_weight(3500.0, f_lo, f_hi, skirt), 0.0);
        assert_eq!(notch_blend_weight(7500.0, f_lo, f_hi, skirt), 0.0);
        // Core band
        assert_eq!(notch_blend_weight(5000.0, f_lo, f_hi, skirt), 1.0);
        assert_eq!(notch_blend_weight(5365.9, f_lo, f_hi, skirt), 1.0);
        // Center of edges has weight exactly 0.5
        let w_lo = notch_blend_weight(f_lo, f_lo, f_hi, skirt);
        let w_hi = notch_blend_weight(f_hi, f_lo, f_hi, skirt);
        assert!((w_lo - 0.5).abs() < 1e-6);
        assert!((w_hi - 0.5).abs() < 1e-6);
        // Monotonic ramp in lower skirt
        let w1 = notch_blend_weight(4100.0, f_lo, f_hi, skirt);
        let w2 = notch_blend_weight(4200.0, f_lo, f_hi, skirt);
        assert!(w1 < w2 && w2 < w_lo);

        // Narrow band test (5000 to 5200 Hz, span < skirt_oct): continuous, bounded, no cliff
        let mut prev = notch_blend_weight(4500.0, 5000.0, 5200.0, skirt);
        for f in (4501..6000).step_by(2) {
            let w = notch_blend_weight(f as f64, 5000.0, 5200.0, skirt);
            assert!(
                (w - prev).abs() < 0.02,
                "narrow band discontinuity at {f} Hz: prev={prev}, curr={w}"
            );
            assert!((0.0..=1.0).contains(&w));
            prev = w;
        }
    }

    /// A ceiling that is not a positive finite number is not a ceiling. The guard read
    /// `safe_cap_db <= 0.0`, which NaN fails outright, and `+inf * tanh(g / +inf)` is `inf * 0` —
    /// so both non-finite caps passed the guard and painted NaN over every knot in the notch band,
    /// which the filter then carried into its coefficients.
    #[test]
    fn a_non_finite_safe_ceiling_is_refused_rather_than_spreading_nan() {
        let c = [(3703.0, 8.0), (4000.0, 2.0)];
        let range = Some((2800.0, 4400.0));
        for cap in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.5] {
            let e = apply_notch_mode(&c, NotchMode::Safe, cap, range)
                .err()
                .unwrap_or_else(|| panic!("a ceiling of {cap} was accepted"));
            assert!(
                e.to_string().contains("safe_cap_db"),
                "the refusal did not name the field: {e}"
            );
        }
        // When mode is Safe and a range is set, safe_cap_db must be validated before the loop
        // even if knots inside the band have non-positive gain or sit outside the range:
        let non_positive_knots = [(3703.0, -2.0), (4000.0, 0.0)];
        assert!(apply_notch_mode(&non_positive_knots, NotchMode::Safe, -5.0, range).is_err());
        assert!(apply_notch_mode(&non_positive_knots, NotchMode::Safe, f64::NAN, range).is_err());
        let outside = [(1000.0, 8.0)];
        assert!(apply_notch_mode(&outside, NotchMode::Safe, f64::NAN, range).is_err());
        assert_eq!(
            apply_notch_mode(&c, NotchMode::Original, f64::NAN, range).unwrap(),
            c,
            "original mode never consults the ceiling"
        );
        assert_eq!(
            apply_notch_mode(&c, NotchMode::Safe, f64::NAN, None).unwrap(),
            c,
            "no notch range never consults the ceiling"
        );
    }

    #[test]
    fn notch_mode_round_trips_names() {
        for name in NotchMode::CHOICES {
            let m = NotchMode::parse(name).expect("a choice names a mode");
            assert_eq!(m.as_str(), name);
            assert_eq!(m.to_string(), name);
        }
        assert!(
            NotchMode::parse("  safe ").is_some(),
            "surrounding space is trimmed"
        );
        assert!(
            NotchMode::parse("SAFE").is_some(),
            "case is folded for library callers; the CLI validates exactly"
        );
        assert!(NotchMode::parse("aggressive").is_none());
    }

    /// `--notch-mode` trusts `CHOICES` to be the whole vocabulary: the argument phase rejects
    /// anything outside it, and the mode is then read with an infallible fallback instead of a
    /// second error message. That is only honest if the list and the variants cover each other, so
    /// it is checked rather than asserted in a comment.
    #[test]
    fn the_choice_list_and_the_variants_cover_each_other() {
        let parsed: Vec<NotchMode> = NotchMode::CHOICES
            .iter()
            .map(|c| NotchMode::parse(c).unwrap_or_else(|| panic!("'{c}' names nothing")))
            .collect();
        let all = [NotchMode::Safe, NotchMode::Original, NotchMode::Zero];
        assert_eq!(parsed.len(), all.len());
        for m in all {
            assert!(
                parsed.contains(&m),
                "{} is not reachable from CHOICES {:?}",
                m.as_str(),
                NotchMode::CHOICES
            );
        }
        // And nothing in the list folds onto another entry, or two names would mean one mode.
        let names: Vec<&str> = parsed.iter().map(|m| m.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "two choices name one mode");
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
        // Zero taps is not an error here: `(0*4 - 1)` underflows to a huge width in the
        // reference's own arithmetic, so the floor is the answer. Pinned because the
        // snippet is printed even for a bypass run whose filter is trivially short.
        assert_eq!(mpv_n_fft(0), 65536, "zero taps lands on the floor");
        assert!(
            mpv_n_fft(usize::MAX) >= 65536,
            "an absurd tap count saturates instead of wrapping"
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

    /// A raw curve can carry the two ears as `[freq, left, right]`. This table is what a reviewer
    /// diffs the shipped filter against, so a row scored by its last column prints one ear under a
    /// heading the rest of the report computes as the ear mean.
    #[test]
    fn the_audit_table_scores_a_two_ear_raw_curve_by_its_ear_mean() {
        let dir = std::env::temp_dir().join(format!("hypreq_audit_ears_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let raw = vec![
            Point::from(vec![20.0, -6.0, -2.0]),
            Point::from(vec![1000.0, 1.0, 3.0]),
        ];
        let target = vec![(20.0, 0.0), (1000.0, 0.0)];
        let ctrl = vec![(20.0, 0.0), (1000.0, 0.0)];
        let p = export_audit_csv_in(&dir, "earmodel", &raw, &target, 0.0, &ctrl).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        let raw_dbr = |line: &str| {
            line.split(',')
                .nth(1)
                .unwrap_or_else(|| panic!("no raw column in {line:?}"))
                .trim()
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("raw column is not a number in {line:?}: {e}"))
        };
        let rows: Vec<&str> = text
            .lines()
            .skip(1)
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(rows.len(), 2, "{text}");
        assert_eq!(
            raw_dbr(rows[0]),
            -4.0,
            "the 20 Hz row must be the ear mean, not the right ear"
        );
        assert_eq!(
            raw_dbr(rows[1]),
            2.0,
            "the 1 kHz row must be the ear mean, not the right ear"
        );
        std::fs::remove_dir_all(&dir).ok();
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
        assert!(text.starts_with("frequency_hz,raw_dBr,target_dBr,raw_error_dB,applied_eq_dB\r\n"));
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
