//! Complex-IR / excess-phase equalization combining:
//!
//! * **Complex fractional-octave smoothing** — P. D. Hatziantoniou & J. N. Mourjopoulos,
//!   *"Generalized Fractional-Octave Smoothing of Audio and Acoustic Responses,"* JAES 48(4),
//!   pp. 259–280 (April 2000; addendum p. 940). Smooths complex transfer functions directly so
//!   narrow reverberant/pinna nulls are attenuated while stable excess-phase structure survives.
//! * **Multi-capture spatial complex coherence gating** — motivated by B. D. Radlovic,
//!   R. C. Williamson & R. A. Kennedy, *"Equalization in an Acoustic Reverberant Environment:
//!   Robustness Results,"* IEEE Trans. Speech and Audio Processing 8(3), pp. 311–319 (May 2000),
//!   DOI `10.1109/89.841287`. Repeated time-of-flight-aligned complex IR captures yield spatial
//!   coherence `gamma^2(f) = |E[H(f)]|^2 / E[|H(f)|^2] in [0, 1]`, driving frequency-dependent
//!   regularization so placement-volatile high-frequency excess phase is never inverted while
//!   placement-stable excess phase (e.g. a multi-way crossover) is inverted safely.
//! * **Causal regularized complex inversion with modeling delay** — O. Kirkeby, P. A. Nelson,
//!   H. Hamada & F. Orduña-Bustamante, *"Fast Deconvolution of Multichannel Systems Using
//!   Regularization,"* IEEE TSAP 6(2), pp. 189–194 (March 1998), and O. Kirkeby & P. A. Nelson,
//!   *"Digital Filter Design for Inversion Problems in Sound Reproduction,"* JAES 47(3),
//!   pp. 107–126 (March 1999):
//!   `G(w) = H^*(w) * D(w) * exp(-j w Delta) / (|H(w)|^2 + beta(w))`.
//!
//! Prior art vs. HyprEQ design choices:
//! * The Kirkeby–Nelson inversion formula with frequency-dependent `beta(w)` and modeling delay
//!   `Delta`, and complex fractional-octave smoothing on `H(e^{jw})`, follow the published papers.
//! * The specific bulk time-of-flight cross-correlation alignment, additive coherence penalty
//!   `beta(w) = beta_in + beta_out * w_oob(w) + beta_incoh * (1 - gamma^2(w))`, passband RMS
//!   normalization, and two-sided half-Kaiser FIR windowing are HyprEQ's engineering design.

use crate::accum::{sum_of_squares, RunningSum};
use crate::fft::{irfft, rfft, rfftfreq, Complex64};
use crate::ir::{db_floor, IrError};
use crate::pchip::Pchip;
use crate::window::half_kaiser_tail_apply;

/// Configuration for [`design_kirkeby_inverse`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexEqConfig {
    /// Minimum FFT length (auto-bumped to a power of two `>= max(2 * max_ir_len, 2 * n_taps)`).
    pub n_fft: usize,
    /// Output causal FIR length in samples.
    pub n_taps: usize,
    /// Half-Kaiser edge fade length (samples) applied symmetrically at leading and trailing edges.
    pub n_tail: usize,
    /// Causal modeling delay `Delta` in milliseconds so non-minimum-phase (acausal) inverse
    /// energy lands cleanly inside `[0..n_taps]`.
    pub delay_ms: f64,
    /// Hatziantoniou & Mourjopoulos complex fractional-octave smoothing width in octaves
    /// (`0.0` disables complex smoothing; `1.0 / 6.0` is 1/6-octave).
    pub smooth_oct: f64,
    /// Dimensionless in-band Kirkeby–Nelson regularization `beta_in` (relative to passband RMS power).
    pub beta_in: f64,
    /// Dimensionless out-of-band Kirkeby–Nelson regularization `beta_out`.
    pub beta_out: f64,
    /// Dimensionless Radlovic spatial-incoherence penalty `beta_incoh` multiplying `1 - gamma^2(f)`.
    pub beta_incoh: f64,
    /// Lower passband edge (Hz).
    pub f_lo_hz: f64,
    /// Upper passband edge (Hz).
    pub f_hi_hz: f64,
    /// Steady-state headroom ceiling (dBFS) applied to peak passband filter gain.
    pub headroom_db: f64,
    /// Kaiser shape parameter for leading/trailing FIR edge fades.
    pub kaiser_beta: f64,
}

impl Default for ComplexEqConfig {
    fn default() -> Self {
        Self {
            n_fft: 32768,
            n_taps: 16384,
            n_tail: 4096,
            delay_ms: 40.0,
            smooth_oct: 1.0 / 6.0,
            beta_in: 0.01,
            beta_out: 0.25,
            beta_incoh: 2.0,
            f_lo_hz: 20.0,
            f_hi_hz: 20000.0,
            headroom_db: -1.50,
            kaiser_beta: 16.0,
        }
    }
}

/// Output and verification diagnostics of [`design_kirkeby_inverse`].
#[derive(Clone, Debug)]
pub struct ComplexEqResult {
    /// Causal mixed-phase equalizer FIR impulse response of length `n_taps`.
    pub ir: Vec<f64>,
    /// Sample rate (Hz).
    pub fs: u32,
    /// Applied integer modeling delay `Delta` (samples).
    pub delay_samples: usize,
    /// Applied modeling delay `Delta` (ms).
    pub delay_ms: f64,
    /// Number of complex IR captures combined.
    pub capture_count: usize,
    /// Per-capture integer bulk time-of-flight shifts removed prior to spatial averaging.
    pub aligned_shifts_samples: Vec<isize>,
    /// Mean spatial coherence `gamma^2` across the passband `[f_lo, f_hi]`.
    pub mean_coherence: f64,
    /// Minimum spatial coherence `gamma^2` across the passband `[f_lo, f_hi]`.
    pub min_coherence: f64,
    /// Peak passband equalizer boost before headroom normalization (dB).
    pub max_boost_db: f64,
    /// Applied steady-state preamp attenuation (dB).
    pub preamp_db: f64,
    /// Peak-to-peak group-delay ripple of the smoothed plant `H` over 100–8000 Hz before EQ (ms).
    pub pre_gd_ripple_ms: f64,
    /// Peak-to-peak group-delay ripple of the equalized plant `H * G` over 100–8000 Hz after EQ (ms).
    pub post_gd_ripple_ms: f64,
    /// Leading-edge decay floor over the first 100 taps (dBFS).
    pub leading_floor_dbfs: f64,
    /// Trailing-edge decay floor over the last 100 taps (dBFS).
    pub trailing_floor_dbfs: f64,
    /// Percentage of full `n_fft` inverse energy retained inside `[0..n_taps]`.
    pub retained_energy_pct: f64,
}

/// Circularly align a set of complex impulse responses to remove bulk time-of-flight offsets.
///
/// The first capture is anchored so its maximum-amplitude sample sits at index `0` (mod `n_fft`),
/// preserving any pre-arrival energy circularly. Each subsequent capture is shifted by the lag
/// maximizing its wideband circular cross-correlation with the anchored reference.
#[must_use]
pub fn align_to_reference(irs: &[Vec<f64>], n_fft: usize) -> (Vec<Vec<f64>>, Vec<isize>) {
    if irs.is_empty() || n_fft < 4 {
        return (Vec::new(), Vec::new());
    }
    let mut aligned = Vec::with_capacity(irs.len());
    let mut shifts = Vec::with_capacity(irs.len());

    // Anchor capture 0 on its main arrival peak so arbitrary recording pre-delay does not consume
    // modeling delay or wrap phase unnecessarily.
    let ref_peak = irs[0]
        .iter()
        .take(n_fft)
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map_or(0, |(idx, _)| idx);
    let mut ref_buf = vec![0.0_f64; n_fft];
    for (i, &v) in irs[0].iter().take(n_fft).enumerate() {
        ref_buf[(i + n_fft - ref_peak) % n_fft] = v;
    }
    let ref_shift = if ref_peak <= n_fft / 2 {
        ref_peak as isize
    } else {
        ref_peak as isize - n_fft as isize
    };
    let ref_spec = rfft(&ref_buf, n_fft);
    aligned.push(ref_buf);
    shifts.push(ref_shift);

    for ir in irs.iter().skip(1) {
        let mut pad = vec![0.0_f64; n_fft];
        let copy_len = ir.len().min(n_fft);
        pad[..copy_len].copy_from_slice(&ir[..copy_len]);
        let spec = rfft(&pad, n_fft);
        let cross: Vec<Complex64> = spec
            .iter()
            .zip(&ref_spec)
            .map(|(&h, &r)| h * r.conj())
            .collect();
        // Maximize signed cross-correlation (not abs) so a true 180-deg polarity flip is
        // preserved as phase scatter for Radlovic spatial coherence gating.
        let cc = irfft(&cross, n_fft);
        let lag = cc
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(idx, _)| idx);
        let signed_lag = if lag <= n_fft / 2 {
            lag as isize
        } else {
            lag as isize - n_fft as isize
        };
        let mut shifted = vec![0.0_f64; n_fft];
        for (i, &v) in pad.iter().enumerate() {
            shifted[(i + n_fft - lag) % n_fft] = v;
        }
        aligned.push(shifted);
        shifts.push(signed_lag);
    }

    (aligned, shifts)
}

/// Hatziantoniou & Mourjopoulos (JAES 2000) generalized complex fractional-octave smoothing.
///
/// Applies a centered, strictly positive raised-cosine (Hann) window over `[f * 2^(-oct/2),
/// f * 2^(+oct/2)]` directly to the complex half-spectrum `spec`, preserving real DC and Nyquist
/// bins. When `oct <= 0.0` or non-finite, returns `spec` unchanged.
#[must_use]
pub fn smooth_complex_octave(spec: &[Complex64], freqs: &[f64], oct: f64) -> Vec<Complex64> {
    let n = spec.len();
    if n == 0 || freqs.len() != n || !oct.is_finite() || oct <= 0.0 {
        return spec.to_vec();
    }
    if n <= 2 {
        return spec.iter().map(|c| Complex64::new(c.re, 0.0)).collect();
    }
    let df = freqs[1] - freqs[0];
    if !df.is_finite() || df <= 0.0 {
        return spec.to_vec();
    }

    let half_ratio = 2.0_f64.powf(0.5 * oct);
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    out[0] = Complex64::new(spec[0].re, 0.0);
    out[n - 1] = Complex64::new(spec[n - 1].re, 0.0);

    for k in 1..n - 1 {
        let f = freqs[k];
        let k_lo = ((f / half_ratio) / df).ceil().max(1.0) as usize;
        let k_hi = ((f * half_ratio) / df).floor().min((n - 2) as f64) as usize;
        let lo = k_lo.min(k);
        let hi = k_hi.max(k);
        if lo == hi {
            out[k] = spec[k];
            continue;
        }
        let half_span = (k - lo).max(hi - k) as f64;
        let denom_span = half_span + 1.0;
        let mut sum_re = RunningSum::new();
        let mut sum_im = RunningSum::new();
        let mut sum_w = RunningSum::new();
        for (m, &c) in spec.iter().enumerate().take(hi + 1).skip(lo) {
            let d = (m as f64 - k as f64).abs();
            let w = 0.5 * (1.0 + (std::f64::consts::PI * d / denom_span).cos());
            sum_re.add(w * c.re);
            sum_im.add(w * c.im);
            sum_w.add(w);
        }
        let w_tot = sum_w.total();
        if w_tot > 0.0 {
            out[k] = Complex64::new(sum_re.total() / w_tot, sum_im.total() / w_tot);
        } else {
            out[k] = spec[k];
        }
    }
    out
}

/// Multi-capture complex mean `E[H(f)]` and spatial coherence
/// `gamma^2(f) = |E[H(f)]|^2 / E[|H(f)|^2] in [0, 1]`.
///
/// Motivated by Radlovic, Williamson & Kennedy (IEEE TSAP 8(3), 2000): frequency regions where
/// repeated seatings agree in both magnitude and phase approach `gamma^2 -> 1`, while regions
/// dominated by seating-dependent phase scatter drop toward `gamma^2 -> 0`.
#[must_use]
pub fn spatial_coherence(spectra: &[Vec<Complex64>]) -> (Vec<Complex64>, Vec<f64>) {
    if spectra.is_empty() || spectra[0].is_empty() {
        return (Vec::new(), Vec::new());
    }
    let m = spectra.len() as f64;
    let n_bins = spectra[0].len();
    let mut mean_spec = vec![Complex64::new(0.0, 0.0); n_bins];
    let mut gamma_sq = vec![0.0_f64; n_bins];

    for k in 0..n_bins {
        let mut re = RunningSum::new();
        let mut im = RunningSum::new();
        let mut pow = RunningSum::new();
        for s in spectra {
            if let Some(&c) = s.get(k) {
                re.add(c.re);
                im.add(c.im);
                pow.add(c.norm_sqr());
            }
        }
        let c_mean = Complex64::new(re.total() / m, im.total() / m);
        let p_mean = pow.total() / m;
        mean_spec[k] = c_mean;
        gamma_sq[k] = if p_mean > 1e-30 {
            (c_mean.norm_sqr() / p_mean).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    (mean_spec, gamma_sq)
}

/// Smooth raised-cosine out-of-band weight: `0.0` inside `[f_lo, f_hi]`, easing to `1.0` outside.
fn out_of_band_weight(f: f64, f_lo: f64, f_hi: f64, nyquist: f64) -> f64 {
    if f >= f_lo && f <= f_hi {
        return 0.0;
    }
    if f < f_lo {
        if f <= 0.0 {
            return 1.0;
        }
        // Smooth raised-cosine transition across the full [0, f_lo] span maximizes transition
        // bandwidth and minimizes low-frequency pre-ringing inside finite modeling delays.
        let t = (f_lo - f) / f_lo;
        return 0.5 * (1.0 - (std::f64::consts::PI * t).cos());
    }
    let f_top = (f_hi * 1.25).min(nyquist);
    if f_top <= f_hi || f >= f_top {
        return 1.0;
    }
    let t = (f - f_hi) / (f_top - f_hi);
    0.5 * (1.0 - (std::f64::consts::PI * t).cos())
}

/// Measure peak-to-peak group delay (ms) of a complex half-spectrum over `[f_lo, f_hi]`, skipping
/// near-null bins below `5%` of passband peak magnitude.
fn group_delay_ripple_ms(spec: &[Complex64], freqs: &[f64], f_lo: f64, f_hi: f64) -> f64 {
    let n = spec.len();
    if n < 3 || freqs.len() != n {
        return 0.0;
    }
    let mut max_mag = 0.0_f64;
    for (&c, &f) in spec.iter().zip(freqs) {
        if f >= f_lo && f <= f_hi {
            max_mag = max_mag.max(c.norm());
        }
    }
    if max_mag <= 1e-18 {
        return 0.0;
    }
    let floor = max_mag * 0.05;

    // Unwrap phase bin-by-bin across valid bins.
    let mut unwrapped = vec![0.0_f64; n];
    let mut valid = vec![false; n];
    let mut prev = 0.0_f64;
    let mut carry = 0.0_f64;
    let mut started = false;
    for k in 0..n {
        if spec[k].norm() > floor {
            let raw = spec[k].im.atan2(spec[k].re);
            if started {
                let mut delta = raw - prev;
                while delta > std::f64::consts::PI {
                    delta -= std::f64::consts::TAU;
                    carry -= std::f64::consts::TAU;
                }
                while delta < -std::f64::consts::PI {
                    delta += std::f64::consts::TAU;
                    carry += std::f64::consts::TAU;
                }
            }
            prev = raw;
            started = true;
            unwrapped[k] = raw + carry;
            valid[k] = true;
        } else {
            unwrapped[k] = prev + carry;
        }
    }

    let mut min_gd = f64::INFINITY;
    let mut max_gd = f64::NEG_INFINITY;
    for k in 1..n - 1 {
        let f = freqs[k];
        if f < f_lo || f > f_hi || !valid[k - 1] || !valid[k] || !valid[k + 1] {
            continue;
        }
        let df = freqs[k + 1] - freqs[k - 1];
        if df <= 0.0 {
            continue;
        }
        let gd_ms = -(unwrapped[k + 1] - unwrapped[k - 1]) / df / std::f64::consts::TAU * 1000.0;
        if gd_ms.is_finite() {
            min_gd = min_gd.min(gd_ms);
            max_gd = max_gd.max(gd_ms);
        }
    }
    if min_gd.is_finite() && max_gd.is_finite() {
        (max_gd - min_gd).max(0.0)
    } else {
        0.0
    }
}

/// Design a causal mixed-phase equalizer IR from one or more measured complex impulse responses.
///
/// Combines bulk ToF alignment, Hatziantoniou–Mourjopoulos complex fractional-octave smoothing,
/// Radlovic multi-capture spatial coherence gating, and Kirkeby–Nelson regularized complex
/// inversion with causal modeling delay `Delta`.
pub fn design_kirkeby_inverse(
    raw_irs: &[Vec<f64>],
    target_ctrl: Option<&[(f64, f64)]>,
    fs: u32,
    cfg: &ComplexEqConfig,
) -> Result<ComplexEqResult, IrError> {
    if fs == 0 {
        return Err(IrError("sample rate must be positive (got 0)".to_string()));
    }
    if raw_irs.is_empty() {
        return Err(IrError(
            "complex EQ requires at least one measured impulse response".to_string(),
        ));
    }
    if cfg.n_taps < 4 {
        return Err(IrError(format!(
            "n_taps must be >= 4 for complex EQ (got {})",
            cfg.n_taps
        )));
    }
    if !cfg.delay_ms.is_finite() || cfg.delay_ms < 0.0 {
        return Err(IrError(format!(
            "delay_ms must be finite and >= 0 (got {})",
            cfg.delay_ms
        )));
    }
    if !cfg.smooth_oct.is_finite() || cfg.smooth_oct < 0.0 {
        return Err(IrError(format!(
            "smooth_oct must be finite and >= 0 (got {})",
            cfg.smooth_oct
        )));
    }
    if !cfg.headroom_db.is_finite() {
        return Err(IrError(format!(
            "headroom_db must be finite (got {})",
            cfg.headroom_db
        )));
    }
    if !cfg.kaiser_beta.is_finite() || cfg.kaiser_beta < 0.0 {
        return Err(IrError(format!(
            "kaiser_beta must be finite and >= 0 (got {})",
            cfg.kaiser_beta
        )));
    }
    if !cfg.f_lo_hz.is_finite()
        || !cfg.f_hi_hz.is_finite()
        || cfg.f_lo_hz <= 0.0
        || cfg.f_hi_hz <= cfg.f_lo_hz
    {
        return Err(IrError(format!(
            "passband edges must be finite with 0 < f_lo_hz < f_hi_hz (got {}, {})",
            cfg.f_lo_hz, cfg.f_hi_hz
        )));
    }
    if !cfg.beta_in.is_finite()
        || cfg.beta_in < 0.0
        || !cfg.beta_out.is_finite()
        || cfg.beta_out < 0.0
        || !cfg.beta_incoh.is_finite()
        || cfg.beta_incoh < 0.0
    {
        return Err(IrError(
            "Kirkeby–Nelson regularization parameters must be finite and non-negative".to_string(),
        ));
    }

    let mut max_len = 0usize;
    for (idx, ir) in raw_irs.iter().enumerate() {
        if ir.is_empty() {
            return Err(IrError(format!(
                "measured impulse response #{idx} is empty"
            )));
        }
        if ir.iter().any(|v| !v.is_finite()) {
            return Err(IrError(format!(
                "measured impulse response #{idx} contains non-finite samples"
            )));
        }
        let peak = ir.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        if peak <= 1e-18 {
            return Err(IrError(format!(
                "measured impulse response #{idx} is silent (zero energy)"
            )));
        }
        max_len = max_len.max(ir.len());
    }

    let n_fft = cfg
        .n_fft
        .max(max_len.saturating_mul(2))
        .max(cfg.n_taps.saturating_mul(2))
        .next_power_of_two()
        .max(64);
    let freqs = rfftfreq(n_fft, fs as f64);
    let nyq = (fs as f64) * 0.5;
    let f_lo = cfg.f_lo_hz.clamp(1.0, nyq * 0.5);
    let f_hi = cfg.f_hi_hz.clamp(f_lo + 1.0, nyq);

    // 1. Bulk time-of-flight alignment across captures.
    let (aligned, shifts) = align_to_reference(raw_irs, n_fft);

    // 2. Complex FFT + Hatziantoniou–Mourjopoulos fractional-octave complex smoothing.
    let mut spectra: Vec<Vec<Complex64>> = Vec::with_capacity(aligned.len());
    for ir in &aligned {
        let raw_spec = rfft(ir, n_fft);
        let smooth_spec = smooth_complex_octave(&raw_spec, &freqs, cfg.smooth_oct);
        spectra.push(smooth_spec);
    }

    // 3. Multi-capture spatial complex coherence (Radlovic et al. 2000).
    let (h_mean, coh) = spatial_coherence(&spectra);

    // 4. Passband RMS normalization so regularization betas are dimensionless and level-invariant.
    let mut ref_pow = RunningSum::new();
    let mut ref_cnt = 0usize;
    let norm_lo = f_lo.max(200.0).min(f_hi);
    let norm_hi = f_hi.min(4000.0).max(norm_lo);
    for (&c, &f) in h_mean.iter().zip(&freqs) {
        if f >= norm_lo && f <= norm_hi {
            ref_pow.add(c.norm_sqr());
            ref_cnt += 1;
        }
    }
    if ref_cnt == 0 {
        for (&c, &f) in h_mean.iter().zip(&freqs) {
            if f >= f_lo && f <= f_hi {
                ref_pow.add(c.norm_sqr());
                ref_cnt += 1;
            }
        }
    }
    let ref_mag = if ref_cnt > 0 {
        (ref_pow.total() / (ref_cnt as f64)).sqrt()
    } else {
        0.0
    };
    if !ref_mag.is_finite() || ref_mag <= 1e-18 {
        return Err(IrError(
            "measured impulse response has zero passband energy".to_string(),
        ));
    }
    let h_norm: Vec<Complex64> = h_mean.iter().map(|&c| c / ref_mag).collect();

    // 5. Optional desired target magnitude response D(f) (zero-phase real target, normalized at 1 kHz).
    let d_mag: Vec<f64> = match target_ctrl {
        Some(pts) => {
            let clean: Vec<(f64, f64)> = pts
                .iter()
                .copied()
                .filter(|&(f, g)| f.is_finite() && f > 0.0 && g.is_finite())
                .collect();
            let sorted = crate::model::sort_and_dedup_pairs(&clean);
            if sorted.len() >= 2 {
                let (tf, tg): (Vec<f64>, Vec<f64>) = sorted.into_iter().unzip();
                let curve = Pchip::new(&tf, &tg);
                let ref_db = curve.eval(1000.0);
                let offset = if ref_db.is_finite() { ref_db } else { 0.0 };
                freqs
                    .iter()
                    .map(|&f| {
                        let db = curve.eval(f) - offset;
                        10.0_f64.powf(db / 20.0)
                    })
                    .collect()
            } else {
                vec![1.0_f64; freqs.len()]
            }
        }
        None => vec![1.0_f64; freqs.len()],
    };

    // 6. Kirkeby–Nelson (1998/1999) causal regularized complex inverse with modeling delay Delta.
    let raw_delay = ((cfg.delay_ms / 1000.0) * (fs as f64)).round() as usize;
    let delay_samples = raw_delay.min(cfg.n_taps.saturating_sub(2));
    let n_bins = freqs.len();
    let mut g_spec = Vec::with_capacity(n_bins);
    let mut max_boost_db = f64::NEG_INFINITY;
    let mut coh_sum = RunningSum::new();
    let mut coh_min = 1.0_f64;
    let mut coh_cnt = 0usize;

    for k in 0..n_bins {
        let f = freqs[k];
        let w_oob = out_of_band_weight(f, f_lo, f_hi, nyq);
        let incoh = if raw_irs.len() >= 2 {
            1.0 - coh[k]
        } else {
            0.0
        };
        let beta = (cfg.beta_in + cfg.beta_out * w_oob + cfg.beta_incoh * incoh).max(1e-12);
        let h = h_norm[k];
        let denom = h.norm_sqr() + beta;
        let angle = -std::f64::consts::TAU * (k as f64) * (delay_samples as f64) / (n_fft as f64);
        let phase_shift = Complex64::new(angle.cos(), angle.sin());
        let mut g = (h.conj() * d_mag[k] / denom) * phase_shift;
        if k == 0 || k == n_bins - 1 {
            g.im = 0.0;
        }
        if f >= f_lo && f <= f_hi {
            let mag = g.norm();
            if mag > 1e-18 {
                max_boost_db = max_boost_db.max(20.0 * mag.log10());
            }
            coh_sum.add(coh[k]);
            coh_min = coh_min.min(coh[k]);
            coh_cnt += 1;
        }
        g_spec.push(g);
    }
    if !max_boost_db.is_finite() {
        max_boost_db = 0.0;
    }

    // 7. Normalize to steady-state headroom ceiling, transform to time domain, window edges.
    let preamp_db = cfg.headroom_db - max_boost_db.max(0.0);
    let scale = 10.0_f64.powf(preamp_db / 20.0);
    let g_scaled: Vec<Complex64> = g_spec.iter().map(|&c| c * scale).collect();
    let full_ir = irfft(&g_scaled, n_fft);
    let energy_full = sum_of_squares(&full_ir);

    let mut ir = full_ir[..cfg.n_taps].to_vec();
    let lead_tail = cfg.n_tail.min(delay_samples / 2);
    let trail_tail = cfg.n_tail.min(cfg.n_taps.saturating_sub(delay_samples) / 2);
    if lead_tail > 0 {
        ir[..lead_tail].reverse();
        half_kaiser_tail_apply(&mut ir[..lead_tail], lead_tail, cfg.kaiser_beta);
        ir[..lead_tail].reverse();
    }
    if trail_tail > 0 {
        half_kaiser_tail_apply(&mut ir, trail_tail, cfg.kaiser_beta);
    }

    let energy_in = sum_of_squares(&ir);
    let retained_energy_pct = if energy_full > 0.0 {
        (energy_in / energy_full * 100.0).min(100.0)
    } else {
        100.0
    };
    let leading_floor_dbfs = db_floor(&ir[..ir.len().min(100)], 100);
    let trailing_floor_dbfs = db_floor(&ir, 100);

    // 8. Measure pre-EQ vs post-EQ group-delay ripple across 100–8000 Hz.
    let gd_lo = f_lo.max(100.0).min(f_hi);
    let gd_hi = f_hi.min(8000.0).max(gd_lo);
    let pre_gd_ripple_ms = group_delay_ripple_ms(&h_norm, &freqs, gd_lo, gd_hi);
    // Demodulate the known constant modeling delay exp(+j * 2*pi * k * delay / n_fft) before
    // unwrapping phase: a constant delay leaves peak-to-peak GD ripple invariant while preventing
    // bin-to-bin phase-wrap aliasing even when delay_samples approaches n_taps.
    let g_realized = rfft(&ir, n_fft);
    let equalized_spec: Vec<Complex64> = h_norm
        .iter()
        .zip(&g_realized)
        .enumerate()
        .map(|(k, (&h, &g))| {
            let angle =
                std::f64::consts::TAU * (k as f64) * (delay_samples as f64) / (n_fft as f64);
            h * g * Complex64::new(angle.cos(), angle.sin())
        })
        .collect();
    let post_gd_ripple_ms = group_delay_ripple_ms(&equalized_spec, &freqs, gd_lo, gd_hi);

    let mean_coherence = if coh_cnt > 0 {
        coh_sum.total() / (coh_cnt as f64)
    } else {
        1.0
    };

    Ok(ComplexEqResult {
        ir,
        fs,
        delay_samples,
        delay_ms: (delay_samples as f64) / (fs as f64) * 1000.0,
        capture_count: raw_irs.len(),
        aligned_shifts_samples: shifts,
        mean_coherence,
        min_coherence: if coh_cnt > 0 { coh_min } else { 1.0 },
        max_boost_db,
        preamp_db,
        pre_gd_ripple_ms,
        post_gd_ripple_ms,
        leading_floor_dbfs,
        trailing_floor_dbfs,
        retained_energy_pct,
    })
}

/// Format a crisp diagnostic summary for [`ComplexEqResult`].
#[must_use]
pub fn report_lines(res: &ComplexEqResult) -> Vec<String> {
    let mut lines = vec![
        "=== Complex-IR Mixed-Phase Inversion (Kirkeby–Nelson / Hatziantoniou–Mourjopoulos / Radlovic) ==="
            .to_string(),
        format!(
            "  Captures          : {} IR(s) @ {} Hz (ToF shifts: {:?} samples)",
            res.capture_count, res.fs, res.aligned_shifts_samples
        ),
        format!(
            "  Spatial coherence : mean gamma^2 = {:.3}, min = {:.3} (1.0 = placement-invariant)",
            res.mean_coherence, res.min_coherence
        ),
        format!(
            "  Modeling delay    : {} samples ({:.2} ms) — causal window for excess-phase inverse",
            res.delay_samples, res.delay_ms
        ),
        format!(
            "  Group-delay ripple: {:.3} ms pre-EQ -> {:.3} ms post-EQ (100–8000 Hz)",
            res.pre_gd_ripple_ms, res.post_gd_ripple_ms
        ),
        format!(
            "  Gain / headroom   : max boost {:+.2} dB, preamp {:+.2} dB",
            res.max_boost_db, res.preamp_db
        ),
        format!(
            "  Decay / energy    : lead {:.2} dBFS, trail {:.2} dBFS, retained {:.6}%",
            res.leading_floor_dbfs, res.trailing_floor_dbfs, res.retained_energy_pct
        ),
    ];
    if res.capture_count == 1 {
        lines.push(
            "  Note              : single IR capture supplied — spatial coherence gating inactive; supply 2+ seatings to gate HF pinna/placement jitter".to_string()
        );
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesize a causal 2nd-order all-pass impulse response (flat unity magnitude, pure excess
    /// phase with a zero outside the unit circle at `r^-1 e^{+-j w0}`).
    fn allpass_ir(fs: f64, f0_hz: f64, q: f64, len: usize) -> Vec<f64> {
        let w0 = std::f64::consts::TAU * f0_hz / fs;
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        let b0 = (1.0 - alpha) / a0;
        let b1 = (-2.0 * w0.cos()) / a0;
        let b2 = (1.0 + alpha) / a0;
        let a1 = b1;
        let a2 = b0;

        let mut y = vec![0.0_f64; len];
        let (mut x1, mut x2) = (0.0, 0.0);
        let (mut y1, mut y2) = (0.0, 0.0);
        for (n, yn) in y.iter_mut().enumerate() {
            let x0 = if n == 0 { 1.0 } else { 0.0 };
            let out = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            *yn = out;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = out;
        }
        y
    }

    #[test]
    fn kirkeby_inverse_cancels_allpass_excess_phase_and_flattens_group_delay() {
        // Pure 2nd-order all-pass at 1 kHz, Q = 2.0: magnitude is 0 dB everywhere (so a
        // minimum-phase magnitude EQ does NOTHING to it), while group delay peaks strongly.
        // Kirkeby–Nelson complex inversion with causal modeling delay must cancel the excess
        // phase and collapse the group-delay ripple.
        let fs = 48000;
        let plant = allpass_ir(fs as f64, 1000.0, 2.0, 2048);
        let cfg = ComplexEqConfig {
            n_fft: 16384,
            n_taps: 8192,
            n_tail: 1024,
            delay_ms: 40.0,
            smooth_oct: 0.0,
            beta_in: 1e-4,
            beta_out: 0.25,
            beta_incoh: 0.0,
            f_lo_hz: 20.0,
            f_hi_hz: 18000.0,
            headroom_db: 0.0,
            kaiser_beta: 16.0,
        };
        let res = design_kirkeby_inverse(&[plant], None, fs, &cfg).expect("kirkeby inverse");
        assert!(
            res.pre_gd_ripple_ms > 1.0,
            "Sabotage check: all-pass plant must have > 1 ms group-delay ripple before EQ (got {:.3} ms)",
            res.pre_gd_ripple_ms
        );
        assert!(
            res.post_gd_ripple_ms < 0.05,
            "Kirkeby–Nelson inverse must flatten all-pass excess group delay to < 0.05 ms (got {:.4} ms, pre was {:.3} ms)",
            res.post_gd_ripple_ms,
            res.pre_gd_ripple_ms
        );
        assert!(res.retained_energy_pct > 99.9);
    }

    #[test]
    fn tof_alignment_recovers_exact_bulk_shift_between_captures() {
        let mut h0 = vec![0.0_f64; 512];
        h0[20] = 1.0;
        h0[21] = -0.4;
        h0[22] = 0.15;
        let mut h1 = vec![0.0_f64; 512];
        h1[37] = 1.0;
        h1[38] = -0.4;
        h1[39] = 0.15;
        let (aligned, shifts) = align_to_reference(&[h0, h1], 1024);
        assert_eq!(shifts[0], 20);
        assert_eq!(shifts[1], 37);
        let max_diff = aligned[0]
            .iter()
            .zip(&aligned[1])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-12,
            "aligned captures differ by {max_diff:.2e}"
        );
    }

    #[test]
    fn radlovic_coherence_gating_distinguishes_stable_from_phase_jittered_bands() {
        // Build 8 captures that share identical spectrum below 3 kHz (stable) and have opposite
        // alternating phases above 6 kHz (spatially volatile pinna/seating jitter).
        let n_fft = 4096;
        let fs = 48000.0;
        let freqs = rfftfreq(n_fft, fs);
        let mut captures = Vec::new();
        for m in 0..8 {
            let mut spec = vec![Complex64::new(0.0, 0.0); freqs.len()];
            let sign_hf = if m % 2 == 0 { 1.0 } else { -1.0 };
            for (k, &f) in freqs.iter().enumerate() {
                if k > 0 && k + 1 < freqs.len() && f > 6000.0 {
                    spec[k] = Complex64::new(sign_hf, 0.0);
                } else {
                    spec[k] = Complex64::new(1.0, 0.0);
                }
            }
            captures.push(irfft(&spec, n_fft));
        }

        let specs: Vec<Vec<Complex64>> = captures.iter().map(|ir| rfft(ir, n_fft)).collect();
        let (_mean, coh) = spatial_coherence(&specs);
        let k_1k = freqs.iter().position(|&f| f >= 1000.0).unwrap();
        let k_10k = freqs.iter().position(|&f| f >= 10000.0).unwrap();
        assert!(
            coh[k_1k] > 0.99,
            "stable 1 kHz band must have coherence ~1.0 (got {:.4})",
            coh[k_1k]
        );
        assert!(
            coh[k_10k] < 0.01,
            "jittered 10 kHz band must have coherence ~0.0 (got {:.4})",
            coh[k_10k]
        );
    }

    #[test]
    fn complex_smoothing_preserves_flat_spectrum_and_attenuates_narrow_spike() {
        let n_fft = 4096;
        let freqs = rfftfreq(n_fft, 48000.0);
        let flat = vec![Complex64::new(1.0, 0.0); freqs.len()];
        let sm_flat = smooth_complex_octave(&flat, &freqs, 1.0 / 6.0);
        for (a, b) in flat.iter().zip(&sm_flat) {
            assert!((a - b).norm() < 1e-12);
        }

        let mut spiked = flat;
        let k_4k = freqs.iter().position(|&f| f >= 4000.0).unwrap();
        spiked[k_4k] = Complex64::new(0.0, 0.0); // single-bin null at 4 kHz
        let sm_spiked = smooth_complex_octave(&spiked, &freqs, 1.0 / 6.0);
        assert!(
            sm_spiked[k_4k].norm() > 0.90,
            "1/6-oct complex smoothing must fill a single-bin null (got {:.3})",
            sm_spiked[k_4k].norm()
        );
    }

    #[test]
    fn radlovic_coherence_gating_suppresses_incoherent_hf_boost_end_to_end() {
        // Construct 6 captures with an identical coherent -6 dB dip around 500 Hz, and a
        // spatially volatile HF band (>6 kHz) whose phase jitters across seatings while keeping
        // a small net mean magnitude. With beta_incoh > 0, Kirkeby–Nelson inversion must strongly
        // attenuate the HF equalizer gain relative to beta_incoh == 0 while leaving the coherent
        // 500 Hz equalization intact.
        let n_fft = 4096;
        let fs = 48000u32;
        let freqs = rfftfreq(n_fft, fs as f64);
        let mut captures = Vec::new();
        for m in 0..6 {
            let mut spec = vec![Complex64::new(0.0, 0.0); freqs.len()];
            // Five captures at +phase, one at -phase -> low coherence above 6 kHz
            let hf_angle = (m as f64) * std::f64::consts::TAU / 6.0 + 0.15;
            for (k, &f) in freqs.iter().enumerate() {
                if k == 0 || k + 1 == freqs.len() {
                    spec[k] = Complex64::new(1.0, 0.0);
                } else if (350.0..=700.0).contains(&f) {
                    spec[k] = Complex64::new(0.5, 0.0); // coherent -6 dB dip
                } else if (7000.0..=12000.0).contains(&f) {
                    // Low coherence (slight bias so mean != 0, gamma^2 ~ 0.03)
                    let bias = if m == 0 { 0.35 } else { 0.0 };
                    spec[k] = Complex64::new(hf_angle.cos() + bias, hf_angle.sin());
                } else {
                    spec[k] = Complex64::new(1.0, 0.0);
                }
            }
            // Shift causal peak to sample 256 so all captures are valid causal IRs.
            let raw = irfft(&spec, n_fft);
            let mut causal = vec![0.0_f64; 1024];
            for i in 0..1024 {
                causal[i] = raw[(i + n_fft - 256) % n_fft];
            }
            captures.push(causal);
        }

        let base_cfg = ComplexEqConfig {
            n_fft: 8192,
            n_taps: 4096,
            n_tail: 512,
            delay_ms: 20.0,
            smooth_oct: 0.0,
            beta_in: 0.005,
            beta_out: 0.25,
            beta_incoh: 0.0,
            f_lo_hz: 20.0,
            f_hi_hz: 18000.0,
            headroom_db: 0.0,
            kaiser_beta: 16.0,
        };
        let gated_cfg = ComplexEqConfig {
            beta_incoh: 4.0,
            ..base_cfg
        };
        let ungated = design_kirkeby_inverse(&captures, None, fs, &base_cfg).expect("ungated");
        let gated = design_kirkeby_inverse(&captures, None, fs, &gated_cfg).expect("gated");

        let g_u = rfft(&ungated.ir, 8192);
        let g_g = rfft(&gated.ir, 8192);
        let f8 = rfftfreq(8192, fs as f64);
        let k_500 = f8.iter().position(|&f| f >= 500.0).unwrap();
        let k_1k = f8.iter().position(|&f| f >= 1000.0).unwrap();
        let k_9k = f8.iter().position(|&f| f >= 9000.0).unwrap();

        // Compare HF gain relative to 1 kHz reference (preamp-invariant).
        let hf_rel_ungated_db = 20.0 * (g_u[k_9k].norm() / g_u[k_1k].norm()).log10();
        let hf_rel_gated_db = 20.0 * (g_g[k_9k].norm() / g_g[k_1k].norm()).log10();
        assert!(
            hf_rel_gated_db < hf_rel_ungated_db - 10.0,
            "Radlovic coherence gating must suppress incoherent 9 kHz boost by >10 dB (ungated {hf_rel_ungated_db:+.2} dB, gated {hf_rel_gated_db:+.2} dB)"
        );

        // Coherent 500 Hz dip boost relative to 1 kHz must be untouched by incoherence gating
        // (gamma^2 == 1 at 500 Hz and 1 kHz) and track the ~+5.8 dB regularized inverse.
        let lf_rel_ungated_db = 20.0 * (g_u[k_500].norm() / g_u[k_1k].norm()).log10();
        let lf_rel_gated_db = 20.0 * (g_g[k_500].norm() / g_g[k_1k].norm()).log10();
        assert!(
            (lf_rel_gated_db - lf_rel_ungated_db).abs() < 0.05 && lf_rel_gated_db > 5.4,
            "coherent 500 Hz boost must be preserved under gating (ungated {lf_rel_ungated_db:+.2} dB, gated {lf_rel_gated_db:+.2} dB)"
        );
    }

    #[test]
    fn rejects_invalid_inputs_cleanly() {
        let cfg = ComplexEqConfig::default();
        assert!(design_kirkeby_inverse(&[], None, 48000, &cfg).is_err());
        assert!(design_kirkeby_inverse(&[vec![]], None, 48000, &cfg).is_err());
        assert!(design_kirkeby_inverse(&[vec![0.0; 64]], None, 48000, &cfg).is_err());
        // Multi-capture with one silent capture must also be rejected cleanly.
        assert!(
            design_kirkeby_inverse(&[vec![0.0; 64], vec![1.0; 64]], None, 48000, &cfg).is_err()
        );
        assert!(design_kirkeby_inverse(&[vec![1.0, f64::NAN]], None, 48000, &cfg).is_err());
        assert!(design_kirkeby_inverse(&[vec![1.0; 64]], None, 0, &cfg).is_err());
        // Invalid config parameters must be rejected instead of poisoning output with NaN.
        let bad_flo = ComplexEqConfig {
            f_lo_hz: f64::NAN,
            ..cfg
        };
        assert!(design_kirkeby_inverse(&[vec![1.0; 64]], None, 48000, &bad_flo).is_err());
        let bad_headroom = ComplexEqConfig {
            headroom_db: f64::INFINITY,
            ..cfg
        };
        assert!(design_kirkeby_inverse(&[vec![1.0; 64]], None, 48000, &bad_headroom).is_err());
        let bad_smooth = ComplexEqConfig {
            smooth_oct: -0.25,
            ..cfg
        };
        assert!(design_kirkeby_inverse(&[vec![1.0; 64]], None, 48000, &bad_smooth).is_err());
    }

    #[test]
    fn zero_modeling_delay_produces_exact_zero_sample_shift() {
        let mut delta = vec![0.0_f64; 128];
        delta[0] = 1.0;
        let cfg = ComplexEqConfig {
            n_fft: 2048,
            n_taps: 1024,
            n_tail: 128,
            delay_ms: 0.0,
            smooth_oct: 0.0,
            beta_in: 1e-4,
            beta_out: 0.25,
            beta_incoh: 0.0,
            headroom_db: 0.0,
            ..ComplexEqConfig::default()
        };
        let res = design_kirkeby_inverse(&[delta], None, 48000, &cfg).expect("zero-delay inverse");
        assert_eq!(res.delay_samples, 0);
        assert_eq!(res.delay_ms, 0.0);
        // Peak impulse must sit at index 0.
        let peak_idx = res
            .ir
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(peak_idx, 0);
    }
}
