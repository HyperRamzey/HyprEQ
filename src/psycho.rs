//! Psychoacoustic evaluation & complex spectral smoothing:
//! - Glasberg & Moore (1990) ERB scale and `roex(p)` auditory excitation patterns
//! - Olive, Welti & Khonsaripour (2017/2018) Harman Predicted Preference Rating (PPR)
//! - Hatziantoniou & Mourjopoulos (2000) complex fractional-octave smoothing

use crate::fit::smooth_octave;
use crate::pchip::Pchip;

/// Equivalent Rectangular Bandwidth in Hz at center frequency `f_hz` (Glasberg & Moore, 1990).
#[must_use]
pub fn erb_hz(f_hz: f64) -> f64 {
    24.7 * (4.37 * (f_hz.max(0.0) / 1000.0) + 1.0)
}

/// Frequency in Hz to ERB-rate number (Cam scale).
#[must_use]
pub fn hz_to_erbs(f_hz: f64) -> f64 {
    21.4 * (4.37 * (f_hz.max(0.0) / 1000.0) + 1.0).log10()
}

/// ERB-rate number back to Hz.
#[must_use]
pub fn erbs_to_hz(erbs: f64) -> f64 {
    (10f64.powf(erbs / 21.4) - 1.0) / 4.37 * 1000.0
}

/// Dimensionless ERB density weight normalized to `1.0` at 1 kHz (`ERB(1000) / ERB(f)`).
#[must_use]
pub fn erb_density(f_hz: f64) -> f64 {
    erb_hz(1000.0) / erb_hz(f_hz)
}

/// Auditory excitation level (dB) over each center frequency using symmetric Glasberg & Moore
/// `roex(p)` filters: `W(g) = (1 + p * |g|) * exp(-p * |g|)`, `p = 4 * fc / ERB(fc)`.
#[must_use]
pub fn roex_excitation_db(f_hz: &[f64], level_db: &[f64], centres_hz: &[f64]) -> Vec<f64> {
    let n = f_hz.len().min(level_db.len());
    if n < 2 || centres_hz.is_empty() {
        return vec![f64::NAN; centres_hz.len()];
    }
    // Linear trapezoid spacing in Hz.
    let mut df = vec![0.0; n];
    df[0] = (f_hz[1] - f_hz[0]).max(0.0);
    for i in 1..n - 1 {
        df[i] = 0.5 * (f_hz[i + 1] - f_hz[i - 1]).max(0.0);
    }
    df[n - 1] = (f_hz[n - 1] - f_hz[n - 2]).max(0.0);

    let power: Vec<f64> = level_db[..n]
        .iter()
        .map(|&db| {
            if db.is_finite() {
                10f64.powf(db / 10.0)
            } else {
                0.0
            }
        })
        .collect();

    centres_hz
        .iter()
        .map(|&fc| {
            if !fc.is_finite() || fc <= 0.0 {
                return f64::NAN;
            }
            let p = 4.0 * fc / erb_hz(fc);
            let mut num = crate::accum::RunningSum::new();
            let mut den = crate::accum::RunningSum::new();
            for i in 0..n {
                let g = (f_hz[i] - fc).abs() / fc;
                let w = (1.0 + p * g) * (-p * g).exp() * df[i];
                num.add(w * power[i]);
                den.add(w);
            }
            let d = den.total();
            if d > 0.0 {
                let ratio = (num.total() / d).max(1e-30);
                10.0 * ratio.log10()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect()
}

/// Headphone form factor for Harman Predicted Preference Rating (Olive, Welti & Khonsaripour).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PprMode {
    OverEar,
    InEar,
}

/// Output of [`harman_ppr`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PprScore {
    pub score: f64,
    pub sd_db: f64,
    pub abs_slope: f64,
    pub mean_err_db: f64,
}

/// Olive, Welti & Khonsaripour (2017/2018) Predicted Preference Rating on 1/48-octave spacing.
///
/// - `OverEar` (AES 9919, 2018): `114.49 - 12.62 * s - 15.52 * g` over 50 Hz–10 kHz.
/// - `InEar` (AES 143rd, 2017): `68.685 - 3.238 * s - 4.473 * g - 2.658 * m`
///   (`s, g` over 20 Hz–10 kHz, `m` = `|mean|` over 40 Hz–10 kHz).
#[must_use]
pub fn harman_ppr(f_hz: &[f64], error_db: &[f64], mode: PprMode) -> Option<PprScore> {
    let valid: Vec<(f64, f64)> = f_hz
        .iter()
        .zip(error_db)
        .filter(|(&f, &e)| f.is_finite() && f > 0.0 && e.is_finite())
        .map(|(&f, &e)| (f, e))
        .collect();
    if valid.len() < 2 {
        return None;
    }
    let (vf, ve): (Vec<f64>, Vec<f64>) = valid.into_iter().unzip();
    let interp = Pchip::new(&vf, &ve);

    let f_start = match mode {
        PprMode::OverEar => 50.0,
        PprMode::InEar => 20.0,
    };
    let grid = crate::global_fit::uniform_log_grid(f_start, 10_000.0, 48);
    if grid.len() < 2 {
        return None;
    }
    let err = interp.eval_grid(&grid);
    let n = err.len() as f64;
    let mean = err.iter().sum::<f64>() / n;
    let var = err.iter().map(|&e| (e - mean) * (e - mean)).sum::<f64>() / (n - 1.0);
    let sd = var.sqrt();

    let lx: Vec<f64> = grid.iter().map(|f| f.log10()).collect();
    let lx_mean = lx.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut lx_var = 0.0;
    for (&x, &y) in lx.iter().zip(&err) {
        let dx = x - lx_mean;
        cov += dx * (y - mean);
        lx_var += dx * dx;
    }
    let abs_slope = if lx_var > 0.0 {
        (cov / lx_var).abs()
    } else {
        0.0
    };

    let (score, mean_err_db) = match mode {
        PprMode::OverEar => (114.49 - 12.62 * sd - 15.52 * abs_slope, mean),
        PprMode::InEar => {
            let g40 = crate::global_fit::uniform_log_grid(40.0, 10_000.0, 48);
            let e40 = interp.eval_grid(&g40);
            let m40 = (e40.iter().sum::<f64>() / (e40.len() as f64)).abs();
            (68.685 - 3.238 * sd - 4.473 * abs_slope - 2.658 * m40, m40)
        }
    };

    Some(PprScore {
        score,
        sd_db: sd,
        abs_slope,
        mean_err_db,
    })
}

/// Hatziantoniou & Mourjopoulos (JAES 2000) complex fractional-octave smoothing over
/// real and imaginary spectral components on the log-frequency integral grid.
#[must_use]
pub fn smooth_complex_octave(
    f_hz: &[f64],
    re: &[f64],
    im: &[f64],
    fraction: f64,
) -> (Vec<f64>, Vec<f64>) {
    (
        smooth_octave(f_hz, re, fraction),
        smooth_octave(f_hz, im, fraction),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erb_and_ppr_match_published_baselines() {
        assert!((erb_hz(1000.0) - 132.639).abs() < 0.01);
        let grid = crate::global_fit::uniform_log_grid(20.0, 20000.0, 48);
        let flat = vec![0.0; grid.len()];
        let ppr = harman_ppr(&grid, &flat, PprMode::OverEar).unwrap();
        assert!((ppr.score - 114.49).abs() < 1e-6);

        let ex = roex_excitation_db(&grid, &flat, &[100.0, 1000.0, 8000.0]);
        for &v in &ex {
            assert!(v.abs() < 1e-9, "flat excitation must be 0 dB, got {v}");
        }
    }
}
