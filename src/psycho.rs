//! Psychoacoustic evaluation:
//! - Glasberg & Moore (1990) ERB scale and `roex(p)` auditory excitation patterns
//! - Olive, Welti & Khonsaripour (2017/2018) Harman Predicted Preference Rating (PPR)

use crate::pchip::Pchip;

/// Equivalent Rectangular Bandwidth in Hz at center frequency `f_hz` (Glasberg & Moore, 1990).
#[must_use]
pub fn erb_hz(f_hz: f64) -> f64 {
    24.7 * (4.37 * (f_hz.max(0.0) / 1000.0) + 1.0)
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
    /// `|slope|` of the deviation regressed on `ln(f)`, in dB per neper of log-frequency.
    pub abs_slope: f64,
    /// `InEar`: `ME`, the mean of `|error|` over 40 Hz–10 kHz. `OverEar`: the signed mean, which
    /// the published over-ear model does not use — reported for context only.
    pub mean_err_db: f64,
}

/// Olive, Welti & Khonsaripour (2017/2018) Predicted Preference Rating on 1/48-octave spacing.
///
/// Form as published — US 2019/0087739 A1 (*Linear model to predict listener preference ratings of
/// headphones*, Olive, Welti & Khonsaripour, Harman International, priority 2017-09-15), claims
/// 11–13 and 20, which give all three predictors explicitly:
///
/// - `AS = SUM[(ln x − ln x̄)(y − ȳ)] / SUM[(ln x − ln x̄)²]` — a regression of the deviation on the
///   **natural** logarithm of frequency. Regressing on log₁₀ instead scales the slope by ln 10 ≈
///   2.302585, so every `AS`-weighted penalty is over-applied by that factor.
/// - `SD = sqrt(SUM (y − ȳ)² / (n − 1))` — the sample standard deviation (the `n − 1` here is what
///   this function's `n - 1.0` divisor is).
/// - `ME_40Hz–10kHz = SUM |y_i| / n` — the mean of the **absolute** deviations. `|mean y|` is a
///   different statistic entirely: boosts and cuts of similar size cancel, so a curve thrashing
///   ±6 dB scores like a flat one on the term that exists to catch exactly that.
///
/// Ranges as published: `OverEar` scores `s, g` over 50 Hz–10 kHz as
/// `114.49 − 12.62·s − 15.52·AS`; `InEar` scores `s, g` over 20 Hz–10 kHz and `m` over 40 Hz–10 kHz
/// as `68.685 − 3.238·s − 4.473·AS − 2.658·ME`.
///
/// `error_db` is the headphone response error curve — the equalized magnitude response minus its
/// design target — not a rig-tracking residual. See [`crate::verify::psycho_summary`].
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

    // Natural log, as published. See the `AS` term in the docs above.
    let lx: Vec<f64> = grid.iter().map(|f| f.ln()).collect();
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
            // ME = SUM|y|/n: the absolute values are taken per bin, before the mean.
            let m40 = e40.iter().map(|e| e.abs()).sum::<f64>() / (e40.len() as f64);
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

    /// A deviation of exactly 6 dB per decade has a slope of 6 / ln 10 = 2.602697 dB per neper when
    /// regressed on `ln f`, as published. Regressed on `log10 f` the same curve reports 6.0 — a 2.3×
    /// over-penalty on a term weighted 15.52 (over-ear) or 4.473 (in-ear).
    #[test]
    fn the_absolute_slope_is_regressed_on_the_natural_log_of_frequency() {
        let grid = crate::global_fit::uniform_log_grid(50.0, 10_000.0, 48);
        let err: Vec<f64> = grid.iter().map(|f| 6.0 * (f / 1000.0).log10()).collect();
        let s = harman_ppr(&grid, &err, PprMode::OverEar).expect("score");
        let expected = 6.0 / 10f64.ln();
        assert!(
            (s.abs_slope - expected).abs() < 1e-6,
            "AS must be {expected:.6} dB per neper, got {:.6}",
            s.abs_slope
        );
        assert!(
            (s.abs_slope * 10f64.ln() - 6.0).abs() < 1e-6,
            "ln(10) is exactly the factor a log10 regression adds"
        );
    }

    /// ME is the mean of the absolute deviations, not the absolute value of the mean: an
    /// alternating ±3 dB curve has a signed mean of zero and a mean absolute error of 3 dB, and the
    /// term exists to charge the thrash that the signed mean cannot see.
    #[test]
    fn the_in_ear_mean_error_averages_absolute_deviations() {
        let grid = crate::global_fit::uniform_log_grid(20.0, 10_000.0, 48);
        let err: Vec<f64> = grid
            .iter()
            .enumerate()
            .map(|(i, _)| if i % 2 == 0 { 3.0 } else { -3.0 })
            .collect();
        let signed_mean = err.iter().sum::<f64>() / err.len() as f64;
        assert!(
            signed_mean.abs() < 0.02,
            "the fixture must be one the two statistics disagree on, got {signed_mean:.4}"
        );
        let s = harman_ppr(&grid, &err, PprMode::InEar).expect("score");
        assert!(
            (s.mean_err_db - 3.0).abs() < 0.25,
            "ME must be mean|y| (3 dB), got {:.3}",
            s.mean_err_db
        );
        assert!(
            s.score < 68.685 - 2.6,
            "a 3 dB thrash must pay the 2.658 x ME term, score {:.3}",
            s.score
        );
    }
}
