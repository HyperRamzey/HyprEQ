//! Psychoacoustic evaluation:
//! - Glasberg & Moore (1990) ERB scale and `roex(p)` auditory excitation patterns
//! - Olive, Welti & Khonsaripour (2017/2018) Harman Predicted Preference Rating (PPR)

use crate::pchip::Pchip;

/// Equivalent Rectangular Bandwidth in Hz at center frequency `f_hz` (Glasberg & Moore, 1990).
#[must_use]
pub fn erb_hz(f_hz: f64) -> f64 {
    24.7 * (4.37 * (f_hz.max(0.0) / 1000.0) + 1.0)
}

/// Dimensionless ERB density weight on log-frequency grids, normalized to `1.0` at 1 kHz:
/// `(f / ERB(f)) / (1000 / ERB(1000))`.
///
/// On a log-frequency grid ($\Delta \ln f = \mathrm{const}$), the differential frequency step
/// is $df = f\, d(\ln f)$. Weighting an error integral by ERB density
/// $d(\mathrm{Cam}) = df / \mathrm{ERB}(f)$ therefore requires weighting by
/// $f / \mathrm{ERB}(f)$ to avoid double-counting the logarithmic bin density.
#[must_use]
pub fn erb_log_density(f_hz: f64) -> f64 {
    if f_hz <= 0.0 || !f_hz.is_finite() {
        return 0.0;
    }
    let norm = 1000.0 / erb_hz(1000.0);
    (f_hz / erb_hz(f_hz)) / norm
}

/// Auditory excitation level (dB) over each center frequency using symmetric Glasberg & Moore
/// `roex(p)` filters: `W(g) = (1 + p * |g|) * exp(-p * |g|)`, `p = 4 * fc / ERB(fc)`.
#[must_use]
pub fn roex_excitation_db(f_hz: &[f64], level_db: &[f64], centres_hz: &[f64]) -> Vec<f64> {
    let n = f_hz.len().min(level_db.len());
    if n < 2 || centres_hz.is_empty() {
        return vec![f64::NAN; centres_hz.len()];
    }
    // Composite-trapezoid spacing in Hz: interior bins carry the mean of their two
    // neighbour intervals, the two endpoint bins carry HALF the interval they border —
    // that is what `np.trapz` weights are. Handing the endpoints the full interval (as
    // this did) doubles the extreme-frequency bins against every other bin, skewing the
    // excitation of centres near either edge of the data.
    let mut df = vec![0.0; n];
    df[0] = 0.5 * (f_hz[1] - f_hz[0]).max(0.0);
    for i in 1..n - 1 {
        df[i] = 0.5 * (f_hz[i + 1] - f_hz[i - 1]).max(0.0);
    }
    df[n - 1] = 0.5 * (f_hz[n - 1] - f_hz[n - 2]).max(0.0);

    // A non-finite level bin is missing data, not silence. Reading it as 0 power while
    // still counting its weight in the denominator diluted every centre it overlaps —
    // pulling excitation toward an invented noise floor. Such a bin leaves both the
    // numerator **and** the denominator; a centre whose usable bins are all non-finite
    // lands on the empty-denominator path below.
    let usable: Vec<bool> = (0..n)
        .map(|i| level_db[i].is_finite() && f_hz[i].is_finite())
        .collect();
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
                if !usable[i] {
                    continue;
                }
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

enum PprSegment {
    Curve {
        f_min: f64,
        f_max: f64,
        pchip: Pchip,
    },
    Single {
        f: f64,
        val: f64,
    },
}

impl PprSegment {
    fn eval(&self, g: f64) -> Option<f64> {
        match self {
            Self::Curve {
                f_min,
                f_max,
                pchip,
            } => {
                if g >= *f_min && g <= *f_max {
                    Some(pchip.eval(g))
                } else {
                    None
                }
            }
            Self::Single { f, val } => {
                if (g - *f).abs() < 1e-9 {
                    Some(*val)
                } else {
                    None
                }
            }
        }
    }
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
///
/// Returns `None` if the input curve is empty, has fewer than two valid knots,
/// or produces degenerate statistics.
#[must_use]
pub fn harman_ppr(f_hz: &[f64], error_db: &[f64], mode: PprMode) -> Option<PprScore> {
    let n_in = f_hz.len().min(error_db.len());
    if n_in < 2 {
        return None;
    }
    let mut raw_pairs: Vec<(f64, f64)> = f_hz[..n_in]
        .iter()
        .copied()
        .zip(error_db[..n_in].iter().copied())
        .filter(|(f, _)| f.is_finite() && *f > 0.0)
        .collect();
    raw_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(raw_pairs.len());
    for p in raw_pairs {
        if pairs.last().is_some_and(|prev| prev.0 == p.0) {
            let n = pairs.len();
            pairs[n - 1] = p;
        } else {
            pairs.push(p);
        }
    }

    // Partition into contiguous segments where error is finite, avoiding Hermite tangent
    // bleed across unmeasured NaN gaps while supporting sparse curves without arbitrary octave caps.
    let mut segments: Vec<PprSegment> = Vec::new();
    let mut cur_f: Vec<f64> = Vec::new();
    let mut cur_e: Vec<f64> = Vec::new();

    let flush_segment =
        |cur_f: &mut Vec<f64>, cur_e: &mut Vec<f64>, segments: &mut Vec<PprSegment>| {
            if cur_f.is_empty() {
                return;
            }
            if cur_f.len() >= 2 {
                let f_min = cur_f[0];
                let f_max = cur_f[cur_f.len() - 1];
                let pchip = Pchip::new(cur_f, cur_e);
                segments.push(PprSegment::Curve {
                    f_min,
                    f_max,
                    pchip,
                });
            } else {
                segments.push(PprSegment::Single {
                    f: cur_f[0],
                    val: cur_e[0],
                });
            }
            cur_f.clear();
            cur_e.clear();
        };

    for (f, e) in pairs {
        if e.is_finite() {
            cur_f.push(f);
            cur_e.push(e);
        } else {
            flush_segment(&mut cur_f, &mut cur_e, &mut segments);
        }
    }
    flush_segment(&mut cur_f, &mut cur_e, &mut segments);

    if segments.is_empty() {
        return None;
    }

    let eval_at = |g: f64| -> Option<f64> {
        for seg in &segments {
            if let Some(v) = seg.eval(g) {
                return Some(v);
            }
        }
        None
    };

    let f_start = match mode {
        PprMode::OverEar => 50.0,
        PprMode::InEar => 20.0,
    };
    let grid = crate::global_fit::uniform_log_grid(f_start, 10_000.0, 48);
    if grid.len() < 2 {
        return None;
    }

    let mut eval_grid = Vec::with_capacity(grid.len());
    let mut err = Vec::with_capacity(grid.len());
    for &g in &grid {
        if let Some(v) = eval_at(g) {
            eval_grid.push(g);
            err.push(v);
        }
    }
    if err.len() < 2 {
        return None;
    }

    let n = err.len() as f64;
    let mean = err.iter().sum::<f64>() / n;
    let var = err.iter().map(|&e| (e - mean) * (e - mean)).sum::<f64>() / (n - 1.0);
    let sd = var.sqrt();

    // Natural log, as published. See the `AS` term in the docs above.
    let lx: Vec<f64> = eval_grid.iter().map(|f| f.ln()).collect();
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
            let mut e40 = Vec::with_capacity(g40.len());
            for &g in &g40 {
                if let Some(v) = eval_at(g) {
                    e40.push(v);
                }
            }
            if e40.is_empty() {
                return None;
            }
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
        assert!((erb_log_density(1000.0) - 1.0).abs() < 1e-12);
        assert!((erb_log_density(20.0) - 0.0988).abs() < 0.001);
        assert!((erb_log_density(20000.0) - 1.2149).abs() < 0.001);
        assert_eq!(erb_log_density(0.0), 0.0);
        assert_eq!(erb_log_density(-50.0), 0.0);
        assert_eq!(erb_log_density(f64::NAN), 0.0);
        let grid = crate::global_fit::uniform_log_grid(20.0, 20000.0, 48);
        let flat = vec![0.0; grid.len()];
        let ppr = harman_ppr(&grid, &flat, PprMode::OverEar).unwrap();
        assert!((ppr.score - 114.49).abs() < 1e-6);

        let ex = roex_excitation_db(&grid, &flat, &[100.0, 1000.0, 8000.0]);
        for &v in &ex {
            assert!(v.abs() < 1e-9, "flat excitation must be 0 dB, got {v}");
        }
    }

    /// A bin with no level is missing data, not silence. It must leave **both** sums:
    /// substituting 0 power while keeping the weight dragged every centre it overlaps —
    /// measured: a 60 dB / missing / 60 dB triple read −920 dB through the dilution.
    #[test]
    fn a_missing_level_bin_leaves_the_average_instead_of_diluting_it() {
        let f = [100.0, 1000.0, 2000.0];
        let lv = [60.0, f64::NAN, 60.0];
        let ex = roex_excitation_db(&f, &lv, &[1000.0]);
        assert!(
            (ex[0] - 60.0).abs() < 1e-9,
            "a hole at the centre must not dilute two finite 60 dB bins: {}",
            ex[0]
        );
        // And the hole must not be reported as a number at all when nothing else is there.
        let all_bad = roex_excitation_db(&f, &[f64::NAN; 3], &[1000.0]);
        assert!(
            all_bad[0].is_infinite() && all_bad[0] < 0.0,
            "{:?}",
            all_bad[0]
        );
    }

    /// Endpoint bins border one interval and carry HALF of it under the composite
    /// trapezoid rule; giving them the full interval doubled them against every other
    /// bin. Both numbers below are computed, not asserted: an independent Python pass
    /// over the same sums reproduces 54.64928534604776 exactly and puts the
    /// full-interval variant 1.50 dB higher — the endpoint bin sitting right at the
    /// centre is the one the skew inflates, so a centre near either edge of the data
    /// reads hot.
    #[test]
    fn endpoint_bins_take_half_the_interval_they_border() {
        let f = [20.0, 30.0, 40.0];
        let lv = [60.0, 0.0, 0.0];
        let ex = roex_excitation_db(&f, &lv, &[25.0]);
        assert!(
            (ex[0] - 54.64928534604776).abs() < 1e-9,
            "expected the trapezoid-weighted 54.64928534604776 dB, got {ex:.17} \
             (full-interval endpoints give 56.14707173051804)",
            ex = ex[0]
        );
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

    #[test]
    fn harman_ppr_evaluates_valid_sparse_curves_without_dropping_bins() {
        // 4-point sparse curve spanning 50 Hz to 10 kHz with 0 dB error
        let f = [50.0, 100.0, 1000.0, 10000.0];
        let e = [0.0, 0.0, 0.0, 0.0];
        let s = harman_ppr(&f, &e, PprMode::OverEar).expect("sparse curve must succeed");
        assert!(
            s.sd_db < 1e-9,
            "sd on flat curve must be 0.0, got {}",
            s.sd_db
        );
        assert!(
            s.abs_slope < 1e-9,
            "abs_slope must be 0.0, got {}",
            s.abs_slope
        );
        assert!(
            (s.score - 114.49).abs() < 1e-6,
            "score must be 114.49, got {}",
            s.score
        );
    }

    #[test]
    fn harman_ppr_excludes_unmeasured_gaps_and_does_not_bridge_them() {
        let grid = crate::global_fit::uniform_log_grid(100.0, 8000.0, 48);
        // Error is 0 dB below 400 Hz and +6 dB above 800 Hz, with a 1-octave gap at 400..800 Hz (NaN)
        let err: Vec<f64> = grid
            .iter()
            .map(|&f| {
                if (400.0..=800.0).contains(&f) {
                    f64::NAN
                } else if f < 400.0 {
                    0.0
                } else {
                    6.0
                }
            })
            .collect();
        let s = harman_ppr(&grid, &err, PprMode::OverEar).expect("ppr score");
        // Non-bridged: bins interior to the 400..800 Hz gap are completely excluded.
        // Valid edge bins (395 Hz and 802 Hz) are retained.
        // With gap exclusion: SD = 2.9150 dB, abs_slope = 1.9319, score = 47.720.
        // If unmeasured gap bridging was enabled (sabotaged), PCHIP would interpolate across
        // the 1-octave gap, adding spurious bridged bins which shift SD down to 2.8068 dB
        // and score up to 48.788 dB (>1.0 dB drift), failing this test.
        assert!(
            (s.sd_db - 2.9150).abs() < 0.001,
            "sd_db expected ~2.9150, got {}",
            s.sd_db
        );
        assert!(
            (s.abs_slope - 1.9319).abs() < 0.001,
            "abs_slope expected ~1.9319, got {}",
            s.abs_slope
        );
        assert!(
            (s.score - 47.720).abs() < 0.002,
            "score expected ~47.720, got {}",
            s.score
        );
    }
}
