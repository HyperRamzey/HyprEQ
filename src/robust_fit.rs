//! Multi-measurement ensemble statistics, power-domain RMS averaging, peak-safe
//! placement envelopes, and whole-curve bootstrap confidence intervals.
//!
//! Prior art vs. HyprEQ design choices:
//!
//! * **Power-domain RMS averaging** ([`EnsembleMode::Rms`]): Schärer & Lindau
//!   (126th AES Convention, Paper 7721, May 2009) and Lindau & Brinkmann
//!   (JAES 60(1/2), pp. 54–62, Jan./Feb. 2012) average linear power
//!   `10 * log10((1/M) * sum 10^(L_j / 10))` across placements so narrow,
//!   position-dependent notches are naturally suppressed relative to dB-domain
//!   arithmetic averaging.
//! * **Upper-variance peak-safe envelope** ([`EnsembleMode::UpperVariance`]):
//!   Masiero & Fels (130th AES Convention, Paper 8388, May 2011) define an
//!   upper-variance curve `mean(f) + k * std(f)` (commonly `k = 2`) over
//!   1/6-octave smoothed measurements. HyprEQ substitutes an outlier-resistant
//!   location/scale pair `median(f) + alpha * (1.4826 * MAD(f))` — our own
//!   robust adaptation so a single bad clamp or seal outlier cannot inflate
//!   both the mean and standard deviation.

use crate::fit::{median_db, quantile, smooth_octave};
use crate::pchip::Pchip;

/// How an ensemble of repeated headphone placements is collapsed before EQ fitting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnsembleMode {
    /// Pointwise dB-domain arithmetic mean (baseline reference only).
    Average,
    /// Pointwise median across placements (survives one bad clamp/seal outlier).
    Median,
    /// Linear power-domain RMS average (Schärer & Lindau, AES 2009; Lindau &
    /// Brinkmann, JAES 2012): `10 * log10(mean_j 10^(L_j(f) / 10))`.
    /// Strictly `>= Average` by Jensen's inequality, naturally weighting down
    /// shifting narrowband notches across placements.
    Rms,
    /// Outlier-resistant upper-variance envelope inspired by Masiero & Fels
    /// (AES 130th, Paper 8388, May 2011): `median(f) + alpha * (1.4826 * MAD(f))`.
    /// (Masiero & Fels use `mean + k * std`; replacing mean/std with median/MAD
    /// is HyprEQ's own design choice.)
    UpperVariance { alpha: f64 },
}

impl EnsembleMode {
    /// CLI vocabulary for `--ensemble-mode`.
    pub const CHOICES: [&'static str; 4] = ["median", "rms", "upper-variance", "average"];

    /// Parse a CLI mode word together with `alpha` (used only for `upper-variance`).
    #[must_use]
    pub fn parse(s: &str, alpha: f64) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "median" => Some(Self::Median),
            "rms" => Some(Self::Rms),
            "upper-variance" => Some(Self::UpperVariance { alpha }),
            "average" => Some(Self::Average),
            _ => None,
        }
    }

    /// Canonical label for reports.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Average => "average",
            Self::Median => "median",
            Self::Rms => "rms",
            Self::UpperVariance { .. } => "upper-variance",
        }
    }
}

/// Summary across repeated placement measurements on a common log-frequency grid.
#[derive(Clone, Debug)]
pub struct EnsembleStats {
    pub f_hz: Vec<f64>,
    pub mean_db: Vec<f64>,
    /// Linear power-domain RMS average across placements (`10 * log10(mean 10^(L/10))`).
    pub rms_db: Vec<f64>,
    pub median_db: Vec<f64>,
    /// `1.4826 * MAD(f)`, consistent with Gaussian standard deviation.
    pub sigma_r_db: Vec<f64>,
    /// Conservative target-inversion curve under the chosen [`EnsembleMode`].
    pub safe_db: Vec<f64>,
    /// Lower 2.5% quantile from whole-curve bootstrap resampling.
    pub ci95_lo_db: Vec<f64>,
    /// Upper 97.5% quantile from whole-curve bootstrap resampling.
    pub ci95_hi_db: Vec<f64>,
    pub count: usize,
}

/// Deterministic SplitMix64 PRNG so bootstrap intervals are reproducible without external crates.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() % (bound as u64)) as usize
    }
}

/// Resample each raw measurement curve onto `f_grid` (via log-frequency PCHIP), optionally
/// applying fractional-octave smoothing (e.g. `Some(1.0 / 6.0)` for Masiero & Fels), and
/// compute robust placement statistics + whole-curve bootstrap `CI_95`.
///
/// Resampling **whole curves** rather than independent frequency bins preserves spectral
/// correlations across adjacent frequencies.
#[must_use]
pub fn analyze_ensemble(
    curves: &[Vec<(f64, f64)>],
    f_grid: &[f64],
    smooth_oct: Option<f64>,
    mode: EnsembleMode,
    n_boot: usize,
) -> Option<EnsembleStats> {
    if curves.is_empty() || f_grid.is_empty() {
        return None;
    }

    let mut resampled: Vec<Vec<f64>> = Vec::with_capacity(curves.len());
    for raw in curves {
        let pts = crate::model::sort_and_dedup_pairs(raw);
        if pts.len() < 2 {
            continue;
        }
        let (fs, gs): (Vec<f64>, Vec<f64>) = pts.into_iter().unzip();
        let vals = Pchip::new(&fs, &gs).eval_grid(f_grid);
        let smoothed = match smooth_oct {
            Some(oct) if oct.is_finite() && oct > 0.0 => smooth_octave(f_grid, &vals, oct),
            _ => vals,
        };
        if smoothed.iter().all(|v| v.is_finite()) {
            resampled.push(smoothed);
        }
    }

    let m = resampled.len();
    if m == 0 {
        return None;
    }
    let k = f_grid.len();

    let mut mean_db = vec![0.0; k];
    let mut rms_db = vec![0.0; k];
    let mut med_db = vec![0.0; k];
    let mut sigma_r_db = vec![0.0; k];
    let mut safe_db = vec![0.0; k];

    let mut col = vec![0.0; m];
    for bin in 0..k {
        let mut sum = crate::accum::RunningSum::new();
        let mut max_v = f64::NEG_INFINITY;
        for (j, curve) in resampled.iter().enumerate() {
            let v = curve[bin];
            col[j] = v;
            sum.add(v);
            if v > max_v {
                max_v = v;
            }
        }
        let mu = sum.total() / (m as f64);
        // Log-sum-exp in linear power prevents overflow/underflow at arbitrary SPL offsets.
        let mut psum = crate::accum::RunningSum::new();
        for &v in &col {
            psum.add(10.0_f64.powf((v - max_v) / 10.0));
        }
        let rms = max_v + 10.0 * (psum.total() / (m as f64)).log10();
        let med = median_db(&col);
        let devs: Vec<f64> = col.iter().map(|&v| (v - med).abs()).collect();
        let sig = 1.4826 * median_db(&devs);

        mean_db[bin] = mu;
        rms_db[bin] = rms;
        med_db[bin] = med;
        sigma_r_db[bin] = sig;
        safe_db[bin] = match mode {
            EnsembleMode::Average => mu,
            EnsembleMode::Median => med,
            EnsembleMode::Rms => rms,
            EnsembleMode::UpperVariance { alpha } => {
                let a = if alpha.is_finite() { alpha } else { 0.0 };
                med + a * sig
            }
        };
    }

    let mut ci95_lo_db = med_db.clone();
    let mut ci95_hi_db = med_db.clone();
    if n_boot > 0 && m >= 2 {
        let mut rng = SplitMix64(0x4879_7072_4551_0001);
        let mut boot: Vec<Vec<f64>> = Vec::with_capacity(n_boot);
        let mut sample = vec![0.0; m];
        for _ in 0..n_boot {
            let picked: Vec<usize> = (0..m).map(|_| rng.next_usize(m)).collect();
            let mut meds = vec![0.0; k];
            for bin in 0..k {
                for (j, &idx) in picked.iter().enumerate() {
                    sample[j] = resampled[idx][bin];
                }
                meds[bin] = median_db(&sample);
            }
            boot.push(meds);
        }
        for bin in 0..k {
            let col: Vec<f64> = boot.iter().map(|curve| curve[bin]).collect();
            ci95_lo_db[bin] = quantile(&col, 0.025);
            ci95_hi_db[bin] = quantile(&col, 0.975);
        }
    }

    Some(EnsembleStats {
        f_hz: f_grid.to_vec(),
        mean_db,
        rms_db,
        median_db: med_db,
        sigma_r_db,
        safe_db,
        ci95_lo_db,
        ci95_hi_db,
        count: m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mad_and_upper_envelope_protect_against_unstable_notches() {
        let grid = crate::fft::geomspace(20.0, 20000.0, 121);
        // 5 placements: stable at 1 kHz, wild notch [-10, -2, -5, -8, -3] near 4 kHz
        let depths = [-10.0, -2.0, -5.0, -8.0, -3.0];
        let curves: Vec<Vec<(f64, f64)>> = depths
            .iter()
            .map(|&d| {
                grid.iter()
                    .map(|&f| {
                        let z = (f / 4000.0).log2() / 0.1;
                        (f, d * (-0.5 * z * z).exp())
                    })
                    .collect()
            })
            .collect();

        let stats = analyze_ensemble(
            &curves,
            &grid,
            None,
            EnsembleMode::UpperVariance { alpha: 1.5 },
            200,
        )
        .expect("valid ensemble");

        let i_1k = grid
            .iter()
            .position(|&f| (f - 1000.0).abs() < 40.0)
            .unwrap();
        let i_4k = grid
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - 4000.0).abs().total_cmp(&(b.1 - 4000.0).abs()))
            .unwrap()
            .0;

        assert!(stats.sigma_r_db[i_1k] < 0.05);
        assert!(
            stats.sigma_r_db[i_4k] > 2.5,
            "expected large MAD at 4 kHz, got {}",
            stats.sigma_r_db[i_4k]
        );
        assert!(
            stats.safe_db[i_4k] > stats.median_db[i_4k] + 2.0,
            "upper variance envelope must lift above median at unstable notch"
        );
        assert!(stats.ci95_lo_db[i_4k] <= stats.median_db[i_4k]);
        assert!(stats.ci95_hi_db[i_4k] >= stats.median_db[i_4k]);
    }

    #[test]
    fn rms_mode_suppresses_shifting_notch_and_satisfies_jensen_inequality() {
        let grid = vec![100.0, 500.0, 1000.0, 4000.0, 8000.0];
        // Three clean placements at 0 dB and one placement with a -20 dB shifting notch at 4 kHz.
        let c1: Vec<(f64, f64)> = grid.iter().map(|&f| (f, 0.0)).collect();
        let c2 = c1.clone();
        let c3 = c1.clone();
        let c4: Vec<(f64, f64)> = grid
            .iter()
            .map(|&f| (f, if (f - 4000.0).abs() < 1.0 { -20.0 } else { 0.0 }))
            .collect();
        let stats = analyze_ensemble(&[c1, c2, c3, c4], &grid, None, EnsembleMode::Rms, 0)
            .expect("rms stats");
        // Analytic RMS at 4 kHz: 10*log10((3 + 0.01) / 4) = -1.234942 dB vs mean = -5.0 dB.
        let expected_rms = 10.0 * ((3.0 + 0.01_f64) / 4.0).log10();
        assert!((stats.rms_db[3] - expected_rms).abs() < 1e-12);
        assert!((stats.safe_db[3] - expected_rms).abs() < 1e-12);
        assert!((stats.mean_db[3] - (-5.0)).abs() < 1e-12);
        // Jensen's inequality: power-domain RMS >= dB arithmetic mean everywhere.
        for (&r, &m) in stats.rms_db.iter().zip(&stats.mean_db) {
            assert!(r >= m - 1e-12, "RMS {r} must be >= mean {m}");
        }
    }

    #[test]
    fn rms_mode_is_overflow_safe_at_extreme_spl_offsets() {
        let grid = vec![100.0, 1000.0, 10000.0];
        let hi: Vec<(f64, f64)> = grid.iter().map(|&f| (f, 5000.0)).collect();
        let lo: Vec<(f64, f64)> = grid.iter().map(|&f| (f, -5000.0)).collect();
        let s_hi = analyze_ensemble(&[hi.clone(), hi], &grid, None, EnsembleMode::Rms, 0)
            .expect("hi stats");
        let s_lo = analyze_ensemble(&[lo.clone(), lo], &grid, None, EnsembleMode::Rms, 0)
            .expect("lo stats");
        for &v in &s_hi.rms_db {
            assert!(v.is_finite() && (v - 5000.0).abs() < 1e-9);
        }
        for &v in &s_lo.rms_db {
            assert!(v.is_finite() && (v - (-5000.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn ensemble_mode_parse_round_trips_all_choices() {
        for word in EnsembleMode::CHOICES {
            let parsed = EnsembleMode::parse(word, 0.75).expect("valid choice");
            assert_eq!(parsed.as_str(), word);
        }
        assert!(EnsembleMode::parse("invalid", 0.0).is_none());
    }
}
