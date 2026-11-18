//! Multi-measurement ensemble statistics, Masiero & Fels (2011) peak-safe envelopes,
//! and whole-curve bootstrap confidence intervals.

use crate::fit::{median_db, quantile, smooth_octave};
use crate::pchip::Pchip;

/// How an ensemble of repeated headphone placements is collapsed before EQ fitting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnsembleMode {
    /// Pointwise arithmetic mean (baseline reference only).
    Average,
    /// Pointwise median across placements (survives one bad clamp/seal outlier).
    Median,
    /// Masiero & Fels (AES 2011) upper variance limit on 1/6-oct smoothed curves:
    /// `H_safe(f) = m(f) + alpha * sigma_r(f)` (positive `alpha` suppresses narrow
    /// placement-sensitive notch boosts so repositioning never creates harsh peaks).
    UpperVariance { alpha: f64 },
}

/// Summary across repeated placement measurements on a common log-frequency grid.
#[derive(Clone, Debug)]
pub struct EnsembleStats {
    pub f_hz: Vec<f64>,
    pub mean_db: Vec<f64>,
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
        let mut pts: Vec<(f64, f64)> = raw
            .iter()
            .copied()
            .filter(|(f, g)| f.is_finite() && *f > 0.0 && g.is_finite())
            .collect();
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
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
    let mut med_db = vec![0.0; k];
    let mut sigma_r_db = vec![0.0; k];
    let mut safe_db = vec![0.0; k];

    let mut col = vec![0.0; m];
    for bin in 0..k {
        let mut sum = crate::accum::RunningSum::new();
        for (j, curve) in resampled.iter().enumerate() {
            let v = curve[bin];
            col[j] = v;
            sum.add(v);
        }
        let mu = sum.total() / (m as f64);
        let med = median_db(&col);
        let devs: Vec<f64> = col.iter().map(|&v| (v - med).abs()).collect();
        let sig = 1.4826 * median_db(&devs);

        mean_db[bin] = mu;
        med_db[bin] = med;
        sigma_r_db[bin] = sig;
        safe_db[bin] = match mode {
            EnsembleMode::Average => mu,
            EnsembleMode::Median => med,
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
}
