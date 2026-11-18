//! Driver physical limits: Olive & Lelièvre (2026) level-dependent nonlinear distortion
//! envelopes and Thiele–Small sub-resonance diaphragm excursion scaling.

use crate::pchip::Pchip;

/// Measured distortion at a single output SPL sweep.
#[derive(Clone, Debug)]
pub struct DistortionSweep {
    pub spl_db: f64,
    /// `(freq_hz, distortion_pct)` (e.g. THD% or non-coherent %).
    pub points: Vec<(f64, f64)>,
}

/// Collection of multi-SPL distortion sweeps for one headphone model.
#[derive(Clone, Debug, Default)]
pub struct DistortionProfile {
    pub sweeps: Vec<DistortionSweep>,
}

impl DistortionProfile {
    /// Estimate `L_max(f) = max { L : D(f, L) <= threshold_pct }` per frequency in `f_grid`.
    #[must_use]
    pub fn max_clean_output_db(&self, f_grid: &[f64], threshold_pct: f64) -> Vec<f64> {
        if self.sweeps.is_empty() || f_grid.is_empty() {
            return vec![f64::INFINITY; f_grid.len()];
        }
        let mut sorted = self.sweeps.clone();
        sorted.sort_by(|a, b| a.spl_db.total_cmp(&b.spl_db));

        let eval_sweeps: Vec<(f64, Vec<f64>)> = sorted
            .iter()
            .filter_map(|s| {
                let mut pts: Vec<(f64, f64)> = s
                    .points
                    .iter()
                    .copied()
                    .filter(|(f, d)| f.is_finite() && *f > 0.0 && d.is_finite())
                    .collect();
                pts.sort_by(|a, b| a.0.total_cmp(&b.0));
                if pts.len() < 2 {
                    return None;
                }
                let (fs, ds): (Vec<f64>, Vec<f64>) = pts.into_iter().unzip();
                Some((s.spl_db, Pchip::new(&fs, &ds).eval_grid(f_grid)))
            })
            .collect();

        if eval_sweeps.is_empty() {
            return vec![f64::INFINITY; f_grid.len()];
        }

        (0..f_grid.len())
            .map(|bin| {
                let mut clean_max = f64::NEG_INFINITY;
                for (spl, vals) in &eval_sweeps {
                    if vals[bin] <= threshold_pct {
                        clean_max = *spl;
                    } else {
                        break;
                    }
                }
                clean_max
            })
            .collect()
    }

    /// Translate `L_max(f)` into an upper EQ boost bound `q_max(f)` given listening SPL.
    /// Per Olive & Lelièvre (2026), audible music distortion generally requires high SPLs (~104–112 dBA),
    /// so this stays slack at normal listening volumes and tightens dynamically at high SPLs.
    #[must_use]
    pub fn q_max_bounds(
        &self,
        f_grid: &[f64],
        h_cal_db: &[f64],
        listen_spl_db: f64,
        threshold_pct: f64,
        hard_cap_db: f64,
    ) -> Vec<f64> {
        let l_max = self.max_clean_output_db(f_grid, threshold_pct);
        l_max
            .iter()
            .zip(h_cal_db)
            .map(|(&lm, &hc)| {
                if !lm.is_finite() {
                    hard_cap_db
                } else {
                    (lm - listen_spl_db - hc).min(hard_cap_db).max(0.0)
                }
            })
            .collect()
    }
}

/// Mechanical excursion model below fundamental driver resonance `fs_hz`.
#[derive(Clone, Copy, Debug)]
pub struct ExcursionModel {
    pub fs_hz: f64,
    /// Reference SPL (at `fs_hz`) where the driver reaches 100% linear `X_max`.
    pub ref_xmax_spl_db: f64,
}

impl ExcursionModel {
    /// Estimated excursion margin in dB (`> 0` means within `X_max`, `< 0` means over-driven).
    /// Below sealed/planar resonance, diaphragm displacement rises roughly as `(fs / f)^2`
    /// (~+12 dB per octave below `fs_hz`).
    #[must_use]
    pub fn margin_db(&self, f_hz: f64, eq_boost_db: f64, band_spl_db: f64) -> f64 {
        if f_hz <= 0.0 || self.fs_hz <= 0.0 {
            return 0.0;
        }
        let sub_rise_db = if f_hz < self.fs_hz {
            40.0 * (self.fs_hz / f_hz).log10()
        } else {
            0.0
        };
        self.ref_xmax_spl_db - (band_spl_db + eq_boost_db + sub_rise_db)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distortion_envelope_caps_boost_only_at_high_spl() {
        let prof = DistortionProfile {
            sweeps: vec![
                DistortionSweep {
                    spl_db: 94.0,
                    points: vec![(20.0, 1.5), (1000.0, 0.2), (20000.0, 0.3)],
                },
                DistortionSweep {
                    spl_db: 104.0,
                    points: vec![(20.0, 5.0), (1000.0, 0.4), (20000.0, 0.5)],
                },
            ],
        };
        let f = vec![20.0, 1000.0];
        let h = vec![0.0, 0.0];
        let q_quiet = prof.q_max_bounds(&f, &h, 75.0, 3.0, 12.0);
        let q_loud = prof.q_max_bounds(&f, &h, 92.0, 3.0, 12.0);
        assert_eq!(q_quiet[0], 12.0);
        assert!((q_loud[0] - 2.0).abs() < 1e-9);
    }
}
