//! Parameterized target families, spatial-audio slope (Mülleder et al. 2025),
//! Bradley–Terry pairwise preference learning, and ISO 226:2023 Formula (1) contour deltas.

use crate::pchip::Pchip;
use std::path::Path;

/// Selectable target family profiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetFamily {
    Harman,
    DiffuseField,
    /// Diffuse field + ~ -1.0 dB/octave spectral slope for spatial audio (Mülleder et al., AES 2025).
    SpatialDiffuseSlope,
    Custom,
}

/// Continuous listener preference modifiers over any baseline target curve.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TargetParams {
    /// Bass shelf gain around 105 Hz (dB).
    pub theta_bass: f64,
    /// Presence / ear-gain trim around 2.5 kHz (dB).
    pub theta_mid: f64,
    /// Log-frequency tilt (dB/octave, anchored to 0 dB at 1 kHz).
    pub theta_tilt: f64,
}

impl TargetParams {
    #[must_use]
    pub fn features(f_hz: f64) -> [f64; 3] {
        // Smooth 2nd-order-like low-shelf basis, broad mid bell, and octave tilt.
        let x_bass = (f_hz / 105.0).powi(2);
        let b = 1.0 / (1.0 + x_bass);
        let z_mid = (f_hz / 2500.0).log2() / 0.65;
        let m = (-0.5 * z_mid * z_mid).exp();
        let s = (f_hz / 1000.0).log2();
        [b, m, s]
    }

    #[must_use]
    pub fn delta_db(&self, f_hz: f64) -> f64 {
        let [b, m, s] = Self::features(f_hz);
        self.theta_bass * b + self.theta_mid * m + self.theta_tilt * s
    }
}

/// Evaluate `T(f; theta)` over `f_grid` from a baseline curve and optional spatial slope.
#[must_use]
pub fn evaluate_target(
    base_pts: &[(f64, f64)],
    f_grid: &[f64],
    family: TargetFamily,
    params: TargetParams,
) -> Vec<f64> {
    if base_pts.len() < 2 {
        return vec![0.0; f_grid.len()];
    }
    let (fs, gs): (Vec<f64>, Vec<f64>) = base_pts.iter().copied().unzip();
    let base = Pchip::new(&fs, &gs).eval_grid(f_grid);
    let spatial_slope = match family {
        TargetFamily::SpatialDiffuseSlope => -1.0,
        _ => 0.0,
    };
    f_grid
        .iter()
        .zip(base)
        .map(|(&f, b)| {
            let slope = if f > 0.0 {
                spatial_slope * (f / 1000.0).log2()
            } else {
                0.0
            };
            b + slope + params.delta_db(f)
        })
        .collect()
}

/// One pairwise preference outcome: `winner` preferred over `loser` in `(bass, mid, tilt)` space.
#[derive(Clone, Copy, Debug)]
pub struct PairwiseChoice {
    pub winner: [f64; 3],
    pub loser: [f64; 3],
}

/// Fit MAP Bradley–Terry preference weights via regularized Newton–Raphson (IRLS).
#[must_use]
pub fn fit_bradley_terry(choices: &[PairwiseChoice], prior_var: f64) -> TargetParams {
    let mut theta = [0.0f64; 3];
    if choices.is_empty() || prior_var <= 0.0 {
        return TargetParams::default();
    }
    let inv_prior = 1.0 / prior_var;
    for _ in 0..20 {
        let mut grad = [
            -theta[0] * inv_prior,
            -theta[1] * inv_prior,
            -theta[2] * inv_prior,
        ];
        let mut hess = [
            [inv_prior, 0.0, 0.0],
            [0.0, inv_prior, 0.0],
            [0.0, 0.0, inv_prior],
        ];
        for c in choices {
            let d = [
                c.winner[0] - c.loser[0],
                c.winner[1] - c.loser[1],
                c.winner[2] - c.loser[2],
            ];
            let dot = theta[0] * d[0] + theta[1] * d[1] + theta[2] * d[2];
            let p = 1.0 / (1.0 + (-dot).exp());
            let w = p * (1.0 - p);
            for i in 0..3 {
                grad[i] += (1.0 - p) * d[i];
                for j in 0..3 {
                    hess[i][j] += w * d[i] * d[j];
                }
            }
        }
        let Some(step) = solve_3x3(hess, grad) else {
            break;
        };
        let norm = (step[0] * step[0] + step[1] * step[1] + step[2] * step[2]).sqrt();
        for i in 0..3 {
            theta[i] += step[i];
        }
        if norm < 1e-10 {
            break;
        }
    }
    TargetParams {
        theta_bass: theta[0],
        theta_mid: theta[1],
        theta_tilt: theta[2],
    }
}

fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let mut m = [
        [a[0][0], a[0][1], a[0][2], b[0]],
        [a[1][0], a[1][1], a[1][2], b[1]],
        [a[2][0], a[2][1], a[2][2], b[2]],
    ];
    for col in 0..3 {
        let mut piv = col;
        for row in (col + 1)..3 {
            if m[row][col].abs() > m[piv][col].abs() {
                piv = row;
            }
        }
        m.swap(col, piv);
        let div = m[col][col];
        if !div.is_finite() || div.abs() < 1e-15 {
            return None;
        }
        let pivot = m[col];
        for below in m.iter_mut().skip(col + 1) {
            let f = below[col] / div;
            for (k, cell) in below.iter_mut().enumerate().skip(col) {
                *cell -= f * pivot[k];
            }
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let mut s = m[row][3];
        for col in (row + 1)..3 {
            s -= m[row][col] * x[col];
        }
        x[row] = s / m[row][row];
    }
    Some(x)
}

/// One row of ISO 226:2023 Table 1 (supplied externally by the user; ISO table data is
/// copyrighted by ISO and is deliberately not embedded in this MIT crate).
#[derive(Clone, Copy, Debug)]
pub struct Iso226Entry {
    pub freq_hz: f64,
    pub alpha_f: f64,
    pub l_u: f64,
    pub t_f: f64,
}

/// External ISO 226:2023 Table 1 loader and Formula (1) contour-delta evaluator.
#[derive(Clone, Debug)]
pub struct Iso226Table {
    entries: Vec<Iso226Entry>,
}

impl Iso226Table {
    pub fn from_entries(mut entries: Vec<Iso226Entry>) -> Option<Self> {
        entries.retain(|e| {
            e.freq_hz.is_finite()
                && e.freq_hz > 0.0
                && e.alpha_f.is_finite()
                && e.alpha_f > 0.0
                && e.l_u.is_finite()
                && e.t_f.is_finite()
        });
        entries.sort_by(|a, b| a.freq_hz.total_cmp(&b.freq_hz));
        (entries.len() >= 2).then_some(Self { entries })
    }

    /// Load user-provided JSON array of `{"freq_hz": ..., "alpha_f": ..., "l_u": ..., "t_f": ...}`.
    pub fn from_json_file(path: &Path) -> crate::error::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let val: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| crate::error::Error::msg(format!("invalid ISO 226 JSON: {e}")))?;
        let arr = val
            .as_array()
            .ok_or_else(|| crate::error::Error::msg("ISO 226 file must be a JSON array"))?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            let f = item.get("freq_hz").and_then(|v| v.as_f64());
            let a = item.get("alpha_f").and_then(|v| v.as_f64());
            let u = item.get("l_u").and_then(|v| v.as_f64());
            let t = item.get("t_f").and_then(|v| v.as_f64());
            if let (Some(freq_hz), Some(alpha_f), Some(l_u), Some(t_f)) = (f, a, u, t) {
                out.push(Iso226Entry {
                    freq_hz,
                    alpha_f,
                    l_u,
                    t_f,
                });
            }
        }
        Self::from_entries(out)
            .ok_or_else(|| crate::error::Error::msg("ISO 226 file holds fewer than 2 valid rows"))
    }

    /// ISO 226:2023 Formula (1) pure-tone SPL (dB) at loudness level `phon`.
    #[must_use]
    pub fn spl_at_entry(e: &Iso226Entry, phon: f64) -> f64 {
        const ALPHA_R: f64 = 0.30;
        const T_R: f64 = 2.4;
        const P_RATIO_SQ: f64 = 4.0e-10; // (20 uPa / 1 Pa)^2
        let term1 = P_RATIO_SQ.powf(ALPHA_R - e.alpha_f)
            * (10f64.powf(ALPHA_R * phon / 10.0) - 10f64.powf(ALPHA_R * T_R / 10.0));
        let term2 = 10f64.powf(e.alpha_f * (e.t_f + e.l_u) / 10.0);
        let a_f = (term1 + term2).max(1e-30);
        (10.0 / e.alpha_f) * a_f.log10() - e.l_u
    }

    /// Contour-delta compensation `(L_p(listen, f) - L_p(listen, 1k)) - (L_p(ref, f) - L_p(ref, 1k))`,
    /// interpolated in `log10(f)` and clamped flat outside the tabulated band.
    #[must_use]
    pub fn contour_delta_db(&self, f_grid: &[f64], listen_phon: f64, ref_phon: f64) -> Vec<f64> {
        let fs: Vec<f64> = self.entries.iter().map(|e| e.freq_hz).collect();
        let diff_at_knots: Vec<f64> = self
            .entries
            .iter()
            .map(|e| {
                let l_cur = Self::spl_at_entry(e, listen_phon);
                let l_ref = Self::spl_at_entry(e, ref_phon);
                l_cur - l_ref
            })
            .collect();
        let p = Pchip::new(&fs, &diff_at_knots);
        let at_1k = p.eval(1000.0);
        p.eval_grid(f_grid).into_iter().map(|v| v - at_1k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso226_2023_formula1_passes_through_phon_at_1khz() {
        let e = Iso226Entry {
            freq_hz: 1000.0,
            alpha_f: 0.30,
            l_u: 0.0,
            t_f: 2.4,
        };
        for phon in [20.0, 40.0, 65.0, 83.0] {
            let spl = Iso226Table::spl_at_entry(&e, phon);
            assert!((spl - phon).abs() < 1e-12, "{spl} vs {phon}");
        }
    }

    #[test]
    fn bradley_terry_recovers_preference_direction() {
        let true_w = [2.0, -0.5, -1.0];
        let mut choices = Vec::new();
        for i in 0..12 {
            let a = [(i % 5) as f64 - 2.0, 0.5, -1.0];
            let b = [1.0 - (i % 3) as f64, -1.0, 1.0];
            let sa: f64 = true_w.iter().zip(a).map(|(w, x)| w * x).sum();
            let sb: f64 = true_w.iter().zip(b).map(|(w, x)| w * x).sum();
            if sa >= sb {
                choices.push(PairwiseChoice {
                    winner: a,
                    loser: b,
                });
            } else {
                choices.push(PairwiseChoice {
                    winner: b,
                    loser: a,
                });
            }
        }
        let est = fit_bradley_terry(&choices, 4.0);
        let dot =
            true_w[0] * est.theta_bass + true_w[1] * est.theta_mid + true_w[2] * est.theta_tilt;
        assert!(dot > 0.0, "estimated preference misaligned: {est:?}");
    }
}
