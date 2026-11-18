//! Global curvature-constrained box-QP regularizer on a uniform log-frequency grid.
//!
//! Minimizes:
//! `J(q) = sum_k w_k (q_k - r_k)^2 + lambda_0 sum_k (q_k - a_k)^2 + lambda_2 sum_k (Delta^2 q_k)^2`
//! subject to `q_min_k <= q_k <= q_max_k`, where `Delta^2 q_k = q_{k+1} - 2 q_k + q_{k-1}`
//! operates in log2-frequency (octave) space.
//!
//! Solved in O(K) per iteration via a single pentadiagonal LDL^T factorization paired with
//! over-relaxed ADMM (`alpha = 1.6`), with zero external linear-algebra crates.

/// Uniform log-frequency grid at `bins_per_octave` resolution (48 => Harman PPR grid).
#[must_use]
pub fn uniform_log_grid(f_min_hz: f64, f_max_hz: f64, bins_per_octave: usize) -> Vec<f64> {
    if !f_min_hz.is_finite()
        || !f_max_hz.is_finite()
        || f_min_hz <= 0.0
        || f_max_hz <= f_min_hz
        || bins_per_octave == 0
    {
        return Vec::new();
    }
    let octaves = (f_max_hz / f_min_hz).log2();
    let steps = (octaves * (bins_per_octave as f64)).round() as usize;
    let mut grid = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let f = f_min_hz * 2f64.powf((i as f64) / (bins_per_octave as f64));
        if f <= f_max_hz * (1.0 + 1e-12) {
            grid.push(f.min(f_max_hz));
        }
    }
    grid
}

/// Configuration for the global curvature-constrained box-QP optimizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalFitConfig {
    /// Unity-anchor penalty (keeps uncertain regions near `anchor_db`, typically 0.0 dB).
    pub lambda_0: f64,
    /// Second-difference curvature penalty in octave space.
    pub lambda_2: f64,
    /// Variance floor `(dB)` preventing infinite weights where `sigma_total == 0`.
    pub sigma_floor_db: f64,
    /// Scale per-bin weights by ERB bandwidth density (`erb_hz(1000) / erb_hz(f)`).
    pub use_erb_weights: bool,
    /// Over-relaxation factor in `(1.0, 1.8)` for ADMM.
    pub admm_alpha: f64,
    /// Maximum ADMM iterations when box constraints activate.
    pub max_iters: usize,
    /// Stopping tolerance on primal and dual residuals.
    pub tol: f64,
}

impl Default for GlobalFitConfig {
    fn default() -> Self {
        Self {
            lambda_0: 0.02,
            lambda_2: 50.0,
            sigma_floor_db: 0.30,
            use_erb_weights: true,
            admm_alpha: 1.6,
            max_iters: 600,
            tol: 1e-8,
        }
    }
}

/// Problem vectors on a common log-frequency grid.
#[derive(Clone, Debug)]
pub struct GlobalFitInput {
    pub f_hz: Vec<f64>,
    pub raw_error_db: Vec<f64>,
    pub total_sigma_db: Vec<f64>,
    pub anchor_db: Vec<f64>,
    pub q_min_db: Vec<f64>,
    pub q_max_db: Vec<f64>,
}

/// Output of [`solve_global_qp`].
#[derive(Clone, Debug)]
pub struct GlobalFitResult {
    pub f_hz: Vec<f64>,
    pub q_db: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

struct PentadiagLdlt {
    d: Vec<f64>,
    l1: Vec<f64>,
    l2: Vec<f64>,
}

impl PentadiagLdlt {
    fn factor(d0: &[f64], d1: &[f64], d2: &[f64]) -> Option<Self> {
        let k = d0.len();
        if k == 0 || d1.len() != k.saturating_sub(1) || d2.len() != k.saturating_sub(2) {
            return None;
        }
        let mut d = vec![0.0; k];
        let mut l1 = vec![0.0; k.saturating_sub(1)];
        let mut l2 = vec![0.0; k.saturating_sub(2)];
        for i in 0..k {
            let mut val = d0[i];
            if i >= 1 {
                val -= l1[i - 1] * l1[i - 1] * d[i - 1];
            }
            if i >= 2 {
                val -= l2[i - 2] * l2[i - 2] * d[i - 2];
            }
            if !val.is_finite() || val <= 1e-18 {
                return None;
            }
            d[i] = val;
            if i + 1 < k {
                let mut num = d1[i];
                if i >= 1 {
                    num -= l2[i - 1] * l1[i - 1] * d[i - 1];
                }
                l1[i] = num / val;
            }
            if i + 2 < k {
                l2[i] = d2[i] / val;
            }
        }
        Some(Self { d, l1, l2 })
    }

    fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        let k = self.d.len();
        let mut y = rhs.to_vec();
        for i in 0..k {
            if i >= 1 {
                y[i] -= self.l1[i - 1] * y[i - 1];
            }
            if i >= 2 {
                y[i] -= self.l2[i - 2] * y[i - 2];
            }
        }
        let mut x = vec![0.0; k];
        for i in (0..k).rev() {
            let mut v = y[i] / self.d[i];
            if i + 1 < k {
                v -= self.l1[i] * x[i + 1];
            }
            if i + 2 < k {
                v -= self.l2[i] * x[i + 2];
            }
            x[i] = v;
        }
        x
    }
}

fn build_system(w: &[f64], lam0: f64, lam2: f64, rho: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let k = w.len();
    let mut d0: Vec<f64> = w.iter().map(|&wi| wi + lam0 + rho).collect();
    let mut d1 = vec![0.0; k.saturating_sub(1)];
    let mut d2 = vec![0.0; k.saturating_sub(2)];
    for j in 0..k.saturating_sub(2) {
        d0[j] += lam2;
        d0[j + 1] += 4.0 * lam2;
        d0[j + 2] += lam2;
        d1[j] += -2.0 * lam2;
        d1[j + 1] += -2.0 * lam2;
        d2[j] += lam2;
    }
    (d0, d1, d2)
}

/// Solve the global curvature-constrained box-QP.
///
/// First attempts a single exact unconstrained `LDL^T` solve; if all box bounds already hold,
/// returns immediately in 1 solve. Otherwise runs over-relaxed ADMM with cached `LDL^T`.
#[must_use]
pub fn solve_global_qp(input: &GlobalFitInput, cfg: &GlobalFitConfig) -> Option<GlobalFitResult> {
    let k = input.f_hz.len();
    if k == 0
        || input.raw_error_db.len() != k
        || input.total_sigma_db.len() != k
        || input.anchor_db.len() != k
        || input.q_min_db.len() != k
        || input.q_max_db.len() != k
    {
        return None;
    }

    let lam0 = cfg.lambda_0.max(1e-9);
    let lam2 = cfg.lambda_2.max(0.0);
    let floor_sq = cfg.sigma_floor_db.max(1e-4).powi(2);

    let mut w = Vec::with_capacity(k);
    for (&f, &sig) in input.f_hz.iter().zip(&input.total_sigma_db) {
        let var = if sig.is_finite() {
            sig * sig + floor_sq
        } else {
            1e12
        };
        let erb_w = if cfg.use_erb_weights && f.is_finite() && f > 0.0 {
            crate::psycho::erb_density(f)
        } else {
            1.0
        };
        w.push(erb_w / var);
    }

    let rhs0: Vec<f64> = w
        .iter()
        .zip(&input.raw_error_db)
        .zip(&input.anchor_db)
        .map(|((&wi, &ri), &ai)| {
            let r_safe = if ri.is_finite() { ri } else { 0.0 };
            let a_safe = if ai.is_finite() { ai } else { 0.0 };
            wi * r_safe + lam0 * a_safe
        })
        .collect();

    // Fast path: unconstrained exact solve; return in 1 step if bounds are inactive.
    let (u0, u1, u2) = build_system(&w, lam0, lam2, 0.0);
    let unconst = PentadiagLdlt::factor(&u0, &u1, &u2)?;
    let q0 = unconst.solve(&rhs0);
    let bounds_ok = q0
        .iter()
        .zip(input.q_min_db.iter().zip(&input.q_max_db))
        .all(|(&q, (&lo, &hi))| q >= lo && q <= hi);
    if bounds_ok {
        return Some(GlobalFitResult {
            f_hz: input.f_hz.clone(),
            q_db: q0,
            iterations: 1,
            converged: true,
        });
    }

    // Box-constrained ADMM with cached pentadiagonal LDL^T.
    let w_med = crate::fit::median_db(&w);
    let rho = (2.0 * (w_med + lam0)).clamp(0.5, 10.0);
    let (d0, d1, d2) = build_system(&w, lam0, lam2, rho);
    let factor = PentadiagLdlt::factor(&d0, &d1, &d2)?;

    let mut z: Vec<f64> = q0
        .iter()
        .zip(input.q_min_db.iter().zip(&input.q_max_db))
        .map(|(&q, (&lo, &hi))| q.clamp(lo.min(hi), hi.max(lo)))
        .collect();
    let mut u = vec![0.0; k];
    let mut rhs = vec![0.0; k];
    let alpha = cfg.admm_alpha.clamp(1.0, 1.8);

    for it in 0..cfg.max_iters {
        for i in 0..k {
            rhs[i] = rhs0[i] + rho * (z[i] - u[i]);
        }
        let q = factor.solve(&rhs);
        let mut r_pri = 0.0f64;
        let mut r_dual = 0.0f64;
        for i in 0..k {
            let q_hat = alpha * q[i] + (1.0 - alpha) * z[i];
            let lo = input.q_min_db[i].min(input.q_max_db[i]);
            let hi = input.q_min_db[i].max(input.q_max_db[i]);
            let z_new = (q_hat + u[i]).clamp(lo, hi);
            u[i] += q_hat - z_new;
            r_pri = r_pri.max((q[i] - z_new).abs());
            r_dual = r_dual.max(rho * (z_new - z[i]).abs());
            z[i] = z_new;
        }
        if r_pri <= cfg.tol && r_dual <= cfg.tol {
            return Some(GlobalFitResult {
                f_hz: input.f_hz.clone(),
                q_db: z,
                iterations: it + 1,
                converged: true,
            });
        }
    }

    Some(GlobalFitResult {
        f_hz: input.f_hz.clone(),
        q_db: z,
        iterations: cfg.max_iters,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_uncertainty_notch_is_bridged_while_low_uncertainty_peak_holds() {
        let f = uniform_log_grid(20.0, 20000.0, 48);
        let k = f.len();
        let mut err = vec![0.0; k];
        let mut sig = vec![0.3; k];
        for (i, &hz) in f.iter().enumerate() {
            let n_z = (hz / 3700.0).log2() / 0.08;
            let p_z = (hz / 8200.0).log2() / 0.12;
            err[i] = 8.0 * (-0.5 * n_z * n_z).exp() - 5.0 * (-0.5 * p_z * p_z).exp();
            sig[i] += 4.5 * (-0.5 * n_z * n_z).exp();
        }
        let q_max: Vec<f64> = f
            .iter()
            .map(|&hz| if hz > 7000.0 { 0.0 } else { 5.5 })
            .collect();
        let inp = GlobalFitInput {
            f_hz: f.clone(),
            raw_error_db: err,
            total_sigma_db: sig,
            anchor_db: vec![0.0; k],
            q_min_db: vec![-12.0; k],
            q_max_db: q_max.clone(),
        };
        let res = solve_global_qp(&inp, &GlobalFitConfig::default()).expect("qp converges");
        assert!(res.converged);
        let i_3k7 = f
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - 3700.0).abs().total_cmp(&(b.1 - 3700.0).abs()))
            .unwrap()
            .0;
        let i_8k2 = f
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - 8200.0).abs().total_cmp(&(b.1 - 8200.0).abs()))
            .unwrap()
            .0;
        assert!(
            res.q_db[i_3k7] < 4.0,
            "uncertain 3.7k notch was not smoothed/bridged: {}",
            res.q_db[i_3k7]
        );
        assert!(
            res.q_db[i_8k2] < -4.2,
            "certain 8.2k cut was erased: {}",
            res.q_db[i_8k2]
        );
        for (i, &q) in res.q_db.iter().enumerate() {
            assert!(q <= q_max[i] + 1e-9 && q >= -12.0 - 1e-9);
        }
    }
}
