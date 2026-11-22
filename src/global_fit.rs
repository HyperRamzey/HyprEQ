//! Global curvature-constrained box-QP regularizer on a uniform log-frequency grid.
//!
//! Minimizes:
//! `J(q) = sum_k w_k phi_gamma(q_k - r_k) + lambda_0 sum_k (q_k - a_k)^2 + lambda_2 sum_k (Delta^2 q_k)^2`
//! subject to `q_min_k <= q_k <= q_max_k`, where `Delta^2 q_k = q_{k+1} - 2 q_k + q_{k-1}`
//! operates in log2-frequency (octave) space and
//! `phi_gamma(e) = gamma * e^2` for `e > 0` (residual post-EQ peak above target) else `e^2`
//! (residual dip/notch below target).
//!
//! * When `peak_penalty_ratio == 1.0` (the default), `phi_gamma(e) = e^2` is symmetric.
//! * When `peak_penalty_ratio > 1.0`, positive post-EQ overshoot (`q_k - r_k > 0`) is penalized
//!   more heavily than negative residual dips, reflecting Bücklein (JAES 29(3), pp. 126–131,
//!   March 1981), who showed that narrowband peaks are substantially more audible than dips of
//!   equal magnitude and bandwidth. The convex asymmetric-quadratic (expectile) formulation is
//!   HyprEQ's own optimization model of that perceptual asymmetry.
//! * [`fec_uncertainty_sigma_db`] adds an optional acoustic-impedance loading uncertainty prior
//!   in the 4–10 kHz ear-canal standing-wave region for non-FEC (closed-back / in-ear) models,
//!   motivated by Møller, Hammershøi, Jensen & Sørensen (JAES 43(4), pp. 203–217, April 1995).
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
    /// Scale per-bin weights by ERB log density (`(f / erb_hz(f)) / (1000 / erb_hz(1000))`).
    pub use_erb_weights: bool,
    /// Over-relaxation factor in `(1.0, 1.8)` for ADMM.
    pub admm_alpha: f64,
    /// Maximum ADMM iterations when box constraints activate.
    pub max_iters: usize,
    /// Stopping tolerance on primal and dual residuals.
    pub tol: f64,
    /// Bücklein (1981) asymmetric peak-vs-notch penalty ratio `gamma >= 1.0`.
    ///
    /// `1.0` (default) is the symmetric quadratic loss. Values `> 1.0` weight positive post-EQ
    /// overshoot `(q_k - r_k > 0, residual peaks)` by `gamma * w_k` while keeping negative
    /// residual dips at `w_k`, keeping the objective strictly convex and `C^1`-smooth.
    pub peak_penalty_ratio: f64,
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
            peak_penalty_ratio: 1.0,
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
            crate::psycho::erb_log_density(f)
        } else {
            1.0
        };
        w.push(erb_w / var);
    }
    // A bin with no measurement must leave the fit entirely. Leaving its error mapped to
    // `0.0` at full weight (as the rhs0 guard did on its own) is not neutral: it tells
    // the solver the hole measured exactly 0 dB deviation, and the fit then chases flat
    // right where nothing was measured. Zeroing the row weight drops the data term; the
    // anchor and curvature terms legitimately still pass over the bin.
    for (wi, &r) in w.iter_mut().zip(&input.raw_error_db) {
        if !r.is_finite() {
            *wi = 0.0;
        }
    }

    let gamma = if cfg.peak_penalty_ratio.is_finite() && cfg.peak_penalty_ratio > 1.0 {
        cfg.peak_penalty_ratio
    } else {
        1.0
    };
    let r_safe: Vec<f64> = input
        .raw_error_db
        .iter()
        .map(|&ri| if ri.is_finite() { ri } else { 0.0 })
        .collect();
    let rhs0: Vec<f64> = w
        .iter()
        .zip(&r_safe)
        .zip(&input.anchor_db)
        .map(|((&wi, &ri), &ai)| {
            let a_safe = if ai.is_finite() { ai } else { 0.0 };
            wi * ri + lam0 * a_safe
        })
        .collect();

    // Fast path: unconstrained exact solve; return in 1 step if bounds and peak asymmetry
    // are both inactive (for gamma == 1.0 this is identical to the symmetric QP).
    let (q_lo, q_hi) = box_bounds(&input.q_min_db, &input.q_max_db);
    let (u0, u1, u2) = build_system(&w, lam0, lam2, 0.0);
    let unconst = PentadiagLdlt::factor(&u0, &u1, &u2)?;
    let q0 = unconst.solve(&rhs0);
    let bounds_ok = q0
        .iter()
        .zip(q_lo.iter().zip(&q_hi))
        .all(|(&q, (&lo, &hi))| q >= lo && q <= hi);
    let asym_inactive = gamma == 1.0
        || q0
            .iter()
            .zip(r_safe.iter().zip(&w))
            .all(|(&q, (&r, &wi))| wi == 0.0 || q <= r);
    if bounds_ok && asym_inactive {
        return Some(GlobalFitResult {
            f_hz: input.f_hz.clone(),
            q_db: q0,
            iterations: 1,
            converged: true,
        });
    }

    // Box-constrained (and optionally Bücklein asymmetric) ADMM with cached pentadiagonal LDL^T.
    // The symmetric data term `0.5 * w_k (q_k - r_k)^2` stays in the q-step; only the excess
    // peak penalty `0.5 * (gamma - 1) * w_k * max(0, z_k - r_k)^2` and box `[lo, hi]` enter
    // the separable z-step proximal operator, keeping the pentadiagonal factorization constant.
    let w_med = crate::fit::median_db(&w);
    let rho = if gamma > 1.0 {
        (2.0 * (0.5 * (1.0 + gamma) * w_med + lam0)).clamp(0.5, 20.0)
    } else {
        (2.0 * (w_med + lam0)).clamp(0.5, 10.0)
    };
    let (d0, d1, d2) = build_system(&w, lam0, lam2, rho);
    let factor = PentadiagLdlt::factor(&d0, &d1, &d2)?;

    let excess: Vec<f64> = w.iter().map(|&wi| (gamma - 1.0) * wi).collect();
    let mut z: Vec<f64> = q0
        .iter()
        .zip(q_lo.iter().zip(&q_hi))
        .map(|(&q, (&lo, &hi))| q.clamp(lo, hi))
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
            let v = q_hat + u[i];
            let z_star = if excess[i] > 0.0 && v > r_safe[i] {
                (excess[i] * r_safe[i] + rho * v) / (excess[i] + rho)
            } else {
                v
            };
            let z_new = z_star.clamp(q_lo[i], q_hi[i]);
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

/// Normalize caller-supplied box bounds to finite, ordered `(lo, hi)` pairs.
///
/// A bound that is not a number is not a value to clamp against — it is the absence of a bound on
/// that side, which is exactly what a driver model with no distortion ceiling, or a QP row over a
/// measurement hole, is saying. It becomes the extreme of the range instead. A pair delivered the
/// wrong way round is swapped. Both cases come out of data files rather than code paths, and
/// `f64::clamp` panics on either ("min > max, or either was NaN"), which would take the whole run
/// down over one bad row.
fn box_bounds(q_min_db: &[f64], q_max_db: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = q_min_db.len().min(q_max_db.len());
    let mut lo = Vec::with_capacity(n);
    let mut hi = Vec::with_capacity(n);
    for i in 0..n {
        let is_q_min_inf = q_min_db[i] == f64::INFINITY;
        let is_q_max_neg_inf = q_max_db[i] == f64::NEG_INFINITY;

        let a = if q_min_db[i].is_nan() || q_min_db[i] == f64::NEG_INFINITY {
            f64::MIN
        } else {
            q_min_db[i]
        };
        let b = if q_max_db[i].is_nan() || q_max_db[i] == f64::INFINITY {
            f64::MAX
        } else {
            q_max_db[i]
        };

        let (l, h) = if is_q_min_inf && q_max_db[i].is_finite() {
            (q_max_db[i], q_max_db[i])
        } else if is_q_max_neg_inf && q_min_db[i].is_finite() {
            (q_min_db[i], q_min_db[i])
        } else if a <= b {
            (a, b)
        } else if a.is_finite() && b.is_finite() {
            (b, a)
        } else {
            (0.0, 0.0)
        };
        lo.push(l);
        hi.push(h);
    }
    (lo, hi)
}

/// Acoustic-impedance loading uncertainty prior `sigma_fec(f)` in dB, motivated by the
/// Free-Air Equivalent Coupling (FEC) criterion of Møller, Hammershøi, Jensen & Sørensen
/// (JAES 43(4), pp. 203–217, April 1995).
///
/// Møller et al. showed that headphones whose radiation impedance seen from the ear canal
/// differs from free air (non-FEC sealed closed-back and in-ear designs) exhibit ear-canal
/// standing-wave loading divergence between couplers and human ears above ~2 kHz, concentrated
/// across 4–10 kHz.
///
/// The smooth `C^1` log-frequency raised-cosine envelope and peak amplitudes (`0.0 dB` for
/// [`AcousticClass::OpenBack`] and [`AcousticClass::Unknown`], `1.5 dB` at 7 kHz for
/// [`AcousticClass::ClosedBack`], and `2.5 dB` at 7 kHz for [`AcousticClass::InEar`]) are
/// HyprEQ's conservative engineering prior rather than a table from the paper.
#[must_use]
pub fn fec_uncertainty_sigma_db(f_hz: f64, class: crate::fixture::AcousticClass) -> f64 {
    use crate::fixture::AcousticClass;
    if !f_hz.is_finite() || f_hz <= 0.0 {
        return 0.0;
    }
    let (f_lo, f_peak, f_hi, peak_db) = match class {
        AcousticClass::OpenBack | AcousticClass::Unknown => return 0.0,
        AcousticClass::ClosedBack => (2000.0_f64, 7000.0_f64, 16000.0_f64, 1.5_f64),
        AcousticClass::InEar => (1500.0_f64, 7000.0_f64, 16000.0_f64, 2.5_f64),
    };
    if f_hz <= f_lo || f_hz >= f_hi {
        return 0.0;
    }
    let lf = f_hz.log2();
    let lp = f_peak.log2();
    let t = if f_hz <= f_peak {
        (lf - f_lo.log2()) / (lp - f_lo.log2())
    } else {
        (f_hi.log2() - lf) / (f_hi.log2() - lp)
    };
    // Smooth raised-cosine ramp: value and derivative vanish at endpoints and at the peak.
    let w = 0.5 * (1.0 - (std::f64::consts::PI * t).cos());
    peak_db * w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A NaN bound came from a driver ceiling sentinel and a reversed pair from a hand-edited JSON;
    /// both reached `f64::clamp`, which panics on either, and took the whole run down.
    ///
    /// The demand below is the one the bridging test uses, because its ceiling is genuinely active:
    /// with inert bounds the solver returns on the unconstrained fast path and never runs the
    /// projection, and a test that only asked for "no panic" would pass without touching the line
    /// that was broken. `iterations > 1` is what makes that visible.
    #[test]
    fn a_non_finite_or_reversed_box_bound_constrains_nothing_instead_of_panicking() {
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
        let ceiling: Vec<f64> = f
            .iter()
            .map(|&hz| if hz > 7000.0 { 0.0 } else { 5.5 })
            .collect();
        // Two rows the ceiling is inert on, so their treatment can be compared against the plain
        // meaning of "unconstrained" without the active ceiling moving anything else.
        let i_nan = 3;
        let i_rev = 4;

        let solve = |q_min: Vec<f64>, q_max: Vec<f64>| {
            solve_global_qp(
                &GlobalFitInput {
                    f_hz: f.clone(),
                    raw_error_db: err.clone(),
                    total_sigma_db: sig.clone(),
                    anchor_db: vec![0.0; k],
                    q_min_db: q_min,
                    q_max_db: q_max,
                },
                &GlobalFitConfig::default(),
            )
        };

        let mut q_min = vec![-12.0; k];
        q_min[i_nan] = f64::NAN;
        q_min[i_rev] = 5.5;
        let mut q_max = ceiling.clone();
        q_max[i_rev] = -12.0;

        let got = solve(q_min.clone(), q_max.clone()).expect("a bad bound must not abort the run");
        assert!(
            got.converged,
            "the solver gave up instead of ignoring the bound"
        );
        assert!(
            got.iterations > 1,
            "the ceiling never engaged, so the projection step was never run: {}",
            got.iterations
        );
        assert!(got.q_db.iter().all(|q| q.is_finite()), "{:?}", got.q_db);
        let i_high = f
            .iter()
            .position(|&hz| hz > 7000.0)
            .expect("grid reaches 7 kHz");
        assert!(
            got.q_db[i_high] <= 1e-9,
            "the active ceiling was ignored: {}",
            got.q_db[i_high]
        );

        // The same run, with the two rows written as what they mean: no floor, and a sorted pair.
        let mut want_min = q_min.clone();
        want_min[i_nan] = f64::MIN;
        let mut want_max = q_max.clone();
        std::mem::swap(&mut want_min[i_rev], &mut want_max[i_rev]);
        let want = solve(want_min, want_max).expect("unconstrained twin");
        assert_eq!(
            got.q_db, want.q_db,
            "a non-finite or reversed bound did not behave as the absence of a bound"
        );
    }

    #[test]
    fn infinite_bounds_do_not_invert_into_unbounded() {
        let (lo, hi) = box_bounds(&[f64::INFINITY], &[6.0]);
        assert_eq!(lo[0], 6.0);
        assert_eq!(hi[0], 6.0);

        let (lo2, hi2) = box_bounds(&[-6.0], &[f64::NEG_INFINITY]);
        assert_eq!(lo2[0], -6.0);
        assert_eq!(hi2[0], -6.0);

        let (lo3, hi3) = box_bounds(&[f64::NEG_INFINITY], &[f64::INFINITY]);
        assert_eq!(lo3[0], f64::MIN);
        assert_eq!(hi3[0], f64::MAX);

        let (lo4, hi4) = box_bounds(&[f64::INFINITY], &[f64::INFINITY]);
        assert_eq!(lo4[0], 0.0);
        assert_eq!(hi4[0], 0.0);

        let (lo5, hi5) = box_bounds(&[f64::INFINITY], &[f64::NEG_INFINITY]);
        assert_eq!(lo5[0], 0.0);
        assert_eq!(hi5[0], 0.0);
    }

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

    /// A non-finite error row must leave the data term entirely. Mapping it to `0.0`
    /// at full weight fabricated a "measured 0 dB deviation" at the hole and pulled the
    /// curve to flat exactly there; with the row's weight zeroed, the hole interpolates
    /// between its neighbours like any unmeasured bin.
    #[test]
    fn a_measurement_hole_leaves_the_fit_instead_of_fabricating_a_flat_row() {
        let f = vec![100.0, 200.0, 400.0, 800.0, 1600.0];
        let k = f.len();
        let solve = |err: Vec<f64>| {
            solve_global_qp(
                &GlobalFitInput {
                    f_hz: f.clone(),
                    raw_error_db: err,
                    total_sigma_db: vec![0.3; k],
                    anchor_db: vec![0.0; k],
                    q_min_db: vec![-12.0; k],
                    q_max_db: vec![12.0; k],
                },
                &GlobalFitConfig::default(),
            )
            .expect("small well-scaled system")
        };
        let solid = solve(vec![6.0; k]);
        assert!(
            solid.q_db.iter().all(|&q| q > 5.0),
            "sanity: a flat +6 demand should fit flat-ish: {:?}",
            solid.q_db
        );
        let hole = solve(vec![6.0, 6.0, f64::NAN, 6.0, 6.0]);
        assert!(
            hole.q_db[2] > 4.0,
            "the hole bin was pulled toward an invented 0 dB reading: {} (neighbours {:?})",
            hole.q_db[2],
            hole.q_db
        );
        // The real bins must be untouched to solver tolerance.
        for i in [0usize, 1, 3, 4] {
            assert!(
                (hole.q_db[i] - solid.q_db[i]).abs() < 0.05,
                "hole changed real bin {i}: {} vs {}",
                hole.q_db[i],
                solid.q_db[i]
            );
        }
    }

    #[test]
    fn bucklein_asymmetric_penalty_reduces_positive_overshoot_and_satisfies_kkt() {
        let f = uniform_log_grid(40.0, 12000.0, 12);
        let k = f.len();
        // Alternating sharp dips and peaks so smoothing creates both positive and negative residuals.
        let raw: Vec<f64> = (0..k)
            .map(|i| {
                let x = i as f64 / (k as f64);
                4.0 * (6.0 * std::f64::consts::PI * x).sin() + if i % 5 == 0 { -3.5 } else { 1.0 }
            })
            .collect();
        let inp = GlobalFitInput {
            f_hz: f,
            raw_error_db: raw.clone(),
            total_sigma_db: vec![0.5; k],
            anchor_db: vec![0.0; k],
            q_min_db: vec![-12.0; k],
            q_max_db: vec![3.5; k],
        };
        let cfg_sym = GlobalFitConfig::default();
        let cfg_asym = GlobalFitConfig {
            peak_penalty_ratio: 4.0,
            max_iters: 1200,
            tol: 1e-9,
            ..GlobalFitConfig::default()
        };
        let sym = solve_global_qp(&inp, &cfg_sym).expect("sym");
        let asym = solve_global_qp(&inp, &cfg_asym).expect("asym");
        assert!(sym.converged && asym.converged);

        // Positive residual (q_k - r_k > 0, post-EQ peak above target) must strictly shrink.
        let mean_peak_sym: f64 = sym
            .q_db
            .iter()
            .zip(&raw)
            .map(|(&q, &r)| (q - r).max(0.0))
            .sum::<f64>()
            / (k as f64);
        let mean_peak_asym: f64 = asym
            .q_db
            .iter()
            .zip(&raw)
            .map(|(&q, &r)| (q - r).max(0.0))
            .sum::<f64>()
            / (k as f64);
        assert!(
            mean_peak_asym < mean_peak_sym * 0.65,
            "asymmetric penalty failed to suppress positive peaks: sym={mean_peak_sym:.4}, asym={mean_peak_asym:.4}"
        );

        // Verify KKT projected-gradient optimality of the asymmetric solution.
        let floor_sq = cfg_asym.sigma_floor_db.powi(2);
        let w: Vec<f64> = inp
            .f_hz
            .iter()
            .zip(&inp.total_sigma_db)
            .map(|(&hz, &sig)| crate::psycho::erb_log_density(hz) / (sig * sig + floor_sq))
            .collect();
        let mut l2q = vec![0.0; k];
        for j in 0..k.saturating_sub(2) {
            let t = asym.q_db[j] - 2.0 * asym.q_db[j + 1] + asym.q_db[j + 2];
            l2q[j] += t;
            l2q[j + 1] -= 2.0 * t;
            l2q[j + 2] += t;
        }
        let mut max_kkt = 0.0_f64;
        for i in 0..k {
            let q = asym.q_db[i];
            let we = if q > raw[i] { 4.0 * w[i] } else { w[i] };
            let grad = we * (q - raw[i]) + cfg_asym.lambda_0 * q + cfg_asym.lambda_2 * l2q[i];
            let pg = if q <= inp.q_min_db[i] + 1e-8 {
                grad.min(0.0)
            } else if q >= inp.q_max_db[i] - 1e-8 {
                grad.max(0.0)
            } else {
                grad
            };
            max_kkt = max_kkt.max(pg.abs());
        }
        assert!(
            max_kkt < 1e-5,
            "asymmetric QP KKT residual too large: {max_kkt:.2e}"
        );
    }

    #[test]
    fn bucklein_asymmetric_penalty_ignores_non_finite_data_holes() {
        let f = uniform_log_grid(100.0, 4000.0, 12);
        let k = f.len();
        let mut raw = vec![3.0; k];
        raw[k / 2] = f64::NAN;
        let inp = GlobalFitInput {
            f_hz: f,
            raw_error_db: raw,
            total_sigma_db: vec![0.5; k],
            anchor_db: vec![3.0; k],
            q_min_db: vec![-12.0; k],
            q_max_db: vec![12.0; k],
        };
        let cfg = GlobalFitConfig {
            peak_penalty_ratio: 5.0,
            ..GlobalFitConfig::default()
        };
        let res = solve_global_qp(&inp, &cfg).expect("converges");
        assert!(res.converged);
        // Because the NaN bin has w[i] = 0 and excess[i] = 0, it must NOT be dragged toward 0 dB!
        assert!(
            (res.q_db[k / 2] - 3.0).abs() < 1e-6,
            "NaN hole dragged away from +3 dB flat line: {:.6}",
            res.q_db[k / 2]
        );
    }

    #[test]
    fn fec_uncertainty_prior_distinguishes_open_closed_and_inear() {
        use crate::fixture::AcousticClass;
        // Open-back and Unknown carry zero added FEC prior everywhere.
        for &hz in &[100.0, 1000.0, 4000.0, 7000.0, 10000.0, 20000.0] {
            assert_eq!(fec_uncertainty_sigma_db(hz, AcousticClass::OpenBack), 0.0);
            assert_eq!(fec_uncertainty_sigma_db(hz, AcousticClass::Unknown), 0.0);
        }
        // Closed-back and In-ear vanish in bass/mid and peak at 7 kHz with InEar > ClosedBack.
        assert_eq!(
            fec_uncertainty_sigma_db(1000.0, AcousticClass::ClosedBack),
            0.0
        );
        assert_eq!(fec_uncertainty_sigma_db(1000.0, AcousticClass::InEar), 0.0);
        assert_eq!(
            fec_uncertainty_sigma_db(18000.0, AcousticClass::ClosedBack),
            0.0
        );
        let closed_7k = fec_uncertainty_sigma_db(7000.0, AcousticClass::ClosedBack);
        let inear_7k = fec_uncertainty_sigma_db(7000.0, AcousticClass::InEar);
        assert!((closed_7k - 1.5).abs() < 1e-12);
        assert!((inear_7k - 2.5).abs() < 1e-12);
        assert!(fec_uncertainty_sigma_db(f64::NAN, AcousticClass::InEar) == 0.0);
        assert!(fec_uncertainty_sigma_db(-500.0, AcousticClass::ClosedBack) == 0.0);
    }
}
