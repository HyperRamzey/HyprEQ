//! Alternative synthesis & deployment structures:
//! - Vicanek (2016) matched (decramped) second-order peaking EQ + PEQ cascade fitter
//! - Balázs Bank (2008/2010) fixed-pole parallel second-order filter (`freqpoles` + Cholesky LS)
//!
//! Non-uniform partitioned convolution (Gardner 1995 / Wefers 2014) is deliberately *not* here: this
//! crate renders filters, it does not stream them, and the partition planner formerly in this module
//! had no caller and produced a schedule that was not zero-latency. See the note at the foot of the
//! file for the rule a real implementation has to satisfy.

use crate::fft::Complex64;
use std::f64::consts::PI;

/// One parametric peaking band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeqBand {
    pub f0_hz: f64,
    pub q: f64,
    pub gain_db: f64,
}

impl PeqBand {
    /// Vicanek (2016) *Matched Second Order Digital Filters* (§4.4, Eqs. 12, 26–29, 44–45).
    /// Returns `[b0, b1, b2, a1, a2]` for `y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]`.
    #[must_use]
    pub fn vicanek_coeffs(&self, fs: f64) -> [f64; 5] {
        let g = 10f64.powf(self.gain_db / 20.0);
        if (g - 1.0).abs() < 1e-12 || self.f0_hz <= 0.0 || self.f0_hz >= 0.5 * fs || self.q <= 0.0 {
            return [1.0, 0.0, 0.0, 0.0, 0.0];
        }
        let w0 = 2.0 * PI * self.f0_hz / fs;
        let q_pole = 1.0 / (2.0 * g.sqrt() * self.q);
        let e = (-q_pole * w0).exp();
        let a1 = if q_pole <= 1.0 {
            -2.0 * e * ((1.0 - q_pole * q_pole).sqrt() * w0).cos()
        } else {
            -2.0 * e * ((q_pole * q_pole - 1.0).sqrt() * w0).cosh()
        };
        let a2 = e * e;

        let s_half = (0.5 * w0).sin();
        let phi1 = s_half * s_half;
        let phi0 = 1.0 - phi1;
        let phi2 = 4.0 * phi0 * phi1;

        let a0_sq = (1.0 + a1 + a2).powi(2);
        let a1_sq = (1.0 - a1 + a2).powi(2);
        let a2_sq = -4.0 * a2;

        let g2 = g * g;
        let b0_sq = a0_sq;
        let r1 = (a0_sq * phi0 + a1_sq * phi1 + a2_sq * phi2) * g2;
        let r2 = (-a0_sq + a1_sq + 4.0 * (phi0 - phi1) * a2_sq) * g2;
        let b2_sq = (r1 - r2 * phi1 - b0_sq) / (4.0 * phi1 * phi1);
        let b1_sq = r2 + b0_sq + 4.0 * (phi1 - phi0) * b2_sq;

        let w = 0.5 * (b0_sq.max(0.0).sqrt() + b1_sq.max(0.0).sqrt());
        let b0 = 0.5 * (w + (w * w + b2_sq).max(0.0).sqrt());
        let b1 = 0.5 * (b0_sq.max(0.0).sqrt() - b1_sq.max(0.0).sqrt());
        let b2 = if b0 > 0.0 { -b2_sq / (4.0 * b0) } else { 0.0 };
        [b0, b1, b2, a1, a2]
    }
}

/// Evaluate the magnitude response in dB of a biquad cascade across `f_hz`.
#[must_use]
pub fn cascade_mag_db(bands: &[PeqBand], f_hz: &[f64], fs: f64) -> Vec<f64> {
    let coeffs: Vec<[f64; 5]> = bands.iter().map(|b| b.vicanek_coeffs(fs)).collect();
    f_hz.iter()
        .map(|&f| {
            let w = 2.0 * PI * f / fs;
            let z1 = Complex64::from_polar(1.0, -w);
            let z2 = z1 * z1;
            let mut db = 0.0;
            for &[b0, b1, b2, a1, a2] in &coeffs {
                let num = Complex64::new(b0, 0.0) + z1 * b1 + z2 * b2;
                let den = Complex64::new(1.0, 0.0) + z1 * a1 + z2 * a2;
                db += 20.0 * (num / den).norm().max(1e-18).log10();
            }
            db
        })
        .collect()
}

/// Fit an `n_bands` Vicanek PEQ cascade via greedy peak initialization + coordinate search.
#[must_use]
pub fn fit_peq_cascade(
    f_hz: &[f64],
    target_db: &[f64],
    n_bands: usize,
    fs: f64,
    q_penalty: f64,
) -> Vec<PeqBand> {
    let k = f_hz.len().min(target_db.len());
    if k < 2 || n_bands == 0 || fs <= 0.0 {
        return Vec::new();
    }
    let f_grid = &f_hz[..k];
    let target = &target_db[..k];
    let f_hi_lim = (0.42 * fs).min(18_000.0);

    let mut bands = Vec::with_capacity(n_bands);
    let mut cur = vec![0.0; k];
    for _ in 0..n_bands {
        let (mut best_i, mut best_abs) = (0usize, -1.0f64);
        for i in 0..k {
            let d = (target[i] - cur[i]).abs();
            if d > best_abs {
                best_abs = d;
                best_i = i;
            }
        }
        bands.push(PeqBand {
            f0_hz: f_grid[best_i].clamp(20.0, f_hi_lim),
            q: 1.4,
            gain_db: (target[best_i] - cur[best_i]).clamp(-12.0, 12.0),
        });
        cur = cascade_mag_db(&bands, f_grid, fs);
    }

    let loss = |bs: &[PeqBand]| -> f64 {
        let pred = cascade_mag_db(bs, f_grid, fs);
        let mse = pred
            .iter()
            .zip(target)
            .map(|(&p, &t)| (p - t) * (p - t))
            .sum::<f64>()
            / (k as f64);
        let qp: f64 = bs.iter().map(|b| (b.q / 10.0).powi(2)).sum();
        mse + q_penalty * qp
    };

    let mut step = [0.15, 0.25, 1.5];
    let mut best_l = loss(&bands);
    for _ in 0..25 {
        let mut improved = false;
        for b_idx in 0..n_bands {
            for &sgn in &[-1.0, 1.0] {
                let mut cand = bands.clone();
                cand[b_idx].f0_hz =
                    (cand[b_idx].f0_hz * 10f64.powf(sgn * step[0])).clamp(20.0, f_hi_lim);
                let l = loss(&cand);
                if l < best_l {
                    bands = cand;
                    best_l = l;
                    improved = true;
                }
            }
            for &sgn in &[-1.0, 1.0] {
                let mut cand = bands.clone();
                cand[b_idx].q = (cand[b_idx].q * (sgn * step[1]).exp()).clamp(0.3, 10.0);
                let l = loss(&cand);
                if l < best_l {
                    bands = cand;
                    best_l = l;
                    improved = true;
                }
            }
            for &sgn in &[-1.0, 1.0] {
                let mut cand = bands.clone();
                cand[b_idx].gain_db = (cand[b_idx].gain_db + sgn * step[2]).clamp(-12.0, 12.0);
                let l = loss(&cand);
                if l < best_l {
                    bands = cand;
                    best_l = l;
                    improved = true;
                }
            }
        }
        if !improved {
            step = [step[0] * 0.5, step[1] * 0.5, step[2] * 0.5];
            if step[2] < 0.02 {
                break;
            }
        }
    }
    bands
}

/// Format fitted bands as an Equalizer APO preset block.
#[must_use]
pub fn to_equalizer_apo_text(bands: &[PeqBand], preamp_db: f64) -> String {
    let mut out = format!("Preamp: {preamp_db:.2} dB\n");
    for (i, b) in bands.iter().enumerate() {
        out.push_str(&format!(
            "Filter {}: ON PK Fc {:.1} Hz Gain {:.2} dB Q {:.2}\n",
            i + 1,
            b.f0_hz,
            b.gain_db,
            b.q
        ));
    }
    out
}

/// Balázs Bank (2008/2010) Fixed-Pole Parallel Second-Order Filter:
/// `H(z) = d0 + sum_k (b0_k + b1_k z^-1) / (1 + a1_k z^-1 + a2_k z^-2)`.
#[derive(Clone, Debug)]
pub struct BankParallelIir {
    pub d0: f64,
    /// Each section holds `[b0, b1, a1, a2]`.
    pub sections: Vec<[f64; 4]>,
    pub fs: f64,
}

impl BankParallelIir {
    /// Balázs Bank `freqpoles` pole placement (`p_k = exp(-dwp/2) * exp(j * wp)`).
    #[must_use]
    pub fn freqpoles(pole_freqs_hz: &[f64], fs: f64) -> Vec<Complex64> {
        let n = pole_freqs_hz.len();
        if n == 0 || fs <= 0.0 {
            return Vec::new();
        }
        let wp: Vec<f64> = pole_freqs_hz
            .iter()
            .map(|&f| 2.0 * PI * f.clamp(1.0, 0.49 * fs) / fs)
            .collect();
        let mut dwp = vec![0.0; n];
        if n == 1 {
            dwp[0] = 0.1 * wp[0];
        } else {
            dwp[0] = (wp[1] - wp[0]).abs();
            for k in 1..(n - 1) {
                dwp[k] = 0.5 * (wp[k + 1] - wp[k - 1]).abs();
            }
            dwp[n - 1] = (wp[n - 1] - wp[n - 2]).abs();
        }
        wp.iter()
            .zip(dwp)
            .map(|(&w, dw)| Complex64::from_polar((-0.5 * dw).exp(), w))
            .collect()
    }

    /// Fit a minimum-phase magnitude curve `ctrl_pts` (`(freq_hz, gain_db)`) with `n_sections`
    /// log-spaced parallel second-order sections via weighted least squares + Cholesky.
    #[must_use]
    pub fn fit_from_ctrl(
        ctrl_pts: &[(f64, f64)],
        fs: f64,
        n_sections: usize,
        reg: f64,
    ) -> Option<Self> {
        if ctrl_pts.len() < 2 || n_sections == 0 || fs <= 0.0 {
            return None;
        }
        let n_fft = 16384usize;
        let ir = crate::ir::synthesize_minimum_phase_ir(
            ctrl_pts, fs as u32, n_fft, n_fft, 0, 0.0, false, 16.0,
        )
        .ok()?;
        // The minimum-phase synthesis applies a headroom preamp so the loudest band sits at
        // 0 dB. Undo that constant gain here: the caller's control curve is the magnitude to
        // fit, not the headroom-shifted spectrum the IR was built from.
        let preamp_scale = 10f64.powf(-ir.preamp_db / 20.0);
        let spec: Vec<Complex64> = crate::fft::rfft(&ir.ir, n_fft)
            .into_iter()
            .map(|c| c * preamp_scale)
            .collect();
        let spec_freqs = crate::fft::rfftfreq(n_fft, fs);
        let re_all: Vec<f64> = spec.iter().map(|c| c.re).collect();
        let im_all: Vec<f64> = spec.iter().map(|c| c.im).collect();

        let f_eval = crate::fft::geomspace(20.0, (0.47 * fs).min(20_000.0), 240);
        let h_eval: Vec<Complex64> = f_eval
            .iter()
            .map(|&f| {
                Complex64::new(
                    crate::fft::interp(f, &spec_freqs, &re_all),
                    crate::fft::interp(f, &spec_freqs, &im_all),
                )
            })
            .collect();

        let pole_freqs = crate::fft::geomspace(25.0, (0.45 * fs).min(19_000.0), n_sections);
        let poles = Self::freqpoles(&pole_freqs, fs);
        let a1s: Vec<f64> = poles.iter().map(|p| -2.0 * p.re).collect();
        let a2s: Vec<f64> = poles.iter().map(|p| p.norm_sqr()).collect();

        let cols = 1 + 2 * n_sections;
        let mut ata = vec![0.0; cols * cols];
        let mut aty = vec![0.0; cols];
        let mut row_re = vec![0.0; cols];
        let mut row_im = vec![0.0; cols];

        for (&f, &h) in f_eval.iter().zip(&h_eval) {
            let wt = 1.0 / h.norm_sqr().max(1e-6);
            let w = 2.0 * PI * f / fs;
            let z1 = Complex64::from_polar(1.0, -w);
            let z2 = z1 * z1;
            row_re[0] = 1.0;
            row_im[0] = 0.0;
            for k in 0..n_sections {
                let den = Complex64::new(1.0, 0.0) + z1 * a1s[k] + z2 * a2s[k];
                let b0 = Complex64::new(1.0, 0.0) / den;
                let b1 = z1 * b0;
                row_re[1 + 2 * k] = b0.re;
                row_im[1 + 2 * k] = b0.im;
                row_re[1 + 2 * k + 1] = b1.re;
                row_im[1 + 2 * k + 1] = b1.im;
            }
            for i in 0..cols {
                aty[i] += wt * (row_re[i] * h.re + row_im[i] * h.im);
                for j in 0..=i {
                    let v = wt * (row_re[i] * row_re[j] + row_im[i] * row_im[j]);
                    ata[i * cols + j] += v;
                }
            }
        }
        for i in 0..cols {
            ata[i * cols + i] += reg.max(1e-12);
            for j in 0..i {
                ata[j * cols + i] = ata[i * cols + j];
            }
        }

        let x = cholesky_solve(cols, &ata, &aty)?;
        let d0 = x[0];
        let sections = (0..n_sections)
            .map(|k| [x[1 + 2 * k], x[1 + 2 * k + 1], a1s[k], a2s[k]])
            .collect();
        Some(Self { d0, sections, fs })
    }

    #[must_use]
    pub fn eval_mag_db(&self, f_hz: &[f64]) -> Vec<f64> {
        f_hz.iter()
            .map(|&f| {
                let w = 2.0 * PI * f / self.fs;
                let z1 = Complex64::from_polar(1.0, -w);
                let z2 = z1 * z1;
                let mut acc = Complex64::new(self.d0, 0.0);
                for &[b0, b1, a1, a2] in &self.sections {
                    acc += (Complex64::new(b0, 0.0) + z1 * b1)
                        / (Complex64::new(1.0, 0.0) + z1 * a1 + z2 * a2);
                }
                20.0 * acc.norm().max(1e-18).log10()
            })
            .collect()
    }
}

fn cholesky_solve(n: usize, a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !s.is_finite() || s <= 0.0 {
                    return None;
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    Some(x)
}

// The Gardner (1995) / Wefers (2014) NUPOLS partition planner that used to live here was removed:
// nothing in this crate convolves with a partitioned schedule, so it was an unused public API, and
// the schedule it produced was not usable even if something had called it. A segment of length L at
// IR offset O needs O >= L input samples before it can be computed with zero added I/O latency, and
// doubling `blk` after two equal blocks preserves exactly that while quadrupling it breaks it — for
// a 128-sample direct head the planner emitted offset 384 / len 512 (then 1408 < 2048, 5504 < 8192),
// i.e. it asked for 512 samples of a block that had not arrived yet. The rule is recorded here so a
// real implementation in the streaming side project starts from it: build the schedule as
// `offset >= len`, assert that per segment, and grow by a factor of two.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pchip::Pchip;

    #[test]
    fn vicanek_peaking_matches_exact_gain_near_nyquist() {
        let fs = 48_000.0;
        let band = PeqBand {
            f0_hz: 19_500.0,
            q: 1.8,
            gain_db: -6.0,
        };
        let mag = cascade_mag_db(&[band], &[1.0, 19_500.0], fs);
        assert!(mag[0].abs() < 1e-6);
        assert!((mag[1] + 6.0).abs() < 1e-5);
    }

    #[test]
    fn bank_parallel_iir_fits_headphone_curve_accurately() {
        let ctrl = vec![
            (20.0, 3.5),
            (60.0, 4.0),
            (250.0, 0.0),
            (1800.0, 2.5),
            (8200.0, -5.0),
            (20000.0, 0.0),
        ];
        let bank = BankParallelIir::fit_from_ctrl(&ctrl, 48_000.0, 32, 1e-8).expect("fits");
        let f_eval = crate::fft::geomspace(25.0, 18_000.0, 64);
        let got = bank.eval_mag_db(&f_eval);
        let (fs, gs): (Vec<f64>, Vec<f64>) = ctrl.into_iter().unzip();
        let want = Pchip::new(&fs, &gs).eval_grid(&f_eval);
        let rms = (got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / (got.len() as f64))
            .sqrt();
        assert!(rms < 0.25, "parallel IIR RMS too high: {rms:.4} dB");
    }
}
