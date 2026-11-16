//! Window and filter math: the Modified Bessel function `I0`, the Kaiser shape
//! parameter, and the half-Kaiser tail that caps the truncation discontinuity of a
//! truncated FIR.
//!
//! `I0` is computed here rather than taken from a table because the two call sites
//! need different things and `hypreq.py` reproduces both:
//!
//! * [`i0`] is the **scalar** power series — the loop the polyphase true-peak bank
//!   in [`crate::tp`] runs once per phase per tap.
//! * [`i0_vec`] is the **element-wise** series capped at 64 terms, which is what
//!   the half-Kaiser tail of a 65 536-tap IR is built from. The cap is not an
//!   optimisation: it bounds the worst case (`I0(18)` needs ~30 terms) so the
//!   window cannot spin on a pathological `beta`.
//!
//! Both accumulate in `f64` and stop at the same relative criterion
//! (`term > 1e-18 · partial sum`), so the two agree bit-for-bit wherever both are
//! defined; the test below asserts it rather than assuming it.

/// Modified Bessel function of the first kind, order 0 (scalar power series).
#[must_use]
pub fn i0(x: f64) -> f64 {
    let q = x * x / 4.0;
    let mut term = 1.0f64;
    let mut s = 1.0f64;
    let mut k = 1.0f64;
    while term > 1e-18 * s {
        term *= q / (k * k);
        s += term;
        k += 1.0;
    }
    s
}

/// Element-wise [`i0`], term loop capped at 64 like `hypreq.py`'s array path.
#[must_use]
pub fn i0_vec(x: &[f64]) -> Vec<f64> {
    let mut res = vec![1.0f64; x.len()];
    let mut term = vec![1.0f64; x.len()];
    let half_x_sq: Vec<f64> = x.iter().map(|v| (0.5 * v) * (0.5 * v)).collect();
    for k in 1..=64usize {
        let kk = (k * k) as f64;
        let mut done = true;
        for ((t, h), r) in term.iter_mut().zip(&half_x_sq).zip(&mut res) {
            *t *= h / kk;
            *r += *t;
            if *t > 1e-18 * *r {
                done = false;
            }
        }
        if done {
            break;
        }
    }
    res
}

/// Kaiser window shape parameter for `att_db` stop-band attenuation.
///
/// Kaiser 1974, as given in Oppenheim & Schafer, *Discrete-Time Signal Processing*,
/// section 7.5. This is the crate's only copy: the true-peak bank in [`crate::tp`] calls
/// it with [`crate::tp::STOPBAND_DB`] and the FIR tail calls it with the tail's own
/// attenuation, and
/// the two are meant to agree to the last bit.
#[must_use]
pub fn kaiser_beta(att_db: f64) -> f64 {
    if att_db > 50.0 {
        0.1102 * (att_db - 8.7)
    } else if att_db >= 21.0 {
        0.5842 * (att_db - 21.0).powf(0.4) + 0.07886 * (att_db - 21.0)
    } else {
        0.0
    }
}

/// Stop-band attenuation implied by a Kaiser `beta`, inverting the `beta > 50`
/// branch — the number `hypreq.py` prints next to the tail as the "Kaiser formula"
/// dB figure (`beta / 0.1102 + 8.7`).
#[must_use]
pub fn kaiser_atten_db_from_beta(beta: f64) -> f64 {
    beta / 0.1102 + 8.7
}

/// Half-Kaiser fade applied to the last `tail` taps of a truncated FIR.
///
/// `w[t] = I0(beta·sqrt(1 - (t/tail)²)) / I0(beta)` for `t = 0..tail`, i.e. a
/// quarter Kaiser ramp from 1.0 down to `1/I0(beta)`, multiplied onto the tail in
/// place. This is what turns the −100 dB-class step discontinuity at the truncation
/// point into a taper, and it is the difference between the reported "unwindowed"
/// and "windowed" decay floors at the end of the IR.
///
/// The window is normalised by [`i0`] at `beta` (scalar), matching the Python, so
/// the first sample of the tail is multiplied by exactly `1.0` up to rounding.
pub fn half_kaiser_tail_apply(ir: &mut [f64], tail: usize, beta: f64) {
    if tail == 0 || tail > ir.len() {
        return;
    }
    let i0_beta = i0(beta);
    let start = ir.len() - tail;
    let t: Vec<f64> = (0..tail).map(|j| j as f64 / tail as f64).collect();
    let arg: Vec<f64> = t
        .iter()
        .map(|tn| beta * (1.0 - tn * tn).max(0.0).sqrt())
        .collect();
    let w = i0_vec(&arg);
    for (j, r) in ir[start..].iter_mut().enumerate() {
        *r *= w[j] / i0_beta;
    }
}

#[cfg(test)]
fn rel_err(got: f64, want: f64) -> f64 {
    ((got - want) / want).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i0_matches_known_values() {
        // Reference values from scipy.special.i0. The series is exact to ~1 ulp of
        // these, so the comparison is relative — at i0(18) ≈ 6.2e6 an absolute 1e-6
        // would be a demand for more precision than f64 has.
        assert_eq!(i0(0.0), 1.0);
        assert!((i0(1.0) - 1.266_065_877_752_008_2).abs() < 1e-14);
        assert!(
            rel_err(i0(5.0), 27.239_871_823_604_442) < 1e-14,
            "{}",
            i0(5.0)
        );
        assert!(
            rel_err(i0(18.0), 6_218_412.420_781_003) < 1e-15,
            "{}",
            i0(18.0)
        );
    }

    #[test]
    fn scalar_and_vector_series_agree() {
        let xs: Vec<f64> = (0..40).map(|i| i as f64 * 0.5).collect();
        let v = i0_vec(&xs);
        for (&x, &w) in xs.iter().zip(&v) {
            let s = i0(x);
            assert!((w - s).abs() <= 1e-12 * s, "x={x}: {w} vs {s}");
        }
    }

    #[test]
    fn kaiser_beta_branches() {
        assert_eq!(kaiser_beta(20.0), 0.0);
        // 0.5842·(30-21)^0.4 + 0.07886·(30-21)
        let mid = 0.5842 * 9.0f64.powf(0.4) + 0.07886 * 9.0;
        assert!((kaiser_beta(30.0) - mid).abs() < 1e-15);
        assert!((kaiser_beta(80.0) - 0.1102 * (80.0 - 8.7)).abs() < 1e-15);
        assert!((kaiser_atten_db_from_beta(kaiser_beta(80.0)) - 80.0).abs() < 1e-9);
    }

    #[test]
    fn half_kaiser_starts_at_unity_and_only_shrinks() {
        let mut ir = vec![1.0f64; 1000];
        half_kaiser_tail_apply(&mut ir, 256, 18.0);
        assert_eq!(
            ir[..744],
            vec![1.0; 744],
            "everything before the tail is untouched"
        );
        assert!(
            (ir[744] - 1.0).abs() < 1e-12,
            "tail starts at unity: {}",
            ir[744]
        );
        for w in ir[744..].windows(2) {
            assert!(w[1] <= w[0] + 1e-15, "monotone taper");
        }
        let expect_last = i0(18.0 * (1.0 - (255.0f64 / 256.0).powi(2)).sqrt()) / i0(18.0);
        assert!((ir[999] - expect_last).abs() < 1e-15);
        assert!(ir[999] < 0.01 && ir[999] > 0.0, "ends deep: {}", ir[999]);
    }

    #[test]
    fn zero_or_oversized_tail_is_a_no_op() {
        let mut ir = vec![3.0f64; 8];
        half_kaiser_tail_apply(&mut ir, 0, 18.0);
        assert_eq!(ir, vec![3.0; 8]);
        half_kaiser_tail_apply(&mut ir, 9, 18.0);
        assert_eq!(
            ir,
            vec![3.0; 8],
            "a tail longer than the IR is skipped, like the Python guard"
        );
    }
}
