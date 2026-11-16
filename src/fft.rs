//! NumPy-compatible FFT wrappers over `rustfft`.
//!
//! The whole DSP chain — the Hilbert-cepstral minimum-phase fold, the FIR
//! magnitude/phase used by the verification plot, and the overlap-add convolver —
//! was written against `numpy.fft`, whose conventions are not the ones `rustfft`
//! exposes. Three differences matter, and this module hides exactly those and
//! nothing else:
//!
//! 1. **Normalisation.** `numpy.fft.ifft` divides by `n`; `rustfft`'s inverse
//!    does not (`ifft(fft(x)) == n·x`). [`ifft`] and [`irfft`] carry the `1/n`.
//! 2. **Half-spectrum convention.** `numpy.fft.rfft` returns `n/2+1` bins of the
//!    DFT of the zero-padded input, which is what `rustfft`'s real planner
//!    returns too — so [`rfft`] is a straight pass, and [`irfft`] rebuilds the
//!    Hermitian extension `numpy` assumes (`X[n-k] = conj(X[k])`, with `X[0]` and
//!    `X[n/2]` their own mirrors).
//! 3. **Frequency grid.** [`rfftfreq`] matches `numpy.fft.rfftfreq(n, d=1/fs)`,
//!    including the `k = n/2` Nyquist bin landing on `+fs/2` rather than `-fs/2`.
//!
//! Rounding is not bit-identical to NumPy's pocketfft: two correct f64 DFTs of the
//! same input differ by O(ε·log n) relative, which is ~1e-15 here. The parity
//! tests in this crate pin the consequence (IR samples and dB floors) rather than
//! pretending the bit patterns match.

use rustfft::FftPlanner;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// The complex scalar every transform in this crate runs on — `f64` real and
/// imaginary parts, which is what `numpy.fft` uses for a `float64` input.
pub type Complex64 = rustfft::num_complex::Complex64;

/// A planned transform. `rustfft`'s only transform trait is the complex one; the
/// real transforms below are built on it rather than on a real-input plan.
type Plan = Arc<dyn rustfft::Fft<f64>>;

thread_local! {
    /// Plans are `Send + Sync` and expensive to design, so they are cached once per
    /// thread and shared by `Arc`. The planner itself is not `Sync`, which is why it
    /// lives in its own thread-local rather than inside the cache.
    static PLANS: RefCell<HashMap<(usize, bool), Plan>> = RefCell::new(HashMap::new());
    static PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

/// The cached plan for a complex transform of length `n` in the given direction.
fn plan(n: usize, inverse: bool) -> Plan {
    if let Some(existing) = PLANS.with(|c| c.borrow().get(&(n, inverse)).cloned()) {
        return existing;
    }
    let fresh = PLANNER.with(|p| {
        let mut p = p.borrow_mut();
        if inverse {
            p.plan_fft_inverse(n)
        } else {
            p.plan_fft_forward(n)
        }
    });
    PLANS.with(|c| {
        c.borrow_mut().insert((n, inverse), Arc::clone(&fresh));
    });
    fresh
}

/// Forward DFT, unnormalised, sign `-1`: `X[k] = Σ x[m]·e^{-2πimk/n}`.
#[must_use]
pub fn fft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut buf = x.to_vec();
    plan(n, false).process(&mut buf);
    buf
}

/// Inverse DFT with NumPy's `1/n`: `x[m] = (1/n)·Σ X[k]·e^{+2πimk/n}`.
#[must_use]
pub fn ifft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut buf = x.to_vec();
    plan(n, true).process(&mut buf);
    let inv_n = 1.0 / n as f64;
    for v in &mut buf {
        *v *= inv_n;
    }
    buf
}

/// Zero-pad or truncate `x` to `n`, then return the `n/2+1` non-negative bins of
/// its DFT — `numpy.fft.rfft(x, n)`.
#[must_use]
pub fn rfft(x: &[f64], n: usize) -> Vec<Complex64> {
    let mut buf = vec![Complex64::new(0.0, 0.0); n];
    for (dst, &src) in buf.iter_mut().zip(x) {
        dst.re = src;
    }
    plan(n, false).process(&mut buf);
    buf.truncate(n / 2 + 1);
    buf
}

/// Inverse of [`rfft`]: take the `n/2+1` independent bins, inverse-transform, and
/// divide by `n` — `numpy.fft.irfft(X, n)`.
///
/// The negative-frequency half is *implied*, exactly as in NumPy: the bins are
/// mirror-conjugated into the upper half (`X[n-k] = conj(X[k])`), and the two bins
/// that are their own mirror — DC, and Nyquist when `n` is even — keep only their
/// real part, because a real signal's spectrum has none to give them. NumPy's
/// pocketfft reaches the same result through a packed length-`n/2` complex
/// transform; the arithmetic is identical, the rounding is not, which is the one
/// place in this crate where a bit-for-bit match with NumPy is not on offer.
#[must_use]
pub fn irfft(spec: &[Complex64], n: usize) -> Vec<f64> {
    if n == 0 {
        // NumPy raises `Invalid number of FFT data points (0)`. Here the generic path
        // would write `buf[0]` into a zero-length buffer, so the degenerate size is
        // answered directly with the empty time signal it stands for.
        return Vec::new();
    }
    let bins = n / 2 + 1;
    let mut buf = vec![Complex64::new(0.0, 0.0); n];
    if let Some(dc) = spec.first() {
        buf[0] = Complex64::new(dc.re, 0.0);
    }
    for k in 1..bins {
        let v = spec.get(k).copied().unwrap_or(Complex64::new(0.0, 0.0));
        if k == n - k {
            // The exact self-mirror: the Nyquist bin of an even-length transform.
            buf[k] = Complex64::new(v.re, 0.0);
        } else {
            buf[k] = v;
            buf[n - k] = v.conj();
        }
    }
    plan(n, true).process(&mut buf);
    let inv_n = 1.0 / n as f64;
    buf.into_iter().map(|c| c.re * inv_n).collect()
}

/// `numpy.fft.rfftfreq(n, d = 1.0 / fs)`: `k·fs/n` for `k = 0..=n/2`.
#[must_use]
pub fn rfftfreq(n: usize, fs: f64) -> Vec<f64> {
    (0..=n / 2).map(|k| (k as f64) * fs / (n as f64)).collect()
}

/// `numpy`'s `np.geomspace(a, b, n)`: exact endpoints, geometrically spaced
/// between them (computed in the log domain the way the caller in `plot.rs`
/// reads them, so the endpoints are never a rounding step off).
#[must_use]
pub fn geomspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![a],
        _ => {
            let la = a.log10();
            let lb = b.log10();
            let mut v: Vec<f64> = (0..n)
                .map(|i| 10f64.powf(la + (lb - la) * (i as f64) / (n - 1) as f64))
                .collect();
            v[0] = a;
            v[n - 1] = b;
            v
        }
    }
}

/// `np.interp(x, xp, fp)` for **ascending** `xp`, with NumPy's clamping at both
/// ends and linear (not log) interpolation in `xp`.
#[must_use]
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    debug_assert_eq!(xp.len(), fp.len());
    debug_assert!(!xp.is_empty());
    if x.is_nan() {
        // NumPy propagates NaN through `np.interp`. Every comparison against NaN is false,
        // so without this the query falls through the end clamps and lands on an interior
        // knot, quietly replacing a missing measurement with a guessed one.
        return f64::NAN;
    }
    if xp.len() == 1 || x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[xp.len() - 1];
    }
    // First index whose knot is strictly greater than x; x lies in [i-1, i].
    let i = match xp.binary_search_by(|p| p.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(k) => return fp[k],
        Err(k) => k,
    };
    let (x0, x1) = (xp[i - 1], xp[i]);
    if x1 == x0 {
        return fp[i];
    }
    fp[i - 1] + (fp[i] - fp[i - 1]) * ((x - x0) / (x1 - x0))
}

/// Slice-aware [`interp`] over a query grid.
#[must_use]
pub fn interp_grid(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| interp(v, xp, fp)).collect()
}

/// Smallest power of two `>= n`.
#[must_use]
pub fn next_pow2(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

/// `int.bit_length()` — the number of bits needed to represent `v`, `0` for zero.
///
/// The convolver's FFT size in `hypreq.py` is written as
/// `1 << (block_len + n_ir - 2).bit_length()`, which is *strictly* greater than
/// its argument when that argument is a power of two. Reproducing the size exactly
/// matters: it decides which FFT plan runs, and therefore the rounding of every
/// convolution sample.
#[must_use]
pub fn bit_length(v: usize) -> usize {
    usize::BITS as usize - v.leading_zeros() as usize
}

/// `1 << v.bit_length()` — the Python expression, verbatim.
#[must_use]
pub fn shl_bit_length(v: usize) -> usize {
    1usize << bit_length(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference DFT written straight from the definition — the oracle for the
    /// wrappers, since the point of this module is the conventions, not speed.
    fn naive_dft(x: &[Complex64], sign: f64) -> Vec<Complex64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut acc = Complex64::new(0.0, 0.0);
                for (m, &xm) in x.iter().enumerate() {
                    let ang = sign * std::f64::consts::TAU * (k * m) as f64 / n as f64;
                    acc += xm * Complex64::from_polar(1.0, ang);
                }
                acc
            })
            .collect()
    }

    fn impulse_response_of_real(x: &[f64]) -> Vec<Complex64> {
        let n = x.len();
        (0..n / 2 + 1)
            .map(|k| {
                let mut acc = Complex64::new(0.0, 0.0);
                for (m, &xm) in x.iter().enumerate() {
                    let ang = -std::f64::consts::TAU * (k * m) as f64 / n as f64;
                    acc += xm * Complex64::from_polar(1.0, ang);
                }
                acc
            })
            .collect()
    }

    #[test]
    fn forward_matches_the_definition() {
        let x: Vec<Complex64> = (0..32)
            .map(|i| Complex64::new((i as f64).cos(), -(i as f64).sin()))
            .collect();
        let got = fft(&x);
        let want = naive_dft(&x, -1.0);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).norm() < 1e-11, "{g} vs {w}");
        }
    }

    #[test]
    fn inverse_carries_the_numpy_one_over_n() {
        let x: Vec<Complex64> = (0..64)
            .map(|i| Complex64::new(i as f64, -(i as f64) / 3.0))
            .collect();
        let back = ifft(&fft(&x));
        for (b, o) in back.iter().zip(&x) {
            assert!((b - o).norm() < 1e-9, "{b} vs {o}");
        }
        // And explicitly: rustfft's raw inverse is n× the NumPy one, which is why
        // [`ifft`] above carries the 1/n. Forward first, then the raw inverse.
        let raw = {
            let mut buf = fft(&x);
            plan(x.len(), true).process(&mut buf);
            buf
        };
        let worst = raw
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - *b * x.len() as f64).norm())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-6, "{worst}");
    }

    #[test]
    fn rfft_matches_a_half_spectrum_dft_and_round_trips() {
        let x: Vec<f64> = (0..257).map(|i| (i as f64 * 0.37).sin() - 0.2).collect();
        let got = rfft(&x, x.len());
        let want = impulse_response_of_real(&x);
        assert_eq!(got.len(), x.len() / 2 + 1);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).norm() < 1e-10, "{g} vs {w}");
        }
        let back = irfft(&got, x.len());
        for (b, o) in back.iter().zip(&x) {
            assert!((b - o).abs() < 1e-11, "{b} vs {o}");
        }
    }

    #[test]
    fn rfft_zero_pads_to_the_requested_length() {
        // np.rfft(x, n) for n > len(x) is the transform of the zero-padded array,
        // which is what synthesize_minimum_phase_ir's caller relies on.
        let x = vec![1.0, 0.5, -0.25];
        let got = rfft(&x, 8);
        assert_eq!(got.len(), 5);
        assert!((got[0].re - 1.25).abs() < 1e-15);
        let want = impulse_response_of_real(&[1.0, 0.5, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0]);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).norm() < 1e-15, "{g} vs {w}");
        }
    }

    #[test]
    fn rfftfreq_grid_includes_the_positive_nyquist_bin() {
        let f = rfftfreq(8, 48000.0);
        assert_eq!(f.len(), 5);
        assert_eq!(f[0], 0.0);
        assert_eq!(f[1], 6000.0);
        assert_eq!(f[4], 24000.0);
    }

    #[test]
    fn interp_clamps_like_numpy_and_hits_knots_exactly() {
        let xp = [1.0, 10.0, 100.0];
        let fp = [-6.0, 0.0, 3.0];
        assert_eq!(interp(0.5, &xp, &fp), -6.0);
        assert_eq!(interp(1e6, &xp, &fp), 3.0);
        assert_eq!(interp(10.0, &xp, &fp), 0.0);
        assert!((interp(5.5, &xp, &fp) - (-6.0 + 6.0 * (4.5 / 9.0))).abs() < 1e-15);
    }

    #[test]
    fn geomspace_hits_the_endpoints_exactly() {
        let g = geomspace(20.0, 20000.0, 1200);
        assert_eq!(g.len(), 1200);
        assert_eq!(g[0], 20.0);
        assert_eq!(g[1199], 20000.0);
        assert!(g.windows(2).all(|w| w[1] > w[0]));
        // np.geomspace(20, 20000, 1200)[600] == 634.2800302
        assert!((g[600] - 634.280_030_2).abs() < 1e-5);
    }

    #[test]
    fn bit_length_reproduces_the_python_fft_size() {
        // The two tables are `1 << v.bit_length()` evaluated in Python.
        let cases: [(usize, usize); 9] = [
            (0, 1),
            (1, 2),
            (2, 4),
            (3, 4),
            (65534, 65536),
            (65536, 131072),
            (81918, 131072),
            (262142, 262144),
            (262143, 262144),
        ];
        for (v, want) in cases {
            assert_eq!(shl_bit_length(v), want, "1 << {v}.bit_length()");
        }
        assert_eq!(bit_length(0), 0);
        assert_eq!(bit_length(262143), 18);
        // And the difference from next_power_of_two that the convolver depends on:
        assert_eq!(next_pow2(65536), 65536);
        assert_eq!(shl_bit_length(65536), 131072);
    }

    #[test]
    fn a_degenerate_transform_size_returns_empty_instead_of_panicking() {
        // numpy raises `Invalid number of FFT data points (0)`; the caller here is a
        // library, and the non-panicking answer to "what is the spectrum of nothing" is
        // nothing. A length-1 transform is legal in numpy and has to stay correct.
        // Behaviour pinned to NumPy, whatever the transform machinery does underneath:
        // `np.fft.rfft(x, 0)` and `irfft(X, 0)` are the sizes it refuses, and length 1 is
        // the size it answers with the sample itself.
        assert!(rfft(&[1.0, 2.0], 0).is_empty());
        assert!(irfft(&[Complex64::new(1.0, 0.0)], 0).is_empty());
        assert_eq!(rfft(&[3.0], 1), vec![Complex64::new(3.0, 0.0)]);
        assert_eq!(rfft(&[], 1), vec![Complex64::new(0.0, 0.0)]);
        assert_eq!(irfft(&[Complex64::new(2.0, 0.0)], 1), vec![2.0]);
        assert_eq!(
            irfft(&[Complex64::new(2.0, 7.0)], 1),
            vec![2.0],
            "imag part is unused"
        );
        // Zero-padding is the same rule at the other end of the size range.
        assert_eq!(rfft(&[1.0], 4).len(), 3);
    }

    #[test]
    fn a_nan_query_stays_nan_through_interp() {
        // np.interp propagates NaN. Comparisons against NaN are all false, so without an
        // explicit rule a missing measurement was answered with an interior knot value.
        let xp = [1.0, 2.0, 3.0];
        let fp = [10.0, 20.0, 30.0];
        assert!(interp(f64::NAN, &xp, &fp).is_nan());
        // The end clamps are what numpy does for the infinities.
        assert_eq!(interp(f64::NEG_INFINITY, &xp, &fp), 10.0);
        assert_eq!(interp(f64::INFINITY, &xp, &fp), 30.0);
        assert_eq!(interp(2.5, &xp, &fp), 25.0);
    }
}
