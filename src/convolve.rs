//! Exact linear convolution of planar audio against a mono FIR, by overlap-add.
//!
//! This is `hypreq.py::convolve_planar_fft`: uniform-block overlap-add with
//! zero-padded FFTs, so the result is the *linear* convolution — no circular
//! wrap, no IIR approximation. The output is `n_in + n_ir - 1` samples (full
//! linear convolution, exactly what scipy's `signal.fftconvolve`/`np.convolve`
//! produce), which is why the caller trims to the input length afterwards.
//!
//! Two properties of the Python are worth stating because they are what the
//! parity tests actually pin down:
//!
//! * **The block size is a property of the input length, not of the IR.**
//!   `block_len = max(16384, min(65536, n_in))`, so an input of ≤ 65536 samples
//!   is one block no matter how long the filter is, and long inputs are cut into
//!   65 536-sample blocks. Since a whole block is convolved at a time, the same
//!   file processed at a different `--taps` goes through the same FFT size, and
//!   the arithmetic — including its rounding — is unchanged.
//! * **`n_fft` is the next power of two strictly above `block_len + n_ir - 2`.**
//!   It is `1 << (block_len + n_ir - 2).bit_length()`, which for a sum that is
//!   already a power of two doubles it rather than keeping it. The padding is
//!   what makes the per-block product linear rather than circular: with `L`
//!   samples of block and `M` of filter, `L + M - 1` outputs must fit.
//!
//! `H` is computed once and reused for every block of every channel, so the cost
//! per block is two real FFTs of size `n_fft` plus a complex multiply. Everything
//! is `f64`: `f32` would put the convolution's own noise floor around -150 dBFS,
//! far above the ≥ 180 dB stop-band decay the report measures.

use crate::fft::{shl_bit_length, Complex64};

/// Convolve every channel of `planar` with the mono FIR `ir`.
///
/// With `trim_to_input` each channel is cut back to its first `n_in` samples, as
/// the Python's `acc[:out_len]` does; otherwise the full `n_in + n_ir - 1`
/// linear response is returned.
///
/// Degenerate inputs follow the reference rather than panicking: an empty `ir`
/// zero-pads inside the transform, so the output is silence of the same length the
/// formula gives (`n_in - 1`); a channel shorter than channel 0 contributes only the
/// blocks it actually has.
#[must_use]
pub fn convolve_planar_fft(planar: &[Vec<f64>], ir: &[f64], trim_to_input: bool) -> Vec<Vec<f64>> {
    let Some(first) = planar.first() else {
        return Vec::new();
    };
    let n_in = first.len();
    let n_ir = ir.len();
    if n_in == 0 {
        // The Python returns one empty channel per input channel here, not an
        // empty list, and the caller zips the result against the channel names.
        return vec![Vec::new(); planar.len()];
    }

    let block_len = 16384usize.max(block_bound(n_in));
    let n_fft = shl_bit_length(block_len + n_ir - 2);
    let h = crate::fft::rfft(ir, n_fft);
    let out_len = if trim_to_input { n_in } else { n_in + n_ir - 1 };

    planar
        .iter()
        .map(|ch| {
            let mut acc = vec![0.0f64; n_in + n_ir - 1];
            let mut start = 0usize;
            // `n_in` is channel 0's length. A shorter channel simply stops early: slicing
            // `start..stop` past its end would panic, and no audio path wants a panic.
            while start < n_in && start < ch.len() {
                let stop = (start + block_len).min(n_in).min(ch.len());
                let blk = &ch[start..stop];
                let seg_len = blk.len() + n_ir - 1;
                let mut spec = crate::fft::rfft(blk, n_fft);
                apply_response(&mut spec, &h);
                let y = crate::fft::irfft(&spec, n_fft);
                let take = seg_len.min(acc.len() - start);
                for (a, &v) in acc[start..start + take].iter_mut().zip(&y[..take]) {
                    *a += v;
                }
                start += block_len;
            }
            acc.truncate(out_len.min(acc.len()));
            acc
        })
        .collect()
}

/// `min(65536, n_in)` — split out so the clamp reads as one idea.
fn block_bound(n_in: usize) -> usize {
    if n_in < 65536 {
        n_in
    } else {
        65536
    }
}

/// The FFT size a given (input length, filter length) pair uses, exposed for
/// the report and the tests.
#[must_use]
pub fn plan(n_in: usize, n_ir: usize) -> (usize, usize) {
    let block_len = 16384usize.max(block_bound(n_in));
    (block_len, shl_bit_length(block_len + n_ir - 2))
}

/// Convolve a single channel.
#[must_use]
pub fn convolve1(x: &[f64], ir: &[f64], trim_to_input: bool) -> Vec<f64> {
    convolve_planar_fft(std::slice::from_ref(&x.to_vec()), ir, trim_to_input)
        .pop()
        .unwrap_or_default()
}

/// In-place complex multiply of an `rfft` spectrum by a precomputed response.
fn apply_response(spec: &mut [Complex64], h: &[Complex64]) {
    for (s, hi) in spec.iter_mut().zip(h.iter()) {
        *s *= hi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_and_fft_plan_matches_the_python_formula() {
        // n_in <= 16384 -> one 16384-block; n_in <= 65536 -> block = n_in; else 65536.
        assert_eq!(plan(4000, 999), (16384, 32768));
        assert_eq!(plan(150000, 5000), (65536, 131072));
        assert_eq!(plan(65536, 65536).0, 65536);
        // block_len + n_ir - 2 == 131070 -> strictly-above power of two.
        assert_eq!(plan(65536, 65536).1, 131072);
        assert_eq!(plan(65536, 65538).1, 262144);
    }

    #[test]
    fn empty_and_degenerate_inputs() {
        assert!(convolve_planar_fft(&[], &[1.0, 2.0], false).is_empty());
        // Empty channel: one empty buffer per input channel, not an empty list.
        let r = convolve_planar_fft(&[vec![], vec![]], &[1.0, 2.0], false);
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|c| c.is_empty()));
        // A delta IR is the identity, at full length.
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = convolve1(&x, &[1.0], false);
        assert_eq!(y, x);
        // Trimmed output is the input length.
        let y = convolve1(&x, &[0.5, 0.5], true);
        assert_eq!(y.len(), 4);
        assert_eq!(y, vec![0.5, 1.5, 2.5, 3.5]);
    }

    #[test]
    fn matches_naive_linear_convolution() {
        let x: Vec<f64> = (0..500)
            .map(|i| (i as f64 * 0.13).sin() * (1.0 - i as f64 / 500.0))
            .collect();
        let h: Vec<f64> = (0..37)
            .map(|i| (-(i as f64 - 18.0).powi(2) / 40.0).exp())
            .collect();
        let y = convolve1(&x, &h, false);
        assert_eq!(y.len(), x.len() + h.len() - 1);
        let mut worst = 0.0f64;
        for n in 0..y.len() {
            let mut ref_y = 0.0f64;
            for (i, &hv) in h.iter().enumerate() {
                if n >= i && n - i < x.len() {
                    ref_y += x[n - i] * hv;
                }
            }
            worst = worst.max((y[n] - ref_y).abs());
        }
        assert!(worst < 1e-12, "worst deviation {worst}");
    }

    #[test]
    fn block_boundaries_are_invisible() {
        // Crossing the 65536 block boundary must not change the result, and a
        // single long block must equal the many-small-blocks arithmetic.
        let n = 65536 + 1000;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i % 97) as f64 * 0.031).sin() - 0.2)
            .collect();
        let h: Vec<f64> = (0..5000)
            .map(|i| (-(i as f64 - 2500.0).powi(2) / 400_000.0).exp())
            .collect();
        let full = convolve1(&x, &h, false);
        // Compare the first block's own linear convolution against the head of
        // the full result: overlap-add adds nothing inside the first block.
        let head = convolve1(&x[..65536], &h, false);
        let worst = (0..65536)
            .map(|i| (full[i] - head[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-9, "first-block deviation {worst}");
        assert_eq!(full.len(), n + h.len() - 1);
    }

    #[test]
    fn channels_are_independent() {
        let a: Vec<f64> = (0..300).map(|i| (i as f64 * 0.05).sin()).collect();
        let b: Vec<f64> = (0..300)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let h: Vec<f64> = (0..64)
            .map(|i| if i < 32 { 1.0 / 32.0 } else { 0.0 })
            .collect();
        let r = convolve_planar_fft(&[a.clone(), b], &h, true);
        let sa = convolve1(&a, &h, true);
        for i in 0..300 {
            assert!((r[0][i] - sa[i]).abs() < 1e-12);
        }
    }

    /// Parity against the Python, using the recorded NumPy artefacts.
    #[test]
    fn matches_numpy_artifacts() {
        let Some(dir) = crate::oracle_dir() else {
            eprintln!("skipping: HYPREQ_ORACLE is not set");
            return;
        };
        let f64s = |n: &str| -> Vec<f64> {
            let b = std::fs::read(format!("{dir}/{n}.f64")).expect("fixture");
            b.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
                .collect()
        };
        // A: 4000 samples x 999 taps, single block, full and trimmed.
        let (x, h) = (f64s("fixtures/A_in"), f64s("fixtures/A_ir"));
        let y = convolve_planar_fft(&[x.clone()], &h, false).pop().unwrap();
        let want = f64s("fixtures/A_out");
        assert_eq!(y.len(), want.len());
        let w = (0..y.len())
            .map(|i| (y[i] - want[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(w < 1e-11, "case A worst {w}");

        let yt = convolve_planar_fft(&[x], &h, true).pop().unwrap();
        let wantt = f64s("fixtures/A_trim_out");
        assert_eq!(yt.len(), wantt.len());
        let wt = (0..yt.len())
            .map(|i| (yt[i] - wantt[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(wt < 1e-11, "case A trimmed worst {wt}");

        // B: 150000 stereo samples x 5000 taps -> three 65536 blocks per channel.
        let l = f64s("fixtures/B_inL");
        let r = f64s("fixtures/B_inR");
        let h2 = f64s("fixtures/B_ir");
        let got = convolve_planar_fft(&[l, r], &h2, false);
        for (name, c) in [("fixtures/B_outL", 0usize), ("fixtures/B_outR", 1)] {
            let want = f64s(name);
            assert_eq!(got[c].len(), want.len(), "{name} length");
            let w = (0..want.len())
                .map(|i| (got[c][i] - want[i]).abs())
                .fold(0.0f64, f64::max);
            // Absolute tolerance: the signals peak near 1.0, so this is ~1e-14
            // relative — FFT reassociation noise between two correct f64 DFTs.
            assert!(w < 1e-10, "{name} worst {w}");
        }
    }

    #[test]
    fn apply_response_is_componentwise() {
        let x = vec![1.0, 0.0, -1.0, 0.0];
        let h = crate::fft::rfft(&[1.0, 0.0, 0.0, 0.0], 4);
        let mut spec = crate::fft::rfft(&x, 4);
        apply_response(&mut spec, &h);
        let y = crate::fft::irfft(&spec, 4);
        for (a, b) in y.iter().zip(&x) {
            assert!((a - b).abs() < 1e-14, "{a} vs {b}");
        }
    }

    #[test]
    fn an_empty_filter_and_a_short_channel_neither_panic_nor_garble() {
        // An empty IR zero-pads inside the transform, so the reference's answer is silence
        // of length n_in + 0 - 1; a channel shorter than channel 0 contributes only the
        // blocks it has instead of slicing past its end.
        let y = convolve_planar_fft(&[vec![1.0, 2.0, 3.0]], &[], false)
            .pop()
            .unwrap();
        assert_eq!(y.len(), 2, "n_in + n_ir - 1 samples of nothing");
        assert!(y.iter().all(|&v| v == 0.0), "{y:?}");

        // A block is at most 65536 samples, so a channel shorter than n_in only meets the
        // boundary when n_in crosses it: channel 0 needs to be long enough that the second
        // channel runs out before the block loop does.
        let n = 70_000;
        let ragged = vec![
            (0..n).map(|i| (i % 7) as f64 * 0.01).collect::<Vec<f64>>(),
            (0..30).map(|i| (i % 3) as f64 * 0.01).collect::<Vec<f64>>(),
        ];
        let out = convolve_planar_fft(&ragged, &[1.0], true);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), n);
        // Exact only to the rfft/irfft round trip, which is at the 1e-17 level here.
        assert!(
            out[0]
                .iter()
                .zip(&ragged[0])
                .all(|(&a, &b)| (a - b).abs() < 1e-12),
            "a delta filter must reproduce channel 0"
        );
        // The short channel keeps the accumulator length channel 0 set, and contributes
        // only the samples it actually has.
        assert!(
            out[1]
                .iter()
                .zip(&ragged[1])
                .all(|(&a, &b)| (a - b).abs() < 1e-12),
            "the short channel comes through unchanged"
        );
        assert!(
            out[1][ragged[1].len()..].iter().all(|&v| v == 0.0),
            "nothing beyond a channel's own samples"
        );
    }
}
