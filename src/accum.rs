//! Summation that holds up at the precision the reports print.
//!
//! The reference sums with NumPy, which accumulates a `float64` reduction in wider
//! storage than the array's dtype and combines the array in blocks. A plain Rust
//! `Iterator::sum` walks the array left to right in `f64`, and at the sizes this tool
//! works with — a 262 144-bin spectrum, a 65 536-tap filter — the two answers differ by
//! about 6e-15 relative.
//!
//! That is invisible until a number is printed with twelve significant digits. It is not
//! invisible then: the causal-energy line of the report read `99.9999999999%` from the
//! reference and `100.0000000000%` from a sequential sum, on a filter whose real energy
//! loss is 1 part in 10^11. Both numbers describe the same filter; only one of them is
//! the correctly rounded value of the sum.
//!
//! Neumaier's compensated sum recovers it: the running compensation collects exactly the
//! low-order bits each addition drops, and the final `sum + comp` lands on the correctly
//! rounded result for the inputs here. That is at least as accurate as the reference's
//! wider accumulator — the two can only disagree in the last bit, and only when the true
//! value sits within 2^-64 of a rounding boundary.

/// One running Neumaier sum: `add` a term, `total` reads the compensated result.
///
/// This is the same recurrence `sum_of_squares` uses, exposed so a *prefix* sum can be built
/// with it. A prefix sum is where the loss actually bites: `prefix[i+1] = prefix[i] + x[i]`
/// walks the same left-to-right path as a plain sum, so every later entry inherits the bits the
/// earlier additions dropped, and a window total `prefix[hi] - prefix[lo]` can come out as
/// exactly zero for a window of values that are all nonzero.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunningSum {
    sum: f64,
    comp: f64,
}

impl RunningSum {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one term. A non-finite term gives up on compensation: `inf` has to propagate as `inf`
    /// and `NaN` as `NaN` rather than turn into a compensated approximation of either.
    pub fn add(&mut self, v: f64) {
        if !v.is_finite() || !self.sum.is_finite() {
            self.sum += v;
            return;
        }
        let next = self.sum + v;
        if !next.is_finite() {
            self.sum = next;
            self.comp = 0.0;
            return;
        }
        self.comp += if self.sum.abs() >= v.abs() {
            (self.sum - next) + v
        } else {
            (v - next) + self.sum
        };
        self.sum = next;
    }

    #[must_use]
    pub fn total(&self) -> f64 {
        self.sum + self.comp
    }
}

/// Prefix sums, compensated: `out[0] = 0.0` and `out[i] = Σ x[0..i]`.
#[must_use]
pub fn prefix_sums(x: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len() + 1);
    let mut acc = RunningSum::new();
    out.push(0.0f64);
    for &v in x {
        acc.add(v);
        out.push(acc.total());
    }
    out
}

/// `Σ x[i]²`, compensated.
///
/// Written for the shape the reports need — energy over a whole filter or spectrum — so
/// the squares are never materialised: a 262 144-element temporary would cost more than
/// the sum itself.
#[must_use]
pub fn sum_of_squares(x: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for &v in x {
        let t = v * v;
        // A non-finite square has no compensation to speak of — `(inf - inf) + inf` is
        // NaN, which would report an energy of NaN for a filter that simply overflows.
        // NumPy's sum returns the inf (or NaN) itself, so propagate it the plain way.
        if !t.is_finite() {
            return x.iter().map(|&w| w * w).fold(0.0f64, |acc, w| acc + w);
        }
        let next = sum + t;
        if !next.is_finite() {
            return x.iter().map(|&w| w * w).fold(0.0f64, |acc, w| acc + w);
        }
        // Whichever operand is larger keeps its high bits; the other one's contribution
        // is what the subtraction below recovers exactly.
        comp += if sum.abs() >= t.abs() {
            (sum - next) + t
        } else {
            (t - next) + sum
        };
        sum = next;
    }
    sum + comp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero() {
        assert_eq!(sum_of_squares(&[]), 0.0);
        assert_eq!(sum_of_squares(&[0.0, -0.0]), 0.0);
    }

    #[test]
    fn recovers_what_a_left_to_right_walk_drops() {
        // (1e8)² is 1e16, whose ulp is 2: adding one more 1.0 to a running total that has
        // reached 1e16 changes nothing at all, so a plain walk reports 1e16 for 2000
        // samples' worth of extra energy. 1e16 + 2000 *is* representable; the sum just
        // has to keep the bits instead of throwing them away each step.
        let mut v = vec![1e8];
        v.extend(std::iter::repeat_n(1.0f64, 2000));
        let naive: f64 = v.iter().map(|x| x * x).sum();
        assert_eq!(naive, 1e16, "the plain walk must be the broken one here");
        assert_eq!(sum_of_squares(&v), 1e16 + 2000.0);
    }

    #[test]
    fn order_does_not_move_it() {
        let mut v: Vec<f64> = (0..5000).map(|i| 1.0 / (i + 1) as f64).collect();
        let a = sum_of_squares(&v);
        v.reverse();
        let b = sum_of_squares(&v);
        assert_eq!(
            a, b,
            "a compensated sum is order-insensitive to the last bit"
        );
    }

    #[test]
    fn overflow_and_nan_propagate_instead_of_becoming_compensated_noise() {
        assert!(sum_of_squares(&[1.0, 1e200]).is_infinite());
        assert!(sum_of_squares(&[f64::NAN, 1.0]).is_nan());
        // Inf first and inf later both have to read the same way round.
        assert!(sum_of_squares(&[1e200, 1.0]).is_infinite());
        // Addition overflow of finite squares must yield +inf, never NaN:
        assert_eq!(sum_of_squares(&[1e154, 1e154]), f64::INFINITY);
        let mut r = RunningSum::new();
        r.add(1e308);
        r.add(1e308);
        assert_eq!(r.total(), f64::INFINITY);
    }

    #[test]
    fn a_window_of_small_values_survives_a_prefix_that_reached_the_sky() {
        // The shape that breaks a plain prefix sum: one enormous term, then a long run of small
        // ones. Once the running total has reached 1e16, adding 1.0 to it changes nothing, so
        // every later prefix entry equals the one before it and a window of a thousand 1.0s
        // reads a total of exactly 0.0 — a mean of 0 for values that are all 1.
        let mut v = vec![1e16];
        v.extend(std::iter::repeat_n(1.0f64, 1000));
        let plain: Vec<f64> = std::iter::once(0.0)
            .chain(v.iter().scan(0.0f64, |acc, &x| {
                *acc += x;
                Some(*acc)
            }))
            .collect();
        assert_eq!(
            plain[1001] - plain[1],
            0.0,
            "the uncompensated prefix must be the broken one here"
        );
        let pre = prefix_sums(&v);
        assert_eq!(pre[1001] - pre[1], 1000.0);
        assert_eq!(
            pre[501] - pre[1],
            500.0,
            "any window, not just the last one"
        );
    }

    #[test]
    fn a_tiny_tail_stays_visible_against_a_big_filter() {
        // The report's actual failure shape: energy over a long filter where the last
        // taps carry ~1e-11 of the total. Rounded away, the ratio prints as exactly 100.
        let mut full = vec![0.0f64; 262_144];
        for (i, s) in full.iter_mut().enumerate().take(65_536) {
            *s = ((i as f64) * 0.37).sin() * 1e-3;
        }
        for s in full.iter_mut().skip(65_536) {
            *s = 1.3e-9;
        }
        let head = &full[..65_536];
        let naive: f64 = head.iter().map(|v| v * v).sum::<f64>()
            / full.iter().map(|v| v * v).sum::<f64>()
            * 100.0;
        assert_eq!(
            naive, 100.0,
            "the plain walk must round the tail away: {naive:.10}"
        );
        let ratio = sum_of_squares(head) / sum_of_squares(&full) * 100.0;
        assert!(
            (99.9999999980..100.0).contains(&ratio),
            "the tail is 1 part in 10^10 of the energy: {ratio:.10}"
        );
    }
}
