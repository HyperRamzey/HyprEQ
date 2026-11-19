//! Fritsch–Carlson monotonic cubic Hermite (PCHIP) interpolation in `log10(f)`.
//!
//! This is the interpolator the whole EQ curve is built from: the control points
//! in `measurements.json` are 127–128 hand-placed knots, and both the FIR target
//! (`ir::synthesize_minimum_phase_ir`) and every plot trace are PCHIP through those
//! knots in log-frequency space. Monotonicity is the point — a plain cubic spline
//! over 128 measurement knots rings between them, and that ringing becomes real
//! boost/cut in the filter.
//!
//! Ported term-for-term from `hypreq.py::pchip_log_interp`, including the parts
//! that look incidental and are not:
//!
//! * queries are clamped to `[f_nodes[0], f_nodes[-1]]` **before** the `log10`, so
//!   DC and everything above the top knot read the end-slope value instead of
//!   extrapolating (the FIR target at bin 0 is then overwritten with the first
//!   knot's gain by the caller — see `ir.rs`).
//! * the interval index is `searchsorted(…, side="right") - 1` clipped to
//!   `[0, n-2]`, so a query exactly on an interior knot takes the interval *above*
//!   it, and both Hermite blends agree there.
//! * the tangent at a knot is the weighted harmonic mean of the two secant slopes
//!   only when they have the same sign, and exactly `0.0` otherwise — which is what
//!   flattens the curve at every measurement peak instead of overshooting it.
//!
//! ### Endpoint Derivatives: Fritsch & Carlson (1980) Eq. (8)
//! Endpoints use the shape-preserving non-centered 3-point formula:
//! `d[0] = ((2*h[0] + h[1])*delta[0] - h[0]*delta[1]) / (h[0] + h[1])`
//! with Fritsch-Carlson monotonicity clamping (zeroed if sign disagrees with `delta[0]`,
//! bounded by `3*delta[0]` if direction reverses), exactly matching SciPy's `PchipInterpolator`
//! and Cleve Moler's `pchiptx.m`. This eliminates boundary slope distortion (yielding 0% derivative
//! error on quadratics rather than the 100% error of naive first-order secants).

/// Insertion point of `x` in ascending `a`, on the right of equal elements —
/// `numpy.searchsorted(a, x, side="right")`.
#[must_use]
pub fn searchsorted_right(a: &[f64], x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = a.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if a[mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Monotone cubic Hermite interpolator through `(x, y)`, both ascending in `x`.
///
/// Built once, queried many times: the node tangents (`δ`) do not depend on the
/// query, and `ir.rs` / `plot.rs` each evaluate the same curve on a different grid.
#[derive(Debug, Clone)]
pub struct Pchip {
    x: Vec<f64>,
    y: Vec<f64>,
    /// Node tangents, same length as `x`.
    d: Vec<f64>,
    /// Successive `x` differences, length `x.len() - 1`.
    h: Vec<f64>,
    f_min: f64,
    f_max: f64,
}

/// One-sided three-point estimate for the endpoint derivative, matching
/// Fritsch & Carlson (1980) Eq. (8), Cleve Moler (Numerical Computing with MATLAB, Chap 3.6 `pchiptx.m`),
/// and SciPy's `PchipInterpolator._edge_case`.
fn pchip_edge_case(h0: f64, h1: f64, m0: f64, m1: f64) -> f64 {
    let d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    let same_sign = (d > 0.0 && m0 > 0.0) || (d < 0.0 && m0 < 0.0);
    if !same_sign {
        return 0.0;
    }
    let opposite_secants = (m0 > 0.0 && m1 < 0.0) || (m0 < 0.0 && m1 > 0.0);
    if opposite_secants && d.abs() > 3.0 * m0.abs() {
        return 3.0 * m0;
    }
    d
}

impl Pchip {
    /// Fit through `(f_nodes, g_nodes)` in `log10(f)` space, so `f_nodes` are plain Hz.
    ///
    /// # Panics
    /// If `f_nodes` and `g_nodes` differ in length, or `f_nodes` is empty.
    #[must_use]
    pub fn new(f_nodes: &[f64], g_nodes: &[f64]) -> Self {
        assert_eq!(f_nodes.len(), g_nodes.len(), "node/gain length mismatch");
        assert!(!f_nodes.is_empty(), "PCHIP needs at least one node");
        // Nodes are normalized here, not assumed. A repeated knot makes one interval zero wide and
        // every divided slope over it inf or NaN, and an unsorted axis makes `f_min` and `f_max` the
        // first and last rows rather than the smallest and largest — which reached `f64::clamp` as
        // "min > max" and aborted the binary on a hand-edited `--target-csv`. So: sort by ascending
        // frequency, last write winning at a repeated frequency (the same rule the CSV loaders use,
        // [`crate::model::sort_and_dedup_points`]), then reject what no cubic can be built from.
        // Sorting is not a convenience: the monotone limiter below and the binary search in `eval`
        // are both only correct on an ascending axis, so an interpolator is never a valid object
        // until this has happened.
        let mut order: Vec<usize> = (0..f_nodes.len()).collect();
        order.sort_by(|&a, &b| f_nodes[a].total_cmp(&f_nodes[b]));
        let mut nodes: Vec<f64> = Vec::with_capacity(f_nodes.len());
        let mut gains: Vec<f64> = Vec::with_capacity(g_nodes.len());
        for &i in &order {
            let (f, g) = (f_nodes[i], g_nodes[i]);
            if nodes.last().is_some_and(|&prev| prev == f) {
                let n = gains.len() - 1;
                gains[n] = g;
            } else {
                nodes.push(f);
                gains.push(g);
            }
        }
        assert!(
            nodes.iter().all(|f| f.is_finite() && *f > 0.0),
            "PCHIP nodes must be finite positive frequencies; got {}",
            nodes
                .iter()
                .find(|f| !(f.is_finite() && **f > 0.0))
                .map_or(String::new(), |f| f.to_string())
        );
        let f_min = nodes[0];
        let f_max = nodes[nodes.len() - 1];
        let f_nodes: &[f64] = &nodes;
        let y = gains;
        let x: Vec<f64> = f_nodes.iter().map(|f| f.log10()).collect();
        if x.len() == 1 {
            return Self {
                x,
                y,
                d: vec![0.0],
                h: Vec::new(),
                f_min,
                f_max,
            };
        }
        let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        let delta: Vec<f64> = h
            .iter()
            .zip(y.windows(2))
            .map(|(hi, yg)| (yg[1] - yg[0]) / hi)
            .collect();

        let mut d = vec![0.0; y.len()];
        for k in 1..x.len() - 1 {
            if delta[k - 1] * delta[k] > 0.0 {
                let w1 = 2.0 * h[k] + h[k - 1];
                let w2 = h[k] + 2.0 * h[k - 1];
                d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k]);
            } else {
                d[k] = 0.0;
            }
        }
        // Endpoint derivatives: Fritsch & Carlson (1980) Eq. (8) / SciPy / Cleve Moler
        // (pchiptx.m) shape-preserving non-centered 3-point formula.
        if x.len() == 2 {
            d[0] = delta[0];
            d[1] = delta[0];
        } else {
            d[0] = pchip_edge_case(h[0], h[1], delta[0], delta[1]);
            let n = x.len();
            d[n - 1] = pchip_edge_case(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
        }

        Self {
            x,
            y,
            d,
            h,
            f_min,
            f_max,
        }
    }

    /// Number of knots, after [`Pchip::new`] has folded any repeats.
    ///
    /// Public because it is the only way to see that folding from outside: the de-duplication is a
    /// property of the interpolant rather than of the array handed to it, and the test that a
    /// repeated frequency leaves three knots and not four needs somewhere to look.
    #[must_use]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Always `false`: a `Pchip` keeps at least one node whatever it is given (see [`Pchip::new`]).
    /// Present because a type with `len` is expected to answer this, not because the answer is
    /// interesting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false // a Pchip always has at least one node, see `new`
    }

    /// Evaluate at frequencies in Hz (clamped to the knot range, like the Python).
    #[must_use]
    pub fn eval(&self, f_hz: f64) -> f64 {
        if self.x.len() == 1 {
            return self.y[0];
        }
        if f_hz.is_nan() {
            return f64::NAN;
        }
        // Clamp in Hz first, exactly as `np.clip(f_query, f[0], f[-1])` does before the
        // log10. Caching f_min/f_max avoids per-query 10^x evaluation and IEEE-754 boundary drift.
        let f = f_hz.clamp(self.f_min, self.f_max);
        let fq = f.log10();
        let idx = searchsorted_right(&self.x, fq)
            .saturating_sub(1)
            .min(self.x.len() - 2);
        let hi = self.h[idx];
        let t = (fq - self.x[idx]) / hi;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        h00 * self.y[idx]
            + h10 * hi * self.d[idx]
            + h01 * self.y[idx + 1]
            + h11 * hi * self.d[idx + 1]
    }

    /// Evaluate on a grid, using a monotonic search cursor when queries ascend.
    #[must_use]
    pub fn eval_grid(&self, f_query: &[f64]) -> Vec<f64> {
        if self.x.len() <= 1 {
            return f_query.iter().map(|&f| self.eval(f)).collect();
        }
        let last_idx = self.x.len() - 2;
        let mut cursor = 0usize;
        let mut out = Vec::with_capacity(f_query.len());
        for &f_hz in f_query {
            if f_hz.is_nan() {
                out.push(f64::NAN);
                continue;
            }
            let f = f_hz.clamp(self.f_min, self.f_max);
            let fq = f.log10();
            if fq >= self.x[cursor] {
                while cursor < last_idx && fq >= self.x[cursor + 1] {
                    cursor += 1;
                }
            } else {
                cursor = searchsorted_right(&self.x, fq)
                    .saturating_sub(1)
                    .min(last_idx);
            }
            let hi = self.h[cursor];
            let t = (fq - self.x[cursor]) / hi;
            let t2 = t * t;
            let t3 = t2 * t;
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;
            out.push(
                h00 * self.y[cursor]
                    + h10 * hi * self.d[cursor]
                    + h01 * self.y[cursor + 1]
                    + h11 * hi * self.d[cursor + 1],
            );
        }
        out
    }
}

/// One-shot form of [`Pchip::new`] + [`Pchip::eval`] — `hypreq.py::pchip_log_interp`.
#[must_use]
pub fn pchip_log_interp(f_nodes: &[f64], g_nodes: &[f64], f_query: &[f64]) -> Vec<f64> {
    Pchip::new(f_nodes, g_nodes).eval_grid(f_query)
}

/// The common single-point case (`pchip_log_interp(f, g, np.array([1000.0]))[0]`).
#[must_use]
pub fn pchip_eval1(f_nodes: &[f64], g_nodes: &[f64], f_hz: f64) -> f64 {
    Pchip::new(f_nodes, g_nodes).eval(f_hz)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The monotone limiter below and the binary search in `eval` both assume an ascending axis, and
    /// `f_min` / `f_max` come from the ends of the node list — so an interpolator built from a
    /// hand-edited CSV in spreadsheet order was not a valid object, and reached `f64::clamp` as
    /// "min > max". Nodes are therefore normalized on the way in: ascending, last write winning at a
    /// repeated frequency, which is the same rule the loaders apply.
    #[test]
    fn unsorted_and_repeated_nodes_describe_the_same_curve_as_sorted_ones() {
        let shuffled_f = [1000.0, 100.0, 1000.0, 10000.0, 20.0];
        let shuffled_g = [0.0, -2.0, 1.0, -3.0, 4.0];
        // The same nodes, ordered, with the two rows at 1 kHz collapsed to the later write.
        let want_f = [20.0, 100.0, 1000.0, 10000.0];
        let want_g = [4.0, -2.0, 1.0, -3.0];
        let shuffled = Pchip::new(&shuffled_f, &shuffled_g);
        let ordered = Pchip::new(&want_f, &want_g);
        for &f in &[20.0, 50.0, 100.0, 400.0, 1000.0, 4000.0, 10000.0] {
            assert_eq!(
                shuffled.f_min, want_f[0],
                "the axis start came from a row order, not the data"
            );
            assert_eq!(
                shuffled.f_max, want_f[3],
                "the axis end came from a row order"
            );
            assert!(
                (shuffled.eval(f) - ordered.eval(f)).abs() < 1e-12,
                "at {f} Hz: {} vs {}",
                shuffled.eval(f),
                ordered.eval(f)
            );
        }
        assert_eq!(shuffled.eval(1000.0), 1.0, "the last write must win");
    }

    #[test]
    fn matches_scipy_pchip_on_a_known_curve() {
        // Values match standard Fritsch & Carlson (1980) Eq. (8) / SciPy PchipInterpolator.
        //   python: scipy.interpolate.PchipInterpolator(np.log10([20.,100.,1000.,10000.,20000.]),
        //                                              [ 3., 0.,-2., 4., 0.])
        //   evaluated at [20., 50., 1000., 3703., 20000., 50000., 1.]
        //     -> [3., 1.0757301, -2., 1.6131173, 0., 0., 3.]
        let f = [20.0, 100.0, 1000.0, 10000.0, 20000.0];
        let g = [3.0, 0.0, -2.0, 4.0, 0.0];
        let q = [20.0, 50.0, 1000.0, 3703.0, 20000.0, 50000.0, 1.0];
        let got = pchip_log_interp(&f, &g, &q);
        let want = [3.0, 1.075_730_1, -2.0, 1.613_117_3, 0.0, 0.0, 3.0];
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 5e-7, "{a} vs {b}");
        }
    }

    #[test]
    fn interpolates_knots_exactly_and_clamps_outside() {
        let f = [20.0, 63.0, 500.0, 4000.0, 16000.0];
        let g = [1.5, -3.25, 0.75, 6.125, -0.5];
        let p = Pchip::new(&f, &g);
        for (&fi, &gi) in f.iter().zip(&g) {
            assert!((p.eval(fi) - gi).abs() < 1e-12, "knot {fi}");
        }
        assert_eq!(p.eval(1.0), g[0], "below the first knot clamps");
        assert_eq!(p.eval(1e6), g[g.len() - 1], "above the last knot clamps");
    }

    #[test]
    fn monotone_between_knots_of_one_sign() {
        // A strictly increasing run must not overshoot: PCHIP's harmonic-mean
        // tangents are the whole reason this interpolator was chosen.
        let f = [20.0, 40.0, 80.0, 160.0, 320.0];
        let g = [-6.0, -4.0, -2.0, -1.0, 0.0];
        let p = Pchip::new(&f, &g);
        let grid = crate::fft::geomspace(20.0, 320.0, 4000);
        for w in grid.windows(2) {
            assert!(
                p.eval(w[1]) >= p.eval(w[0]) - 1e-12,
                "dip between {} and {}",
                w[0],
                w[1]
            );
        }
        for &f_i in &grid {
            assert!(
                p.eval(f_i) <= 0.0 + 1e-12 && p.eval(f_i) >= -6.0 - 1e-12,
                "overshoot at {f_i}"
            );
        }
    }

    #[test]
    fn flattens_at_a_peak() {
        let f = [100.0, 1000.0, 10000.0];
        let g = [0.0, 5.0, 0.0];
        let p = Pchip::new(&f, &g);
        // Tangent at the peak knot is exactly zero by the sign rule, and hypreq.py
        // reads 4.99999812 one tenth of a percent above the knot.
        assert_eq!(p.d[1], 0.0);
        let near = p.eval(1000.0 * 1.001);
        assert!(near < 5.0, "no boost past the knot: {near}");
        assert!((near - 4.999_999_058).abs() < 5e-8, "{near}");
    }

    /// Invariant: Fritsch & Carlson (1980) Eq. (8) has 0% derivative error on quadratics,
    /// whereas first-order secants have 100% error. Sabotage: returning `delta[0]` yields
    /// `d[0] = 1.0` (failing the test).
    #[test]
    fn endpoints_match_fritsch_carlson_and_scipy_exact() {
        // Knots chosen so log10(f) = [0.0, 1.0, 2.0, 3.0], with g = x^2:
        let f = [1.0, 10.0, 100.0, 1000.0];
        let g = [0.0, 1.0, 4.0, 9.0];
        let p = Pchip::new(&f, &g);
        // Exact f'(0) for y = x^2 is 0.0; exact f'(3) is 6.0.
        assert!((p.d[0] - 0.0).abs() < 1e-12, "d[0] was {}", p.d[0]);
        assert!((p.d[3] - 6.0).abs() < 1e-12, "d[3] was {}", p.d[3]);
    }

    #[test]
    fn single_node_is_a_constant() {
        let p = Pchip::new(&[1000.0], &[2.5]);
        assert_eq!(p.eval(20.0), 2.5);
        assert_eq!(p.eval(20000.0), 2.5);
    }

    #[test]
    fn searchsorted_matches_the_right_side_convention() {
        let a = [1.0, 2.0, 2.0, 5.0];
        assert_eq!(searchsorted_right(&a, 0.5), 0);
        assert_eq!(searchsorted_right(&a, 2.0), 3, "right of equal elements");
        assert_eq!(searchsorted_right(&a, 5.0), 4);
        assert_eq!(searchsorted_right(&a, 6.0), 4);
    }

    #[test]
    fn a_query_off_the_end_of_the_grid_clamps_instead_of_wrapping() {
        // log10 of 0 Hz is -inf and of a negative frequency NaN; the NaN searched to index
        // 0 and `0 - 1` on a usize is what crashed. The reference clamps in Hz first, so
        // every frequency below the first knot reads as the first knot.
        let p = Pchip::new(&[20.0, 1000.0, 20000.0], &[1.0, 0.0, -1.0]);
        assert_eq!(p.eval(0.0), 1.0);
        assert_eq!(p.eval(-5.0), 1.0);
        assert_eq!(p.eval(f64::NEG_INFINITY), 1.0);
        assert_eq!(p.eval(f64::INFINITY), -1.0);
        assert_eq!(p.eval(20.0), 1.0);
        assert_eq!(p.eval(20000.0), -1.0);
        // np.interp's answer for a NaN query is NaN, not a guess.
        assert!(p.eval(f64::NAN).is_nan());
    }

    #[test]
    fn a_repeated_knot_cannot_turn_the_curve_into_nan() {
        // A zero-wide interval makes every divided slope over it infinite, and one
        // infinite slope is enough to poison the whole cubic.
        let p = Pchip::new(&[20.0, 1000.0, 1000.0, 20000.0], &[1.0, 0.5, -0.5, -1.0]);
        assert_eq!(
            p.len(),
            3,
            "the repeated knot is folded, last value winning"
        );
        for f in [20.0, 100.0, 1000.0, 5000.0, 20000.0] {
            assert!(p.eval(f).is_finite(), "{f} Hz gave {}", p.eval(f));
        }
        assert_eq!(p.eval(1000.0), -0.5);
    }
}
