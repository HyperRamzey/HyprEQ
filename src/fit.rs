//! Fit-aware regularization: how much of a correction to believe, band by band.
//!
//! The engine's whole correction is `target − raw`, evaluated at every measurement point.
//! That is the textbook inversion, and it has one structural flaw: it trusts every wiggle in
//! the raw curve equally. A broad shelf is the room-and-ear system telling you something real
//! and worth fixing. A 3 dB notch one twelfth of an octave wide is the measurement — a coupler
//! resonance, a standing wave in the pinna, the fixture — and a filter that "fixes" it puts a
//! corresponding spike into the headphone's response and does nothing for the sound.
//!
//! Regularization is the fix, and the *form* of it matters more than the amount. The
//! conventional published form (Norcross, Bouchard & Soulodre, AES 121st Convention Paper
//! 6929, and what `pyfar.dsp.RegularizedSpectrumInversion` implements) penalises the filter's
//! own magnitude:
//!
//! ```text
//! G = S*·D / (S*S + β|ε|²)
//! ```
//!
//! Penalising `‖G‖²` is wrong here, and visibly so: in a band the inversion cannot win, the
//! answer it converges to is *stop trying*, which in dB is −∞. What that band needed was
//! *leave it alone*, which is 0 dB. Turn the knob up on the conventional form and the high end
//! droops — that droop is the penalty, not the headphone. See `docs/ROADMAP.md` §3.6.
//!
//! This module therefore penalises the deviation from an all-pass reference instead:
//!
//! ```text
//! minimise ‖S·G − D‖² + β·‖G − A‖²   ⇒   G = (S*·D + β·A) / (|S|² + β)
//! ```
//!
//! with `A = 1`. Two properties follow, and they are the tests in this file rather than
//! aspirations: `β → 0` reproduces the plain inversion exactly (`G = D/S`, today's curve), and
//! `β → ∞` gives `|G| → 1`, a bypass. The load `β` is frequency-dependent and referred to the
//! *unity* level — it is added to `|S|²`, so it is in absolute power units and one value means
//! the same thing on a headphone with a 2 dB null and one with a 20 dB null. Scaling it by
//! `max|S|` would not, which is `docs/ROADMAP.md` §3.6's other warning.
//!
//! `β(f)` comes from the curve's own deviation from its octave-smoothed version: a band that
//! sits where a smoothed curve says it should is the system, and gets `β ≈ 0`; a band that
//! sticks out is the measurement, and gets a `β` large enough to walk it back to unity.
//!
//! Everything here is `f64`, pure, and infallible. Nothing in this module is reached when
//! `--regularize` is left at its default, so the default output stays byte-identical to the
//! reference program.


/// The level of a raw measurement, which is what its magnitude has to be reduced by before the
/// curve can stand for `S`.
///
/// The load `β` is defined against a *normalized* transfer function — one that reads about 0 dB
/// in the mid-band — because `β` is compared with `|S|²`. A curve that arrives in dB SPL, or one
/// that was never normalized because its knots were hand-placed and `normalize_1khz` was off,
/// carries an arbitrary offset `C`, and feeding it in as-is is not a cosmetic slip: `|S|²` scales
/// by `10^(C/10)`, which is the same thing as dividing the load by that factor. A 90 dB offset
/// with [`KNEE_PER_DB`] at 8 moves the knee by `90 / 8` ≈ 11 dB, so a measurement recorded at a
/// different calibration is silently a different policy — `--regularize` would do nothing at all
/// on a dB-SPL curve and everything on the same curve written in dBFS.
///
/// The statistic is the **median** of the curve, deliberately not its value at 1 kHz. A reference
/// read at one frequency is only a reference if nothing is happening at that frequency, and the
/// curves here are full of narrow features: a measurement with a 12 dB coupler null at 1 kHz would
/// be referenced 12 dB low, which raises every `|S|²` by that much and switches the load off over
/// the whole band — the same failure the offset was. The median is the level the curve actually
/// spends its time at, and it moves by exactly the offset it is meant to remove.
#[must_use]
pub fn reference_db(raw: &[(f64, f64)]) -> f64 {
    median_db(&raw.iter().map(|&(_, y)| y).collect::<Vec<_>>())
}

/// The median of a list of levels, ignoring the ones that are not numbers.
///
/// The same rule [`reference_db`] documents, exposed for callers that have a plain curve rather
/// than a `(frequency, gain)` table: the median of the left ear's gains and the median of the right
/// ear's gains are the pair's balance, and they are the same statistic for the same reason.
pub fn median_db(g: &[f64]) -> f64 {
    let mut v: Vec<f64> = g.iter().copied().filter(|y| y.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) / 2.0
    }
}

/// Octave width of the window that decides what "smooth" means for the fit residual.
///
/// Half an octave, which is the choice Gómez-Bolaños (2016) makes for this estimate: wide
/// enough that a shelf or a broad rise is entirely inside its own window and so contributes
/// almost no residual, narrow enough that a notch a twelfth of an octave wide is nothing but
/// residual.
pub const RESIDUAL_SMOOTH_OCTAVE: f64 = 0.5;

/// Octave width the residual itself is smoothed by before it becomes `β`.
///
/// One sixth of an octave: narrower than the residual window, so it does not widen the region
/// that gets regularized, but enough that `β` varies bin to bin smoothly instead of flickering
/// — a `β` that jumps between "full correction" and "bypass" between adjacent knots puts a step
/// into the correction curve, and a step in the magnitude is a ridge in the impulse response.
pub const BETA_SMOOTH_OCTAVE: f64 = 1.0 / 6.0;

/// How fast `β` grows per dB of residual past the knee.
///
/// Eight. This looks arbitrary and is not: `β` is added to `|S|²`, so to be *invisible* on a
/// trusted band it has to be three or four orders of magnitude below it, and to be *decisive*
/// on a wild band it has to be above it. Traversing that range across the handful of dB
/// between "measurement texture" and "that is a notch" needs about one order of magnitude of
/// `β` per 1.25 dB of residual. Anything gentler either bites into real corrections or fails to
/// bypass the notch. Measured, with `λ = 5`: a 1 dB residual leaves a 6 dB dip at `+5.99 dB`,
/// a 2 dB residual at `+5.93 dB`, a 7 dB residual at `+0.05 dB`.
pub const KNEE_PER_DB: f64 = 8.0;

/// Lowest `λ` the automatic estimate will use, in dB of deviation.
///
/// Three, which is where the shipped measurements put it. Measured on the three curves this
/// project ships, the smoothed deviation of the raw response has a median of 0.06 dB (FT1 Pro),
/// 0.32 dB (JBL 770NC) and 0.34 dB (DT 770 Pro) — ordinary texture — while its worst bands are
/// 3.6 to 8.4 dB and sit at 3.7 kHz and between 8 and 17 kHz: the narrow HF pinna and coupler
/// notches that are the textbook case for not being chased. A knee at 3 dB therefore leaves a
/// 6 dB correction at `+5.87 dB` where the curve is ordinary and takes it to `+0.64 dB` or
/// below at those notches. A knee an octave higher (5 dB) leaves the JBL's worst band almost
/// fully corrected, which is the behaviour this feature exists to prevent.
pub const LAMBDA_FLOOR_DB: f64 = 3.0;

/// Quantile of the deviation that the automatic `λ` is read from.
///
/// The top decile, not the median. The deviation of a real curve is near zero almost
/// everywhere with a sparse tail, so a central statistic (a median, or a MAD around it) reads
/// about 0 on every curve and the estimate never engages — measured: MAD is 0.00 on a flat
/// curve *and* on one with a 12 dB notch in it. Reading the tail is also what makes the
/// estimate do the right thing about the features themselves: a lone narrow notch barely moves
/// the 90th percentile, so it cannot raise the knee and negotiate its way out of being
/// bypassed.
pub const AUTO_TAIL_QUANTILE: f64 = 0.90;

/// How far above that quantile the automatic `λ` sits.
///
/// Thirty percent, which is what leaves ordinary curves at the floor and lifts only the ones
/// whose texture is genuinely broad. Measured on synthetic curves at this grid: a ±3 dB ripple
/// every twelfth of an octave gives a 90th percentile of 2.02 dB (knee stays at the floor), a
/// ±9 dB ripple gives 6.07 dB (knee lifts to 7.89 dB, so a measurement that is twitchy
/// everywhere is not bypassed everywhere).
pub const AUTO_TAIL_FACTOR: f64 = 1.3;

/// A regularization policy: the knee's position, its steepness, and the two window widths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Policy {
    /// Residual level, in dB, at which a band is half-trusted. `λ` above the knee is
    /// progressively bypassed; `λ` below it is corrected in full.
    pub lambda_db: f64,
    /// Order of magnitude `β` gains per dB of residual. See [`KNEE_PER_DB`].
    pub kappa: f64,
    /// Window that defines the residual. See [`RESIDUAL_SMOOTH_OCTAVE`].
    pub residual_octave: f64,
    /// Window the residual is smoothed by before becoming `β`. See [`BETA_SMOOTH_OCTAVE`].
    pub beta_octave: f64,
}

impl Default for Policy {
    fn default() -> Self {
        Policy {
            lambda_db: LAMBDA_FLOOR_DB,
            kappa: KNEE_PER_DB,
            residual_octave: RESIDUAL_SMOOTH_OCTAVE,
            beta_octave: BETA_SMOOTH_OCTAVE,
        }
    }
}

impl Policy {
    /// A policy with `λ` set by hand, everything else at its default.
    ///
    /// `λ` is in dB of residual and is trusted as given: a huge value disables the regularizer
    /// (no band is ever that far past the knee) and `f64::INFINITY` disables it exactly.
    #[must_use]
    pub fn fixed(lambda_db: f64) -> Self {
        Policy {
            lambda_db,
            ..Policy::default()
        }
    }

    /// Estimate `λ` from the curve itself: `max(floor, AUTO_TAIL_FACTOR · q90(deviation))`.
    ///
    /// The floor is what a well-behaved curve gets (all three shipped measurements do); the
    /// estimate only ever *raises* `λ`, on a curve whose texture is broad enough that the floor
    /// would walk back most of the correction. See [`AUTO_TAIL_QUANTILE`] for why it is a tail
    /// statistic and not a median.
    #[must_use]
    pub fn auto(raw_f: &[f64], raw_g: &[f64]) -> Self {
        let s = deviation(raw_f, raw_g, RESIDUAL_SMOOTH_OCTAVE, BETA_SMOOTH_OCTAVE);
        let tail = quantile(&s, AUTO_TAIL_QUANTILE);
        let knee = if tail.is_finite() {
            AUTO_TAIL_FACTOR * tail
        } else {
            0.0
        };
        Policy {
            lambda_db: knee.max(LAMBDA_FLOOR_DB),
            ..Policy::default()
        }
    }
}

/// Fractional-octave moving average of `g` over the grid `f`, in `O(N)`.
///
/// The window at bin `i` is `[f_i · 2^(-fraction/2), f_i · 2^(+fraction/2)]` and the mean is
/// taken over the samples that are actually inside it, divided by *their* count — not by the
/// nominal window width. That is what keeps the first and last bins honest: dividing an
/// edge window by the full width pulls the curve's ends toward zero, which on a correction
/// curve means a fabricated tilt at exactly the two places a headphone measurement is least
/// trustworthy.
///
/// `f` must be ascending (a frequency axis is). Two cursors walk the window's edges while the
/// bins advance, and the window's content comes from compensated prefix sums, so neither the
/// width of the window nor the length of the curve multiplies the cost.
///
/// The mean is **weighted by each sample's own slice of the log axis** — the trapezoid rule on a
/// log-frequency axis, where a sample owns the interval halfway to its neighbours and the two end
/// samples own half a spacing each. An unweighted mean of "the samples that fell in the window"
/// is that integral only when the samples are log-uniform, and the curves here are not: a CSV
/// model hand-placed its knots, and where they sit further than the window apart the window holds
/// a single knot, which makes the smoothing a silent no-op at exactly the frequencies someone
/// took the trouble to place.
///
/// Non-finite input propagates, but *only to the windows that contain it*. A curve with a `NaN` in
/// it is a broken curve and a `NaN` that reaches the report is easier to find than a smoothing
/// window that quietly went missing; what must not happen is one `NaN` value rewriting the
/// smoothed curve at every frequency above it, or one `NaN` frequency stopping the edge cursors
/// so that every later window comes back empty. Both did.
#[must_use]
pub fn smooth_octave(f: &[f64], g: &[f64], fraction: f64) -> Vec<f64> {
    let n = f.len().min(g.len());
    if n == 0 || !(fraction.is_finite() && fraction > 0.0) {
        return vec![f64::NAN; n];
    }
    let half = fraction / 2.0;

    // Only a finite, positive frequency can take a place on a log axis. Those that cannot are
    // left off the axis instead of sitting on it: a cursor advanced with `<` or `<=` stops dead
    // at a `NaN`, because both comparisons are false against `NaN`, and every window after it
    // then reads empty.
    let mut hz = Vec::with_capacity(n);
    let mut val = Vec::with_capacity(n);
    let mut keep = Vec::with_capacity(n);
    for i in 0..n {
        let ok = f[i].is_finite() && f[i] > 0.0;
        if ok {
            hz.push(f[i]);
            val.push(g[i]);
        }
        keep.push(ok);
    }
    let m = hz.len();
    let mut out = vec![f64::NAN; n];
    if m == 0 {
        return out;
    }

    // Cell edges in log2, and the compensated prefix sums of "weight" and "weight × value".
    // A sample whose *value* is not finite keeps its place on the axis (its neighbours must not
    // grow to cover a gap that is really a broken measurement) but carries no weight; the badness
    // prefix is what turns it back into a NaN for every window that overlaps it.
    let c: Vec<f64> = hz.iter().map(|h| h.log2()).collect();
    let mut edge = Vec::with_capacity(m + 1);
    if m == 1 {
        edge.push(c[0] - 0.5);
        edge.push(c[0] + 0.5);
    } else {
        edge.push(c[0] - (c[1] - c[0]) / 2.0);
        for i in 1..m {
            edge.push((c[i - 1] + c[i]) / 2.0);
        }
        edge.push(c[m - 1] + (c[m - 1] - c[m - 2]) / 2.0);
    }
    let mut w = Vec::with_capacity(m);
    let mut wg = Vec::with_capacity(m);
    let mut bad = Vec::with_capacity(m);
    for j in 0..m {
        let width = (edge[j + 1] - edge[j]).max(0.0);
        let finite = val[j].is_finite();
        w.push(if finite { width } else { 0.0 });
        wg.push(if finite { width * val[j] } else { 0.0 });
        bad.push(if finite { 0.0 } else { 1.0 });
    }
    let pw = crate::accum::prefix_sums(&w);
    let pwg = crate::accum::prefix_sums(&wg);
    let pbad = crate::accum::prefix_sums(&bad);

    // Both cursors only ever move forward: the window's edges rise with its centre.
    let mut full_lo = 0usize;
    let mut full_hi = 0usize;
    let mut p = 0usize;
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        let lo2 = c[p] - half;
        let hi2 = c[p] + half;
        // Cells wholly inside the window are [full_lo, full_hi); at most one more cell hangs
        // over each edge, and those two are added by hand.
        while full_lo < m && edge[full_lo] < lo2 {
            full_lo += 1;
        }
        if full_hi < full_lo {
            full_hi = full_lo;
        }
        while full_hi < m && edge[full_hi + 1] <= hi2 {
            full_hi += 1;
        }
        let mut first = full_lo;
        let mut last_plus = full_hi;
        let mut weight = pw[full_hi] - pw[full_lo];
        let mut wsum = pwg[full_hi] - pwg[full_lo];
        if full_lo > 0 && edge[full_lo] > lo2 {
            let over = edge[full_lo].min(hi2) - lo2;
            weight += over;
            wsum += over * val[full_lo - 1];
            first = full_lo - 1;
        }
        if full_hi < m && edge[full_hi] < hi2 {
            let over = hi2 - edge[full_hi];
            weight += over;
            wsum += over * val[full_hi];
            last_plus = full_hi + 1;
        }
        // A non-finite value anywhere under the window makes this window non-finite, and nothing
        // else's.
        if pbad[last_plus] - pbad[first] > 0.0 {
            out[i] = f64::NAN;
        } else if weight > 0.0 {
            out[i] = wsum / weight;
        }
        p += 1;
    }
    out
}

/// Absolute deviation of the curve from its own octave-smoothed version, in dB.
///
/// This is the quantity the whole policy hangs on: it says how much of a point is *not*
/// explainable by a curve that varies smoothly by an octave. Two smoothings are applied in
/// sequence — the residual against the wide window, then a light smoothing of its magnitude —
/// because a raw `|residual|` is one-sided and spiky, and a spiky `β` puts steps in the filter.
#[must_use]
pub fn deviation(f: &[f64], g: &[f64], residual_octave: f64, beta_octave: f64) -> Vec<f64> {
    let n = f.len().min(g.len());
    let smooth = smooth_octave(f, g, residual_octave);
    let residual: Vec<f64> = (0..n).map(|i| (g[i] - smooth[i]).abs()).collect();
    smooth_octave(f, &residual, beta_octave)
}

/// The `q`-th quantile of a sample, by nearest rank: index `⌈q·n⌉ − 1` of the sorted values.
///
/// Nearest rank rather than an interpolated quantile because this feeds a knee position, not a
/// reported statistic — a value that is actually in the sample is easier to defend against the
/// curve than a weighted average of two neighbours. Non-finite samples are excluded; an empty
/// one is `f64::NAN`, which [`Policy::auto`] reads as "no information, use the floor".
#[must_use]
pub fn quantile(xs: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() || !(q.is_finite() && (0.0..=1.0).contains(&q)) {
        return f64::NAN;
    }
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (q * v.len() as f64).ceil() as usize;
    v[rank.saturating_sub(1).min(v.len() - 1)]
}

/// The diagonal load `β` for a band whose smoothed deviation is `s` dB.
///
/// `β = 10^(κ·(s − λ)/10)`. It is continuous and monotone in `s`, which is the property that
/// keeps the resulting filter smooth: `β` is `1` exactly at the knee, far below it on ordinary
/// texture (so the correction proceeds untouched), far above it on a narrow feature (so the
/// band is bypassed). A non-finite `s` is a broken measurement and gets an infinite `β`: the
/// band is not corrected, which is the safe reading of a number that does not exist.
#[must_use]
pub fn beta(s_db: f64, policy: &Policy) -> f64 {
    if !s_db.is_finite() {
        return f64::INFINITY;
    }
    if !policy.lambda_db.is_finite() {
        // λ = +∞: no band is ever past the knee, which is exactly "no regularization".
        return if policy.lambda_db > 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
    }
    let exponent = policy.kappa * (s_db - policy.lambda_db) / 10.0;
    // `powf` overflow would give +inf, which is a legitimate β (full bypass); underflow gives
    // 0, also legitimate (no regularization). Neither is a panic.
    10f64.powf(exponent)
}

/// The correction a regularized inversion applies at one frequency, in dB.
///
/// `G = (|S|·|D| + β) / (|S|² + β)` with `S = 10^(raw_db/20)` the measured magnitude, `D` the
/// target magnitude, and the all-pass reference taken as `A = 1` (flat, zero phase — this is a
/// magnitude-only correction; the minimum-phase conversion happens later, in `ir.rs`).
///
/// `β = 0` gives `|D|/|S|`, the plain inversion, i.e. `target_db − raw_db`: the curve the
/// engine writes today. `β → ∞` gives 1, i.e. 0 dB. Both ends are asserted in the tests, and
/// the second one is the reason this form was chosen over the one that penalises `‖G‖²`.
#[must_use]
///
/// `raw_db` is the measurement *relative to its own reference level* — see [`reference_db`]. The
/// load is compared against `|S|²`, so an absolute reading is a policy change in disguise.
#[cfg(test)]
pub fn regularized_gain_db(raw_db: f64, target_db: f64, beta: f64) -> f64 {
    if !raw_db.is_finite() || !target_db.is_finite() {
        return f64::NAN;
    }
    if !beta.is_finite() && beta > 0.0 {
        return 0.0; // infinite load: bypass, exactly.
    }
    if beta <= 0.0 {
        return target_db - raw_db; // no load: today's curve, exactly.
    }
    let h = 10f64.powf(raw_db / 20.0);
    let d = 10f64.powf(target_db / 20.0);
    let g = (h * d + beta) / (h * h + beta);
    if g.is_finite() && g > 0.0 {
        20.0 * g.log10()
    } else {
        f64::NAN
    }
}

/// Shrink an existing correction toward a bypass by the trust factor at that band.
///
/// `raw_db` is the measurement relative to its own reference level, exactly as in
/// [`regularized_gain_db`]: the load is weighed against `|S|²`, so `S` has to be a normalized
/// transfer function rather than an absolute reading of the rig.
///
/// `G_reg = (|S|²·G + β) / (|S|² + β)`. This is [`regularized_gain_db`] with the target taken
/// to be whatever the correction already implies — `D = S·G`, the compensated response the
/// curve is asking for — substituted into the same closed form: `|S|·|S·G| = |S|²·G`, the same
/// numerator. That identity is asserted in the tests, and it is why one policy serves both
/// places this is used: a curve derived by inverting a target, and a hand-tuned curve whose
/// target was decided by a person.
///
/// It answers the question the hand-tuned case otherwise dodges. A shipped curve's `+11 dB` at
/// 9.5 kHz may be correcting a real dip or chasing a coupler resonance, and the difference is
/// visible in the measurement: the second sits on a band whose own deviation from a smoothed
/// curve is large. `β` reads that and walks the boost back to 0 dB, while a broad shelf two
/// octaves lower — where `|S|² ≫ β` — is returned untouched, to the last bit when `β = 0`.
#[must_use]
pub fn regularize_correction(raw_db: f64, gain_db: f64, beta: f64) -> f64 {
    if !raw_db.is_finite() || !gain_db.is_finite() {
        return f64::NAN;
    }
    if !beta.is_finite() && beta > 0.0 {
        return 0.0; // infinite load: bypass.
    }
    if beta <= 0.0 {
        return gain_db; // no load: the curve as written, exactly, not a round trip through 10^x.
    }
    let power = 10f64.powf(raw_db / 10.0); // |S|²
    let g = 10f64.powf(gain_db / 20.0);
    let reg = (power * g + beta) / (power + beta);
    if reg.is_finite() && reg > 0.0 {
        20.0 * reg.log10()
    } else {
        f64::NAN
    }
}

/// The deviation-based load `β` at each of the given frequencies, for a curve that is not
/// itself being inverted.
///
/// `raw_f`/`raw_g` are the measured response; the deviation is read from it and evaluated at
/// `query_hz` (a hand-tuned correction curve's own knot frequencies, typically). This is the
/// piece a caller outside the derivation path needs, so the policy's two window widths stay in
/// the [`Policy`] rather than being re-decided at each call site.
#[must_use]
pub fn beta_at_freqs(raw_f: &[f64], raw_g: &[f64], query_hz: &[f64], policy: &Policy) -> Vec<f64> {
    if raw_f.len() != raw_g.len() || raw_f.len() < 2 {
        return vec![f64::INFINITY; query_hz.len()];
    }
    let valid: Vec<(f64, f64)> = raw_f
        .iter()
        .zip(raw_g)
        .filter(|(&f, &g)| f.is_finite() && f > 0.0 && g.is_finite())
        .map(|(&f, &g)| (f, g))
        .collect();
    if valid.len() < 2 {
        return vec![f64::INFINITY; query_hz.len()];
    }
    let (vf, vg): (Vec<f64>, Vec<f64>) = valid.into_iter().unzip();
    let s = deviation(&vf, &vg, policy.residual_octave, policy.beta_octave);
    // Interpolate the deviation itself onto the query frequencies. It must not be smoothed a
    // second time here: `deviation` already closes with a 1/6-octave window, and a further
    // half-octave one erases exactly the narrow features the load is supposed to find.
    let curve = crate::pchip::Pchip::new(&vf, &s);
    query_hz
        .iter()
        .map(|&hz| {
            let v = curve.eval(hz);
            // A non-finite reading is a band nothing is known about, which is the safe end:
            // infinite load, band bypassed.
            beta(v, policy)
        })
        .collect()
}

/// The regularized correction curve, as `(freq_hz, gain_dB)` at the measurement frequencies.
///
/// `raw` and `target` are the same knot lists the rest of the engine uses; the target is
/// carried onto the raw grid with the same log-frequency PCHIP the control-point derivation
/// uses, so the two paths differ in the regularizer and in nothing else. Frequencies are the
/// measurement frequencies, so a caller can hand the result straight to
/// [`crate::ir::synthesize_minimum_phase_ir`] the way it would hand it the plain knots.
#[cfg(test)]
#[must_use]
pub fn regularized_curve(
    raw: &[(f64, f64)],
    target: &[(f64, f64)],
    policy: &Policy,
) -> Vec<(f64, f64)> {
    if raw.len() < 2 || target.len() < 2 {
        return Vec::new();
    }
    let f: Vec<f64> = raw.iter().map(|&(x, _)| x).collect();
    let g: Vec<f64> = raw.iter().map(|&(_, y)| y).collect();
    let tf: Vec<f64> = target.iter().map(|&(x, _)| x).collect();
    let tg: Vec<f64> = target.iter().map(|&(_, y)| y).collect();
    let target_on_raw = crate::pchip::Pchip::new(&tf, &tg).eval_grid(&f);
    let s = deviation(&f, &g, policy.residual_octave, policy.beta_octave);
    // Both sides of `S·D` are put on the curve's own reference. They have to move together: the
    // error this produces when the load is zero is `target − raw`, and a shared offset cancels out
    // of that, so the unregularized curve is bit-for-bit what it always was.
    let base = reference_db(raw);

    raw.iter()
        .zip(target_on_raw)
        .zip(&s)
        .map(|((raw_pt, target_db), &deviation_db)| {
            let (freq, raw_db) = *raw_pt;
            (
                freq,
                regularized_gain_db(raw_db - base, target_db - base, beta(deviation_db, policy)),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1/12-octave grid from 20 Hz to 20 kHz, the density the shipped measurements use.
    fn grid(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 20.0 * 10f64.powf(f64::from(i as u32) / f64::from((n - 1) as u32) * 3.0))
            .collect()
    }

    /// The smoother must be the mean of the window it claims, including at the ends where the
    /// window hangs off the grid. Recomputed the naive way, bin by bin, on a random curve.
    /// The window's mean is the trapezoid integral of the curve over the window divided by the
    /// window's own measure, computed here independently and slowly from the definition.
    ///
    /// The previous version of this test asserted the *unweighted* mean of the samples whose
    /// centres happened to fall inside the window, which is the same thing only on a log-uniform
    /// axis; it is now the reference implementation's job to say what the mean is.
    #[test]
    fn the_smoother_is_the_log_domain_integral_of_the_window() {
        let f = grid(361);
        // Deterministic pseudo-noise: no RNG, no global state, same curve every run.
        let g: Vec<f64> = (0..f.len())
            .map(|i| ((i as f64) * 0.732).sin() * 3.0 + ((i as f64) * 2.11).cos())
            .collect();
        let half = 0.25f64;
        let got = smooth_octave(&f, &g, 0.5);
        let want = brute_log_trapezoid(&f, &g, half);
        for i in 0..f.len() {
            assert!(
                (got[i] - want[i]).abs() < 1e-9,
                "bin {i} ({:.1} Hz): smoothed {}, integral of the window {}",
                f[i],
                got[i],
                want[i]
            );
        }
        // The first window is genuinely partial, and a flat curve must stay flat at the ends:
        // divide a truncated window by the nominal width and the ends of the curve get pulled
        // toward zero, at exactly the two places a headphone measurement is least trustworthy.
        let count0 = f
            .iter()
            .filter(|y| {
                let lo = f[0] * 2f64.powf(-half);
                **y >= lo && **y <= f[0] * 2f64.powf(half)
            })
            .count();
        assert!(count0 < 30, "edge window should be truncated, got {count0}");
        let flat = vec![1.0f64; f.len()];
        let s = smooth_octave(&f, &flat, 0.5);
        for (i, &v) in s.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-12,
                "bin {i} of a flat curve smoothed to {v}"
            );
        }
    }

    /// The definition, written slowly: build each sample's log-domain cell, clip the cells to the
    /// window, and take the value-weighted mean of the clipped widths.
    fn brute_log_trapezoid(f: &[f64], g: &[f64], half: f64) -> Vec<f64> {
        let n = f.len();
        let c: Vec<f64> = f.iter().map(|x| x.log2()).collect();
        let mut edge: Vec<f64> = Vec::with_capacity(n + 1);
        edge.push(c[0] - (c[1] - c[0]) / 2.0);
        for i in 1..n {
            edge.push((c[i - 1] + c[i]) / 2.0);
        }
        edge.push(c[n - 1] + (c[n - 1] - c[n - 2]) / 2.0);
        (0..n)
            .map(|i| {
                let (lo, hi) = (c[i] - half, c[i] + half);
                let mut w = 0.0;
                let mut s = 0.0;
                for j in 0..n {
                    let over = (edge[j + 1].min(hi) - edge[j].max(lo)).max(0.0);
                    w += over;
                    s += over * g[j];
                }
                s / w
            })
            .collect()
    }

    /// One `NaN` value must reach the windows that contain it and stop there. It used to poison
    /// every prefix sum after it, so the smoothed curve came back `NaN` at every frequency above
    /// the bad sample — and a `NaN` frequency wedged both cursors, which emptied every later
    /// window as well.
    #[test]
    fn one_bad_sample_reaches_the_windows_that_hold_it_and_no_others() {
        let f = grid(361);
        let g: Vec<f64> = (0..f.len())
            .map(|i| ((i as f64) * 0.732).sin() * 3.0 + ((i as f64) * 2.11).cos())
            .collect();
        let clean = smooth_octave(&f, &g, 0.5);

        let mut poisoned = g.clone();
        let k = 180;
        poisoned[k] = f64::NAN;
        let got = smooth_octave(&f, &poisoned, 0.5);
        assert!(got[k].is_nan(), "the broken sample itself must stay broken");
        let mut near = 0usize;
        for i in 0..f.len() {
            let window = f[i] * 2f64.powf(-0.25)..=f[i] * 2f64.powf(0.25);
            let holds_it = window.contains(&f[k]);
            if holds_it {
                near += 1;
                assert!(got[i].is_nan(), "bin {i} holds the NaN and must read NaN");
            } else {
                assert!(
                    (got[i] - clean[i]).abs() < 1e-12,
                    "bin {i} ({:.1} Hz) does not hold the NaN but smoothed to {} instead of {}",
                    f[i],
                    got[i],
                    clean[i]
                );
            }
        }
        assert!(
            near > 0 && near < 40,
            "a half-octave window is narrow: {near} bins"
        );

        // An infinity behaves the same way: it is a value, not a window.
        let mut infinite = g.clone();
        infinite[k] = f64::NEG_INFINITY;
        let got = smooth_octave(&f, &infinite, 0.5);
        assert!(
            got[k].is_nan(),
            "an infinite value cannot be averaged into a mean"
        );
        assert!(
            (got[f.len() - 1] - clean[f.len() - 1]).abs() < 1e-12,
            "the far end of the curve is past the infinity: {}",
            got[f.len() - 1]
        );

        // A `NaN` frequency: no window can be built for it, but the axis below and above it still
        // has to be walked to the end.
        let mut broken_hz = f.clone();
        broken_hz[k] = f64::NAN;
        let got = smooth_octave(&broken_hz, &g, 0.5);
        assert!(
            got[k].is_nan(),
            "a frequency that is not a frequency has no window"
        );
        for i in (k + 30)..f.len() {
            assert!(
                got[i].is_finite(),
                "bin {i} ({:.1} Hz) went non-finite behind a NaN frequency: {}",
                f[i],
                got[i]
            );
            assert!(
                (got[i] - clean[i]).abs() < 1.0,
                "bin {i} smoothed to {} where the clean curve says {}",
                got[i],
                clean[i]
            );
        }
    }

    /// Knots that are unevenly spaced cover uneven slices of the log axis, and an unweighted mean
    /// of "the samples whose centres fell in the window" treats a knot standing for two thirds of
    /// an octave and a knot standing for a fortieth as if they said the same amount.
    ///
    /// Measured on the axis below, at the 120 Hz knot with a sixth-octave window: the window
    /// (113.3–127.1 Hz) is 69 % inside 120 Hz's own cell and 31 % inside 115 Hz's, so the curve
    /// there is 8.21 dB. Counting two samples and averaging them gives 6.00 dB — 2.2 dB of
    /// regularization that simply is not there, at the low end of a hand-placed curve where knots
    /// are sparsest.
    #[test]
    fn uneven_knot_spacing_is_weighted_by_the_axis_it_covers() {
        let f = vec![100.0, 105.0, 110.0, 115.0, 120.0, 400.0, 1000.0];
        let g = vec![0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0];
        let got = smooth_octave(&f, &g, 1.0 / 6.0);
        let want = brute_log_trapezoid(&f, &g, 1.0 / 12.0);
        for i in 0..f.len() {
            assert!(
                (got[i] - want[i]).abs() < 1e-9,
                "knot {i} ({} Hz): smoothed {}, the window's integral is {}",
                f[i],
                got[i],
                want[i]
            );
        }
        assert!(
            (got[4] - 8.21).abs() < 0.02,
            "the 120 Hz knot carries most of its window: got {}",
            got[4]
        );
        // What counting samples instead of widths would have said, and how far off it is.
        let half = 1.0 / 12.0;
        let (lo, hi) = (f[4] * 2f64.powf(-half), f[4] * 2f64.powf(half));
        let in_window: Vec<f64> = f
            .iter()
            .zip(&g)
            .filter(|(x, _)| **x >= lo && **x <= hi)
            .map(|(_, y)| *y)
            .collect();
        let counted = in_window.iter().sum::<f64>() / in_window.len() as f64;
        assert!(
            (got[4] - counted).abs() > 1.0,
            "counting two samples instead of integrating the axis they cover: {counted} vs {}",
            got[4]
        );
    }

    /// The prefix sums behind a window are compensated, and this is the case that shows it: one
    /// absurd sample, then a long run of ordinary ones. Once the running total has reached
    /// 2.6e13, adding the next cell's weighted contribution (0.0026) to it changes nothing, so an
    /// uncompensated prefix reports a window total of exactly zero and the smoother answers 0 dB
    /// for a stretch of curve that is uniformly 1 dB — at every frequency above the outlier, not
    /// just at it.
    ///
    /// The tolerance is not tighter than it looks. Compensation keeps the small terms from being
    /// dropped on the way in, but a window total is still the *difference of two prefixes*, and
    /// those are quantised to one ulp of 2.6e13, which is 0.0039 — about 0.008 dB once it is
    /// divided by the window's weight. An uncompensated prefix is off by the whole 1 dB.
    #[test]
    fn a_window_far_from_an_outlier_still_averages_what_it_covers() {
        let f = grid(4001);
        let mut g = vec![1.0f64; f.len()];
        g[0] = 1e16;
        let s = smooth_octave(&f, &g, 0.5);
        assert!(
            s[0].is_finite(),
            "the outlier's own window has a finite mean"
        );
        for i in (400..f.len()).step_by(251) {
            assert!(
                (s[i] - 1.0).abs() < 0.05,
                "bin {i} ({:.0} Hz) is decades above the outlier and read {:+} instead of +1",
                f[i],
                s[i]
            );
        }
    }

    /// The load is weighed against `|S|²`, so `S` has to mean a normalized transfer function. A
    /// measurement saved in absolute units — a rig logging dB SPL puts every sample near +94 —
    /// used to carry that offset straight into `|S|²`, which is arithmetically the same as
    /// dividing the load by `10^9.4`: the same curve, written in different units, was corrected in
    /// full where it should have been bypassed. The target moves with it, because the two are
    /// stated on one scale, and the correction it asks for is a difference that does not care.
    #[test]
    fn a_curve_saved_in_absolute_units_carries_the_same_policy() {
        let f = grid(241);
        // A 12 dB null a sixth of an octave wide at 9.5 kHz on an otherwise flat curve.
        let raw: Vec<(f64, f64)> = f
            .iter()
            .map(|&x| {
                let null = -12.0 * (-2.0 * ((x / 9500.0).log2() / (1.0 / 6.0)).powi(2)).exp();
                (x, null)
            })
            .collect();
        let target = vec![(20.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)];
        let policy = Policy::default();
        let plain = regularized_curve(&raw, &target, &Policy::fixed(f64::INFINITY));
        let want = regularized_curve(&raw, &target, &policy);
        assert_eq!(want.len(), plain.len());
        let bites = (0..want.len()).any(|i| (want[i].1 - plain[i].1).abs() > 1.0);
        assert!(
            bites,
            "the fixture must actually regularize something: max difference {}",
            (0..want.len())
                .map(|i| (want[i].1 - plain[i].1).abs())
                .fold(0.0f64, f64::max)
        );

        for offset in [94.0f64, -40.0, 0.5] {
            let shifted: Vec<(f64, f64)> = raw.iter().map(|&(x, y)| (x, y + offset)).collect();
            let shifted_target: Vec<(f64, f64)> =
                target.iter().map(|&(x, y)| (x, y + offset)).collect();
            let got = regularized_curve(&shifted, &shifted_target, &policy);
            for i in 0..got.len() {
                assert!(
                    (got[i].1 - want[i].1).abs() < 1e-9,
                    "the same measurement written {offset:+.1} dB high regularized {:.0} Hz to {} \
                     dB instead of {} dB",
                    got[i].0,
                    got[i].1,
                    want[i].1
                );
            }
        }
    }

    /// The property `--regularize` is sold on: it can only ever make a correction gentler. The
    /// shrink is a weighted mean of the curve's own gain and a bypass, in linear amplitude, with
    /// the measurement's power `|S|²` on the curve and the load `β` on the bypass, so the result
    /// is trapped between them — it cannot overshoot past 0 dB, and it cannot deepen a correction
    /// that was already there.
    ///
    /// The pin is the worst knot on a shipped model, measured rather than invented: the
    /// Beyerdynamic DT 770 Pro at 3703 Hz, where the raw measurement sits 4.87 dB *below* unity,
    /// the shipped curve asks for `+4.26 dB` to fill that null, and the load there is 9.7. The
    /// answer is `+0.18 dB`: the boost is walked almost all the way back, which is the whole
    /// point of the flag. Two wrong implementations land on numbers this test rejects — putting
    /// the amplitude `|S|` where the form wants the power `|S|²` gives `+0.30 dB`, and dropping
    /// the `|S|²` weight altogether (shrinking by `β` alone, so a loud band is as suspect as a
    /// silent one) gives `+0.50 dB`.
    #[test]
    fn the_shrink_lands_between_the_curve_and_bypass_and_never_past_either() {
        assert!(
            (regularize_correction(-4.87, 4.26, 9.7) - 0.177).abs() < 0.005,
            "the DT 770's 3.7 kHz knot read {}",
            regularize_correction(-4.87, 4.26, 9.7)
        );

        let mut beta = 1e-3f64;
        while beta < 1e4 {
            for &gain_db in &[-12.0, -6.0, -3.0, 0.0, 3.0, 6.0, 12.0] {
                for &raw_db in &[-18.0, -9.0, -3.0, 0.0, 3.0, 9.0, 18.0] {
                    let g = regularize_correction(raw_db, gain_db, beta);
                    assert!(
                        g.is_finite(),
                        "finite inputs gave a non-finite shrink at beta {beta}"
                    );
                    assert!(
                        g.abs() <= gain_db.abs() + 1e-12,
                        "beta {beta}, measurement {raw_db} dB, curve {gain_db} dB: the shrink came \
                         out at {g} dB, past the curve it was asked to soften"
                    );
                    assert!(
                        g.abs() < 1e-12 || g.signum() == gain_db.signum(),
                        "beta {beta}, measurement {raw_db} dB, curve {gain_db} dB: the shrink \
                         changed the sign of the correction, landing on {g} dB"
                    );
                }
            }
            // More load, more walk-back: monotone toward bypass, never away from it.
            for &gain_db in &[-6.0, 6.0] {
                let softer = regularize_correction(-4.0, gain_db, beta * 10.0);
                let harder = regularize_correction(-4.0, gain_db, beta);
                assert!(
                    softer.abs() < harder.abs() + 1e-12,
                    "load {beta} gave {harder} dB and ten times the load gave {softer} dB"
                );
            }
            beta *= 10.0;
        }

        // And the weight on `|S|²` is what makes the load bite where it should: at the same load,
        // the deeper the null the boost is aimed at, the less of that boost survives.
        let mut previous = f64::MAX;
        for raw_db in [6.0, 0.0, -6.0, -12.0, -18.0] {
            let g = regularize_correction(raw_db, 6.0, 5.0);
            assert!(
                g < previous,
                "a boost into a {raw_db} dB band survived at {g} dB, no less than the louder band \
                 before it"
            );
            previous = g;
        }
        assert!(
            regularize_correction(6.0, 6.0, 5.0) > 3.0,
            "a boost where the measurement is actually loud should mostly stand: {}",
            regularize_correction(6.0, 6.0, 5.0)
        );
    }

    /// The limit of all this, stated rather than hidden: when knots sit further apart than the
    /// window, a window holds one knot's cell and there is nothing to average — the smoother
    /// returns the knot itself. That is a property of smoothing a *sampled* curve at a fractional
    /// octave width, not of the weighting, and it is why `deviation()` on a hand-placed CSV axis
    /// with 0.7-octave gaps reads no deviation at all. A curve that dense has to be resampled
    /// before it is smoothed, and nothing here does that; the assertion exists so that changing it
    /// later is a decision and not an accident.
    #[test]
    fn a_window_narrower_than_a_knot_spacing_has_nothing_to_average() {
        let f = vec![
            20.0, 55.0, 140.0, 400.0, 1100.0, 2900.0, 5300.0, 8100.0, 16000.0,
        ];
        let mut g = vec![0.0f64; f.len()];
        g[6] = 9.0;
        let s = smooth_octave(&f, &g, 1.0 / 6.0);
        assert!(
            (s[6] - 9.0).abs() < 1e-12,
            "a sixth-octave window inside a 0.7-octave cell reads the knot: {}",
            s[6]
        );
        // The next knot is 0.62 octave away, so a one-octave window does reach it.
        let s = smooth_octave(&f, &g, 1.0);
        assert!(
            s[6] < 8.0,
            "an octave window on the same axis must mix: {}",
            s[6]
        );
    }

    /// A flat curve smooths to itself, and adding an offset changes nothing about the fit.
    #[test]
    fn the_fit_measures_shape_not_level() {
        let f = grid(241);
        let flat = vec![0.0f64; f.len()];
        let s_flat = deviation(&f, &flat, RESIDUAL_SMOOTH_OCTAVE, BETA_SMOOTH_OCTAVE);
        assert!(
            s_flat.iter().all(|&v| v.abs() < 1e-12),
            "a flat curve has no deviation, got max {}",
            s_flat.iter().copied().fold(0.0f64, f64::max)
        );

        // A shelf is broad: its residual is confined to the transition, and small.
        let shelf: Vec<f64> = f
            .iter()
            .map(|&x| 6.0 / (1.0 + (1000.0 / x).powf(4.0)))
            .collect();
        let s_shelf = deviation(&f, &shelf, RESIDUAL_SMOOTH_OCTAVE, BETA_SMOOTH_OCTAVE);
        // Shifted up by 12 dB — same shape, different level. λ must not notice.
        let lifted: Vec<f64> = shelf.iter().map(|&y| y + 12.0).collect();
        let s_lifted = deviation(&f, &lifted, RESIDUAL_SMOOTH_OCTAVE, BETA_SMOOTH_OCTAVE);
        for (a, b) in s_shelf.iter().zip(&s_lifted) {
            assert!(
                (a - b).abs() < 1e-12,
                "the fit is level-sensitive: {a} vs {b}"
            );
        }
        assert!(
            s_shelf.iter().all(|&v| v < 2.0),
            "a broad shelf should not read as a fit failure: max {}",
            s_shelf.iter().copied().fold(0.0f64, f64::max)
        );
    }

    /// The automatic knee must ignore a lone notch (or it would negotiate its way out of being
    /// bypassed) and follow texture that is broad. Both halves are measured, not assumed.
    #[test]
    fn the_auto_knee_ignores_a_sparse_notch_and_follows_broad_texture() {
        let f = grid(361);
        let notch = |depth: f64, width_oct: f64| -> Vec<f64> {
            f.iter()
                .map(|&x| {
                    let d = (x / 1000.0).log2() / width_oct;
                    -depth * (-d * d * 2.0).exp()
                })
                .collect()
        };
        let ripple = |amp: f64, per_oct: f64| -> Vec<f64> {
            f.iter()
                .map(|&x| amp * (2.0 * std::f64::consts::PI * (x / 1000.0).log2() / per_oct).sin())
                .collect()
        };
        let flat = vec![0.0f64; f.len()];

        // A 12 dB notch one twelfth of an octave wide touches a handful of bins, so the top
        // decile never sees it and the knee stays at the floor — which is what lets that notch
        // be bypassed at all.
        assert_eq!(Policy::auto(&f, &flat).lambda_db, LAMBDA_FLOOR_DB);
        assert_eq!(
            Policy::auto(&f, &notch(12.0, 1.0 / 12.0)).lambda_db,
            LAMBDA_FLOOR_DB,
            "a sparse notch must not raise the knee"
        );
        // Broad texture does raise it, and by how much was measured at this grid: a ±3 dB
        // ripple every twelfth of an octave has q90 = 2.02 dB (still under the floor), a ±9 dB
        // one has q90 = 6.07 dB → knee at 1.3 × 6.07 = 7.89 dB.
        assert_eq!(
            Policy::auto(&f, &ripple(3.0, 1.0 / 12.0)).lambda_db,
            LAMBDA_FLOOR_DB,
            "mild ripple should sit at the floor"
        );
        let wild = Policy::auto(&f, &ripple(9.0, 1.0 / 12.0)).lambda_db;
        assert!(
            (wild - 7.89).abs() < 0.6,
            "broad ±9 dB texture should lift the knee to about 7.9 dB, got {wild}"
        );
        assert!(wild > LAMBDA_FLOOR_DB, "and it must be above the floor");
        // A curve noisier than that lifts it further: the estimate is monotone in the texture.
        let wilder = Policy::auto(&f, &ripple(14.0, 1.0 / 12.0)).lambda_db;
        assert!(
            wilder >= wild,
            "more texture, higher knee: {wilder} vs {wild}"
        );
        // A hand-set λ is used verbatim, in either direction.
        assert_eq!(Policy::fixed(11.5).lambda_db, 11.5);
        assert_eq!(Policy::fixed(1.0).lambda_db, 1.0);
    }

    /// The two ends of the closed form, which is the whole reason for the all-pass reference:
    /// no load gives today's curve, infinite load gives a bypass and not silence.
    #[test]
    fn infinite_load_bypasses_and_zero_load_is_todays_curve() {
        for &(raw, target) in &[
            (-6.0, 0.0),
            (3.5, 0.0),
            (-20.0, 0.0),
            (0.0, -4.25),
            (12.0, 1.5),
        ] {
            let plain = regularized_gain_db(raw, target, 0.0);
            assert!(
                (plain - (target - raw)).abs() < 1e-12,
                "β=0 must reproduce `target - raw` exactly: {raw}, {target} -> {plain}"
            );
            let bypass = regularized_gain_db(raw, target, f64::INFINITY);
            assert!(
                bypass.abs() < 1e-12,
                "β=∞ must give 0 dB, not silence: {raw}, {target} -> {bypass}"
            );
        }
        // Approaching the limit numerically, the same way a real policy does.
        let deep = regularized_gain_db(-20.0, 0.0, 1e12);
        assert!(
            deep.abs() < 1e-6,
            "a huge load should be a bypass, got {deep} dB"
        );

        // The trap this form exists to avoid, at the same load. The conventional published form
        // (Norcross et al., and `pyfar.dsp.RegularizedSpectrumInversion`) penalises the filter's
        // own magnitude: `|G| = |S|·|D| / (|S|² + β|ε|²)`. At a load large enough to matter, the
        // same band that this module sends to 0 dB goes to silence instead — here -230 dB, about
        // fourteen hundred dB below the bypass a listener needed. Asserted so that a refactor
        // cannot swap the forms in without a test failing.
        let h = 10f64.powf(-20.0 / 20.0);
        let d = 1.0;
        let naive = 20.0 * ((h * d) / (h * h + 1e12)).log10();
        assert!(
            naive < -100.0,
            "the ‖G‖² form is supposed to collapse toward silence; measured {naive} dB"
        );
        assert!(regularized_gain_db(-20.0, 0.0, 1e12) > naive + 100.0);
    }

    /// The policy's shape, with the numbers measured when `KNEE_PER_DB` was chosen: ordinary
    /// texture keeps its correction, a real notch loses it, and nothing in between jumps.
    #[test]
    fn texture_is_left_alone_and_a_notch_is_walked_back() {
        let p = Policy::default();
        assert!((p.lambda_db - 3.0).abs() < 1e-12);
        // s = smoothed deviation, in dB. The numbers below are the measured shape of the knee
        // at λ = 3, κ = 8: the shipped curves have a median deviation of 0.06–0.34 dB and worst
        // bands of 3.6–8.4 dB, so the first two assertions are ordinary measurement and the
        // last is a narrow HF notch.
        let gain_at = |s: f64| regularized_gain_db(-6.0, 0.0, beta(s, &p));
        assert!(
            (gain_at(0.34) - 5.874).abs() < 0.02,
            "median texture on the shipped curves must keep the correction, got {}",
            gain_at(0.34)
        );
        assert!(
            (gain_at(1.0) - 5.597).abs() < 0.02,
            "1 dB of texture must keep the correction, got {}",
            gain_at(1.0)
        );
        assert!(
            gain_at(8.4) < 0.01,
            "the worst band measured on the DT 770 must be bypassed, got {}",
            gain_at(8.4)
        );
        assert!(
            gain_at(7.0) < 0.1,
            "a 7 dB excursion must be bypassed, got {}",
            gain_at(7.0)
        );
        // Half-trust at the knee, and monotone all the way through.
        let half = gain_at(p.lambda_db);
        assert!(
            half > 1.0 && half < 2.0,
            "at the knee a 6 dB dip halves: {half}"
        );
        let mut prev = f64::INFINITY;
        for i in 0..60 {
            let g = gain_at(f64::from(i) / 10.0);
            assert!(
                g <= prev + 1e-12,
                "the response to the knee is not monotone at {i}"
            );
            assert!(g.is_finite(), "gain went non-finite at {i}");
            prev = g;
        }
        // λ is the knob: raise it and the same notch is believed again.
        let believed = regularized_gain_db(-6.0, 0.0, beta(7.0, &Policy::fixed(20.0)));
        assert!(
            believed > 5.9,
            "λ=20 should leave a 7 dB excursion alone, got {believed}"
        );
        // A non-finite deviation is a broken measurement: bypass the band, do not correct it.
        assert!(beta(f64::NAN, &p).is_infinite());
        assert_eq!(regularized_gain_db(-6.0, 0.0, beta(f64::NAN, &p)), 0.0);
    }

    /// End to end on a curve: a narrow notch stops being corrected, a broad region beside it
    /// keeps every dB it had.
    #[test]
    fn a_narrow_notch_stops_being_corrected_and_its_neighbourhood_does_not() {
        let f = grid(361);
        // Raw: flat, plus a 12 dB notch one twelfth of an octave wide at 1 kHz.
        let raw: Vec<f64> = f
            .iter()
            .map(|&x| {
                let d = (x / 1000.0).log2() * 12.0;
                -12.0 * (-d * d * 2.0).exp()
            })
            .collect();
        let raw_pts: Vec<(f64, f64)> = f.iter().zip(&raw).map(|(&x, &y)| (x, y)).collect();
        // Target: +6 dB everywhere, so the plain correction is +6 dB at every knot.
        let target = vec![(20.0, 6.0), (20_000.0, 6.0)];

        let off = regularized_curve(&raw_pts, &target, &Policy::fixed(f64::INFINITY));
        let on = regularized_curve(&raw_pts, &target, &Policy::default());
        assert_eq!(off.len(), raw_pts.len());

        let at = |curve: &[(f64, f64)], hz: f64| -> f64 {
            curve
                .iter()
                .min_by(|a, b| {
                    ((a.0 / hz).log10())
                        .abs()
                        .partial_cmp(&((b.0 / hz).log10()).abs())
                        .unwrap()
                })
                .map(|&(_, g)| g)
                .expect("grid is non-empty")
        };
        // With the regularizer off, the plain inversion: +6 dB where the raw curve is flat, and
        // the full +18 dB it would have applied inside the notch (6 − (−12)). That +18 dB into a
        // feature one twelfth of an octave wide is precisely the demand being negotiated away.
        // Compared against the knot that is actually nearest 1 kHz: this grid has no bin at
        // exactly 1000 Hz (it falls at index 203.72), so hard-coding 18.0 would assert a
        // frequency the curve never had.
        let nearest_raw = raw_pts
            .iter()
            .min_by(|a, b| {
                ((a.0 / 1000.0).log10())
                    .abs()
                    .partial_cmp(&((b.0 / 1000.0).log10()).abs())
                    .unwrap()
            })
            .map(|&(_, g)| g)
            .expect("grid is non-empty");
        assert!(
            (at(&off, 1000.0) - (6.0 - nearest_raw)).abs() < 1e-9,
            "off must be the plain inversion: {} against {}",
            at(&off, 1000.0),
            6.0 - nearest_raw
        );
        assert!((at(&off, 8000.0) - 6.0).abs() < 1e-9);
        // At the notch, the correction is gone; an octave away, it is not.
        let in_notch = at(&on, 1000.0);
        let away = at(&on, 4000.0);
        assert!(
            in_notch.abs() < 1.0,
            "the 12 dB notch should be bypassed, still correcting {in_notch:.2} dB"
        );
        assert!(
            (away - 6.0).abs() < 0.05,
            "a flat region an octave away keeps its correction, got {away:.2} dB"
        );
        assert!(
            in_notch < away - 4.0,
            "the two must be clearly different: {in_notch} vs {away}"
        );
    }

    /// The two paths must be the same maths. Shrinking a hand-tuned curve that happens to be
    /// the plain inversion has to equal inverting the target directly — the numerators are
    /// identical (`|S|·|S·G| = |S|²·G`), so any difference is an implementation slip.
    #[test]
    fn shrinking_a_plain_inversion_is_the_same_as_inverting_the_target() {
        let p = Policy::default();
        for &(raw, target) in &[
            (-6.0, 0.0),
            (4.0, 0.0),
            (-20.0, 0.0),
            (0.0, -3.5),
            (-11.0, 2.0),
            (12.0, 1.5),
        ] {
            for s in [-2.0, 0.0, 1.5, 3.0, 5.0, 9.0] {
                let b = beta(s, &p);
                let via_curve = regularize_correction(raw, target - raw, b);
                let via_target = regularized_gain_db(raw, target, b);
                assert!(
                    (via_curve - via_target).abs() < 1e-9,
                    "raw {raw}, target {target}, s {s}: correction path {via_curve} vs inversion path {via_target} (beta {b})"
                );
            }
        }
        // And the two ends still hold on this path.
        assert_eq!(regularize_correction(-6.0, 6.0, 0.0), 6.0);
        assert_eq!(regularize_correction(-6.0, 6.0, f64::INFINITY), 0.0);
        assert!(regularize_correction(f64::NAN, 6.0, 1.0).is_nan());
        assert!(regularize_correction(-6.0, f64::INFINITY, 1.0).is_nan());
    }

    /// A hand-tuned curve keeps its shelf and loses its notch chase, at the frequencies the
    /// shipped measurements actually put their worst deviation.
    #[test]
    fn a_hand_tuned_curve_keeps_its_shelf_and_loses_its_notch_chase() {
        // Raw: flat except a narrow dip at 9.5 kHz (the deviation peak all three shipped
        // curves have) and a broad 4 dB bass rise, which is real system response.
        let f = grid(361);
        // σ = 1/12 octave: a coupler notch, which is what the worst bands on the shipped
        // measurements actually are. Wider than that and it is no longer narrow relative to the
        // half-octave reference curve — at 1/4 octave it stops reading as a fit failure, which
        // is the smoother behaving correctly, not a bug.
        let raw: Vec<f64> = f
            .iter()
            .map(|&x| {
                let notch = -10.0 * (-2.0 * ((x / 9500.0).log2() / (1.0 / 12.0)).powi(2)).exp();
                let bass = 4.0 / (1.0 + (x / 150.0).powf(4.0))
                    + 0.4 * (2.0 * std::f64::consts::PI * (x / 1000.0).log2() * 24.0).sin();
                notch + bass
            })
            .collect();
        let p = Policy::default();
        let hs: Vec<f64> = vec![60.0, 1000.0, 9500.0];
        let loads = beta_at_freqs(&f, &raw, &hs, &p);
        let (bass_load, mid_load, notch_load) = (loads[0], loads[1], loads[2]);
        assert!(
            notch_load > bass_load * 100.0 && notch_load > mid_load * 100.0,
            "the notch must carry by far the largest load: {loads:?}"
        );

        // A +10 dB boost at each of the three, regularized.
        let bass = regularize_correction(2.0, 10.0, bass_load);
        let mid = regularize_correction(0.0, 10.0, mid_load);
        let notch = regularize_correction(-10.0, 10.0, notch_load);
        assert!(
            (bass - 10.0).abs() < 0.1,
            "a broad bass shelf must survive: {bass}"
        );
        assert!(
            (mid - 10.0).abs() < 0.5,
            "a flat mid-band boost must survive: {mid}"
        );
        // Numbers measured for this curve at this grid: deviation 0.25 / 0.24 / 3.53 dB and
        // load 6.3e-3 / 6.2e-3 / 2.6, turning a +10 dB boost into +9.98 / +9.96 / +0.64 dB.
        assert!(
            notch < 1.0,
            "the 9.5 kHz notch chase must be walked back, still {notch:.2} dB"
        );
        assert!(
            notch_load > 100.0 * mid_load,
            "a narrow notch must carry two orders of magnitude more load, got {notch_load:.3} vs {mid_load:.3}"
        );
        // The load lookup must not invent a load for a curve it cannot read.
        assert!(beta_at_freqs(&[], &[], &hs, &p)
            .iter()
            .all(|&b| b.is_infinite()));
        assert!(beta_at_freqs(&[1.0], &[1.0], &hs, &p)
            .iter()
            .all(|&b| b.is_infinite()));
    }

    /// Degenerate inputs must return nothing or a `NaN`, never a panic and never a wild number.
    #[test]
    fn degenerate_curves_are_answered_not_panicked_on() {
        let f = grid(24);
        let g = vec![0.0; f.len()];
        let pts: Vec<(f64, f64)> = f.iter().zip(&g).map(|(&x, &y)| (x, y)).collect();

        assert!(regularized_curve(&[], &[], &Policy::default()).is_empty());
        assert!(regularized_curve(&pts[..1], &pts, &Policy::default()).is_empty());
        assert!(regularized_curve(&pts, &pts[..1], &Policy::default()).is_empty());
        // A zero frequency is not a frequency, and must not become a divide-by-zero.
        let junk_f = vec![0.0, -30.0, 1000.0, 2000.0];
        let junk_g = vec![1.0, 2.0, 3.0, 4.0];
        let s = smooth_octave(&junk_f, &junk_g, 0.5);
        assert!(
            s[..2].iter().all(|&v| v.is_nan()),
            "no valid centre, no value"
        );
        assert!(
            s[2..].iter().all(|&v| v.is_finite()),
            "valid centres still work: {s:?}"
        );
        // A NaN in the curve propagates rather than silently shortening a window.
        let with_nan = vec![0.0, f64::NAN, 0.0];
        let got = smooth_octave(&[100.0, 1000.0, 10_000.0], &with_nan, 1.0);
        assert!(
            got.iter().any(|&v| v.is_nan()),
            "NaN must be visible: {got:?}"
        );
        // Zero and negative window widths are refused, not trusted.
        assert!(smooth_octave(&f, &g, 0.0).iter().all(|&v| v.is_nan()));
        assert!(smooth_octave(&f, &g, f64::NAN).iter().all(|&v| v.is_nan()));
        // The quantile is nearest rank, so the value it returns is one that was measured.
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 0.5), 2.0);
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 0.9), 4.0);
        assert_eq!(quantile(&[4.0, 1.0, 3.0, 2.0], 0.0), 1.0);
        assert_eq!(quantile(&[1.0], 0.9), 1.0);
        assert!(quantile(&[], 0.9).is_nan());
        assert!(quantile(&[f64::NAN, f64::INFINITY], 0.9).is_nan());
        // A quantile outside [0,1] is a bug in the caller, not a silent clamp.
        assert!(quantile(&[1.0, 2.0], 1.5).is_nan());
        // `auto` reads a sample of nothing as "no information" and takes the floor.
        assert_eq!(Policy::auto(&[], &[]).lambda_db, LAMBDA_FLOOR_DB);
    }
}
