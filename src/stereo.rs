//! Left and right: what a pair of ear measurements agrees about, and how far a correction may
//! pull the two ears apart.
//!
//! The shipped convolver is one filter, loaded on both channels — that is what a PipeWire
//! `afir` with a mono impulse response *is*, and it is what the reference program derives. So a
//! model measured per ear contributes an average curve to the DSP and its two ears become two
//! dashed traces on the figure: the difference between them is neither corrected nor mentioned.
//!
//! That difference is not small on this database. Measured on the shipped captures:
//!
//! * **DT 770 Pro (80 Ω)** — the two ears' median levels differ by 1.27 dB, and their shapes by up
//!   to 7.72 dB at 15 110 Hz, most of which does not survive a sixth-octave window (so it is one
//!   ear's own wiggle, not a difference between the units).
//! * **FT1 Pro** — medians within 0.05 dB, shapes 4.10 dB apart at 3 839 Hz, and that one *does*
//!   survive the window: a broad difference is a pairing difference.
//! * **JBL 770NC** — captured as a single curve, so there is nothing to compare.
//!
//! A single filter leaves all of that standing: the delivered left/right difference after the
//! correction is exactly the difference that was measured, because the same curve is added to both
//! sides. [`trim`] is the alternative — a per-ear additive correction whose **mean is the shipped
//! curve** (so the centre of the image does not move) and whose **difference cancels the pair's
//! own difference** (so the image stops leaning).
//!
//! The cancellation is not free, and it is the same bargain [`crate::fit`] already strikes: part of
//! a left/right difference is the two units being different, which is real and worth cancelling,
//! and part of it is one ear sitting 2 mm differently in a coupler, which is a fixture and gets
//! cancelled into a spike only the other ear needed. So the shape part of the mismatch goes
//! through the same knee as everything else — see [`trim`] for where the level part is handled and
//! why it is not subject to the same doubt.
//!
//! The quantities here are named for what they are and attributed to what they come from:
//!
//! * Band edges and nominal names are IEC 61260-1, reused from [`crate::assess`].
//! * The reading of a correlation coefficient (+1 identical, 0 uncorrelated, −1 perfectly
//!   anticorrelated) is the standards-body definition: ASA/ANSI acoustical terminology,
//!   "cross correlation; interaural cross correlation", annotation 2.
//! * The per-ear **mismatch**, the **narrow fraction**, the **level offset** and the **skew bound**
//!   below are HyprEQ's, chosen for this database and documented with the numbers that picked
//!   them. Nothing in the standards reviewed defines a "correct" amount of left/right difference
//!   between two headphones, and none of these thresholds should be read as if one did.

use crate::assess;
use crate::fft::{rfft, rfftfreq};
use crate::fit;
use crate::pchip::Pchip;

/// A curve: frequency in Hz and gain in dB, ascending in frequency. The shape every curve in this
/// crate takes; named here because a pair of them is what a per-ear filter is built from and the
/// bare type is unreadable in a signature.
pub type Curve = Vec<(f64, f64)>;

/// The left and right curves a [`Trim`] produces from one shared curve.
pub type CurvePair = (Curve, Curve);

/// What `--stereo` asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// One filter from the average curve, and nothing said about the ears. Today's behaviour, and
    /// the default: the report and the bytes are the ones the reference program produces.
    Off,
    /// The same filter, plus the block that says what the two ears differ by. Audio unchanged.
    Report,
    /// One filter per ear, applied to a stereo file, bounded by the skew ceiling. Requires a model
    /// measured per ear and a file with exactly two channels.
    Split,
}

impl Mode {
    /// The whole vocabulary, in the order the usage message lists it.
    pub const CHOICES: [&'static str; 3] = ["off", "report", "split"];

    /// The words this mode answers to. The CLI validates against [`Mode::CHOICES`] with argparse's
    /// own message, so this is the library's courtesy: trimmed and case-folded, `None` for anything
    /// else rather than an error carrying a message no caller would print.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Mode::Off),
            "report" => Some(Mode::Report),
            "split" => Some(Mode::Split),
            _ => None,
        }
    }

    /// The word this mode prints as, and the one `parse` accepts for it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Off => "off",
            Mode::Report => "report",
            Mode::Split => "split",
        }
    }
}

/// Largest filter difference between the ears that `--stereo split` allows in one band, in dB,
/// when the operator does not name a bound.
///
/// Chosen against this database, not out of a book: at 6 dB the bound is slack everywhere on both
/// stereo captures except the DT 770's 15.1 kHz band, where the mismatch is 7.7 dB *and* narrow,
/// which is the case the regularizer is already refusing. It is therefore a ceiling on the damage
/// the flag can do to an image — a per-ear trim can lean a centre channel 9 dB to one side and
/// still sound "corrected" on each ear measured alone — rather than a claim about perception.
pub const DEFAULT_MAX_SKEW_DB: f64 = 6.0;

/// Below this peak shape mismatch the ears are called [`Verdict::Matched`], in dB.
///
/// Ours. The shipped captures are quantised to 0.01 dB and their sixth-octave deviation is a third
/// of a dB, so this is deliberately far above measurement granularity: it is the level at which a
/// shared filter's leftover difference stops being a rounding detail of the pair.
pub const MATCH_CEILING_DB: f64 = 3.0;

/// Above this level difference between the ears, a pair whose shapes agree is [`Verdict::Level`]
/// rather than [`Verdict::Matched`], in dB.
///
/// Ours, and deliberately tighter than [`MATCH_CEILING_DB`]: a pure level difference has no
/// spectral excuse at all. It moves the centre of everything in the mix, and nothing about a
/// shared filter can put it back.
pub const LEVEL_FLOOR_DB: f64 = 1.0;

/// How many capture points a sixth-of-an-octave window has to cover before "is this feature narrow
/// or is it a shelf?" is a question the grid can answer.
///
/// This is the same scope note [`crate::assess`] prints for a short recording, in the frequency
/// domain: a third-octave band below about 8.7 FFT bins is attributed rather than measured, and a
/// sixth-octave window over fewer than three samples smooths nothing — it reads one capture point
/// and calls the result a feature. The shipped captures are spaced ~0.078 of an octave apart, so
/// their window covers **~2.1 points** and the question is not asked of them.
pub const RESOLVED_WINDOW_POINTS: f64 = 3.0;

/// A pair of ears, on the grid the model was captured at.
#[derive(Debug, Clone, PartialEq)]
pub struct Ears {
    /// Capture frequencies in Hz, ascending.
    pub f: Vec<f64>,
    /// Left ear, dB.
    pub l: Vec<f64>,
    /// Right ear, dB.
    pub r: Vec<f64>,
    /// The column the single-filter path fits, which is what `--stereo` is being asked about.
    #[cfg(test)]
    pub avg: Vec<f64>,
}

impl Ears {
    /// Read the per-ear columns out of a raw table.
    ///
    /// The convention is the figure's, and it predates this module: a row with four or more
    /// columns is `[freq, left, right, average]`, and the last column is what the DSP fits. A
    /// two-column table has no ears to compare, and returns `None` — which is not an error, it is
    /// the common case.
    pub fn from_raw_pts(pts: &[Vec<f64>]) -> Option<Ears> {
        if pts.is_empty() || !pts.iter().all(|p| p.len() >= 3) {
            return None;
        }
        let col = |i: usize| -> Vec<f64> { pts.iter().map(|p| p[i]).collect() };
        let e = Ears {
            f: col(0),
            l: col(1),
            r: col(2),
            #[cfg(test)]
            avg: pts
                .iter()
                .map(|p| {
                    if p.len() >= 4 {
                        *p.last().unwrap()
                    } else {
                        (p[1] + p[2]) / 2.0
                    }
                })
                .collect(),
        };
        (e.l.len() >= 2).then_some(e)
    }

    /// Number of capture points.
    pub fn len(&self) -> usize {
        self.f.len()
    }

    /// True when there is nothing on the axis. Present because [`Ears::len`] is.
    pub fn is_empty(&self) -> bool {
        self.f.is_empty()
    }

    /// The difference as captured: `left − right`, absolute, in dB.
    ///
    /// This is what the listener is offered after a shared filter, and what [`trim`] cancels.
    pub fn total_db(&self) -> Vec<f64> {
        self.l.iter().zip(&self.r).map(|(&l, &r)| l - r).collect()
    }

    /// The difference with each ear referred to its own median level, in dB.
    ///
    /// The shape question, kept separate from the level question because they deserve different
    /// treatment: a capture made with the right ear's analyser gain 94 dB off has a 94 dB *total*
    /// difference and no shape difference at all, and reading it the other way round would call a
    /// level bookkeeping error a pairing defect. Referred to the median for the same reason the
    /// regularizer refers a measurement to its median ([`fit::reference_db`]): an absolute offset
    /// must not decide a policy.
    pub fn shape_db(&self) -> Vec<f64> {
        let offset = self.level_offset_db();
        self.total_db().iter().map(|d| d - offset).collect()
    }

    /// The pair's level difference: the median of one ear minus the median of the other, in dB.
    pub fn level_offset_db(&self) -> f64 {
        fit::median_db(&self.l) - fit::median_db(&self.r)
    }
}

/// What the mismatch analysis concluded about a pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The ears agree inside [`MATCH_CEILING_DB`] and are within [`LEVEL_FLOOR_DB`] of each other.
    Matched,
    /// Shapes agree, levels do not: the shared filter leaves everything leaning.
    Level,
    /// A difference above the ceiling. Whether it is the two units or one ear in a coupler is a
    /// question [`Mismatch::resolved`] says whether this capture can even be asked.
    Mismatch,
}

impl Verdict {
    /// The word printed in the report.
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Matched => "matched",
            Verdict::Level => "level difference, same shape",
            Verdict::Mismatch => "mismatch",
        }
    }
}

/// The summary of one pair of ear measurements.
#[derive(Debug, Clone, Copy)]
pub struct Mismatch {
    /// Median left minus median right, dB. Positive means the left ear captured louder.
    pub level_offset_db: f64,
    /// Largest shape mismatch, signed as `left − right`.
    pub peak_db: f64,
    /// Frequency of [`Self::peak_db`] in Hz.
    pub peak_hz: f64,
    /// [`Self::peak_hz`] named by the IEC 61260-1 nominal value of its third-octave band.
    pub nominal_hz: f64,
    /// Root-mean-square of the shape mismatch, dB. A spread, not a level.
    pub spread_db: f64,
    /// Spacing of the capture grid, in octaves (median of the log2 differences).
    pub grid_oct: f64,
    /// How many capture points a sixth-of-an-octave window covers on this grid.
    ///
    /// Below [`RESOLVED_WINDOW_POINTS`] the curve cannot distinguish a spike from a shelf — every
    /// feature is one sample and whatever the interpolant makes of it — and [`Self::resolved`] is
    /// false. Both shipped stereo captures sit above 2.1.
    pub window_points: f64,
    /// Whether this capture resolves narrow and broad apart.
    pub resolved: bool,
    /// Correlation of the two ears' wiggles, or `None` when one ear has no wiggle to correlate.
    ///
    /// `None` is not zero. Zero means the two ears' features are independent, which is a finding;
    /// `None` means there was nothing to compare, which is a flat curve or a single point.
    pub correlation: Option<f64>,
    /// The call, from [`MATCH_CEILING_DB`] and [`LEVEL_FLOOR_DB`]. Narrowness plays no part in it,
    /// because [`Mismatch::resolved`] says the shipped grids cannot measure narrowness.
    pub verdict: Verdict,
}

/// Compare a pair of ears.
pub fn analyse(e: &Ears) -> Mismatch {
    let shape = e.shape_db();
    let n = e.f.len().min(shape.len());
    let mut peak_db = 0.0_f64;
    let mut peak_hz = 0.0_f64;
    let mut sq = 0.0_f64;
    let mut count = 0usize;
    for (&d, &hz) in shape[..n].iter().zip(&e.f[..n]) {
        if !d.is_finite() {
            continue;
        }
        if d.abs() > peak_db.abs() {
            peak_db = d;
            peak_hz = hz;
        }
        sq += d * d;
        count += 1;
    }
    let spread_db = if count > 0 {
        (sq / count as f64).sqrt()
    } else {
        0.0
    };

    // The grid's own resolution, stated next to every claim about feature width.
    let mut gaps: Vec<f64> = (1..n)
        .map(|i| (e.f[i] / e.f[i - 1]).log2())
        .filter(|g| g.is_finite() && *g > 0.0)
        .collect();
    let grid_oct = if gaps.is_empty() {
        f64::NAN
    } else {
        gaps.sort_unstable_by(|a, b| a.total_cmp(b));
        gaps[gaps.len() / 2]
    };
    let window_points = if grid_oct.is_finite() && grid_oct > 0.0 {
        (1.0 / 6.0) / grid_oct
    } else {
        0.0
    };

    Mismatch {
        level_offset_db: e.level_offset_db(),
        peak_db,
        peak_hz,
        nominal_hz: assess::nearest_nominal(peak_hz),
        spread_db,
        grid_oct,
        window_points,
        resolved: window_points >= RESOLVED_WINDOW_POINTS,
        correlation: correlation(e),
        verdict: classify(peak_db.abs(), e.level_offset_db().abs()),
    }
}

/// The call itself, split out so the thresholds are testable without building a whole curve.
///
/// It stops short of the interesting question on purpose. Distinguishing "the two units are
/// different" from "one ear sat differently in the coupler" needs a feature's width, and a width
/// needs a grid that can see one: at the shipped ~0.078-octave spacing a sixth-octave window covers
/// ~2.1 samples, so the measurement says *how much* the ears differ and *where*, and not *why*. The
/// amount that gets cancelled is still decided band by band, by [`crate::fit`]'s knee exactly as it
/// is for a single curve.
fn classify(peak_db: f64, level_db: f64) -> Verdict {
    if peak_db <= MATCH_CEILING_DB {
        if level_db > LEVEL_FLOOR_DB {
            Verdict::Level
        } else {
            Verdict::Matched
        }
    } else {
        Verdict::Mismatch
    }
}

/// Pearson's coefficient between the two ears' wiggles, each measured against its own
/// sixth-octave smoothed self.
///
/// The shapes are correlated by construction — two ears of one headphone share their bass shelf —
/// so correlating the curves measures the shelf. Correlating what is *left* after each ear's own
/// envelope answers the question this module cares about: are the two ears' features the same
/// feature, or two independent accidents of placement? `None` when either side has no spread,
/// which the caller must not print as a `0.00` that would read as "orthogonal".
pub fn correlation(e: &Ears) -> Option<f64> {
    let n = e.f.len();
    if n < 3 {
        return None;
    }
    let a = residual_of(&e.f, &e.l, n);
    let b = residual_of(&e.f, &e.r, n);
    let mut ma = 0.0;
    let mut mb = 0.0;
    let mut cnt = 0usize;
    for i in 0..n {
        if a[i].is_finite() && b[i].is_finite() {
            ma += a[i];
            mb += b[i];
            cnt += 1;
        }
    }
    if cnt < 3 {
        return None;
    }
    ma /= cnt as f64;
    mb /= cnt as f64;
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..n {
        if a[i].is_finite() && b[i].is_finite() {
            let (xa, xb) = (a[i] - ma, b[i] - mb);
            num += xa * xb;
            da += xa * xa;
            db += xb * xb;
        }
    }
    if da <= 0.0 || db <= 0.0 {
        return None;
    }
    Some(num / (da.sqrt() * db.sqrt()))
}

fn residual_of(f: &[f64], g: &[f64], n: usize) -> Vec<f64> {
    let smooth = fit::smooth_octave(f, g, 1.0 / 6.0);
    (0..n).map(|i| g[i] - smooth[i]).collect()
}

/// What a per-ear trim did, per knot of the curve the filter is built from.
#[derive(Debug, Clone)]
pub struct Trim {
    /// Knot frequencies in Hz, shared by both ears.
    pub f: Vec<f64>,
    /// Added to the left ear's curve, dB.
    pub l: Vec<f64>,
    /// Added to the right ear's curve, dB.
    pub r: Vec<f64>,
    /// How much of the level difference was cancelled, dB. Always the pair's full level offset.
    pub level_db: f64,
    /// Knots where the skew bound was the last word rather than the measurement.
    pub bound_bands: usize,
    /// Knots where the notch ceiling was re-applied after the trim and moved at least one ear's
    /// curve — distinct knots, so a knot it cut back on both sides counts once.
    ///
    /// Zero is the normal case and the one the design intends: the trim is applied as `±`, so the
    /// pair's mean is the shared curve. The ceiling is the last word on a notch band, though, so a
    /// trim that pushed one ear's fill past `safe_cap_db` gets cut back on that side alone, and the
    /// middle of the image moves by half of it. Counted here because a report that said "centre
    /// held" over such a curve would be describing the design rather than the result.
    pub cap_bites: usize,
}

impl Trim {
    /// Record where the ceiling bit, so [`report_lines`] can say the centre moved.
    #[must_use]
    pub fn with_cap_bites(mut self, bites: usize) -> Self {
        self.cap_bites = bites;
        self
    }

    /// The filter difference between the ears at each knot: `left − right`, dB.
    pub fn skew_db(&self) -> Vec<f64> {
        self.l.iter().zip(&self.r).map(|(&l, &r)| l - r).collect()
    }

    /// Largest filter difference between the ears, and the knot frequency it happens at.
    pub fn worst(&self) -> (f64, f64) {
        let mut best = (0.0_f64, 0.0_f64);
        for (i, &s) in self.skew_db().iter().enumerate() {
            if s.abs() > best.0.abs() {
                best = (s, self.f[i]);
            }
        }
        best
    }

    /// The two curves the filter pair is built from, given the curve both ears would have shared.
    ///
    /// Their **mean is `ctrl` at every knot**, exactly: the trim is applied as `±`, so centring the
    /// image costs nothing in level and a listener's centre stays where the model put it.
    pub fn curves(&self, ctrl: &Curve) -> CurvePair {
        let tl = Pchip::new(&self.f, &self.l);
        let tr = Pchip::new(&self.f, &self.r);
        // The trim is defined on the shared curve's own knots, so this reads its own nodes and
        // returns them exactly; one interpolant each, not one per knot.
        let hz: Vec<f64> = ctrl.iter().map(|&(x, _)| x).collect();
        let (gl, gr) = (tl.eval_grid(&hz), tr.eval_grid(&hz));
        let out = |add: &[f64]| -> Curve {
            ctrl.iter()
                .zip(add)
                .map(|(&(h, g), &a)| (h, g + a))
                .collect()
        };
        (out(&gl), out(&gr))
    }
}

/// The per-ear trim for a pair: what each ear's filter gets added to the shared curve.
///
/// Written as `left += t`, `right −= t` so the pair's mean stays on the shipped curve. With `D`
/// the captured difference and `level` its median, the cancellation asked for at a knot is
/// `−(D − level)/2 − level/2`: the shape part is regularized with ear-swap antisymmetry by `1 / (1 + β)`,
/// so a band the knee disbelieves is left leaning exactly as much as today's single filter leaves it,
/// and a band the knee believes is cancelled; the level part is applied in full and unasked, because
/// a consistent offset between two ears is the least doubtful number in the file — it is not a claim
/// about a feature, it is the pair's balance, and the knee exists to doubt features.
///
/// `max_skew_db` then bounds the *result*: no knot may differ by more than that, and the excess is
/// taken off both sides equally so the mean still does not move. [`Trim::bound_bands`] counts the
/// knots where that happened, and the report says so — a bound that silently refused work would be
/// the same kind of lie as a limiter that reported nothing while engaging.
pub fn trim(e: &Ears, ctrl: &Curve, policy: &fit::Policy, max_skew_db: f64) -> Trim {
    let n = e.f.len();
    let shape = e.shape_db();
    let level = e.level_offset_db();

    // Doubt is a function of the shape mismatch's own deviation, on the same windows the
    // regularizer uses everywhere else.
    let dev = fit::deviation(&e.f, &shape, policy.residual_octave, policy.beta_octave);

    let ctrl_f: Vec<f64> = ctrl.iter().map(|&(x, _)| x).collect();
    let mut t_l = vec![0.0_f64; ctrl_f.len()];
    let mut t_r = vec![0.0_f64; ctrl_f.len()];
    // Translation invariance is what lets the balance be handled separately from the shape: a
    // monotone cubic through `shape + level` is the same curve lifted by `level`, so reading the
    // shape at a knot and adding the balance gives what reading the captured difference at that
    // knot would have given.
    let shape_interp = Pchip::new(&e.f[..n], &shape[..n]);
    let dev_interp = Pchip::new(&e.f[..n], &dev[..n]);

    let shape_at = shape_interp.eval_grid(&ctrl_f);
    let dev_at = dev_interp.eval_grid(&ctrl_f);

    let mut bound_bands = 0usize;
    for i in 0..ctrl_f.len() {
        let s = shape_at[i];
        let beta = fit::beta(dev_at[i], policy);
        // The shape part, doubted band by band with ear-swap antisymmetry:
        // regularized by 1 / (1 + beta) in dB, so L and R trims remain exact inverses
        // and swapping L and R leads negates the trim identically.
        let mut half = if beta <= 0.0 {
            -s / 2.0
        } else if !beta.is_finite() {
            0.0
        } else {
            (-s / 2.0) / (1.0 + beta)
        };
        // The balance, in full, split down the middle.
        half -= level / 2.0;
        let cap = (max_skew_db / 2.0).max(0.0);
        if half.abs() > cap {
            bound_bands += 1;
            half = half.clamp(-cap, cap);
        }
        t_l[i] = half;
        t_r[i] = -half;
    }

    let lim = if max_skew_db.is_finite() {
        max_skew_db.max(0.0)
    } else if max_skew_db.is_sign_positive() {
        f64::INFINITY
    } else {
        0.0
    };

    Trim {
        f: ctrl_f,
        l: t_l,
        r: t_r,
        level_db: level.clamp(-lim, lim),
        bound_bands,
        cap_bites: 0,
    }
}

/// The report block. `trim` is `Some` when a per-ear pair was actually built, `None` for a report
/// of a shared filter, which is the difference between "here is what is wrong" and "here is what I
/// did about it".
pub fn report_lines(m: &Mismatch, points: usize, t: Option<&Trim>) -> Vec<String> {
    let mut lines = vec![format!(
        "=== Ears ({} capture points measured per ear) ===",
        points
    )];
    lines.push(format!(
        "  Level difference  : {:+.2} dB left vs right ({})",
        m.level_offset_db,
        if m.level_offset_db.abs() > LEVEL_FLOOR_DB {
            "the centre leans"
        } else {
            "inside a dB"
        }
    ));
    if m.peak_db.abs() == 0.0 || m.peak_hz <= 0.0 {
        lines.push(format!(
            "  Shape mismatch    : none (identical L/R response), spread {:.2} dB",
            m.spread_db
        ));
    } else {
        lines.push(format!(
            "  Shape mismatch    : {:+.2} dB @ {:.0} Hz (band {}), spread {:.2} dB",
            m.peak_db,
            m.peak_hz,
            assess::band_name(m.nominal_hz),
            m.spread_db
        ));
    }
    lines.push(format!("  Reads as          : {}", m.verdict.label()));
    if !m.resolved {
        lines.push(format!(
            "  Resolution        : a sixth of an octave is {:.1} capture points here (grid {:.3} oct), so one tenth of an octave is one sample and the reason for the difference is not knowable from this curve",
            m.window_points, m.grid_oct
        ));
    }
    match m.correlation {
        Some(c) => lines.push(format!(
            "  Feature match     : r = {c:+.3} (+1 the same feature, 0 independent accidents)"
        )),
        None => {
            lines.push("  Feature match     : n/a (a curve with nothing to correlate)".to_string())
        }
    }
    if let Some(t) = t {
        let (worst, hz) = t.worst();
        if worst.abs() == 0.0 || hz <= 0.0 {
            lines.push(format!(
                "  Per-ear trim      : balance {:+.2} dB cancelled, no per-band filter difference",
                -t.level_db
            ));
        } else {
            lines.push(format!(
                "  Per-ear trim      : balance {:+.2} dB cancelled, filter difference up to {:+.2} dB @ {:.0} Hz",
                -t.level_db, worst, hz
            ));
        }
        if t.bound_bands > 0 {
            lines.push(format!(
                "  Bound             : {} knot(s) held to the skew ceiling instead of cancelling what was measured",
                t.bound_bands
            ));
        }
        // The claim the design makes and the result this run got, printed as whichever it is.
        lines.push(if t.cap_bites == 0 {
            "  Centre            : held (the two curves average to the shared curve at every knot)"
                .to_string()
        } else {
            format!(
                "  Centre            : moved in {} knot(s) — the notch ceiling was applied after the trim and cut one or both ears back",
                t.cap_bites
            )
        });
    }
    lines
}

/// Low-frequency dichotic interaural group-delay (IGD) lateral-shift threshold, in microseconds,
/// reported by Møller, Minnaar, Olesen, Christensen & Plogsties (*"On the Audibility of All-Pass
/// Phase in Electroacoustical Transfer Functions,"* JAES 55(3), pp. 115–134, March 2007).
///
/// Above ~35 µs of low-frequency interaural group-delay mismatch, listeners hear a lateral shift
/// of the stereo image that is independent of the audio signal.
pub const DICHOTIC_IGD_THRESHOLD_US: f64 = 35.0;

/// Lower frequency bound (Hz) for the low-frequency ITD / dichotic IGD audit band.
pub const IGD_BAND_LO_HZ: f64 = 50.0;

/// Upper frequency bound (Hz) for the low-frequency ITD / dichotic IGD audit band.
pub const IGD_BAND_HI_HZ: f64 = 2000.0;

/// Measured interaural group-delay difference between a stereo filter pair `(ir_l, ir_r)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterauralGroupDelay {
    /// Peak `|GD_L(f) - GD_R(f)|` across `[band_lo_hz, band_hi_hz]`, in microseconds.
    pub max_igd_us: f64,
    /// Frequency at which `max_igd_us` occurs, in Hz.
    pub peak_hz: f64,
    /// Lower edge of the evaluated frequency band, in Hz.
    pub band_lo_hz: f64,
    /// Upper edge of the evaluated frequency band, in Hz.
    pub band_hi_hz: f64,
    /// `true` when `max_igd_us > DICHOTIC_IGD_THRESHOLD_US` (35 µs, Møller et al. 2007).
    pub exceeds_threshold: bool,
}

/// Measure interaural group-delay mismatch `|GD_L(f) - GD_R(f)|` in microseconds between two
/// impulse responses over `[band_lo_hz, band_hi_hz]`.
///
/// Computes the cross-spectrum `H_L(f) * H_R^*(f)` so common-mode phase wraps cancel before
/// phase unwrapping and differentiation (`-1/(2π) · d(Δφ)/df · 1e6`). Returns `None` if inputs
/// are empty, sample rate is zero, or the frequency band has fewer than two usable bins.
#[must_use]
pub fn measure_interaural_group_delay(
    ir_l: &[f64],
    ir_r: &[f64],
    fs: u32,
    band_lo_hz: f64,
    band_hi_hz: f64,
) -> Option<InterauralGroupDelay> {
    if ir_l.is_empty()
        || ir_r.is_empty()
        || fs == 0
        || !band_lo_hz.is_finite()
        || !band_hi_hz.is_finite()
        || band_lo_hz <= 0.0
        || band_hi_hz <= band_lo_hz
        || ir_l.iter().any(|v| !v.is_finite())
        || ir_r.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let nyq = (fs as f64) * 0.5;
    let hi = band_hi_hz.min(nyq);
    if hi <= band_lo_hz {
        return None;
    }
    let max_len = ir_l.len().max(ir_r.len());
    let n_fft = (max_len.saturating_mul(2)).next_power_of_two().max(16384);
    let hl = rfft(ir_l, n_fft);
    let hr = rfft(ir_r, n_fft);
    let freqs = rfftfreq(n_fft, fs as f64);
    let n_bins = freqs.len();
    if n_bins < 3 {
        return None;
    }

    let max_mag_l = hl.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
    let max_mag_r = hr.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
    if max_mag_l <= 0.0 || max_mag_r <= 0.0 {
        return None;
    }
    let floor_l = max_mag_l * 1e-12;
    let floor_r = max_mag_r * 1e-12;

    // Cross-spectrum phase Δφ(f) = arg(H_L * H_R^*), unwrapped bin-by-bin.
    let mut dphi = vec![0.0_f64; n_bins];
    let mut valid = vec![false; n_bins];
    let mut prev_raw = 0.0_f64;
    let mut carry = 0.0_f64;
    let mut started = false;
    for k in 0..n_bins {
        let ml = hl[k].norm();
        let mr = hr[k].norm();
        if ml > floor_l && mr > floor_r {
            let cross = hl[k] * hr[k].conj();
            let raw = cross.im.atan2(cross.re);
            if started {
                let mut delta = raw - prev_raw;
                while delta > std::f64::consts::PI {
                    delta -= std::f64::consts::TAU;
                    carry -= std::f64::consts::TAU;
                }
                while delta < -std::f64::consts::PI {
                    delta += std::f64::consts::TAU;
                    carry += std::f64::consts::TAU;
                }
            }
            prev_raw = raw;
            started = true;
            dphi[k] = raw + carry;
            valid[k] = true;
        } else {
            dphi[k] = prev_raw + carry;
        }
    }

    let mut max_igd_us = 0.0_f64;
    let mut peak_hz = 0.0_f64;
    let mut measured = 0usize;
    for k in 1..n_bins - 1 {
        let f = freqs[k];
        if f < band_lo_hz || f > hi {
            continue;
        }
        if !valid[k - 1] || !valid[k] || !valid[k + 1] {
            continue;
        }
        let df = freqs[k + 1] - freqs[k - 1];
        if df <= 0.0 {
            continue;
        }
        let igd_us = (-(dphi[k + 1] - dphi[k - 1]) / df / std::f64::consts::TAU * 1e6).abs();
        if igd_us.is_finite() {
            measured += 1;
            if measured == 1 || igd_us > max_igd_us {
                max_igd_us = igd_us;
                peak_hz = f;
            }
        }
    }
    if measured == 0 {
        return None;
    }
    Some(InterauralGroupDelay {
        max_igd_us,
        peak_hz,
        band_lo_hz,
        band_hi_hz: hi,
        exceeds_threshold: max_igd_us > DICHOTIC_IGD_THRESHOLD_US,
    })
}

/// Format the dichotic IGD measurement against Møller et al. (2007)'s 35 µs threshold.
#[must_use]
pub fn igd_report_line(igd: &InterauralGroupDelay) -> String {
    let status = if igd.exceeds_threshold {
        format!(
            "exceeds {:.0} µs dichotic lateral-shift threshold (Møller et al. 2007; use --stereo-phase-match)",
            DICHOTIC_IGD_THRESHOLD_US
        )
    } else {
        format!(
            "within {:.0} µs dichotic lateral-shift threshold (Møller et al. 2007)",
            DICHOTIC_IGD_THRESHOLD_US
        )
    };
    format!(
        "  Interaural GD     : {:.1} µs @ {:.0} Hz ({:.0}–{:.0} Hz) — {status}",
        igd.max_igd_us, igd.peak_hz, igd.band_lo_hz, igd.band_hi_hz
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::Policy;

    #[test]
    fn the_mode_vocabulary_and_the_variants_cover_each_other() {
        // The CLI rejects anything outside CHOICES before this is consulted, so an entry with no
        // variant behind it would be a word the usage message offers and the program cannot honour.
        for name in Mode::CHOICES {
            let m = Mode::parse(name).unwrap_or_else(|| panic!("{name} is offered but not parsed"));
            assert_eq!(m.as_str(), name, "{name} does not round-trip");
        }
        assert_eq!(Mode::CHOICES.len(), 3);
        assert_eq!(Mode::parse("  SPLIT "), Some(Mode::Split));
        assert_eq!(Mode::parse("both"), None);
        assert_eq!(Mode::parse(""), None);
    }

    /// A capture grid: log-spaced, 20 Hz to 20 kHz, like the shipped ones.
    fn grid(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 20.0_f64 * (1000.0_f64).powf(i as f64 / (n as f64 - 1.0)))
            .collect()
    }

    /// A Gaussian feature in log frequency: `sigma_oct` is its width in octaves.
    fn bump(f: &[f64], hz: f64, depth_db: f64, sigma_oct: f64) -> Vec<f64> {
        f.iter()
            .map(|&x| depth_db * (-0.5 * ((x / hz).log2() / sigma_oct).powi(2)).exp())
            .collect()
    }

    fn pair(f: &[f64], l: &[f64], r: &[f64]) -> Ears {
        Ears {
            f: f.to_vec(),
            l: l.to_vec(),
            r: r.to_vec(),
            avg: l.iter().zip(r).map(|(&a, &b)| (a + b) / 2.0).collect(),
        }
    }

    fn flat(f: &[f64]) -> Vec<f64> {
        vec![0.0; f.len()]
    }

    /// The shared curve the pair would have been corrected with: no correction at all, so every
    /// number in the report is the difference that survives a flat filter.
    fn no_ctrl(f: &[f64]) -> Vec<(f64, f64)> {
        f.iter().map(|&h| (h, 0.0_f64)).collect()
    }

    /// What the listener is offered at the two ears after a pair of curves went through the
    /// measurement: `delivered left − delivered right`.
    fn delivered(e: &Ears, gl: &[(f64, f64)], gr: &[(f64, f64)]) -> Vec<f64> {
        let ll = Pchip::new(
            &gl.iter().map(|p| p.0).collect::<Vec<_>>(),
            &gl.iter().map(|p| p.1).collect::<Vec<_>>(),
        )
        .eval_grid(&e.f);
        let rr = Pchip::new(
            &gr.iter().map(|p| p.0).collect::<Vec<_>>(),
            &gr.iter().map(|p| p.1).collect::<Vec<_>>(),
        )
        .eval_grid(&e.f);
        (0..e.f.len())
            .map(|i| (e.l[i] + ll[i]) - (e.r[i] + rr[i]))
            .collect()
    }

    #[test]
    fn from_raw_pts_reads_the_figures_own_column_convention() {
        // [freq, left, right, average], and the average is the file's own column - which is rounded
        // to the hundredth of a dB in the shipped data, so recomputing it here would be a second,
        // slightly different answer to a question the model already answered.
        let pts = vec![
            vec![20.0, 3.82, 1.27, 2.54],
            vec![1000.0, -1.0, -1.4, -1.21],
            vec![20000.0, -7.16, -6.62, -6.89],
        ];
        let e = Ears::from_raw_pts(&pts).expect("four columns is a pair");
        assert_eq!(e.len(), 3);
        assert!(!e.is_empty());
        assert_eq!(e.l, vec![3.82, -1.0, -7.16]);
        assert_eq!(e.r, vec![1.27, -1.4, -6.62]);
        assert_eq!(e.avg, vec![2.54, -1.21, -6.89]);

        // Three columns [freq, left, right]: average is computed as mean of left and right.
        let three = vec![
            vec![20.0, 3.82, 1.28],
            vec![1000.0, -1.0, -1.4],
            vec![20000.0, -7.16, -6.62],
        ];
        let e3 = Ears::from_raw_pts(&three).expect("three columns is a pair");
        assert_eq!(e3.len(), 3);
        assert_eq!(e3.l, vec![3.82, -1.0, -7.16]);
        assert_eq!(e3.r, vec![1.28, -1.4, -6.62]);
        assert_eq!(e3.avg.len(), 3);
        assert!((e3.avg[0] - 2.55).abs() < 1e-12);
        assert!((e3.avg[1] - (-1.2)).abs() < 1e-12);
        assert!((e3.avg[2] - (-6.89)).abs() < 1e-12);

        // Two columns is the common case, not a failure: a model measured as one curve.
        let mono = vec![vec![20.0, 3.82], vec![1000.0, -1.0]];
        assert!(Ears::from_raw_pts(&mono).is_none());
        assert!(Ears::from_raw_pts(&[]).is_none());

        // Ragged rows where later rows have fewer than 4 columns must not panic.
        let ragged = vec![vec![20.0, 3.82, 1.27, 2.54], vec![1000.0, -1.0]];
        assert!(Ears::from_raw_pts(&ragged).is_none());
    }

    #[test]
    fn ear_swap_is_strictly_antisymmetric() {
        let f = grid(61);
        let bump_l = bump(&f, 3000.0, 4.0, 0.3);
        let bump_r = bump(&f, 5000.0, -3.0, 0.3);
        let e_lr = pair(&f, &bump_l, &bump_r);
        let e_rl = pair(&f, &bump_r, &bump_l);
        let ctrl = no_ctrl(&f);
        let policy = Policy::default();
        let t_lr = trim(&e_lr, &ctrl, &policy, 12.0);
        let t_rl = trim(&e_rl, &ctrl, &policy, 12.0);
        for i in 0..ctrl.len() {
            assert!(
                (t_lr.l[i] - t_rl.r[i]).abs() < 1e-12,
                "L trim failed swap antisymmetry at index {i}: {} vs {}",
                t_lr.l[i],
                t_rl.r[i]
            );
            assert!(
                (t_lr.r[i] - t_rl.l[i]).abs() < 1e-12,
                "R trim failed swap antisymmetry at index {i}: {} vs {}",
                t_lr.r[i],
                t_rl.l[i]
            );
        }
    }

    #[test]
    fn a_mirrored_pair_reports_nothing_to_fix() {
        let f = grid(61);
        let shape = bump(&f, 4000.0, -6.0, 0.25);
        let e = pair(&f, &shape, &shape);
        let m = analyse(&e);
        assert_eq!(m.verdict, Verdict::Matched);
        assert!(m.peak_db.abs() < 1e-12, "peak {}", m.peak_db);
        assert!(m.level_offset_db.abs() < 1e-12);
        assert!(m.spread_db < 1e-12);
        // Both ears carry the identical wiggle, which is the maximally correlated answer.
        assert!(m.correlation.is_some());
        assert!(
            (m.correlation.unwrap() - 1.0).abs() < 1e-6,
            "r {:?}",
            m.correlation
        );
    }

    #[test]
    fn spread_db_is_the_true_rms_spread() {
        // RMS of [-2.0, 2.0, -2.0, 2.0] is exactly 2.0 dB.
        // Old bug divided by count instead of sqrt(count), yielding 2.0 / 2 = 1.0 dB.
        let f = vec![100.0, 200.0, 400.0, 800.0];
        let l = vec![-2.0, 2.0, -2.0, 2.0];
        let r = vec![0.0, 0.0, 0.0, 0.0];
        let e = pair(&f, &l, &r);
        let m = analyse(&e);
        assert!(
            (m.spread_db - 2.0).abs() < 1e-9,
            "expected RMS 2.0 dB, got {} dB",
            m.spread_db
        );
    }

    #[test]
    fn a_curve_with_no_wiggle_has_no_correlation_to_report() {
        // The trap this guards: 0.00 is a *finding* under the standards-body reading (+1 identical,
        // 0 uncorrelated, -1 anticorrelated). A flat pair has no features to compare, and printing
        // 0.00 for it would tell the reader the ears are independent when they are identical.
        let f = grid(61);
        let e = pair(&f, &flat(&f), &flat(&f));
        let m = analyse(&e);
        assert_eq!(m.correlation, None);
        assert_eq!(m.verdict, Verdict::Matched);
        let lines = report_lines(&m, e.len(), None);
        assert!(
            lines.iter().any(|l| l.contains("n/a")),
            "a pair with nothing to correlate must not print a number: {lines:?}"
        );
        assert!(
            !lines.iter().any(|l| l.contains("r = +0.000")),
            "a None printed as a zero: {lines:?}"
        );
    }

    #[test]
    fn an_absolute_offset_on_one_ear_is_balance_not_shape() {
        // Same measurement, analyser gain 94 dB off on one ear. That is a bookkeeping difference in
        // how the capture was saved, and it must read as balance, not as a 94 dB pairing defect -
        // the same unit-invariance rule fit::reference_db exists for.
        let f = grid(61);
        let shape = bump(&f, 4000.0, -6.0, 0.25);
        let right: Vec<f64> = shape.iter().map(|&g| g - 40.0).collect();
        let e = pair(&f, &shape, &right);
        let m = analyse(&e);
        assert!(
            (m.level_offset_db - 40.0).abs() < 1e-9,
            "offset {:?}",
            m.level_offset_db
        );
        assert!(m.peak_db.abs() < 1e-9, "shape read {:+.2} dB", m.peak_db);
        assert_eq!(m.verdict, Verdict::Level);
    }

    #[test]
    fn a_difference_above_the_ceiling_is_a_mismatch_however_it_is_shaped() {
        // A quarter-octave shelf and a twenty-fourth-octave spike, same 6 dB in the same band. They
        // are different problems - one is the two units, the other is one ear in a coupler - and on
        // this grid nothing can tell them apart, which is the finding: the verdict says mismatch and
        // the resolution line says why it stops there.
        let f = grid(121);
        for sigma in [0.25, 1.0 / 24.0] {
            let e = pair(&f, &flat(&f), &bump(&f, 4000.0, 6.0, sigma));
            let m = analyse(&e);
            assert!(m.peak_db.abs() > 5.0, "peak {:?}", m.peak_db);
            assert_eq!(
                m.verdict,
                Verdict::Mismatch,
                "sigma {sigma} read {:?}",
                m.verdict
            );
            assert!(
                !m.resolved,
                "a 121-point decade-and-a-half grid cannot see a width"
            );
        }

        // A difference inside the ceiling is not a mismatch, however sharp it is.
        let small = bump(&f, 4000.0, 2.0, 1.0 / 24.0);
        assert_eq!(
            analyse(&pair(&f, &flat(&f), &small)).verdict,
            Verdict::Matched
        );
    }

    #[test]
    fn narrowness_is_only_claimed_when_the_grid_can_see_it() {
        // The claim is tied to the grid, not to a taste for a threshold: a sixth-octave window has
        // to land on at least three capture points before "narrow" means anything, and that is a
        // spacing of a twentieth of an octave. The shipped captures are at ~0.078.
        let coarse = grid(121);
        let m_coarse = analyse(&pair(&coarse, &flat(&coarse), &flat(&coarse)));
        assert!(
            (m_coarse.grid_oct - 0.0827).abs() < 1e-3,
            "grid {:?}",
            m_coarse.grid_oct
        );
        assert!(
            (m_coarse.window_points - 2.01).abs() < 0.05,
            "window covers {:?} points",
            m_coarse.window_points
        );
        assert!(!m_coarse.resolved);

        let fine = grid(601);
        let m_fine = analyse(&pair(&fine, &flat(&fine), &flat(&fine)));
        assert!(m_fine.grid_oct < 0.02, "grid {:?}", m_fine.grid_oct);
        assert!(m_fine.resolved, "window {:?}", m_fine.window_points);

        // And the caveat is printed exactly when it is owed.
        let seen = report_lines(&m_coarse, coarse.len(), None);
        assert!(
            seen.iter().any(|l| l.starts_with("  Resolution        :")),
            "an unresolvable grid printed no scope note: {seen:?}"
        );
        let fine_lines = report_lines(&m_fine, fine.len(), None);
        assert!(
            !fine_lines
                .iter()
                .any(|l| l.starts_with("  Resolution        :")),
            "a grid that can see a width still apologised: {fine_lines:?}"
        );
    }

    #[test]
    fn the_worst_band_is_named_in_the_units_the_rest_of_the_report_uses() {
        let f = grid(121);
        // 3839 Hz, the FT1 Pro's real worst band, whose IEC 61260-1 nominal is 4 kHz.
        let r = bump(&f, 3839.0, 4.1, 0.25);
        let m = analyse(&pair(&f, &flat(&f), &r));
        assert!(
            m.peak_hz > 3400.0 && m.peak_hz < 4300.0,
            "peak hz {:?}",
            m.peak_hz
        );
        assert_eq!(m.nominal_hz, 4000.0);
        let lines = report_lines(&m, f.len(), None);
        let shape_line = lines
            .iter()
            .find(|l| l.starts_with("  Shape mismatch"))
            .expect("the shape line");
        assert!(shape_line.contains("4 kHz"), "{shape_line}");
    }

    #[test]
    fn the_feature_match_reads_the_same_feature_twice_and_two_accidents_once() {
        let f = grid(121);
        let wiggle = bump(&f, 4000.0, 6.0, 1.0 / 24.0);
        let other = bump(&f, 9000.0, 6.0, 1.0 / 24.0);

        // One feature, heard by both ears: perfectly correlated residuals.
        let same = analyse(&pair(&f, &wiggle, &wiggle));
        assert!((same.correlation.unwrap() - 1.0).abs() < 1e-6);

        // Two features in two places, on top of the shelf both ears share. The shelf is the point:
        // correlating the *curves* measures it and reports near +1 for a pair whose features have
        // nothing to do with each other, which is the opposite of the reading the report prints.
        // Correlating what is left after each ear's own envelope sees the two accidents for what
        // they are.
        let shelf: Vec<f64> = f
            .iter()
            .map(|&h| -6.0 * (1.0 / ((h / 120.0).log2().powi(2) + 1.0)).exp())
            .collect();
        let add =
            |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(&x, &y)| x + y).collect() };
        let l_both = add(&shelf, &wiggle);
        let r_both = add(&shelf, &other);
        let apart = analyse(&pair(&f, &l_both, &r_both));
        let r_apart = apart.correlation.unwrap();
        assert!(
            r_apart < 0.5,
            "two separate spikes on a shared shelf read r = {r_apart}"
        );

        // Same pair, both ears carrying the same feature: the shelf must not wash it out either.
        let together = analyse(&pair(&f, &l_both, &add(&shelf, &wiggle)));
        assert!(
            (together.correlation.unwrap() - 1.0).abs() < 1e-6,
            "a shared shelf hid a shared feature: {:?}",
            together.correlation
        );

        // The same feature with the leads swapped: the standard's -1.
        let anti = analyse(&pair(
            &f,
            &wiggle,
            &wiggle.iter().map(|&g| -g).collect::<Vec<_>>(),
        ));
        assert!((anti.correlation.unwrap() + 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_knee_that_believes_nothing_leaves_the_shape_and_fixes_the_balance() {
        // The limit the flag has to satisfy to be an opt-in rather than a different product: with
        // every band disbelieved, the delivered *shape* difference is exactly what a shared filter
        // leaves, because the trim's shape part has been shrunk to zero. The balance is not subject
        // to that doubt and is cancelled anyway.
        let f = grid(121);
        let shape = bump(&f, 4000.0, 6.0, 0.25);
        let l: Vec<f64> = shape.iter().map(|&g| g + 2.0).collect();
        let e = pair(&f, &l, &flat(&f));
        let ctrl = no_ctrl(&f);
        let t = trim(&e, &ctrl, &Policy::fixed(-90.0), f64::INFINITY);
        let (gl, gr) = t.curves(&ctrl);
        let d = delivered(&e, &gl, &gr);
        // Delivered difference = (level + shape) + 2·(−level/2) = shape: untouched.
        let want = e.shape_db();
        let worst = (0..f.len())
            .map(|i| (d[i] - want[i]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst < 0.05,
            "a disbelieved pair's shape difference moved by {worst:.3} dB: {:?}",
            d.iter().take(6).collect::<Vec<_>>()
        );
        // And the balance is gone. Median, not mean: the pair's balance is *defined* as the
        // difference of the two medians, so that is what the trim removes, and a one-sided bump
        // leaves a mean-minus-median residue of its own (0.37 dB on this fixture) which is the
        // shape, not a leaning centre. `d` equals `want` above, so the median check is the same
        // statement made about the whole curve rather than point by point.
        let mut sorted = d.clone();
        sorted.sort_unstable_by(|a, b| a.total_cmp(b));
        let mid = sorted.len() / 2;
        let median = if sorted.len() % 2 == 1 {
            sorted[mid]
        } else {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        };
        assert!(
            median.abs() < 0.05,
            "the centre still leans by {median:.3} dB (the capture was {} off)",
            e.level_offset_db()
        );
        assert_eq!(t.bound_bands, 0, "an unbounded run reported the bound");
    }

    #[test]
    fn a_knee_that_believes_everything_cancels_the_difference_it_was_shown() {
        let f = grid(121);
        let shape = bump(&f, 4000.0, 6.0, 0.25);
        let l: Vec<f64> = shape.iter().map(|&g| g + 2.0).collect();
        let e = pair(&f, &l, &flat(&f));
        let ctrl = no_ctrl(&f);
        let t = trim(&e, &ctrl, &Policy::fixed(90.0), f64::INFINITY);
        let (gl, gr) = t.curves(&ctrl);
        let d = delivered(&e, &gl, &gr);
        let worst = d.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(
            worst < 0.1,
            "a trusted pair still differs by {worst:.3} dB: {:?}",
            d.iter().take(6).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_ceiling_holds_the_pair_together_without_moving_its_middle() {
        // 9 dB of difference, a 2 dB ceiling: the filter pair may not pull the ears more than 2 dB
        // apart, and the middle of the image - the mean of the two curves - stays on the shipped
        // curve. The bound is the feature; a bound that moved the centre would trade a lean for a
        // level error the model never asked for.
        let f = grid(121);
        let shape = bump(&f, 4000.0, 7.0, 0.25);
        let l: Vec<f64> = shape.iter().map(|&g| g + 2.0).collect();
        let e = pair(&f, &l, &flat(&f));
        let ctrl = f.iter().map(|&h| (h, 1.5_f64)).collect::<Vec<_>>();
        let t = trim(&e, &ctrl, &Policy::fixed(90.0), 2.0);
        assert!(
            t.bound_bands > 0,
            "a 9 dB difference with a 2 dB ceiling bound nothing"
        );
        let skew = t.skew_db();
        let worst = skew.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(worst <= 2.0 + 1e-9, "the ceiling leaked to {worst:.4} dB");

        let (gl, gr) = t.curves(&ctrl);
        let mut mid_err = 0.0_f64;
        for i in 0..ctrl.len() {
            mid_err = mid_err.max(((gl[i].1 + gr[i].1) / 2.0 - ctrl[i].1).abs());
        }
        assert!(
            mid_err < 1e-9,
            "the middle of the image moved by {mid_err:.2e} dB"
        );

        // Slack enough, the ceiling is not mentioned and nothing changes.
        let slack = trim(&e, &ctrl, &Policy::fixed(90.0), 90.0);
        assert_eq!(slack.bound_bands, 0);
        let free = slack.skew_db();
        assert!(
            free.iter().fold(0.0_f64, |m, &v| m.max(v.abs())) > 2.0,
            "a 90 dB ceiling bound a 9 dB difference"
        );
    }

    #[test]
    fn the_report_says_what_it_did_and_not_only_what_it_saw() {
        let f = grid(121);
        let shape = bump(&f, 4000.0, 7.0, 0.25);
        let l: Vec<f64> = shape.iter().map(|&g| g + 2.0).collect();
        let e = pair(&f, &l, &flat(&f));
        let m = analyse(&e);
        let ctrl = no_ctrl(&f);

        let seen = report_lines(&m, e.len(), None);
        assert!(seen.iter().any(|l| l.starts_with("=== Ears (121")));
        assert!(
            !seen.iter().any(|l| l.contains("Per-ear trim")),
            "a report of a shared filter claimed to have built a pair: {seen:?}"
        );

        let t = trim(&e, &ctrl, &Policy::fixed(90.0), 2.0);
        let done = report_lines(&m, e.len(), Some(&t));
        assert!(done.iter().any(|l| l.contains("Per-ear trim")), "{done:?}");
        assert!(
            done.iter().any(|l| l.starts_with("  Bound  ")),
            "the bound bit and said nothing: {done:?}"
        );
        // The wording follows the measurement, not the template.
        assert!(
            seen.iter().any(|l| l.contains("the centre leans")),
            "a 2 dB offset did not lean: {seen:?}"
        );
        assert!(
            seen.iter()
                .any(|l| l.contains("Reads as          : mismatch")),
            "a 7 dB shape difference was not called one: {seen:?}"
        );
        let quiet = pair(&f, &flat(&f), &bump(&f, 4000.0, 0.4, 0.25));
        let q = report_lines(&analyse(&quiet), f.len(), None);
        assert!(q.iter().any(|l| l.contains("inside a dB")), "{q:?}");
    }

    #[test]
    fn level_offset_clamped_to_ceiling_in_report() {
        let f = grid(121);
        let l = flat(&f).iter().map(|&x| x + 10.0).collect::<Vec<_>>();
        let e = pair(&f, &l, &flat(&f));
        let m = analyse(&e);
        let ctrl = no_ctrl(&f);
        let t = trim(&e, &ctrl, &Policy::fixed(90.0), 6.0);
        assert_eq!(t.level_db, 6.0);
        let lines = report_lines(&m, e.len(), Some(&t));
        assert!(lines
            .iter()
            .any(|line| line.contains("balance -6.00 dB cancelled")));
    }

    #[test]
    fn a_ceiling_that_bit_after_the_trim_says_the_centre_moved() {
        // "The centre is held" is a property of the design — the trim is applied as `±` — and the
        // notch ceiling is allowed to break it, because the ceiling is the last word on a notch
        // band. When it does, the report has to describe the curve it produced rather than the one
        // it planned.
        let f = grid(121);
        let shape = bump(&f, 4000.0, 7.0, 0.25);
        let l: Vec<f64> = shape.iter().map(|&g| g + 2.0).collect();
        let e = pair(&f, &l, &flat(&f));
        let m = analyse(&e);
        let ctrl = no_ctrl(&f);
        let t = trim(&e, &ctrl, &Policy::fixed(90.0), f64::INFINITY);

        let held = report_lines(&m, e.len(), Some(&t));
        assert!(
            held.iter()
                .any(|l| l.starts_with("  Centre            : held")),
            "{held:?}"
        );

        let bitten = report_lines(&m, e.len(), Some(&t.clone().with_cap_bites(15)));
        assert!(
            bitten.iter().any(|l| l.contains("moved in 15 knot(s)")),
            "the ceiling bit and the report still claimed the centre was held: {bitten:?}"
        );
        assert!(
            !bitten
                .iter()
                .any(|l| l.contains("Centre            : held")),
            "both claims printed: {bitten:?}"
        );
    }

    #[test]
    fn interaural_group_delay_measures_known_sample_shift_and_flags_threshold() {
        // 48 kHz: 1 sample = 20.833 µs (< 35 µs), 3 samples = 62.5 µs (> 35 µs).
        let mut ir_l = vec![0.0_f64; 1024];
        let mut ir_r1 = vec![0.0_f64; 1024];
        let mut ir_r3 = vec![0.0_f64; 1024];
        ir_l[100] = 1.0;
        ir_r1[101] = 1.0;
        ir_r3[103] = 1.0;

        let g1 =
            measure_interaural_group_delay(&ir_l, &ir_r1, 48000, IGD_BAND_LO_HZ, IGD_BAND_HI_HZ)
                .expect("1-sample shift");
        assert!(
            (g1.max_igd_us - 1e6 / 48000.0).abs() < 0.05,
            "got {}",
            g1.max_igd_us
        );
        assert!(!g1.exceeds_threshold);

        let g3 =
            measure_interaural_group_delay(&ir_l, &ir_r3, 48000, IGD_BAND_LO_HZ, IGD_BAND_HI_HZ)
                .expect("3-sample shift");
        assert!(
            (g3.max_igd_us - 3e6 / 48000.0).abs() < 0.05,
            "got {}",
            g3.max_igd_us
        );
        assert!(g3.exceeds_threshold);
        let line = igd_report_line(&g3);
        assert!(
            line.contains("35 µs") && line.contains("Møller et al. 2007"),
            "{line}"
        );
    }

    #[test]
    fn identical_irs_have_zero_interaural_group_delay() {
        let mut ir = vec![0.0_f64; 512];
        ir[32] = 0.8;
        ir[33] = -0.3;
        ir[34] = 0.1;
        let g = measure_interaural_group_delay(&ir, &ir, 48000, IGD_BAND_LO_HZ, IGD_BAND_HI_HZ)
            .expect("identical");
        assert!(g.max_igd_us < 1e-6, "got {}", g.max_igd_us);
        assert!(!g.exceeds_threshold);
    }

    #[test]
    fn interaural_group_delay_rejects_non_finite_and_silent_irs() {
        let mut bad = vec![0.0_f64; 128];
        bad[10] = f64::NAN;
        let mut ok = vec![0.0_f64; 128];
        ok[10] = 1.0;
        assert!(
            measure_interaural_group_delay(&bad, &ok, 48000, IGD_BAND_LO_HZ, IGD_BAND_HI_HZ)
                .is_none()
        );
        assert!(measure_interaural_group_delay(
            &vec![0.0; 128],
            &ok,
            48000,
            IGD_BAND_LO_HZ,
            IGD_BAND_HI_HZ
        )
        .is_none());
    }
}
