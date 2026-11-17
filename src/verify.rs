//! Checking a correction against a measurement taken after it.
//!
//! Everything else in this crate runs from the capture to the filter, and stops. The filter is
//! then loaded, and nobody looks again: the loop the roadmap calls §3.5 is open. This module closes
//! it, one measurement at a time — it takes a CSV someone measured *after* the correction was
//! applied, and answers the only question that matters about a correction: did the thing it was
//! supposed to do, happen?
//!
//! # Two curves, two different failures
//!
//! The measurement is compared to two predictions, because they fail for different reasons:
//!
//! * **residual** = `measured − predicted`, where `predicted = raw + |H(f)|` is the captured
//!   response plus the *realised* magnitude of the FIR that was written (not the design's control
//!   knots — the filter that exists, with its truncation ripple). This is the rig check: if it is
//!   large, then either the capture was taken with a different filter, or the fixture moved, or the
//!   unit moved. Pass/fail is judged on this one, which is what the roadmap asked for.
//! * **design shortfall** = `target − predicted`, computable before any new measurement exists.
//!   This is the design's own error: the bands where the correction never asked for what the target
//!   wanted, because regularization, the notch ceiling or a finite tap count got there first. A rig
//!   that reproduces the design *perfectly* still misses by this much, so a verifier that reported
//!   only the residual would call those bands fine and a reader would rightly distrust the next
//!   number.
//!
//! # What the comparison refuses to know
//!
//! **Level.** A post-EQ capture is taken through whatever interface gain the measurer happens to
//! have. The median residual is therefore removed before any band is judged, and printed as its own
//! line: it is a fact about the capture chain, not about the filter. Delay needs no handling
//! because these are magnitude curves.
//!
//! **Quiet.** A band whose capture sits [`DEFAULT_SNR_DB`] dB below the capture's own median band
//! level is reported as *not measured* and is never scored. Below that the number in the CSV is
//! mostly the rig's noise floor, and noise floors do not move when a filter is loaded — so a
//! verifier that scored such a band would report a stable, reproducible, meaningless failure. The
//! trade is stated in the report rather than hidden: a real deep null in that band is also read as
//! unmeasured, and at that level the capture genuinely cannot tell the two apart.
//!
//! **Absolute level, and its blind spot.** The units of a CSV are not knowable from the file: one
//! measurer writes dBFS off a soundcard, another writes dB SPL off a calibrated HATS. So the noise
//! gate is *relative*, and a capture that is entirely noise floor cannot be recognised as such — it
//! is uniformly quiet, which relative gating cannot see. The report therefore prints the capture's
//! own median band level as an anchor: the number a reader who knows their rig checks first, and the
//! one thing standing between "the residual is flat" and "the input was never connected".
//!
//! **Which number in a band.** A band is a region, and a curve can be flat on average inside one
//! while sitting 6 dB off at its middle. So the *mean* residual across the band is reported as
//! context, and pass/fail is judged on the **worst sample inside the band** — the direction an
//! under-report is unforgivable, and the reason the acceptance test can place a 6 dB error in a band
//! and expect 6 dB back (a bump a sixth of an octave wide averages down to about four fifths of
//! itself over a third-octave band, and a narrower one to half).
//!
//! **Scatter.** A band average hides what it averaged. Each band is sampled
//! [`BAND_SAMPLES`] times across its own third-octave width, and the spread of those samples is
//! reported next to the average: a band that sits inside the threshold while moving more than the
//! threshold *inside itself* is noise-limited, and the report says the number is scatter rather
//! than bias. That is the distinction the acceptance criteria asked for — "the measurement is
//! noisy" and "the fit is wrong" are different findings and get different words.
//!
//! # Where the thresholds come from
//!
//! No standard reviewed defines how far a headphone may sit from a target curve after EQ. What
//! exists in the literature is preference data, not acceptance criteria: Fleischmann, Silzle and
//! Plogsties (*Headphone Equalization — Measurement, Design and Psychoacoustic Evaluation*,
//! Microelectronic Systems, Springer 2011, pp. 301–311) equalize measured headphones onto several
//! targets and evaluate by listening test, and state in their abstract that "in the literature,
//! there is no consensus about the optimal transfer function and equalization of headphones" —
//! which is exactly why a shipped default cannot be cited to a document. So:
//!
//! * [`DEFAULT_PASS_DB`] is a *configuration default*, and the report always prints the number it
//!   used, so a pass is never read as compliance with an unnamed standard.
//! * [`DEFAULT_SNR_DB`] is derived, not chosen: additive noise of level `N` in a band whose signal
//!   is `S` biases the band's level by up to `20·log10(1 + N/S)`, which is 0.83 dB at 20 dB of
//!   signal-to-noise and 0.22 dB at 34 dB. Twenty dB puts the noise's contribution under 1 dB —
//!   under a third of the default threshold — and is the level at which measurement practice
//!   (averaging until the noise is well below the tone) lands in the same place.

use crate::assess;
use crate::fit::median_db;
use crate::pchip::Pchip;

/// Residual, in dB, beyond which a band is called a failure. A configuration default; see the
/// module docs for why it cannot be cited to a standard.
pub const DEFAULT_PASS_DB: f64 = 3.0;
/// How far a band's capture may sit below the capture's own median band level before it is
/// treated as unmeasured. Derived from the noise bias — see the module docs.
pub const DEFAULT_SNR_DB: f64 = 20.0;
/// Samples taken across one third-octave band, log-spaced and including both edges.
///
/// Twenty-five is not decoration: the band average and the within-band spread are read off these
/// samples, and at a handful of points a wiggle whose period happens to match the sample spacing
/// aliases straight through — the scatter would read as smooth on the capture that motivated this
/// field. 25 samples over a third of an octave is 0.0133 octave apart, so structure down to about
/// 0.03 octave period is seen rather than invented.
pub const BAND_SAMPLES: usize = 25;

/// What the verifier is told, as opposed to what it measures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Residual beyond which a band fails.
    pub pass_db: f64,
    /// Band level, relative to the capture's median band, below which a band is unmeasured.
    pub snr_db: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            pass_db: DEFAULT_PASS_DB,
            snr_db: DEFAULT_SNR_DB,
        }
    }
}

/// What became of one band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Inside the threshold after the capture's level offset was removed.
    Pass,
    /// Outside it.
    Fail,
    /// The capture is too close to its own noise floor here for the residual to mean anything.
    TooQuiet,
    /// The capture, or the prediction, does not extend this far.
    Uncovered,
}

impl Status {
    /// The word written in the CSV and the report.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::TooQuiet => "unmeasured",
            Self::Uncovered => "uncovered",
        }
    }
}

/// One band's verdict, with every number the sentence about it is built from.
#[derive(Clone, Copy, Debug)]
pub struct BandResult {
    /// IEC 61260-1 band index, 0 at 1 kHz.
    pub index: i32,
    /// Exact centre, `1000 · 10^(k/10)`.
    pub centre_hz: f64,
    /// The standard's nominal centre, the name the report prints.
    pub nominal_hz: f64,
    /// Band average of the post-EQ capture, dB.
    pub measured_db: f64,
    /// Band average of `raw + |H|`, dB.
    pub predicted_db: f64,
    /// Band average of the design target, dB; `NaN` when no target was supplied.
    pub target_db: f64,
    /// Band *average* of `measured − predicted`, before the level offset: context, and the number
    /// that tells a bias apart from structure. See the module docs.
    pub residual_db: f64,
    /// The worst residual inside the band, offset removed — the number pass/fail is judged on.
    /// Signed, and taken at the sample that deviated most, so a band can average near zero and
    /// still fail: that is the case averaging would have hidden.
    pub shape_db: f64,
    /// Spread of the residual across the band's own width, dB.
    pub scatter_db: f64,
    /// Worst `target − predicted` inside the band, dB: what the design itself did not ask for.
    pub design_db: f64,
    /// [`Status::Pass`] / [`Status::Fail`] / one of the two reasons it was not scored.
    pub status: Status,
    /// True when [`BandResult::scatter_db`] exceeds the threshold: the band average is then a
    /// description of a wiggly trace, not a measured bias.
    pub noise_limited: bool,
}

/// The whole comparison.
#[derive(Clone, Debug)]
pub struct Report {
    /// Every band on the grid, scored or not, in frequency order.
    pub bands: Vec<BandResult>,
    /// The capture's level offset, removed from every residual.
    pub offset_db: f64,
    /// Band level below which a band is unmeasured, in the capture's own units.
    pub gate_db: f64,
    /// The capture's median band level, in whatever units the file carries.
    ///
    /// Printed because the gate is relative and cannot detect a capture that is noise floor
    /// everywhere; this is the number a reader who knows their rig looks at.
    pub median_db: f64,
    /// Bands that were scored at all.
    pub scored: usize,
    /// Scored bands inside the threshold.
    pub pass: usize,
    /// Scored bands outside it.
    pub fail: usize,
    /// Bands dropped for being at the noise floor.
    pub quiet: usize,
    /// Bands dropped for falling outside the capture or the prediction.
    pub uncovered: usize,
    /// Scored bands whose within-band spread exceeds the threshold.
    pub noise_limited: usize,
    /// Largest `|shape_db|` among the failures, with its frequency.
    pub worst: Option<(f64, f64)>,
    /// Largest `|design_db|` among the scored bands, with its frequency: the design's own error.
    pub design_worst: Option<(f64, f64)>,
    /// Capture points the comparison had to work with.
    pub points: usize,
    /// Frequency range the capture covered.
    pub f_lo: f64,
    /// … to …
    pub f_hi: f64,
    /// Threshold in force when this was produced.
    pub pass_db: f64,
    /// Noise gate in force, dB below the capture's median band.
    pub snr_db: f64,
}

impl Report {
    /// True when there was something to check and nothing fell outside the threshold.
    ///
    /// "Nothing to check" is not a pass: a capture entirely below the gate produces zero failures
    /// and would otherwise be reported as a good result.
    #[must_use]
    pub fn ok(&self) -> bool {
        self.scored > 0 && self.fail == 0
    }

    /// True when no band could be scored at all.
    #[must_use]
    pub fn empty(&self) -> bool {
        self.scored == 0
    }
}

/// A curve on its own frequency axis, ready to be asked for band averages.
struct Curve {
    f: Vec<f64>,
    g: Vec<f64>,
}

impl Curve {
    /// Keep the usable points, in ascending frequency, one per frequency (last wins, which is what
    /// the loaders do too).
    fn new(pts: &[(f64, f64)]) -> Option<Self> {
        let mut kept: Vec<(f64, f64)> = pts
            .iter()
            .copied()
            .filter(|(f, g)| f.is_finite() && *f > 0.0 && g.is_finite())
            .collect();
        kept.sort_by(|a, b| a.0.total_cmp(&b.0));
        // A repeated frequency is a broken capture either way; "last row wins" is what the model
        // loader and `Pchip::new` both do, so the same file means the same thing in both places.
        let mut f: Vec<f64> = Vec::with_capacity(kept.len());
        let mut g: Vec<f64> = Vec::with_capacity(kept.len());
        for (ff, gg) in kept {
            if f.last().is_some_and(|&prev| prev == ff) {
                let last = g.len() - 1;
                g[last] = gg;
            } else {
                f.push(ff);
                g.push(gg);
            }
        }
        (f.len() >= 2).then_some(Self { f, g })
    }

    fn lo(&self) -> f64 {
        self.f[0]
    }

    fn hi(&self) -> f64 {
        *self.f.last().unwrap_or(&0.0)
    }

    /// True when the whole interval lies on the axis. No extrapolation: a curve continued past its
    /// last point invents a response, and a verifier that invents the answer cannot fail.
    fn covers(&self, lo: f64, hi: f64) -> bool {
        lo >= self.lo() && hi <= self.hi()
    }
}

/// Compensated mean, because that is how this crate adds numbers up; for 25 samples it makes no
/// difference and for zero samples it says so instead of returning a plausible `0.0`.
fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut acc = crate::accum::RunningSum::new();
    for &x in v {
        acc.add(x);
    }
    acc.total() / v.len() as f64
}

/// The log-spaced samples across one band's own width, both edges included.
fn band_samples(lo_hz: f64, hi_hz: f64) -> Vec<f64> {
    let n = u32::try_from(BAND_SAMPLES).unwrap_or(25) - 1;
    (0..=n)
        .map(|i| lo_hz * (hi_hz / lo_hz).powf(f64::from(i) / f64::from(n)))
        .collect()
}

/// Read capture rows into `(freq_hz, dB)`, following the database's own column convention.
///
/// The shipped files are `[freq, left, right, average]`, and the average column is what the DSP
/// fits, so a capture exported the same way is read the same way: with four or more columns the
/// **last** one is the curve. With two or three there is no average column, so the non-frequency
/// columns are averaged — which for the usual two-column `(freq, dB)` file is exactly that column,
/// and for a `[freq, L, R]` export is the mean of the ears, the same thing the shared filter was
/// built to hit. (Note: if a 3-column CSV contains `[freq, spl, phase]`, strip the phase degree
/// column to avoid averaging SPL decibels with phase degrees).
#[must_use]
pub fn read_capture(rows: &[Vec<f64>]) -> Option<Vec<(f64, f64)>> {
    let pts: Vec<(f64, f64)> = rows
        .iter()
        .filter(|r| r.len() >= 2)
        .map(|r| {
            let g = if r.len() >= 4 {
                *r.last().unwrap()
            } else {
                mean(&r[1..])
            };
            (r[0], g)
        })
        .collect();
    Curve::new(&pts).map(|c| c.f.iter().copied().zip(c.g).collect())
}

/// Compare a post-EQ capture with what the filter was supposed to produce.
///
/// * `post` — the new measurement, `(freq_hz, dB)`.
/// * `predicted` — `raw + |H(f)|`, the realised filter, `(freq_hz, dB)`.
/// * `target` — the design target the correction aimed at; optional in the sense that `None`
///   (or fewer than two usable points) simply omits the design-shortfall column.
///
/// `None` when either required curve has fewer than two usable points, which is a broken input
/// rather than a flat result.
#[must_use]
pub fn verify(
    post: &[(f64, f64)],
    predicted: &[(f64, f64)],
    target: Option<&[(f64, f64)]>,
    cfg: &Config,
) -> Option<Report> {
    let post = Curve::new(post)?;
    let predicted = Curve::new(predicted)?;
    let target = target.and_then(Curve::new);

    // The grid stops where the shorter curve stops: beyond that there is nothing to compare, and
    // an extrapolated PCHIP is a polite way of writing the answer in before measuring it.
    let grid = assess::bands_below(post.hi().min(predicted.hi()));
    if grid.is_empty() {
        return None;
    }

    let (mut rows, resid) = measure_bands(&grid, &post, &predicted, target.as_ref());
    let tally = score(&mut rows, &resid, cfg);

    Some(Report {
        scored: tally.pass + tally.fail,
        bands: rows,
        offset_db: tally.offset_db,
        gate_db: tally.gate_db,
        median_db: tally.median_db,
        pass: tally.pass,
        fail: tally.fail,
        quiet: tally.quiet,
        uncovered: grid.len() - tally.pass - tally.fail - tally.quiet,
        noise_limited: tally.noise_limited,
        worst: tally.worst,
        design_worst: tally.design_worst,
        points: post.f.len(),
        f_lo: post.lo(),
        f_hi: post.hi(),
        pass_db: cfg.pass_db,
        snr_db: cfg.snr_db,
    })
}

/// Every band on the grid, measured against both curves. Nothing is judged here: the level offset
/// that judgement needs does not exist until all of these numbers are in.
fn measure_bands(
    grid: &[assess::Band],
    post: &Curve,
    predicted: &Curve,
    target: Option<&Curve>,
) -> (Vec<BandResult>, Vec<Vec<f64>>) {
    let mut rows: Vec<BandResult> = Vec::with_capacity(grid.len());
    let mut resid_of: Vec<Vec<f64>> = Vec::with_capacity(grid.len());
    let p_post = Pchip::new(&post.f, &post.g);
    let p_pred = Pchip::new(&predicted.f, &predicted.g);
    let p_target = target.map(|t| Pchip::new(&t.f, &t.g));

    for b in grid {
        let samples = band_samples(b.lo_hz, b.hi_hz);
        let covered = post.covers(b.lo_hz, b.hi_hz) && predicted.covers(b.lo_hz, b.hi_hz);
        let target_covered = target.is_some_and(|t| t.covers(b.lo_hz, b.hi_hz));

        let (measured_db, predicted_db, target_db, scatter_db, resid, design_db) = if covered {
            let m = p_post.eval_grid(&samples);
            let h = p_pred.eval_grid(&samples);
            let m_mean = mean(&m);
            let h_mean = mean(&h);

            let t_data = if target_covered {
                p_target.as_ref().map(|pt| pt.eval_grid(&samples))
            } else {
                None
            };
            let t_mean = t_data.as_ref().map_or(f64::NAN, |t| mean(t));

            let d: Vec<f64> = m.iter().zip(&h).map(|(&x, &y)| x - y).collect();
            let max = d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = d.iter().cloned().fold(f64::INFINITY, f64::min);

            let design = t_data.map_or(f64::NAN, |t| {
                t.iter().zip(&h).map(|(&x, &y)| x - y).fold(
                    f64::NAN,
                    |worst: f64, v: f64| {
                        if !worst.is_finite() || v.abs() > worst.abs() {
                            v
                        } else {
                            worst
                        }
                    },
                )
            });
            (m_mean, h_mean, t_mean, max - min, d, design)
        } else {
            (f64::NAN, f64::NAN, f64::NAN, f64::NAN, Vec::new(), f64::NAN)
        };
        resid_of.push(resid);

        rows.push(BandResult {
            index: b.index,
            centre_hz: b.centre_hz,
            nominal_hz: b.nominal_hz,
            measured_db,
            predicted_db,
            target_db,
            scatter_db,
            design_db,
            residual_db: measured_db - predicted_db,
            shape_db: f64::NAN,
            status: if covered {
                Status::Pass
            } else {
                Status::Uncovered
            },
            noise_limited: false,
        });
    }
    (rows, resid_of)
}

/// What judging every band produced. Kept apart from [`Report`] so the two-pass structure below is
/// visible instead of being twelve statements interleaved with the loop that needs their results.
struct Tally {
    offset_db: f64,
    gate_db: f64,
    median_db: f64,
    pass: usize,
    fail: usize,
    quiet: usize,
    noise_limited: usize,
    worst: Option<(f64, f64)>,
    design_worst: Option<(f64, f64)>,
}

/// The two passes: measure the capture's own level and noise floor first, then judge.
///
/// The order is not cosmetic. A post-EQ capture is taken through whatever interface gain the
/// measurer happened to have, so the residual's *constant* is a fact about the capture chain; and
/// a band at the rig's noise floor reproduces itself beautifully and would be reported as a stable,
/// reproducible, meaningless failure. Both have to be settled before a single band is scored.
fn score(rows: &mut [BandResult], resid: &[Vec<f64>], cfg: &Config) -> Tally {
    // The gate is relative to the capture itself. An absolute dBFS floor would be a statement about
    // the interface gain, which is the one thing known to be arbitrary here.
    let levels: Vec<f64> = rows
        .iter()
        .filter(|r| r.status != Status::Uncovered)
        .map(|r| r.measured_db)
        .collect();
    let median_level = median_db(&levels);
    let gate_db = if levels.is_empty() {
        f64::NEG_INFINITY
    } else {
        median_level - cfg.snr_db
    };

    // The median of the *band means*: a level difference is a constant across the axis, so each
    // band's mean estimates it with as little noise as one number can carry, and the median across
    // bands keeps one wild band (or a filter that was never loaded) from dragging the whole curve.
    let offset_db = {
        let r: Vec<f64> = rows
            .iter()
            .filter(|r| r.status != Status::Uncovered && r.measured_db >= gate_db)
            .map(|r| r.residual_db)
            .collect();
        if r.is_empty() {
            f64::NAN
        } else {
            median_db(&r)
        }
    };

    let mut t = Tally {
        offset_db,
        gate_db,
        median_db: median_level,
        pass: 0,
        fail: 0,
        quiet: 0,
        noise_limited: 0,
        worst: None,
        design_worst: None,
    };

    for (r, d) in rows.iter_mut().zip(resid) {
        if r.status == Status::Uncovered {
            continue;
        }
        if r.measured_db < gate_db || !offset_db.is_finite() {
            r.status = Status::TooQuiet;
            t.quiet += 1;
            continue;
        }
        // Worst-in-band, signed: the sample furthest from what the prediction said once the
        // capture's level is accounted for.
        r.shape_db = d
            .iter()
            .map(|&x| x - offset_db)
            .fold(f64::NAN, |worst: f64, v: f64| {
                if !worst.is_finite() || v.abs() > worst.abs() {
                    v
                } else {
                    worst
                }
            });
        r.noise_limited = r.scatter_db.is_finite() && r.scatter_db > cfg.pass_db;
        r.status = if r.shape_db.abs() > cfg.pass_db {
            t.fail += 1;
            if t.worst.is_none_or(|(_, w)| r.shape_db.abs() > w.abs()) {
                t.worst = Some((r.centre_hz, r.shape_db));
            }
            Status::Fail
        } else {
            t.pass += 1;
            Status::Pass
        };
        if r.noise_limited {
            t.noise_limited += 1;
        }
        if r.design_db.is_finite()
            && t.design_worst
                .is_none_or(|(_, x)| r.design_db.abs() > x.abs())
        {
            t.design_worst = Some((r.centre_hz, r.design_db));
        }
    }
    t
}

/// The report block, in the same shape as every other section: a banner, then aligned pairs.
///
/// `note` is for the one thing the numbers cannot say about the run they belong to — that the
/// filter was split per ear, say, so the curve being compared is the shared design and not what
/// either ear was given.
#[must_use]
pub fn report_lines(r: &Report, note: Option<&str>) -> Vec<String> {
    let mut out = Vec::new();
    out.push(format!(
        "=== Verification (post-EQ capture: {} points, {} - {}) ===",
        r.points,
        crate::assess::band_name(r.f_lo),
        crate::assess::band_name(r.f_hi)
    ));

    let mut unscored = Vec::new();
    if r.quiet > 0 {
        unscored.push(format!(
            "{} too quiet (< {:.1} dB, {:.0} dB below the capture's median band)",
            r.quiet, r.gate_db, r.snr_db
        ));
    }
    if r.uncovered > 0 {
        unscored.push(format!("{} outside the compared range", r.uncovered));
    }
    out.push(format!(
        "  Scored          : {} of {} band(s){}",
        r.scored,
        r.bands.len(),
        if unscored.is_empty() {
            String::new()
        } else {
            format!("; {}", unscored.join(", "))
        }
    ));
    if r.empty() {
        out.push(
            "  Verdict         : NOTHING TO CHECK - no band was both covered and above the noise gate"
                .to_string(),
        );
        if let Some(n) = note {
            out.push(format!("  Note            : {n}"));
        }
        return out;
    }

    out.push(format!(
        "  Level offset    : {} dB removed before judging shape (the capture is not calibrated, so its absolute level says nothing about the filter)",
        crate::measure::sgn(r.offset_db, 2)
    ));
    out.push(format!(
        "  Capture level   : median band {:.1} dB, gate at {:.1} dB (its own units; a capture that is nothing but noise floor is uniformly quiet and this section cannot see that - check this number against your rig)",
        r.median_db, r.gate_db
    ));

    out.push(format!(
        "  Threshold       : +/-{:.2} dB of residual after the offset",
        r.pass_db
    ));
    if let Some((hz, shape)) = r.worst {
        out.push(format!(
            "  Worst band      : {} dB @ {} Hz (band {})",
            crate::measure::sgn(shape, 2),
            hz.round(),
            assess::band_name(assess::nearest_nominal(hz))
        ));
    }
    out.push(format!(
        "  Verdict         : {} - {} of {} band(s) outside +/-{:.2} dB{}",
        if r.fail == 0 { "PASS" } else { "FAIL" },
        r.fail,
        r.scored,
        r.pass_db,
        if r.noise_limited > 0 {
            format!("; {} scored band(s) noise-limited", r.noise_limited)
        } else {
            String::new()
        }
    ));
    if r.noise_limited > 0 {
        let worst_scatter = r
            .bands
            .iter()
            .filter(|b| b.noise_limited)
            .map(|b| b.scatter_db)
            .fold(0.0_f64, f64::max);
        out.push(format!(
            "  Noise-limited   : the residual moves up to {:.1} dB inside those bands, so the number there is scatter across the band, not a bias",
            worst_scatter
        ));
    }
    if r.quiet > 0 {
        out.push(format!(
            "  Not claimed     : the {} quiet band(s) are reported as unmeasured, not as passing; a real null that deep cannot be told from the rig's own noise by this file",
            r.quiet
        ));
    }
    if let Some((hz, d)) = r.design_worst {
        out.push(format!(
            "  Design shortfall: {} dB @ {} Hz (band {}) - the filter itself does not reach the target here, so a rig that reproduces it exactly still misses by this much",
            crate::measure::sgn(d, 2),
            hz.round(),
            assess::band_name(assess::nearest_nominal(hz))
        ));
    }
    if let Some(n) = note {
        out.push(format!("  Note            : {n}"));
    }
    out
}

/// The per-band table, written next to the audit CSV when verification is asked for.
///
/// The audit CSV itself is byte-compared against the reference program, so it cannot gain columns
/// here; this file exists so the numbers survive the terminal.
#[must_use]
pub fn csv_text(r: &Report) -> String {
    let mut s = String::from(
        "band_index,centre_hz,nominal_hz,measured_db,predicted_db,target_db,residual_db,shape_db,scatter_db,design_db,noise_limited,status\n",
    );
    for b in &r.bands {
        let num = |v: f64| -> String {
            if v.is_finite() {
                format!("{v:.3}")
            } else {
                String::new()
            }
        };
        s.push_str(&format!(
            "{},{:.2},{:.2},{},{},{},{},{},{},{},{},{}\n",
            b.index,
            b.centre_hz,
            b.nominal_hz,
            num(b.measured_db),
            num(b.predicted_db),
            num(b.target_db),
            num(b.residual_db),
            num(b.shape_db),
            num(b.scatter_db),
            num(b.design_db),
            if b.noise_limited { "1" } else { "0" },
            b.status.as_str(),
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assess::{self, bands_below};

    /// A densely sampled axis: 60 points per decade from 20 Hz to 20 kHz, so a third-octave band
    /// holds about 18 capture points and interpolation is not the thing under test.
    fn axis() -> Vec<f64> {
        let n = (60.0_f64 * (1000.0_f64).log10()).round() as usize;
        (0..=n)
            .map(|i| 20.0_f64 * (1000.0_f64).powf(f64::from(i as u32) / f64::from(n as u32)))
            .collect()
    }

    /// A curve on that axis, built from a closure so a test can state the shape it means.
    fn curve(g: impl Fn(f64) -> f64) -> Vec<(f64, f64)> {
        axis().iter().map(|&f| (f, g(f))).collect()
    }

    /// A bump of `amp` dB centred on `f0`, narrow enough to belong to one band.
    ///
    /// The width matters: a Gaussian in `log f` with `sigma` of a third of an octave falls to about
    /// 6 % of its peak one band away, so it lands on one band and the neighbours stay flat. At a
    /// wider sigma the bump lifts most of the axis, the median level offset swallows it, and the
    /// residual is largest where the bump is *absent* — which is a true reading of what the module
    /// does, and useless for a test that wants a named band.
    fn bump(f: f64, f0: f64, amp: f64) -> f64 {
        let sigma = 0.115 / (2.0_f64 * 2.0_f64.ln()).sqrt(); // half height about a sixth of an octave out
        let x = f.ln() - f0.ln();
        amp * (-0.5 * (x / sigma).powi(2)).exp()
    }

    fn cfg() -> Config {
        Config::default()
    }

    #[test]
    fn a_capture_that_reproduces_the_prediction_passes_every_band() {
        let predicted = curve(|f| 4.0 - 6.0 * (f / 20000.0).ln() / (1000.0_f64).ln().abs());
        let post = predicted.clone();
        let r = verify(&post, &predicted, None, &cfg()).expect("curves are usable");
        assert_eq!(r.fail, 0, "an identical capture was scored as a failure");
        assert!(r.ok());
        assert!(
            r.offset_db.abs() < 1e-9,
            "an identical capture reported a level offset: {}",
            r.offset_db
        );
        // The axis cannot cover the half-band hanging off each end, so those are uncovered rather
        // than passed — a curve is not asked to describe frequencies it does not reach.
        assert_eq!(r.quiet, 0);
        let inside = r
            .bands
            .iter()
            .filter(|b| b.status != Status::Uncovered)
            .count();
        assert!(inside > 20, "only {inside} bands were inside the axis");
        assert!(
            r.bands
                .iter()
                .all(|b| b.status == Status::Pass || b.status == Status::Uncovered),
            "a covered band did not pass"
        );
    }

    #[test]
    fn a_known_residual_is_named_by_band_and_by_number() {
        // The acceptance criterion is that the tool *names* the band and the dB error, so the
        // residual here is placed on a band centre and given a size the report must repeat.
        let predicted = curve(|_| 0.0);
        let f0 = bands_below(24000.0)
            .into_iter()
            .find(|b| b.index == 5)
            .expect("grid")
            .centre_hz;
        let post = curve(|f| bump(f, f0, 6.0));
        let r = verify(&post, &predicted, None, &cfg()).expect("curves are usable");
        assert!(r.fail > 0, "a 6 dB bump was not failed anywhere");
        let (hz, shape) = r.worst.expect("a worst band");
        assert!(
            (hz - f0).abs() < 1.0,
            "the bump at {f0} Hz was blamed on {hz} Hz"
        );
        assert!(
            (shape - 6.0).abs() < 0.6,
            "a 6 dB bump was reported as {shape} dB"
        );
        let band = r
            .bands
            .iter()
            .find(|b| (b.centre_hz - f0).abs() < 1.0)
            .expect("the band is in the table");
        assert_eq!(band.status, Status::Fail);
        let lines = report_lines(&r, None).join("\n");
        assert!(
            lines.contains(&format!("{} Hz", f0.round())),
            "the report does not name the band it failed: {lines}"
        );
        assert!(lines.contains("FAIL"), "{lines}");
    }

    #[test]
    fn an_absolute_level_difference_is_reported_and_not_failed() {
        let predicted = curve(|f| -bump(f, 4000.0, 3.0));
        let post = curve(|f| -bump(f, 4000.0, 3.0) + 7.5);
        let r = verify(&post, &predicted, None, &cfg()).expect("curves are usable");
        assert!(
            (r.offset_db - 7.5).abs() < 0.05,
            "the offset was reported as {} dB",
            r.offset_db
        );
        assert_eq!(r.fail, 0, "a level offset was judged as a shape error");
        assert!(r.ok());
        let lines = report_lines(&r, None).join("\n");
        assert!(
            lines.contains("+7.50 dB removed"),
            "the offset is not in the report: {lines}"
        );
    }

    #[test]
    fn one_wild_band_cannot_drag_the_offset_and_shift_every_other_band() {
        // This is why the offset is a median. With a mean, one band measured in a different session
        // (a re-plug, a level bump at one frequency) moves every other band's verdict at once.
        // The wild region has to sit where the axis actually *covers* a band: an earlier version of
        // this fixture put it above 18 kHz, where the top band is uncovered anyway, and the test
        // passed while asserting nothing.
        let predicted = curve(|_| 0.0);
        let post = curve(|f| {
            if (900.0..1300.0).contains(&f) {
                40.0
            } else {
                0.0
            }
        });
        let r = verify(&post, &predicted, None, &cfg()).expect("curves are usable");
        assert!(
            r.bands
                .iter()
                .any(|b| (b.centre_hz - 1000.0).abs() < 1.0 && b.status == Status::Fail),
            "the wild band was not even failed, so this fixture tests nothing"
        );
        assert!(
            r.offset_db.abs() < 1.0,
            "one wild band moved the offset to {} dB",
            r.offset_db
        );
        assert!(
            r.fail <= 2,
            "the wild band shifted the whole curve: {} bands failed",
            r.fail
        );
    }

    #[test]
    fn the_design_shortfall_describes_the_design_not_the_measurement() {
        // The two numbers answer different questions and must not be allowed to merge: if the
        // shortfall were measured against the capture, a rig that measured badly would appear to
        // have a design that missed by whatever the rig did.
        let predicted = curve(|_| 0.0);
        let target = curve(|f| bump(f, 2000.0, 5.0));
        // The rig is bad: +6 dB at the same place the design fell 5 dB short.
        let post = curve(|f| bump(f, 2000.0, 6.0));
        let r = verify(&post, &predicted, Some(&target), &cfg()).expect("curves are usable");
        let (hz, design) = r.design_worst.expect("the design misses the target");
        assert!(
            (design - 5.0).abs() < 0.7,
            "the design shortfall read {design} dB, which is the measurement's error, not the design's"
        );
        assert!(hz > 1600.0 && hz < 2500.0, "shortfall blamed on {hz} Hz");
        let band = r
            .bands
            .iter()
            .find(|b| (b.centre_hz - hz).abs() < 1.0)
            .expect("band");
        assert!(
            band.shape_db.abs() > design.abs() + 0.5,
            "the residual did not stay separate from the shortfall: {:?} vs {:?}",
            band.shape_db,
            design
        );
    }

    #[test]
    fn fine_ripple_is_resolved_rather_than_averaged_into_smoothness() {
        // 20 periods per decade is 0.05 of an octave — about a fifth of a third-octave band. Measured
        // on this fixture, not guessed: 3 or 5 samples per band read **0.00 dB** of scatter in every
        // band, because the samples land at the same phase of the ripple each time, and the module
        // would report a perfectly flat residual for a capture swinging ±4 dB inside each band. At 7
        // samples and above the answer is 6.78 dB in every band — 85 % of the true 8.0 dB
        // peak-to-peak, and stable band to band, which is what resolved (rather than aliased) looks
        // like. 25 is chosen for margin against the phase relationships real captures bring, at a
        // cost of a few hundred interpolations; 7 already passes this test, so that choice is
        // explicitly not pinned here.
        let predicted = curve(|_| 0.0);
        let post = curve(|f| 4.0 * (20.0 * f.log10() * std::f64::consts::TAU).sin());
        let r = verify(&post, &predicted, None, &cfg()).expect("curves are usable");
        let scored: usize = r
            .bands
            .iter()
            .filter(|b| b.status != Status::Uncovered)
            .count();
        assert!(scored > 10, "too little of the axis was scored: {scored}");
        assert!(
            r.noise_limited * 2 > scored,
            "a ±4 dB ripple every fifth of a band wide was flagged in only {} of {scored} bands",
            r.noise_limited
        );
        let worst_scatter = r.bands.iter().map(|b| b.scatter_db).fold(0.0_f64, f64::max);
        assert!(
            worst_scatter > 5.0,
            "the ripple's within-band spread was read as {worst_scatter} dB"
        );
    }

    #[test]
    fn an_unrelated_curve_fails_loudly() {
        let predicted = curve(|f| -bump(f, 4000.0, 3.0));
        // A different headphone: a gross tonal tilt from +8 dB in the bass to -8 dB in the treble,
        // plus a resonance. The tilt is deliberately *not* a constant, because the level offset is
        // removed by design and would swallow a constant whole — which is the right behaviour, and
        // the wrong fixture for "fails loudly".
        let post =
            curve(|f| 8.0 - 16.0 * ((f / 20.0).ln() / (1000.0_f64).ln()) + bump(f, 900.0, 7.0));
        let r = verify(&post, &predicted, None, &cfg()).expect("curves are usable");
        assert!(
            r.fail * 2 > r.scored,
            "an unrelated curve failed only {} of {} bands",
            r.fail,
            r.scored
        );
        assert!(!r.ok());
        assert!(r.worst.is_some(), "no band was blamed");
        let lines = report_lines(&r, None).join("\n");
        assert!(lines.contains("FAIL"), "{lines}");
        assert!(lines.contains("Worst band"), "{lines}");
    }

    #[test]
    fn a_capture_with_nothing_in_the_bass_is_unmeasured_not_wrong() {
        // The distinction the acceptance list asked for. The bass is 45 dB below the rest of the
        // capture; the residual there is enormous, and none of it is information.
        let predicted = curve(|_| 0.0);
        let post = curve(|f| if f < 100.0 { -45.0 } else { 0.0 });
        let r = verify(&post, &predicted, None, &cfg()).expect("curves are usable");
        assert!(r.quiet > 0, "the quiet bass was scored as a finding");
        let grid = assess::bands_below(24000.0);
        for b in &r.bands {
            let g = grid.iter().find(|g| g.index == b.index).expect("grid");
            // `Uncovered` is allowed here: the band at the very bottom of the axis reaches below
            // the first capture point, so it is dropped for a different and equally good reason.
            if g.hi_hz <= 100.0 && b.status != Status::Uncovered {
                assert_eq!(
                    b.status,
                    Status::TooQuiet,
                    "the whole band at {} Hz is at the rig's floor and was scored anyway",
                    b.centre_hz
                );
            }
        }
        // Only the one band the step cuts in half may be blamed, and it is right to blame it: a 45 dB
        // step inside a single third-octave band is a deviation in the capture, not an artefact.
        let blamed: Vec<&BandResult> = r
            .bands
            .iter()
            .filter(|b| b.status == Status::Fail)
            .collect();
        assert!(
            blamed.iter().all(|b| {
                let g = grid.iter().find(|g| g.index == b.index).expect("grid");
                g.lo_hz < 100.0 && g.hi_hz > 100.0
            }),
            "a band the step did not cut was failed: {:?}",
            blamed.iter().map(|b| b.centre_hz).collect::<Vec<_>>()
        );
        let lines = report_lines(&r, None).join("\n");
        assert!(lines.contains("too quiet"), "{lines}");
        assert!(lines.contains("Not claimed"), "{lines}");
    }

    #[test]
    fn a_capture_that_covers_no_whole_band_is_nothing_to_check() {
        // A file spanning a fraction of one third-octave band interpolates fine and scores nothing:
        // every band falls outside it. That has to be said out loud rather than reported as zero
        // failures, which is what a pass looks like.
        let predicted = curve(|_| 0.0);
        let post = vec![(1000.0, 0.0), (1000.4, 0.0), (1000.8, 0.0)];
        let r = verify(&post, &predicted, None, &cfg()).expect("three points interpolate");
        assert!(r.empty(), "{:?} bands were scored", r.scored);
        assert_eq!(r.pass, 0);
        assert!(!r.ok(), "nothing to check was reported as a pass");
        let lines = report_lines(&r, None).join("\n");
        assert!(lines.contains("NOTHING TO CHECK"), "{lines}");
        assert!(lines.contains("outside the compared range"), "{lines}");
    }

    #[test]
    fn the_capture_s_own_level_is_printed_because_relative_gating_cannot_see_a_dead_file() {
        // The gate is relative to the capture, so a capture that is noise floor everywhere is
        // uniformly quiet and passes its own gate. The mitigation is not a second threshold in
        // unknown units; it is printing the anchor for a reader who knows their rig.
        let predicted = curve(|_| 0.0);
        let post = curve(|_| -90.0);
        let r = verify(&post, &predicted, None, &cfg()).expect("curves are usable");
        assert!(
            (r.median_db + 90.0).abs() < 0.01,
            "the median band level was {}",
            r.median_db
        );
        let lines = report_lines(&r, None).join("\n");
        assert!(lines.contains("-90.0 dB"), "{lines}");
        assert!(lines.contains("gate at"), "{lines}");
    }

    #[test]
    fn a_wiggly_capture_is_called_scatter_and_not_bias() {
        let predicted = curve(|_| 0.0);
        // A 1/8-octave ripple of +/- 4 dB: far more than a third of an octave's worth of samples'
        // resolution, and averaging to about nothing over a band.
        let post = curve(|f| 4.0 * (6.0 * f.log10() * std::f64::consts::TAU).sin());
        let r = verify(&post, &predicted, None, &cfg()).expect("curves are usable");
        assert!(
            r.noise_limited > 0,
            "a 4 dB ripple was never flagged as scatter"
        );
        // The discriminator, not the fail count: a band carrying this ripple sits near zero on
        // average while moving more than the threshold inside itself. That is structure. A fit that
        // is wrong sits *off* zero on average, with a flat trace inside the band.
        let flagged: Vec<&BandResult> = r.bands.iter().filter(|b| b.noise_limited).collect();
        assert!(!flagged.is_empty());
        for b in &flagged {
            assert!(
                b.residual_db.abs() < b.scatter_db / 2.0,
                "band {} looked noisy but its mean was {} dB of bias against {} dB of spread",
                b.centre_hz,
                b.residual_db,
                b.scatter_db
            );
        }
        let lines = report_lines(&r, None).join("\n");
        assert!(lines.contains("Noise-limited"), "{lines}");
        assert!(lines.contains("scatter"), "{lines}");
    }

    #[test]
    fn a_ripply_capture_the_filter_explains_is_not_called_noisy() {
        // The case the discriminator exists to protect. A headphone with real narrow resonances is
        // a ripply trace; if the correction reproduces them, the residual is flat and the fit is
        // right. Scattering the *capture* rather than the *residual* would call every one of those
        // bands noise-limited and teach the reader to ignore the line.
        let ripple = |f: f64| 4.0 * (6.0 * f.log10() * std::f64::consts::TAU).sin();
        let predicted = curve(ripple);
        let post = predicted.clone();
        let r = verify(&post, &predicted, None, &cfg()).expect("curves are usable");
        assert_eq!(
            r.noise_limited, 0,
            "a filter that matched a ripply capture was called noisy"
        );
        assert_eq!(r.fail, 0, "a perfect realisation of a bumpy design failed");
        assert!(r.ok());
        // The capture itself is wildly non-flat inside a band; it is the *residual* that is flat.
        assert!(
            r.bands
                .iter()
                .filter(|b| b.status == Status::Pass)
                .any(|b| b.measured_db.is_finite()),
            "nothing was scored"
        );
    }

    #[test]
    fn a_smooth_capture_is_not_noise_limited() {
        // The other half of the discriminator: without this, "noise-limited" would be a line that
        // always prints and proves nothing.
        let predicted = curve(|_| 0.0);
        let post = curve(|f| -bump(f, 3000.0, 1.5));
        let r = verify(&post, &predicted, None, &cfg()).expect("curves are usable");
        assert_eq!(r.noise_limited, 0, "a smooth curve was called noisy");
        let lines = report_lines(&r, None).join("\n");
        assert!(
            !lines.contains("Noise-limited"),
            "the noise line printed with no noise: {lines}"
        );
    }

    #[test]
    fn a_short_capture_reports_the_bands_it_cannot_reach() {
        let predicted = curve(|_| 0.0);
        let post: Vec<(f64, f64)> = axis()
            .iter()
            .filter(|&&f| (100.0..=6000.0).contains(&f))
            .map(|&f| (f, 0.0))
            .collect();
        let r = verify(&post, &predicted, None, &cfg()).expect("curves are usable");
        assert!(r.uncovered > 0, "the missing range was scored anyway");
        // Coverage is about the band's *edges*: a band whose centre sits at 100 Hz reaches down to
        // 89 Hz, where this capture does not exist, and must not be scored.
        for b in &r.bands {
            let band = assess::bands_below(24000.0)
                .into_iter()
                .find(|g| g.index == b.index)
                .expect("same grid");
            if band.lo_hz >= 100.0 && band.hi_hz <= 6000.0 {
                assert_ne!(
                    b.status,
                    Status::Uncovered,
                    "band {} is inside the capture and was marked uncovered",
                    b.centre_hz
                );
            }
        }
        let lines = report_lines(&r, None).join("\n");
        assert!(lines.contains("outside the compared range"), "{lines}");
    }

    #[test]
    fn the_grid_is_the_grid_the_assessment_uses() {
        // Two report sections naming different bands for one frequency is how band names become
        // decoration. Both must come from the same function.
        let predicted = curve(|_| 0.0);
        let r = verify(&predicted, &predicted, None, &cfg()).expect("curves are usable");
        let grid = bands_below(20000.0);
        assert_eq!(r.bands.len(), grid.len());
        for (b, g) in r.bands.iter().zip(&grid) {
            assert_eq!(b.index, g.index);
            assert_eq!(b.centre_hz, g.centre_hz);
            assert_eq!(b.nominal_hz, g.nominal_hz);
        }
    }

    #[test]
    fn the_design_shortfall_is_visible_even_when_the_rig_is_perfect() {
        // The rig reproduces the design exactly, and the design does not reach the target: the
        // report must still say so, or a pass reads as "the headphone now matches the target".
        let predicted = curve(|_| 0.0);
        let post = predicted.clone();
        let target = curve(|f| bump(f, 2000.0, 5.0));
        let r = verify(&post, &predicted, Some(&target), &cfg()).expect("curves are usable");
        assert!(r.ok(), "a perfect rig was failed: {:?}", r.worst);
        let (hz, d) = r
            .design_worst
            .expect("the design misses the target somewhere");
        assert!((d - 5.0).abs() < 0.6, "shortfall reported as {d} dB");
        assert!(hz > 1500.0 && hz < 2600.0, "shortfall blamed on {hz} Hz");
        let lines = report_lines(&r, None).join("\n");
        assert!(lines.contains("Design shortfall"), "{lines}");
    }

    #[test]
    fn a_capture_too_short_to_interpolate_is_refused() {
        let predicted = curve(|_| 0.0);
        assert!(
            verify(&[(1000.0, 0.0)], &predicted, None, &cfg()).is_none(),
            "one point was enough to interpolate"
        );
        assert!(verify(&[], &predicted, None, &cfg()).is_none());
        assert!(verify(&predicted, &[], None, &cfg()).is_none());
        // Non-finite points are dropped, and a curve left with one point is refused.
        let half_broken = vec![(1000.0, 0.0), (2000.0, f64::NAN)];
        assert!(
            verify(&half_broken, &predicted, None, &cfg()).is_none(),
            "a curve with one finite point was interpolated"
        );
    }

    #[test]
    fn read_capture_follows_the_databases_own_columns() {
        // Two columns: the curve. Three: the mean of the ears. Four and more: the last column,
        // which in the shipped files *is* the average the DSP fits.
        let two = vec![vec![100.0, 1.0], vec![1000.0, 2.0]];
        assert_eq!(read_capture(&two).unwrap()[1], (1000.0, 2.0));
        let three = vec![vec![100.0, 1.0, 3.0], vec![1000.0, 2.0, 6.0]];
        assert_eq!(read_capture(&three).unwrap()[1], (1000.0, 4.0));
        let four = vec![vec![100.0, 1.0, 3.0, -9.0], vec![1000.0, 2.0, 6.0, -7.0]];
        assert_eq!(read_capture(&four).unwrap()[1], (1000.0, -7.0));
        assert!(read_capture(&[vec![100.0], vec![1000.0]]).is_none());
        assert!(read_capture(&[]).is_none());
    }

    #[test]
    fn the_csv_row_for_a_band_carries_the_numbers_the_report_quotes() {
        let predicted = curve(|_| 0.0);
        let f0 = bands_below(24000.0)
            .into_iter()
            .find(|b| b.index == -3)
            .expect("grid")
            .centre_hz;
        let post = curve(|f| bump(f, f0, -4.5));
        let r = verify(&post, &predicted, Some(&curve(|_| 1.0)), &cfg()).expect("usable");
        let csv = csv_text(&r);
        let head = csv.lines().next().expect("header");
        assert!(
            head.starts_with("band_index,centre_hz,nominal_hz,measured_db"),
            "{head}"
        );
        assert_eq!(csv.lines().count(), r.bands.len() + 1);
        let row = csv
            .lines()
            .find(|l| l.starts_with("-3,"))
            .expect("the 125 Hz row");
        let cols: Vec<&str> = row.split(',').collect();
        assert_eq!(*cols.last().expect("status"), "fail");
        let shape: f64 = cols[7].parse().expect("shape column parses");
        assert!((shape + 4.5).abs() < 0.6, "the CSV says {shape} dB");
        assert_eq!(cols[10], "0", "a bias was flagged as noise: {row}");
    }

    #[test]
    fn the_note_line_appears_only_when_the_run_needs_one() {
        let predicted = curve(|_| 0.0);
        let r = verify(&predicted, &predicted, None, &cfg()).expect("usable");
        assert!(
            !report_lines(&r, None).join("\n").contains("Note"),
            "a plain run grew a note"
        );
        let lines = report_lines(&r, Some("the filter was split per ear")).join("\n");
        assert!(
            lines.contains("Note            : the filter was split per ear"),
            "{lines}"
        );
    }
}
