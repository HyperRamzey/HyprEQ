//! What a correction *demands*, as opposed to what it looks like.
//!
//! The report has always printed the biggest boost in the curve, and a boost is a promise
//! about the headphone, not about the music. A `+9 dB` shelf at 40 Hz costs nothing on a
//! podcast and asks a small driver for excursion it does not have on a bass-heavy track.
//! This module joins the two: it puts the applied correction on the same bands the content
//! occupies and reports, per band, how much level the filter is about to move.
//!
//! Everything here is `f64`, infallible and side-effect free: band levels from a transform,
//! the curve sampled onto those bands, and a comparison against a budget. It measures; it
//! does not model a driver, an amplifier or a distortion threshold, and it is not asked to.
//! The output is a claim about a measured quantity — "this filter adds 8.4 dB where the
//! file already has −12 dBFS of content" — which is the sort of claim this code can support.
//!
//! Bands follow **IEC 61260-1**: reference 1 kHz, the base-10 system, so a third-octave
//! step is `10^(1/10)` and a band's edges are its centre times `10^(∓1/20)`. Exact centres
//! are what the maths uses; the report names the band by the standard's nominal value (20,
//! 25, 31.5, 40 … 16 k, 20 k), because nobody reasons about `997.0 Hz`.
//!
//! It is reached from the CLI behind `--assess` only. The default report is byte-compared
//! against the reference program, so a line added there is a parity failure; this section
//! is printed separately for exactly that reason.

use crate::fft::{rfft, Complex64};
use crate::pchip::Pchip;
use crate::wav::comma_sep;

/// Nominal third-octave mantissas within one decade, IEC 61260-1 base-10.
const MANTISSAS: [f64; 10] = [1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0];

/// One third-octave band: where it is, and what the filter does to it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Band {
    /// Band index, IEC 61260-1, with 0 at 1 kHz.
    pub index: i32,
    /// Exact centre, `1000 · 10^(k/10)`, the frequency the maths uses.
    pub centre_hz: f64,
    /// The standard's nominal centre, the name the report prints.
    pub nominal_hz: f64,
    /// Lower edge, `centre · 10^(-1/20)`.
    pub lo_hz: f64,
    /// Upper edge, `centre · 10^(1/20)`.
    pub hi_hz: f64,
    /// Content already in the band, dBFS (windowed RMS).
    pub signal_db: f64,
    /// What the correction applies at the band centre, dB.
    pub eq_db: f64,
    /// `signal_db + eq_db`: where the band sits after the filter, before headroom.
    pub after_db: f64,
}

impl Band {
    /// True when the band carries content worth warning about.
    ///
    /// A large boost into a band the file has nothing in is inert, and listing it as a
    /// demand would train the reader to ignore the section. The floor is deliberately
    /// generous: `-80` dBFS is silence in anything this program is asked to process, and a
    /// band that quiet cannot be driven anywhere.
    #[must_use]
    pub fn carries_content(&self, floor_db: f64) -> bool {
        self.signal_db > floor_db
    }

    /// True when this band's boost is large enough to name, *and* there is content to act
    /// on. Both halves matter: see [`Band::carries_content`].
    #[must_use]
    pub fn over_budget(&self, budget_db: f64, floor_db: f64) -> bool {
        self.eq_db >= budget_db && self.carries_content(floor_db)
    }
}

/// Boost level, in dB, at which a band is named in the report.
pub const DEFAULT_BUDGET_DB: f64 = 6.0;
/// Content level, in dBFS, below which a band is treated as empty.
pub const CONTENT_FLOOR_DB: f64 = -80.0;
/// The standard's range, as band indices: nominal 20 Hz to nominal 20 kHz. The bounds are
/// indices rather than frequencies because the exact centres do not land on the round numbers the
/// series is named by — band `-17` is 19.9526 Hz, not 20.000 Hz.
const LOWEST_K: i32 = -17;
/// Nominal 20 kHz.
const HIGHEST_K: i32 = 13;
/// Samples analysed per channel. Third-octave resolution wants a long window: this is 5.5 s
/// at 48 kHz, and a longer one costs memory for detail the band levels cannot use.
pub const ANALYSIS_WINDOW: usize = 1 << 18;

/// The third-octave grid a signal can actually fill: the standard's nominal 20 Hz through nominal
/// 20 kHz, trimmed at Nyquist for the rate in hand.
///
/// The index range is wide enough for any sample rate up to 192 kHz and trimmed by the
/// window, so a low rate yields few bands rather than an error.
#[must_use]
pub fn bands(fs: u32) -> Vec<Band> {
    let nyquist = f64::from(fs) / 2.0;
    if fs == 0 || !nyquist.is_finite() {
        return Vec::new();
    }
    // The standard's range is 20 Hz to 20 kHz by *nominal* value, so the lower bound is an index
    // bound. Testing the exact centre against 20 Hz drops the 20 Hz band, whose exact centre is
    // 19.9526 Hz, and the grid then started at 25.12 Hz while the module docs and `nominal(-17)`
    // both say 20 Hz. The upper bound stays a frequency test, because a band whose centre is above
    // Nyquist cannot be measured at this sample rate at all.
    (LOWEST_K..=HIGHEST_K)
        .filter_map(|k| {
            let centre = 1000.0 * 10f64.powf(f64::from(k) / 10.0);
            (centre < nyquist).then(|| at_index(k, centre))
        })
        .collect()
}

/// A band at index `k`, with its exact centre, nominal name and edges.
fn at_index(k: i32, centre: f64) -> Band {
    let edge = 10f64.powf(1.0 / 20.0);
    Band {
        index: k,
        centre_hz: centre,
        nominal_hz: nominal(k),
        lo_hz: centre / edge,
        hi_hz: centre * edge,
        signal_db: f64::NEG_INFINITY,
        eq_db: 0.0,
        after_db: f64::NEG_INFINITY,
    }
}

/// The standard's nominal centre for band `k`: the exact `10^(k/10)` value written with the
/// two significant figures the series is named by.
///
/// The mantissa index is `k - 10·floor(k/10)`, not Rust's truncated remainder: at `k = -17`
/// the truncated remainder gives 50 Hz where the standard's table says 20 Hz.
#[must_use]
pub fn nominal(k: i32) -> f64 {
    let decade = k.div_euclid(10);
    let mantissa = MANTISSAS[usize::try_from(k.rem_euclid(10)).unwrap_or(0)];
    mantissa * 10f64.powf(f64::from(decade) + 3.0)
}

/// The correction curve sampled at the band centres, in dB.
///
/// `ctrl` is the same `(freq_hz, gain_dBr)` knot list the FIR is built from, so the numbers
/// describe the filter that is about to be written rather than an approximation of it. A
/// curve with fewer than two usable knots contributes 0 dB everywhere: no correction, no
/// demand, and no `Pchip` panic on a degenerate curve.
#[must_use]
pub fn curve_at(bands: &[Band], ctrl: &[(f64, f64)]) -> Vec<f64> {
    let usable: Vec<(f64, f64)> = ctrl
        .iter()
        .filter(|(f, g)| f.is_finite() && *f > 0.0 && g.is_finite())
        .copied()
        .collect();
    if usable.len() < 2 {
        return vec![0.0; bands.len()];
    }
    let nodes: Vec<f64> = usable.iter().map(|&(f, _)| f).collect();
    let gains: Vec<f64> = usable.iter().map(|&(_, g)| g).collect();
    Pchip::new(&nodes, &gains).eval_grid(&bands.iter().map(|b| b.centre_hz).collect::<Vec<_>>())
}

/// Third-octave RMS levels of one channel, in dBFS.
///
/// One Hann-windowed real transform of at most [`ANALYSIS_WINDOW`] samples, then `|X|²`
/// summed inside each band and divided by `n · Σw²` — Parseval for an unnormalized
/// transform with the window's own energy taken out, which is what makes a full-scale sine
/// read `-3.01 dBFS` instead of a number that depends on the window length. The window is
/// not decoration: without it a tone that misses a bin by a third smears a few dB into its
/// neighbours, and the section's whole job is saying *which* band.
///
/// Interior bins are doubled because a real transform keeps only half the spectrum; DC and
/// Nyquist are not.
#[must_use]
pub fn band_levels(signal: &[f64], fs: u32, bands: &[Band]) -> Vec<f64> {
    if signal.is_empty() || fs == 0 || bands.is_empty() {
        return vec![f64::NEG_INFINITY; bands.len()];
    }
    // Window the record, then pad it — never pad it and window the padding. A taper sized to the
    // transform stops at its peak where the samples run out, which is a rectangular edge into the
    // zeros: measured two bands away from a 1 kHz tone, leakage went from -124 dBFS at an exact
    // power of two to -45 dBFS at 8193 samples, and the level itself read -6.05 instead of -3.01
    // dBFS because `window_energy` integrated the taper over silence as well.
    let keep = signal.len().min(ANALYSIS_WINDOW);
    let n = keep.next_power_of_two().max(16);
    let mut buf = vec![0.0f64; n];

    // Hann, and its energy over the samples that exist, which the normalization divides by.
    let mut window_energy = 0.0f64;
    for i in 0..keep {
        let w = 0.5
            - 0.5
                * (2.0 * std::f64::consts::PI * f64::from(i as u32) / f64::from(keep as u32)).cos();
        buf[i] = signal[i] * w;
        window_energy += w * w;
    }
    if window_energy <= 0.0 {
        return vec![f64::NEG_INFINITY; bands.len()];
    }

    let spec: Vec<Complex64> = rfft(&buf, n);
    let denom = f64::from(n as u32) * window_energy;
    let bin_hz = f64::from(fs) / f64::from(n as u32);

    // Partition, not window. Every bin is given to the band whose centre is nearest in log
    // frequency; for a 1/3-octave grid that boundary is the geometric mean of two adjacent
    // centres, which is exactly each band's edge. Summing the bins whose frequency falls
    // between a band's edges instead would `floor` one edge down and `ceil` the other up,
    // and a bin shared by two bands is then counted twice: measured on a 60 Hz tone, that
    // read the neighbouring band 13 dB high.
    //
    // DC is dropped rather than folded into the lowest band — no third-octave band contains
    // it, and a record-player rumble is not bass content.
    let first = bands[0].index;
    let mut sums = vec![0.0f64; bands.len()];
    for (k, &x) in spec.iter().enumerate().skip(1) {
        let hz = f64::from(k as u32) * bin_hz;
        let idx = (10.0 * (hz / 1000.0).log10()).round() as i32;
        let Ok(pos) = usize::try_from(idx - first) else {
            continue;
        };
        let Some(b) = bands.get(pos) else { continue };
        // The grid is trimmed at both ends, so the nearest centre can lie outside the band
        // that actually exists; those bins belong to no band.
        if hz < b.lo_hz || hz > b.hi_hz {
            continue;
        }
        let power = x.re * x.re + x.im * x.im;
        // A real transform keeps half the spectrum, so interior bins stand for a conjugate
        // pair. DC was skipped and Nyquist is the final bin, which is not doubled.
        let doubled = k < spec.len() - 1;
        sums[pos] += if doubled { 2.0 * power } else { power };
    }

    sums.into_iter()
        .map(|s| {
            let ms = s / denom;
            if ms.is_finite() && ms > 0.0 {
                10.0 * ms.log10()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect()
}

/// The largest boost in the curve and where it sits, and the largest cut.
///
/// Knots that are not a positive frequency with a finite gain are skipped: a curve with a
/// `NaN` in it should report the rest of itself, not a `NaN` extreme.
#[must_use]
fn curve_extremes(ctrl: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let (mut boost, mut boost_hz, mut cut, mut cut_hz) = (0.0, 0.0, 0.0, 0.0);
    for &(f, g) in ctrl {
        if !f.is_finite() || f <= 0.0 || !g.is_finite() {
            continue;
        }
        if g > boost {
            boost = g;
            boost_hz = f;
        }
        if g < cut {
            cut = g;
            cut_hz = f;
        }
    }
    (boost, boost_hz, cut, cut_hz)
}

/// Per-band content level, taking the worst channel.
///
/// Channels are analysed separately and combined by maximum: a narrow spike in one ear is
/// the demanding one, and averaging it with the other hides it. Non-finite content (a `NaN`
/// in a hand-edited file) reads as no content in that band, which cannot be compared.
fn worst_levels(planar: &[Vec<f64>], fs: u32, grid: &[Band]) -> (Vec<f64>, usize) {
    let mut worst = vec![f64::NEG_INFINITY; grid.len()];
    let mut analysed = 0usize;
    for ch in planar {
        analysed = analysed.max(ch.len().min(ANALYSIS_WINDOW));
        if ch.is_empty() {
            continue;
        }
        for (w, &v) in worst.iter_mut().zip(&band_levels(ch, fs, grid)) {
            if v.is_finite() && v > *w {
                *w = v;
            }
        }
    }
    (worst, analysed)
}

/// The gain staging this run will actually apply between the filter and the limiter.
///
/// These four travel together, and the first draft of this report passed three of them and got
/// the arithmetic wrong: the level "after the filter" left out the FIR's own preamp, and the
/// ceiling test then added the headroom a second time. They are one value now so a caller cannot
/// hand over half a staging.
#[derive(Clone, Copy, Debug)]
pub struct Staging {
    /// Steady-state convolver headroom the operator asked for, dB (`--headroom-db`).
    pub headroom_db: f64,
    /// The preamp baked into every FIR tap, dB: `headroom_db - max_boost_db` as computed by
    /// [`crate::ir::synthesize_minimum_phase_ir`]. Pass the field of the result, not a guess.
    pub preamp_db: f64,
    /// The limiter's true-peak ceiling, dBTP (`--tp-ceiling`).
    pub ceiling_db: f64,
    /// Whether a limiter will run to enforce that ceiling.
    pub limiter_on: bool,
}

/// What the filter demands of the content, band by band.
#[derive(Clone, Debug)]
pub struct Demand {
    /// Biggest boost in the curve and where it sits, from the curve alone.
    pub max_boost_db: f64,
    pub max_boost_hz: f64,
    /// Biggest cut in the curve (negative, or 0 for a curve that never cuts).
    pub max_cut_db: f64,
    pub max_cut_hz: f64,
    /// Every band with content, in ascending frequency order.
    pub bands: Vec<Band>,
    /// Bands whose boost exceeded the budget while carrying content.
    pub triggered: Vec<Band>,
    /// The loudest of those bands after the filter — the one that decides what the run
    /// demands. `None` when no band was over budget with content.
    pub peak: Option<Band>,
    /// Bands that reach the ceiling once the convolver's headroom is applied, i.e. the ones
    /// the limiter will be asked to deal with.
    pub at_ceiling: Vec<Band>,
    /// Samples per channel actually analysed.
    pub analysed_samples: usize,
    /// Spacing of the analysis bins, Hz: `fs` over the transform length. Zero when nothing was
    /// analysed. Below about `8.7 · bin_hz` a third-octave band is narrower than one bin, which is
    /// what limits how far down the grid a level can be attributed to a band; see
    /// See `low_frequency_limit`.
    pub bin_hz: f64,
}

/// Join a correction curve to content levels and report the demand.
///
/// `planar` is the input audio, one `Vec` per channel. Channels are analysed separately and
/// combined by maximum, because a narrow spike in one ear is the demanding one and averaging
/// it with the other hides it. An empty channel list — the convolver-only path, which has no
/// audio — still yields the curve extremes and no bands, which is the honest answer: the
/// filter's shape is known, the content is not.
#[must_use]
pub fn assess(
    ctrl: &[(f64, f64)],
    planar: &[Vec<f64>],
    fs: u32,
    budget_db: f64,
    run: &Staging,
) -> Demand {
    let grid = bands(fs);
    let (boost, boost_hz, cut, cut_hz) = curve_extremes(ctrl);
    let mut d = Demand {
        max_boost_db: boost,
        max_boost_hz: boost_hz,
        max_cut_db: cut,
        max_cut_hz: cut_hz,
        bands: Vec::new(),
        triggered: Vec::new(),
        peak: None,
        at_ceiling: Vec::new(),
        analysed_samples: 0,
        bin_hz: 0.0,
    };
    if planar.is_empty() || fs == 0 || grid.is_empty() {
        return d;
    }

    let gains = curve_at(&grid, ctrl);
    let (worst, analysed) = worst_levels(planar, fs, &grid);
    d.analysed_samples = analysed;
    // The spacing the analysis actually achieved, which is what decides how far down the grid a
    // level can be attributed to a band at all. The longest analysed channel sets it, matching
    // the transform `band_levels` runs for that channel.
    let n_fft = analysed.min(ANALYSIS_WINDOW).next_power_of_two().max(16);
    d.bin_hz = f64::from(fs) / f64::from(n_fft as u32);

    for (b, (&signal, &eq)) in grid.iter().zip(worst.iter().zip(&gains)) {
        let mut band = *b;
        band.signal_db = signal;
        band.eq_db = eq;
        // The filter's own preamp belongs in this number. `synthesize_minimum_phase_ir` bakes
        // `headroom_db - max_boost_db` into every tap, so the level the convolver delivers is not
        // `signal + eq`: leaving the preamp out read a -3.01 dBFS tone through the shipped
        // `dt770_80` curve as -4.20 dBFS when the filter actually delivered -9.80 dBFS, and the
        // ceiling test below then added the headroom a second time.
        band.after_db = if signal.is_finite() {
            signal + eq + run.preamp_db
        } else {
            signal
        };
        if band.over_budget(budget_db, CONTENT_FLOOR_DB) {
            d.triggered.push(band);
        }
        // `after_db` already carries the preamp, which *is* the headroom by the time it reaches
        // the limiter, so the comparison is against the ceiling alone.
        if band.after_db.is_finite() && band.after_db >= run.ceiling_db {
            d.at_ceiling.push(band);
        }
        d.bands.push(band);
    }

    d.peak = d
        .triggered
        .iter()
        .max_by(|a, b| cmp_level(a.after_db, b.after_db))
        .copied();
    d
}

/// Compare two levels with `NaN` ordered below everything, so a `max_by` over levels that
/// include a non-finite value still returns an answer rather than nothing.
fn cmp_level(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Less)
}

/// A nominal frequency as people write it: `63 Hz`, `1 kHz`, `3.15 kHz`, with no trailing
/// zeros and no bare decimal point.
#[must_use]
pub fn band_name(nominal_hz: f64) -> String {
    if !nominal_hz.is_finite() || nominal_hz <= 0.0 {
        return "unknown band".to_string();
    }
    let (value, unit) = if nominal_hz >= 1000.0 {
        (nominal_hz / 1000.0, "kHz")
    } else {
        (nominal_hz, "Hz")
    };
    let text = format!("{value:.2}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    format!("{text} {unit}")
}

/// The report lines. Returned rather than printed, so the wording is testable and the call
/// site decides where they go.
///
/// Three facts, in the order a reader acts on them: how many bands the filter is boosting
/// hard into content, which of them is the loudest after the filter, and which will actually
/// reach the ceiling once the convolver's headroom is applied. Only the last one lists
/// bands, because a 1/3-octave analysis reports the same tone in three adjacent bands and a
/// list of all of them is noise.
#[must_use]
pub fn report_lines(d: &Demand, budget_db: f64, run: &Staging) -> Vec<String> {
    let ceiling_db = run.ceiling_db;
    let mut out = Vec::new();
    out.push(format!(
        "=== Correction Demand (budget {budget_db:+.2} dB, preamp {:+.2} dB, ceiling {ceiling_db:.2} dBTP) ===",
        run.preamp_db,
    ));
    out.push(format!(
        "  Curve extreme : boost {}, cut {}",
        extreme_text(d.max_boost_db, d.max_boost_hz),
        extreme_text(d.max_cut_db, d.max_cut_hz)
    ));
    if d.bands.is_empty() {
        out.push(
            "  Content       : no input audio analysed, so these lines describe the filter only"
                .to_string(),
        );
        return out;
    }
    out.push(format!(
        "  Content       : {} samples/channel, 1/3-octave bands, worst channel kept",
        comma_sep(d.analysed_samples as u64)
    ));
    if d.bin_hz > 0.0 && low_frequency_limit(d.bin_hz) > grid_low_hz() {
        out.push(format!(
            "  Resolution    : bin {bin:.2} Hz; below about {limit:.0} Hz a band is narrower than \
             one bin, so levels there name the nearest band",
            bin = d.bin_hz,
            limit = low_frequency_limit(d.bin_hz)
        ));
    }
    // A budget of zero or less is a floor, not an amount of boost, and saying "boost -9.00 dB or
    // more" reads as though every band cleared it — which, at a negative threshold, they do.
    let threshold = if budget_db > 0.0 {
        format!("boost {budget_db:+.2} dB or more")
    } else {
        format!("sit at or above {budget_db:+.2} dB")
    };
    out.push(format!(
        "  Boost demand  : {} band(s) {threshold} into audible content",
        d.triggered.len()
    ));
    if let Some(p) = d.peak {
        out.push(format!(
            "    loudest     : {} -> content {:7.2} dBFS, filter {:+6.2} dB, preamp {:+6.2} dB, {:7.2} dBFS after",
            band_name(p.nominal_hz),
            p.signal_db,
            p.eq_db,
            run.preamp_db,
            p.after_db
        ));
    }
    if d.at_ceiling.is_empty() {
        out.push(format!(
            "  Ceiling       : no band reaches {ceiling_db:.2} dBTP after the filter"
        ));
        return out;
    }
    out.push(format!(
        "  Ceiling       : {} band(s) reach {ceiling_db:.2} dBTP after the filter; {}",
        d.at_ceiling.len(),
        if run.limiter_on {
            "the limiter is doing this work"
        } else {
            "no limiter is running, so this is what gets written"
        }
    ));
    for b in &d.at_ceiling {
        out.push(format!(
            "    {:>9} : {:7.2} dBFS after the filter, {:.2} dB past the ceiling",
            band_name(b.nominal_hz),
            b.after_db,
            b.after_db - ceiling_db
        ));
    }
    out
}

/// One end of the curve, named, or a plain "none" when the curve never goes that way. A curve
/// that never boosts has no frequency to report, and printing the placeholder `0 Hz` read as
/// "boost +0.00 dB @ unknown band".
#[must_use]
fn extreme_text(db: f64, hz: f64) -> String {
    if db == 0.0 || !hz.is_finite() || hz <= 0.0 {
        return "none".to_string();
    }
    format!("{db:+.2} dB @ {}", band_name(nearest_nominal(hz)))
}

/// The frequency below which a third-octave band is narrower than one analysis bin. A band spans
/// `10^(1/20)` to `10^(-1/20)` times its centre, so its width is
/// `10^(1/20) - 10^(-1/20)` ≈ 0.2308 times its centre, and two bins fit at
/// `2 · bin / 0.2308` ≈ 8.7 bins. At the CLI's 48 kHz, 8192-sample analysis (bin 5.86 Hz) that is
/// 51 Hz, which is why the bottom of the grid is indicative and not a measurement.
#[must_use]
fn low_frequency_limit(bin_hz: f64) -> f64 {
    2.0 * bin_hz / (10f64.powf(1.0 / 20.0) - 10f64.powf(-1.0 / 20.0))
}

/// Lowest centre the grid can hold (`1000·10^(-17/10)` = 19.95 Hz). The resolution line is only
/// printed when the limit is above this, because below it no band on the grid is affected.
#[must_use]
fn grid_low_hz() -> f64 {
    1000.0 * 10f64.powf(f64::from(LOWEST_K) / 10.0)
}

/// The nominal centre of the band containing `hz`, for naming a frequency that came from
/// the curve rather than from the grid.
#[must_use]
pub fn nearest_nominal(hz: f64) -> f64 {
    if !hz.is_finite() || hz <= 0.0 {
        return f64::NAN;
    }
    let k = (10.0 * (hz / 1000.0).log10()).round() as i32;
    nominal(k)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The convolver defaults the CLI would use, so the tests describe a real run.
    const HEADROOM: f64 = -1.5;
    const CEILING: f64 = -3.0;

    /// The staging a run applies to `ctrl`: the preamp the FIR synthesis itself would compute,
    /// and a limiter that will enforce the ceiling.
    fn staging(ctrl: &[(f64, f64)]) -> Staging {
        let (boost, _, _, _) = curve_extremes(ctrl);
        Staging {
            headroom_db: HEADROOM,
            preamp_db: HEADROOM - boost,
            ceiling_db: CEILING,
            limiter_on: true,
        }
    }

    /// The level the convolver actually hands the limiter for one band: the content, the curve,
    /// and the preamp the FIR was built with. Every "after the filter" number in this module is
    /// checked against this, because the three terms being summed by hand is where the report
    /// first went wrong.
    fn delivered(b: &Band, run: &Staging) -> f64 {
        b.signal_db + b.eq_db + run.preamp_db
    }

    /// A full-scale sine of `freq` over `n` samples.
    fn sine(freq: f64, fs: u32, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * freq * f64::from(i as u32) / f64::from(fs)).sin()
            })
            .collect()
    }

    /// The grid is IEC 61260-1 base-10: `10^(1/10)` steps, `10^(±1/20)` edges, and the
    /// nominal names from the standard's table rather than the floats between them.
    #[test]
    fn the_grid_is_iec_61260_base_10() {
        let g = bands(48_000);
        assert!(
            g.len() >= 28,
            "20 Hz to 24 kHz is about 30 bands, got {}",
            g.len()
        );
        assert!(
            g.windows(2).all(|w| w[0].index < w[1].index),
            "indices must ascend"
        );
        for w in g.windows(2) {
            let ratio = w[1].centre_hz / w[0].centre_hz;
            assert!(
                (ratio - 10f64.powf(0.1)).abs() < 1e-9,
                "{} -> {} is not a base-10 third octave ({ratio:.6})",
                w[0].centre_hz,
                w[1].centre_hz
            );
        }
        for b in &g {
            let edge = 10f64.powf(1.0 / 20.0);
            assert!(
                (b.hi_hz / b.lo_hz - edge * edge).abs() < 1e-9,
                "band {} is a third of an octave wide",
                b.centre_hz
            );
            // The grid runs to the standard's nominal ends, which are not the same as the exact
            // centres: the lowest band's centre is just *below* 20 Hz, not above it.
            assert!(
                b.centre_hz > 19.9 && b.centre_hz < 24_000.0,
                "band centre {} outside the grid",
                b.centre_hz
            );
            assert!(
                b.nominal_hz >= 20.0 && b.nominal_hz <= 20_000.0,
                "nominal {} outside the standard's range",
                b.nominal_hz
            );
        }
        // A rate too low to hold many bands is not an error, and a zero rate holds none.
        assert!(bands(8_000).len() > 5);
        assert!(bands(0).is_empty());
        // The standard's own table, both sides of 1 kHz and across a decade boundary.
        assert_eq!(nominal(0), 1000.0);
        assert_eq!(nominal(1), 1250.0);
        assert_eq!(nominal(7), 5000.0);
        assert_eq!(nominal(13), 20_000.0);
        // The series near 1 kHz runs 500, 630, 800, 1000: k = -1 is 800 Hz (exact 794.3),
        // not 630. Reading the table by ear is how this one gets written wrong.
        assert_eq!(nominal(-1), 800.0);
        assert_eq!(nominal(-2), 630.0);
        assert_eq!(nominal(-7), 200.0);
        assert_eq!(nominal(-10), 100.0);
        // k = -17 is the one a truncated remainder gets wrong: 20 Hz, not 50.
        assert_eq!(nominal(-17), 20.0);
        assert_eq!(band_name(3150.0), "3.15 kHz");
        assert_eq!(band_name(1000.0), "1 kHz");
        assert_eq!(band_name(63.0), "63 Hz");
        assert_eq!(band_name(20_000.0), "20 kHz");
    }

    #[test]
    fn the_twenty_hertz_band_is_on_the_grid() {
        // The standard names a band at 20 Hz and `nominal(-17)` says so, but its *exact* centre
        // is 1000·10^(-17/10) = 19.9526 Hz, so a grid bounded by "centre > 20 Hz" silently drops
        // it and starts at 25.12 Hz instead.
        let g = bands(48_000);
        let first = &g[0];
        assert_eq!(first.index, -17);
        assert_eq!(first.nominal_hz, 20.0);
        assert!(
            (first.centre_hz - 19.952_623).abs() < 1e-5,
            "exact centre of the 20 Hz band is {}",
            first.centre_hz
        );
        assert!(first.lo_hz < first.centre_hz && first.hi_hz > first.centre_hz);
        assert!(
            g.windows(2).all(|w| w[1].lo_hz > w[0].hi_hz - 1e-9),
            "bands must not overlap once the lowest one is back"
        );

        // The top of the grid is nominal 20 kHz where the rate can hold it, and stops short where
        // it cannot: that bound is a real frequency test, unlike the lower one.
        assert_eq!(
            g.last().unwrap().index,
            13,
            "48 kHz holds the nominal 20 kHz band"
        );
        assert_eq!(bands(44_100).last().unwrap().index, 13);
        assert_eq!(
            bands(8_000).last().unwrap().index,
            6,
            "8 kHz tops out at nominal 4 kHz"
        );
        assert_eq!(bands(8_000)[0].nominal_hz, 20.0);
        assert_eq!(bands(48_000).len(), 31, "k = -17..=13");
    }

    /// A record of any length is measured the same way: same level, same rejection two bands
    /// away. This is the window-and-divisor test for lengths that are not powers of two, which
    /// is where sizing the taper to the transform instead of to the record shows up.
    #[test]
    fn a_record_that_is_not_a_power_of_two_reads_the_same_level() {
        // The taper and its energy have to cover the samples that exist. Windowing the padded
        // FFT length instead stops the record at the peak of the taper — a rectangular edge into
        // silence — so the level falls by about 3 dB and the edge leaks. Every other test in this
        // module used an exact power of two, where the two agree by construction.
        //
        // These lengths are chosen so the arithmetic is not a coincidence either: none is a
        // power of two, 48 000 samples of 1 kHz is not a whole number of periods at the padded
        // length, and 8193 is the case that pushes the transform to the next size.
        let fs = 48_000u32;
        let g = bands(fs);
        let idx = g
            .iter()
            .position(|b| (b.centre_hz - 1000.0).abs() < 1e-9)
            .expect("1 kHz is a grid centre");
        let exact = band_levels(&sine(1000.0, fs, 1 << 13), fs, &g);

        for len in [(1 << 13) + 1, 48_000, (1 << 17) + 1] {
            let got = band_levels(&sine(1000.0, fs, len), fs, &g);
            assert!(
                (got[idx] - exact[idx]).abs() < 0.5,
                "{len} samples read {:.2} dBFS where {} read {:.2}: the window is being applied \
                 over the zero padding",
                got[idx],
                1 << 13,
                exact[idx]
            );
            assert!(
                got[idx + 2] < got[idx] - 40.0,
                "{len} samples leak {:.1} dBFS two bands away from a {:.1} dBFS tone",
                got[idx + 2],
                got[idx]
            );
        }
    }

    /// A tone is found in its own band at the level a full-scale sine really has, and not
    /// in its neighbours. One test covers the band edges, the Parseval divisor and the
    /// window: without the Hann window the neighbour leaks by ~24 dB, and with the wrong
    /// `n · Σw²` normalization the level is off by tens of dB.
    #[test]
    fn a_tone_lands_in_its_own_band_at_its_own_level() {
        let fs = 48_000u32;
        let n = 1 << 13;
        let g = bands(fs);
        let idx = g
            .iter()
            .position(|b| (b.centre_hz - 1000.0).abs() < 1e-9)
            .expect("1 kHz is a grid centre");

        // A full-scale sine is -3.01 dBFS RMS. Tolerances come from measuring 200 in-band
        // tones: the error reaches -0.33 dB at a band edge, so 0.5 dB is the honest bound.
        for tone in [1000.0, 964.0, 1000.0 * 10f64.powf(0.1 / 10.0)] {
            let levels = band_levels(&sine(tone, fs, n), fs, &g);
            assert!(
                (levels[idx] - (-3.0103)).abs() < 0.5,
                "tone {tone} Hz read {:.2} dBFS in its band, want about -3.01",
                levels[idx]
            );
        }

        // A tone centred in its band must not show up two bands away. Measured with Hann:
        // 74 dB down or more; 40 dB leaves room without hiding a missing window.
        let levels = band_levels(&sine(1000.0, fs, n), fs, &g);
        for k in [idx - 2, idx + 2] {
            assert!(
                levels[k] < levels[idx] - 40.0,
                "band {k} holds {:.1} dBFS against the tone's {:.1}: leakage unwindowed?",
                levels[k],
                levels[idx]
            );
        }

        // Silence is "no content", not -0 dBFS, or every band would look full.
        let quiet = band_levels(&vec![0.0; n], fs, &g);
        assert!(quiet.iter().all(|&v| v == f64::NEG_INFINITY));
    }

    /// Nothing is demanded where nothing was changed.
    #[test]
    fn a_flat_curve_demands_nothing() {
        let fs = 48_000u32;
        let ctrl = vec![(20.0, 0.0), (1000.0, 0.0), (20_000.0, 0.0)];
        let bass = sine(60.0, fs, 1 << 14);
        let d = assess(
            &ctrl,
            std::slice::from_ref(&bass),
            fs,
            DEFAULT_BUDGET_DB,
            &staging(&ctrl),
        );
        assert_eq!(d.max_boost_db, 0.0);
        assert_eq!(d.max_cut_db, 0.0);
        assert!(
            d.triggered.is_empty(),
            "a flat curve triggered {} bands",
            d.triggered.len()
        );
        assert!(
            !d.bands.is_empty(),
            "content was supplied, so bands are expected"
        );
        for b in &d.bands {
            assert!(b.eq_db.abs() < 1e-9, "curve contributed {} dB", b.eq_db);
            if b.signal_db.is_finite() {
                // The curve contributes nothing, and yet the level the convolver delivers is
                // 1.5 dB lower: a flat curve has 0 dB of max boost, so its preamp is the whole
                // steady-state headroom. "after" is a delivered level, not a statement about
                // the curve, and pretending otherwise is what made the ceiling test double-count
                // the headroom.
                assert!(
                    (b.after_db - delivered(b, &staging(&ctrl))).abs() < 1e-9,
                    "after must be content + curve + preamp: {} vs {}",
                    b.after_db,
                    delivered(b, &staging(&ctrl))
                );
                assert!(
                    (b.after_db - (b.signal_db + HEADROOM)).abs() < 1e-9,
                    "a flat curve leaves only the preamp: {} vs {}",
                    b.after_db,
                    b.signal_db + HEADROOM
                );
            }
        }
        let lines = report_lines(&d, DEFAULT_BUDGET_DB, &staging(&ctrl));
        assert!(
            lines.iter().any(|l| l.contains("0 band(s) boost")),
            "{lines:?}"
        );
    }

    /// The case the section exists for: a bass shelf is only a demand because the file has
    /// bass in it.
    #[test]
    fn a_bass_shelf_is_only_a_demand_when_the_file_has_bass() {
        let fs = 48_000u32;
        // +9 dB to 100 Hz, flat above.
        let ctrl = vec![
            (20.0, 9.0),
            (60.0, 9.0),
            (100.0, 9.0),
            (200.0, 0.0),
            (1000.0, 0.0),
            (20_000.0, 0.0),
        ];

        let bass = sine(60.0, fs, 1 << 14);
        let d = assess(
            &ctrl,
            std::slice::from_ref(&bass),
            fs,
            DEFAULT_BUDGET_DB,
            &staging(&ctrl),
        );
        assert!(
            (d.max_boost_db - 9.0).abs() < 1e-12,
            "curve max boost {}",
            d.max_boost_db
        );
        // A 1/3-octave analysis of one 60 Hz tone reports it across the neighbouring bands
        // as well (Hann's main lobe is two bins wide and the tone sits 4 Hz from a band
        // edge), so "triggered" counts more than one band. What must be unambiguous is
        // *which* band the demand belongs to, and that is the peak.
        assert!(
            !d.triggered.is_empty(),
            "a 60 Hz tone with +9 dB below 100 Hz must warn"
        );
        let b = d.peak.expect("one band must be the loudest");
        assert_eq!(
            b.nominal_hz, 63.0,
            "the tone's own band must be the peak, got {}",
            b.nominal_hz
        );
        assert!(b.eq_db > 8.5, "filter gain at the band was {}", b.eq_db);
        // What the curve *demands* is +9 dB; what the convolver *delivers* is that demand minus
        // its own preamp, and for this curve the preamp is `headroom - max_boost` = -1.5 - 9 =
        // -10.5 dB, so a -3 dBFS tone lands at about -4.5 dBFS. The old expectation of "+6 dBFS"
        // was the curve sum with the preamp left out — a level no shipped FIR outputs.
        let run = staging(&ctrl);
        assert!(
            (b.after_db - delivered(&b, &run)).abs() < 1e-9,
            "after is content + filter + preamp, got {}",
            b.after_db
        );
        assert!(
            (b.after_db + 4.5).abs() < 0.6,
            "a 9 dB shelf pays for itself out of the headroom, got {:.2} dBFS",
            b.after_db
        );
        assert!(
            d.at_ceiling.is_empty(),
            "that is exactly what the preamp is for: {:?}",
            d.at_ceiling
        );
        let quiet = report_lines(&d, DEFAULT_BUDGET_DB, &run);
        assert!(
            quiet.iter().any(|l| l.contains("no band reaches")),
            "{quiet:?}"
        );

        // The ceiling question still fires when the ceiling is set below what the preamp leaves
        // behind, and it is asked of the delivered level: -4.5 dBFS is 1.5 dB past -6.0.
        let tight = Staging {
            ceiling_db: -6.0,
            ..run
        };
        let hot = assess(
            &ctrl,
            std::slice::from_ref(&bass),
            fs,
            DEFAULT_BUDGET_DB,
            &tight,
        );
        assert_eq!(
            hot.at_ceiling.len(),
            1,
            "only the tone's band clears a -6 dBTP ceiling: {:?}",
            hot.at_ceiling
        );
        assert_eq!(hot.at_ceiling[0].nominal_hz, 63.0);
        let hot_lines = report_lines(&hot, DEFAULT_BUDGET_DB, &tight);
        assert!(
            hot_lines.iter().any(|l| l.contains("1.39 dB past the ceiling")),
            "the margin is measured against the ceiling alone, not headroom-then-ceiling: {hot_lines:?}"
        );
        assert!(
            hot_lines
                .iter()
                .any(|l| l.contains("the limiter is doing this work")),
            "{hot_lines:?}"
        );
        let no_lim = Staging {
            limiter_on: false,
            ..tight
        };
        let off_lines = report_lines(&hot, DEFAULT_BUDGET_DB, &no_lim);
        assert!(
            off_lines
                .iter()
                .any(|l| l.contains("no limiter is running, so this is what gets written"))
                && !off_lines
                    .iter()
                    .any(|l| l.contains("the limiter is doing this work")),
            "--no-tp-limiter must not be credited with work it is not doing: {off_lines:?}"
        );

        let lines = report_lines(&d, DEFAULT_BUDGET_DB, &staging(&ctrl));
        assert!(
            lines.iter().any(|l| l.contains("Boost demand")),
            "{lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|l| l.contains("7 band(s) boost +6.00 dB or more")),
            "the budget is printed with its sign, and only real boosts count: {lines:?}"
        );

        // The same filter against content with nothing in the bass is inert. The boost is
        // still in the curve — the extreme line still says so — but there is no demand.
        let treble = sine(8000.0, fs, 1 << 14);
        let d2 = assess(
            &ctrl,
            std::slice::from_ref(&treble),
            fs,
            DEFAULT_BUDGET_DB,
            &staging(&ctrl),
        );
        assert!(
            d2.triggered.is_empty(),
            "a bass boost on bass-free content is not a demand: {:?}",
            d2.triggered
        );
        assert!(
            (d2.max_boost_db - 9.0).abs() < 1e-12,
            "the curve extreme is still reported"
        );
        // And the budget is a knob, not a constant: at +8.5 dB the bass band still speaks.
        // The budget is a knob, not a constant.
        let d3 = assess(&ctrl, std::slice::from_ref(&bass), fs, 8.5, &staging(&ctrl));
        assert_eq!(
            d3.triggered.len(),
            d.triggered.len(),
            "8.5 dB still catches the shelf"
        );
        let d4 = assess(&ctrl, std::slice::from_ref(&bass), fs, 9.5, &staging(&ctrl));
        assert!(
            d4.triggered.is_empty(),
            "above the shelf there is nothing to warn about"
        );
        assert!(d4.peak.is_none());
    }

    /// Two channels, one hot in the bass and one not: the demanding ear is the answer.
    #[test]
    fn the_worst_channel_is_the_one_that_matters() {
        let fs = 48_000u32;
        let ctrl = vec![(20.0, 9.0), (60.0, 9.0), (200.0, 0.0), (20_000.0, 0.0)];
        let hot = sine(60.0, fs, 1 << 14);
        let mut quiet = hot.clone();
        for v in quiet.iter_mut() {
            *v *= 0.001; // -60 dB
        }
        let both = vec![quiet, hot.clone()];
        let d = assess(&ctrl, &both, fs, DEFAULT_BUDGET_DB, &staging(&ctrl));
        assert!(!d.triggered.is_empty(), "the hot channel should trigger");
        let peak = d.peak.expect("a peak band");
        assert!(
            peak.signal_db > -10.0,
            "worst channel kept: {}",
            peak.signal_db
        );
        assert_eq!(peak.nominal_hz, 63.0);

        // A ragged pair (one channel shorter) is analysed as far as it goes, not panicked on.
        let short = vec![hot.clone(), vec![0.5; 7]];
        let d2 = assess(&ctrl, &short, fs, DEFAULT_BUDGET_DB, &staging(&ctrl));
        assert!(!d2.bands.is_empty(), "bands are still produced");
    }

    /// A budget is a signed threshold, and the sentence has to read correctly on both sides of
    /// zero. A negative budget is a floor: "boost -9.00 dB or more" is true of every band and
    /// tells the reader nothing, and the header used to print `+-9.00 dB`.
    #[test]
    fn a_negative_budget_is_phrased_as_a_floor() {
        let fs = 48_000u32;
        let ctrl = vec![(60.0, 9.0), (100.0, 9.0), (1000.0, 0.0)];
        let bass = sine(60.0, fs, 1 << 14);
        let run = staging(&ctrl);
        let d = assess(&ctrl, std::slice::from_ref(&bass), fs, -9.0, &run);
        let lines = report_lines(&d, -9.0, &run);
        let text = lines.join("\n");
        assert!(!text.contains("+-"), "sign printed twice: {text}");
        assert!(
            text.contains("budget -9.00 dB"),
            "the header must carry the real sign: {text}"
        );
        assert!(
            text.contains("sit at or above -9.00 dB"),
            "a negative budget is a floor: {text}"
        );
        assert!(!text.contains("boost -9.00 dB or more"), "{text}");
        assert!(
            !text.contains("unknown band"),
            "a band name is a band name: {text}"
        );

        // And a curve that only cuts has no boost to name, and says so rather than blaming the
        // `0 Hz` placeholder for it.
        let cuts = vec![(1000.0, -4.0), (6000.0, -9.0)];
        let flat = vec![(100.0, 0.0), (10_000.0, 0.0)];
        for (curve, want) in [(&cuts, "boost none"), (&flat, "boost none")] {
            let d = assess(
                curve,
                std::slice::from_ref(&bass),
                fs,
                DEFAULT_BUDGET_DB,
                &staging(curve),
            );
            let text = report_lines(&d, DEFAULT_BUDGET_DB, &staging(curve)).join("\n");
            assert!(text.contains(want), "{curve:?} -> {text}");
            assert!(!text.contains("unknown band"), "{curve:?} -> {text}");
        }
        let d = assess(
            &cuts,
            std::slice::from_ref(&bass),
            fs,
            DEFAULT_BUDGET_DB,
            &staging(&cuts),
        );
        let text = report_lines(&d, DEFAULT_BUDGET_DB, &staging(&cuts)).join("\n");
        assert!(
            text.contains("cut -9.00 dB @ 6.3 kHz"),
            "a real cut is still named: {text}"
        );
        // A flat curve has neither extreme, so neither is named.
        let d = assess(
            &flat,
            std::slice::from_ref(&bass),
            fs,
            DEFAULT_BUDGET_DB,
            &staging(&flat),
        );
        let text = report_lines(&d, DEFAULT_BUDGET_DB, &staging(&flat)).join("\n");
        assert!(
            text.contains("boost none, cut none"),
            "a bypass curve claims nothing: {text}"
        );
    }

    /// How far down the grid a level can be *attributed* to a band is limited by the analysis,
    /// not by the standard: a third-octave band is 0.159 · centre wide, so below about 12.6 bins
    /// of frequency there is not a bin inside it. This is documented behaviour, and the two cases
    /// are pinned so a change in the window or the grid cannot silently move the boundary.
    #[test]
    fn a_low_tone_is_named_by_the_nearest_band_when_the_bins_are_coarse() {
        let fs = 48_000u32;
        let g = bands(fs);
        let at = |nominal_hz: f64| {
            g.iter()
                .position(|b| b.nominal_hz == nominal_hz)
                .expect("band on the grid")
        };
        let (b25, b315) = (at(25.0), at(31.5));

        // 8192 samples at 48 kHz: bin 5.86 Hz, and the 25 Hz band is 22.28..28.05 Hz wide, which
        // is 0.98 bins. A 26.5 Hz tone is inside that band and has no bin inside it — the nearest
        // bin is 29.30 Hz, which belongs to the 31.5 Hz band. That is the resolution limit, not a
        // misassignment, and the report says so on its own line.
        let levels = band_levels(&sine(26.5, fs, 1 << 13), fs, &g);
        // Measured: the 31.5 Hz band reads -5.9 dBFS and the 25 Hz band -6.3 dBFS. The tone's
        // main lobe covers the bins at 23.4 and 29.3 Hz, which belong to those two bands, so the
        // energy splits and each band reads about 3 dB under the tone's true -3.01 dBFS. The
        // partition is not picking a winner here; there simply is no bin inside 22.28..28.05 Hz.
        assert!(
            (levels[b315] - levels[b25]).abs() < 2.0,
            "a band-edge tone at this resolution spreads over both neighbours: {} vs {}",
            levels[b315],
            levels[b25]
        );
        assert!(
            levels[b25] < -4.5 && levels[b315] < -4.5,
            "and both read it low: {} / {} against a true -3.01",
            levels[b25],
            levels[b315]
        );

        // The same tone with the full analysis window: bin 0.183 Hz, dozens of bins inside the
        // band, and the level lands where the standard says it belongs.
        let levels = band_levels(&sine(26.5, fs, ANALYSIS_WINDOW), fs, &g);
        assert!(
            levels[b25] > levels[b315] + 20.0,
            "fine bins: 25 Hz band {:.1} dBFS, 31.5 Hz band {:.1} dBFS",
            levels[b25],
            levels[b315]
        );
        assert!(
            (levels[b25] + 3.01).abs() < 0.5,
            "and at that resolution the level is the tone's own: {:.2} dBFS",
            levels[b25]
        );

        // The report states the limit when it applies, and stays quiet when it does not.
        let ctrl = vec![(25.0, 7.0), (1000.0, 0.0)];
        let run = staging(&ctrl);
        let short = sine(26.5, fs, 1 << 13);
        let d = assess(
            &ctrl,
            std::slice::from_ref(&short),
            fs,
            DEFAULT_BUDGET_DB,
            &run,
        );
        let text = report_lines(&d, DEFAULT_BUDGET_DB, &run).join("\n");
        assert!(
            text.contains("Resolution    : bin 5.86 Hz; below about 51 Hz"),
            "{text}"
        );
        let long = sine(26.5, fs, ANALYSIS_WINDOW);
        let d = assess(
            &ctrl,
            std::slice::from_ref(&long),
            fs,
            DEFAULT_BUDGET_DB,
            &run,
        );
        let text = report_lines(&d, DEFAULT_BUDGET_DB, &run).join("\n");
        assert!(
            !text.contains("Resolution"),
            "a 0.18 Hz bin needs no caveat: {text}"
        );
        assert!(d.bin_hz < 0.2, "long window bin {}", d.bin_hz);
    }

    /// No audio is an answer, not a crash: the filter's shape is known, the content is not.
    #[test]
    fn without_audio_the_curve_is_still_described() {
        let ctrl = vec![(20.0, -6.0), (1000.0, 3.0), (20_000.0, 0.0)];
        let d = assess(&ctrl, &[], 48_000, DEFAULT_BUDGET_DB, &staging(&ctrl));
        assert!(d.bands.is_empty());
        assert!(d.triggered.is_empty());
        assert!((d.max_boost_db - 3.0).abs() < 1e-12);
        assert!((d.max_cut_db + 6.0).abs() < 1e-12);
        assert!((nearest_nominal(d.max_boost_hz) - 1000.0).abs() < 1.0);
        let lines = report_lines(&d, DEFAULT_BUDGET_DB, &staging(&ctrl));
        assert!(lines.iter().any(|l| l.contains("no input audio analysed")));
    }

    /// Degenerate inputs are the ones that turn into panics and NaNs.
    #[test]
    fn degenerate_input_produces_numbers_not_panics() {
        let ctrl = vec![(20.0, 3.0), (1000.0, 3.0)];
        // A zero sample rate has no bands to speak of, and must not divide by it.
        let d = assess(
            &ctrl,
            &[vec![0.1, 0.2, 0.3]],
            0,
            DEFAULT_BUDGET_DB,
            &staging(&ctrl),
        );
        assert!(d.bands.is_empty(), "no rate, no bands");

        // One sample, and a channel of NaNs.
        let one = sine(1000.0, 48_000, 1);
        let d1 = assess(
            &ctrl,
            std::slice::from_ref(&one),
            48_000,
            DEFAULT_BUDGET_DB,
            &staging(&ctrl),
        );
        assert!(d1
            .bands
            .iter()
            .all(|b| b.signal_db.is_finite() || b.signal_db == f64::NEG_INFINITY));

        let nans = vec![f64::NAN; 1 << 12];
        let d2 = assess(
            &ctrl,
            std::slice::from_ref(&nans),
            48_000,
            DEFAULT_BUDGET_DB,
            &staging(&ctrl),
        );
        assert!(
            d2.bands.iter().all(|b| !b.signal_db.is_nan()),
            "NaN content must read as no content, not as a demand"
        );
        assert!(d2.triggered.is_empty(), "and must not be warned about");

        // A lone knot is not a curve, and must not steer the filter gain.
        let d3 = assess(
            &[(1000.0, 9.0)],
            &[vec![0.5; 1 << 12]],
            48_000,
            DEFAULT_BUDGET_DB,
            &staging(&[(1000.0, 9.0)]),
        );
        assert!(
            d3.bands.iter().all(|b| b.eq_db == 0.0),
            "one knot cannot define a correction"
        );
        // A curve with junk in it is filtered out rather than propagated.
        let junk = vec![
            (f64::NAN, 3.0),
            (-40.0, 3.0),
            (0.0, 5.0),
            (20.0, f64::INFINITY),
            (100.0, 2.0),
            (2000.0, -1.0),
        ];
        let d4 = assess(
            &junk,
            &[vec![0.5; 1 << 12]],
            48_000,
            DEFAULT_BUDGET_DB,
            &staging(&junk),
        );
        assert!(
            d4.bands.iter().all(|b| b.eq_db.is_finite()),
            "non-finite curve values leaked"
        );
    }

    /// A frequency between bands is named by the band it belongs to.
    #[test]
    fn a_bare_frequency_is_named_by_its_band() {
        assert_eq!(nearest_nominal(1000.0), 1000.0);
        assert_eq!(nearest_nominal(1150.0), 1250.0);
        assert_eq!(nearest_nominal(60.0), 63.0);
        assert_eq!(nearest_nominal(19_000.0), 20_000.0);
        assert!(nearest_nominal(0.0).is_nan());
        assert_eq!(band_name(nearest_nominal(60.0)), "63 Hz");
    }
}
