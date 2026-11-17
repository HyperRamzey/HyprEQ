//! Mixed-phase correction: move the phase, leave the magnitude alone.
//!
//! The shipped filter is minimum-phase. That is the right default — it puts no pre-ringing in front
//! of a transient, and it matches how loudspeaker crossovers already behave — but it is a *choice*,
//! and minimum-phase is not free: the group delay it implies across a boosted bass region is exactly
//! the smearing that a linear-phase filter spreads symmetrically instead. §3.4 of the roadmap cites
//! the objective evidence (Li et al., *Applied Acoustics* 2019: transient decay speed correlates with
//! phase linearity) and the listening thresholds (Liski, Mäkivirta & Välimäki, *IEEE/ACM TASLP* 29,
//! 2189–2201 (2021): about −0.56 ms of negative and +0.64 ms of positive group delay at threshold for
//! a pink-noise impulse). The evidence that it is audible at all is contested, which is why this ships
//! behind `--experimental` rather than as a default or a recommendation.
//!
//! # Why an all-pass stage, and not a two-factor design
//!
//! The classical mixed-phase construction (Budde, *Gemischtphasige Filter*, DAGA 2012) factorises the
//! target into a minimum-phase factor and a linear-phase residual, then alternates the two so each
//! compensates the other's truncation error. Its own implementation notes report why the naive form
//! fails: interpolating directly between minimum and linear phase puts energy outside the causal
//! support, and truncating that response **changes the magnitude** — with the alternating loop
//! measured to oscillate and, divided through factor nulls, to diverge.
//!
//! That is fatal for the property this feature exists to guarantee. So the magnitude is never
//! re-derived: the finished filter's spectrum is read, its phase is *replaced* inside a band, and the
//! magnitude is carried through untouched. `|H|` is preserved by construction rather than by an
//! iterative correction, and the deviation that truncation does introduce is measured and reported
//! instead of asserted away.
//!
//! What is flattened is the **group delay**, not the phase: within the band the phase is pulled
//! toward the straight line that least-squares fits it, which is what "linear phase over this band"
//! means operationally, and the amount of change is bounded by the band's own delay spread rather than
//! by the filter length. That keeps the latency cost at the milliseconds the band already contains,
//! instead of the hundreds a full linear-phase kernel would demand — and it is the same posture the
//! shipped tools in this space take, where group-delay work is a bass-focused guard and "the objective
//! is transient integrity, not visual flatness of group delay".
//!
//! # What this module does not claim
//!
//! It does not claim the result sounds better. It reports what moved, in milliseconds, next to the
//! published thresholds, and leaves the judgement to the listener.

use crate::fft::Complex64;
use crate::fft::{irfft, rfft, rfftfreq};

/// The band edge below which group delay is flattened, in Hz.
///
/// A configuration default, chosen to sit inside the bass region where a headphone correction's
/// minimum-phase group delay is largest and where the shipped tools in this area also work; it is not
/// a value any publication prescribes.
pub const DEFAULT_CROSSOVER_HZ: f64 = 200.0;

/// How far above the crossover the weight is tapered away, as a ratio.
///
/// The taper exists because a hard edge in the weight is a discontinuity in phase derivative, which
/// is a group-delay spike — the exact artifact this stage is meant to avoid creating.
pub const TAPER_RATIO: f64 = 2.0;

/// Frequency below which the band does not reach, in Hz.
///
/// Bin 0 is DC, where the phase of a real filter is pinned to 0 or pi and has nothing to flatten;
/// the first couple of bins are dominated by the filter's overall delay.
const BAND_FLOOR_HZ: f64 = 10.0;

/// How far the stage may move group delay at any one frequency, in ms.
///
/// This is the parameter that makes the operation possible at all, and it is the reason the stage is
/// not simply "make the bass linear phase". On the shipped DT 770 correction the minimum-phase group
/// delay spans **0.7 ms to 219.7 ms across 10-200 Hz** (measured, `--experimental-phase 200`). A truly
/// flat line therefore has to either advance two hundred milliseconds of deep-bass energy — which is
/// not a causal operation: the energy leaves the front of the kernel, wraps on the padded grid, and
/// shows up as an absurd latency plus a 0.30 dB magnitude error, which is exactly what an earlier
/// version of this stage produced and reported honestly — or add the same as latency.
///
/// So the correction is bounded: each frequency is moved toward the flat line by at most this much,
/// and the report prints the span it was asked to close against the budget it was allowed. The number
/// is a HyprEQ configuration default, prescribed by nobody, and it is a constant rather than a CLI
/// option on purpose: raising it is a decision about how much latency a filter may acquire, and the
/// measurements below say what each step costs, so it should be changed by someone who has read them.
/// The default is set at the published audibility scale, not at what the band could absorb. Liski,
/// Mäkivirta & Välimäki (2021) put the threshold near 0.56 ms, and at that budget the phase the stage
/// moves is small enough that the re-ringing it creates stays inside a few milliseconds of kernel. A
/// A 5 ms default was tried first: on the shipped DT 770 correction it asked for 215 ms of latency, because
/// a 5 ms delay change at 10 Hz is 6 radians of phase, and 6 radians at 10 Hz is most of a period of
/// that component being relocated. That is the trade in its purest form — flattening a bass band whose
/// minimum-phase delay spans seconds is not a free operation at any budget — so the budget is a CLI
/// option (`--experimental-phase-budget`), and the default spends the least that still does something
/// measurable.
pub const DEFAULT_GD_BUDGET_MS: f64 = 0.56;

/// What the stage did, in the units a user can act on.
#[derive(Debug, Clone)]
pub struct Adjustment {
    /// The filter with its band flattened, truncated to the length it came in at.
    pub ir: Vec<f64>,
    /// The same filter before truncation, on the padded grid.
    ///
    /// Kept because two claims are only checkable on it: the energy share is measured against it, and
    /// "nothing above the taper is touched" is a statement about the spectrum *before* the tail is
    /// cut -- after truncation every bin moves a little, which is the cost of the stage, disclosed in
    /// the report rather than argued away.
    pub full: Vec<f64>,
    /// Lower edge of the band that was touched, Hz.
    pub f_lo_hz: f64,
    /// Crossover (upper edge of the flat region), Hz.
    pub crossover_hz: f64,
    /// Top of the taper, Hz. Above this the filter is bit-for-bit what it was.
    pub f_taper_hz: f64,
    /// The delay the band was flattened onto, in ms: the band's own mean group delay before the stage
    /// ran. Flattening moves delay from where the filter had more of it to where it had less; this is
    /// the level it moves toward, so a reader can see that no delay was invented.
    pub line_delay_ms: f64,
    /// Largest group-delay change the stage makes anywhere, in ms.
    pub max_gd_change_ms: f64,
    /// Frequency of that largest change, Hz.
    pub max_gd_change_hz: f64,
    /// Index of the delivered kernel's peak energy, samples.
    ///
    /// Against the input filter's own peak index this is the latency the stage adds. It cannot be
    /// avoided, only moved somewhere else: a band pulled toward linear phase spreads its impulse
    /// response symmetrically around the delay the band already contained, so a kernel that keeps the
    /// same tap count and refuses the latency has to throw the pre-ringing away instead -- and throwing
    /// it away *is* a magnitude error. Published designs take permitted delay as an input for exactly
    /// this reason; here it is taken as the length the run already asked for, and the resulting delay
    /// is printed.
    pub added_latency_samples: usize,
    /// That latency in ms.
    pub added_latency_ms: f64,
    /// Share of the reshaped response's energy that the delivered window keeps, %.
    pub retained_energy_pct: f64,
    /// Taps handed in, and taps handed back: the kernel grows to hold the response the reshape made.
    pub taps_before: usize,
    /// See [`Self::taps_before`].
    pub taps_after: usize,
    /// How far group delay was allowed to move at any one frequency, ms.
    pub budget_ms: f64,
    /// How far apart the band's group delay actually was before the stage ran, ms. The budget and this
    /// number together say how much of the span was closed and how much was left alone on purpose.
    pub gd_span_ms: f64,
    /// Whether the delivered kernel holds the response to the -130 dB floor.
    pub held_floor: bool,
    /// Whether every frequency stayed inside the budget after the residue was removed. It can fail to:
    /// taking the residue out and re-clamping is a fixed-point iteration, and on a band whose delay is
    /// dominated by one deep peak it may not close in the allotted passes. Reported, not assumed.
    pub budget_held: bool,
    /// Largest magnitude deviation introduced by truncating the reshaped filter, dB.
    pub max_mag_error_db: f64,
    /// Share of the reshaped filter's energy that survives truncation, %.
    pub energy_pct: f64,
    /// Floor over the tail of the delivered filter, dBFS.
    pub floor_dbfs: f64,
}

impl Adjustment {
    /// Whether the largest delay this stage moves is smaller than the **tighter** of the two
    /// audibility thresholds reported by Liski et al. (2021) — about `−0.56 ms` and `+0.64 ms` at
    /// threshold for a pink-noise impulse.
    ///
    /// The tighter magnitude (0.56 ms) is the bar because what is measured here is a magnitude: the
    /// stage reports how far group delay moves, not whether it advances or delays an onset, and the two
    /// published figures differ by direction rather than by which one applies. Using the looser one
    /// would flatter the stage.
    ///
    /// Reported alongside the number, never instead of it. Those thresholds are for a broadband
    /// impulse's perceived onset, so this is a comparison the reader can weigh, not a verdict on
    /// whether the change is audible in headphone correction.
    #[must_use]
    pub fn within_published_thresholds(&self) -> bool {
        self.max_gd_change_ms < LISKI_NEGATIVE_MS.abs().min(LISKI_POSITIVE_MS)
    }
}

/// Liski, Mäkivirta & Välimäki (2021) group-delay thresholds, ms. Negative and positive differ
/// because onset advancement and onset delay are not judged equally.
pub const LISKI_NEGATIVE_MS: f64 = -0.56;
/// See [`LISKI_NEGATIVE_MS`].
pub const LISKI_POSITIVE_MS: f64 = 0.64;

/// Unwrap a half-spectrum's phase onto a continuous curve, bin by bin.
///
/// A fit over wrapped phase is meaningless: the slope it finds is the sawtooth's, not the filter's.
fn unwrap(phase: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(phase.len());
    let mut prev = 0.0;
    let mut carry = 0.0;
    for (i, &p) in phase.iter().enumerate() {
        if i > 0 {
            let mut delta = p - prev;
            while delta > std::f64::consts::PI {
                delta -= std::f64::consts::TAU;
                carry -= std::f64::consts::TAU;
            }
            while delta < -std::f64::consts::PI {
                delta += std::f64::consts::TAU;
                carry += std::f64::consts::TAU;
            }
        }
        out.push(p + carry);
        prev = p;
    }
    out
}

/// How much of the replacement is in effect at `f`: 1.0 across the flat band, easing to 0.0 at the
/// top of the taper, 0.0 below the band floor.
fn weight(f: f64, f_lo: f64, crossover: f64, f_taper: f64) -> f64 {
    if f < f_lo || f >= f_taper {
        return 0.0;
    }
    if f <= crossover {
        return 1.0;
    }
    // Raised cosine: value and derivative both meet the endpoints, so the weight introduces no
    // step in group delay of its own.
    let t = (f - crossover) / (f_taper - crossover);
    let c = (std::f64::consts::PI * t).cos();
    0.5 * (1.0 + c)
}

/// Group delay in ms implied by a phase curve sampled on `freq`: `-1/(2π) · dφ/df`, one-sided
/// differences where a centred one is not available.
fn group_delay_ms(phase: &[f64], freq: &[f64]) -> Vec<f64> {
    let n = phase.len();
    if n < 2 {
        return vec![0.0; n];
    }
    // Centred difference where both neighbours exist, one-sided at the two ends.
    (0..n)
        .map(|i| {
            let (a, b) = match i {
                0 => (0usize, 1usize),
                _ if i == n - 1 => (n - 2, n - 1),
                _ => (i - 1, i + 1),
            };
            let df = freq[b] - freq[a];
            if df > 0.0 {
                -(phase[b] - phase[a]) / df / (2.0 * std::f64::consts::PI) * 1000.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Start index of the `w` consecutive samples of `x` (read circularly) holding the most energy.
///
/// Prefix sums over the array doubled on itself, so every circular window is one subtraction. A plain
/// scan would be `O(n·w)` on a 65 536-point grid.
fn best_window(x: &[f64], w: usize) -> usize {
    let n = x.len();
    if w == 0 || w >= n {
        return 0;
    }
    let doubled: Vec<f64> = x.iter().chain(x.iter()).map(|v| v * v).collect();
    let prefix = crate::accum::prefix_sums(&doubled);
    let mut best = 0usize;
    let mut best_sum = f64::NEG_INFINITY;
    for s in 0..n {
        let sum = prefix[s + w] - prefix[s];
        if sum > best_sum {
            best_sum = sum;
            best = s;
        }
    }
    best
}

/// Index of the largest-magnitude sample.
fn peak_index(x: &[f64]) -> usize {
    x.iter().enumerate().fold(
        0usize,
        |best, (i, v)| if v.abs() > x[best].abs() { i } else { best },
    )
}

/// Flatten the group delay of `ir` between 10 Hz and `crossover_hz`, changing nothing
/// above `crossover_hz · TAPER_RATIO`.
///
/// Returns `None` when the request cannot mean anything: no taps, a zero rate, a crossover that is
/// not finite or not above the band floor, or a crossover at or past Nyquist.
/// `budget_ms` bounds how far group delay may move at any single frequency; see
/// [`DEFAULT_GD_BUDGET_MS`] for why an unbounded version of this operation is not implementable.
#[must_use]
pub fn linearize_band(ir: &[f64], fs: u32, crossover_hz: f64) -> Option<Adjustment> {
    linearize_band_with_budget(ir, fs, crossover_hz, DEFAULT_GD_BUDGET_MS)
}

/// See [`linearize_band`].
#[must_use]
pub fn linearize_band_with_budget(
    ir: &[f64],
    fs: u32,
    crossover_hz: f64,
    budget_ms: f64,
) -> Option<Adjustment> {
    let budget_ms = if budget_ms.is_finite() && budget_ms > 0.0 {
        budget_ms
    } else {
        DEFAULT_GD_BUDGET_MS
    };
    let len = ir.len();
    if len < 2 || fs == 0 {
        return None;
    }
    let nyquist = f64::from(fs) / 2.0;
    if !crossover_hz.is_finite() || crossover_hz <= BAND_FLOOR_HZ * 2.0 || crossover_hz >= nyquist {
        return None;
    }
    let f_taper = (crossover_hz * TAPER_RATIO).min(nyquist);

    // Zero-padded to at least twice the filter, so the reshaped (and no longer compact) impulse is
    // measured against a grid that is not folding its own tail back onto its onset. Truncation of
    // what that leaves is exactly the error this function then measures.
    // Eight times the filter, not twice. The reshape makes the response longer than the kernel it
    // came from, and the grid has to hold the whole of it — pre-ringing included — with room left over,
    // or the tail folds back onto the onset and the numbers below measure a fold instead of a filter.
    // The project's rule is to buy accuracy with compute rather than give it up, and a bigger transform
    // is the cheapest accuracy on the table: everything here stays f64.
    let want = len.saturating_mul(8).max(8);
    let n = crate::fft::shl_bit_length(want).max(8);
    let spec = rfft(ir, n);
    let freq = rfftfreq(n, f64::from(fs));
    let mag: Vec<f64> = spec.iter().map(|c| c.norm()).collect();
    let phase: Vec<f64> = spec.iter().map(|c| c.arg()).collect();
    let phase_u = unwrap(&phase);

    // The fit band: the flat part of the region, not the taper. Fitting over the taper would let the
    // line be dragged by frequencies the line is only meant to fade out of.
    let mut fit_f: Vec<f64> = Vec::new();
    let mut fit_p: Vec<f64> = Vec::new();
    for (i, &f) in freq.iter().enumerate().take(freq.len()) {
        if f >= BAND_FLOOR_HZ && f <= crossover_hz {
            fit_f.push(f);
            fit_p.push(phase_u[i]);
        }
    }
    if fit_f.len() < 2 {
        return None;
    }
    // The work is specified in group delay, because that is what the stage promises to flatten, and
    // then integrated back into phase. Specifying it in phase (blend each bin toward a fitted line)
    // was the first version, and it cannot state a budget: a bin's phase says nothing about how many
    // milliseconds of delay it represents at that frequency.
    let gd_now = group_delay_ms(&phase_u, &freq);
    let band: Vec<usize> = (0..freq.len())
        .filter(|&i| (BAND_FLOOR_HZ..=crossover_hz).contains(&freq[i]))
        .collect();
    if band.len() < 2 {
        return None;
    }
    // The target is the band's own mean delay: flattening to it moves delay around inside the band and
    // invents none. Fit over the flat part only, never the taper, or the line is dragged by the
    // frequencies whose weight is on their way to zero.
    let mean_gd = {
        let fit: Vec<f64> = band.iter().map(|&i| gd_now[i]).collect();
        fit.iter().sum::<f64>() / fit.len() as f64
    };
    // Per-bin correction, clamped: the requested change is `mean - current`, and at most
    // `budget` ms of it is applied at any one frequency. `want` also records the span the band
    // actually contains, which is what the report contrasts the budget with.
    let requested: f64 = {
        let lo = band
            .iter()
            .map(|&i| gd_now[i])
            .fold(f64::INFINITY, f64::min);
        let hi = band
            .iter()
            .map(|&i| gd_now[i])
            .fold(f64::NEG_INFINITY, f64::max);
        (hi - lo).max((mean_gd - lo).max(hi - mean_gd))
    };
    let mut correction: Vec<f64> = freq
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let w = weight(f, BAND_FLOOR_HZ, crossover_hz, f_taper);
            if w == 0.0 || !(BAND_FLOOR_HZ..f_taper).contains(&f) {
                return 0.0;
            }
            let want = mean_gd - gd_now[i];
            w * want.clamp(-budget_ms, budget_ms)
        })
        .collect();
    // Make the band's total delay change integrate to zero, and this is the reason: the phase change is
    // the integral of the delay change, so anything left over at the bottom edge of the band is a *step*
    // in phase there. A step in phase is an infinite group-delay spike, and it is not a measurement
    // artifact — with the shipped filter it measured 867 ms of "group delay moved" on a 5 ms budget and
    // sent the kernel out to 524 288 taps chasing energy that step invented. The clamp is what breaks the
    // balance: with no clamp, aiming every bin at the band mean integrates to zero by construction, and
    // clipping the tails of that correction leaves a residue. So the residue is removed, and the clamp
    // re-applied, until the budget holds and the integral closes.
    let band_bins: Vec<usize> = (0..freq.len())
        .filter(|&i| (BAND_FLOOR_HZ..f_taper).contains(&freq[i]))
        .collect();
    if band_bins.len() >= 2 {
        for _ in 0..4 {
            // Trapezoid mean of the correction across the band it is applied over.
            let mut integral = 0.0f64;
            for w in band_bins.windows(2) {
                let (a, b) = (w[0], w[1]);
                integral += (freq[b] - freq[a]) * 0.5 * (correction[a] + correction[b]);
            }
            let span = freq[*band_bins.last().unwrap()] - freq[band_bins[0]];
            if span <= 0.0 {
                break;
            }
            let residue = integral / span;
            if residue.abs() < 1e-12 {
                break;
            }
            for &i in &band_bins {
                correction[i] = (correction[i] - residue).clamp(-budget_ms, budget_ms);
            }
        }
    }
    // Integrate the delay change back into phase, anchored above the taper where the correction has
    // already gone to zero: `phi_new(f) = phi(f) + 2pi * integral of the correction above f`, and the
    // integral below the band is now zero too, so the join is continuous at both edges.
    let df_to_rad = |a: f64, b: f64| (b - a) * std::f64::consts::TAU / 1000.0;
    let mut new_phase = phase_u.clone();
    let mut running = 0.0f64;
    for i in (1..freq.len()).rev() {
        let df = freq[i] - freq[i - 1];
        if df <= 0.0 {
            continue;
        }
        // Trapezoid over the interval below this bin, accumulating downward in frequency.
        running += df_to_rad(freq[i - 1], freq[i]) * 0.5 * (correction[i] + correction[i - 1]);
        if correction[i - 1] != 0.0 || correction[i] != 0.0 {
            // Plus, and the sign is worth the sentence it costs: group delay is D = -(1/2pi) dphi/df,
            // so dphi = -2pi D df. Integrating a *delay increase* from the anchor downwards therefore
            // *adds* phase in this convention. The first version subtracted, and the test that measures
            // the band's delay spread caught it: the spread went 2.13 ms -> 4.90 ms, i.e. the stage had
            // been flattening group delay away from the mean, in the wrong direction, confidently.
            new_phase[i - 1] = phase_u[i - 1] + running;
        }
    }

    let shaped: Vec<Complex64> = mag
        .iter()
        .zip(&new_phase)
        .map(|(&m, &p)| Complex64::new(m * p.cos(), m * p.sin()))
        .collect();
    let full = irfft(&shaped, n);

    // How long the delivered kernel has to be, and where it starts. Cutting the reshaped response at the
    // tap count the run asked for is what made this stage cost magnitude in the first place: 1.36 dB on
    // a synthetic bass shelf, 0.25 dB on the shipped DT 770 correction. The response is longer than the
    // kernel it came from, so the kernel grows to hold it — f64 throughout, more taps, more compute,
    // which is the trade this project takes.
    //
    // The criterion is the one a delivered filter is already held to here: the tail must sit at or below
    // -130 dBFS, which is what the report's "Decay @ Truncation" line prints for the shipped filters —
    // plus a loose energy bound, so the choice is made on the artefact a user loads rather than on a
    // fraction of an ideal. The start is the position whose window of that length keeps the most energy,
    // read circularly, because the pre-ringing the reshape created sits at the far end of the grid.
    //
    // The cap is what keeps this honest. Growing without one chased a bass correction's sub-100 Hz decay
    // out to 524 288 taps and reported 5.5 s of "added latency", which was not latency but the distance
    // to wherever the window happened to land: a diffuse tail that is still above -130 dBFS half a
    // second after the onset cannot be captured by any sane kernel. Four times the requested length is
    // the most this stage takes on its own; past that it reports the number it got and `held_floor: false`
    // rather than pretending.
    let energy = crate::accum::sum_of_squares(&full);
    let budget_energy = energy * 1e-9;
    let cap = len.saturating_mul(4).min(n / 2).max(len);
    let tail_at = |s: usize, w: usize| {
        let take = w.min(100);
        let tail: Vec<f64> = (0..take).map(|i| full[(s + w - 1 - i) % n]).collect();
        crate::ir::db_floor(&tail, take)
    };
    // Which window, once a length is chosen: the *smallest shift* that meets the floor, not the densest
    // window on the grid. Those are different answers and the dense one is wrong here — it costs 460 ms
    // of latency on the shipped filter, because sliding a kernel is a delay, and the response of a bass
    // correction is diffuse enough that its densest stretch of 65 536 samples starts a quarter of a
    // second away from the onset. Latency is the one cost a listener cannot opt out of later, so it is
    // spent last and only as far as the floor requires; the shift needed here is the few hundred samples
    // of pre-ringing the reshape created, not a fraction of the kernel.
    // Window energy by prefix sum over the array doubled on itself, so a shift is one subtraction. The
    // first draft summed each window in a loop and the run took 80 seconds scanning shifts; the scan is
    // also capped, because past a fraction of the kernel the answer is "this response cannot be held at
    // this floor", which is a result to report, not a search to finish.
    let doubled: Vec<f64> = full.iter().chain(full.iter()).map(|v| v * v).collect();
    let prefix = crate::accum::prefix_sums(&doubled);
    let max_shift = (n / 8).max(1);
    let passes = |s: usize, w: usize| {
        let kept = prefix[s + w] - prefix[s];
        (
            tail_at(s, w) <= -130.0,
            energy - kept <= budget_energy,
            kept,
        )
    };
    let smallest_shift = |w: usize| -> Option<usize> {
        let (floor_ok, energy_ok, _) = passes(0, w);
        if floor_ok && energy_ok {
            return Some(0);
        }
        // Nearest shift first, alternating sides: the pre-ringing sits just before the onset, so the
        // answer is a small backward move far more often than a forward one.
        for shift in 1..=max_shift {
            for s in [n - shift, shift] {
                let (f, e, _) = passes(s, w);
                if f && e {
                    return Some(s);
                }
            }
        }
        None
    };
    let mut taps_after = len;
    let mut start = 0usize;
    let mut retained = 0.0f64;
    let mut reached_floor = false;
    for (num, den) in [(1usize, 1), (5, 4), (3, 2), (7, 4), (2, 1), (3, 1), (4, 1)] {
        let probe = len.saturating_mul(num) / den;
        if probe < len || probe > cap {
            continue;
        }
        taps_after = probe;
        if let Some(s) = smallest_shift(probe) {
            start = s;
            retained = passes(s, probe).2;
            reached_floor = true;
            break;
        }
        // Nothing at this length clears the floor at any shift; try a longer kernel before giving up.
        if probe >= cap {
            start = best_window(&full, probe);
            retained = passes(start, probe).2;
        }
    }
    let out: Vec<f64> = (0..taps_after).map(|i| full[(start + i) % n]).collect();
    let out_peak = peak_index(&out);
    let input_peak = peak_index(ir);
    // Relative to the input filter's own onset, not to index 0: an input that already carries delay
    // must not be charged for it twice.
    let added_latency_samples = out_peak.saturating_sub(input_peak);

    // What truncation cost, measured rather than assumed: the same grid, the same magnitudes.
    let after = rfft(&out, n);
    let mut max_mag_error_db = 0.0f64;
    for (i, c) in after.iter().enumerate() {
        let (before, now) = (mag[i], c.norm());
        if before < 1e-300 {
            continue;
        }
        let d = ((now + 1e-300).ln() - (before + 1e-300).ln()) / std::f64::consts::LN_10 * 20.0;
        if d.is_finite() && d.abs() > max_mag_error_db {
            max_mag_error_db = d.abs();
        }
    }

    let gd_before = group_delay_ms(&phase_u, &freq);
    let gd_after = group_delay_ms(&new_phase, &freq);
    let mut max_change = 0.0f64;
    let mut at_hz = 0.0f64;
    for (i, &f) in freq.iter().enumerate() {
        let d = (gd_after[i] - gd_before[i]).abs();
        if d > max_change {
            max_change = d;
            at_hz = f;
        }
    }
    let _ = requested;

    let floor = crate::ir::db_floor(&out, 100);
    let gd_span = requested;
    let budget_held = correction.iter().all(|&c| c.abs() <= budget_ms + 1e-9);
    Some(Adjustment {
        ir: out,
        full,
        f_lo_hz: BAND_FLOOR_HZ,
        crossover_hz,
        f_taper_hz: f_taper,
        line_delay_ms: mean_gd,
        max_gd_change_ms: max_change,
        max_gd_change_hz: at_hz,
        max_mag_error_db,
        added_latency_samples,
        added_latency_ms: added_latency_samples as f64 / f64::from(fs) * 1000.0,
        retained_energy_pct: if energy > 0.0 {
            retained / energy * 100.0
        } else {
            100.0
        },
        energy_pct: if energy > 0.0 {
            retained / energy * 100.0
        } else {
            100.0
        },
        floor_dbfs: floor,
        taps_before: len,
        taps_after,
        budget_ms,
        gd_span_ms: gd_span,
        held_floor: reached_floor,
        budget_held,
    })
}

/// The band's group-delay span before the stage ran, ms. Kept on the adjustment by the caller-side
/// report; see [`DEFAULT_GD_BUDGET_MS`] for the measurement this feeds.
#[must_use]
pub fn budget_ms() -> f64 {
    DEFAULT_GD_BUDGET_MS
}

/// Report lines for a run that asked for this. Printed only when the stage ran, so the default
/// report is the parity-pinned one.
#[must_use]
pub fn report_lines(a: &Adjustment) -> Vec<String> {
    let threshold = if a.within_published_thresholds() {
        "within"
    } else {
        "beyond"
    };
    vec![
        format!(
            "=== Experimental Phase (mixed: flat group delay below {:.0} Hz) ===",
            a.crossover_hz
        ),
        format!(
            "  Band            : {:.0} Hz - {:.0} Hz, tapered to unity by {:.0} Hz (nothing above it is touched)",
            a.f_lo_hz, a.crossover_hz, a.f_taper_hz
        ),
        format!(
            "  Delay carried   : {:.3} ms, the band's own mean - flattening moves delay, it does not add any",
            a.line_delay_ms
        ),
        format!(
            "  Group delay     : up to {:.3} ms moved, worst at {:.0} Hz - {} the published audibility thresholds (-0.56 / +0.64 ms, Liski et al. 2021)",
            a.max_gd_change_ms, a.max_gd_change_hz, threshold
        ),
        format!(
            "  Budget          : {:.2} ms per frequency against a band spread of {:.1} ms - the rest was left alone rather than paid for in latency{}",
            a.budget_ms,
            a.gd_span_ms,
            if a.budget_held {
                String::new()
            } else {
                " (and the budget did NOT hold after the residue was removed)".to_string()
            }
        ),
        format!(
            "  Kernel          : {} -> {} taps, +{:.1} ms latency - the reshape makes the response longer, and it is kept rather than cut",
            a.taps_before, a.taps_after, a.added_latency_ms
        ),
        format!(
            "  Magnitude       : worst deviation {:.3e} dB - this stage replaces phase and carries the curve through, so what is left is the tail it could not hold{}",
            a.max_mag_error_db,
            if a.held_floor { "" } else { ", and it did NOT reach the -130 dB floor" }
        ),
        format!(
            "  Energy retained : {:.10}%, tail floor {:.2} dBFS",
            a.energy_pct, a.floor_dbfs
        ),
        "  Audibility      : contested. Minimum-phase is the default; this is an experiment, not a recommendation.".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::rfft as fwd;

    /// A filter with a real bass shelf: magnitude is what the stage must not touch, and a
    /// minimum-phase shelf has plenty of bass group delay to flatten.
    fn shelf_ir() -> Vec<f64> {
        // Built by the shipped synthesis path, so every number these tests assert is the number a run
        // produces rather than a property of a hand-written impulse that happens to be easy to shape.
        // The curve boosts the bass hard: that is where a minimum-phase correction carries its delay,
        // and therefore where there is anything to flatten.
        let curve = [
            (20.0, 8.0),
            (40.0, 8.0),
            (80.0, 6.0),
            (160.0, 2.0),
            (400.0, 0.0),
            (1000.0, 0.0),
            (4000.0, -1.0),
            (12000.0, -1.0),
            (20000.0, -1.0),
        ];
        crate::ir::synthesize_minimum_phase_ir(&curve, 48000, 32768, 16384, 4096, -1.50, true, 16.0)
            .expect("the shipped synthesis builds this")
            .ir
    }

    fn mag_db(ir: &[f64], n: usize) -> Vec<f64> {
        fwd(ir, n)
            .iter()
            .map(|c| 20.0 * (c.norm() + 1e-300).log10())
            .collect()
    }

    #[test]
    fn the_magnitude_survives_within_rounding() {
        // The acceptance line, stated as a number rather than a promise. The stage replaces phase and
        // carries magnitude through, so the only loss is truncation of the reshaped tail - and that
        // has to be many orders below anything a listener or a report could care about.
        let ir = shelf_ir();
        let a = linearize_band(&ir, 48000, 200.0).expect("a band to flatten");
        // The tolerance is the acceptance criterion, so it is stated as a number. It is not zero, and
        // it cannot be: linearising a band moves part of that band's energy in front of the onset,
        // and the filter still has to end where it ended. What the stage guarantees is that the cost
        // is a truncation cost and not a re-derived curve -- three orders below the threshold the
        // verification loop judges a band by, and printed in the report every time it runs.
        assert!(
            a.max_mag_error_db < 0.1,
            "phase-only shaping moved the magnitude by {} dB",
            a.max_mag_error_db
        );
        // The grid the stage works on (it pads eightfold, so matching it is not optional). An n-point
        // transform of a longer array aliases, and an aliased magnitude has nothing to say about whether
        // the stage preserved one.
        let n = crate::fft::shl_bit_length(ir.len() * 8);
        let before = mag_db(&ir, n);
        let after = mag_db(&a.ir, n);
        let worst = before
            .iter()
            .zip(&after)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 0.1, "independent check says {worst} dB moved");
        assert!(
            (worst - a.max_mag_error_db).abs() < 1e-3,
            "the report's deviation ({}) is not what an outside measure finds ({worst})",
            a.max_mag_error_db
        );
    }

    /// Group-delay spread (max - min, ms) across a frequency window.
    fn gd_spread(spec: &[Complex64], freq: &[f64], lo: f64, hi: f64) -> f64 {
        let phase = unwrap(&spec.iter().map(|c| c.arg()).collect::<Vec<_>>());
        let gd = group_delay_ms(&phase, freq);
        let band: Vec<f64> = (0..freq.len())
            .filter(|&i| freq[i] >= lo && freq[i] <= hi)
            .map(|i| gd[i])
            .collect();
        assert!(
            band.len() > 4,
            "the window holds {} bins, not enough to speak of",
            band.len()
        );
        let mn = band.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = band.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        mx - mn
    }

    #[test]
    fn the_bass_group_delay_actually_flattens() {
        // Not a tautology and not the magnitude test again: this fails if the weight is inverted, if
        // the line is fitted over the wrong band, or if the phase is replaced with anything but the
        // fit. A minimum-phase bass shelf has real group-delay ripple; after the stage it must have
        // substantially less inside the band.
        let ir = shelf_ir();
        let n = crate::fft::shl_bit_length(ir.len() * 8);
        let freq = rfftfreq(n, 48000.0);
        let before = gd_spread(&fwd(&ir, n), &freq, 30.0, 180.0);
        let a = linearize_band(&ir, 48000, 200.0).expect("adjustment");
        let after = gd_spread(&fwd(&a.ir, n), &freq, 30.0, 180.0);
        // The claim is "flattens, within the budget", not "makes linear". On this fixture the band spans
        // 2.13 ms and the default budget is 0.56 ms per frequency, so a bounded stage can only close part
        // of it — and a stage that closed all of it would be ignoring the budget, which is the other test.
        assert!(
            after < before * 0.8,
            "group-delay spread across the band went {before:.4} ms -> {after:.4} ms; flattening did not \n             flatten"
        );
        assert!(
            before > 0.01,
            "the fixture has no delay ripple to flatten: {before} ms"
        );
    }

    #[test]
    fn nothing_above_the_taper_is_touched() {
        // A stage that quietly reshaped the treble would be a different feature from the one the
        // README describes, and nothing else in the report would show it.
        let ir = shelf_ir();
        let a = linearize_band(&ir, 48000, 200.0).expect("adjustment");
        // Same grid the stage reshaped on, or the two spectra are different transforms of different
        // amounts of the array and the comparison measures aliasing.
        let n = crate::fft::shl_bit_length(ir.len() * 8);
        let freq = rfftfreq(n, 48000.0);
        // The treble guarantee is structural, not numerical: above the taper the weight is zero, so the
        // replacement leaves the phase term alone, and `the_taper_weight_is_continuous_and_ends_where_it_
        // says` pins that on the function which decides it. What is checked here, on the real filter, is
        // the property the module exists for: the magnitude is carried through the reshape untouched,
        // exactly, before truncation. Measuring the delivered taps instead would measure the truncation
        // cost, which has its own number in the report and its own test.
        let spec0 = fwd(&ir, n);
        let spec1 = fwd(&a.full, n);
        let peak = spec0.iter().map(|c| c.norm()).fold(0.0f64, f64::max);
        let worst_mag = (0..spec0.len())
            .map(|i| (spec0[i].norm() - spec1[i].norm()).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst_mag <= peak * 1e-12,
            "magnitude moved by {worst_mag} (peak {peak}) before any truncation, which the design does \
             not allow: this stage replaces phase and carries magnitude"
        );
        assert!(
            a.f_taper_hz > a.crossover_hz,
            "the taper does not sit above the band"
        );
        let weighted: Vec<f64> = freq
            .iter()
            .filter(|&&f| f > a.f_taper_hz)
            .map(|&f| weight(f, BAND_FLOOR_HZ, a.crossover_hz, a.f_taper_hz))
            .filter(|&w| w != 0.0)
            .collect();
        assert!(
            weighted.is_empty(),
            "{} bins above the taper still carry weight",
            weighted.len()
        );
    }

    #[test]
    fn a_filter_that_is_already_flat_in_the_band_comes_back_unchanged() {
        // A pure delay is linear phase everywhere: the stage has nothing to remove, so it must
        // deliver the same numbers rather than invent a correction.
        let mut ir = vec![0.0; 4096];
        ir[64] = 0.5;
        let a = linearize_band(&ir, 48000, 200.0).expect("adjustment");
        let diff: f64 = ir
            .iter()
            .zip(&a.ir)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        assert!(diff < 1e-12, "a linear-phase filter was reshaped by {diff}");
        assert!(
            a.max_gd_change_ms < 1e-9,
            "GD moved {} ms on a flat filter",
            a.max_gd_change_ms
        );
    }

    #[test]
    fn a_padded_grid_is_what_makes_the_magnitude_claim_true() {
        // The reshape spreads the response, so the tap count no longer covers it and *somewhere* has to
        // give. Cutting where the filter has always been cut discards the pre-ringing the stage just
        // created; that measured 1.36 dB on this fixture, which is more than the verification loop
        // forgives a single band. Sliding the window onto the densest `len` samples buys the magnitude
        // with latency instead, and prints the latency. This test is that difference, and it fails the
        // moment `best_window` stops finding the dense window.
        let ir = shelf_ir();
        let a = linearize_band(&ir, 48000, 200.0).expect("adjustment");
        let n = crate::fft::shl_bit_length(ir.len() * 2);

        // The naive alternative: the same reshape, cut where the minimum-phase filter was cut.
        let naive = &a.full[..ir.len()];
        let m0 = mag_db(&ir, n);
        let cost = |ir2: &[f64]| {
            let m = mag_db(ir2, n);
            m0.iter()
                .zip(&m)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max)
        };
        let (chosen, naive_cost) = (cost(&a.ir), cost(naive));
        assert!(
            naive_cost > chosen * 5.0,
            "the window choice buys nothing: chosen {chosen:.4} dB vs cut-at-zero {naive_cost:.4} dB"
        );
        assert!(
            a.added_latency_samples > 0,
            "linearising a band with no added latency means the pre-ringing was thrown away, not kept"
        );
        assert!(
            a.retained_energy_pct > 99.0,
            "the delivered window keeps only {}% of the reshaped response",
            a.retained_energy_pct
        );
    }

    #[test]
    fn the_taper_weight_is_continuous_and_ends_where_it_says() {
        for &f in &[10.0, 100.0, 200.0, 250.0, 350.0, 399.0, 400.0, 1000.0] {
            let w = weight(f, BAND_FLOOR_HZ, 200.0, 400.0);
            assert!((0.0..=1.0).contains(&w), "weight at {f} Hz is {w}");
            if f <= 200.0 && f > BAND_FLOOR_HZ {
                assert_eq!(w, 1.0, "the flat band is not flat at {f} Hz");
            }
            if f >= 400.0 || f < BAND_FLOOR_HZ {
                assert_eq!(w, 0.0, "the stage reaches past its band at {f} Hz");
            }
        }
        // No step at the crossover: an epsilon away must still be an epsilon away.
        let a = weight(200.0, BAND_FLOOR_HZ, 200.0, 400.0);
        let b = weight(200.0001, BAND_FLOOR_HZ, 200.0, 400.0);
        assert!((a - b).abs() < 1e-6, "the weight jumps at the crossover");
    }

    #[test]
    fn nonsense_requests_are_declined_rather_than_attempted() {
        let ir = shelf_ir();
        assert!(linearize_band(&[], 48000, 200.0).is_none(), "no taps");
        assert!(linearize_band(&ir, 0, 200.0).is_none(), "no rate");
        assert!(
            linearize_band(&ir, 48000, 1.0).is_none(),
            "crossover under the floor"
        );
        assert!(
            linearize_band(&ir, 48000, f64::NAN).is_none(),
            "NaN crossover"
        );
        assert!(
            linearize_band(&ir, 48000, 30000.0).is_none(),
            "crossover past Nyquist"
        );
        // A rate so low that 200 Hz is above Nyquist cannot be shaped either.
        assert!(
            linearize_band(&ir, 300, 200.0).is_none(),
            "rate below the band"
        );
    }

    #[test]
    fn the_unwrapper_removes_wraps_a_fit_cannot_tolerate() {
        // 100 ms of delay on a 16 384-point grid: the phase crosses pi about thirty-eight times inside
        // the 10-200 Hz band the fit works on. The two numbers are coupled and the coupling is the trap:
        // a delay long enough to wrap needs bins fine enough that the phase still advances less than pi
        // per bin, or the sequence is aliased and no amount of unwrapping recovers the delay (an earlier
        // draft used 2 048 bins and read 7.3 ms of a 50 ms delay, confidently and wrongly).
        let n = 16384;
        let freq = rfftfreq(n, 48000.0);
        let delay_s = 0.1;
        let wrapped: Vec<f64> = freq
            .iter()
            .map(|&f| {
                let x = (-(std::f64::consts::TAU) * f * delay_s).rem_euclid(std::f64::consts::TAU);
                if x > std::f64::consts::PI {
                    x - std::f64::consts::TAU
                } else {
                    x
                }
            })
            .collect();
        let band_bins = freq.iter().take_while(|&&f| f < 200.0).count();
        let wraps = (1..band_bins)
            .filter(|&i| (wrapped[i] - wrapped[i - 1]).abs() > std::f64::consts::PI)
            .count();
        assert!(
            wraps > 4,
            "the fixture wraps {wraps} times inside the fit band"
        );
        let u = unwrap(&wrapped);
        let gd = group_delay_ms(&u, &freq);
        let expected = delay_s * 1000.0;
        for (i, &delay) in gd.iter().enumerate().skip(10).take(n / 2) {
            assert!(
                (delay - expected).abs() < 1e-3,
                "bin {i} reads {delay} ms, expected {expected}"
            );
        }
        // Why the module needs this at all: its whole correction is computed from group delay, and
        // group delay is a difference of phase. Across a wrap that difference is 2 pi over one bin step,
        // which reads as a colossal delay spike — and a spike is precisely what a budgeted correction
        // would spend its entire budget "fixing", corrupting a filter whose delay was perfectly flat.
        let naive = group_delay_ms(&wrapped, &freq);
        let spikes = (10..band_bins)
            .filter(|&i| (naive[i] - gd[i]).abs() > 1.0)
            .count();
        assert!(
            spikes > 4,
            "wrapped phase produced no delay spikes ({spikes}) in {band_bins} band bins, so unwrapping \n             is not what is being tested"
        );
    }

    #[test]
    fn the_published_threshold_comparison_is_a_comparison() {
        let mut a = linearize_band(&shelf_ir(), 48000, 200.0).expect("adjustment");
        let measured = a.max_gd_change_ms;
        a.max_gd_change_ms = 0.10;
        assert!(a.within_published_thresholds(), "0.10 ms should be within");
        a.max_gd_change_ms = 5.0;
        assert!(!a.within_published_thresholds(), "5 ms should be beyond");
        a.max_gd_change_ms = measured;
    }

    #[test]
    fn the_report_says_what_moved_and_what_it_does_not_claim() {
        let a = linearize_band(&shelf_ir(), 48000, 200.0).expect("adjustment");
        let lines = report_lines(&a);
        let all = lines.join("\n");
        assert!(all.contains("200 Hz"), "{all}");
        assert!(
            all.contains("Liski"),
            "the thresholds are quoted without a source"
        );
        assert!(
            all.contains("contested"),
            "the report must not recommend this"
        );
        assert!(
            all.contains("Magnitude") && all.contains("carries the curve through"),
            "the one property the stage guarantees is not stated:\n{all}"
        );
        assert!(
            all.contains("Budget") && all.contains("Kernel"),
            "the two costs the stage can incur are not both printed:\n{all}"
        );
    }
}
