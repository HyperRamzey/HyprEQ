# HyprEQ as an EQ base: assessment, comparison, and DSP roadmap

Repository: <https://github.com/HyperRamzey/HyprEQ>

This document answers a question that was asked directly: *as an EQ base, how good
is this — is it an 8/10?* It then sets out where the DSP actually falls short of the
literature, and what to build next, with sources.

---

## 1. The rating, and what it is a rating of

**8/10 as a measurement-driven EQ *foundation*. It is not an 8/10 guarantee of sound
quality, and it should not be read as one.**

Those are different claims, and the difference is the whole content of this section.

### What earns the 8

| Property | Why it matters |
| --- | --- |
| The correction is a **true minimum-phase FIR**, synthesized cepstrally and shipped as a WAV convolver (`pcm_f64le` for `afir=precision=double`, `pcm_f32le` mono/stereo for PipeWire and DAW convolvers) | A magnitude-only IPEQ cannot fix the decay of a resonance; a minimum-phase filter corrects magnitude while keeping the impulse response causal and compact, and it loads into any convolution engine |
| Internals stay `f64` end to end; `f32` appears only in a 32-bit container | The report prints twelve significant digits of a causal-energy ratio; that is only meaningful if the pipeline is not losing bits on the way |
| **True-peak** limiting, not sample-peak: interpolating detector, two-stage slow/transient gain, lookahead and release accounting, per-channel and group meters | Post-EQ clipping is the most common way "correcting" a headphone makes it worse, and sample-peak meters do not see inter-sample overs |
| Every run writes an **audit CSV** (raw, target, raw error, applied EQ per control knot) and a 4-panel verification figure, and both are checked against a reference implementation | The curve can be audited after the fact, which is the difference between a tool and a preset |
| Reproducible numerical contract: NumPy-compatible FFT/PCHIP/dither semantics, TPDF dither with a fixed stream, ties-to-even rounding, and a parity harness that diffs artefacts against the reference | The port is testable rather than plausible |
| The engine is small and legible: `model → pchip → ir → convolve → limiter → plot`, with 236 library tests, 12 CLI tests and 3 that run the built binary | A base you can read is a base you can extend |

### What caps it — the honest list

1. **The fit is unregularized in a specific sense.** Smoothing is applied to the
   *measurement* (sub-bass Gaussian leak discount, log-cosine HF taper, HF bypass) and
   notch handling is a mode switch (`safe`/`original`/`zero`), but the amount of
   correction is not derived from *how badly the curve can be fit*. Deep, narrow notches
   are exactly where automatic equalization goes wrong, and the literature says the
   regularization should be estimated from the response itself (§3.1).
2. **A target is assumed to travel.** Each model records its fixture — as free text in
   `rig_info` ("Filk (Clone GRAS 43AG / KB006x Pinna + IEC 60318-4 711 Coupler, via
   AutoEq") — which both implementations *load and never use*: it reaches the model struct
   and no report line, no plot, and no check ever reads it. The target side records no
   fixture at all: there is one `harman_2018` curve with no attribution, and a target curve
   is a property of a fixture — that is the central finding of the 2025 Olive/Clark work
   (§3.3). Equalizing a B&K 5128 measurement to a GRAS-derived Harman target is a real and
   common error, and this tool would do it silently, with the warning sitting unread in a
   JSON string.
3. **The loop is open.** The tool predicts what the EQ will do and never checks it. The
   prediction is good — headphones turn out to be obedient to EQ (§3.5) — so a
   remeasurement, when someone has one, is cheap and decisive evidence, and there is no
   place to put it.
4. **Channels are fit independently.** Nothing compares L to R, so a correction can
   introduce a spectral skew between ears, which is a lateralization and imaging problem,
   not a frequency-response problem.
5. **Magnitude only.** The filter is deliberately minimum-phase, which is a defensible
   default, but the phase column in the verification figure is descriptive: there is no
   option to trade a little minimum-phase purity for a flatter group delay where the
   literature says that is audible (§3.4).
6. **No distortion or excursion model.** A +9 dB bass shelf is reported as a +9 dB bass
   shelf. Whether the driver can deliver it, and what it costs, is out of scope — even
   though the tool already computes and reports true peak before and after.
7. **Single-file, single-measurement.** No averaging across samples of the same model,
   no tolerance band, no "this peak is probably unit variance" judgement.

### Prior context, recorded so it is not mistaken for more than it is

The operator's own listening comparison, on three of their own measurements, found
HyprEQ's correction preferable to AutoEq's on **two of the three**. That is a useful,
real signal — and it is three comparisons, on one person's ears, on three headphones,
with one reference chain. It supports "this base is competitive with the strongest
open-source baseline." It does not support "better than AutoEq", and nothing in the
codebase is claimed on that basis. AutoEq is treated here as the baseline to beat, not
as something to replace wholesale: its measurement coverage is the asset, and this
project's engine shape is the asset. They are different tools, and §2 says where.

---

## 2. Where it sits next to AutoEq, REW, CamillaDSP

| | HyprEQ | AutoEq | REW | CamillaDSP / CamillaFIR |
| --- | --- | --- | --- | --- |
| What it is | Convolver generator: measurement + target → minimum-phase FIR WAV, plus the report and figure | Measurement database + parametric-EQ optimizer + web app and PyPI library | Measurement application with filter design and export | A real-time DSP engine (CamillaDSP) and a REW-export → FIR generator (CamillaFIR) |
| Input | Your own CSV/JSON measurement, plus a target curve | A curated database of oratory1990 / crinacle / Innerfidelity / Rtings measurements, plus targets | A sweep measured on your own rig or an imported response | Configuration; CamillaFIR consumes REW exports |
| Output | Minimum-phase FIR as 64/32-bit WAV (offline and 16 384-tap real-time forms), audit CSV, 4-panel figure, and the input file convolved and limited | Settings for other apps — biquads/PEQ bands, graphic EQ, convolution — mostly generated by an optimizer under Fc/Q/gain limits | Filter sets, and a filter impulse response exported as WAV | Runs the filters; does not decide what they should be |
| Strong at | The part after the decision: filter realization, headroom, true-peak limiting, provenance, verifiable numerics | Coverage (thousands of headphones) and a tuned, constrained PEQ optimizer; it deliberately treats the region above ~10 kHz as an average instead of chasing it | Measuring — including non-minimum-phase behaviour, which its predicted-EQ view folds in, and exporting the result | Playback: real-time convolution, pipelines, cross-platform |
| Weak at | Fixing a bad *fit* decision (§3.1, since answered by `--regularize`), fixture awareness (§3.3), stereo coherence (measured, and cancellable per ear, since `--stereo`), phase choice | Anything requiring your own measurement pipeline or a convolution-specific artifact of this shape | Automation and reproducibility; it is an interactive app | Deciding anything |

The consequence: **HyprEQ's differentiation is the last mile, not the first.** Anyone can
get a flat-ish target and a smoothed error curve. What this project does well is turn a
curve into a *filter that survives contact with a real signal* — headroom, true peak,
dither, truncation, and an auditable record. The roadmap below is therefore about
improving the *decisions* fed into that engine without damaging what the engine already
does well.

---

## 3. What the literature says

### 3.1 Regularization should be estimated from the response, not fixed

> Gómez-Bolaños, Mäkivirta & Pulkki, *Automatic Regularization Parameter for Headphone
> Transfer Function Inversion*, **JAES 64(10), 752–761 (2016)**, DOI
> [`10.17743/jaes.2016.0030`](https://doi.org/10.17743/jaes.2016.0030).

The paper estimates the regularization amount by comparing the measured response with its
**half-octave-smoothed** version, so the parameter depends only on the headphone being
equalized, and shows in a perceptual test that this beats both a fixed-λ regularized
inverse and complex smoothing with a half-octave window. It is precisely the "the notches
are where it goes wrong" problem, and it gives an implementable recipe rather than a
principle.

**Implication here:** HyprEQ currently applies one smoothing philosophy to all curves.
A per-model λ derived from the raw-vs-smoothed residual — and a cap on notch depth that
follows from it — is a direct, testable implementation of this result, and it is the
single highest-value change in this document.

### 3.2 Perceptual robustness beats inversion accuracy

> Masiero & Fels, *Perceptually Robust Headphone Equalization for Binaural
> Reproduction*, **130th AES Convention, London (2010; Curran 2011)**, RWTH-CONV-198599.

Inverting a headphone-to-eardrum response faithfully is not the same as making it sound
right; over-fitting narrow nulls is a known failure mode, especially where the null is a
measurement artefact of the coupling rather than an acoustic property. The smoothing-based
routines in this line of work trade accuracy for robustness on purpose.

**Implication here:** `apply_notch_mode` is the right place to hang this — but the mode
should be chosen from evidence about the curve, not set once on the command line.

### 3.3 A target curve belongs to a fixture

> Olive & Clark, *A Preferred Headphone Target Curve Defined for the B&K 5128 and GRAS
> 45CA-10 Test Fixtures*, **AES 2025 (Newport Beach)**, eLibrary
> [`22928`](https://www.aes.org/publications/elibrary-page/?id=22928).

The abstract states the problem plainly: the Harman around-/on-ear target was defined on
a GRAS 45CA fixture with an RA045 ("711") ear simulator and a custom pinna, and is
"often misused with other fixtures, leading to measurement and interpretation errors due
to differing acoustic transfer impedances". The authors measured headphones on all three
fixtures and derived per-fixture calibration and target equivalents, then ran listening
tests on how similar the results sounded. The reported pattern (in the discussion of the
paper accompanying its presentation) is that a target fitted **per headphone** transfers
well, while a generalized, averaged translation transfers noticeably worse — and worse
again for the B&K 5128.

**Implication here:** the database has no place to say "this curve was measured on X".
Adding a fixture field, a per-fixture target table, and a loud warning when a curve's
fixture and the target's fixture disagree costs almost nothing and prevents the most
likely way this tool produces a bad filter.

### 3.4 Phase is not free, and minimum-phase is a choice

> Li et al., *Evaluation of headphone phase equalization on sound reproduction*,
> **Applied Acoustics (2019)**, [`S0003682X18311587`](https://www.sciencedirect.com/science/article/pii/S0003682X18311587)
> — objective and subjective evaluation of magnitude and phase equalization of the HETF;
> phase linearity was found to matter for the decay speed of the corresponding transient.
>
> Further: Juho Liski, Aki Mäkivirta & Vesa Välimäki, *Audibility of Group-Delay
> Equalization*, **IEEE/ACM Transactions on Audio, Speech and Language Processing, vol. 29,
> pp. 2189–2201 (2021)**, art. [`9450008`](https://doi.org/10.1109/TASLP.2021.3094413)
> — a peer-reviewed journal article (not a thesis), which is where the thresholds quoted in
> PR 5 come from: about −0.56 ms of negative and +0.64 ms of positive group delay at threshold
> for a pink-noise impulse.

Minimum-phase is the sane default for headphone correction: it puts no pre-ringing in
front of a transient and matches how loudspeaker crossovers already behave. But it is a
choice, and the objective evidence that transient decay correlates with phase linearity
means the choice deserves a switch rather than an assumption — with the caveat that this
is the most contested area in the list and should ship behind a flag.

### 3.5 The prediction is worth checking, because EQ is obedient

> Rtings, *EQ With Confidence: 200 Post-EQ Remeasurements Validate That Headphones Follow
> a Linear Response to Equalization*
> ([rtings.com/headphones/learn/research/eq-remeasurements](https://www.rtings.com/headphones/learn/research/eq-remeasurements)).

Across 200 remeasurements, the measured post-EQ curve tracked the calculated one closely;
the article's conclusion is that headphones are far more linear and obedient to EQ than
the hobby consensus assumes.

**Implication here:** this cuts *for* a verification loop and *against* the excuse that
remeasurement is hopeless. If the predicted and remeasured post-EQ curves disagree, the
interesting conclusion is not "EQ is unreliable" — it is that something specific is wrong
here (fixture mismatch, a bad measurement, an EQ applied at the wrong stage, or a bug in
this tool). That is a much better use of a remeasurement than the tool currently has.

---

### 3.6 Regularize toward unity, not toward zero

The textbook form of this problem is: design `g` so that `h * g ≈ d`, where `h` is the
measured response and `d` a pure delay (flat magnitude, linear phase). Tikhonov gives
`minimise ‖Hg − d‖² + λ‖g‖²`, solved by `g = (HᴴH + λI)⁻¹ Hᴴd`.

The trap is the second term. What it penalises is `‖g‖²`, the filter's own energy. In a
band where the inversion is hopeless — a deep coupling null, or simply far past anywhere
anyone is listening — the least-squares answer is not "leave this band alone", it is
"stop trying", and those two differ completely in dB: *leave it alone* is 0 dB, *stop
trying* is −∞ dB. A penalty on the filter's magnitude therefore cuts hardest exactly where
a bypass was wanted, and it does it more visibly the larger λ gets. Any regularized path
added here has to be looked at for this: an HF region that droops as λ rises is showing
the penalty, not the headphone.

What the published routines actually compute is a regularized least-squares inversion whose
penalty is on the **filter's own magnitude**, referred to a regularization curve `ε(f)`:

```text
G(f) = H*(f)·d(f) / (H*(f)H(f) + β·|ε(f)|²)          β ≥ 0

g  = (HᴴH + diag(β))⁻¹ (Hᴴ d + diag(β)·a)            the general weighted form
```

The first line is what AKtools and pyfar ship (§3.6's prior-art note), and it cites Norcross
2006; the second is the same weighted least-squares problem with the penalty taken about a
**reference `a`** instead of about zero. The first line has a consequence this project will not
ship: as `β·|ε|² → ∞` the filter goes to zero, so a band the measurement discredits is not
left alone, it is *silenced*. Putting the reference at unity — `a = 1`, no correction — is what
makes `β → ∞` mean "give up on this band" rather than "mute this band", and **that substitution
is HyprEQ's, not the literature's**: no source reviewed here penalises deviation from an
all-pass reference. Everything else is straight from the sources: `β` is derived from the
response itself (§3.1's residual between the raw curve and its octave-smoothed version is
exactly such a source), so a smooth band gets a small `β` (correction proceeds) and a wild band
gets a large one (correction falls back to the reference). Two properties follow, and both are
testable rather than aspirational:

* **the weight is referred to the unity level, not to `max|H|`.** A `λ` scaled by the peak
  of the response means one thing on a set with a 6 dB null and another thing on a set with
  a 2 dB null; scaled by the unity reference it means the same thing on both. This is
  `λ_unity` in PR 1.
* **`β(f) → ∞` gives `|G(f)| → 1`, not `0`.** One assertion, and it is what stops the naive
  form from coming back in a later refactor.

Prior art, read at the source rather than from a summary:

* **AKtools** (TU Berlin Audio Techniques, MATLAB): `AKregulatedInversion.m` calls
  `AKffdInverseFilter.m`, which computes `G = conj(C)·A / (|C|² + β|ε|²)` and names
  *Norcross 2006* in its own header. Licence is the **EUPL 1.1 or later** (TU Berlin's page and
  every source header), not GPL.
* **3GPP TR 26.818 §8.3.8** used that same `AKregulatedInversion` to build **16 384-sample**
  compensation FIRs by "regularized least mean squares inversion", with **1/6-octave-smoothed
  inverted copies of the measured responses as the regularization curves and β = 0.2** — an
  industry document choosing the smoothed-inverse formulation over the raw one, which is §3.1's
  argument made from a different direction. (The document then smooths and time-windows the
  filters to 256 samples; that later step is not the FIR length.)
* **pyfar** (MIT) exports `RegularizedSpectrumInversion` and the older
  `regularized_spectrum_inversion`, publishing `S⁻¹(f) = S*(f)·D(f) / (S*(f)S(f) + β|ε(f)|²)`;
  the function carries a `PyfarDeprecationWarning` saying it goes away **in pyfar 0.10.0** in
  favour of the class, and cites Norcross 2006 (the function additionally cites Kirkeby &
  Nelson).

None of them is a dependency. AKtools is EUPL and MATLAB, so it is maths to re-derive rather
than code to link into a tree that ships under MIT; pyfar is a numerics package this crate has
no reason to pull in for one equation. What we took from each is the equation and the citation,
which is also the rule in §5.

### 3.7 Robust statistics, and a smoothing loop that costs `O(N)`

Working with repeated measurements — several seats of one head, several placements of one
fixture — needs two things the arithmetic mean is bad at.

* **Location and spread that survive one bad seat.** Use the median, and `1.4826 × MAD` as
  the scale estimate; the constant is what makes the median absolute deviation a consistent
  estimator of σ for a Gaussian sample, so it stays comparable with the σ-based thresholds
  this project already prints. One headband clamped hard moves the mean by several dB and
  barely moves the median.
* **A combined curve that keeps real resonances.** Averaging repeated seats in the dB domain
  *smears* a resonance that has shifted a few hertz between placements: the mean shows a
  feature shallower and wider than any single measurement had. The perceptual line of work
  (§3.2) reaches for an **upper envelope** instead — per frequency bin, the maximum across
  the repeats. It keeps every seat's resonance at full depth, and it is the conservative
  choice for anything that will gate a boost, because it never under-reads a peak.

Both of the above feed on a fractional-octave smoother, and that smoother should cost
`O(N)`. On a 1/12-octave grid from 20 Hz to 20 kHz the window is roughly 120 samples wide,
so the obvious "average the window at every bin" form is `O(N·W)` and needs an edge
correction that nobody implements identically twice. The prefix-sum form — cumulative sums
of the dB array and of the count array, then a difference per bin, divided by the **actual**
number of samples inside the window rather than by `W` — is two walks and a division, and it
is exact at the boundaries instead of pulling edge bins toward zero. Keep the accumulation
order fixed and compensated (`accum.rs`'s Neumaier sum) and the difference from the naive
form is below the last bit rather than an unexplained drift.

**Implication here:** PR 1 needs the raw-vs-smoothed residual, PR 2 needs the robust
location/spread and the envelope, and both want the cheap smoother — so the smoother and the
two statistics land as one small module with its own tests *before* either PR depends on
them.

## 4. Roadmap

Every PR below keeps one invariant: **with no new flags, output is byte-identical to
today's**, and `scripts/parity.py` stays green. New behaviour is additive and opt-in, so
the parity oracle remains a live check on the existing engine while each change is
proved by its own tests.

### PR 1 — Fit-aware regularization — **shipped** as `--regularize`

**Problem:** one smoothing/regularization policy for every curve (§3.1, §3.2).

**Method:** estimate λ per model from the raw curve's own deviation from its
half-octave-smoothed version (Gómez-Bolaños 2016), referred to the **unity** level rather
than to the response's peak (§3.6's `λ_unity`), and use it to (a) cap notch depth in
`apply_notch_mode`, and (b) weight the control-point derivation in
`model::normalize_model_entry` where the raw curve is being reduced to knots. Where the
correction is computed as an inversion rather than a knot edit, the diagonal load is
`diag(β(f))`, not `λI`.

**The test that keeps it honest:** in a band whose `β` is driven large, the resulting
magnitude goes to **0 dB**, not toward silence (§3.6). That single assertion is what
distinguishes a regularizer from a shrinker.

**Shipped as:** `src/fit.rs` (pure `f64`) plus `--regularize off|auto|<λ>`, wired at both places
a correction curve is formed — `model::derive_ctrl_pts_from_raw_regularized` for a model whose
knots come from inverting a target, and `Model::regularize`'s knot path for a model that ships
hand-tuned knots, which every model in this database does.

`ir::apply_notch_mode` **still applies, after** the regularizer. That is the opposite of an early
note in this document, which claimed the regularizer "supersedes" the cap; the code never said so
and the claim was wrong on its merits. The two answer different questions: `β(f)` asks how well
*this measurement* supports a band, and `safe_cap_db` says what this *model's metadata* allows in
the band where a coupler resonance lives. A model can be measured beautifully and still not be
allowed a +12 dB boost inside its notch range. So the pipeline is regularize, then cap, in that
order — the ceiling is the last word — and
`the_notch_cap_still_binds_on_a_regularized_curve` pins it: with a knee that trusts everything the
cap is what stops a +12 dB knot; with a knee that shrinks it to about +6 dB the curve still comes
back *at* the ceiling rather than below it, which is what cap-first would have produced.

Three properties of the shipped form are worth stating because they are not the obvious
implementation:

* **The deviation smoother is an integral, not a count.** Each knot is weighted by its own slice
  of the log-frequency axis (the trapezoid rule on a log axis: halfway to its neighbours, half a
  spacing at the two ends). An unweighted mean of "the samples that landed in the window" is that
  integral only on a log-uniform grid, and a hand-placed CSV axis is not: unevenly spaced knots
  then carry weight by how many happen to sit nearby, worth 2.2 dB of phantom regularization on
  the axis in `uneven_knot_spacing_is_weighted_by_the_axis_it_covers`.
* **`|S|²` is taken from the curve referred to its own level** (`fit::reference_db`, the median).
  `β` is compared against `|S|²`, so feeding in a curve saved in dB SPL divides the load by
  `10^9.4` — the *same measurement*, written in different units, was a different policy. The
  median, not the value at 1 kHz, because a reference read at one frequency is only a reference if
  nothing is happening there, and a 12 dB coupler null at 1 kHz is exactly the thing these curves
  contain.
* **The fixture shaping is applied before the load, not after.** Below 65 Hz the derivation
  discounts sub-bass leak; applied to an already-bypassed band, that discount subtracts itself
  from zero and delivers a fabricated cut where the measurement said to change nothing. A fully
  bypassed curve is 0 dB everywhere, `a_fully_bypassed_derivation_is_a_flat_wire_below_65_hz_too`
  says, including under 65 Hz.

The shape differs from the sketch in one place worth naming. `regularization_lambda(g)` and
`notch_cap(lambda)` became a *frequency-dependent* `β(f)` plus a knee, because the §3.6 argument
is that the load has to be per-band — a scalar λ plus a depth cap cannot leave a broad shelf
untouched while bypassing a notch an octave away, which is the whole behaviour.

**Acceptance, and how it was met:**
- `infinite_load_bypasses_and_zero_load_is_todays_curve` is the §3.6 test: `β → ∞` gives 0 dB,
  `β = 0` gives `target − raw` to 1e-12, and the same band under the `‖G‖²` form is asserted to
  collapse to silence, so the two forms cannot be interchanged quietly.
- `the_smoother_is_the_log_domain_integral_of_the_window` compares every bin against the window's
  trapezoid integral computed independently and slowly from the definition, and checks that a flat
  curve stays flat at the two truncated ends; `uneven_knot_spacing_is_weighted_by_the_axis_it_covers`
  pins the 8.21 dB that an axis with uneven knots actually says where counting samples says 6.00;
  `one_bad_sample_reaches_the_windows_that_hold_it_and_no_others` and
  `a_window_far_from_an_outlier_still_averages_what_it_covers` cover propagation and the
  compensated prefix; `a_window_narrower_than_a_knot_spacing_has_nothing_to_average` states the
  limit that weighting cannot fix.
- `the_fit_measures_shape_not_level` (invariant to a 12 dB offset; a broad shelf does not read as
  a fit failure), `the_auto_knee_ignores_a_sparse_notch_and_follows_broad_texture` (a sparse 12 dB
  notch leaves the knee at the floor; ±9 dB broad texture lifts it to the measured 7.9 dB),
  `texture_is_left_alone_and_a_notch_is_walked_back` (measured knee shape),
  `a_narrow_notch_stops_being_corrected_and_its_neighbourhood_does_not` and
  `a_hand_tuned_curve_keeps_its_shelf_and_loses_its_notch_chase` end to end.
- `a_curve_saved_in_absolute_units_carries_the_same_policy` and
  `hand_tuned_knots_are_shrunk_the_same_whatever_units_the_measurement_came_in` re-save the same
  measurement +94, −40 and +0.5 dB off and require the regularized curve to be the same curve;
  `a_fully_bypassed_derivation_is_a_flat_wire_below_65_hz_too` and
  `an_infinite_knee_reproduces_the_plain_derivation` pin the two ends of the load;
  `the_shrink_lands_between_the_curve_and_bypass_and_never_past_either` sweeps the form and
  asserts it can only ever soften a correction, pinned to the DT 770's worst knot.
- `regularize_off_hands_back_the_models_own_curve` asserts `off` returns `model.ctrl_pts`
  untouched and prints nothing, and that a knee far above every deviation is an exact no-op — and
  that the no-op is *reported* as nothing moving rather than as a largest change of `+0.00 dB` at
  a `0 Hz` knot that does not exist. Parity stays **98/98** with the flag present in the parser,
  and `tests/cli_report.rs` runs the real binary to hold the report's section order.
- Sabotage runs, each killing tests: swapping the all-pass reference for the `‖G‖²` penalty;
  dividing the smoother's window by its nominal width instead of integrating it; counting samples
  where the form wants log-domain widths; dropping the compensation from the prefix sums; letting
  a non-finite value poison every later window; leaving a `NaN` frequency on the axis to stall the
  cursors; putting amplitude where the form wants power, or dropping the `|S|²` weight; skipping
  the reference level on either regularizing path; shaping after the shrink; printing the block
  before the model banner; accepting `--notch-mode SAFE` the way the library tolerates it; and
  reporting a worst knot when nothing moved.
- End to end on a shipped model, measured after the level reference and the integral smoother
  landed: `--regularize auto` on `dt770_80` estimates a **4.27 dB** knee, moves **33 of 128**
  knots, the largest **−4.16 dB @ 3703 Hz** — the measurement there sits 4.87 dB *below* unity and
  the shipped curve asks +4.26 dB to fill it, which is the notch-chase the flag exists to refuse.
  The largest walk-back of a *cut* on any shipped model is 0.78 dB (JBL 770NC at 8481 Hz), and
  FT1 Pro's is 0.40 dB. `--regularize off` writes the same bytes as before the flag existed.

**The symmetry decision, stated.** The shrink is symmetric: a band the load discredits is moved
toward 0 dB whether it is a boost or a cut. Nothing in the reviewed literature asks for anything
else — Norcross, Kirkeby, pyfar's `RegularizedSpectrumInversion` and AKtools' `AKffdInverseFilter`
are all symmetric about their reference, and the JAES paper estimates a single amount per response
— so an asymmetric penalty would be a HyprEQ product decision, not a citation, and may not be
described as one if it is ever added. The argument for adding it is real (the damage this flag
manages — headroom, noise floor, true peak — is done by boosts, and a cut cannot do it), and the
argument against is that a sharp cut is the same chase of measurement texture seen from the other
side. The measurements decide it stays symmetric for now: at `auto`, boosts are walked back by up
to 4.16 dB and cuts by at most 0.78 dB, so the asymmetry the literature does not require is
already the shape the form produces.

**What did not ship: the figure overlay.** The acceptance list asked for a panel drawing auto and
off together. That item was substituted, not implemented, and the reason is scope on this turn,
not a blocker: `scripts/parity.py` never passes `--regularize` (checked: no occurrence), so with
the flag at its default the figure's bytes are untouched and an overlay could have gone in with
parity green. What it actually takes is a second trace on the gain panel, an entry in the
hand-drawn legend (`plot.rs` draws those itself, to matplotlib's geometry), a test that renders
with the flag on and counts ink, and one that the flag-off figure is unchanged.

What shipped instead is the same quantity as text: `Knots moved` and `Extreme shifted`, plus
running both into separate directories and diffing the reports. Text gives the size of the change
at the worst knot; the overlay would show its *shape* — how wide the backed-off region is and how
fast it recovers, which is exactly the judgment "is this the fixture or the headphone" needs. If
the printed numbers say the flag is worth keeping, the overlay is the right next change, and it
belongs with the decision to make `auto` the default rather than with this PR.

### PR 2 — Fixture-aware targets — **shipped** as `fixture` + `--force`

**The premise held, and the data was worse than the schema.** §3.3 quoted Olive & Clark (AES 2025):
the Harman around-ear target was defined on a GRAS 45CA fixture with an RA045 ear and is "often
misused with other fixtures, leading to measurement and interpretation errors due to differing
acoustic transfer impedances". Checking what the shipped database actually says about its own rigs:

| model | `rig_info` says | fixture | equalised to |
|---|---|---|---|
| `jbl770nc` | Filk, *clone* GRAS 43AG, KB006x pinna, IEC 60318-4 (711) | `gras_43ag_clone` | `harman_2018`, defined on GRAS 45CA |
| `dt770_80` | Crinacle, *genuine* GRAS 43AG-7, RA0402 711, KB500X pinna | `gras_43ag` | the same |
| `ft1pro` | oratory1990 / Kuulokenurkka, GRAS 43AG, KB50xx pinna, IEC 711 | `gras_43ag` | the same |

So every shipped model is a 43AG measurement against a 45CA-defined target — the exact misuse the
paper names, in the tool that was supposed to prevent it. `rig_info` had described the rigs accurately
all along; nothing could *read* it.

**What shipped.** A fixture key on models, a `target_fixtures` table beside the targets, a comparison,
and a refusal. Six fixtures are nameable (`gras_43ag`, `gras_43ag_clone`, `gras_45ca`, `gras_kb5000`,
`bk5128`, `iec60318_4`); a declared fixture must also appear in the model's `name`, and a name naming a
second fixture is refused — two places to write one fact is where drift lives, and the check is
range-based so `GRAS 43AG` inside `Clone GRAS 43AG` reads as one mention rather than two. An
unrecognised fixture is a database error at load (exit 1, choices listed) rather than a quiet "no
fixture", because "no fixture" switches the comparison off and a typo must not buy that. A real
conflict exits 2, names both fixtures and both keys, and offers three remedies; it fires **before any
file is written**, including in an `-H all` run, which is why the guard walks the models the run will
actually use (`used_models`) — including the audio path's rule that `-H all` with `--input` means the
*first* model, not all of them.

**Three acceptance lines moved, and each move is a fact rather than a convenience.**

* *"Print the fixture in the report header and the figure's title block … `--list-models` shows
  fixtures."* Shipped — **through the data.** The banner, the figure title, `--list-models` and the
  audio-mode header all print the model's `name`, and the reference program prints the same field, so
  the fixture label went into `name` and appears in all four places on both programs:
  `Beyerdynamic DT 770 Pro (80 Ω) [GRAS 43AG]`. A line added in *code* would have to be added to the
  oracle too, and the oracle is frozen; the parity harness confirms it — 98/98 scenarios, byte-identical
  WAVs, with the three longer names in place. What the roadmap wanted the new line to *achieve* — nobody
  reads a curve without seeing the rig it came from — is achieved.
* *"Ship the equivalents that have a defensible published curve (… GRAS KB5000 and B&K 5128
  translations)."* **Not shipped, and not shippable here.** Those curves are the substance of a paywalled
  AES paper; the numbers are not in anything readable, and inventing a target curve and citing a paper
  for it is the one thing that would make this tool quietly wrong. What ships instead is the guard that
  makes the substitution impossible to do by accident, and a fixture vocabulary ready for the day the
  numbers are legitimately available.
* *"The reference curve is the upper envelope across repeats and the disagreement between seats is
  reported as 1.4826 × MAD" (§3.7).* **Deferred.** No model in this database has repeated seats — each
  carries one curve, and the two models with more than one column carry two *ears*, which PR 4 already
  handles and which are not seats. An envelope/MAD estimator with no repeats to aggregate would be dead
  code that its own tests could not distinguish from the single-curve path, which is the standard this
  project has applied to every other unused helper.

**The shipped conflict is recorded, not waved through.** The refusal exists for the person adding a
model and picking a target off a list. It is the wrong instrument for a pairing the database *inherited*
and cannot fix, so a model may carry `fixture_conflict_ok` with the reason; empty is refused, since an
acknowledgement that records no reason acknowledges nothing. All three shipped models carry it, with
the reason in their words: no published 43AG equivalent of the Harman target exists to switch to. The
behaviour is unchanged from the reference program — the filters are byte-identical — but the pairing is
now a decision on file, in the database, where someone can disagree with it and edit it.

**Adjacency is claimed once and only once.** Genuine 43AG ↔ 43AG clone is allowed: same fixture design,
same ear, differing in pinna and build, and the literature's finding is about transfer impedance between
fixture *designs*. Everything else, including 43AG against 45CA (both "a GRAS fixture with a 711 ear",
and the single most likely pairing to get shipped by accident), is refused rather than judged — no
published source licences judging it.

### PR 3 — Acoustic verification loop — **shipped** as `--verify-with`

**Problem:** the loop is open (§3.5). The program goes from a capture to a filter and stops. Nothing
ever asks what the headphone did after the filter went in.

**The specification asked for two things that had to be separated.** "Align it to the predicted
response" and "report the residual" describe one error, and the obvious single number conflates two
that a user acts on differently:

* **residual** = measured − predicted: the rig, the fit, the unit. *"your filter is not doing what it
  was told."*
* **design shortfall** = target − predicted: the filter never asked for what the target wanted,
  because regularization, the notch ceiling, or the refusal above `hf_bypass_hz` decided so.
  *"even a perfect rig cannot hit the target here."*

A verifier that reports only the residual calls the second a pass — which is precisely the case where
the user should be told the design gave up. Both are in the report, and the shortfall is scored on
band means, not worst points, because it describes a curve rather than a measurement.

**Predicted is `raw + |H(f)|`, not the control knots.** The written FIR carries the half-Kaiser
apodizer and its own truncation ripple; predicting from the design books the filter's deviation as a
rig error. The run's preamp is subtracted back out — it is a headroom constant baked into the taps,
and letting it through moves every band's shortfall by whatever `--headroom-db` says. A test moves
that knob by 8 dB and requires the shortfall line to be byte-identical.

**The acceptance criterion moved, and the moved line is still met.** The original said "a synthetic
test where the post-EQ curve is the predicted curve convolved with a known residual". That cannot be
computed in a test without an FFT, so instead the integration tests use a model whose correction is
provably zero — then `predicted = raw` exactly, and a capture equal to the raw curve must pass while
one carrying **6 dB at 1 kHz** must be blamed on the 1 kHz band. It is named, and the reported number
is within 0.5 dB of the 6.00 dB that was injected:

```text
  Worst band      : +5.99 dB @ 1000 Hz (band 1 kHz)
  Verdict         : FAIL - 3 of 29 band(s) outside +/-3.00 dB, 2 of them noise-limited
```

The rest of the acceptance list is met as written: an unrelated curve fails loudly (a raw curve
against a corrected prediction: 18 of 29 bands); "the measurement is quiet" arrives as `uncovered`
bands and a `NOTHING TO CHECK` verdict that is not a pass, while "the measurement is noisy" arrives
as a per-band scatter flag; the two never collapse into one another. And because pass/fail uses the
**worst** point in a band, the injected 6 dB comes back as 6 dB instead of a third-octave average of
it — the mutation that swapped in the band mean was killed by a fixture built to discriminate it.

**No prior art defines the tolerance, so the thresholds are defaults, not compliance.** The searches
found measurement methods (IEC 60268-7 for headphone measurement, IEC 61260-1 for the band grid,
ITU-T P.34 for a transmission-quality mask) and one 2022 study of *fit to a target* with an
intelligibility model, but no published acceptance region for post-EQ headphone response — consistent
with the one qualitative source available (Fleischmann, Silzle & Plogsties, *Headphone Equalization*:
"no consensus can be found in literature" on an ideal transfer function, and a listening study finding
no significant preference between two automatic EQ methods). So `DEFAULT_PASS_DB = 3.0` is wired as a
default, chosen to sit above a normal repeat-measurement difference and well under the 10–12 dB the
shipped database corrects, and **the report always prints the threshold it used**. The one constant
that *is* derived is the quiet gate: additive noise 20 dB below a signal biases a level reading by up
to **0.83 dB**, which is under a third of the threshold; by 6 dB of SNR the bias is 0.97 dB and the
noise *is* the failure.

**Where it refuses, and what it will not claim.** `--verify-with` with `--bypass-eq` (no filter) and
with `--stereo split` (two filters, one capture) exit 2 before a file is written, as does a
non-positive threshold; an unreadable capture exits 1 before the convolvers, so a mistyped path
cannot leave half a model behind. It is report-only — the same input with and without the flag writes
byte-identical audio and convolvers, tested — and the table is its own `<prefix>_verification.csv`,
because the audit CSV is byte-compared against the reference program and cannot gain a column.
A `FAIL` exits 0: the run did what it was asked, and the verdict is a finding about a measurement.

**Two claims were dropped as unreachable rather than fudged.** Alignment in *time* is meaningless for
a frequency-domain capture, and a post-EQ measurement's phase is mostly the filter's; and no residual
can separate "this unit deviates" from "this rig deviates" — the report says so in the line the user
reads.

**Not done on purpose:** no fifth figure panel. The residual is 31 numbers with four columns each and
the table is the right shape for them; a panel would be an ink test and nothing else. If one is ever
wanted, `--stereo`'s trace is the pattern to follow: opt-in, and the default figure must stay
comparable with the reference.

### PR 4 — Stereo-aware correction — **shipped** as `--stereo`

**The premise did not survive checking.** This section said "L and R are fit independently
today". They are not. `convolve_planar_fft` takes *one* impulse response and applies it to every
channel — which is also what a PipeWire `afir` loaded with a mono IR is — so there was never a
per-ear fit to make shared. What is true is the other half: two of the three shipped models measured
both ears, the DSP has always fit the average column, and the difference between the ears was
something only the figure looked at. Measured, and it is not small:

| model | level difference, left vs right | shape difference | where |
|---|---|---|---|
| DT 770 Pro (80 Ω) | **+1.27 dB** (the centre leans) | **−7.72 dB** | 15 110 Hz, band 16 kHz |
| FiiO FT1 Pro | +0.05 dB | **−4.10 dB** | 3 839 Hz, band 4 kHz |
| JBL Live 770NC | measured as one curve | — | nothing to leave standing |

**The method had to change too.** "Fit a shared correction where the two ears disagree only in a
narrow, uncorrelated way" first has to measure narrowness, and these captures are spaced **0.079 of
an octave** apart: a sixth-octave window covers **2.1 capture points**, so a feature one sample wide
is not distinguishable from a shelf a tenth of an octave wide. That is the same category of error
PR 6 refuses in the amplitude domain — attributing a third-octave band to fewer than one FFT bin —
and the answer is the same: state the resolution rather than invent it, and let the amount cancelled
be decided band by band by PR 1's knee, which is already the machinery for "how much of this feature
does the measurement support" and is already tested.

**Shipped as:** `src/stereo.rs` (pure `f64`) plus `--stereo off|report|split` and
`--stereo-max-skew <dB>`, applied where ears actually exist — the WAV path. Per-ear filters are
impossible in the real-time path by definition (one mono IR is loaded on both channels), so a run
that writes convolvers refuses `split` with exit 2 instead of quietly becoming `report`.

* The **balance** is `median(left) − median(right)` and is cancelled in full; the **shape mismatch**
  is the captured difference with that balance removed, cancelled band by band through the same
  `fit::regularize_correction` as everything else. Median, not mean, for the reason `reference_db`
  already gives: a capture saved with one ear's analyser 94 dB off has a 94 dB *difference* and no
  shape difference at all.
* The knee comes from the **model's own captured curve**, never from the mismatch being cancelled —
  otherwise a pair with one enormous narrow difference raises its own knee and negotiates its way out
  of the correction, which is the failure `Policy::auto` was written to avoid.
* The trim is applied `left += t`, `right −= t`, so **the pair's mean is the shared curve at every
  knot**; the delivered difference is `D + 2t`, which pins both limits: `t = 0` is today's answer and
  `t = −D/2` is a perfectly matched pair.
* `--stereo-max-skew` (default 6.00 dB) bounds `|G_L − G_R|` per knot and takes the excess **off both
  sides equally**, so the ceiling cannot move the centre. It is a ceiling on the damage the flag can
  do to an image, chosen against this database — not a claim about perception, and no standard is
  cited for it, because none reviewed defines a correct amount of left/right difference between two
  headphones.
* `--notch-mode safe` re-applies **per ear, after the trim**, because `safe_cap_db` is the last word
  on what a model may ask in its concha-null band. Where it bites the centre moves by half, and the
  report prints `Centre : moved in 15 knot(s)` rather than reciting the design's promise (measured on
  the shipped DT 770 at the default knee: 6 left, 9 right).
* **The figure gained the trace.** `left − right` is drawn on panel 2 — the panel where the reader is
  already looking for what is wrong — in the ears' own blue, with the worst disagreement quoted in
  its legend entry. Opt-in like everything else here: `show_ears` comes from `--stereo` being
  something other than `off`, and a test counts the trace's pixels inside panel 2 and asserts the
  default figure contains none of that colour there.

**Sources, and what they do not say.** The correlation reading printed in the report (+1 the same
feature, 0 independent, −1 inverted) is the standards-body definition: ASA/ANSI acoustical
terminology, *cross correlation; interaural cross correlation*, annotation 2, read at
[asastandards.org](https://asastandards.org/terms/interaural-cross-correlation/) rather than
paraphrased. The band names on the mismatch line are IEC 61260-1, reused from PR 6. Everything else
in the module — the balance, the shape mismatch, the skew ceiling, `RESOLVED_WINDOW_POINTS`, the
`±`-trim that holds the centre — is HyprEQ's, chosen against the numbers above. Nothing reviewed
prescribes how different two headphone ears are allowed to be, and the report does not imply
otherwise: below the grid's resolution the verdict stops at "mismatch".

**Acceptance, and how it was met.** The original list asked for a bounded-skew mirrored pair, a
flagged asymmetric pair, a skew curve on the figure, and mono models byte-identical. All four are
met, and the shape of the last one changed with the premise: a mono model has no pair, so it says so
in words instead of comparing a curve against itself and printing a flattering zero.

- `a_mirrored_pair_reports_nothing_to_fix` pins the mirrored case at 0.00 / 0.00 with `r = +1.000`,
  and `one_filter_handed_to_both_channels_is_the_shared_filter` carries it down to the bytes of the
  convolution, so `split` on an identical pair cannot sound different from `off`.
- `a_difference_above_the_ceiling_is_a_mismatch_however_it_is_shaped` runs the same 6 dB difference at
  two widths ten times apart and asserts both come back `mismatch` with `resolved == false`;
  `narrowness_is_only_claimed_when_the_grid_can_see_it` pins the boundary in grid terms (0.083 oct →
  2.01 points, unresolved; 601 capture points → resolved) and asserts the resolution line appears
  exactly when it is owed.
- `the_feature_match_reads_the_same_feature_twice_and_two_accidents_once` is the trap the module
  exists for: two unrelated spikes sitting on a shared bass shelf correlate at **+0.97** if the
  *curves* are correlated, while the report claims to be reporting features.
- `a_curve_with_no_wiggle_has_no_correlation_to_report` keeps `None` from becoming `0.00`, which under
  the cited definition would call two identical ears independent.
- `the_ceiling_holds_the_pair_together_without_moving_its_middle` pins the bound (2 dB on a 9 dB
  difference), the slack case that changes nothing, and that the pair's mean stayed on the curve to
  1e-9; `a_ceiling_that_bit_after_the_trim_says_the_centre_moved` pins the wording that admits it.
- `a_knee_that_believes_nothing_leaves_the_shape_and_fixes_the_balance` and
  `a_knee_that_believes_everything_cancels_the_difference_it_was_shown` are the two limits.
- End to end (`tests/cli_report.rs`): the block sits under the model banner and names the shipped
  numbers (`+1.27 dB`, `-7.72 dB @ 15110 Hz`); `report` writes byte-identical convolvers and audio to
  the default run; `split` does not, prints both ear filters, never prints a filter it did not apply,
  and accounts for the 15 knots the ceiling moved; the refusals exit 2 before anything is written and
  a mono file exits 1 naming its channel count.
- Sabotage runs, each killing tests: the balance folded into the shape (3 tests); a `None` correlation
  printed as `0.00`; `resolved`'s threshold lowered to 1.0; the balance applied to one ear only; the
  skew bound computed but never applied; the cancellation asked at full depth instead of half;
  correlating the curves instead of their residuals (+0.965 where the test wants < 0.5); the trim line
  printed when nothing was trimmed; `median_db` replaced by a mean (2 fit tests); `report` building a
  pair; `split` applying the shared filter to both channels; the mono refusal replaced by a silent
  fallback; the `split`-without-audio refusal removed; the per-ear ceiling skipped; the ears block
  moved above the model banner; and the figure trace not drawn.

**Not shipped, on purpose:** per-ear *real-time* convolvers — a second IR file and a second `afir`
invocation per model, for a chain that can split the channels. The CLI refuses that case rather than
half-doing it, and the WAV path is where a two-filter chain can actually be built.

### PR 5 — Experimental phase correction — **shipped** as `--experimental-phase <HZ>`

**Problem, as scoped:** minimum-phase was assumed (§3.4) with no way to ask the question.

**What shipped.** `src/phase.rs`. The finished filter's spectrum is read, its **group delay**
is corrected inside a band, and the magnitude is carried through untouched; the stage never
re-derives `|H|`. Costs are printed rather than assumed: the group delay moved, the delay the
band was flattened onto, the kernel growth, the latency, the magnitude deviation and the
tail floor. Off by default, and the default report is the parity-pinned one (98/98 with the
flag absent).

**Three findings, all measured, none of them visible in the plan:**

1. *Flattening a bass band is not a cheap operation, and the plan assumed it was.* On the
   shipped DT 770 correction the minimum-phase group delay spans **9.6 ms across 10–200 Hz**,
   and the delay below 20 Hz is enormous because a steep low shelf hides it inside a decay.
   A truly flat line therefore has to either advance ~220 ms of deep-bass energy — which is
   not a causal operation: the energy leaves the front of the kernel, wraps on the padded
   grid, and came out as an absurd latency plus a **0.30 dB** magnitude error — or add the
   same as latency. So the correction is bounded per frequency
   (`DEFAULT_GD_BUDGET_MS`, 0.56 ms, the audibility scale rather than what the band could
   absorb), and the report prints the budget against the band's real span so the reader sees
   how much was closed and how much was left alone on purpose. Published designs take
   *permitted delay* as an input for this reason; here the budget is the same quantity.
2. *The band's delay change must integrate to zero.* The phase change is the integral of the
   delay change, so any residue at the bottom edge of the band is a **step in phase**, and a
   step is an infinite group-delay spike. With the clamp in place and no residue removal this
   measured **867 ms** of "group delay moved" on a 5 ms budget. The residue is now removed and
   the clamp re-applied until the integral closes.
3. *Where the kernel is cut is a latency decision, not an energy one.* The reshape spreads
   the response, so the delivered window has to start somewhere. Taking the densest stretch of
   the grid — the obvious choice — cost **460 ms** of latency on the shipped filter, because
   sliding a kernel *is* a delay and a bass correction's response is diffuse enough that its
   densest stretch starts a quarter of a second from the onset. The smallest shift that meets
   the floor is used instead. Uncapped growth was also tried: it reached **524 288 taps and
   5.5 s** chasing a tail that never reaches −130 dBFS, so the kernel now grows on a capped
   ladder (≤ 4× the requested length) and prints `held_floor: false` when the floor is not met
   rather than pretending.

**Prior art, read rather than described.** Budde's DAGA 2012 mixed-phase method (its own
implementation notes report that interpolating between minimum and linear phase puts energy
outside the causal support and that truncating it changes the magnitude — which is why HyprEQ
never re-derives `|H|` instead of factorising it, and why the alternating loop's divergence
risk does not exist here). The CamillaFIR manual, whose posture this copies: group-delay work
is a bass-focused guard, not a wideband phase shaper. NYU's minimum-phase FIR design notes,
quoted: "the delay of a linear-phase filter is equal to (N − 1)/2", and a narrow transition
band with high stop-band attenuation "requires making the filter long" — the trade this stage
lives inside, and the reason accuracy here is bought with taps and an 8× padded transform
rather than given up. Liski, Mäkivirta & Välimäki (TASLP 29, 2189–2201, 2021) supplies the
threshold the budget is set against; that paper's own abstract records that the ear tolerates
**more** phase processing in the low frequencies, which argues for treating a bass
GD-flattening stage as experimental rather than as a recommendation, and that is how it ships.
The full text was behind a paywall at the time of writing, so no per-frequency threshold
numbers are quoted that could not be read.

**Acceptance, and how it was met:**
- `the_magnitude_survives_within_rounding`, and end to end in
  `the_stage_moves_the_phase_and_leaves_the_curve_alone`: the written filter's magnitude
  against the default run's, **< 0.1 dB** across the band, measured on the artefact.
- `the_bass_group_delay_actually_flattens`, plus the same spread reduction measured on the
  delivered WAV: the band's delay spread goes down, not merely somewhere else.
- `nothing_above_the_taper_is_touched`: magnitude exact to 1e-12 relative before truncation,
  and zero weight above the taper.
- `a_filter_that_is_already_flat_in_the_band_comes_back_unchanged`,
  `nonsense_requests_are_declined_rather_than_attempted`,
  `the_taper_weight_is_continuous_and_ends_where_it_says`,
  `keeping_the_window_with_the_most_energy_is_what_buys_the_magnitude`,
  `the_unwrapper_removes_wraps_a_fit_cannot_tolerate`, `the_report_says_what_moved_and_what_it_does_not_claim`.
- Figure: no panel was added, because none was needed — the plots are drawn from the IR that
  gets written, so the phase and IR panels show the difference on their own.

**Two measurements that were wrong before they were fixed, recorded because both passed at
the time.** An energy centroid cannot see bass phase: it read 1.2 samples before *and* after a
correct stage, because the untouched treble dominates the average. Absolute group delay cannot
see a delay: it moves by π everywhere by definition. Both assertions now compare *shape* with
each filter's own median removed.

**Not claimed:** that this sounds better, or that it is audible — see §6 and the README, which
say the audibility of phase correction is contested. The budget is a HyprEQ configuration
constant, not a compliance limit, and the stage's latency (about 120 ms on the shipped DT 770
filter at the default budget) makes it wrong for anything live.

### PR 6 — Distortion-aware assessment — **shipped** as `--assess`

**Problem:** a big boost is reported without any statement about what it demands (§1, cap 6).

**Shipped:** `src/assess.rs` and the `--assess` / `--assess-budget` options. It puts the
applied correction on the content's own third-octave spectrum (IEC 61260-1 base-10, Hann,
Parseval-normalized; the numerical rules are in `docs/SETUP.md` §5) and reports three facts:
how many bands the filter boosts past the budget while carrying content, which of them is the
loudest after the filter, and which reach the limiter's ceiling once the convolver's headroom
is applied. Channels are combined by maximum, never averaged.

It is a warning system on measured quantities, exactly as scoped: it models no driver, no
amplifier and no distortion threshold, and prints no excursion number.

**Acceptance, and how it was met:**
- `a_flat_curve_demands_nothing` — a 0 dB curve names nothing.
- `a_bass_shelf_is_only_a_demand_when_the_file_has_bass` — a +9 dB shelf against a 60 Hz tone
  names 63 Hz as the peak, at about +6 dBFS after the filter; the same shelf against content
  with nothing in the bass names nothing; the budget is a knob, not a constant.
- `the_worst_channel_is_the_one_that_matters`, `without_audio_the_curve_is_still_described`,
  `degenerate_input_produces_numbers_not_panics`, `the_grid_is_iec_61260_base_10`,
  `a_tone_lands_in_its_own_band_at_its_own_level`, `a_bare_frequency_is_named_by_its_band`.
- Sabotage runs: replacing the nearest-centre bin partition with permissive edge rounding, and
  replacing `n · Σw²` with `n²`, each fail tests; dropping the Hann window leaves a tone only
  ~32 dB out of its neighbours against the ~105 dB the window delivers, and the 40 dB bound
  catches it.

**One acceptance line moved.** The plan asked for "the limiter report gains a line stating the
measured margin after the filter". The limiter report is byte-compared against the reference
program in `scripts/parity.py`, so a line added there is a parity failure by construction. The
margin statement ships in the opt-in section instead (`Ceiling : N band(s) reach …; the limiter
is doing this work`), which is where it can exist without breaking parity. The default report is
byte-identical with and without the flag — checked end to end, not just by inspection: the same
input through both paths writes byte-identical WAVs.

---

## 5. What this crate links

Six crates, no build-script code generation, and no C library except the ones `plotters`
pulls through `font-kit`. Every one of them is MIT or Apache-2.0, so the whole tree stays
inside the MIT licence this project ships under. Versions are the ones `Cargo.lock` pins.

| Crate | Locked version | Licence | What it does here |
| --- | --- | --- | --- |
| [`rustfft`](https://crates.io/crates/rustfft) | 6.4.1 | MIT OR Apache-2.0 | The transforms behind the NumPy-convention wrappers in `src/fft.rs` (`ifft` scaling, `n/2+1` bins, positive Nyquist). It is reached only through those wrappers, so rustfft never gets to decide a convention |
| [`serde`](https://crates.io/crates/serde) | 1.0.229 | MIT OR Apache-2.0 | Two `derive` implementations: the measurement database and a model entry |
| [`serde_json`](https://crates.io/crates/serde_json) | 1.0.154 | MIT OR Apache-2.0 | Loads `measurements.json` with `preserve_order` on. Model order is load-bearing — it is the order `--list-models`, `--headphone all` and the default model come out in — and the default map would sort it alphabetically |
| [`clap`](https://crates.io/crates/clap) | 4.6.7 | MIT OR Apache-2.0 | Argument parsing. The argparse-shaped domain validation, exit code `2` and message wording are layered on top of it, not delegated to it |
| [`plotters`](https://crates.io/crates/plotters) | 0.3.7 | MIT | The verification figure's bitmap and TrueType text backend, with `default-features = false`. Axes, ticks, clipping, legends and callouts are hand-drawn on purpose, to match the reference's matplotlib geometry |
| [`png`](https://crates.io/crates/png) | 0.17.16 (dev) | MIT OR Apache-2.0 | Reads the rendered figure back in tests, so "the panel has ink in it" is an assertion rather than a hope |

**Licensing rule.** Only MIT/Apache-2.0-compatible Rust crates get linked, and only for
plumbing — decoding, encoding, transforms, text. The numerics this project is graded
against stay in-tree. Where an algorithm in this roadmap has a published reference
implementation, it is cited for its equations and not for its code: the regularized
inversion work in AKtools is MATLAB under the **EUPL 1.1 or later** — a copyleft the MIT tree
cannot absorb, so it could not be linked or translated even if a translation were worth the
effort — and the MIT-licensed equivalent in pyfar is not a dependency here either. Anything the roadmap
adopts is re-implemented in `f64` from the published maths, with the paper cited in the
commit that adds it.

**What stays hand-rolled, and why.** PCHIP in log-frequency, the PCG64 dither stream, the
true-peak interpolation filter, the two-stage limiter envelope, the half-Kaiser tail, and
the ties-to-even rounding are all pinned to reference behaviour that a generic crate is
entitled to implement differently (a spline library may use natural cubic splines; a
random library may use a different stream; a limiter library will not share this
lookahead bookkeeping). Those are the parts where "equivalent" is not the requirement, so
they stay here with tests.

**Nothing else is linked, deliberately.** PR 1–6 are expected to need no new dependency:
regularization is a change to the least-squares target, smoothing statistics are a few
prefix sums, and loudness or new containers would only matter if the CLI's inputs changed.
If a dependency is ever added, it joins this table with its licence and the one sentence
explaining why it could not stay hand-rolled.

---

## 6. Non-goals

- **Not** a measurement application. There is no sweep engine here and PR 3 does not add
  one; it accepts somebody else's CSV.
- **Not** a replacement for AutoEq's database. Importing third-party measurement sets is a
  data-licensing question before it is a code question.
- **Not** a real-time engine. CamillaDSP and mpv/FFmpeg `afir` already run the filters
  this project produces; the product is the filter plus the evidence.
- **Not** a claim that phase or distortion correction is audible in every case. PR 5 and
  PR 6 ship as measurement and warning, with the contested evidence cited rather than
  quietly dropped.
