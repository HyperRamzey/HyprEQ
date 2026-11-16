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
| The engine is small and legible: `model → pchip → ir → convolve → limiter → plot`, with ~190 unit tests | A base you can read is a base you can extend |

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
| Weak at | Fixing a bad *fit* decision (§3.1), fixture awareness (§3.3), stereo coherence, phase choice | Anything requiring your own measurement pipeline or a convolution-specific artifact of this shape | Automation and reproducibility; it is an interactive app | Deciding anything |

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
> Further: *Audibility of Group-Delay Equalization* (Aalto, 2021), on when group-delay
> correction is and is not audible.

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

## 4. Roadmap

Every PR below keeps one invariant: **with no new flags, output is byte-identical to
today's**, and `scripts/parity.py` stays green. New behaviour is additive and opt-in, so
the parity oracle remains a live check on the existing engine while each change is
proved by its own tests.

### PR 1 — Fit-aware regularization *(highest value)*

**Problem:** one smoothing/regularization policy for every curve (§3.1, §3.2).

**Method:** estimate λ per model from the raw curve's own deviation from its
half-octave-smoothed version (Gómez-Bolaños 2016), and use it to (a) cap notch depth in
`apply_notch_mode`, and (b) weight the control-point derivation in
`model::normalize_model_entry` where the raw curve is being reduced to knots.

**Implementation shape:** a new `fit.rs` — `smooth_octave(f, g, fraction)`,
`regularization_lambda(g)`, `notch_cap(lambda)`-style pure f64 functions — called from
`ir::apply_notch_mode` and from the model normalizer, with `--regularize auto|off|<λ>`.
`auto` is *not* the default in PR 1; it becomes the default only after the figure shows
the two side by side on real curves.

**Acceptance:** unit tests for the smoother on known curves (a pure tone, an impulse
notch, a shelf) proving the λ estimate rises with notch depth/width and is invariant to
flat gain; a test that `--regularize off` reproduces today's bytes exactly; a figure panel
overlay of auto vs off; parity unchanged.

### PR 2 — Fixture-aware targets

**Problem:** targets are fixture-specific and nothing records the fixture (§3.3).

**Method:** promote what `rig_info` already says into structure — a `fixture` key on the
model and a `fixture` key on each target — and finally *use* it: print the fixture in the
report header and the figure's title block, with `rig_info` kept as the human-readable
caption beside it. Ship the equivalents that have a defensible published curve (the
Harman family on GRAS 45CA/711 and the newer GRAS KB5000 and B&K 5128 translations), and
refuse — or require `--force` — when model fixture and target fixture disagree. Where no
equivalent is available, say so in the report instead of silently EQ-ing to a foreign
target.

**Acceptance:** the three shipped models gain the fixture their `rig_info` already
describes and keep producing byte-identical filters; a model with `fixture: bk5128` and
`target: harman_2018` fails with a message naming both; `--list-models` shows fixtures; the
figure's title block names the fixture it assumed; an unknown fixture string is reported as
a database error the way a bad target name already is.

### PR 3 — Acoustic verification loop

**Problem:** the loop is open (§3.5).

**Method:** accept a post-EQ measurement (`--verify-with post_eq.csv`), align it to the
predicted response (same smoothing, same frequency grid, delay/gain-insensitive), and
report the residual per band with pass/fail thresholds, in the audit CSV and as a fifth
panel or overlay.

**Implementation shape:** reuse `model::load_csv_points` and `pchip` for alignment; reuse
`measure.rs` for the statistics; a new `verify.rs` for the band residual and thresholds.

**Acceptance:** a synthetic test where the post-EQ curve is the predicted curve convolved
with a known residual — the tool must name the band and the dB error; a test that an
unrelated curve fails loudly; the report distinguishes "measurement is quiet/noisy" from
"the fit is wrong".

### PR 4 — Stereo-aware correction

**Problem:** L and R are fit independently today.

**Method:** when a model carries L/R columns (the plot already draws raw L/R when present),
fit a *shared* correction where the two ears disagree only in a narrow, uncorrelated way,
and bound the L/R gain difference of the applied filter per band; report the induced
inter-aural skew.

**Acceptance:** a test that a mirrored L/R pair produces a correction with bounded skew; a
test that a genuinely asymmetric pair still gets corrected but is flagged; the figure shows
the skew curve; mono models are byte-identical.

### PR 5 — Experimental phase correction *(behind `--experimental`, documented as such)*

**Problem:** minimum-phase is assumed (§3.4).

**Method:** keep the minimum-phase core, and add a mixed-phase option: minimum-phase
magnitude above a crossover plus a linear-phase (or selectively all-pass-equalized) low
end, which is the shape CamillaFIR and the group-delay literature converge on.

**Acceptance:** the flag must produce a filter whose magnitude matches the minimum-phase
one within a stated tolerance (so phase is the only thing that moved); the figure's phase
and IR panels must show the difference; the README states that audibility is contested.

### PR 6 — Distortion-aware assessment

**Problem:** a big boost is reported without any statement about what it demands (§1, cap 6).

**Method:** from the applied EQ curve and the measured signal's own spectrum, estimate the
per-band excursion/level demand the filter adds, and warn where the demanded boost exceeds
a configurable budget — using the headroom, pre/post true-peak, and limiter statistics that
are already computed. This is deliberately a *warning system* on measured quantities, not a
driver model.

**Acceptance:** a test that a 0 dB curve triggers nothing; a test that a +9 dB bass shelf
on bass-heavy content triggers a named band; the limiter report gains a line stating the
measured margin after the filter, which is the number a listener actually cares about.

---

## 5. Library notes for the Rust side

What the crate uses today (`rustfft` behind the NumPy-compatible wrappers in `fft.rs`,
`serde`/`serde_json` with `preserve_order`, `clap`, `plotters` for the figure) and what is
worth knowing about for the roadmap. All versions below were checked against crates.io
when this was written.

| Crate | Version | What it would be for here |
| --- | --- | --- |
| [`realfft`](https://crates.io/crates/realfft) | 3.5.1 | Real-to-complex / complex-to-real transforms directly; would shorten `fft.rs`'s rfft/irfft wrappers, at the cost of re-deriving the NumPy-exact semantics those wrappers exist to guarantee |
| [`microfft`](https://crates.io/crates/microfft) | 0.6.0 | Small fixed-size transforms with a compile-time plan — a candidate for the true-peak interpolation filters if the detector ever moves to a fixed 4×/32× path |
| [`neco-minphase`](https://crates.io/crates/neco-minphase) | 0.1.2 | Minimum-phase spectrum and overlap-add convolution kernels — the closest existing crate to `ir::synthesize_minimum_phase_ir` + `convolve.rs`; worth reading before writing PR 5's mixed-phase variant |
| [`fft-convolver`](https://crates.io/crates/fft-convolver) / [`convolution-dsp`](https://crates.io/crates/convolution-dsp) / [`rconvolve`](https://crates.io/crates/rconvolve) | 0.4.0 / 0.3.0 / 0.1.2 | Real-time-capable partitioned convolution. Only relevant if a streaming mode is ever wanted; `convolve_planar_fft` is offline and deliberately matches the reference's block structure |
| [`ebur128`](https://crates.io/crates/ebur128) | 0.1.11 | EBU R128 loudness. The reports already carry peak/RMS/ITU-x4 true-peak; integrated loudness and LRA are the obvious additions to a mastering-oriented report (the meter in `meter.rs` is a console peak-hold meter, not a loudness meter, and is not a substitute) |
| [`hound`](https://crates.io/crates/hound) / [`symphonia`](https://crates.io/crates/symphonia) | 3.5.1 / 0.6.1 | WAV I/O and general container/codec decoding. `riff.rs`/`wav.rs` exist for Python-`wave`-compatible semantics (chunk layout, ragged frames, extensible GUID, TPDF PCM); symphonia is the answer only if non-WAV input is ever accepted |
| [`plotters`](https://crates.io/crates/plotters) | (in use) | The figure's bitmap/text backend. Axes, ticks, clipping and callouts are hand-drawn on purpose, to match the reference's matplotlib geometry |

**What stays hand-rolled, and why.** PCHIP in log-frequency, the PCG64 dither stream, the
true-peak interpolation filter, the two-stage limiter envelope, the half-Kaiser tail, and
the ties-to-even rounding are all pinned to reference behaviour that a generic crate is
entitled to implement differently (a spline library may use natural cubic splines; a
random library may use a different stream; a limiter library will not share this
lookahead bookkeeping). Those are the parts where "equivalent" is not the requirement, so
they stay here with tests. Anything that is only *plumbing* — decoding, encoding,
transforms — is fair game to delegate.

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
