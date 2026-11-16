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

The published routines regularize the *deviation from the all-pass reference* instead, or
equivalently load the diagonal with a frequency-dependent weight so that a band is only
flattened where it has earned it:

```text
g = (HᴴH + diag(β))⁻¹ Hᴴ d ,     β(f) ≥ 0
```

with `β` derived from the response itself — §3.1's residual between the raw curve and its
octave-smoothed version is exactly such a source — so a smooth band gets a small `β`
(correction proceeds) and a wild band gets a large one (correction falls back to `d`,
which is unity). Two properties follow, and both are testable rather than aspirational:

* **the weight is referred to the unity level, not to `max|H|`.** A `λ` scaled by the peak
  of the response means one thing on a set with a 6 dB null and another thing on a set with
  a 2 dB null; scaled by the unity reference it means the same thing on both. This is
  `λ_unity` in PR 1.
* **`β(f) → ∞` gives `|G(f)| → 1`, not `0`.** One assertion, and it is what stops the naive
  form from coming back in a later refactor.

Prior art worth reading before writing it: the same construction ships as
`AKregulatedInversion` in **AKtools** (TU Berlin Audio Techniques; MATLAB), and the
3GPP VR characterisation in **TR 26.818 §8.3.8** used it to build 16 384-tap compensation
filters as regularized least-squares inversions with **1/6-octave-smoothed inverted copies
of the measured responses as the regularization curves** — an industry document choosing the
smoothed-inverse formulation over the raw one, which is the same argument §3.1 makes from
perception. In Python it exists as `pyfar.dsp.regularized_spectrum_inversion` (pyfar 0.8.1,
MIT). Neither is a dependency here: AKtools ships its licence as a PDF bundled with the
toolbox, so it is maths to re-derive rather than code to link, and pyfar is a numerics
package this crate has no reason to pull in for one equation. §5 states the rule; PR 1
implements the maths in `f64` against this document.

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

### PR 1 — Fit-aware regularization *(highest value)*

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

**Statistics:** where a fixture has repeated measurements available, the reference curve is
the **upper envelope** across repeats and the disagreement between seats is reported as
`1.4826 × MAD` beside the fixture name (§3.7) — a fixture whose seats disagree by 4 dB
around 200 Hz is telling you the target there means something different per head, and the
report should say so rather than average it away.

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

**Envelope first:** a mixed-phase stage that chases a raw magnitude will invent steep
skirts that no headphone has. Constrain the target log-magnitude's continuity *before* the
minimum-phase conversion — the order DRC-FIR made its name on — so the all-pass section
handles departure from minimum phase rather than fighting an over-detailed target.

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
inversion work in AKtools is MATLAB under GPL, so it could not be linked or translated
into a tree that ships under MIT even if a translation were worth the effort, and the
MIT-licensed equivalent in pyfar is not a dependency here either. Anything the roadmap
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
