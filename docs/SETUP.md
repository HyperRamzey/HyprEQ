# Setting up HyprEQ

HyprEQ ships as one Rust crate (`hypreq`, a library plus a binary) and one Python
script (`hypreq.py`). The Rust crate is the implementation; `hypreq.py` is kept only as
the **parity oracle** the harness diffs the Rust output against, and it is expected to
drift behind the Rust side as features are added.

```text
Cargo.toml            crate manifest (numerical policy lives in its comments)
src/                  the implementation
hypreq.py             parity oracle — not the thing you build
measurements.json     target curves + headphone definitions, in file order
models/               optional drop-in .json / .csv models (scanned at startup)
scripts/parity.py     Rust-vs-oracle comparison harness
docs/                 this file
```

## 1. Prerequisites

| What | Needed for | Notes |
| --- | --- | --- |
| Rust ≥ 1.87 (`rustc`, `cargo`) | everything | `rust-version` in `Cargo.toml` is 1.87, edition 2021 |
| `pkg-config`, `fontconfig`, `freetype` (C dev files) | the verification PNG | the `ttf` feature of `plotters` goes through `font-kit`, which links against the system fontconfig/freetype and resolves `"sans-serif"` **at runtime** |
| a usable `sans-serif` font | the verification PNG | any family fontconfig can resolve; DejaVu is enough |
| Python ≥ 3.10, `numpy`, `soundfile` | `scripts/parity.py` | `soundfile` wrapslibsndfile for payload decoding |
| Python `matplotlib` | the oracle's PNG only | the Rust figure does not use it |

The font dependencies are not optional if you want the figure: with the `ttf` feature
enabled a machine with no fontconfig configured fails to *link*, and a machine with
fontconfig but no resolvable `sans-serif` panics inside `plotters` while drawing text.
`--no-plot` avoids both paths entirely.

On Arch:

```bash
sudo pacman -S --needed rust fontconfig freetype2 ttf-dejavu python-numpy python-soundfile python-matplotlib
```

## 2. Build

```bash
cargo build --release          # ./target/release/hypreq
cargo build                    # dev profile, faster compile, same numerics
```

The release profile sets `lto = "fat"`, `codegen-units = 1`, `opt-level = 3` and
**nothing else**. Read the comment in `Cargo.toml` before touching it: the DSP must stay
IEEE-754 exact, so no `-C target-cpu=native` with FMA, no `-C llvm-args=--fp-contract=fast`
and no fast-math. A fused multiply-add rounds once where the reference rounds twice, and
the true-peak interpolator, the limiter's gain scheduler and the Hilbert-cepstral
minimum-phase fold would drift off the oracle.

## 3. Run

```bash
# everything in the database: f64 convolver, two f32 real-time convolvers, PNG
./target/release/hypreq --headphone all --export-csv

# one model, lighter filter, different rate
./target/release/hypreq -H dt770_80 --notch-mode safe
./target/release/hypreq -H ft1pro --taps 16384 --fs 44100

# what is loaded, in database order
./target/release/hypreq --list-models

# measure / EQ / true-peak-limit a WAV
./target/release/hypreq -H jbl770nc --input song.wav --out-bits 24
./target/release/hypreq --measure-wav song.wav          # limiter + meters, no EQ, no write
```

Artefacts are written to the **current directory**, named from the model's
`file_prefix` (so `dt770_80` writes `dt770_80ohm_*`):

| File | Contents |
| --- | --- |
| `<prefix>_f64_<rate>.wav` | 64-bit float mono convolver for `mpv`/FFmpeg (`afir=precision=double`) |
| `<prefix>_mastering_mono_<rate>.wav` | 32-bit float, 16,384 taps, PipeWire/DAW |
| `<prefix>_mastering_stereo_<rate>.wav` | the same IR on both channels |
| `<prefix>_final_verification.png` | 1950 × 1725, 4-panel verification figure |
| `<prefix>_audit_data.csv` | frequency, raw, target, raw error, applied EQ (`--export-csv`) |
| `ir.wav` | copy of the single selected model's f64 convolver |
| `<input>_<model>_tp_<ratio>.wav` | audio-mode output |

`<rate>` is `48k`, `96k`, `44.1k`, `88.2k` and so on.

`--data-file` and `--models-dir` point at a different database/drop-in directory;
without them the defaults (`measurements.json`, `models/`) are resolved relative to the
current directory.

## 4. Verification gates

Every one of these has to be clean before a change is pushed:

```bash
cargo fmt --check
cargo build
cargo build --release
cargo test                      # unit + integration, default parallelism
cargo clippy --all-targets      # zero warnings, including tests
python3 scripts/parity.py       # all scenarios

# documentation links must resolve — this is what catches a doc that still names a
# function the code no longer has
env -u RUSTC_WORKSPACE_WRAPPER \
  RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc --no-deps
```

`cargo test` runs a few deliberately expensive tests (full-length convolutions and
figure renders); two are `#[ignore]`d because they take minutes and are covered by the
parity harness instead:

```bash
cargo test -- --include-ignored
```

### The parity harness

```bash
python3 scripts/parity.py                 # everything
python3 scripts/parity.py --only audio    # scenarios whose name contains "audio"
python3 scripts/parity.py --keep          # leave the scratch tree behind
python3 scripts/parity.py --selftest      # check the harness's own normalisers, no binaries
python3 scripts/parity.py --rust ./target/release/hypreq --python python3 --ref hypreq.py
```

Each of the 28 scenarios runs `hypreq.py` and the Rust binary in **separate scratch
directories** seeded with the same `measurements.json` and the same input WAV, then
compares:

* **stdout** byte-for-byte after normalising the absolute paths both tools print;
* **WAV containers** exactly (chunk order, format tag, channels, rate, bits, byte size);
* **WAV payloads** — dithered PCM must be byte-identical, float payloads within a
  per-bit-depth tolerance (a 32-bit payload is judged in f32 ULP; comparing it against an
  f64-scale tolerance is a harness bug, not a divergence);
* **audit CSVs** byte-for-byte;
* **usage errors**: message wording and exit code `2`;
* the **PNG**: the Rust side must write a decodable `1950 × 1725` image **with plotted ink
  inside each of the four panels** — counted strictly inside each panel, inset past the
  frame and the tick marks, so an image whose panels hold nothing but their own frames
  fails. matplotlib and `plotters` are different renderers, so pixels are never compared;
  exact geometry is covered by unit tests that decode the PNG the same way.

One deliberate divergence, because the honest version of "the figures match" is "the default
figures match": with `--stereo` set to anything but `off`, the Rust figure draws a fifth trace the
reference has no word for — `left − right` on panel 2, in the ears' own blue, with the worst
disagreement quoted in its legend entry. It is opt-in for exactly that reason, and
`the_ear_trace_draws_ink_in_the_error_panel` holds both halves: the trace's pixels present when the
flag asks, and none of that colour anywhere in panel 2 when it does not.

The harness exits non-zero if any comparison fails, so it is usable as a CI gate.

#### One usage-error message is spelled differently by different CPythons

`usage wording` is compared word for word, with two exceptions the harness owns: the
`hypreq.py:` / `error:` prefix, and the quoting of argparse's `invalid choice:` message. That
exception is not laziness — argparse has changed the spelling of this message twice and reverted
one of its own changes, and it did so at **patch** granularity, so there is no version range that
is "the" wording:

| interpreter | bad `--notch-mode` (string choices) | bad `--out-bits` (int choices) |
| --- | --- | --- |
| ≤ 3.12.7, 3.13.0 | `'bogus' (choose from 'safe', 'original', 'zero')` | `20 (choose from 16, 24, 32, 64)` |
| 3.12.8 – 3.14.3 | `'bogus' (choose from safe, original, zero)` | `'20' (choose from 16, 24, 32, 64)` |
| 3.13.7+, 3.14.4+ | `'bogus' (choose from 'safe', 'original', 'zero')` | `'20' (choose from '16', '24', '32', '64')` |

The middle row is `str(value)` + `map(str, choices)` rendered through `%(value)r`, which is why
the *value* stays quoted while the list loses its quotes; the last row is `repr(str(x))` in both
slots. The Rust binary prints one fixed string (the last row, matching what people actually run
now), so a strict comparison would make parity red on the interpreters in rows one and two while
the CLI is telling the user something entirely correct. `canonicalize_choice` therefore strips
quotes from inside that message and nothing else; the flag name, the phrase, the value, the set
of choices and the exit code are all still compared.

`--selftest` asserts both halves — that the three spellings collapse, and that the normaliser
does not swallow content (a different flag, a different value, a shortened choice list, and every
non-choice message must survive byte for byte). It runs without a binary, numpy, or the reference.

To reproduce the table, or to check a new interpreter:

```bash
uv python install 3.11 3.12.7 3.13.0            # or any patch level you care about
uv venv /tmp/v311 --python 3.11 && uv pip install -p /tmp/v311/bin/python numpy soundfile
python3 scripts/parity.py --python /tmp/v311/bin/python --only usage   # 33/33
/tmp/v311/bin/python hypreq.py --out-bits 20 | grep 'invalid choice'   # the raw spelling
```

`numpy` and `soundfile` are needed because the reference imports them before it parses the
command line; `matplotlib` is not, because the usage-error scenarios never reach the plot.

## 5. Numerical rules

* All DSP state is `f64`. Only a container may drop to `f32`, and only at write time.
* `+`, `-`, `*` must not be contracted into FMA. Do not enable fast-math anywhere.
* Where the oracle uses `round()` / `np.round()`, Rust uses `.round_ties_even()`:
  Python rounds half to even, and `44.1 kHz` exposes it (`5 ms * 44.1 = 220.5` → `220`).
* `src/fft.rs` wraps `rustfft` to NumPy conventions (`ifft` scales by `1/n`, `rfft`
  returns `n/2+1` bins, `rfftfreq` keeps a positive Nyquist bin). rustfft and NumPy's
  pocketfft factor the same transform differently, so the convolver matches the oracle to
  a couple of ULP rather than bit-for-bit — that residual is transform structure, and it
  is what the f32 real-time convolvers' 1-ULP parity tolerance absorbs.
* PCM output uses a fixed TPDF dither stream whose draw order is part of the byte
  pattern: `(frame, channel)` row-major, one pair of uniform draws per sample. Changing
  the layout changes the bytes.

**Band assessment (`--assess`, `src/assess.rs`).** Nothing here is compared against the
reference program, because the reference has no equivalent; the rules below are what makes the
numbers defensible on their own:

- Bands are **IEC 61260-1** base-10: centre `1000 · 10^(k/10)`, edges `centre · 10^(±1/20)`,
  and bands named by the standard's nominal value (20, 25, 31.5 … 16 k, 20 k). The nominal
  mantissa index is `k - 10·⌊k/10⌋`, not a truncated remainder: at `k = -17` the truncated
  version reports 50 Hz where the standard says 20 Hz.
- The grid's ends are **indices** (`k = -17 … 13`), trimmed at Nyquist. They cannot be frequency
  tests: band `-17`'s exact centre is `19.9526 Hz`, so `centre > 20.0` drops the nominal 20 Hz
  band and quietly starts the grid at 25.12 Hz.
- Levels are windowed RMS with a Hann window **sized to the record**, zero-padded out to the
  transform length, divided by `n · Σw²` with `Σw²` summed over the record only. Sizing the taper
  to the transform instead ends the record at the peak of the taper — a rectangular edge into the
  padding — and integrates the window's energy over silence: a full-scale 1 kHz sine read `-6.05
  dBFS` at 8193 samples where it reads `-3.01` at 8192, and leakage two bands away went from
  `-124` to `-45 dBFS`. `n · Σw²` is Parseval for an unnormalized transform with the window's
  energy taken out; it is what makes the level independent of the record length.
- Below about **8.7 bins** a third-octave band is narrower than one FFT bin (band width is
  `10^(1/20) − 10^(−1/20) ≈ 0.2308` of its centre), so the bottom of the grid is attributed to the
  nearest band rather than measured: at 48 kHz with an 8192-sample record (bin 5.86 Hz, limit
  51 Hz) a 26.5 Hz tone splits between the 25 Hz and 31.5 Hz bands, each reading about 3 dB under
  the tone. The report prints its bin spacing and that limit whenever it is above the grid's
  lowest centre, which is the honest scope note; with the full 262144-sample window (bin 0.18 Hz)
  the line is not printed and the tone lands in its own band.
- Every FFT bin is assigned to the band whose centre is nearest in log frequency. Testing each
  bin against a band's edges instead rounds one edge down and the other up, so a bin on a
  boundary is counted in two bands: measured on a 60 Hz tone, the neighbouring band read 13 dB
  high.
- Channels are analysed separately and combined by **maximum**, never averaged: a narrow spike
  in one ear is the demanding one.
- A `NaN` in the input reads as *no content* in that band, never as a demand.
- "After the filter" is the level the convolver **delivers**: content + curve + the FIR's own
  preamp (`headroom_db − max_boost_db`, the number `synthesize_minimum_phase_ir` reports). Leaving
  the preamp out read a `-3.01 dBFS` tone through the shipped `dt770_80` curve as `-4.20 dBFS`
  where the filter actually delivers `-9.80 dBFS`, and the ceiling test then added the headroom a
  second time. So the ceiling comparison is against `--tp-ceiling` alone, and the budget comparison
  is against the curve alone — the demand is what the filter asks for, the ceiling is what arrives.
- The ceiling line names the limiter only when one is running: under `--no-tp-limiter` it says the
  bands will be written as they are.

**Regularization (`--regularize`, `src/fit.rs`).** Also unreferenced by the oracle, and off by
default for the same reason the assessment section is: the default report and curve are compared
byte for byte. What it does is arithmetic, so the rules are numbers:

- The load is added to `|S|²`, so it is in **absolute power units referred to unity**, never
  scaled by `max|S|`. One knee value therefore means the same thing on a headphone with a 2 dB
  null and one with a 20 dB null.
- The closed form is `G = (S*D + βA) / (|S|² + β)` with `A = 1`, which penalises the deviation
  from an all-pass reference. `β → 0` reproduces `target − raw` exactly; `β → ∞` gives 0 dB, a
  bypass. The better-known published form (`S*D / (S*S + β|ε|²)`, Norcross et al. 2006 — what
  AKtools' `AKffdInverseFilter.m` and `pyfar`'s `RegularizedSpectrumInversion` both ship) penalises
  the filter's magnitude instead and sends the same band to silence — at `β = 10¹²` and a 20 dB
  null, `-230 dB` where `0 dB` was needed. Asserted in
  `infinite_load_bypasses_and_zero_load_is_todays_curve` so the two cannot be swapped quietly.
  **Substituting the unity reference for zero is HyprEQ's, not the literature's**: no source
  reviewed penalises deviation from an all-pass, and the sentence must not be written as if one
  did.
- `S` is the measurement **referred to its own level** (`fit::reference_db`, the median of the
  curve). `β` is weighed against `|S|²`, so an absolute offset `C` on the curve is arithmetically
  the same as dividing `β` by `10^(C/10)` and moves the knee by `C/κ`: a capture saved in dB SPL
  (+94 dB, `κ = 8`) is an 11 dB knee change, and `--regularize` would have done nothing at all on
  it while doing everything on the same curve written in dBFS. The **median** and not the value at
  1 kHz because a reference read at one frequency is only a reference where nothing is happening,
  and the test fixture that caught that has a 12 dB null at 1 kHz. Pinned by
  `a_curve_saved_in_absolute_units_carries_the_same_policy` and
  `hand_tuned_knots_are_shrunk_the_same_whatever_units_the_measurement_came_in`.
- The shrink is a weighted mean of the curve's gain and a bypass in **linear amplitude**, so it is
  trapped between them: it can soften a correction, never deepen one, and it changes no sign.
  Pinned across a sweep in `the_shrink_lands_between_the_curve_and_bypass_and_never_past_either`,
  which also pins the DT 770's worst knot at `+0.18 dB` — an amplitude weighting (`|S|` where the
  form wants `|S|²`) gives `+0.30`, and dropping the `|S|²` weight gives `+0.50`.
- The window mean is the **trapezoid integral of the curve over the window**, each knot weighted by
  its own slice of the log axis (halfway to its neighbours, half a spacing at the two ends). The
  unweighted mean of the samples whose centres fell inside is that integral only on a log-uniform
  axis, and hand-placed knots are not: on the axis in
  `uneven_knot_spacing_is_weighted_by_the_axis_it_covers` the window is 69 % inside one knot's cell
  and 31 % inside its neighbour's, so the curve there is **8.21 dB** where counting two samples
  says 6.00. What weighting cannot do is reach a knot further away than the window:
  `a_window_narrower_than_a_knot_spacing_has_nothing_to_average` pins that, because a curve that
  sparse has to be resampled before it is smoothed and nothing here does that.
- The prefix sums behind the window are **compensated** (`accum::prefix_sums`, the same Neumaier
  recurrence as the energy sums). A plain `prefix[i+1] = prefix[i] + x[i]` inherits every bit the
  earlier additions dropped, so one absurd sample makes a window of ordinary ones total exactly
  zero: measured, a curve of 1 dB reads a mean of **0.00 dB** at every frequency above a 1e16
  outlier. Compensation recovers the terms; the residual error is then one ulp of the prefix
  (0.0039 there, about 0.008 dB of mean), which is the tolerance
  `a_window_far_from_an_outlier_still_averages_what_it_covers` allows.
- A non-finite value reaches **the windows that contain it and no others**, and a non-finite
  frequency leaves the log axis instead of sitting on it. Both used to be global: the poisoned
  prefix made every later window `NaN`, and a `NaN` frequency stalls both edge cursors for good,
  because `NaN < lo` and `NaN <= hi` are both false. The documented policy did not change — a
  broken curve still reads broken where the break is.
- The fixture shaping (HF taper, sub-65 Hz leak discount) is applied to the correction **before**
  the load, not after it. After the load, the leak discount subtracts itself from a band the load
  already took to 0 dB and delivers a fabricated sub-bass cut: `−6 dB at 20 Hz` on a curve whose
  deviation said "change nothing". So a fully bypassed derivation is 0 dB everywhere,
  `a_fully_bypassed_derivation_is_a_flat_wire_below_65_hz_too`, and with no load anywhere the
  regularized derivation is the plain one to the last bit,
  `an_infinite_knee_reproduces_the_plain_derivation`.
- In the CLI the notch ceiling runs **after** the regularizer, and both run. `β(f)` knows what one
  measurement supports; `safe_cap_db` knows what this model's metadata allows in the band where a
  coupler resonance lives, and neither answers the other's question.
  `the_notch_cap_still_binds_on_a_regularized_curve` pins the order: at a knee that trusts
  everything the cap is what stops a +12 dB knot, and at a knee that shrinks it to about +6 dB the
  curve still lands *at* the ceiling (0.45 dB is what cap-first produces).
- A hand-tuned curve is regularized with the same form, taking its implied target to be
  `D = S·G`, which makes it `G_reg = (|S|²G + β) / (|S|² + β)`. That is algebraically the same
  expression, and `shrinking_a_plain_inversion_is_the_same_as_inverting_the_target` asserts the
  two paths agree to 1e-9 rather than trusting the derivation.
- The knee sits at **3 dB** of deviation because the shipped measurements put it there: median
  deviation 0.06 / 0.32 / 0.34 dB on the FT1 Pro, JBL 770NC and DT 770 Pro, worst bands 3.6 to
  8.4 dB at 3.7 kHz and 8–17 kHz. Steepness is 8 (about one order of magnitude of `β` per 1.25 dB
  of deviation) because `β` has to be three orders below `|S|²` to be invisible on a trusted band
  and above it to be decisive on a wild one, and a gentler knee either shaves real corrections or
  fails to bypass the notch.
- `auto` reads the **90th percentile** of the deviation, times 1.3, floored at 3 dB. Not a median
  or a MAD: the deviation of a real curve is near zero almost everywhere with a sparse tail, so
  a central statistic reads ~0 on every curve (measured MAD: 0.00 on a flat curve *and* on one
  with a 12 dB notch), and a lone notch must not raise the knee and negotiate its way out of
  being bypassed.
- The deviation is smoothed **once** (half-octave residual, then 1/6-octave on its magnitude,
  both by the weighted integral above).
  Smoothing it a second time when looking it up at a knot frequency erases the narrow features
  it exists to find — measured: a 10 dB coupler notch's peak went from 3.5 dB to 1.8 dB and
  stopped being bypassed. `regularized_curve` interpolates the deviation, it does not re-filter
  it.

**Per-ear correction (`--stereo`, `src/stereo.rs`).** Unreferenced by the oracle, `off` by default,
and `report` byte-identical on both the report's filter and the audio. The quantities:

- The pair's **balance** is `median(left) − median(right)` and the **shape mismatch** is the captured
  difference with that balance removed. Median for the reason `fit::reference_db` already gives: a
  capture saved with one ear's analyser 94 dB off has a 94 dB *difference* and no shape difference,
  and reading it the other way turns a bookkeeping fact into a pairing defect. The balance is
  cancelled in full and the shape is cancelled band by band — a consistent offset between two ears
  is not a claim about a feature, and the knee exists to doubt features.
- The trim is applied as `left += t`, `right −= t`, so **the pair's mean is the shared curve at every
  knot** (test: 1e-9). Delivered difference after the trim is `D + 2t`, so `t = −D/2` cancels it
  and `t = 0` leaves today's answer; the regularizer moves `t` between those two.
- The knee that decides `t` comes from the **model's own captured curve**, never from the mismatch it
  is about to cancel. A pair with one enormous narrow difference would otherwise raise its own knee
  and negotiate out of being corrected, which is the failure `Policy::auto` exists to avoid. With
  `--regularize off` the trim keeps the documented 3.00 dB default knee and the report names it,
  because `off` declines to touch *the curve* — it is not a vote of confidence in a difference that
  stands on one sample.
- `--stereo-max-skew` bounds `|G_L − G_R|` per knot and takes the excess **off both sides equally**,
  so the bound cannot move the centre. Default 6.00 dB, chosen against this database rather than out
  of a book: it is slack everywhere on both stereo captures except the DT 770's 15.1 kHz band, where
  the mismatch is 7.7 dB. A negative ceiling is a usage error; `0.00` means no per-ear difference at
  all.
- **Narrowness is not claimed below the grid's resolution.** A sixth-octave window spans
  `1/6 ÷ grid_oct` capture points: 2.1 on the DT 770 (grid 0.079 oct) and 2.2 on the FT1 Pro (0.077).
  Below `RESOLVED_WINDOW_POINTS = 3` the verdict stops at "mismatch" and the report prints the
  resolution line; asking "fixture or unit" of a curve whose features are one sample wide is the
  same category of error as attributing a third-octave band to fewer than one FFT bin, which
  `--assess` already refuses to do.
- The **feature match** is Pearson's coefficient between the two ears' residuals from their *own*
  sixth-octave smoothed curves — not between the curves. Correlating the curves measures the bass
  shelf both ears share and printed **+0.97** for a fixture pair whose two features are in different
  octaves; correlating the leftovers reads what the report claims to read. The `+1 / 0 / −1` reading
  is the standards-body definition (ASA/ANSI acoustical terminology, "cross correlation; interaural
  cross correlation", annotation 2). A curve with no wiggle returns **no number**, not `0.00`, which
  under the same definition would say "independent" about two identical ears.
- `--stereo split` is refused at three doors: no `--input` (a run that writes convolvers loads one
  mono impulse response on both channels), `--bypass-eq`, and a file that is not exactly two
  channels — exit 2 for the first two (argparse's code, before any file is written) and exit 1 with
  the channel count named for the last, because only the file knows that.
- The ceiling is the last word **per ear, after the trim**, and where it bites the report says the
  centre moved: on the shipped DT 770 at the default knee it cuts 6 left and 9 right knots, so the
  claim printed is `moved in 15 knot(s)` rather than the design's promise.

### What verification is held to numerically

`--verify-with` compares a later capture against the response the written filter predicts. The
rules that decide a verdict are arithmetic, and the constants were measured rather than chosen.

- The grid is [`hypreq::assess::bands_below`], the same IEC 61260-1 third-octave grid `--assess`
  builds, clamped to the range **both** files cover. Bands are named `1 kHz`, not `997.0 Hz`: the
  same measurement must not be called by two names in two sections of one report. A band is scored
  only if the whole band is inside the compared range; the rest are counted as `uncovered`.
- The prediction is `raw(f) + |H(f)| - preamp`, where `|H|` is the realised magnitude of the impulse
  response that was actually written or convolved — not the control knots. A FIR built with the
  half-Kaiser apodizer has its own truncation ripple, and predicting from the design would book that
  ripple as a rig error. The preamp comes out because it is a headroom constant the taps carry; a
  test moves `--headroom-db` by 8 dB and requires the design-shortfall line to be byte-identical.
- That transform is padded to at least **32,768** points, not merely to 2·taps. The magnitude is
  interpolated *in dB* between bins, and a straight line between two dB samples is a lie across a
  null: a 2-tap average padded only to its own 2·taps gets four bins, one of them a null at Nyquist,
  and a query between them returns −3,001 dB instead of −8.34 dB. At 4,096–65,536 taps the bins are
  well under a hertz apart and the error is nil; the floor is what keeps a `--taps 256` run from
  being quietly wrong.
- Each band is sampled at **25** log-spaced points. That number is a measurement, not a preference:
  on a ripple with 20 periods per decade, sampling at the IEC 61260-1 limit of 5 points per octave
  reads **0.00 dB** of scatter — the samples alias and land on the same phase every time, so the
  tool reports "smooth" about a curve that is anything but. Seven points recover the truth (6.78 dB
  of scatter); 25 leaves margin for narrower wiggles than the fixture.
- Pass/fail is judged on the **worst** point in a band, never the band mean, for the same reason as
  the sample count: a band whose mean is 0.2 dB while its centre sits 6 dB off is a 6 dB error, and
  an averaging verifier under-reports. Means are still in the table (`measured_db`, `predicted_db`)
  because mean-plus-worst is what distinguishes a shifted band from a ripply one.
- The capture's level offset is the **median** of the band-mean residuals — median because a capture
  with one enormous band should not have that band's size subtracted from all the others — removed
  before any band is judged, and printed afterwards.
- `noise_limited` is scatter **of the residual** inside a band exceeding the threshold, not the
  capture's own ripple. A capture whose ripple the filter reproduces is not noisy, and calling it so
  would send the user off to fix a rig that is working.
- Two constants are wired but not claimed as standards. `DEFAULT_PASS_DB = 3.0` is a default chosen
  to sit above a normal repeat-measurement difference and well below the 10–12 dB errors the shipped
  database corrects; no published acceptance region for post-EQ headphone response could be found,
  so the report always prints the threshold it used. `DEFAULT_SNR_DB = 20.0` is the one that is
  derived: additive noise `N` dB below a signal `S` biases a level reading by up to
  `20·log10(1 + N/S)` — **0.83 dB** at 20 dB of SNR, 1.58 dB at 14, 2.39 dB at 10, 3.53 dB at 6.
  At 20 dB the noise is under a third of the threshold; by 6 dB it *is* the failure.
- A band below the capture's own median band level by `snr_db` is `too_quiet`, not passed and not
  failed. The gate is relative because a CSV carries no units — and the blind spot that creates (a
  capture that is noise floor everywhere is uniformly quiet) is printed as `Capture level` instead of
  being hidden.
- **Nothing to check is not a pass**: `Report::ok()` is false when no band was scored, and the report
  says `NOTHING TO CHECK` rather than showing a zero count.
- A `FAIL` verdict still exits 0: the run did what it was asked. Usage refusals (`--bypass-eq`,
  `--stereo split`, a non-positive or non-finite threshold) exit 2 before anything is written; a
  capture that cannot be read or interpolated exits 1, also before anything is written, so one
  mistyped path cannot leave a half-finished model behind.

### Where the port refuses what the reference accepts

Parity is the rule, and two places break it deliberately. Both are sizes, both would make
the reference produce a file or a number that cannot be used, and no parity scenario
reaches either one, so the harness stays green while they differ.

* **The 4 GiB write ceiling is two ceilings, not one.** The oracle refuses a payload when
  `36 + data_size` exceeds `u32::MAX`. That measures a classic 44-byte header, so under a
  68-byte `WAVE_FORMAT_EXTENSIBLE` header (every file above two channels) a payload 24
  bytes too large passes, and the size field the writer then stores wraps to a few dozen
  bytes. The file plays and is silently truncated at the wrap. `src/wav.rs` keeps the
  reference's threshold and its exact sentence for the classic header, and adds
  `riff::riff_sizes`, which checks the payload against *both* size fields the container
  carries. Pinned by `a_payload_too_wide_for_the_riff_field_is_refused_not_wrapped` and
  `the_write_is_refused_before_it_can_pass_a_size_it_cannot_hold`.
* **`--taps` far above the usable range is answered, not panicked on.** The snippet the
  report prints contains `1 << (taps * 4 - 1).bit_length()`. Python's integer is
  unbounded, so `--taps 5000000000000000000` prints an enormous number; the Rust shift at
  that width was an overflow panic. `fft::shl_bit_length` now saturates at `usize::MAX`
  so the arithmetic stays total and the caller reports the size. Pinned by
  `a_shift_wider_than_a_machine_word_saturates_instead_of_panicking`.

## 6. Adding headphone measurements

Three ways, none of which touch the DSP:

1. **Drop a CSV in `models/`** — 2 columns (`freq_hz,raw_dBr`) or 4
   (`freq_hz,left_dBr,right_dBr,avg_dBr`); the filename stem becomes the `-H` key. The
   raw curve is normalised at 1 kHz, differenced against the default target, given the
   sub-bass leak discount and tapered to `0.00 dB` at the model's `hf_bypass_hz`.
2. **Drop `models/<key>.json`** — same schema as a database entry; supply `raw_pts` (or
   `raw_csv`/`ctrl_csv`) and either explicit `ctrl_pts` or let them be derived.
3. **Add an entry under `"models"` in `measurements.json`** — order in the file is order
   on screen: `--list-models`, `--headphone all` and the default model all follow it.

Useful keys: `name`, `file_prefix`, `rig_info`, `target`, `harm_offset`, `gauss_apodize`,
`hf_bypass_hz`, `notch_range_hz`, `safe_cap_db`, `subbass_leak_discount_db`,
`autoeq_geq_str`, `plot_meta`, `plot_annotations`.

`--upgrade-json` writes the built-in `plot_meta` and callout annotations back into the
database in place, which is the easiest way to see what a model can carry.

## 7. Troubleshooting

| Symptom | Cause |
| --- | --- |
| build fails linking `fontconfig`/`freetype` | missing C dev packages for the `ttf` feature — install them or build with `--no-plot` runs |
| panic while drawing text | fontconfig cannot resolve `sans-serif`; install a font family |
| `Error: ... measurements.json` | the database is resolved against the current directory — run from the repo root or pass `--data-file` |
| `error: ... exit code 2` | usage error, same shape argparse used; `--help` lists the accepted values |
| parity scenario says "neither side wrote it" | the scenario expects the wrong filename — output files are named from `file_prefix`, not the model key |
