# HyprEQ

Rig-compensated minimum-phase FIR impulse response generator (`pcm_f64le` for `mpv` / `FFmpeg` and `pcm_f32le` for PipeWire / DAW convolvers) targeting the **Harman Over-Ear 2018** curve.

All target curves and headphone measurement datasets are decoupled from the DSP into an external `measurements.json` database (with automatic discovery of standalone `.json` and `.csv` model files in `models/`), allowing users to add new headphone models without touching the code.

## Implementation

The DSP is a single self-contained Rust crate — a library plus a `hypreq` binary — with no
external service, plugin host, or path dependency. `hypreq.py` is the same pipeline kept as
a **parity oracle**: `scripts/parity.py` runs both sides scenario by scenario and diffs
stdout, WAV containers and payloads, audit CSVs, and usage errors. See
[`docs/SETUP.md`](docs/SETUP.md) for prerequisites (the verification figure needs
fontconfig/freetype), the build and verification gates, the numerical rules (f64 internals,
no FMA/fast-math), and the artefact naming.

[`docs/ROADMAP.md`](docs/ROADMAP.md) is the assessment around that implementation: what the
engine is good at and where it stops, how it differs from AutoEq, REW and CamillaDSP, and
the six DSP changes worth making next — with the papers they come from.

```bash
cargo build --release
./target/release/hypreq --headphone all --export-csv
```

## Supported Headphones & Data Sources

1. **JBL Live 770NC (Passive / Wired Mode)** — `--headphone jbl770nc`
   - **Measurement Source:** **Filk** (Clone GRAS 43AG fixture with `KB006x` KEMAR pinna + clone IEC 60318-4 `711` occluded-ear simulator), via the [AutoEq](https://github.com/jaakkopasanen/AutoEq) database.
   - **Target:** Harman Over-Ear 2018.
     <img width="1950" height="1725" alt="jbl770nc_final_verification" src="https://github.com/user-attachments/assets/8077a438-9547-4bcd-937a-5b97cdca29fd" />

2. **Beyerdynamic DT 770 Pro (80 Ω)** — `--headphone dt770_80`
   - **Measurement Source:** **Crinacle** (Genuine `GRAS 43AG-7` fixture: `RA0402` high-frequency 711 coupler + `KB500X` anthropometric pinna), via AutoEq.
   - **Target:** Harman Over-Ear 2018.
     <img width="1950" height="1725" alt="dt770_80ohm_final_verification" src="https://github.com/user-attachments/assets/bb9cb3a1-7519-494b-b826-211ef1710715" />

3. **FiiO FT1 Pro (Open-Back Planar Magnetic)** — `--headphone ft1pro`
   - **Measurement Source:** **oratory1990 / Kuulokenurkka** (Genuine `GRAS 43AG` fixture with `KB50xx` anthropometric pinna + IEC 60318-4 `711` occluded-ear simulator), via AutoEq.
   - **Target:** Harman Over-Ear 2018.
<img width="1950" height="1725" alt="ft1pro_final_verification" src="https://github.com/user-attachments/assets/63e24940-7c91-4df8-b0b4-14480a36a0f6" />

### Acoustic Scope & Limitations
A coupler measurement captures one unit on one artificial pinna/ear simulator. Pad wear, seal leakage, and individual pinna geometry shift sub-bass (`<60 Hz`) and high-frequency pinna interference notches (`>3 kHz`) by several dB from ear to ear. For example, even between the Left and Right earcups of the measured DT 770 Pro 80 Ω unit on Crinacle's GRAS 43AG-7 rig, the `~3.76 kHz` concha reflection null shifts by `135 Hz` and `6.6 dB` (`-2.16 dBr` Left vs. `-8.78 dBr` Right at `3,838 Hz`), while the FiiO FT1 Pro exhibits a `4.05 dB` L/R channel split across its `~3.84–4.05 kHz` concha/driver notch (`+4.32 dBr` Left vs. `+8.37 dBr` Right at `3,839 Hz`).

To avoid over-correcting rig-specific artifacts while still fixing audible driver/cup resonances:
- **Sub-bass pad/pinna leak compensation:** Discounts stiff-pinna air-leakage roll-off below `40 Hz` on the `KB006x` plate (`jbl770nc`), asymmetric velour pad leakage (`dt770_80`), and damps the `35 Hz` controlled front-volume pad-leak resonance ($f_s$) on the `ft1pro` while moderating its sub-bass Harman shelf (`+2.70 dB` at `55 Hz`, `+2.37 dB` at `20 Hz`) to prevent `1 μm` planar diaphragm excursion limits at high SPLs.
- **Minimum-phase cup & breakup pole inversion:** Uses Hilbert-Cepstral minimum-phase reconstruction (`0.00 ms` impulse peak delay, zero pre-ringing) to invert lumped-parameter bass/cup modes (`160 Hz` / `455 Hz` on JBL; `59 Hz` / `205 Hz` on DT 770), open-back planar ear-gain recession and upper-midrange glare (`+4.07 dB` at `1.82 kHz` and `-2.64 dB` at `2.82 kHz` on FT1 Pro), and diaphragm breakup peaks (`6.88 kHz` / `8.48 kHz` on JBL; `5.34 kHz`, `6.64 kHz`, and `8.16 kHz` on DT 770; `5.75 kHz` on FT1 Pro) without letting wide `2.0-octave` treble smoothing bleed adjacent dip errors into boosting sibilance peaks.
- **Safe concha notch bridging (`--notch-mode safe`):** Caps narrow boosts inside the DT 770's `2.8–4.4 kHz` and FT1 Pro's `3.4–4.4 kHz` non-minimum-phase concha nulls to `+1.50 dB` (`safe_cap_db * tanh(g / safe_cap_db)`) by default (preventing L/R and head-to-head upper-midrange glare), or allows `--notch-mode original` (`+4.26 dB` fill on DT 770; `+2.01 dB` fill on FT1 Pro) and `--notch-mode zero` (`0.00 dB`).
- **Uncorrected coupler-null & HF bypass zone:** Leaves `>9.1 kHz` (`jbl770nc`), `>8.8 kHz` / `>15.8 kHz` (`dt770_80`), and `>9.75 kHz` (`ft1pro`) coupler half-wave cylinder nulls unmodified (`0.00 dB` ultrasonic wire to Nyquist, shaded on the verification plots).

---

## Generating the Impulse Responses & Verification Plots

Built with `cargo build --release` (see [`docs/SETUP.md`](docs/SETUP.md) for the fontconfig
requirement the verification figure has):

```bash
# Generate all headphones (64-bit float WAVs, 32-bit float DAW WAVs, audit CSVs, and 4-panel PNG plots):
./target/release/hypreq --headphone all --export-csv

# Generate only JBL Live 770NC, Beyerdynamic DT 770 Pro (80 Ohm), or FiiO FT1 Pro:
./target/release/hypreq --headphone jbl770nc
./target/release/hypreq --headphone dt770_80 --notch-mode safe
./target/release/hypreq --headphone ft1pro --notch-mode safe

# Lighter 16,384-tap IR (already reaches <-147 dBFS unwindowed floor) or custom sample rate (e.g. 44.1 kHz):
./target/release/hypreq --headphone dt770_80 --taps 16384 --fs 44100
./target/release/hypreq --headphone ft1pro --taps 16384 --fs 44100

# Specify a custom measurement database JSON or external models directory:
./target/release/hypreq --data-file measurements.json --models-dir models --headphone all

# Measure / EQ / true-peak-limit an existing recording:
./target/release/hypreq -H jbl770nc --input song.wav --out-bits 24
./target/release/hypreq --measure-wav song.wav

# Ask what the correction demands of that recording before applying it:
./target/release/hypreq -H dt770_80 --input song.wav --assess
./target/release/hypreq -H dt770_80 --assess --assess-budget 9

# Regularize the correction, so a narrow notch the measurement cannot support is not chased:
./target/release/hypreq -H dt770_80 --regularize auto
./target/release/hypreq -H dt770_80 --regularize 4.5
```

### Regularizing the Correction (`--regularize`)

The correction is `target − raw`, evaluated at every measurement point, and that formula trusts
every wiggle equally. A broad shelf is the system telling you something real; a 3 dB notch one
twelfth of an octave wide is the fixture — a coupler resonance, a pinna standing wave — and a
filter that "fixes" it puts a matching spike into the headphone and does nothing for the sound.
`--regularize` asks each band how well the measurement fits a smoothed copy of itself and
negotiates the demand down where it does not:

```text
================================================================================
MODEL: Beyerdynamic DT 770 Pro (80 Ω) | Notch: safe
================================================================================
=== Regularization (knee 4.27 dB of deviation, shrunk the shipped knots toward a bypass) ===
  Knots moved     : 33 of 128 (largest -4.16 dB @ 3703 Hz)
  Extreme shifted : +4.26 dB -> +0.10 dB at that knot
```

That knot is the case the flag exists for: the measurement sits 4.87 dB *below* unity at 3703 Hz,
the published curve asks `+4.26 dB` to fill the hole, and the load walks it back to almost
nothing. On the DT 770 Pro the knee lands on the 3.7 kHz and 8–17 kHz deviations — the narrow HF
notches — and leaves the bass and mid-band corrections alone; the largest change to a *cut* on
any shipped model at `auto` is 0.78 dB. `auto` reads the knee from the curve's own top-decile
deviation with a 3 dB floor; a number sets the knee directly, in dB of deviation.

Two things it does not do. It does not depend on how the measurement was saved: the load is
weighed against `|S|²`, so the curve is referred to its own level first, and a capture written in
dB SPL regularizes the same as the same curve in dBFS. And it does not lift the notch ceiling:
`--notch-mode safe` still applies afterwards, because `safe_cap_db` says what this model allows in
its concha-null band whatever the measurement looks like.

It is **not the default**, and `off` writes the curve the reference program writes, byte for byte.
The reason is the honest one: the regularizer decides which parts of a published correction are
measurement, and that judgment should be seen in a side-by-side comparison before it becomes the
thing everyone gets. `docs/ROADMAP.md` §3.6 covers why the fallback is 0 dB and not silence.

### What the Two Ears Disagree About (`--stereo`)

The filter is one curve, and one curve is what gets loaded on both channels — that is what a mono
impulse response *is*. Two of the three models in this database were measured per ear, and until
now their two ears were two dashed lines on the figure and nothing else: the DSP fit the average
column and left the difference standing, which is what the listener hears afterwards.

```text
./target/release/hypreq -H dt770_80 --stereo report          # look, change nothing
./target/release/hypreq --input mix.wav --stereo split       # one filter per ear
./target/release/hypreq --input mix.wav --stereo split --stereo-max-skew 3
```

```text
=== Ears (129 capture points measured per ear) ===
  Level difference  : +1.27 dB left vs right (the centre leans)
  Shape mismatch    : -7.72 dB @ 15110 Hz (band 16 kHz), spread 1.70 dB
  Reads as          : mismatch
  Resolution        : a sixth of an octave is 2.1 capture points here (grid 0.079 oct), so one tenth
                      of an octave is one sample and the reason for the difference is not knowable
                      from this curve
  Feature match     : r = +0.704 (+1 the same feature, 0 independent accidents)
  Per-ear trim      : balance -1.27 dB cancelled, filter difference up to -5.70 dB @ 3838 Hz
  Centre            : moved in 15 knot(s) — the notch ceiling was applied after the trim and cut one
                      side back
```

`report` changes no audio and no bytes — it is the same filter with the truth attached. `split`
builds one curve per ear out of the curve the shared filter would have used, applies it as `±`, and
so leaves the **centre** of the image where the model put it while cancelling the pair's own
difference: on the DT 770 that is a 1.27 dB lean plus up to 5.70 dB of per-band difference, and on
the FT1 Pro 0.05 dB and 3.30 dB.

Two things it says rather than hides. The **resolution** line is the honest scope note: these
captures are spaced 0.08 of an octave apart, so a sixth-octave window covers about two samples and
nothing in the file can tell you whether a difference is the two units or one ear sitting
differently in a coupler — the amount to cancel is decided band by band by the same knee
`--regularize` uses, and the label stops at "mismatch". And the **centre** line reports where
`--notch-mode safe` cut one ear back after the trim, because the ceiling is the last word on a
notch band and where it bites, the middle of the image moves by half of it.

`split` needs a model measured per ear and a **stereo** file: a mono file has no right ear, and a
run that writes convolvers has no channels to apply two filters to, so it refuses with exit 2
instead of quietly becoming `report`. The per-ear difference is capped at 6 dB per band by default
(`--stereo-max-skew`) — a ceiling on the damage the flag can do to an image, not a claim about
perception. `off` is the default and writes exactly what it wrote before the flag existed.

### What a Correction Demands (`--assess`)

The report has always printed the biggest boost in the curve, and a boost is a promise about
the headphone, not about the music: a `+9 dB` shelf at 40 Hz costs nothing on a podcast and
asks a small driver for excursion it does not have on a bass-heavy track. `--assess` puts the
applied correction on the same bands the content occupies — third-octave bands per
**IEC 61260-1** (reference 1 kHz, base-10 spacing, edges at `centre · 10^(±1/20)`) — and prints
what the filter moves in each:

```text
=== Correction Demand (budget +6.00 dB, headroom -1.50 dB, ceiling -3.00 dBTP) ===
  Curve extreme : boost +4.10 dB @ 200 Hz, cut -5.04 dB @ 8 kHz
  Content       : 96,000 samples/channel, 1/3-octave bands, worst channel kept
  Boost demand  : 0 band(s) boost 6.00 dB or more into audible content
  Ceiling       : no band reaches -3.00 dBTP after -1.50 dB of headroom
```

Report-only, and off by default: it changes no filter, no level and no output file, and the
default report is byte-compared against the reference program, so a line added there would be
a parity failure. What it states is a claim about measured quantities — content level in a
band, the filter's gain there, and where the sum lands relative to the limiter ceiling. It does
not model a driver, an amplifier or a distortion threshold, and no number in it should be read
as one.

`hypreq.py` accepts the same options and is kept only as the parity oracle for
`scripts/parity.py`.

### Checking the Correction Against What Actually Happened (`--verify-with`)

Everything above runs from a capture to a filter and stops. The filter gets loaded, and nobody looks
again — the loop is open. This closes it: measure the headphone *after* the correction, hand over the
CSV, and the program compares the two, band by band, on the same IEC 61260-1 third-octave grid
`--assess` uses.

```text
./target/release/hypreq -H dt770_80 --verify-with post_eq.csv
./target/release/hypreq --input mix.wav --verify-with post_eq.csv --verify-pass-db 2
```

The output below is a synthetic capture — the filter's own predicted response with **6 dB injected
at 1 kHz** — so that the numbers can be checked against the thing that was done to them:

```text
=== Verification (post-EQ capture: 400 points, 20 Hz - 20 kHz) ===
  Scored          : 29 of 31 band(s); 2 outside the compared range
  Level offset    : -0.00 dB removed before judging shape (the capture is not calibrated, so its absolute level says nothing about the filter)
  Capture level   : median band 2.1 dB, gate at -17.9 dB (its own units; a capture that is nothing but noise floor is uniformly quiet and this section cannot see that - check this number against your rig)
  Threshold       : +/-3.00 dB of residual after the offset
  Worst band      : +5.99 dB @ 1000 Hz (band 1 kHz)
  Verdict         : FAIL - 3 of 29 band(s) outside +/-3.00 dB, 2 of them noise-limited
  Noise-limited   : the residual moves up to 4.0 dB inside those bands, so the number there is scatter across the band, not a bias
  Design shortfall: +12.79 dB @ 3981 Hz (band 4 kHz) - the filter itself does not reach the target here, so a rig that reproduces it exactly still misses by this much
  Note            : predicted from the 65,536-tap 64-bit convolver written above, with its -5.60 dB preamp taken back out
  Written Verify CSV  : dt770_80ohm_verification.csv
```

What each line is owed, and where the honesty lives:

* **The prediction is the filter, not the design.** `raw + |H(f)|`, the realised magnitude of the
  impulse response that was written — including its truncation ripple and its apodizer. Verifying
  against the control knots would blame the rig for the filter's own deviation. The run's preamp is
  taken back out, because it is a headroom choice and not a spectral one; `--headroom-db` therefore
  moves no number in this section.
* **The 6 dB error comes back as 6 dB, at the band it was put in.** Pass/fail is judged on the
  *worst* point inside each band, not the band's average: averaging a third of an octave is how a
  narrow error gets reported as a small one. The mean is in the table too, and the two together are
  what separate a bias from a ripple.
* **`Level offset` is subtracted, then printed.** A capture's absolute level is a fact about the
  interface gain. The median residual is removed before any band is judged, and reported, because it
  is still worth knowing.
* **`Design shortfall` is a different question, and it survives a perfect rig.** Where regularization,
  the notch ceiling or the deliberate refusal to correct above `hf_bypass_hz` got in the way, the
  filter never asked for what the target wanted — and a verifier that only reported the residual
  would call those bands fine.
* **Quiet bands are called unmeasured, not wrong.** A band sitting `--verify-snr-db` (default 20 dB)
  below the capture's own median band level is mostly the rig's noise floor, and a noise floor
  reproduces itself beautifully: scored, it yields a stable, reproducible, meaningless failure. It is
  excluded and counted. The trade is in the report — a real null that deep is also read as
  unmeasured, because at that level the file genuinely cannot tell them apart.
* **Noise-limited says *scatter*, not *bias*.** Each band is sampled 25 times across its own width.
  A band that sits inside the threshold on average while moving more than the threshold *inside
  itself* is flagged, so "the measurement is noisy" and "the fit is wrong" arrive as different
  findings. A ripply capture the filter matched is **not** flagged — the residual is what is
  scattered, not the curve.
* **The gate is relative, and the blind spot is printed.** A CSV carries no units: one measurer writes
  dBFS off a soundcard, another writes dB SPL off a calibrated HATS. So there is no absolute floor,
  and a capture that is nothing but noise floor everywhere is uniformly quiet — invisible to relative
  gating. `Capture level` is the anchor to check against your own rig.
* **Refused, before a single file is written:** `--verify-with` with `--bypass-eq` (no filter to
  verify) and with `--stereo split` (two filters, one capture — the residual would be read as rig
  error when it is the trim). A capture path that cannot be read fails immediately, not after the
  convolvers. Both refusals exit `2`, as every usage error does.
* **It changes nothing else.** `report`-like: the same input with and without the flag writes
  byte-identical audio and byte-identical convolvers (a test compares them). The verdict is a finding
  about a measurement, so it does not become the exit code — `PASS` and `FAIL` both exit `0`, and
  `FAIL` is spelled out in the report.
* **The table is its own file** (`<prefix>_verification.csv`), one row per band: measured, predicted,
  target, residual, shape, scatter, design, and the status word. The audit CSV is byte-compared
  against the reference program, so it cannot gain a column; this can.

### Which Rig a Curve Belongs To (`fixture`)

A frequency response is not a property of a headphone. It is a property of the headphone *on a
fixture*, and a target curve was **defined** on one fixture and one only. The paper that made this a
published problem for target curves puts the failure mode plainly: the Harman around-ear target was
defined on a GRAS 45CA fixture with an RA0045 ear, and is "often misused with other fixtures, leading
to measurement and interpretation errors due to differing acoustic transfer impedances" (Olive &
Clark, *A Preferred Headphone Target Curve Defined for the B&K 5128 and GRAS 45CA-10 Test Fixtures*,
AES 157th Convention 2024, eLibrary `23068`, Express Paper 394). This database used to have no field in which to record any of it.

Three keys, all optional:

| key | where | meaning |
|---|---|---|
| `fixture` | a model | the rig the **measurement** was made on: `gras_43ag`, `gras_43ag_clone`, `gras_45ca`, `gras_kb5000`, `bk5128`, `iec60318_4` |
| `target_fixtures` | top level, beside `targets` | the rig each **target curve** was defined on, e.g. `"harman_2018": "gras_45ca"` |
| `fixture_conflict_ok` | a model | a written reason for shipping a cross-fixture pairing anyway |

A pairing whose two fixtures are different pieces of hardware is refused, and says which two it means:

```console
$ hypreq --data-file fixtures.json -H bk --no-plot
error: 'bk' was measured on B&K 5128 (fixture: bk5128), but the target 'harman_2018' was defined on
GRAS 45CA (fixture: gras_45ca). Different fixtures present different acoustic transfer impedances, so
this filter would be fitting one rig's curve onto another's; a target is not portable between them.
Choose a target recorded for bk5128, record the target's real fixture in 'target_fixtures', record
the pairing on the model with 'fixture_conflict_ok' and its reason, or pass --force to equalise
anyway. ...
```

* **`--force` is narrower than the word suggests.** It overrides this one refusal and prints the
  refusal it overrode. It does not let a malformed database load.
* **Two units of one fixture design are not the difference being guarded.** A genuine GRAS 43AG and a
  43AG clone are allowed to share a target; every other cross-fixture pairing is refused rather than
  judged, because nobody has published a translation that would justify judging it.
* **A target that records no fixture is unknown, not universal.** No refusal is made — refusing on the
  absence of a claim would punish a database for being honest about what it does not know — and no
  report claims a check happened.
* **A fixture the program does not recognise is a database error** (exit 1), listed with its choices,
  the way an unknown target name already is. "No fixture" switches the comparison off, so a typo must
  not be allowed to buy that silence.
* **The name has to say what the key says.** A model declaring `"fixture": "gras_45ca"` whose `name`
  mentions `GRAS 43AG`, or mentions two fixtures, is refused at load: two places to write one fact is
  where drift lives, and a name that mentions both rigs says which one the measurement was made on
  is a coin flip.
* **The shipped models record their rigs, and one inherited decision.** All three are AutoEq-derived
  GRAS 43AG captures — `jbl770nc` a clone, the other two genuine — and their `name` now says so, since
  the name is what `--list-models`, the report banner and the figure's title block all print:
  `Beyerdynamic DT 770 Pro (80 Ω) [GRAS 43AG]`. The Harman target they are equalised to was defined on
  a different fixture, and there is no published 43AG equivalent of it to switch to. So each model
  carries that pairing as a reason on file rather than as an accident nobody noticed. It is the same
  decision the reference program has always made; it is now one you can read, and disagree with.

### Experimental: Moving the Phase, Leaving the Curve (`--experimental-phase`)

The shipped filter is **minimum-phase**, and that stays the default. Minimum-phase is the right
default: it puts no pre-ringing in front of a transient, and it matches what loudspeaker crossovers
already do. But it is a choice, and it is not free — the group delay it implies across a boosted
bass region is real delay, just hidden inside a decay instead of spread symmetrically. This flag
lets you ask what that costs, and it prints what it moved.

```bash
hypreq -H dt770_80 --experimental-phase 200
```

```text
=== Experimental Phase (mixed: flat group delay below 200 Hz) ===
  Band            : 10 Hz - 200 Hz, tapered to unity by 400 Hz (nothing above it is touched)
  Delay carried   : -0.232 ms, the band's own mean - flattening moves delay, it does not add any
  Group delay     : up to 0.560 ms moved, worst at 139 Hz - within the published audibility thresholds (-0.56 / +0.64 ms, Liski et al. 2021)
  Budget          : 0.56 ms per frequency against a band spread of 9.6 ms - the rest was left alone rather than paid for in latency
  Kernel          : 65536 -> 65536 taps, +119.6 ms latency - the reshape makes the response longer, and it is kept rather than cut
  Magnitude       : worst deviation 1.011e-2 dB - this stage replaces phase and carries the curve through
  Energy retained : 99.9999999003%, tail floor -165.88 dBFS
  Audibility      : contested. Minimum-phase is the default; this is an experiment, not a recommendation.
```

* **The curve is the thing that must not move.** The stage reads the finished filter's spectrum,
  corrects its group delay inside the band, and carries the magnitude through — `|H|` is never
  re-derived. What is left is the cost of holding a longer response in a finite kernel: **0.010 dB**
  on the DT 770 above, against the 3 dB a verification band is judged by.
* **It costs latency, and that is the honest price, not a defect.** Flattening a band's delay makes
  delay the filter already had in the bass *explicit* instead of hidden in a decay. About 120 ms on
  the DT 770 at the default settings, printed every time. Do not use this for gaming, monitoring or
  anything a human is playing along to.
* **The band is bounded on purpose, and the report says by how much.** Group delay below 200 Hz on a
  real correction spans roughly a decade of values, and closing all of it would mean advancing
  hundreds of milliseconds of deep-bass energy, which no causal filter can do. The per-frequency
  budget (0.56 ms, the audibility scale from Liski et al., TASLP 29, 2189–2201 (2021)) is a HyprEQ
  configuration default that nobody prescribed; the block prints it next to the band's real spread so
  you can see how much was closed and how much was left alone.
* **Audibility is genuinely contested, including in the study the budget comes from.** That paper
  finds the ear tolerates *more* phase processing in the low frequencies — which is an argument for
  treating a bass phase stage as an experiment, not as an upgrade. Nothing here says this sounds
  better. A/B it with your own ears or leave it off.
* **Off means off.** The default run is byte-for-byte the shipped one; the plots, the convolvers and
  the report are untouched. When the flag is given, the stage runs on the filter that actually gets
  written (both kernels of a `--stereo split` too), and the two report numbers that describe those
  taps — causal energy and the decay floor — are recomputed from them rather than inherited.

### Generated Files
- `jbl770nc_f64_48k.wav` / `dt770_80ohm_f64_48k.wav` / `ft1pro_f64_48k.wav`: 64-bit IEEE float (`pcm_f64le`) mono impulse responses for `mpv` / `FFmpeg` (`afir=precision=double`). When a single headphone is selected, it is also copied to `ir.wav`.
- `*_mastering_mono_48k.wav` / `*_mastering_stereo_48k.wav`: 32-bit IEEE float (`pcm_f32le`, `16,384` taps) for real-time PipeWire `filter-chain` or DAW convolver plugins.
- `jbl770nc_final_verification.png` / `dt770_80ohm_final_verification.png` / `ft1pro_final_verification.png`: 4-panel before/after frequency, filter gain, minimum-phase angle, and time-domain impulse response plots.
- `*_audit_data.csv`: Auditable CSV tables (when `--export-csv` is passed) listing frequency, raw measurement, Harman target, raw error, and applied EQ.
- `*_verification.csv`: Per-band verification table (only when `--verify-with` is passed) with measured, predicted, target, residual, in-band scatter, design shortfall, and the pass/fail/uncovered status for every third-octave band.

---

## Adding More Headphone Models (Modular Architecture)

All target curves (`targets`) and headphone definitions (`models`) are stored outside the code in `measurements.json`, and the loader also scans the `models/` directory at startup. You can add new headphone models in three ways without editing any source:

### 1. Drop a Raw Measurement CSV into `models/` (Simplest)
Create a `models/` directory next to the binary's working directory and drop in any 2-column (`freq_hz,raw_dBr`) or 4-column (`freq_hz,left_dBr,right_dBr,avg_dBr`) CSV file, e.g. `models/hd600.csv`:
- HyprEQ automatically registers the filename stem as a `--headphone` key (e.g. `hypreq --headphone hd600`).
- When no manual `ctrl_pts` curve is specified, HyprEQ automatically normalizes the raw measurement near `1 kHz`, computes the inverse error against `harman_2018`, applies sub-bass leak discounting (`<65 Hz`), and cosine-tapers the filter to `0.00 dB` at `9.5 kHz` (`hf_bypass_hz`) so 711 coupler half-wave nulls are never boosted.

### 2. Drop a Standalone JSON File into `models/`
Create `models/<model_key>.json` (e.g. `models/hd600.json`). You can either supply explicit `ctrl_pts` alongside `raw_pts` (or `raw_csv` / `ctrl_csv` paths), or omit `ctrl_pts` to let HyprEQ derive the control curve automatically:

```json
{
  "key": "hd600",
  "name": "Sennheiser HD 600",
  "file_prefix": "hd600",
  "rig_info": "GRAS 43AG / KB50xx Anthropometric Pinna + IEC 711",
  "fixture": "gras_43ag",
  "target": "harman_2018",
  "harm_offset": 0.0,
  "gauss_apodize": true,
  "hf_bypass_hz": 9500.0,
  "notch_range_hz": [3000.0, 4500.0],
  "safe_cap_db": 1.50,
  "subbass_leak_discount_db": 1.0,
  "raw_pts": [
    [20.0, -3.20],
    [100.0, 0.85],
    [1000.0, 0.00],
    [3000.0, 8.90],
    [10000.0, -2.10],
    [19955.5, -18.50]
  ]
}
```

### 3. Add a Model Entry to `measurements.json`
Add a new object under `"models"` in `measurements.json` using the same schema as `jbl770nc`, `dt770_80`, and `ft1pro` (supporting optional `autoeq_geq_str` for AutoEq GraphicEQ overlay comparison, `plot_annotations`, `audit_notes`, `notch_range_hz`, and `hf_bypass_hz`).

---

## `mpv` Configuration (`FFmpeg >= 6.0` Required)

HyprEQ prints the exact `mpv.conf` block matching your chosen `--fs` and file path at the end of every run.

For a `48,000 Hz` setup, add the following to `~/.config/mpv/mpv.conf` (Linux/macOS) or `%APPDATA%\mpv\mpv.conf` (Windows) (replace `jbl770nc_f64_48k.wav` with `dt770_80ohm_f64_48k.wav`, `ft1pro_f64_48k.wav`, or `ir.wav` as needed):

```ini
# Lock mpv output rate to match the IR sample rate so mpv never inserts a second default resampler
audio-samplerate=48000

# 64-bit double Kaiser (beta=16, 153.9 dB formula) resampler -> non-uniform partitioned f64 convolution
# minp=1024:maxp=8192 keeps head block latency to ~21.3 ms for video A/V sync
af=lavfi=[aformat=sample_fmts=dblp,aresample=48000:filter_size=256:phase_shift=12:cutoff=0.98:filter_type=kaiser:kaiser_beta=16:osf=dblp[in];amovie=/etc/mpv/jbl770nc_f64_48k.wav,aformat=sample_fmts=dblp[ir];[in][ir]afir=irfmt=mono:irnorm=-1:wet=1:minp=1024:maxp=8192:precision=double]
```

### Platform & Path Notes
- **FFmpeg Version:** The `afir` options `minp`, `maxp`, and `precision=double` require **FFmpeg 6.0 or newer**.
- **Linux Paths:** `/etc/mpv/` is system-wide and requires root (`sudo`). For a rootless user config, place the `.wav` in `/home/<username>/.config/mpv/` and set `amovie=/home/<username>/.config/mpv/jbl770nc_f64_48k.wav`.
- **Windows Paths:** Inside FFmpeg's `amovie=`, use forward slashes and escape the drive colon (`amovie=C\\:/Users/<username>/AppData/Roaming/mpv/jbl770nc_f64_48k.wav`).

---

## Credits & License
- Generator code (Rust crate and `hypreq.py`) is licensed under the **MIT License**.
- Raw frequency response measurements courtesy of **Filk** (`JBL Live 770NC`), **Crinacle** (`Beyerdynamic DT 770 Pro 80 Ω`), and **oratory1990 / Kuulokenurkka** (`FiiO FT1 Pro`) via **Jaakko Pasanen's [AutoEq](https://github.com/jaakkopasanen/AutoEq)** database (used under their respective measurement terms). Target curve derived from **Harman International** (2018 Over-Ear target).
