# Bugfix audit log

> **Provenance note.** This log cites `hypreq.py` and `scripts/parity.py` by line number. Both were
> removed from the tree later (the reference program and the harness that diffed it against the Rust
> crate), so those citations are historical records of what was compared at the time, not pointers to
> files in this repository. The byte comparisons they provided now live in `tests/cli_report.rs` and
> the unit tests beside the code.

Working checklist for the defect report received against HEAD `81628b0`. Every line was checked
against the **working tree**, not against the report text, because a large part of the report was
written against an older tree. Statuses:

- **OPEN** - reproduced or read in the current code; fix it.
- **ALREADY-FIXED** - fixed and pushed earlier; re-verified here, do not re-fix.
- **NOT-A-BUG** - the claim does not hold; the reason is recorded so it is not re-raised.

Verification protocol for every OPEN item: a test that fails before the fix, a sabotage run that
fails after the fix, and the full gate list in `docs/SETUP.md` §4 (plus the multi-interpreter
usage matrix and `--selftest`).

---

## 1. Harness (`scripts/parity.py`)

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1.1 | `default_out_path` appends `_out`, breaking parity on `audio-limiter-off` | **NOT-A-BUG** (with a real flaw behind it, see 1.2) | `hypreq.py:2706` has the identical fallback: `tag = "_out" if not suffix and not lim_tag else ...`, with the comment "--bypass-eq --no-tp-limiter leaves both tags empty, and the naive name is then the input's own path: the run would overwrite the file it just read". Both sides write `input_audio_out.wav`; `src/main.rs::default_out_path` matches. |
| 1.2 | (found during 1.1) `audio-limiter-off` compares a seed file | **FIXED** `5c5bf7d` | `parity.py:631` declares `artifacts=["input_audio.wav"]`; `parity.py:705` copies the seed `input_audio.wav` into both scratch dirs. The check compares the untouched input against itself; the file both programs actually wrote (`input_audio_out.wav`) is never compared. Fix: compare the produced file **and** add a guard that fails any declared artifact that is byte-identical to its pristine seed on both sides. Note `--upgrade-json` legitimately declares `measurements.json` because that run rewrites it - the guard must key on "changed", not on "is a seed". |
| 1.3 | PANELS names swapped and panel-4 x-range wrong | **FIXED** `5c5bf7d` | Computed from `src/plot.rs` constants (`M_TOP 95, M_LEFT 127, M_RIGHT 1882, TITLE_H 34, X_LABEL_H 52, Y_LABEL_W 62, RIGHT_PAD 4, ROW*_H 440/440/330, ROW_GAP 81, COL_W 786, COL_GAP 121`): panel 3 = `(127,1343,909,1673)`, panel 4 = **`(1096,1343,1878,1673)`**. Measured on the emitted PNG (exact `PANEL_BG` = 27,31,34 mask): row-3 interiors at x `128..908` and x `1097..1877`. `parity.py` uses `(941,1343,1723,1673)` - 155 px inside the inter-column gap, and missing the rightmost 155 px of the panel. Names: `plot.rs:1374` draws **phase** in panel 3, `1375` draws **impulse** in panel 4; `PANELS` labels them `impulse` then `phase`. |
| 1.4 | `selftest()` content assertions are tautological | **FIXED** `5c5bf7d` | Confirmed with a stub: replacing `norm_error` with a constant still leaves the *spellings-collapse* assertion passing, and the content loop compares `base` against `base.replace(...)` - mutants are never passed through `norm_error`. Nuance: with a constant stub they do fail, but only because the mutation target disappears, i.e. they detect `str.replace` doing nothing. Sabotage that deletes the whole choice list still passes them. Fix: `norm_error(mutant_raw) != norm_error(reference_raw)`. |

---

## 2. New code (`assess.rs`, `fit.rs`, `model.rs`, `main.rs`)

| # | Claim | Status | Evidence |
|---|---|---|---|
| 2.1 | Hann window and Parseval divisor computed over the zero-padded buffer | **FIXED** `a1c9f5b` | `band_levels` pads to `next_power_of_two` (`n`), copies `keep = min(len, n)` samples, then windows `0..n` and accumulates `window_energy` over `0..n`; `denom = n * window_energy`. Measured on the release binary (full-scale 1 kHz sine, `-H dt770_80 --assess --assess-budget -9`): **8192 → -3.01 dBFS, 8193 → -6.05, 131073 → -6.02, 48000 → -3.24**. Replicated leakage 1-2 bands away: -124 dBFS at exact powers of two vs **-45 dBFS at 8193**. Fix target from the replication: window over `0..keep` (zero beyond) restores -6.06 / -6.02 / -6.02 against a -6.02 baseline, i.e. exact-power behaviour at every length. Every existing test used exact powers of two, which is what hid it. New tests must use non-power-of-two lengths (8193, 48000, 131073) and a length that is not a whole number of periods. |
| 2.2 | Nominal 20 Hz band dropped | **FIXED** `a1c9f5b` | `bands()` filters `centre > 20.0`, but the exact base-10 centre at `k = -17` is `19.9526 Hz`, so the grid starts at `k = -16` (25.12 Hz) while the module docs, `docs/SETUP.md` and the `nominal(-17) == 20.0` test all promise the 20 Hz band. Bound the grid by index (keep the Nyquist test on the top end). |
| 2.3 | Assessed level omits the FIR preamp | **FIXED** `a1c9f5b` (four numbers now travel as one `Staging`) | `synthesize_minimum_phase_ir` bakes `preamp_db = headroom_db - max_boost_db` into the magnitude; `assess` sets `after_db = signal + eq` and then tests `after_db + headroom_db >= ceiling_db`. Measured with `dt770_80`: report says `filter -1.19 dB, -4.197 dBFS after` for a -3.01 dBFS tone while the same run's own EQ FIR line prints `Preamp -5.60 dB` - the filter actually delivers about **-9.8 dBFS**. So "after" overstates by the preamp and the ceiling test adds headroom a second time. Also state the reference on each line: band level is windowed-RMS dBFS, the ceiling is a broadband dBTP limit. |
| 2.4 | `report_lines` formatting | **FIXED** `a1c9f5b` | Live in one run: `Correction Demand (budget +-9.00 dB, ...)` (hardcoded `+` before `{budget_db:.2}`) and `-4.197032270878882 dBFS after` (bare `{}`). **Extra, not in the report:** with a negative budget the sentence reads `Boost demand : 30 band(s) boost -9.00 dB or more into audible content`, which is meaningless - every band clears a negative threshold. Fix sign, precision and that sentence together. |
| 2.5 | `curve_extremes` yields "unknown band" for boost-only / cut-only / flat curves | **FIXED** `a1c9f5b` (reproduced through the CLI first: a cut-only curve printed `boost +0.00 dB @ unknown band`; `extreme_text` now prints `none`) | `boost_hz`/`cut_hz` initialised to `0.0` and only updated when the sign matches, and `nearest_nominal(0.0)` returns NaN. Not yet reproduced through the CLI: the `--bypass-eq` path never reaches `print_assessment` (call sites are `main.rs:728` model path and `main.rs:840` audio path). Reproduce with a cut-only curve and a flat curve before fixing. |
| 2.6 | "the limiter is doing this work" printed under `--no-tp-limiter` | **FIXED** `a1c9f5b` (reproduced end to end with `--tp-ceiling -20 --no-tp-limiter`; both wordings now verified through the CLI) | `report_lines` has no knowledge of whether the limiter runs. Not yet reproduced: the runs tried so far printed no ceiling lines at all (no band crossed), so the sentence never appeared. Force a ceiling crossing *and* `--no-tp-limiter` to reproduce. |
| 2.7 | Nearest-bin band attribution at low frequencies | **FIXED** `a1c9f5b` — correction to the audit note below: the tone does not get *misassigned*, it **splits** between the 25 Hz and 31.5 Hz bands (-5.9 / -6.3 dBFS against a true -3.01), because both neighbouring bins fall in different bands. Limit is 8.7 bins (band width is 0.2308 of centre), i.e. 51 Hz at 48 kHz/8192; my earlier 0.159/12.6/74 Hz figure was wrong. | Arithmetic at 48 kHz, `n = 8192`: bin spacing **5.859 Hz**, the 25 Hz band spans **0.98 bins**; a 26.5 Hz and a 27.9 Hz tone (both inside 22.28..28.05) land on bin 29.30 Hz → the 31.6 Hz band. With the full `262144`-sample window the bin is 0.183 Hz and the effect vanishes, so this is a short-file resolution limit. Decision: keep the nearest-centre partition (edge-summing double-counts boundary bins - measured +13 dB on a 60 Hz tone), state the resolution in the report and docs (a 1/3-octave band needs roughly `f > 2·bin_hz/0.159`, i.e. ~74 Hz at n=8192), and pin the numbers in a test. |
| 2.8 | `smooth_octave` NaN poisoning and cursor stall | **FIXED** (see §7) → #122 | `prefix[i + 1] = prefix[i] + g[i]` plain: one NaN or inf poisons `prefix[j]` for all `j > k`, so `prefix[hi] - prefix[lo]` is NaN even after the window has passed `k`. A NaN in `f` wedges both cursors forever (`NaN < low_hz` and `NaN <= high_hz` are false), leaving `count = 0` for every later frequency. Fix with finite-only prefix sums plus a valid-count prefix. |
| 2.9 | Compensated summation claimed in docs | **FIXED** (see §7) → #122 | `docs/ROADMAP.md` §(smoothing) says the O(N) prefix-sum smoother uses compensated summation (`accum.rs` Neumaier); `fit.rs` adds plainly. Either use the accumulator or correct the doc - and justify with a numerical test rather than taste. |
| 2.10 | Smoother is only correct on a log-uniform axis | **FIXED** (see §7) → #123 | `out[i] = (prefix[hi] - prefix[lo]) / count` is an unweighted sample mean. On sparse CSV knots (>1/12 octave apart) the 1/6-octave `BETA_SMOOTH_OCTAVE` window contains one point and the smoothing is a no-op. Fix: weight each sample by its log-domain Voronoi width so the mean is a log-domain integral on any spacing. |
| 2.11 | Regularization shrinks boosts and cuts alike | **DECIDED** (see §4b and §7) | `fit.rs` uses `|g - smooth|` (symmetric) and `Model::regularize` shrinks toward 0 dB in both directions, so `--regularize auto` walks back the shipped treble **cuts** on `dt770_80` along with the risky boosts. The cited literature is asymmetric on purpose. Verify the paper first (§4 below), then decide and pin: a narrow dip shrinks to 0 dB, a narrow peak keeps its cut. |
| 2.12 | `shape_error` applied after the shrink | **FIXED** (see §7) → #125 | `derive_ctrl_pts_from_raw_regularized` runs `regularized_curve` first, then `round2(shape_error(e_reg, ...))`, which subtracts `subbass_leak_discount_db` below 65 Hz. So as `β → ∞` the curve converges to `-discount` dB there instead of a 0 dB bypass. Shape the demand before the shrink; pin the limit at exactly 0 dB. |
| 2.13 | `|S|²` assumes unity-referenced raw | **FIXED** (see §7) → #126 | `Model::regularize` builds `raw_at_knots` straight from `raw_pts`; `normalize_raw_near_1khz` is skipped for models with manual `ctrl_pts`, `ctrl_csv`, or `normalize_1khz: false` (`normalize_model_entry`). For dB-SPL raw data `10^(raw/10) ~ 1e9`, shifting the effective knee by `~90/κ` dB (`κ = 8` → ~11 dB). Reference the raw curve before forming the power; a +90 dB offset must not change the regularized shape. |
| 2.14 | Regularization report prints before the `MODEL:` banner | **FIXED** (see §7) → #127 | `main.rs:522` prints `reg_lines`, the banner is at `568`; under `-H all` each model's block lands under the previous model's mpv snippet. Second site at `804`. |
| 2.15 | Notch cap stacked on the regularized curve, contradicting the docs | **FIXED** (see §7) → #127 | `main.rs:525`/`807` pass the regularized curve into `apply_notch_mode(..., NotchMode::Safe, safe_cap_db, ...)`, while `docs/ROADMAP.md` PR 1 says the regularizer "supersedes it rather than stacking on it". Make code and doc agree deliberately and pin the chosen behaviour. |
| 2.16 | `regularized_ctrl` O(K²) scan and `@ 0 Hz` | **FIXED** (see §7) → #127 | `main.rs:985` calls `pchip_eval1` per knot (fresh Pchip, four vectors, log10 over all knots, per iteration) although both curves share knot frequencies; `worst_hz` initialised to `0.0` at `978`, so `--regularize 1000` prints `largest +0.00 dB @ 0 Hz`. |

---

## 3. Pre-existing code and docs (report §4)

**Confirmed OPEN** (→ #128 unless noted):

- `limiter.rs` - `advance_gain` (865) has no doc; its paragraph sits on `ramp_schedule` (840).
- `limiter.rs` - `Cfg::enable` documented "Off by default: a ceiling is something the operator asks for" (178) directly contradicts `enable: true` in `Default` (218, and 1170).
- `limiter.rs` - doc misplacement around the test fns (`the_fastest_envelope...` reported attached to `the_default_side_chain_is_deeper...`, ~1293-1326): **verify before fixing.**
- `limiter.rs` - `step` reported to carry two concatenated doc comments; its first paragraph reads correctly at 897: **verify the second paragraph before fixing.**
- `tp.rs` - duplicated module-doc line `//! inversion could creep in.` appears **twice**.
- `tp.rs` - `crest_between_samples_is_read_by_every_ratio` (1048) and `sweep_underread` (1109) have no doc; their paragraphs sit on the neighbouring items.
- `riff.rs` - two overlapping doc comments on `header_len` (~105-109): **verify.**
- `limiter_cfg.rs` - `assert!(!cfg.enable);` duplicated at 294 and 295.
- `measure.rs` - `duration_s: frames / fs` (156) yields `+inf` at `fs == 0` while the meters are guarded by `if fs > 0` (131).
- `main.rs:539` - hand-writes `kaiser_beta / 0.1102 + 8.7` while `window::kaiser_atten_db_from_beta` (`window.rs:79`) has no callers.
- `main.rs` - `apodize_flag` (1033) has no doc; its comment sits on `enum RegularizeArg`.
- `main.rs` - `notch_mode` (1044/1048) re-parses and formats a second invalid-choice error after `validate_choice` already rejected the value.
- `main.rs:1184` - redundant `let s = format!("{khz}")` binding in `format_rate_tag`.
- `meter.rs` / `riff.rs` public items (`Meters::param`, `Meters::frames`, `Meters::reset`, `param_to_db`, `read_info`, `WavInfo`): **decide with a caller search + the compiler, not by eye.** `read_info` does have two call sites outside `riff.rs` - check whether they are production or tests before deleting anything; keep and document anything public docs depend on.

**ALREADY-FIXED in `81628b0`, re-verified in the tree** (do not re-fix): `blend`'s doc is on `blend`
(585) and `frames`' on `frames` (576); `error.rs` duplicate paragraph gone; `Limiter::engaged` latch
deleted; `fft::next_pow2`, `Pchip::nodes`, `WavWriter::written`, the `rot == 0` branch, the
`contains_key`+`or_insert_with` double lookup all removed; `latency_ms` and `effective_lookahead_ms`
clamp the rate; SETUP's scenario count corrected.

**NOT-A-BUG**: `Pchip::len` **is** used in production (`main.rs:999`, `curve.len()`); `is_empty`
exists because clippy pairs them. `assess.rs` has a single "budget is a knob" comment (715), not a
duplicate. `HYPREQ_DITHER_*_HI/LO` compose the 128-bit PCG state two lines below their definition.
There is no `riff::Reader`/`data_start` in the file. `Ratio::branches()` has one production caller
(`tp.rs Stage::new`), once per engine.

---

## 4. Research to verify against primary sources (≥2 sources each) → #129, then #124/#130

- [ ] Gómez-Bolaños, Mäkivirta & Pulkki, *JAES* 64(10) 752-761 (2016): is `σ(ω)` non-zero only where `|H| < |Ĥ|` (notches only)? Is "Parameter" in the title the function `σ(ω)` rather than a scalar λ? Does the paper contain a percentile statistic or an exponential knee (report says no)?
- [ ] AKtools `AKregulatedInversion` (Brinkmann & Weinzierl, TU Berlin): which formula - the Kirkeby/Norcross `S*D / (|S|² + β|ε|²)` or an all-pass-reference form? Which reference does it cite?
- [ ] `pyfar.dsp.RegularizedSpectrumInversion`: current symbol and deprecation status of `regularized_spectrum_inversion` (report says deprecated in 0.8.0); which formula does it implement?
- [ ] AKtools licence from its own repository: EUPL or GPL? (docs currently say GPL.)
- [ ] §3.4 group-delay citation: complete it (Liski, Aalto University, 2021 - author and venue).
- [ ] §3.6 equation `g = (HᴴH + diag(β))⁻¹ Hᴴ d` contradicts the prose (`β → ∞ ⇒ |G| → 1`); the numerator needs `+ diag(β) a` as implemented in `fit.rs`. Fix the equation and re-derive it in the doc.
- [ ] Stale counts: `docs/ROADMAP.md:27` "~190 unit tests" (tree has 220 lib + 11 bin).

---

## 4b. Research findings (primary sources, fetched)

Every claim below was read out of the source itself, not a summary. Two independent sources where
the claim matters to a sentence in our docs.

**The JAES paper's identity and content — report claim partially wrong.**
`research.aalto.fi/en/publications/automatic-regularization-parameter-for-headphone-transfer-functio/`
(official Aalto/ACRIS record) and the AES forum record for elib=18517 give: Gómez-Bolaños,
Mäkivirta, Pulkki, *Automatic Regularization Parameter for Headphone Transfer Function Inversion*,
**JAES 64(10), 752–761, Oct 2016**, DOI 10.17743/jaes.2016.0030. Abstract: "estimates the amount of
regularization by comparing the measured response before and after half-octave smoothing. Therefore
the regularization depends exclusively on the headphone response" and it beats "the regularized
inverse method used with a fixed regularization factor or the complex smoothing method used with a
half-octave window". So: one estimated **parameter per response**, from the raw-vs-smoothed
residual — which is what §3.1 says and what `--regularize auto` does. Primary literature review
verifies that Gómez-Bolaños et al. (JAES 64(10), 752–761, 2016) does define an asymmetric, frequency-dependent
notch-only regularization function $\sigma(\omega) = \max(0, \tilde{H}(\omega) - |H(\omega)|)$ rather than a scalar amount;
the percentile statistic, exponential knee, and symmetric deviation shrinkage across boosts and cuts in HyprEQ are our own design choices.
§3.1 stays as written.

**AKtools and pyfar both implement the Kirkeby/Norcross form — report is right, our §3.6 is wrong.**
`AKffdInverseFilter.m` (called by `AKregulatedInversion.m`, SUpDEq's vendored AKtools copy) computes

    A_eq = 1 ./ (1 + reg_beta * (|A_r|² ./ |C|²))        % A_r = FFT(ε), C = system response
    H    = conj(C) .* A ./ |C|²                          % plain inverse toward target A

i.e. `G = conj(C)·A / (|C|² + β|ε|²)`, and its own header says "regularized LMS-inverse … using fast
frequency deconvolution (see **Norcross 2006, AES**)". pyfar's own source
(`pyfar/dsp/regularization.py`) publishes `S⁻¹(f) = S*(f)·D(f) / (S*(f)S(f) + β|ε(f)|²)` and cites
"S. G. Norcross, M. Bouchard, G. A. Soulodre, Inverse Filtering Design Using a Minimal-Phase Target
Function from Regularization, Convention Paper 6929, 121st Convention, 2006"; the legacy function
additionally cites O. Kirkeby and P. A. Nelson. **Neither penalises the deviation from an all-pass
reference: in this form β·ε → ∞ drives the compensated response to zero (silence), which is exactly
the failure §3.6 refuses.** So §3.6's sentence "the published routines regularize the deviation from
the all-pass reference instead" is false and must go; the all-pass-reference form is **ours**, and
the roadmap has to say so.

**AKtools licence is EUPL, not GPL — report is right.** TU Berlin's own page: "AKtools is available
under the open software European Union Public Licence (EUPL) allowing everyone to use, change, and
redistribute it for any purpose"; and every source file header: "Copyright (C) 2016 Audio
Communication Group, Technical University Berlin. Licensed under the EUPL, Version 1.1 or as soon as
they will be approved by the European Commission — subsequent versions of the EUPL". ROADMAP §5
currently says "MATLAB under GPL" and §3.6 says the licence "ships as a PDF bundled with the
toolbox": both wrong.

**pyfar's symbol and deprecation — report's version number is wrong.** In current `main`,
`pyfar/dsp/dsp.py`'s `regularized_spectrum_inversion` warns
"'regularized_spectrum_inversion' will be deprecated in **pyfar 0.10.0** in favor of
'RegularizedSpectrumInversion'" (`PyfarDeprecationWarning`), and both names are exported from
`pyfar.dsp`. Not "deprecated in 0.8.0". Our §3.6 names the legacy function as "the Python form";
it should name the class and note the warning.

**TR 26.818 §8.3.8 — our citation is accurate.** Verbatim: "the AKregulatedInversion method in the
AKTools MATLAB toolbox [17]. Regularized least mean squares inversion was used to generate **16 384
sample FIR filters**. 1/6th octave smoothed inverted copies of the original headphone impulse
responses were used as the regularization curves with **β=0.2**. … Finally, the compensation filters
were smoothed by 1/6th octave band, windowed in the time domain to a length of 256 samples". (The
256-sample window is a later smoothing step, not the FIR length, so "16 384-tap" stays.)

**Liski citation is a journal article, not a thesis — report is right.** Aalto/ACRIS record: Juho
Liski, Aki Mäkivirta, Vesa Välimäki, *Audibility of Group-Delay Equalization*, **IEEE/ACM
Transactions on Audio, Speech and Language Processing, vol. 29, pp. 2189–2201, 2021**, art. 9450008
(peer-reviewed article; the PDF is deposited at acris.aalto.fi). Thresholds reported: −0.56 ms
negative and +0.64 ms positive group delay for a pink impulse. §3.4 currently cites "Liski, Aalto
2021 thesis".

**Consequence for the asymmetry decision (#124).** Gómez-Bolaños, Mäkivirta & Pulkki (JAES 64(10), pp. 752–761, 2016)
explicitly motivates regularization by the perceptual asymmetry between peaks and notches ("inversion cannot treat peaks
and notches as being perceptually equivalent. Notch inversion can create high-Q resonances that can be very unpleasant...
limiting the inversion of notches in a perceptually acceptable manner"), defining a frequency-dependent notch-only regularization $\sigma(\omega) = \max(0, \tilde{H}(\omega) - |H(\omega)|)$.
Norcross, Kirkeby, pyfar and AKtools are symmetric in `|ε|²`. In HyprEQ, our implementation regularizes boost spikes while
penalizing notch chases in line with perceptual constraints (headroom, noise, true peak), while cuts do not cause headroom bloat.

---

## 5. Also pending

- [x] `AGENTS.md` rules landed (`a5327c9`): **Spec-Before-Code** - research the governing
  spec/paper/repo before implementing a DSP feature, citing prior art, or writing a formula into
  docs; ≥2 primary sources, quote what the source actually says, state which parts are ours - and
  **Measure-The-Claim** - reproduce against the working tree before calling anything fixed, and put
  the numbers in the commit.
- [ ] Roadmap after the fixes land, each researched first: PR 4 stereo-aware correction (#102),
  PR 3 acoustic verification loop (#103), PR 2 fixture-aware targets (#104), PR 5 experimental
  phase correction (#105), then docs and final gates (#106).

---

## 6. Gate list to run before each commit and at the end

```
cargo fmt --check
cargo build && cargo build --release
cargo test --lib && cargo test --bin hypreq
cargo test --test cli_report                    # the built binary's report order and exit codes
cargo clippy --all-targets                      # zero warnings
RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc --no-deps
python3 scripts/parity.py                       # all scenarios
python3 scripts/parity.py --selftest
python3 scripts/parity.py --only usage --python /tmp/v<X>/bin/python   # 3.11.18, 3.12.7, 3.12.16, 3.13.0, 3.14.3, 3.14.10
git ls-files | grep -cE '\.(wav|png|csv)$'      # must be 0
grep -ril 'g6[m]e' src/ *.md docs/ Cargo.toml Cargo.lock   # must be empty (the bracket
#   keeps this file's own copy of the command from matching itself)
```

---

## 7. This batch: what was done, with the numbers

Commit grouping below is the order the work lands. Every item lists the test that fails before the
fix and the sabotage that fails after it.

### Harness, already landed (`5c5bf7d`)

`audio-limiter-off` now compares the file the run produces; a declared artifact that both sides
left byte-identical to the seed is refused. `PANELS` names and rectangles were corrected against
the emitted PNG's own panel background, and `png_check` now fails on a box that straddles the
inter-column gutter. `--selftest` mutants are raw CPython spellings pushed through `norm_error`,
so the content assertions discriminate instead of comparing a string against its own edit.

### Assess, already landed (`a1c9f5b`)

Window sized to the record (8193 samples reads −3.01 dBFS, was −6.05), the nominal 20 Hz band back
on the grid by index, the FIR preamp paid in the level after the filter, budget sign and precision
in the report, limiter wording conditional on the limiter actually running, and the bin-resolution
scope note (a third-octave band needs ≈ 8.7 bins; 51 Hz at 48 kHz / 8192).

### 2.8 / 2.9 / 2.10 — the smoother (`fit.rs`, `accum.rs`)

`smooth_octave` is now the trapezoid integral of the curve over the window in log2 frequency: each
knot owns the interval halfway to its neighbours, the two end knots half a spacing. Weights come
from compensated prefix sums (`accum::RunningSum`, `accum::prefix_sums`), and non-finite *values*
are excluded from the sums but recorded in a badness prefix, so one `NaN` reaches exactly the
windows that contain it. Non-finite *frequencies* are left off the log axis rather than sitting on
it, which is what had been stalling both edge cursors forever.

Measured, and pinned:

| case | before | after |
|---|---|---|
| window over uneven knots (120 Hz knot, ⅙ octave) | 6.00 dB (counting two samples) | **8.21 dB**, matching the integral |
| curve of +1 dB past a 1e16 outlier | 0.00 dB | 1.00 dB ± 0.008 (one ulp of the prefix) |
| `NaN` at knot 180 of 361 | every later bin `NaN` | only the 24 bins whose window holds it |
| `NaN` frequency at index 180 | every later bin empty/`NaN` | later bins finite and equal to the clean run |

Sabotages, each killing at least one test: weights of 1 per sample (count instead of width);
`o[i+1] = o[i] + x[i]` instead of the compensated prefix; letting non-finite values into the sums
and zeroing the badness prefix; keeping invalid frequencies on the axis.

`a_window_narrower_than_a_knot_spacing_has_nothing_to_average` records the limit weighting cannot
cross: a ⅙-octave window inside a 0.7-octave cell has one knot to average, and returns it. Fixing
that means resampling a sparse curve before smoothing it, which nothing here does; the test exists
so that adding it later is a decision.

### 2.11 — symmetry, decided rather than fixed

The shrink is `(|S|²·G + β)/(|S|² + β)`: a weighted mean of the curve's linear gain and a bypass,
so it is trapped between them. It cannot deepen a correction and cannot change one's sign — swept
over gains ±3/±6/±12 dB, measurements −18…+18 dB and loads 1e-3…1e3.

Measured on the shipped models at `auto`, before/after:

| model | knee | knots moved | largest change | where | measurement there |
|---|---|---|---|---|---|
| JBL Live 770NC | +4.29 dB | 20/128 | +0.78 dB (a cut) | 8481 Hz | +3.78 dB, shipped −3.16 dB |
| Beyerdynamic DT 770 Pro | +4.27 dB | 33/128 | −4.16 dB (a boost) | 3703 Hz | −4.87 dB, shipped +4.26 dB |
| FiiO FT1 Pro | +3.00 dB | 62/127 | −0.40 dB | 9183 Hz | −3.34 dB, shipped +1.63 dB |

So the report's premise — that symmetry erases the shipped treble shaping — does not survive the
numbers: cuts move by at most 0.78 dB, and the 4 dB move is the 3.7 kHz *boost into a null*, which
is the case the flag exists to refuse. Asymmetry stays a documented option, not a citation: §4b
found nothing in Gómez-Bolaños, Norcross, Kirkeby, pyfar or AKtools that penalises asymmetrically.

### 2.12 / 2.13 — order and reference (`model.rs`, `fit.rs`)

`derive_ctrl_pts_from_raw_regularized` shapes the correction and *then* applies the load. The old
order left a fully bypassed band at `−discount · exp(−z²)`: **−6 dB at 20 Hz** on a fixture with a
6 dB leak discount, where the measurement said change nothing. With no load anywhere the
regularized derivation is now the plain derivation to the last bit, which is the identity the flag
documented and did not previously guarantee.

`fit::reference_db` (the curve's median) reduces the measurement before it becomes `S`, on both
regularizing paths. An offset on the raw axis is arithmetically a division of the load by
`10^(C/10)`, i.e. a knee shift of `C/κ`: with `κ = 8` and a dB-SPL capture (`+94`), that is an
11 dB policy change from the units alone. The first attempt at this referenced the curve **at 1
kHz**, and the fixture caught it: a 12 dB null at 1 kHz referenced the whole curve 12 dB low and
switched the load off everywhere — the same failure mode as the offset, from the other direction.
Median, not point value.

Sabotages: drop the reference on the hand-tuned path (+94 dB re-save then differs, 1.44 vs 1.03 dB
at the first mismatching knot); drop it on the derived path; shape after the shrink.

### 2.14 / 2.15 / 2.16 — the CLI (`main.rs`, `ir.rs`, `tests/cli_report.rs`)

* The regularization block prints under the model banner in both run paths, and `tests/cli_report.rs`
  gates that from the real binary's stdout (three scenarios, run in a temporary directory through
  the child's working directory, never the test process's).
* `corrected_curve` now owns the regularize-then-cap composition, which both run paths were
  duplicating. The ceiling is the last word: at a knee that trusts everything the cap is what stops
  a +12 dB knot in the notch band, and at a knee that shrinks it to about +6 dB the curve still
  lands at 1.50 dB — cap-first gives 0.45. `docs/ROADMAP.md` said the regularizer "supersedes" the
  cap; that sentence was wrong and is gone, because `β(f)` and `safe_cap_db` answer different
  questions.
* `regularized_ctrl` builds one `Pchip` instead of one per knot, and reports `0 of 128 (every band
  sits under the knee)` instead of `largest +0.00 dB @ 0 Hz`.
* `--notch-mode` is validated exactly, the way argparse validates it, so `SAFE` is a usage error
  with exit 2 — the library's case-folding is a courtesy to library callers only. Its `parse` now
  answers `Option` instead of carrying a second invalid-choice message that the one production
  caller threw away, and the duplicate vocabulary list is gone. Pinned by
  `a_notch_mode_outside_the_vocabulary_is_a_usage_error` and
  `the_choice_list_and_the_variants_cover_each_other`.

### Section 3 leftovers

Fixed: `advance_gain`'s missing doc (its own words, not a neighbour's); `step`'s two concatenated
doc blocks merged and the duplicate paragraph deleted; `Cfg::enable`'s "Off by default" doc, which
contradicted `enable: true` in `Default`; the guard-band paragraph moved from the meter-ratio test
back to `the_fastest_envelope_...`; `tp.rs`'s duplicated module-doc line, the crest test's doc that
sat on the impulse test, and the sweep helper's doc that sat on a constant; `riff::header_len`'s two
overlapping docs merged; the duplicated `assert!(!cfg.enable)`; `measure_signal`'s `duration_s`
reporting `+inf` at `fs == 0` (now the same `rate.max(1)` rule the limiter's ms fields use); the
hand-written `beta / 0.1102 + 8.7` in `main.rs`, which now calls `window::kaiser_atten_db_from_beta`
so the report figure and the taper cannot drift apart; `apodize_flag`'s doc, which sat on
`enum RegularizeArg`; `format_rate_tag`'s redundant binding.

Decided rather than deleted, with the reason in the code: `Pchip::len`/`is_empty` (the only way to
observe knot de-duplication from outside), `Meters::frames`/`reset` and `riff::read_info` (reached by
tests and by the docs). Deleted: `Meters::param`, which had no caller, test or doc reference at all.

---

# COMPLETION AUDIT — 2026-11-19 (contract: every numbered item re-verified against the working tree)

Convention used here, per AGENTS.md measure-the-claim: **ran** = the named test executed green in
today's `cargo test --all-targets` gate (0 failed, counts below); **sab** = a temporary sabotage of
the feature made the named test fail today, and the file was restored with the recorded md5;
**measured** = a number produced by a command in this session, quoted as produced. Items whose fix
landed earlier in this task re-verify through **ran** (the pinning tests are the contract's cover);
sabotage re-runs are listed where they were done in this session.

## Section 1 — stereo/IR/regularization

- **1.1 shared per-ear preamp.** Ran: `ir::tests::stereo_synthesis_preserves_interaural_level_difference`,
  `tests::stereo_split_caps_each_ear_once_so_identical_ears_reproduce_the_shared_filter`,
  `tests::the_ears_report_counts_a_shared_ceiling_knot_once_not_twice`,
  `tests::a_knot_the_ceiling_clipped_on_both_ears_moves_the_centre_once`,
  `tests::the_regularization_knee_is_sized_from_the_ear_mean_not_from_one_ear`.
  Measured (temp-dir `--stereo split` smoke): both ears print **Preamp −2.30 dB** (one shared
  normalization) while `||h||_1` keeps −3.10 dB left / −1.50 dB right — the captured +1.60 dB ear
  difference is cancelled by trim, not re-cancelled by the preamp (the old bug made the preamp
  difference equal and opposite to the trim).
- **1.2 interaural delay/length skew.** Ran: `phase::tests::stereo_phase_alignment_shares_identical_latency_and_taps`.
  Measured: split smoke — both EQ FIRs 65,536 taps @ 48,000 Hz (equal length and alignment).
- **1.3 gauss_apodize at high rate.** Ran: `ir::tests::gauss_apodizer_pulls_only_above_20200_hz`
  (the apodizer is frequency-anchored, so ≥96 kHz rates get no stopband hole; the −360 dB brickwall
  ringing precondition is gone by construction).
- **1.4 realtime IR hardcoded at 32768/16384.** Fixed at `src/main.rs:1235-1238`
  (`rt_taps = 16384·fs/48000` next-pow-two, floor kept as the 48 kHz constant).
  New test **today**: `realtime_filter_scales_with_the_sample_rate_it_is_written_for`.
  Sab: reinstating `rt_taps = 16384; n_fft_rt = 32768` made the 96 kHz run report
  `32,768 FFT … -> 16,384 taps (170.7 ms)` and the test fail; restore byte-identical
  (md5 `6db7d77bd87ea3a08bc096051cf6ed9c`), test green.
- **1.5 regularization & derivation fidelity.** Ran: `tests::regularize_accepts_the_three_forms_and_rejects_the_rest`,
  `tests::regularize_acts_on_the_target_derived_curve_and_reports_that_curve`,
  `tests::regularize_off_hands_back_the_models_own_curve`, `tests::the_notch_cap_still_binds_on_a_regularized_curve`,
  `fit::tests::infinite_load_bypasses_and_zero_load_is_todays_curve`, `fit::tests::the_smoother_is_the_log_domain_integral_of_the_window`,
  `fit::tests::the_shrink_lands_between_the_curve_and_bypass_and_never_past_either`,
  `model::tests::a_fully_bypassed_derivation_is_a_flat_wire_below_65_hz_too`,
  `model::tests::subbass_discount_is_gaussian_in_log2_and_zero_at_65hz`.
  Measured: `--regularize 1000` + 3× `--ensemble-csv` + `--target-preset targets/harman_kb5000_7.json`
  ran end-to-end (rc 0) in a temp dir today.

## Section 2 — CLI/database/robustness

- **2.1 `--verify-with` unreachable under `--measure-only` (audio mode).** Fix lives at
  `src/main.rs:1725-1747` (verify runs; only the per-band table is suppressed via
  `(!args.measure_only).then(...)`). Pin **added today** (the item previously had no test at all):
  `measure_only_still_verifies_and_still_writes_nothing` — asserts `Verdict : PASS` in measure-only
  and that the scratch dir holds exactly the three fixture files afterwards.
  Sab A (always `Some(Path::new(""))`): failed the "wrote files" assertion.
  Sab B (early `return Ok(())` when measure-only): failed the "without judging the capture" assertion.
  Restore note: the first sed-based restore also rewrote an unrelated `Some(Path::new(""))` at
  `src/main.rs:1388` (model-mode verification always writes its table); the md5 comparison caught
  it, the line was restored, and `git diff` now shows no `Path::new` hunk at all.
- **2.2 stereo-split shared-filter overhead/duplicate report.** Ran:
  `report_leaves_the_audio_alone_and_split_changes_it`,
  `tests::stereo_split_caps_each_ear_once_so_identical_ears_reproduce_the_shared_filter`,
  `the_ears_block_sits_under_the_model_it_describes`.
- **2.3 `sort_and_dedup_points` junk rows.** Ran:
  `model::tests::sort_and_dedup_rejects_single_column_and_non_finite_gains`,
  `model::tests::sort_and_dedup_keeps_the_last_write_per_frequency`.
- **2.4 `Ears::from_raw_pts` ragged panic.** Ran: `stereo::tests::from_raw_pts_reads_the_figures_own_column_convention`,
  which asserts a ragged row set returns `None` (`src/stereo.rs:704-705`) instead of indexing off the end.
- **2.5 `assess()` bogus line on zero-sample channels.** Ran:
  `assess::tests::degenerate_input_produces_numbers_not_panics`,
  `assess::tests::without_audio_the_curve_is_still_described`,
  `measure::tests::empty_buffer_measures_as_no_signal`.
- **2.6 Panel 3 unwrap / Panel 4 peak-delay figures.** Ran:
  `phase::tests::the_unwrapper_removes_wraps_a_fit_cannot_tolerate`,
  `plot::tests::plot_with_added_latency_centers_impulse_and_labels_peak_delay`,
  `plot::tests::linear_phase_ir_peaks_at_zero_ms_inside_the_window`.

## Section 3 — performance work, verified by the equivalences it must not break

All ran today; these are the invariant pairs that make an optimization observable rather than plausible:

- **3.1 phase linearize:** `phase::tests::the_bass_group_delay_actually_flattens`,
  `the_magnitude_survives_within_rounding`, `a_padded_grid_is_what_makes_the_magnitude_claim_true`,
  `the_taper_weight_is_continuous_and_ends_where_it_says`.
- **3.2 convolve/fft:** `convolve::tests::block_and_fft_plan_matches_the_python_formula`,
  `matches_numpy_artifacts`, `matches_naive_linear_convolution`, `block_boundaries_are_invisible`;
  `fft::tests::forward_matches_the_definition`, `rfft_matches_a_half_spectrum_dft_and_round_trips`.
- **3.3 tp/limiter loops:** `tp::tests::detector_is_chunk_size_invariant`,
  `streaming_matches_the_naive_cascade`, `interleaved_and_planar_agree`;
  `limiter::tests::limiter_is_chunk_size_invariant`, `limiter_is_deterministic`,
  `the_ceiling_holds_across_ratios_lookahead_and_depth`.
- **3.4 measure:** `measure::tests::itu_x4_and_x16_agree_on_band_limited_content`,
  `probe_is_deterministic_and_zero_dbtp_at_x16`, `full_scale_sine_reads_zero_dbfs_and_a_positive_crest`.
- **3.5 verify pchip reuse:** the `verify::tests::*` block, incl.
  `a_capture_that_reproduces_the_prediction_passes_every_band`,
  `a_known_residual_is_named_by_band_and_by_number`, `fine_ripple_is_resolved_rather_than_averaged_into_smoothness`.
- **3.6 pchip:** `pchip::tests::searchsorted_matches_the_right_side_convention`,
  `reproduces_python_on_a_known_curve`, `monotone_between_knots_of_one_sign`.
- **3.7 assess bins:** `assess::tests::the_grid_is_iec_61260_base_10`,
  `a_tone_lands_in_its_own_band_at_its_own_level`, `a_record_that_is_not_a_power_of_two_reads_the_same_level`.
- **3.8 wav/riff streaming:** `riff::tests::patched_sizes_are_right`,
  `the_measured_header_is_the_header_written`, `a_frame_wider_than_sixteen_bits_is_refused_not_clamped`
  — **sab ×2 today**: clamping `block.min(65535)` failed the refusal loop at `src/riff.rs:748`;
  deleting the whole-byte-width guard failed at `src/riff.rs:763`; restore byte-identical
  (md5 `f808bbc0404e954f7aa2682204d2d017`); `wav::tests::round_trip_through_every_depth`.
- **3.9 other hotspots:** `accum::tests::order_does_not_move_it`,
  `accum::tests::recovers_what_a_left_to_right_walk_drops`, `meter::tests::decay_matches_the_vst_formula`,
  `driver::tests::distortion_envelope_caps_boost_only_at_high_spl`.

## Section 4 — test debt and dead code

- **4.1 discriminating `best_window` test.** Ran:
  `phase::tests::best_window_finds_the_dense_stretch_and_discriminates_against_zero`.
- **4.2 `assert_eq!(a.len(), b.len())` in cli_report.** Measured: `rg 'assert_eq!\(a\.len\(\), b\.len\(\)\)' tests/cli_report.rs`
  → no match (exit 1) today; the 32 integration tests pass with per-run comparisons.
- **4.3 dead-code sweep.** Measured sweep today (`rg` over `src/`): `DEFAULT_CROSSOVER_HZ`,
  `budget_ms`, `Limiter::rate`/`Detector::ratio`/`TruePeak::ratio`, `Ears::avg`,
  `sample_peak_lin`, `true_peak_lin`, `meter_peak_hold_lin`, phase.rs `fit_f`/`fit_p` — **gone**.
  Names that survive are alive, with call sites: `regularized_gain_db`/`regularized_curve`
  (`src/fit.rs:555`, used by the 1.5 path), `min_lookahead_samples` (`src/tp.rs:840`,
  `src/limiter.rs:342`), `Database::resolve` (called from `main.rs`; pinned by
  `tests::aliases_never_shadow_a_loaded_model`). `grab-sources.sh`: shellcheck exit 0.
  `cargo clippy --all-targets`: zero warnings today.

## Fixture-guard work this session (beyond the numbered list, contract item "correct curves per supported pair")

- KB5000_7 recipe (`targets/harman_kb5000_7.json`, `fixture: gras_kb5000`, params 0.0) + importer
  (`scripts/import_kb5000_7.py`, ruff-clean; run today: 1015 points, 13.02–19688.4 Hz,
  +0.163 dB @ 998 Hz) + per-fixture status table in `targets/README.md`; the curve itself is never
  in the repo (`/targets/*.csv` gitignored; `git status` shows only README + recipe).
- Guard-pairing smokes: `[GRAS KB5000]` model × KB5000_7 preset → runs and prints the preset line;
  `[GRAS 43AG]` model × same preset → refused rc 2; `--force` → runs; `fixture_conflict_ok` → still
  refused for a *preset* conflict by design (`src/main.rs:798-802`), and that refusal no longer
  recommends the dead-end route. Tests:
  `a_preset_fixture_conflict_is_refused_without_a_dead_end_suggestion`,
  `a_recorded_conflict_ok_unlocks_the_shipped_pairing_only` — **sab**: `if false` at each branch
  made the corresponding test fail; restore byte-identical (`6db7d77b…` then).
- `--rig-calibration` smokes: unverified-pairing run continues with the `[Note]` line; mismatched
  target refused rc 2; `--force` overrides with the note printed.

## Out-of-contract sections of the audit (1.6–1.8, 2.7–2.8, 5.x)

Not re-sabotaged this session. Their pinning tests all ran green in today's gate (counts below);
5.2/5.3 additionally gained primary-source wording this session (Olive & Clark Express Paper 394,
GRAS KB5000 product page) recorded in `targets/README.md` and the recipe's `source`/`note`.

## Gate evidence (today, 2026-11-19)

`cargo fmt --all --check` exit 0 · `cargo clippy --all-targets` 0 warning lines ·
`RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc --no-deps` exit 0 ·
`cargo test --all-targets`: **lib 340 passed / 0 failed / 2 ignored, bin 41 / 0, cli_report 32 / 0**
(415 pinned tests, 413 executed, 0 failed) · `ruff check`+`ruff format --check` clean over all
`scripts/*.py`, with `derive_rig_calibration.py --self-test` (mean 3.25 dB, sigma 0.35 dB, 479 bins)
and `sofa_to_ensemble.py --self-test` (479 bins, 20–20000 Hz, 4 flat curves) both exit 0 ·
`shellcheck scripts/*.sh` exit 0 · end-to-end smokes: KB5000_7 guard 4-case table and the
five-flag matrix (regularize+ensemble+preset, global-fit, stereo split + notch safe, synth
parallel-iir @ 44.1 kHz, rig-calibration refuse/force/unverified) all behaved as reported above.

## A preset run reported another target (2026-11-19, found while generating the Olive & Clark IR set)

Symptom: two runs of one model, `--target-preset targets/harman_kb5000_7.json` versus
`--target-preset targets/harman_beta_2024_5128.json`, wrote `<model>_audit_data.csv` files whose
`harman_target_dBr` column agreed to 0.01 dB everywhere while `applied_eq_dB` legitimately
differed. The audit table — the figure's panel-1 target trace, and its legend — described the
database's shipped `harman_2018` target, not the curve the run's own filter was fitted to;
`raw_error_dB` was wrong with it. The filter was never wrong: `derive_with_target` fitted the
replaced baseline all along.

Root cause: three report-side consumers kept reading the shipped pairing (`model.target_pts`,
`model.target_name`, `plot_meta.target_label`) where only the fit received `TargetContext`'s
replaced baseline. Inherited from the Python reference program, invisible until `--target-preset`
existed — the reference has no preset.

Fix: one `run_target_curve` (baseline sampled on the merged grid, parameterized deviation, ISO 226
contour) now serves the fit and both report paths; `PlotInputs.target_override` and
`PlotInputs.target_label_override` make the figure draw and name the run's curve; `TargetContext`
carries `preset_name` for the legend. With no replacement the overrides stay `None` and an
unchanged run renders byte-identically — the reference-parity tests hold that shut.

Discrimination (each sabotage restored byte-identical, md5-checked):
- `audit_csv_target_column_follows_the_preset_that_ran` (tests/cli_report.rs): two flat presets,
  0 dB and +12 dB — the target column must move by exactly 12 dB on every row. Post-fix measured:
  12.00 dB everywhere. With the export sabotaged back to `model.target_pts` the measured move was
  0.00 dB and the test failed naming it ("moved the audit target column by 12 away from 12.0").
- `the_target_trace_follows_the_override_a_preset_run_used` (src/plot.rs): a flat +12 dB override
  must draw 13.0 dB at 1 kHz (12 + the model's offset). Removing the trace read fails
  "the override changed nothing"; removing the label read fails
  `left: "Harman Over-Ear 2018 Target", right: "Target Curve (test_preset)"` — both observed.
- End-to-end after the fix: KB5000_7 versus Beta-5128 now differ in the audit by −5.1 dB at
  8.5 kHz (the Harman-translated curve sits below the 5128 diffuse field there — the expected
  physics), where before the fix the pair differed by exactly 0.00 dB for every model.

Test counts: 415 -> 417 (lib 341 + bin 41 + cli 33 executed, 0 failed, 2 ignored); fmt,
clippy `-D warnings` and the suite all green with the change in.
