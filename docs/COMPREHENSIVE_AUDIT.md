# HyprEQ Comprehensive Audit & Bug List (All 3 Reviews - All Resolved & Verified)

This document consolidates every defect, discrepancy, performance hotspot, broken invariant, and literature error identified across all three comprehensive reviews of the HyprEQ codebase (39 files across `src/*.rs`, `tests/*.rs`, `scripts/*`, `docs/*`, `README.md`, `Cargo.toml`, `measurements.json`). All items below have been resolved, fixed in production code/docs, and verified against tests and primary sources.

---

## Section 1: Filter & Equalization Quality Bugs (Acoustic & DSP Impact)

### 1.1 `--stereo split` erases broadband L/R balance (`Verdict::Level`) and breaks per-ear gain staging
- **Locations**: `src/main.rs:1166-1179`, `src/ir.rs:196-197`, `src/stereo.rs:450-510`
- **Mechanism**: `stereo::trim` splits the measured L/R level offset (`median(L) - median(R)`) symmetrically into `left_curve` (`-level / 2`) and `right_curve` (`+level / 2`) so the delivered audio cancels interaural balance lean. However, `run_audio_wav_mode` calls `synthesize_minimum_phase_ir` independently on `left_curve` and `right_curve` with the same `args.headroom_db`. Inside `synthesize_minimum_phase_ir`, each filter independently normalizes by `preamp_db = headroom_db - max_boost_db`.
  - On pure channel imbalance (e.g., +2.0 dB L vs. R, identical shape), Left gets `max_boost = +3.10 dB` (`preamp = -4.60 dB`); Right gets `max_boost = +5.10 dB` (`preamp = -6.60 dB`). The +2.00 dB preamp difference exactly cancels the -2.00 dB trim (`max|ir_l - ir_r| = 3.85e-17`), leaving the 2 dB balance error untouched in audio output.
- **Required Fix**: Synthesize both ears under a single shared preamp:
  `preamp_db = headroom_db - max(left_max_boost, right_max_boost)`
  normalizing the hotter ear to `headroom_db` and applying that exact `preamp_db` to both ear IRs.

### 1.2 `--stereo split` + `--experimental-phase` creates interaural delay (ITD) skew and mismatched channel lengths
- **Locations**: `src/main.rs:1182-1206`, `src/convolve.rs:71-89`, `src/phase.rs:502-546`
- **Mechanism**: `run_audio_wav_mode` calls `phase_stage` independently on left and right. Inside `linearize_band_with_budget`, `smallest_shift` picks a per-filter circular start index `start` (shifting the impulse peak by `added_latency_samples`) and grows `taps_after` along 1x..4x to hit -130 dBFS / 1e-9 retained energy. On a DT 770-like pair with sub-150 Hz L/R difference, Left's peak shifts to sample 4,630 (96.46 ms) and Right's peak to sample 3,587 (74.73 ms)—a +1,043-sample (+21.73 ms) ITD skew (~30× larger than maximum human head ITD), collapsing stereo imaging. Furthermore, if one ear steps to a higher rung on the tap-growth ladder while the other stays at 1x, channels end up with different lengths.
- **Required Fix**: In `--stereo split`, compute `start` and `taps_after` jointly so both ears share identical impulse alignment (`start`) and identical `taps_after`.

### 1.3 `gauss_apodize = true` at `fs >= 96 kHz` causes -360 dB brickwall stopband ringing and impulse peak delay
- **Locations**: `src/ir.rs:8, 187-194, 203`, `src/plot.rs:1299`
- **Mechanism**: The Gaussian rolloff above 20.2 kHz (`-3.0103 * ((f - 20200) / 1650)^2`) is hardcoded in absolute Hz rather than scaled relative to Nyquist (`fs / 2`). At 96 kHz, it demands -854.5 dB (hitting -360 dB floor at 38.2 kHz); at 192 kHz, it demands -6,353 dB, turning the gentle Nyquist taper into a brickwall lowpass around 22–28 kHz that adds causal passband delay (peak tap shifts to 9 / 25), violates "0.00 ms IR Peak Delay", and bloats transient gain bound `||h||_1` by +2.5 dB.
- **Required Fix**: Scale the apodization window relative to Nyquist or transition region (`fs / 2`), or maintain correct taper characteristics at high sample rates.

### 1.4 Real-time IR hardcodes `n_fft_rt = 32768, n_taps = 16384` regardless of `fs`
- **Locations**: `src/main.rs:854-863`, `src/ir.rs:133-140`
- **Mechanism**: Real-time path uses fixed 16,384 taps and 32,768 FFT. At 192 kHz, 16,384 taps is only 85.3 ms (df = 5.86 Hz bin), exhibiting cepstral aliasing and truncated bass resolution.
- **Required Fix**: Scale real-time FFT size and tap count with sample rate (maintaining at least ~340 ms duration and >= 4x-8x cepstral oversampling).

### 1.5 Regularization (`--regularize`) & automatic curve derivation fidelity
- **Locations**: `src/fit.rs:193-205, 451-469, 479-537`, `src/model.rs:361, 421-539, 698, 722-726`, `docs/ROADMAP.md:397`, `BUGFIX.md:161`
- **Mechanism**:
  - Inverted asymmetry: Prior art (Gómez-Bolaños et al. 2016) explicitly treats peaks and notches asymmetrically to prevent sharp boost spikes while preserving cuts.
  - Linear-amplitude blend + unflattened $|S|^2$ varies threshold by 25-30 dB across frequency.
  - `deviation()` on raw SPL flags clean curves at high frequency due to one-sided truncated smoothing window.
  - Automatic curve derivation (`derive_ctrl_pts_from_raw`) lacks AutoEq level centering and fractional-octave smoothing / boost limiting.
  - Boundary/validation: `hf_taper_start_hz >= hf_bypass_hz` ordering, and NaN/inf in gain columns poisoning curves.
- **Required Fix**: Handle asymmetric peak/notch regularization, normalize $|S|^2$, ensure smooth deviation boundaries, add fractional-octave smoothing/limiting to auto-derivation, and validate gain columns.

### 1.6 `--experimental-phase` DSP & integration defects
- **Locations**: `src/phase.rs:204-215, 352-355, 399-421, 437-444`, `src/main.rs:1215`
- **Mechanism**:
  - Unclosed residue integral and boundary step at 10 Hz creating +0.888 ms group-delay spike.
  - Audio-mode `--trim-tail` discards file tail and leaves audio out of sync with FIR latency.
  - Hard discontinuity at 10.0 Hz in `weight(f)` creating pre-ringing tails.
  - Linear-bin arithmetic `mean_gd` weighting 100-200 Hz 10x over 10-20 Hz.
  - Residue solver under-relaxation on tight budgets.
- **Required Fix**: Smooth low-frequency transition, log-weighted mean delay, robust residue solver with boundary half-steps, and correct latency-aware tail trimming.

### 1.7 `--stereo` trim policy inconsistency & reporting mismatch
- **Locations**: `src/main.rs:1673-1681`, `src/stereo.rs:275-279, 454, 472, 492-509, 556`
- **Mechanism**:
  - `spread_db` dividing by $N$ instead of $\sqrt{N}$.
  - Signed $s$ violating ear-swap antisymmetry in `stereo::trim`.
  - Discrepancy between policy evaluated on average curve vs shape mismatch.
  - Clamped balance reporting when `|level_offset_db| > max_skew_db`.
- **Required Fix**: True RMS for `spread_db`, strictly antisymmetric shape regularization, consistent policy application, and reporting actual cancelled balance.

### 1.8 True-Peak Detector & Limiter
- **Locations**: `src/limiter.rs:165, 935`, `src/tp.rs`
- **Mechanism**: Stepped release window under extreme fast settings creating sub-50 Hz modulation ripple.
- **Required Fix**: Smooth release window schedule without stepped discontinuities.

---

## Section 2: Production Logic, Plot, CLI & Verification Bugs

### 2.1 `--verify-with` unreachable with `--measure-only` in audio mode
- **Location**: `src/main.rs:1224-1307`
- **Mechanism**: `run_audio_wav_mode` returns early on `args.measure_only`, skipping verification report.

### 2.2 `--stereo split` + `--experimental-phase` shared filter overhead and duplicate report
- **Location**: `src/main.rs:1119-1205`
- **Mechanism**: Synthesizes and phase-shapes unused shared `mpv` filter, printing two conflicting phase blocks.

### 2.3 `sort_and_dedup_points` admits 1-column rows and non-finite gains
- **Location**: `src/model.rs:361, 681-703, 756-760`
- **Mechanism**: Fails to validate row length >= 2 and gain finiteness, allowing NaNs into Pchip.

### 2.4 `Ears::from_raw_pts` panics on ragged `raw_pts`
- **Location**: `src/stereo.rs:150-162`
- **Mechanism**: Only checks row 0 length; panics if later rows have fewer than 4 columns.

### 2.5 `assess()` bogus line on zero-sample channels
- **Location**: `src/assess.rs:377-388, 478-485`
- **Mechanism**: Empty channel data (`&[vec![]]`) bypasses check, computing resolution caveat at 25,997 Hz.

### 2.6 Panel 3 unwrapped phase interpolation & Panel 4 impulse peak delay with `--experimental-phase`
- **Location**: `src/plot.rs:843-908, 1108-1181`
- **Mechanism**: Panel 4 only plots first 250 samples (0..5.2 ms) while impulse peak is delayed to ~120 ms; Panel 3 plots wrapped phase with ~2,400 delay wraps and interpolates across $\pm 180^\circ$. Truncates if IR grows past 32,768.

### 2.7 Stale tap counts printed after `--experimental-phase` kernel growth
- **Location**: `src/main.rs:823, 896-898, 989-990, 1140, 1197, 1304, 1500`
- **Mechanism**: Report lines print requested tap count and hardcoded strings instead of actual grown `ir.len()`.

### 2.8 `--verify-with` extrapolation and knot range checking
- **Location**: `src/main.rs:1587-1621`, `src/verify.rs:327, 401-443`
- **Mechanism**: Extrapolates onto 20..20,000 Hz before calling verifier; `measure_bands` ignores `target.covers()`; multi-column CSVs without headers average dB SPL with phase degrees.

---

## Section 3: Performance & Allocation Optimizations

### 3.1 `src/phase.rs`: Linearize Band Optimization
- Use 8x FFT grid instead of 16x (`next_power_of_two()`).
- Zero-alloc `tail_at` & prefix bounds check, eliminating up to 1.8M vector allocations.
- Reuse prefix in `best_window`, avoid redundant `passes` and `gd_before` calls.

### 3.2 `src/convolve.rs` & `src/fft.rs`: Convolver Optimization
- Compute IR FFT `H` once per convolver, not per channel.
- Avoid redundant full clones in `convolve1`.

### 3.3 `src/tp.rs` & `src/limiter.rs`: Audio Hot Loop
- Cache polyphase filter branches per ratio.
- Use circular buffer for `Stage::push` to avoid per-sample `copy_within`.
- Skip modulo and division calculations during idle limiter passages; deduplicate `required(c)`.

### 3.4 `src/measure.rs`: Signal Measurement Hotspots
- Avoid duplicate TruePeak X4 pass when ratio is already X4.
- Reuse slice vector in 1024-frame loop; eliminate 27 zeros allocation.

### 3.5 `src/verify.rs`: Verifier Pchip Reuse
- Construct `Pchip` 3 times before loop instead of 6 times per band (186 constructions).

### 3.6 `src/pchip.rs`: PCHIP Interpolation Speedup
- Cache `f_min` and `f_max` to eliminate redundant `powf` calls.
- Monotonic search cursor for sorted queries (`eval_grid`).

### 3.7 `src/assess.rs`: Third-Octave FFT Bin Pre-computation
- Precalculate integer FFT bin bounds for 31 bands instead of 131k log10 evaluations.
- Cache Hann window across channels.

### 3.8 `src/wav.rs` & `src/riff.rs`: Streaming & Buffer Sizing
- Streaming disk chunks for WAV I/O.
- Extensible header capacity 68.

### 3.9 Other Hotspots
- Avoid `wav_in.planar.clone()` duplication when unneeded.
- In-place tail trimming.
- Cache `built_in_metadata`.
- Scalar series for `half_kaiser_tail_apply`.

---

## Section 4: Broken Test Invariants & Dead Code Cleanup

### 4.1 `src/phase.rs:247`: Discriminating test for `best_window`
- Add test where `best_window` is exercised and fails if stubbed to return 0.

### 4.2 `tests/cli_report.rs`: Fix `assert_eq!(a.len(), b.len())`
- Handle 1x..4x kernel-growth ladder and peak delays in integration tests.

### 4.3 Dead code removal
- `src/phase.rs`: `fit_f`/`fit_p`, `DEFAULT_CROSSOVER_HZ`, `budget_ms()`, duplicate `energy_pct`.
- `src/assess.rs`: unread `Staging::headroom_db` and parameter.
- `src/fit.rs`: unused `regularized_gain_db` and `regularized_curve`.
- `src/limiter.rs`: unread `Limiter::rate` and method.
- `src/tp.rs`: unread `Detector::ratio`, unused `Detector::ratio()`, `TruePeak::ratio()`, `TruePeak::min_lookahead_samples()`.
- `src/model.rs`: unused `Database::resolve`.
- `src/stereo.rs`: unread `Ears::avg`.
- `src/measure.rs`: unread `sample_peak_lin`, `true_peak_lin`, `meter_peak_hold_lin`.
- `Cargo.toml`: remove unused `serde` derives on Cfg, Report, Ratio.
- `scripts/grab-sources.sh`: fix clobbered variable `n`.

---

## Section 5: Documentation, Comments & Specification/Citation Corrections

### 5.1 B&K 5128 Standard Citation
- Replace non-existent "IEC 60318-10" with ITU-T P.58, IEC 60318-7, ANSI S3.36-1985, Type 4620 / ITU-T P.57 Type 4.3 (`src/fixture.rs`).

### 5.2 GRAS KB5000 & 45CA-10 Fixture Descriptions
- Clarify KB5000 is an anthropometric pinna; distinguish RA0045/RA045 vs RA0402 (`src/fixture.rs`).

### 5.3 Olive & Clark (2025)
- Update eLibrary ID to 23068, Express Paper 394, venue Long Beach, CA (`src/fixture.rs`, `docs/ROADMAP.md`, `README.md`).

### 5.4 Liski, Mäkivirta & Välimäki (2021)
- Correct DOI to `10.1109/TASLP.2021.3087969`.
- Clarify CC BY 4.0 Open Access (not paywalled).
- Clarify 500 Hz - 4 kHz experimental band (`docs/ROADMAP.md`, `README.md`, `src/phase.rs`).

### 5.5 Masiero & Fels Citation
- Correct 130th AES Convention London to May 2011 (`docs/ROADMAP.md`).

### 5.6 IEC 61260-1 Sampling Limits
- Correct "5 points per octave" claim in `docs/SETUP.md`.

### 5.7 Fleischmann, Silzle & Plogsties Page Numbers
- Correct page numbers to pp. 301–311 in `src/verify.rs`.

### 5.8 Miscellaneous Code & Doc Fixes
- Remove `--experimental-phase-default` from `--help`.
- Remove `--experimental-phase-budget` and fix DT 770 GD span in `src/phase.rs`.
- Unstack three doc comments on `phase_stage` in `src/main.rs`.
- Fix misplaced docs in `src/phase.rs`, `src/limiter.rs`, `src/fit.rs`.
- Fix stale comments in `src/assess.rs`, `src/stereo.rs`, `src/tp.rs`.
- Update module table in `src/lib.rs` (add missing 9 modules).
- Update test counts in `docs/ROADMAP.md`.
- Fix FFmpeg requirement heading and inline paths in `README.md`.
