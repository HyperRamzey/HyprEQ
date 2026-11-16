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
=== Regularization (knee 3.76 dB of deviation, shrunk the shipped knots toward a bypass) ===
  Knots moved     : 36 of 128 (largest -4.07 dB @ 3703 Hz)
  Extreme shifted : +4.26 dB -> +0.19 dB at that knot
```

On the DT 770 Pro that knee lands on the 3.7 kHz and 8–17 kHz deviations — the narrow HF notches
— and leaves the bass and mid-band corrections alone. `auto` reads the knee from the curve's own
top-decile deviation with a 3 dB floor; a number sets the knee directly, in dB of deviation.

It is **not the default**, and `off` writes the curve the reference program writes, byte for byte.
The reason is the honest one: the regularizer decides which parts of a published correction are
measurement, and that judgment should be seen in a side-by-side comparison before it becomes the
thing everyone gets. `docs/ROADMAP.md` §3.6 covers why the fallback is 0 dB and not silence.

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

### Generated Files
- `jbl770nc_f64_48k.wav` / `dt770_80ohm_f64_48k.wav` / `ft1pro_f64_48k.wav`: 64-bit IEEE float (`pcm_f64le`) mono impulse responses for `mpv` / `FFmpeg` (`afir=precision=double`). When a single headphone is selected, it is also copied to `ir.wav`.
- `*_mastering_mono_48k.wav` / `*_mastering_stereo_48k.wav`: 32-bit IEEE float (`pcm_f32le`, `16,384` taps) for real-time PipeWire `filter-chain` or DAW convolver plugins.
- `jbl770nc_final_verification.png` / `dt770_80ohm_final_verification.png` / `ft1pro_final_verification.png`: 4-panel before/after frequency, filter gain, minimum-phase angle, and time-domain impulse response plots.
- `*_audit_data.csv`: Auditable CSV tables (when `--export-csv` is passed) listing frequency, raw measurement, Harman target, raw error, and applied EQ.

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

## `mpv` Configuration (`FFmpeg >= 8.0` Required)

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
