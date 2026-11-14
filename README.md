# HyprEQ

Rig-compensated minimum-phase FIR impulse response generator (`pcm_f64le` for `mpv` / `FFmpeg` and `pcm_f32le` for PipeWire / DAW convolvers) targeting the **Harman Over-Ear 2018** curve.

## Supported Headphones & Data Sources

1. **JBL Live 770NC (Passive / Wired Mode)** — `--headphone jbl770nc`
   - **Measurement Source:** **Filk** (Clone GRAS 43AG fixture with `KB006x` KEMAR pinna + clone IEC 60318-4 `711` occluded-ear simulator), via the [AutoEq](https://github.com/jaakkopasanen/AutoEq) database.
   - **Target:** Harman Over-Ear 2018.
2. **Beyerdynamic DT 770 Pro (80 Ω)** — `--headphone dt770_80`
   - **Measurement Source:** **Crinacle** (Genuine `GRAS 43AG-7` fixture: `RA0402` high-frequency 711 coupler + `KB500X` anthropometric pinna), via AutoEq.
   - **Target:** Harman Over-Ear 2018.

### Acoustic Scope & Limitations
A coupler measurement captures one unit on one artificial pinna/ear simulator. Pad wear, seal leakage, and individual pinna geometry shift sub-bass (`<60 Hz`) and high-frequency pinna interference notches (`>3 kHz`) by several dB from ear to ear. For example, even between the Left and Right earcups of the measured DT 770 Pro 80 Ω unit on Crinacle's GRAS 43AG-7 rig, the `~3.76 kHz` concha reflection null shifts by `135 Hz` and `6.6 dB` (`-2.16 dBr` Left vs. `-8.78 dBr` Right at `3,838 Hz`).

To avoid over-correcting rig-specific artifacts while still fixing audible driver/cup resonances:
- **Sub-bass pad/pinna leak compensation:** Discounts stiff-pinna air-leakage roll-off below `40 Hz` on the `KB006x` plate (`jbl770nc`) and asymmetric velour pad leakage (`dt770_80`).
- **Minimum-phase cup & breakup pole inversion:** Uses Hilbert-Cepstral minimum-phase reconstruction (`0.00 ms` impulse peak delay, zero pre-ringing) to invert lumped-parameter bass/cup modes (`160 Hz` / `455 Hz` on JBL; `59 Hz` / `205 Hz` on DT 770) and diaphragm breakup peaks (`6.88 kHz` / `8.48 kHz` on JBL; `5.34 kHz`, `6.64 kHz`, and `8.16 kHz` on DT 770) without letting wide `2.0-octave` treble smoothing bleed adjacent dip errors into boosting sibilance peaks.
- **Safe concha notch bridging (`--notch-mode safe`):** Caps narrow boosts inside the DT 770's `2.8–4.4 kHz` non-minimum-phase concha null to `+1.50 dB` by default (preventing L/R and head-to-head upper-midrange glare), or allows `--notch-mode original` (`+4.26 dB` fill) and `--notch-mode zero` (`0.00 dB`).
- **Uncorrected coupler-null & HF bypass zone:** Leaves `>9.1 kHz` (`jbl770nc`) and `>8.8 kHz` / `>15.8 kHz` (`dt770_80`) coupler half-wave cylinder nulls unmodified (`0.00 dB` ultrasonic wire to Nyquist, shaded on the verification plots).

---

## Generating the Impulse Responses & Verification Plots

Requires Python 3 and `numpy` (plus `matplotlib` for the 4-panel verification PNGs):

```bash
# Generate both headphones (64-bit float WAVs, 32-bit float DAW WAVs, audit CSVs, and 4-panel PNG plots):
python3 hypreq.py --headphone all --export-csv

# Generate only JBL Live 770NC or Beyerdynamic DT 770 Pro (80 Ohm):
python3 hypreq.py --headphone jbl770nc
python3 hypreq.py --headphone dt770_80 --notch-mode safe

# Lighter 16,384-tap IR (already reaches <-147 dBFS unwindowed floor) or custom sample rate (e.g. 44.1 kHz):
python3 hypreq.py --headphone dt770_80 --taps 16384 --fs 44100
```

### Generated Files
- `jbl770nc_f64_48k.wav` / `dt770_80ohm_f64_48k.wav`: 64-bit IEEE float (`pcm_f64le`) mono impulse responses for `mpv` / `FFmpeg` (`afir=precision=double`). When a single headphone is selected, it is also copied to `ir.wav`.
- `*_mastering_mono_48k.wav` / `*_mastering_stereo_48k.wav`: 32-bit IEEE float (`pcm_f32le`, `16,384` taps) for real-time PipeWire `filter-chain` or DAW convolver plugins.
- `jbl770nc_final_verification.png` / `dt770_80ohm_final_verification.png`: 4-panel before/after frequency, filter gain, minimum-phase angle, and time-domain impulse response plots.
- `*_audit_data.csv`: Auditable CSV tables (when `--export-csv` is passed) listing frequency, raw measurement, Harman target, raw error, and applied EQ.

---

## `mpv` Configuration (`FFmpeg >= 8.0` Required)

`hypreq.py` prints the exact `mpv.conf` block matching your chosen `--fs` and file path at the end of every run.

For a `48,000 Hz` setup, add the following to `~/.config/mpv/mpv.conf` (Linux/macOS) or `%APPDATA%\mpv\mpv.conf` (Windows):

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
- Generator code is licensed under the **MIT License**.
- Raw frequency response measurements courtesy of **Filk** (`JBL Live 770NC`) and **Crinacle** (`Beyerdynamic DT 770 Pro 80 Ω`) via **Jaakko Pasanen's [AutoEq](https://github.com/jaakkopasanen/AutoEq)** database (used under their respective measurement terms). Target curve derived from **Harman International** (2018 Over-Ear target).
