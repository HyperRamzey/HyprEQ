# HyprEQ

Rig-compensated minimum-phase FIR impulse response generators (`pcm_f64le` for `mpv` / `FFmpeg` and `pcm_f32le` for PipeWire / DAW convolvers) targeting the **Harman Over-Ear 2018** curve.

## Supported Headphones & Data Sources

1. **JBL Live 770NC (Passive / Wired Mode)** — `770NCLIVE.py`
   - **Measurement Source:** `Filk` (`KB006x` pinna + clone IEC 60318-4 `711` occluded-ear simulator), via the [AutoEq](https://github.com/jaakkopasanen/AutoEq) database.
   - **Target:** Harman Over-Ear 2018.
     
2. **Beyerdynamic DT 770 Pro (80 Ω)** — `DT770P(80).py`
   - **Measurement Source:** `oratory1990` (`GRAS 43AG-7` / `KB5000` anthropomorphic pinna + IEC `711` coupler), via AutoEq.
   - **Target:** Harman Over-Ear 2018.
<img width="1950" height="1725" alt="image" src="https://github.com/user-attachments/assets/ff34a555-dfef-434f-8498-a6bedfeebb8f" />

### Acoustic Scope & Limitations
A coupler measurement captures one unit on one artificial pinna/ear simulator. Pad wear, seal leakage, and individual pinna anatomy shift sub-bass (`<60 Hz`) and high-frequency pinna interference notches (`>3 kHz`) by several dB from head to head.

Rather than inverting raw coupler errors blindly—or using a wide `2.0-octave` treble smoother that bleeds adjacent dip errors into accidentally boosting narrow sibilance peaks (such as the JBL 770NC's `6.88 kHz` and `8.48 kHz` diaphragm breakup modes)—these scripts:
- Discount stiff-pinna sub-bass air-leakage artifacts below `40 Hz`.
- Invert lumped-parameter cup/driver resonances (`160 Hz` / `455 Hz` on the JBL 770NC) and diaphragm breakup peaks (`6.88 kHz` / `8.48 kHz`) using Hilbert-Cepstral minimum-phase reconstruction (`0.00 ms` impulse peak delay, zero pre-ringing).
- Bridge across non-minimum-phase pinna/concha destructive interference dips without narrow high-Q boosting.
- Bypass `>9.1 kHz` IEC 711 coupler half-wave cylinder nulls (`0.00 dB` flat wire to Nyquist).

---

## Generating `ir.wav`

Requires Python 3 and `numpy`:

```bash
python3 770NCLIVE.py
# Or for a lighter 16,384-tap IR (-152.6 dBFS unwindowed decay floor, 0.0013 dB max FR error):
python3 770NCLIVE.py --taps 16384
# Or for a native 44.1 kHz / 96 kHz system:
python3 770NCLIVE.py --fs 44100
