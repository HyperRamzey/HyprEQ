//! # hypreq
//!
//! Rig-compensated **minimum-phase FIR impulse-response generator** for headphone
//! equalisation from coupler measurements, plus the file-convolution and
//! true-peak-limiting chain around it. A Rust port of `hypreq.py`; the CLI is
//! argument-for-argument compatible, and `scripts/parity.py` diffs the two
//! implementations' artefacts.
//!
//! ## What it produces
//!
//! For a headphone model (`measurements.json`, or any `.json`/`.csv` dropped in
//! `models/`):
//!
//! * a `pcm_f64le` mono IR for `mpv`/FFmpeg `afir=precision=double`;
//! * `pcm_f32le` mono and stereo 16 384-tap IRs for PipeWire `filter-chain` and DAW
//!   convolvers;
//! * an audit CSV (raw, target, raw error, applied EQ per control knot);
//! * a 4-panel PNG verification figure ([`plot`]);
//! * and, with `--input`, the file itself convolved with that IR and put through the
//!   two-stage true-peak ceiling limiter in [`limiter`].
//!
//! ## The shape of the DSP
//!
//! [`model`] turns a coupler measurement into control knots — target minus raw, with
//! the sub-bass pad-leak discount and the cosine taper into the HF bypass zone that
//! keeps a 711-coupler half-wave null from being "corrected" into a boost.
//! [`pchip`] interpolates those knots monotonically in `log10(f)`; [`ir`] folds that
//! magnitude into a minimum-phase impulse response by Hilbert-cepstral reconstruction
//! (zero pre-ringing, no impulse-peak delay), truncates it to `--taps`, and fades the
//! tail with a half-Kaiser. [`convolve`] is exact linear-phase overlap-add, [`tp`]
//! measures true-peak per ITU-R BS.1770-5, and [`limiter`] honours a ceiling against
//! that measurement.
//!
//! ## The numerical contract
//!
//! **Every sample and every coefficient in the DSP path is `f64`.** The IR is
//! synthesised, windowed, convolved, measured and limited in double precision; only
//! two things leave `f64`, and both are format requirements rather than arithmetic:
//! the real-time IRs are *written* as `pcm_f32le` ([`wav`], with the f32-vs-f64 error
//! the report prints), and the peak *meters* in [`meter`] hold `f32` bits, which is
//! what the Python's `np.float32` casts reproduce.
//!
//! Ported numerics are IEEE-754 exact in the sense that matters: no fused
//! multiply-add, no reassociation, no fast-math, no denormal flushing. The one place
//! where bit-identity with NumPy is genuinely impossible is the FFT — a correct `f64`
//! DFT is not unique in its rounding — so the tests pin the *consequences* (`IR`
//! sample deviation, dB decay floors, dBTP) to tolerances instead of claiming
//! bit-equality. See [`fft`] for the three NumPy conventions this crate reproduces.
//!
//! ## Modules
//!
//! | module | contents |
//! |---|---|
//! | [`model`] | `measurements.json` + `models/` loader, control-curve derivation, plot metadata |
//! | [`pchip`] | Fritsch–Carlson monotone cubic in `log10(f)` |
//! | [`ir`] | Hilbert-cepstral minimum-phase synthesis, notch modes, audit CSV |
//! | [`fft`] | NumPy-convention FFT wrappers over `rustfft` |
//! | [`window`] | `I0`, Kaiser `beta`, half-Kaiser IR tail |
//! | [`wav`] | RIFF reader (PCM/float/extensible), IEEE-float and TPDF-dithered PCM writers |
//! | [`dither`] | the exact numpy PCG64 stream that the TPDF dither is drawn from |
//! | [`convolve`] | overlap-add FFT convolution against a mono FIR |
//! | [`measure`] | [`measure::SignalMeasurements`], the deterministic convolver probe |
//! | [`tp`] | ITU-R BS.1770-5 true-peak detector, 4x–32x polyphase interpolators |
//! | [`limiter`] | two-stage true-peak ceiling limiter with lookahead and linking |
//! | [`meter`] | console-scale peak meters (`f32` atomics, 0.30 s release) |
//! | [`riff`] | RIFF header writer/reader, extensible channel masks |
//! | [`error`] | the crate's single error type |
//! | [`limiter_cfg`] | CLI-facing limiter configuration over [`limiter`], default `x16` |

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

pub mod accum;
pub mod assess;
pub mod convolve;
pub mod dither;
pub mod error;
pub mod fft;
pub mod fit;
pub mod ir;
pub mod limiter;
pub mod limiter_cfg;
pub mod measure;
pub mod meter;
pub mod model;
pub mod pchip;
pub mod plot;
pub mod pyfmt;
pub mod riff;
pub mod stereo;
pub mod tp;
pub mod verify;
pub mod wav;
pub mod window;

/// Directory holding the recorded Python reference artefacts, when the operator
/// points `HYPREQ_ORACLE` at them.
///
/// The parity tests that need real NumPy output use this and skip (loudly) when it
/// is unset, so the suite stays runnable on a machine that has only the Rust tree.
#[doc(hidden)]
#[must_use]
pub fn oracle_dir() -> Option<String> {
    std::env::var("HYPREQ_ORACLE")
        .ok()
        .filter(|s| !s.is_empty())
}
