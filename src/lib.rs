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
//! tail with a half-Kaiser. [`convolve`] is exact linear convolution by overlap-add, [`tp`]
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
//! | [`accum`] | Neumaier and pairwise compensated summation and prefix sums |
//! | [`assess`] | Content-demand third-octave spectrum assessment against headroom |
//! | [`convolve`] | Overlap-add FFT convolution against planar FIRs |
//! | [`dither`] | The exact NumPy PCG64 stream that the TPDF dither is drawn from |
//! | [`driver`] | Level-dependent distortion and sub-resonance excursion boost bounds |
//! | [`error`] | The crate's single error type |
//! | [`fft`] | NumPy-convention FFT wrappers over `rustfft` |
//! | [`fit`] | Octave-band smoothing, deviation, quantile, and regularized curve inversion |
//! | [`fixture`] | Measurement rig and target fixture compatibility detection |
//! | [`global_fit`] | Pentadiagonal LDL^T + over-relaxed ADMM box-QP global curve fit |
//! | [`ir`] | Hilbert-cepstral minimum-phase synthesis, notch modes, audit CSV |
//! | [`limiter`] | Two-stage true-peak ceiling limiter with lookahead and linking |
//! | [`limiter_cfg`] | CLI-facing limiter configuration over [`limiter`], default `x16` |
//! | [`measure`] | [`measure::SignalMeasurements`], the deterministic convolver probe |
//! | [`meter`] | Console-scale peak meters (`f32` atomics, 0.30 s release) |
//! | [`model`] | `measurements.json` + `models/` loader, control-curve derivation, plot metadata |
//! | [`pchip`] | Fritsch–Carlson monotone cubic in `log10(f)` |
//! | [`phase`] | Experimental mixed-phase band group-delay linearization |
//! | [`plot`] | Four-panel PNG verification figure renderer |
//! | [`psycho`] | ERB/roex excitation, Harman PPR, complex fractional-octave smoothing |
//! | [`pyfmt`] | Python-style format string and float printer |
//! | [`riff`] | RIFF header writer/reader, extensible channel masks |
//! | [`robust_fit`] | Ensemble MAD statistics and whole-curve bootstrap confidence intervals |
//! | [`stereo`] | Interaural L/R mismatch analysis, balance, and per-ear trim |
//! | [`synth_alt`] | Vicanek matched PEQ, Bank parallel IIR, NUPOLS partition schedule |
//! | [`target`] | Parameterized targets, Bradley–Terry preference learner, ISO 226 delta |
//! | [`tp`] | ITU-R BS.1770-5 true-peak detector, 4x–32x polyphase interpolators |
//! | [`verify`] | Third-octave prediction-vs-capture acoustic verifier |
//! | [`wav`] | RIFF reader (PCM/float/extensible), IEEE-float and TPDF-dithered PCM writers |
//! | [`window`] | `I0`, Kaiser `beta`, half-Kaiser IR tail |

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

pub mod accum;
pub mod assess;
pub mod convolve;
pub mod dither;
pub mod driver;
pub mod error;
pub mod fft;
pub mod fit;
pub mod fixture;
pub mod global_fit;
pub mod ir;
pub mod limiter;
pub mod limiter_cfg;
pub mod measure;
pub mod meter;
pub mod model;
pub mod pchip;
pub mod phase;
pub mod plot;
pub mod psycho;
pub mod pyfmt;
pub mod riff;
pub mod robust_fit;
pub mod stereo;
pub mod synth_alt;
pub mod target;
pub mod tp;
pub mod verify;
pub mod wav;
pub mod window;

/// Directory holding the recorded Python reference artefacts, when the operator
/// points `HYPREQ_ORACLE` at them.
///
/// The parity tests that need real NumPy output use this and skip (loudly) when it
/// is unset, so the suite stays runnable on a machine that has only the Rust tree.
#[cfg(test)]
#[doc(hidden)]
#[must_use]
pub fn oracle_dir() -> Option<String> {
    std::env::var("HYPREQ_ORACLE")
        .ok()
        .filter(|s| !s.is_empty())
}
