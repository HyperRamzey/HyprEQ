//! HyprEQ command line: minimum-phase convolver generation, audio convolution, and the
//! true-peak limiter/measurement suite.
//!
//! Two modes, chosen by whether `--input` (or `--measure-wav`) was given:
//!
//! * **Model mode** — synthesise the minimum-phase FIR for one or every headphone in
//!   the database and write the 64-bit offline convolver, the two 32-bit real-time
//!   convolvers, optionally the audit CSV and the 4-panel verification PNG, then probe
//!   the whole chain with a deterministic 0 dBTP multitone + transient.
//! * **Audio mode** — measure an input WAV, convolve it with the model's IR, true-peak
//!   limit it, verify the result with an independent detector, and write the output.
//!
//! Every number on stdout is printed by the same functions the tests assert on, so the
//! report a user reads is the report the test suite checks.

use std::path::{Path, PathBuf};
use std::process::exit;

use clap::Parser;

use hypreq::convolve::{convolve_planar_fft, convolve_planar_pair};
use hypreq::error::Error;
use hypreq::ir::{
    apply_notch_mode, export_audit_csv, gauss_apodize_params, l1_gain_db, mpv_n_fft,
    synthesize_minimum_phase_ir, NotchMode,
};
use hypreq::limiter::{
    Limiter, DEFAULT_ATTACK_MS, DEFAULT_CEILING_DB, DEFAULT_LINK_RELEASE, DEFAULT_LINK_TRANSIENT,
    DEFAULT_LOOKAHEAD_MS, DEFAULT_RELEASE_MS, MAX_ATTACK_MS, MAX_CEILING_DB, MAX_LOOKAHEAD_MS,
    MAX_RELEASE_MS, MIN_ATTACK_MS, MIN_CEILING_DB, MIN_LOOKAHEAD_MS, MIN_RELEASE_MS,
};
use hypreq::limiter_cfg::{print_limiter_report, TpArgs};
use hypreq::measure::{build_convolver_probe, measure_signal, print_signal_measurements};
use hypreq::model::{
    load_measurement_database, upgrade_measurements_json, Database, Model, MODEL_ALIASES,
};
use hypreq::plot::{PlotData, PlotInputs};
use hypreq::tp::TruePeak;
use hypreq::wav::{comma_sep, read_wav, write_wav_ieee_float, write_wav_pcm_or_float};

/// Exit code argparse uses for a usage error, kept so scripts that branch on it still do.
const USAGE_ERROR: i32 = 2;

/// HyprEQ: Unified Minimum-Phase Convolver Generator, x16 True-Peak Limiter &
/// Measurements Suite.
#[derive(Parser, Debug)]
#[command(name = "hypreq", version, about)]
struct Args {
    /// Headphone profile to generate/apply (default: all)
    #[arg(short = 'H', long, default_value = "all")]
    headphone: String,
    /// Sample rate in Hz (default: 48000 for IR gen, or input WAV rate)
    ///
    /// `allow_negative_numbers` keeps a value like `-1` a *value* rather than an unknown
    /// flag, so it reaches the same domain check argparse sends it to.
    #[arg(long, allow_negative_numbers = true)]
    fs: Option<i64>,
    /// Tap count for 64-bit float WAV (default: 65536; 16384 already reaches
    /// <-134 dBFS unwindowed)
    #[arg(long, default_value_t = 65536, allow_negative_numbers = true)]
    taps: i64,
    /// Steady-state convolver preamp headroom in dB (default: -1.50)
    #[arg(long, default_value_t = -1.50, allow_negative_numbers = true)]
    headroom_db: f64,
    /// Non-minimum-phase concha notch handling: 'safe' (default, caps narrow fill to
    /// +1.50 dB), 'original', or 'zero' (0.0 dB)
    #[arg(long, default_value = "safe")]
    notch_mode: String,
    /// Per-ear correction: 'off' (default, one shared filter, silent), 'report' (same filter,
    /// say what the two ears differ by), 'split' (one filter per ear, needs -i and a stereo WAV)
    #[arg(long, default_value = "off")]
    stereo: String,
    /// Largest left/right filter difference --stereo split may apply in one band, in dB
    #[arg(long, default_value_t = hypreq::stereo::DEFAULT_MAX_SKEW_DB, allow_negative_numbers = true)]
    stereo_max_skew: f64,
    /// Force Gaussian Nyquist roll-off (>20.2 kHz) for legacy brickwall-ringing sources
    #[arg(long, conflicts_with = "no_apodize")]
    apodize: bool,
    /// Force non-apodizing 0.00 dB ultrasonic wire (overrides model JSON gauss_apodize)
    #[arg(long)]
    no_apodize: bool,
    /// Export auditable CSV tables of raw, target, and EQ data
    #[arg(long)]
    export_csv: bool,
    /// Skip the 4-panel verification plot generation
    #[arg(long)]
    no_plot: bool,
    /// List all loaded headphone models from measurements.json and models/ and exit
    #[arg(long)]
    list_models: bool,
    /// Enrich measurements.json in-place with built-in plot_meta and callout annotations
    #[arg(long)]
    upgrade_json: bool,
    /// Path to external JSON measurement database (default: measurements.json)
    #[arg(long, default_value = "measurements.json")]
    data_file: PathBuf,
    /// Optional directory containing additional per-model .json or .csv files
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,

    /// True-peak oversampling ratio: 2/x2, 4/x4, 8/x8, 16/x16, 32/x32 (default: x16)
    #[arg(
        long,
        default_value = "x16",
        help_heading = "True-Peak Limiter & Measurements"
    )]
    tp_ratio: String,
    /// True-peak limiter ceiling in dBTP
    #[arg(
        long,
        default_value_t = DEFAULT_CEILING_DB,
        allow_negative_numbers = true,
        help_heading = "True-Peak Limiter & Measurements",
        help = format!(
            "True-peak limiter ceiling in dBTP, [{MIN_CEILING_DB}, {MAX_CEILING_DB}] (default: {DEFAULT_CEILING_DB})"
        )
    )]
    tp_ceiling: f64,
    /// Transient lookahead in ms
    #[arg(
        long,
        default_value_t = DEFAULT_LOOKAHEAD_MS,
        allow_negative_numbers = true,
        help_heading = "True-Peak Limiter & Measurements",
        help = format!(
            "Transient lookahead in ms, [{MIN_LOOKAHEAD_MS}, {MAX_LOOKAHEAD_MS}] (default: {DEFAULT_LOOKAHEAD_MS})"
        )
    )]
    tp_lookahead: f64,
    /// Slow-stage attack in ms
    #[arg(
        long,
        default_value_t = DEFAULT_ATTACK_MS,
        allow_negative_numbers = true,
        help_heading = "True-Peak Limiter & Measurements",
        help = format!(
            "Slow-stage attack in ms, [{MIN_ATTACK_MS}, {MAX_ATTACK_MS}] (default: {DEFAULT_ATTACK_MS})"
        )
    )]
    tp_attack: f64,
    /// Slow-stage release in ms
    #[arg(
        long,
        default_value_t = DEFAULT_RELEASE_MS,
        allow_negative_numbers = true,
        help_heading = "True-Peak Limiter & Measurements",
        help = format!(
            "Slow-stage release in ms, [{MIN_RELEASE_MS}, {MAX_RELEASE_MS}] (default: {DEFAULT_RELEASE_MS})"
        )
    )]
    tp_release: f64,
    /// Transient stage channel linking
    #[arg(
        long,
        default_value_t = DEFAULT_LINK_TRANSIENT,
        allow_negative_numbers = true,
        help_heading = "True-Peak Limiter & Measurements",
        help = format!(
            "Transient stage channel linking [0..1] (default: {DEFAULT_LINK_TRANSIENT})"
        )
    )]
    tp_link_transient: f64,
    /// Slow stage channel linking
    #[arg(
        long,
        default_value_t = DEFAULT_LINK_RELEASE,
        allow_negative_numbers = true,
        help_heading = "True-Peak Limiter & Measurements",
        help = format!("Slow stage channel linking [0..1] (default: {DEFAULT_LINK_RELEASE})")
    )]
    tp_link_release: f64,
    /// Disable the true-peak ceiling limiter (measurements still run)
    #[arg(long, help_heading = "True-Peak Limiter & Measurements")]
    no_tp_limiter: bool,
    /// Input WAV file to measure, convolve with headphone EQ, and true-peak limit
    #[arg(long, help_heading = "True-Peak Limiter & Measurements")]
    input: Option<String>,
    /// Output WAV path for --input mode (default: <input>_<model>_tp_<ratio>.wav)
    #[arg(long, help_heading = "True-Peak Limiter & Measurements")]
    output: Option<String>,
    /// Output WAV bit depth for --input mode: 16/24 TPDF PCM or 32/64 float
    /// (default: 32)
    ///
    /// Parsed as a plain integer and matched against the choice list by hand, so `20` and
    /// `-16` produce argparse's `invalid choice` message instead of a parse failure.
    #[arg(
        long,
        default_value_t = 32,
        allow_negative_numbers = true,
        help_heading = "True-Peak Limiter & Measurements",
        help = "Output WAV bit depth for --input mode: 16/24 TPDF PCM or 32/64 float (default: 32)"
    )]
    out_bits: i64,
    /// Skip EQ convolution in --input mode (run true-peak limiter + measurements only)
    #[arg(long, help_heading = "True-Peak Limiter & Measurements")]
    bypass_eq: bool,
    /// Trim convolver/limiter trailing latency/tail to match input frame count in
    /// --input mode
    #[arg(long, help_heading = "True-Peak Limiter & Measurements")]
    trim_tail: bool,
    /// Measure audio file / convolved signal and print the true-peak & meter report
    /// without writing WAV
    #[arg(long, help_heading = "True-Peak Limiter & Measurements")]
    measure_only: bool,
    /// Shorthand for --input <wav> --bypass-eq --measure-only
    #[arg(long, help_heading = "True-Peak Limiter & Measurements")]
    measure_wav: Option<String>,
    /// Report what the correction demands of the content, per 1/3-octave band
    ///
    /// Report-only: it changes no filter, no level and no file. Off by default because the
    /// default report is byte-compared against the reference program, and a section added
    /// there would be a parity failure.
    #[arg(long, help_heading = "True-Peak Limiter & Measurements")]
    assess: bool,
    /// Boost level in dB at which --assess names a band (default: 6)
    #[arg(
        long,
        default_value_t = hypreq::assess::DEFAULT_BUDGET_DB,
        allow_negative_numbers = true,
        help_heading = "True-Peak Limiter & Measurements",
        help = format!(
            "Boost level in dB at which --assess names a band (default: {})",
            hypreq::assess::DEFAULT_BUDGET_DB
        )
    )]
    assess_budget: f64,
    /// Regularize the correction by how well the measurement fits a smoothed copy of itself
    ///
    /// `off` (the default) keeps the curve the reference program writes. `auto` sets the knee
    /// from the curve's own deviation; a number sets it directly, in dB of deviation. Not a
    /// default in this release: it changes the filter, and the comparison against the current
    /// behaviour is the point of running it explicitly.
    #[arg(
        long,
        default_value = "off",
        allow_hyphen_values = true,
        help_heading = "True-Peak Limiter & Measurements",
        help = "Regularize the correction: 'off' (default), 'auto', or a lambda in dB of deviation"
    )]
    regularize: String,
    /// Verify the filter against a measurement taken after it (CSV: freq, dB)
    ///
    /// Closes the loop the rest of the program leaves open: the capture is compared band by band
    /// with what the written filter was supposed to produce, after the capture's own level offset
    /// and noise floor are taken out. Report-only - it changes no filter and no audio, and the exit
    /// code does not carry the verdict, because a failed verification is a finding about a
    /// measurement rather than a program error.
    #[arg(
        long,
        value_name = "CSV",
        help_heading = "True-Peak Limiter & Measurements"
    )]
    verify_with: Option<std::path::PathBuf>,
    /// Equalise a measurement to a target defined on a different test fixture anyway.
    ///
    /// Narrower than the word suggests: it overrides exactly one refusal, the one that fires when a
    /// model's `fixture` and the fixture its target was defined on are different pieces of hardware
    /// (Olive & Clark, AES 2025). It does not bypass a malformed database, a bad option, or anything
    /// else, and it prints the refusal it overrode.
    #[arg(long, help_heading = "Database & Model Selection")]
    force: bool,
    /// Residual in dB beyond which --verify-with calls a band a failure
    #[arg(
        long,
        default_value_t = hypreq::verify::DEFAULT_PASS_DB,
        allow_negative_numbers = true,
        help_heading = "True-Peak Limiter & Measurements",
        help = format!(
            "Residual in dB beyond which --verify-with calls a band a failure (default: {})",
            hypreq::verify::DEFAULT_PASS_DB
        )
    )]
    verify_pass_db: f64,
    /// How far below the capture's own median band level a band may sit before it is unmeasured
    #[arg(
        long,
        default_value_t = hypreq::verify::DEFAULT_SNR_DB,
        allow_negative_numbers = true,
        help_heading = "True-Peak Limiter & Measurements",
        help = format!(
            "How far below the capture's median band level a band may sit before --verify-with calls it unmeasured (default: {} dB)",
            hypreq::verify::DEFAULT_SNR_DB
        )
    )]
    verify_snr_db: f64,
    /// Repeated-placement raw measurement CSV (frequency,dB); repeatable
    #[arg(long, value_name = "CSV", help_heading = "Groundwork (opt-in)")]
    ensemble_csv: Vec<PathBuf>,
    /// Masiero & Fels upper-variance weight on 1/6-oct smoothed ensemble curves (0 => median)
    #[arg(
        long,
        default_value_t = 0.0,
        allow_negative_numbers = true,
        help_heading = "Groundwork (opt-in)"
    )]
    ensemble_alpha: f64,
    /// Solve the global curvature-constrained box-QP over the 1/48-octave grid
    #[arg(long, help_heading = "Groundwork (opt-in)")]
    global_fit: bool,
    /// Alternative synthesis backend alongside the default FIR
    #[arg(
        long,
        value_name = "BACKEND",
        default_value = "fir",
        help_heading = "Groundwork (opt-in)"
    )]
    synth: String,
    /// Peaking bands when --synth peq
    #[arg(
        long,
        default_value_t = 8,
        value_name = "N",
        help_heading = "Groundwork (opt-in)"
    )]
    peq_bands: usize,
    /// Target family: harman, diffuse, spatial. Default: harman. The `Option` is load-bearing:
    /// a recorded preset supplies its own family only when this flag was not named, and a plain
    /// `String` defaulting to `"harman"` cannot tell "not named" apart from "named, to pull a
    /// diffuse/spatial preset back to harman."
    #[arg(long, value_name = "FAMILY", help_heading = "Groundwork (opt-in)")]
    target_family: Option<String>,
    /// Bass-shelf trim (dB) of the parameterized target. An explicit value always wins over a
    /// `--target-preset`, `0.0` included: "no shelf" is a decision, not the absence of one.
    #[arg(
        long,
        value_name = "DB",
        allow_negative_numbers = true,
        help_heading = "Groundwork (opt-in)"
    )]
    target_bass: Option<f64>,
    /// Presence-band trim (dB) of the parameterized target. Explicit values win over a
    /// `--target-preset`, `0.0` included.
    #[arg(
        long,
        value_name = "DB",
        allow_negative_numbers = true,
        help_heading = "Groundwork (opt-in)"
    )]
    target_mid: Option<f64>,
    /// Log-frequency tilt (dB/octave) of the parameterized target. Explicit values win over a
    /// `--target-preset`, `0.0` included.
    #[arg(
        long,
        value_name = "DB_PER_OCT",
        allow_negative_numbers = true,
        help_heading = "Groundwork (opt-in)"
    )]
    target_tilt: Option<f64>,
    /// Listening level in phon for the ISO 226:2023 contour-delta compensation
    #[arg(
        long,
        value_name = "PHON",
        allow_negative_numbers = true,
        help_heading = "Groundwork (opt-in)"
    )]
    listen_spl: Option<f64>,
    /// External ISO 226:2023 Table 1 JSON (the copyrighted table is not embedded)
    #[arg(long, value_name = "JSON", help_heading = "Groundwork (opt-in)")]
    iso226_table: Option<PathBuf>,
    /// Form factor used for the Harman PPR after --verify-with
    #[arg(
        long,
        value_name = "MODE",
        default_value = "over-ear",
        help_heading = "Groundwork (opt-in)"
    )]
    ppr_mode: String,
    /// Baseline target curve CSV (frequency,dB), replacing the database target for this run
    #[arg(long, value_name = "CSV", help_heading = "Groundwork (opt-in)")]
    target_csv: Option<PathBuf>,
    /// Rig-target recipe JSON (baseline CSV path + published shelf constants)
    #[arg(long, value_name = "JSON", help_heading = "Groundwork (opt-in)")]
    target_preset: Option<PathBuf>,
    /// Ignore the target recipe the model record ships with; equalise to the database's built-in curve
    #[arg(
        long,
        help_heading = "Groundwork (opt-in)",
        conflicts_with = "target_preset"
    )]
    no_target_preset: bool,
    /// Empirical rig translation JSON (delta mean and sigma, from derive_rig_calibration.py)
    #[arg(long, value_name = "JSON", help_heading = "Groundwork (opt-in)")]
    rig_calibration: Option<PathBuf>,
    /// Experimental: flatten the filter's group delay below <Hz>, leaving the magnitude alone
    #[arg(
        long,
        value_name = "HZ",
        allow_negative_numbers = true,
        help_heading = "Experimental",
        help = concat!(
            "EXPERIMENTAL. Replace the filter's phase below <Hz> with the straight line that best fits "
            ,"it, so the band's group delay flattens and the magnitude is carried through untouched. "
            ,"Off by default: the shipped filter stays minimum-phase. Costs latency, not magnitude, and "
            ,"whether the difference is audible is contested (Liski et al., TASLP 2021). Reports what "
            ,"it moved (e.g. 200 Hz)."
        )
    )]
    experimental_phase: Option<f64>,
}

fn main() {
    let mut args = Args::parse();

    // --measure-wav is a documented shorthand, not a separate mode.
    if let Some(wav) = args.measure_wav.clone() {
        args.input = Some(wav);
        args.bypass_eq = true;
        args.measure_only = true;
    }

    match run(args) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Error: {e}");
            exit(1);
        }
    }
}

/// A usage error in the shape argparse used: `error: <message>`, exit code 2.
fn usage_error(message: impl std::fmt::Display) -> ! {
    eprintln!("error: {message}");
    exit(USAGE_ERROR);
}

/// `python3 -c "print(f'{b}'))"` for a bool that has to read `True`/`False`.
fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

fn run(args: Args) -> Result<(), Error> {
    if args.upgrade_json {
        let (target, shown) = upgrade_target(&args);
        upgrade_measurements_json(&target).map_err(|e| Error::msg(e.0))?;
        println!(
            "Upgraded '{}' with plot_meta and callout annotations.",
            shown.display()
        );
        return Ok(());
    }

    // The Python loads the database before building the parser, so an unparseable
    // database is reported before an unknown flag is. A pure measurement run needs no
    // database at all, and must not fail when there is none.
    let needs_models = args.measure_wav.is_none() && !(args.input.is_some() && args.bypass_eq);
    let (data_file, models_dir) = database_paths(&args);
    let mut models = Database::default();
    if data_file.exists() || needs_models {
        models = load_measurement_database(&data_file, &models_dir).map_err(|e| Error::msg(e.0))?;
    }

    // argparse validates each `choices` option as it consumes the command line, so every
    // one of them is reported before the manual `--taps` check and before any output. -H
    // only carries a choice list when the database actually loaded models (hypreq.py:2882
    // passes `choices=None` otherwise), so a missing database never rejects a valid name.
    let aliases = active_aliases(&models);
    if !models.models.is_empty() && resolve_headphone(&models, &aliases, &args.headphone).is_none()
    {
        usage_error(format!(
            "argument -H/--headphone: invalid choice: '{}' (choose from {})",
            args.headphone,
            choice_list(&models, &aliases)
        ));
    }
    validate_choice(&args.notch_mode, &NotchMode::CHOICES, "--notch-mode");
    validate_choice(&args.stereo, &hypreq::stereo::Mode::CHOICES, "--stereo");
    validate_choice(&args.synth, &["fir", "parallel-iir", "peq"], "--synth");
    if let Some(family) = args.target_family.as_deref() {
        validate_choice(family, &["harman", "diffuse", "spatial"], "--target-family");
    }
    validate_choice(&args.ppr_mode, &["over-ear", "in-ear"], "--ppr-mode");
    for (name, v) in [
        ("--ensemble-alpha", Some(args.ensemble_alpha)),
        ("--target-bass", args.target_bass),
        ("--target-mid", args.target_mid),
        ("--target-tilt", args.target_tilt),
    ] {
        if let Some(v) = v {
            if !v.is_finite() {
                usage_error(format!("argument {name}: not a number: '{v}'"));
            }
        }
    }
    if args.ensemble_alpha < 0.0 {
        usage_error("--ensemble-alpha cannot be negative; 0.00 selects the median");
    }
    if args.target_csv.is_some() && args.target_preset.is_some() {
        usage_error(
            "--target-csv and --target-preset cannot be combined; put the baseline CSV path in the preset",
        );
    }
    if args.peq_bands == 0 || args.peq_bands > 32 {
        usage_error("--peq-bands must be in 1..=32");
    }
    match (args.listen_spl, args.iso226_table.as_ref()) {
        (Some(phon), Some(_)) if phon.is_finite() && phon > 0.0 => {}
        (Some(phon), Some(_)) => usage_error(format!(
            "--listen-spl must be a finite level above 0 phon (got {phon})"
        )),
        (Some(_), None) => usage_error("--iso226-table is required with --listen-spl"),
        (None, Some(_)) => usage_error("--listen-spl is required with --iso226-table"),
        (None, None) => {}
    }
    if !args.headroom_db.is_finite() {
        usage_error(format!(
            "argument --headroom-db: not a number: '{}'",
            args.headroom_db
        ));
    }
    if !args.assess_budget.is_finite() {
        usage_error(format!(
            "argument --assess-budget: not a number: '{}'",
            args.assess_budget
        ));
    }
    if !args.stereo_max_skew.is_finite() {
        usage_error(format!(
            "argument --stereo-max-skew: not a number: '{}'",
            args.stereo_max_skew
        ));
    }
    if args.stereo_max_skew < 0.0 {
        usage_error(
            "--stereo-max-skew cannot be negative; 0.00 means no per-ear difference at all",
        );
    }
    // Asked of a run that writes convolvers, `split` has nothing to act on: a mono impulse response
    // is loaded on both channels by definition, which is what a PipeWire `afir` takes. Refused
    // before any work rather than quietly downgraded to `report`, because a flag that silently did
    // something else is worse than a flag that refuses.
    if stereo_mode(&args) == hypreq::stereo::Mode::Split && (args.input.is_none() || args.bypass_eq)
    {
        usage_error(
            "--stereo split applies one filter per ear, which needs audio to apply them to: pass -i <stereo.wav> (without --bypass-eq), or use --stereo report to see what a shared filter leaves",
        );
    }
    if stereo_mode(&args) == hypreq::stereo::Mode::Split && args.synth != "fir" {
        usage_error(format!(
            "--stereo split is not supported with --synth {} (split per-ear filtering is currently FIR-only); use --synth fir or --stereo report",
            args.synth
        ));
    }
    // Verification thresholds: a non-finite value is a typo, and a zero threshold would fail every
    // band on the interpolation's own rounding, which is not a finding about anybody's headphones.
    for (name, v) in [
        ("--verify-pass-db", args.verify_pass_db),
        ("--verify-snr-db", args.verify_snr_db),
    ] {
        if !v.is_finite() {
            usage_error(format!("argument {name}: not a number: '{v}'"));
        }
        if v <= 0.0 {
            usage_error(format!("{name} must be above 0.00 dB"));
        }
    }
    if let Some(hz) = args.experimental_phase {
        // Same rule as every other numeric option: a NaN or an infinity is a typo, not a request.
        if !hz.is_finite() {
            usage_error("--experimental-phase must be a finite frequency in Hz");
        }
        if hz <= 0.0 {
            usage_error("--experimental-phase must be a frequency above 0 Hz");
        }
    }
    if let Some(path) = args.verify_with.as_ref() {
        if args.bypass_eq {
            usage_error(concat!(
                "--verify-with compares a later measurement against the filter that was supposed ",
                "to change it, and --bypass-eq writes none: drop one of the two"
            ));
        }
        // Two filters, one capture: the residual would be read as rig error when it is the trim.
        if stereo_mode(&args) == hypreq::stereo::Mode::Split {
            usage_error(concat!(
                "--verify-with cannot be combined with --stereo split, which applies two filters ",
                "and would leave one capture to be compared against both; use --stereo report to ",
                "see the pair, or verify with --stereo off"
            ));
        }
        // A mistyped capture path should not cost a full render. The file is read twice — once here,
        // once where the comparison runs — and a capture CSV is a few kilobytes, so the duplicate
        // read buys not threading a parsed curve through three call paths that mostly do not want it.
        let rows = hypreq::model::load_csv_points(path).map_err(|e| Error::msg(e.to_string()))?;
        if hypreq::verify::read_capture(&rows).is_none() {
            return Err(Error::msg(format!(
                "--verify-with: '{}' holds no usable (frequency, dB) pair to interpolate",
                path.display()
            )));
        }
    }
    validate_out_bits(args.out_bits);
    // `--regularize` takes a word or a number, so clap cannot check it; a value that is neither
    // of the documented forms is a usage error like any other bad choice.
    if let Err(msg) = regularize_arg(&args.regularize) {
        usage_error(msg);
    }

    if let Some(fs) = args.fs {
        if fs <= 0 {
            usage_error(format!("sample rate must be positive (got {fs})"));
        }
    }

    if args.taps < 256 {
        usage_error("--taps must be >= 256");
    }

    let tp_args = TpArgs {
        ratio: args.tp_ratio.clone(),
        ceiling_db: args.tp_ceiling,
        lookahead_ms: args.tp_lookahead,
        attack_ms: args.tp_attack,
        release_ms: args.tp_release,
        link_transient: args.tp_link_transient,
        link_release: args.tp_link_release,
        disable: args.no_tp_limiter,
    };
    // hypreq.py wraps build_limiter_cfg() in `except ValueError: parser.error(str(exc))`,
    // so an out-of-range limiter option is a usage error too.
    let lim_cfg = match tp_args.to_cfg() {
        Ok(cfg) => cfg,
        Err(msg) => usage_error(msg),
    };

    if args.list_models {
        for m in models.models.iter() {
            println!(
                "{:<12} | {} | target={} | gauss_apodize={}",
                m.key,
                m.name,
                m.target_name,
                py_bool(m.gauss_apodize)
            );
        }
        return Ok(());
    }

    let selected = match resolve_headphone(&models, &aliases, &args.headphone) {
        Some(k) => k.to_string(),
        // With an empty database `-H` is unvalidated, so the raw name falls through to the
        // lookup below and fails there, the way Python's `models[selected]` KeyError does.
        None => args.headphone.clone(),
    };

    // A curve measured on one fixture and equalised to a target *defined* on another is the most
    // likely way this program produces a confident, wrong filter, and until the fixture had a key
    // there was nothing to compare. Checked before any file is written, for exactly the models this
    // invocation is about to put a filter on.
    // A recipe brings its own baseline, so it brings its own rig: the fixture has to be read off the
    // recipe, because the database entry describes the target the recipe *replaced* — and the same
    // argument holds for a `target_preset` recorded on the model, which *is* that model's target on
    // an ordinary run. The recipe is therefore resolved per model below; the CLI's own recipe, when
    // there is one, outranks every record. Same argument for an empirical translation parsed from
    // `--rig-calibration` — it is one file for every model in this run, so it is read once here.
    // Both reads are the lenient ones: a file that does not parse is reported by the authoritative
    // loader where it has always been reported, and a guard that cannot run simply does not run.
    let cli_preset_fixture = args
        .target_preset
        .as_ref()
        .and_then(|p| preset_fixture_only(p));
    let cli_calibration = args
        .rig_calibration
        .as_ref()
        .and_then(|p| hypreq::fixture::RigCalibration::from_json_file(p).ok());
    for m in used_models(&models, &selected, args.input.is_some(), args.bypass_eq) {
        let recipe_fixture = cli_preset_fixture
            .or_else(|| effective_target_preset(m, &args).and_then(preset_fixture_only));
        let custom_target = args.target_csv.is_some()
            || (effective_target_preset(m, &args).is_some() && recipe_fixture.is_none());
        let target_fixture = if custom_target {
            recipe_fixture
        } else {
            recipe_fixture.or_else(|| models.target_fixtures.get(&m.target_name).copied())
        };
        let active_cal = cli_calibration.as_ref().or(m.rig_calibration.as_ref());
        if let Some(cal) = active_cal {
            rig_calibration_guard(m, target_fixture, cal, args.force)?;
        } else {
            // The recipe alone is the override here; a database that records its target rigs is the
            // ordinary case, not a replacement, and `fixture_conflict_ok` speaks about that case.
            fixture_guard(
                m,
                target_fixture,
                recipe_fixture.is_some(),
                args.target_preset.is_some(),
                args.force,
            )?;
        }
    }

    if let Some(input) = args.input.clone() {
        let model_cfg = if args.bypass_eq {
            None
        } else {
            let key = if selected == "all" {
                let first = (*models.keys().first().copied().unwrap_or("")).to_string();
                if first.is_empty() {
                    usage_error("No headphone models loaded; pass --bypass-eq or --data-file.");
                }
                println!(
                    "  [Note] --headphone=all with --input selects first model '{first}'. \
                     Pass -H <model> to choose."
                );
                first
            } else {
                selected.clone()
            };
            Some(
                models
                    .get(&key)
                    .ok_or_else(|| Error::msg(format!("unknown model '{key}'")))?,
            )
        };
        return run_audio_wav_mode(&args, &input, model_cfg, &lim_cfg);
    }

    if selected == "all" {
        for m in models.models.iter() {
            run_model(m, &args, false, &lim_cfg)?;
        }
    } else {
        let m = models
            .get(selected.as_str())
            .ok_or_else(|| Error::msg(format!("unknown model '{selected}'")))?;
        run_model(m, &args, true, &lim_cfg)?;
    }
    Ok(())
}

/// The figure's title block: the model's name, which is where a database records the fixture a
/// measurement was made on, and the notch mode for a model that has a notch to speak of.
///
/// Pulled out of `run_model` so that "the figure names the fixture it assumed" is a statement a test
/// can check on a string, rather than a restatement of `title = name.clone()`.
fn figure_title(model: &hypreq::model::Model, notch_mode: NotchMode) -> String {
    let mut title = model.name.clone();
    if model.notch_range_hz.is_some() {
        title = format!("{title} | Notch: {}", notch_mode.as_str());
    }
    title
}

/// The models this invocation will actually put a filter on.
///
/// Spelled out rather than re-derived at each check because the rule "`-H all` with `--input` means
/// the first model, not all of them" is already implemented inline in the audio path, and a guard
/// that guessed differently from the code it guards would either refuse runs that work or approve
/// runs it never looked at.
fn used_models<'a>(
    models: &'a Database,
    selected: &str,
    input: bool,
    bypass_eq: bool,
) -> Vec<&'a hypreq::model::Model> {
    if bypass_eq {
        return Vec::new();
    }
    match selected {
        // The audio path picks the first model and says so; a guard looking at all of them would
        // refuse a run that only ever touches one.
        "all" if input => models
            .keys()
            .first()
            .and_then(|k| models.get(k))
            .into_iter()
            .collect(),
        "all" => models.models.iter().collect(),
        key => models.get(key).into_iter().collect(),
    }
}

/// The fixture a `--target-preset` declares, or `None` when the recipe does not say.
///
/// Deliberately a *partial* parse: the authoritative one is [`load_target_preset`], which runs later
/// and reports a malformed recipe in full. Reading only this one key means the guard cannot change
/// the order in which the CLI reports unrelated problems with the same file; a recipe that does not
/// parse yields `None` here and is refused where it has always been refused.
fn preset_fixture_only(path: &Path) -> Option<hypreq::fixture::Fixture> {
    let text = std::fs::read_to_string(path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&text).ok()?;
    hypreq::fixture::Fixture::parse(val.get("fixture")?.as_str()?)
}

/// Refuse a measurement being equalised to a target defined on a different test fixture.
///
/// The check only exists when both sides recorded a fixture; a target with no recorded fixture is
/// unknown, not compatible, and an unknown pairing is not refused — refusing on the absence of a
/// claim would punish a database for being honest about what it does not know. `--force` lets a
/// specific pairing through with a line saying so, because a user who re-measured on another rig may
/// well want it, and it is their call once it is spelled out to them.
///
/// `recipe_fixture` is the rig the effective target recipe says its own baseline was defined on —
/// either a `--target-preset` passed on the command line or the `target_preset` recorded on the
/// model — and it is `None` unless such a recipe actually runs. It replaces the database's entry
/// rather than merging with it: for this run the recipe *is* the target, so a guard that kept
/// consulting `target_fixtures` would certify a pairing it had never looked at — the whole point of
/// the recipe is that it brings its own curve. `recipe_from_cli` records where the recipe came
/// from, so a refusal about the database's own record cannot tell the user to "pick a preset"
/// they never picked.
///
/// The two cases must not be collapsed into one `Option`, because the second use below reads
/// "was the target replaced" and a database that *does* record its target fixtures would then look
/// like a replacement on every ordinary run — which would silence the `fixture_conflict_ok`
/// exemption for exactly the models that carry it, and refuse the program's built-in path.
fn fixture_guard(
    model: &Model,
    target_fixture: Option<hypreq::fixture::Fixture>,
    recipe_active: bool,
    recipe_from_cli: bool,
    force: bool,
) -> Result<(), Error> {
    use hypreq::fixture::{pair, Pairing};
    // The resolved target fixture wins outright. A database entry for the *replaced* target has nothing
    // to say about a baseline that came from a file or custom curve, and consulting it would let a stale
    // entry vouch for a curve it was never measured against.
    let target = target_fixture;
    if pair(model.fixture, target) != Pairing::Different {
        return Ok(());
    }
    // A database can record that it inherited a pairing it cannot fix — which is the honest state of
    // every AutoEq-derived measurement equalised to a Harman target, the shipped three included.
    // The reason is in the file, so a person reading the database sees the decision; the guard's job
    // is to make an *unrecorded* pairing loud rather than to overrule a recorded one. A
    // `fixture_conflict_ok` on the model is a record about the model's *shipped* target, so it does
    // not vouch for a recipe the database has never seen.
    if model.fixture_conflict_ok.is_some() && !recipe_active {
        return Ok(());
    }
    let measured = model.fixture.expect("a Different pairing has both sides");
    let defined = target.expect("a Different pairing has both sides");
    let named = match (recipe_active, recipe_from_cli) {
        (true, true) => format!(
            "the --target-preset baseline (target '{}')",
            model.target_name
        ),
        (true, false) => format!(
            "the 'target_preset' recorded on this model (which replaced its target '{}')",
            model.target_name
        ),
        (false, _) => format!("the target '{}'", model.target_name),
    };
    // The suggestion list has to match what the guard actually honours. A recipe mismatch — from
    // the command line or from the database's own record — is never unlocked by
    // `fixture_conflict_ok` (see the exemption above), so recommending that route from a recipe
    // refusal would send the user down a path the code has just refused to walk.
    let routes = if recipe_active && recipe_from_cli {
        String::from(
            "Pick a preset whose 'fixture' matches the capture, correct the model's recorded \
             fixture, or pass --force to equalise anyway. 'fixture_conflict_ok' vouches for the \
             model's shipped target, never for a recipe chosen at the command line.",
        )
    } else if recipe_active {
        String::from(
            "The database paired this model with its own default recipe, and they disagree about \
             the rig. Correct the model's 'target_preset' or its declared fixture, or pass \
             --force to equalise anyway. 'fixture_conflict_ok' vouches for the built-in target, \
             never for the recorded recipe.",
        )
    } else {
        format!(
            "Choose a target recorded for {}, record the target's real fixture in \
             'target_fixtures', record the pairing on the model with 'fixture_conflict_ok' and its \
             reason, or pass --force to equalise anyway.",
            measured.key()
        )
    };
    let message = format!(
        "'{}' was measured on {} (fixture: {}), but {named} was defined on {} \
         (fixture: {}). Different fixtures present different acoustic transfer impedances, so this \
         filter would be fitting one rig's curve onto another's; a target is not portable between \
         them. {routes} Note that a published curve for one fixture has no free equivalent for \
         another, which is why the shipped models carry that record.",
        model.key,
        measured,
        measured.key(),
        defined,
        defined.key(),
    );
    if force {
        println!("  [Note] {message}");
        Ok(())
    } else {
        usage_error(message)
    }
}

/// Refuse an empirical rig translation that does not describe this measurement and this target.
///
/// A [`RigCalibration`](hypreq::fixture::RigCalibration) is not a filter, it is a *statement about
/// how two rigs differ for one class of headphone*: `delta_mean_db` is the response of the same
/// units on the target rig minus their response on the source rig, stratified by acoustic class.
/// Three things therefore have to line up before subtracting it from a correction is meaningful, and
/// none of them was checked: the rig the measurement came from (`source`), the rig the target curve
/// is defined on (`target`), and the construction the delta was averaged over (`acoustic_class`).
/// A translation that mismatches any of the three still produces a smooth, plausible curve — it is
/// simply the correction for some other headphone on some other rig, which is the failure mode this
/// program's whole design exists to avoid.
///
/// A side that records nothing is [`Pairing::Unknown`], not a passed check: the run continues and
/// says so, because refusing on the absence of a claim would make the honest database the one that
/// fails. A side that records something contradictory is refused, and `--force` overrides it with a
/// line saying which pairing was overridden and why it was objected to — the override is reported,
/// never silent.
fn rig_calibration_guard(
    model: &Model,
    target_fixture: Option<hypreq::fixture::Fixture>,
    cal: &hypreq::fixture::RigCalibration,
    force: bool,
) -> Result<(), Error> {
    use hypreq::fixture::{pair, AcousticClass, Pairing};
    let mut complaints: Vec<String> = Vec::new();
    let mut unverified: Vec<String> = Vec::new();

    // (a) the rig the measurement was made on has to be the rig the delta starts from, and the rig
    // the target curve is defined on has to be the rig the delta lands on.
    for (label, recorded, theirs, said) in [
        ("source", model.fixture, cal.source, "measured"),
        ("target", target_fixture, cal.target, "defined"),
    ] {
        match pair(recorded, Some(theirs)) {
            // `Different` proves both sides recorded something, so no `unwrap_or` hedge is needed.
            Pairing::Different => {
                let mine = recorded.expect("a Different pairing has both sides").key();
                let theirs = theirs.key();
                complaints.push(if label == "source" {
                    format!(
                        "its 'source' is {theirs}, but this headphone was measured on {mine} — the \
                         translation starts from {theirs}, which does not match a measurement made on {mine}"
                    )
                } else {
                    format!(
                        "its 'target' is {theirs}, but this run's target was defined on {mine} — the \
                         translation lands on {theirs}, which does not match a target defined on {mine}"
                    )
                });
            }
            Pairing::Unknown => {
                let theirs = theirs.key();
                unverified.push(format!(
                    "its '{label}' is {theirs}, while this run's curve has no fixture recorded on \
                     which it was {said}, so that half of the pairing could not be checked"
                ));
            }
            Pairing::Same | Pairing::Adjacent => {}
        }
    }

    // (b) the delta is an average over a construction; an open-back leak path is not a sealed one.
    match (model.acoustic_class, cal.acoustic_class) {
        (AcousticClass::Unknown, _) | (_, AcousticClass::Unknown) => unverified.push(format!(
            "the calibration is stratified by acoustic class ({}), and {} one, so the class was not \
             checked",
            class_name(cal.acoustic_class),
            if model.acoustic_class == AcousticClass::Unknown {
                "this model records no"
            } else {
                "the calibration records no"
            }
        )),
        (mine, theirs) if mine != theirs => complaints.push(format!(
            "its 'acoustic_class' is {}, but this model is {} — a rig-to-rig delta is an average \
             over one construction, and the leak path of one does not transfer to the other",
            class_name(theirs),
            class_name(mine)
        )),
        _ => {}
    }

    if complaints.is_empty() {
        for line in unverified {
            println!(
                "  [Note] --rig-calibration {} -> {}: {line}. Pass --rig-calibration for a rig you \
                 measured yourself, or record the fixture/acoustic_class on the model to have this \
                 checked.",
                cal.source.key(),
                cal.target.key()
            );
        }
        return Ok(());
    }
    let message = format!(
        "--rig-calibration '{}' -> '{}' does not describe '{}': {}. It is an empirical statement \
         about how two rigs differ for one class of headphone, not a filter: subtracting one that \
         describes another pairing leaves a smooth curve that answers a different question. Fix the \
         calibration's source/target/acoustic_class, leave 'rig_calibration' out of the model, or \
         pass --force to apply it anyway.",
        cal.source.key(),
        cal.target.key(),
        model.key,
        complaints.join("; ")
    );
    if force {
        println!("  [Note] {message}");
        Ok(())
    } else {
        usage_error(message)
    }
}

/// A class name a person can act on, rather than `Debug` on a unit variant.
fn class_name(class: hypreq::fixture::AcousticClass) -> &'static str {
    use hypreq::fixture::AcousticClass as C;
    match class {
        C::OpenBack => "open_back",
        C::ClosedBack => "closed_back",
        C::InEar => "in_ear",
        C::Unknown => "unknown",
    }
}

/// `--out-bits` accepts exactly these depths.
const OUT_BITS: [i64; 4] = [16, 24, 32, 64];

/// Reject a value outside a choice list with argparse's wording and exit code.
///
/// Quoted value, quoted choices: that is what a current interpreter prints for a list of
/// strings (`repr(str(choice))`), and it is what the oldest ones printed too. CPython changed
/// the quoting of this message twice and reverted one of the changes, so the quote characters
/// here match some patch levels and not others; `docs/SETUP.md` records the table of spellings.
/// The wording is pinned by the usage tests in `tests/cli_report.rs`, and the content — flag name,
/// phrase, value, choice set, exit code — is what they assert. Do not "fix" the quoting to chase a
/// specific patch release.
fn validate_choice(value: &str, choices: &[&str], flag: &str) {
    if !choices.contains(&value) {
        let list = choices
            .iter()
            .map(|c| format!("'{c}'"))
            .collect::<Vec<_>>()
            .join(", ");
        usage_error(format!(
            "argument {flag}: invalid choice: '{value}' (choose from {list})"
        ));
    }
}

/// `--out-bits`: the depths are ints, and a current CPython quotes them anyway — it renders
/// both slots with `repr(str(x))`, giving `invalid choice: '20' (choose from '16', '24', '32',
/// '64')`, which is what this prints. That is not stable across patch releases: 3.12.7 and 3.13.0
/// print the value as `20` with no quotes, and 3.12.8 through 3.14.3 drop the quotes from the
/// list. The string printed here is fixed and cannot follow the interpreter, so the harness
/// forgives quoting inside this message and nothing else (`--selftest`); measured on 3.11.18,
/// 3.12.7, 3.12.16, 3.13.0, 3.14.3 and 3.14.10.
fn validate_out_bits(bits: i64) {
    if !OUT_BITS.contains(&bits) {
        let list = OUT_BITS
            .iter()
            .map(|b| format!("'{b}'"))
            .collect::<Vec<_>>()
            .join(", ");
        usage_error(format!(
            "argument --out-bits: invalid choice: '{bits}' (choose from {list})"
        ));
    }
}

/// `--fs` as a positive rate. argparse takes any int, and the reference only notices a
/// non-positive one when it divides by it, so the message is the reference's own
/// `sample rate must be positive (got N)` and it is a plain error, not a usage error.
fn fs_hz(args: &Args, default: u32) -> Result<u32, Error> {
    match args.fs {
        None => Ok(default),
        Some(v) if v <= 0 => Err(Error::msg(format!(
            "sample rate must be positive (got {v})"
        ))),
        Some(v) => u32::try_from(v)
            .map_err(|_| Error::msg(format!("sample rate must be positive (got {v})"))),
    }
}

/// `--taps` as a count. `run()` has already rejected anything below 256.
fn taps_count(args: &Args) -> usize {
    args.taps as usize
}

/// Model aliases whose target actually loaded, mirroring the Python's filtering.
///
/// An alias that names a loaded model is dropped: keeping it would make that model
/// unreachable by its own name, since `-H` resolves aliases before plain keys.
fn active_aliases(models: &Database) -> Vec<(&'static str, &'static str)> {
    filter_aliases(|name| models.get(name).is_some())
}

/// The alias table reduced against a database predicate: keep an alias when its target
/// loaded and its own name is free.
fn filter_aliases(loaded: impl Fn(&str) -> bool) -> Vec<(&'static str, &'static str)> {
    MODEL_ALIASES
        .iter()
        .filter(|(alias, target)| loaded(target) && !loaded(alias))
        .map(|(a, t)| (*a, *t))
        .collect()
}

/// The valid `-H` values, in the order argparse would list them.
fn choice_list(models: &Database, aliases: &[(&'static str, &'static str)]) -> String {
    let mut items = vec!["'all'".to_string()];
    items.extend(models.keys().iter().map(|k| format!("'{k}'")));
    items.extend(aliases.iter().map(|(a, _)| format!("'{a}'")));
    items.join(", ")
}

/// Resolve `-H` to a model key, or `None` when neither a key nor a live alias matches.
fn resolve_headphone<'a>(
    models: &'a Database,
    aliases: &'a [(&'static str, &'static str)],
    requested: &'a str,
) -> Option<&'a str> {
    if requested == "all" {
        return Some("all");
    }
    if let Some((_, target)) = aliases.iter().find(|(a, _)| *a == requested) {
        return Some(target);
    }
    models.get(requested).map(|m| m.key.as_str())
}

fn limit_ir(ir: &mut [f64], lim_cfg: &hypreq::limiter::Cfg, headroom_db: f64) -> f64 {
    if !lim_cfg.enable {
        return 0.0;
    }
    let ceiling_db = lim_cfg.ceiling_db.min(headroom_db).min(0.0);
    let mut tp = TruePeak::new_floored(lim_cfg.ratio, 1);
    tp.push_ch(0, ir);
    tp.drain();
    let ir_dbtp = tp.dbtp();
    if ir_dbtp > ceiling_db {
        let gr = ir_dbtp - ceiling_db;
        let scale = 10.0f64.powf(-gr / 20.0);
        for x in ir.iter_mut() {
            *x *= scale;
        }
        gr
    } else {
        0.0
    }
}

fn limit_stereo_ir(
    left: &mut [f64],
    right: &mut [f64],
    lim_cfg: &hypreq::limiter::Cfg,
    headroom_db: f64,
) -> f64 {
    if !lim_cfg.enable {
        return 0.0;
    }
    let ceiling_db = lim_cfg.ceiling_db.min(headroom_db).min(0.0);
    let mut tp = TruePeak::new_floored(lim_cfg.ratio, 2);
    tp.push_ch(0, left);
    tp.push_ch(1, right);
    tp.drain();
    let ir_dbtp = tp.dbtp();
    if ir_dbtp > ceiling_db {
        let gr = ir_dbtp - ceiling_db;
        let scale = 10.0f64.powf(-gr / 20.0);
        for x in left.iter_mut() {
            *x *= scale;
        }
        for x in right.iter_mut() {
            *x *= scale;
        }
        gr
    } else {
        0.0
    }
}

/// Generate one model's convolvers and print the four report sections.
fn run_model(
    model: &Model,
    args: &Args,
    copy_default_ir: bool,
    lim_cfg: &hypreq::limiter::Cfg,
) -> Result<(), Error> {
    let fs = fs_hz(args, 48000)?;
    let taps = taps_count(args);
    let rate_tag = format_rate_tag(fs);
    let prefix = &model.file_prefix;
    let headroom_db = args.headroom_db;
    let use_apodize = match apodize_flag(args) {
        Some(b) => b,
        None => model.gauss_apodize,
    };
    let notch_mode = notch_mode(args);

    let title = figure_title(model, notch_mode);

    let stages = corrected_curve_stages(model, args, fs, !args.measure_only)?;
    let (active_ctrl, reg_lines) = stages.shared;

    // Panel 1's legend names the target from the database record and the model's plot_meta,
    // both written when the shipped pairing was the only one. A run that replaced the target
    // must caption its own curve: the preset's name, the csv baseline, or the parameterized
    // shape of the shipped one -- the same honesty the audit table now gets.
    let target_label = stages.target_label;

    let f64_wav = format!("{prefix}_f64_{rate_tag}.wav");
    let mono_rt = format!("{prefix}_mastering_mono_{rate_tag}.wav");
    let stereo_rt = format!("{prefix}_mastering_stereo_{rate_tag}.wav");

    let n_fft_mpv = mpv_n_fft(taps);
    let kaiser_beta = 18.0;
    // The same inversion of the Kaiser rule the window module uses to go the other way, so the
    // figure in the report and the figure that chose the taper cannot drift apart by someone
    // editing one of the two copies of `0.1102`.
    let kaiser_form_db = hypreq::window::kaiser_atten_db_from_beta(kaiser_beta);
    let mut mpv = synthesize_minimum_phase_ir(
        &active_ctrl,
        fs,
        n_fft_mpv,
        taps,
        taps / 8,
        headroom_db,
        use_apodize,
        kaiser_beta,
    )
    .map_err(|e| Error::msg(e.0))?;
    // Before the WAV is written: the shaped filter is the one that gets delivered, so it is also the
    // one every measurement below has to be taken on.
    let mut phase_done: Vec<(&str, hypreq::phase::Adjustment)> = Vec::new();
    if let Some(a) = phase_stage(&mut mpv, fs, "mastering", args)? {
        phase_done.push(("mastering", a));
    }

    let gr_mpv = limit_ir(&mut mpv.ir, lim_cfg, headroom_db);
    if gr_mpv > 0.0 {
        let scale = 10.0f64.powf(-gr_mpv / 20.0);
        for m in &mut mpv.mag_lin {
            *m *= scale;
        }
        mpv.preamp_db -= gr_mpv;
        mpv.unwindowed_floor_dbfs -= gr_mpv;
        mpv.windowed_floor_dbfs -= gr_mpv;
        for (label, a) in &mut phase_done {
            if *label == "mastering" {
                a.floor_dbfs -= gr_mpv;
            }
        }
    }
    write_wav_ieee_float(Path::new(&f64_wav), std::slice::from_ref(&mpv.ir), fs, 64)?;
    if copy_default_ir {
        std::fs::copy(Path::new(&f64_wav), Path::new("ir.wav"))
            .map_err(|e| Error::msg(format!("cannot write 'ir.wav': {e}")))?;
    }

    let ir_mpv_meas = measure_signal(std::slice::from_ref(&mpv.ir), fs, lim_cfg.ratio);
    let ir_l1_db = l1_gain_db(&mpv.ir);

    let (f_apod_start, _) = gauss_apodize_params(fs);
    let apod_desc = if use_apodize {
        format!("Gaussian Apodizer (>{:.1} kHz)", f_apod_start / 1000.0)
    } else {
        "Non-Apodizing 0.00 dB HF Wire".to_string()
    };
    println!();
    println!("{}", "=".repeat(80));
    println!("MODEL: {title}");
    println!("{}", "=".repeat(80));
    // What `--regularize` did belongs under the model's own heading, not in front of it: the
    // block names knots and a knee, which mean nothing before the reader knows which headphone
    // and which notch policy they are looking at. Empty on the default path, so the report the
    // parity harness compares is untouched.
    for line in reg_lines {
        println!("{line}");
    }
    // Same placement rule as the block above it: the ears belong to the model whose heading the
    // reader has just read. `split` cannot mean anything here — this path writes one mono impulse
    // response, and one impulse response is what gets loaded on both channels — so the argument
    // phase has already refused it and only `report` can arrive.
    if stereo_mode(args) != hypreq::stereo::Mode::Off {
        let (lines, _) = stereo_pair(model, args, &stages.pre_cap, stereo_mode(args))?;
        for line in lines {
            println!("{line}");
        }
    }
    println!("=== [1] Offline / mpv 64-Bit Double Convolver ({f64_wav}) ===");
    println!(
        "  Format            : 64-bit IEEE Float (pcm_f64le) | {} Hz | Mono",
        comma_sep(u64::from(fs))
    );
    println!(
        "  FFT / Taps        : {} FFT ({:.4} Hz bin) -> {} taps ({:.1} ms)",
        comma_sep(n_fft_mpv as u64),
        fs as f64 / n_fft_mpv as f64,
        comma_sep(mpv.ir.len() as u64),
        mpv.ir.len() as f64 / f64::from(fs) * 1000.0
    );
    println!(
        "  HF Mode / Taper   : {apod_desc} + Half-Kaiser Fade (beta={:.1}, {:.1} dB Kaiser formula)",
        kaiser_beta, kaiser_form_db
    );
    println!(
        "  Total Preamp      : {:.2} dB (Max Boost +{:.2} dB, {:.2} dB steady-state headroom)",
        mpv.preamp_db, mpv.max_boost_db, headroom_db
    );
    println!("  Causal Energy     : {:.10}%", mpv.energy_pct);
    println!(
        "  Decay @ Truncation: {:.2} dBFS (unwindowed) | {:.2} dBFS (after half-Kaiser fade)",
        mpv.unwindowed_floor_dbfs, mpv.windowed_floor_dbfs
    );
    println!(
        "  IR Peak / Meter   : {:+.3} dBFS sample | {:+.3} dBTP ({}) | {:+.3} dBTP (ITU {}) | ||h||_1 bound: {:+.2} dB",
        ir_mpv_meas.sample_peak_dbfs,
        ir_mpv_meas.true_peak_dbtp,
        lim_cfg.ratio.label(),
        ir_mpv_meas.itu_dbtp,
        ir_mpv_meas.itu_ratio.label(),
        ir_l1_db
    );
    println!(
        "  IR Hold (0.30s)   : {:+.3} dBFS (param {:.4} on -60..0 dB scale)",
        ir_mpv_meas.meter_peak_hold_dbfs, ir_mpv_meas.meter_param
    );

    // The real-time pair: a shorter filter, a gentler taper, and the f32 rounding the
    // PipeWire/DAW convolvers actually load. Scale taps and transform with rate to preserve
    // duration (~341 ms) and cepstral oversampling at higher rates.
    let rt_taps = ((16384usize * (fs as usize)) / 48000)
        .next_power_of_two()
        .max(16384);
    let n_fft_rt = (rt_taps * 2).max(32768);
    let mut rt = synthesize_minimum_phase_ir(
        &active_ctrl,
        fs,
        n_fft_rt,
        rt_taps,
        rt_taps / 8,
        headroom_db,
        use_apodize,
        16.0,
    )
    .map_err(|e| Error::msg(e.0))?;
    if let Some(a) = phase_stage(&mut rt, fs, "real-time", args)? {
        phase_done.push(("real-time", a));
    }
    let gr_rt = limit_ir(&mut rt.ir, lim_cfg, headroom_db);
    if gr_rt > 0.0 {
        let scale = 10.0f64.powf(-gr_rt / 20.0);
        for m in &mut rt.mag_lin {
            *m *= scale;
        }
        rt.preamp_db -= gr_rt;
        rt.unwindowed_floor_dbfs -= gr_rt;
        rt.windowed_floor_dbfs -= gr_rt;
        for (label, a) in &mut phase_done {
            if *label == "real-time" {
                a.floor_dbfs -= gr_rt;
            }
        }
    }
    let ir_rt_f32: Vec<f32> = rt.ir.iter().map(|&x| x as f32).collect();
    let ir_rt_f64: Vec<f64> = ir_rt_f32.iter().map(|&x| f64::from(x)).collect();
    write_wav_ieee_float(
        Path::new(&mono_rt),
        std::slice::from_ref(&ir_rt_f64),
        fs,
        32,
    )?;
    write_wav_ieee_float(
        Path::new(&stereo_rt),
        &[ir_rt_f64.clone(), ir_rt_f64.clone()],
        fs,
        32,
    )?;
    let err_rms_dbfs = {
        let mut acc = 0.0f64;
        for (a, b) in rt.ir.iter().zip(&ir_rt_f32) {
            let d = a - f64::from(*b);
            acc += d * d;
        }
        20.0 * ((acc / rt.ir.len() as f64).sqrt() + 1e-30_f64).log10()
    };
    let ir_rt_meas = measure_signal(std::slice::from_ref(&ir_rt_f64), fs, lim_cfg.ratio);

    println!();
    println!("=== [2] Real-Time PipeWire / DAW 32-Bit Float Convolvers ===");
    println!(
        "  FFT / Taps        : {} FFT ({:.4} Hz bin) -> {} taps ({:.1} ms, Half-Kaiser beta=16.0)",
        comma_sep(n_fft_rt as u64),
        fs as f64 / n_fft_rt as f64,
        comma_sep(rt.ir.len() as u64),
        rt.ir.len() as f64 / fs as f64 * 1000.0
    );
    println!(
        "  Decay @ Truncation: {:.2} dBFS (unwindowed) | {:.2} dBFS (after half-Kaiser fade)",
        rt.unwindowed_floor_dbfs, rt.windowed_floor_dbfs
    );
    println!(
        "  IR Peak / Meter   : {:+.3} dBFS sample | {:+.3} dBTP ({}) | {:+.3} dBTP (ITU {})",
        ir_rt_meas.sample_peak_dbfs,
        ir_rt_meas.true_peak_dbtp,
        lim_cfg.ratio.label(),
        ir_rt_meas.itu_dbtp,
        ir_rt_meas.itu_ratio.label()
    );
    println!("  f32 vs f64 Error  : {:.2} dBFS RMS", err_rms_dbfs);
    println!("  Written WAVs      : {mono_rt}, {stereo_rt}");
    print_phase_block(&phase_done, true);

    if args.export_csv {
        let raw_pts: Vec<Vec<f64>> = match stages.raw_active.as_ref() {
            Some(active) => active.iter().map(|&(f, g)| vec![f, g]).collect(),
            None => model.raw_pts.clone(),
        };
        let csv_path = export_audit_csv(
            &model.key,
            &raw_pts,
            stages.target.as_deref().unwrap_or(&model.target_pts),
            model.harm_offset,
            &active_ctrl,
        )?;
        println!("  Written Audit CSV : {}", csv_path.display());
    }

    if !args.no_plot {
        let inputs = PlotInputs {
            ir_rt: &ir_rt_f64,
            mag_lin: &rt.mag_lin,
            n_fft_rt,
            fs,
            boost_rt: rt.max_boost_db,
            pre_rt: rt.preamp_db,
            unwin_rt: rt.unwindowed_floor_dbfs,
            active_ctrl: &active_ctrl,
            raw_override: stages.raw_active.as_deref(),
            target_override: stages.target.as_deref(),
            target_label_override: target_label.as_deref(),
            notch_mode: notch_mode.as_str(),
            // The ears trace is opt-in like the rest of the flag: an unchanged run writes the
            // unchanged figure, even for a model that happens to carry two ear columns.
            show_ears: stereo_mode(args) != hypreq::stereo::Mode::Off,
            added_latency: phase_done
                .iter()
                .find(|(k, _)| *k == "real-time")
                .map_or(0, |(_, a)| a.added_latency_samples),
        };
        let data = PlotData::build(model, &inputs);
        let path = data.render(prefix)?;
        println!("  Written Plot      : {}", path.display());
    }

    // A deterministic 0 dBTP probe through the *offline* filter, then through the
    // limiter: the pair of numbers that says the convolver has real headroom left.
    let probe = build_convolver_probe(fs, 0.5);
    let convolved_probe = convolve_planar_fft(std::slice::from_ref(&probe), &mpv.ir, false);
    let pre_probe_meas = measure_signal(&convolved_probe, fs, lim_cfg.ratio);

    println!(
        "\n=== [3] True-Peak {} Limiter & Convolver Verification (0.00 dBTP Probe -> IR) ===",
        lim_cfg.ratio.label()
    );
    print_signal_measurements("Convolved Probe (Pre-Limiter)", &pre_probe_meas);
    if lim_cfg.enable {
        let mut probe_work = vec![convolved_probe[0].clone()];
        let mut lim = Limiter::enabled(*lim_cfg, fs, 1);
        {
            let mut refs: Vec<&mut [f64]> =
                probe_work.iter_mut().map(|c| c.as_mut_slice()).collect();
            lim.process(refs.as_mut_slice());
        }
        let tail = lim.flush();
        if let Some(t) = tail.first() {
            probe_work[0].extend_from_slice(t);
        }
        let rep = lim.report();
        let post_probe_meas = measure_signal(&probe_work, fs, lim_cfg.ratio);
        print_limiter_report(&rep, &post_probe_meas, lim_cfg, fs);
        print_signal_measurements("Convolved Probe (Post-Limiter)", &post_probe_meas);
    } else {
        println!("  [Note] True-peak limiter disabled (--no-tp-limiter).");
    }

    if args.verify_with.is_some() {
        // `mpv.ir` and not the real-time filter: section [4] tells the reader to load this one, so
        // this is the filter their post-EQ capture came from. The name says which, because a run
        // that writes two filters and verifies against the other one is worse than no verification.
        run_verification(
            model,
            stages.raw_active.as_deref(),
            stages.target.as_deref(),
            &mpv.ir,
            fs,
            mpv.preamp_db,
            args,
            Some(Path::new("")),
            &format!(
                "predicted from the {}-tap 64-bit convolver written above, with its {:+.2} dB preamp taken back out",
                comma_sep(mpv.ir.len() as u64),
                mpv.preamp_db
            ),
        )?;
    }

    if args.assess {
        // No audio on this path: the section describes the filter, and says so.
        print_assessment(&active_ctrl, &[], fs, args, mpv.preamp_db, lim_cfg.enable);
    }

    // mpv wants an absolute path with escaped separators. `absolute` is the lexical
    // equivalent of Python's `os.path.abspath`: unlike `canonicalize` it neither touches
    // the disk nor hands back a Windows `\\?\` prefix that mpv cannot open.
    let abs_ir_path = mpv_snippet_path(&f64_wav);
    println!(
        "\n=== [4] Matching mpv.conf Snippet ({fs} Hz, minp=1024:maxp=8192 -> {:.1} ms block latency) ===",
        1024.0 / fs as f64 * 1000.0
    );
    println!("audio-samplerate={fs}");
    println!(
        "af=lavfi=[aformat=sample_fmts=dblp,aresample={fs}:filter_size=256:phase_shift=12:cutoff=0.98:filter_type=kaiser:kaiser_beta=16:osf=dblp[in];amovie={abs_ir_path},aformat=sample_fmts=dblp[ir];[in][ir]afir=irfmt=mono:irnorm=-1:wet=1:minp=1024:maxp=8192:precision=double]"
    );
    Ok(())
}

/// Measure, EQ, limit, verify and write one audio file.
fn run_audio_wav_mode(
    args: &Args,
    input: &str,
    model_cfg: Option<&Model>,
    lim_cfg: &hypreq::limiter::Cfg,
) -> Result<(), Error> {
    let wav_in = read_wav(Path::new(input))?;
    // The file's own rate always wins: the EQ is synthesised to match the signal, not the
    // other way round. The reference prints the bare rate here, not a grouped one.
    let file_fs = wav_in.fs;
    let mut fs = fs_hz(args, file_fs)?;
    if let Some(want) = args.fs {
        if want != i64::from(file_fs) {
            println!(
                "  [Warning] --fs={want} differs from input WAV sample rate {file_fs}; synthesizing EQ IR at {file_fs} Hz."
            );
            fs = file_fs;
        }
    }

    // Computed before the banner so the regularization block can be printed under it, where the
    // reader already knows which model and which notch policy the knots belong to. Some for
    // exactly the path that needs it: a model, and the EQ not bypassed.
    let regularized = match model_cfg {
        Some(m) if !args.bypass_eq => {
            Some(corrected_curve_stages(m, args, fs, !args.measure_only)?)
        }
        _ => None,
    };
    // The design curves verification has to judge against, kept before `regularized` is
    // consumed by the filter stage below.
    let design_curves = regularized
        .as_ref()
        .map(|s| (s.raw_active.clone(), s.target.clone()));

    // The ears, asked about the curve the filter is about to be built from — so the block prints
    // under the banner with the regularization lines, and so `split`'s pair is derived from exactly
    // the curve `off` would have synthesized. Computed once, before the banner, because printing it
    // later would put the answer above the question.
    let ears_block = match (model_cfg, regularized.as_ref()) {
        (Some(m), Some(stages)) if stereo_mode(args) != hypreq::stereo::Mode::Off => {
            Some(stereo_pair(m, args, &stages.pre_cap, stereo_mode(args))?)
        }
        _ => None,
    };

    println!();
    println!("{}", "=".repeat(80));
    if args.measure_only && args.bypass_eq {
        println!("AUDIO MEASUREMENT: {input} ({})", wav_in.fmt_desc);
    } else if args.bypass_eq {
        println!(
            "TRUE-PEAK LIMITER ({}): {input} ({})",
            lim_cfg.ratio.label(),
            wav_in.fmt_desc
        );
    } else {
        let name = model_cfg
            .map(|m| m.name.as_str())
            .ok_or_else(|| Error::msg("EQ mode needs a model; pass --bypass-eq or --data-file"))?;
        println!(
            "CONVOLVER + {} LIMITER: {input} ({}) -> {name}",
            lim_cfg.ratio.label().to_uppercase(),
            wav_in.fmt_desc
        );
    }
    println!("{}", "=".repeat(80));
    if let Some(stages) = regularized.as_ref() {
        for line in &stages.shared.1 {
            println!("{line}");
        }
    }
    if let Some((lines, _)) = ears_block.as_ref() {
        for line in lines {
            println!("{line}");
        }
    }

    let m_in = measure_signal(&wav_in.planar, fs, lim_cfg.ratio);
    print_signal_measurements("Input Audio", &m_in);

    let mut work = wav_in.planar;
    let orig_n = work.first().map_or(0, |c| c.len());
    let mut ir_tag = "bypass".to_string();
    // The filter the output was actually made with. Verification has to run against this one rather
    // than a re-synthesis of the design, because what the listener hears is the taps, not the knots.
    let mut applied_ir: Option<(Vec<f64>, f64)> = None;

    let eq_inputs = match (model_cfg, regularized) {
        (Some(model), Some(stages)) => Some((model, stages.shared.0)),
        _ => None,
    };

    if let Some((model, active_ctrl)) = eq_inputs {
        let per_ear = ears_block.as_ref().and_then(|(_, pair)| pair.clone());
        let use_apodize = match apodize_flag(args) {
            Some(b) => b,
            None => model.gauss_apodize,
        };
        let taps = taps_count(args);
        let n_fft_mpv = mpv_n_fft(taps);
        let mut mpv_filter = if per_ear.is_none() {
            Some(
                synthesize_minimum_phase_ir(
                    &active_ctrl,
                    fs,
                    n_fft_mpv,
                    taps,
                    taps / 8,
                    args.headroom_db,
                    use_apodize,
                    18.0,
                )
                .map_err(|e| Error::msg(e.0))?,
            )
        } else {
            None
        };
        let phase = if per_ear.is_none() {
            if let Some(mpv) = mpv_filter.as_mut() {
                let mut p = phase_stage(mpv, fs, "applied", args)?;
                let gr = limit_ir(&mut mpv.ir, lim_cfg, args.headroom_db);
                if gr > 0.0 {
                    let scale = 10.0f64.powf(-gr / 20.0);
                    for m in &mut mpv.mag_lin {
                        *m *= scale;
                    }
                    mpv.preamp_db -= gr;
                    mpv.unwindowed_floor_dbfs -= gr;
                    mpv.windowed_floor_dbfs -= gr;
                    if let Some(ref mut adj) = p {
                        adj.floor_dbfs -= gr;
                    }
                }
                p
            } else {
                None
            }
        } else {
            None
        };
        if let Some(mpv) = mpv_filter.as_ref() {
            applied_ir = Some((mpv.ir.clone(), mpv.preamp_db));
            println!(
                "  EQ FIR ({})   : {} taps @ {} Hz | Preamp {:+.2} dB (Max boost +{:.2} dB, ||h||_1 {:+.2} dB)",
                model.file_prefix,
                comma_sep(mpv.ir.len() as u64),
                comma_sep(u64::from(fs)),
                mpv.preamp_db,
                mpv.max_boost_db,
                l1_gain_db(&mpv.ir)
            );
            if args.assess {
                print_assessment(&active_ctrl, &work, fs, args, mpv.preamp_db, lim_cfg.enable);
            }
            if let Some(a) = phase.as_ref() {
                println!("{}", "-".repeat(100));
                print_phase_block(&[("applied", a.clone())], false);
            }
        }
        match per_ear {
            Some((left_curve, right_curve)) => {
                let (mut left, mut right) = hypreq::ir::synthesize_stereo_minimum_phase_ir(
                    &left_curve,
                    &right_curve,
                    fs,
                    n_fft_mpv,
                    taps,
                    taps / 8,
                    args.headroom_db,
                    use_apodize,
                    18.0,
                )
                .map_err(|e| Error::msg(e.0))?;
                // Two ears, two filters: when experimental phase is enabled, both kernels are reshaped
                // jointly so both channels share identical impulse alignment (0 samples ITD skew)
                // and identical filter lengths.
                let mut ear_phase: Vec<(&str, hypreq::phase::Adjustment)> = if let Some(crossover) =
                    args.experimental_phase
                {
                    let (mut adj_l, mut adj_r) = hypreq::phase::linearize_band_stereo(
                            &left.ir, &right.ir, fs, crossover,
                        )
                        .ok_or_else(|| {
                            Error::msg(format!(
                                "--experimental-phase {:.1} Hz cannot run on the stereo split filters: \
                                 the crossover has to sit above 20 Hz and below Nyquist ({} Hz at {fs} Hz)",
                                crossover,
                                f64::from(fs) / 2.0
                            ))
                        })?;
                    left.ir = std::mem::take(&mut adj_l.ir);
                    left.energy_pct = adj_l.retained_energy_pct;
                    left.windowed_floor_dbfs = adj_l.floor_dbfs;
                    adj_l.full = Vec::new();

                    right.ir = std::mem::take(&mut adj_r.ir);
                    right.energy_pct = adj_r.retained_energy_pct;
                    right.windowed_floor_dbfs = adj_r.floor_dbfs;
                    adj_r.full = Vec::new();

                    vec![("left", adj_l), ("right", adj_r)]
                } else {
                    Vec::new()
                };
                let gr_shared =
                    limit_stereo_ir(&mut left.ir, &mut right.ir, lim_cfg, args.headroom_db);
                if gr_shared > 0.0 {
                    let scale = 10.0f64.powf(-gr_shared / 20.0);
                    for m in &mut left.mag_lin {
                        *m *= scale;
                    }
                    left.preamp_db -= gr_shared;
                    left.unwindowed_floor_dbfs -= gr_shared;
                    left.windowed_floor_dbfs -= gr_shared;

                    for m in &mut right.mag_lin {
                        *m *= scale;
                    }
                    right.preamp_db -= gr_shared;
                    right.unwindowed_floor_dbfs -= gr_shared;
                    right.windowed_floor_dbfs -= gr_shared;

                    for (_, adj) in &mut ear_phase {
                        adj.floor_dbfs -= gr_shared;
                    }
                }
                for (tag, ir) in [("left", &left), ("right", &right)] {
                    // Padded to the width the shared line's `(prefix)   ` occupies, so the two
                    // variants of this line put their numbers in the same column.
                    println!(
                        "  EQ FIR {tag:<16}: {} taps @ {} Hz | Preamp {:+.2} dB (Max boost +{:.2} dB, ||h||_1 {:+.2} dB)",
                        comma_sep(ir.ir.len() as u64),
                        comma_sep(u64::from(fs)),
                        ir.preamp_db,
                        ir.max_boost_db,
                        l1_gain_db(&ir.ir)
                    );
                }
                if args.assess {
                    print_assessment(
                        &active_ctrl,
                        &work,
                        fs,
                        args,
                        left.preamp_db,
                        lim_cfg.enable,
                    );
                }
                let channels = work.len();
                print_phase_block(&ear_phase, false);
                let filter_lat = ear_phase
                    .first()
                    .map_or(0, |(_, adj)| adj.added_latency_samples);
                let trim_to_in = args.trim_tail && filter_lat == 0;
                work = convolve_planar_pair(&work, &left.ir, &right.ir, trim_to_in)
                    .ok_or_else(|| {
                        Error::msg(format!(
                            "--stereo split hands one filter to each ear, which needs a stereo file; '{input}' has {channels} channel{}",
                            if channels == 1 { "" } else { "s" }
                        ))
                    })?;
                if args.trim_tail && filter_lat > 0 {
                    for ch in &mut work {
                        if ch.len() < filter_lat + orig_n {
                            ch.resize(filter_lat + orig_n, 0.0);
                        }
                        let end = filter_lat + orig_n;
                        *ch = ch[filter_lat..end].to_vec();
                    }
                }
            }
            None => {
                let mpv = mpv_filter.as_ref().unwrap();
                let filter_lat = phase.as_ref().map_or(0, |adj| adj.added_latency_samples);
                let trim_to_in = args.trim_tail && filter_lat == 0;
                work = convolve_planar_fft(&work, &mpv.ir, trim_to_in);
                if args.trim_tail && filter_lat > 0 {
                    for ch in &mut work {
                        if ch.len() < filter_lat + orig_n {
                            ch.resize(filter_lat + orig_n, 0.0);
                        }
                        let end = filter_lat + orig_n;
                        *ch = ch[filter_lat..end].to_vec();
                    }
                }
            }
        }
        let m_eq = measure_signal(&work, fs, lim_cfg.ratio);
        print_signal_measurements("Post-EQ (Pre-Limiter)", &m_eq);
        ir_tag = model.file_prefix.clone();
    }

    if !args.measure_only {
        if lim_cfg.enable {
            let mut lim = Limiter::enabled(*lim_cfg, fs, work.len());
            {
                let mut refs: Vec<&mut [f64]> = work.iter_mut().map(|c| c.as_mut_slice()).collect();
                lim.process(refs.as_mut_slice());
            }
            let tail = lim.flush();
            if !tail.is_empty() {
                if args.trim_tail {
                    let lat = lim.latency_samples();
                    work = work
                        .iter()
                        .zip(&tail)
                        .map(|(w, t)| {
                            let joined: Vec<f64> = w.iter().chain(t.iter()).copied().collect();
                            let end = (lat + orig_n).min(joined.len());
                            joined[lat.min(joined.len())..end].to_vec()
                        })
                        .collect();
                } else {
                    for (w, t) in work.iter_mut().zip(&tail) {
                        w.extend_from_slice(t);
                    }
                }
            }
            let rep = lim.report();
            let m_out = measure_signal(&work, fs, lim_cfg.ratio);
            println!();
            println!("=== True-Peak Limiter Report ===");
            print_limiter_report(&rep, &m_out, lim_cfg, fs);
            print_signal_measurements("Output Audio (Verified)", &m_out);
        } else {
            println!();
            println!("  [Note] True-peak limiter disabled (--no-tp-limiter).");
            let m_out = measure_signal(&work, fs, lim_cfg.ratio);
            print_signal_measurements("Output Audio", &m_out);
        }

        let out_path = match &args.output {
            Some(p) => p.clone(),
            None => default_out_path(
                input,
                &ir_tag,
                lim_cfg.enable.then(|| lim_cfg.ratio.label()),
            ),
        };

        let bits = u32::try_from(args.out_bits).unwrap_or(32);
        let frames = write_wav_pcm_or_float(Path::new(&out_path), &work, fs, bits)?;
        println!(
            "  Written Output WAV  : {out_path} ({bits}-bit @ {} Hz, {}ch, {} frames)",
            comma_sep(u64::from(fs)),
            work.len(),
            comma_sep(frames as u64)
        );
    }

    if let Some(path) = args.verify_with.as_ref() {
        // Asked and refused is better than asked and ignored: a run that never applied a filter has
        // nothing to verify, and printing nothing would read as a pass.
        let (Some(model), Some((ir, preamp_db))) = (model_cfg, applied_ir.as_ref()) else {
            return Err(Error::msg(format!(
                "--verify-with: '{}' was given but no filter was applied to this file, so there is nothing to verify",
                path.display()
            )));
        };
        // Measure-only writes no files at all, including this one; the report still prints, because
        // the finding is about the measurement and not about the output.
        run_verification(
            model,
            design_curves
                .as_ref()
                .and_then(|(r, _)| r.as_ref())
                .map(Vec::as_slice),
            design_curves
                .as_ref()
                .and_then(|(_, t)| t.as_ref())
                .map(Vec::as_slice),
            ir,
            fs,
            *preamp_db,
            args,
            (!args.measure_only).then(|| Path::new("")),
            &format!(
                "predicted from the {}-tap filter convolved into this file, before the limiter",
                comma_sep(ir.len() as u64)
            ),
        )?;
    }
    Ok(())
}

/// What `--regularize` asked for.
#[derive(Clone, Copy, Debug, PartialEq)]
enum RegularizeArg {
    /// Write the curve the reference program writes. The default, and the parity-pinned path.
    Off,
    /// Set the knee from the curve's own deviation.
    Auto,
    /// Set the knee directly, in dB of deviation.
    Fixed(f64),
}

/// Parse `--regularize`, accepting the three documented forms and nothing else.
///
/// A value that parses as a number but is not finite (`inf`, `NaN`) is rejected rather than
/// carried into the DSP: `λ = NaN` would quietly regularize nothing and print a `NaN` knee, which
/// is worse than an error message.
fn regularize_arg(raw: &str) -> Result<RegularizeArg, String> {
    let text = raw.trim();
    match text.to_ascii_lowercase().as_str() {
        "" | "off" | "none" => Ok(RegularizeArg::Off),
        "auto" => Ok(RegularizeArg::Auto),
        _ => match text.parse::<f64>() {
            Ok(v) if v.is_finite() => Ok(RegularizeArg::Fixed(v)),
            _ => Err(format!(
                "argument --regularize: expected 'off', 'auto', or a number in dB (got '{raw}')"
            )),
        },
    }
}

/// A curve to synthesize, and the report lines that say how it was arrived at. Empty lines mean
/// nothing was done and nothing was said, which is the default path.
type CurveAndReport = (Vec<(f64, f64)>, Vec<String>);

/// The control curve to build the filter from, and the lines describing what regularization did
/// to it.
///
/// With `--regularize off` this hands back the model's own knots, untouched — the same `Vec` the
/// reference program derives, which is why parity is unaffected by the flag existing. Anything
/// else goes through [`Model::regularize`], and the report says which path it took and how far
/// the knots moved, because "it regularized something" is not a reviewable claim.
fn regularized_ctrl(model: &Model, args: &Args) -> CurveAndReport {
    // Validated in the argument phase, where a bad value becomes a usage error. `Off` is the
    // fallback rather than a panic so this function stays total.
    let arg = regularize_arg(&args.regularize).unwrap_or(RegularizeArg::Off);
    if arg == RegularizeArg::Off {
        return (model.ctrl_pts.clone(), Vec::new());
    }
    let f: Vec<f64> = model.raw_pts.iter().map(|p| p[0]).collect();
    let g: Vec<f64> = model
        .raw_pts
        .iter()
        .map(|p| hypreq::model::point_gain(p))
        .collect();
    let policy = match arg {
        RegularizeArg::Auto => hypreq::fit::Policy::auto(&f, &g),
        RegularizeArg::Fixed(knee) => hypreq::fit::Policy::fixed(knee),
        RegularizeArg::Off => return (model.ctrl_pts.clone(), Vec::new()),
    };
    let done = model.regularize(&policy);
    if done.curve().is_empty() {
        return (
            model.ctrl_pts.clone(),
            vec![
                regularization_banner(&policy, done.describe()).to_string(),
                "  Curve           : unchanged".to_string(),
            ],
        );
    }
    let curve = done.curve().to_vec();
    let lines = regularization_lines(&policy, &model.ctrl_pts, &curve, done.describe());
    (curve, lines)
}

/// The heading every regularization block shares, so both paths print the same banner.
#[must_use]
fn regularization_banner(policy: &hypreq::fit::Policy, describe: &str) -> String {
    format!(
        "=== Regularization (knee {:.2} dB of deviation, {describe}) ===",
        policy.lambda_db
    )
}

/// What the policy did to `curve`, measured against `plain` — the same derivation with the policy
/// off. Shared by the two paths that can regularize: the plain one, where `plain` is the model's
/// shipped knots, and the target one, where it is this run's own unregularized derivation. One
/// sentence, one meaning, whichever way the curve arrived.
#[must_use]
fn regularization_lines(
    policy: &hypreq::fit::Policy,
    plain: &[(f64, f64)],
    curve: &[(f64, f64)],
    describe: &str,
) -> Vec<String> {
    let mut lines = vec![regularization_banner(policy, describe)];
    // Read `plain` at the regularized curve's own frequencies: a PCHIP passes through its nodes, so
    // when the two share a grid this is the exact old value. One interpolant for the whole curve —
    // building it inside the loop made the report quadratic in the knot count.
    let plain_f: Vec<f64> = plain.iter().map(|&(x, _)| x).collect();
    let plain_g: Vec<f64> = plain.iter().map(|&(_, y)| y).collect();
    let plain_curve = hypreq::pchip::Pchip::new(&plain_f, &plain_g);

    let mut worst_db = 0.0f64;
    let mut worst_hz = 0.0f64;
    let mut moved = 0usize;
    let (mut was_at_worst, mut now_at_worst) = (0.0f64, 0.0f64);
    for &(hz, gain) in curve {
        let before = plain_curve.eval(hz);
        let delta = gain - before;
        if delta.abs() > worst_db.abs() {
            worst_db = delta;
            worst_hz = hz;
            was_at_worst = before;
            now_at_worst = gain;
        }
        if delta.abs() >= 0.005 {
            moved += 1;
        }
    }
    if moved == 0 {
        // Say so, rather than reporting a largest move of +0.00 dB at a frequency the curve never
        // had. `worst_hz` starts at 0 Hz, and 0 Hz is not a knot of anything.
        lines.push(format!(
            "  Knots moved     : 0 of {} (every band sits under the knee)",
            curve.len()
        ));
        return lines;
    }
    lines.push(format!(
        "  Knots moved     : {moved} of {} (largest {worst_db:+.2} dB @ {worst_hz:.0} Hz)",
        curve.len()
    ));
    lines.push(format!(
        "  Extreme shifted : {was_at_worst:+.2} dB -> {now_at_worst:+.2} dB at that knot"
    ));
    lines
}

/// The `--regularize` policy for this run, or `None` when the flag is off.
fn regularization_policy(
    args: &Args,
    raw_pts: &[hypreq::model::Point],
) -> Option<hypreq::fit::Policy> {
    // Validated in the argument phase, where a bad value becomes a usage error; `Off` is the
    // fallback rather than a panic so this stays total.
    match regularize_arg(&args.regularize).unwrap_or(RegularizeArg::Off) {
        RegularizeArg::Off => None,
        RegularizeArg::Fixed(knee) => Some(hypreq::fit::Policy::fixed(knee)),
        RegularizeArg::Auto => {
            let f: Vec<f64> = raw_pts.iter().map(|p| p[0]).collect();
            let g: Vec<f64> = raw_pts
                .iter()
                .map(|p| hypreq::model::point_gain(p))
                .collect();
            Some(hypreq::fit::Policy::auto(&f, &g))
        }
    }
}

/// The fixture parameters a re-derivation has to reuse, so a target edit does not also silently
/// change the sub-bass leak discount or the coupler-null taper.
fn derivation_params(model: &Model) -> (f64, f64, f64) {
    match model.derivation {
        Some(d) => (
            d.hf_taper_start_hz,
            d.hf_bypass_hz,
            d.subbass_leak_discount_db,
        ),
        None => (
            model.hf_taper_start_hz.unwrap_or(model.hf_bypass_hz * 0.88),
            model.hf_bypass_hz,
            model.subbass_leak_discount_db.unwrap_or(0.0),
        ),
    }
}

/// A user-supplied rig-target recipe: published shelf constants over a baseline curve the user
/// owns. Copyrighted target tables are never embedded in this MIT crate.
#[derive(Debug)]
struct TargetPreset {
    name: String,
    baseline: Vec<(f64, f64)>,
    fixture: Option<hypreq::fixture::Fixture>,
    family: String,
    params: hypreq::target::TargetParams,
    source: String,
}

fn load_target_preset(path: &Path) -> Result<TargetPreset, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::msg(format!("--target-preset '{}': {e}", path.display())))?;
    let val: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        Error::msg(format!(
            "--target-preset '{}': invalid JSON: {e}",
            path.display()
        ))
    })?;
    let obj = val
        .as_object()
        .ok_or_else(|| Error::msg("--target-preset must be a JSON object"))?;
    let str_field = |k: &str| -> Result<String, Error> {
        obj.get(k)
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| Error::msg(format!("--target-preset needs a string '{k}'")))
    };
    let name = str_field("name")?;
    let source = obj
        .get("source")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unattributed")
        .to_string();
    let baseline_rel = str_field("baseline_csv")?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let baseline_path = base_dir.join(&baseline_rel);
    let rows = hypreq::model::load_csv_points(&baseline_path).map_err(|e| {
        Error::msg(format!(
            "--target-preset '{}': baseline '{}': {e}",
            path.display(),
            baseline_path.display()
        ))
    })?;
    let baseline: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|p| {
            let f = *p.first()?;
            // `point_gain`, not the last column: a `[freq, L, R]` baseline is a two-ear curve,
            // and reading the right ear as the whole curve is what `model::point_gain` exists
            // to prevent. A two-column row still reads its second column — no change there.
            let g = hypreq::model::point_gain(p);
            (f.is_finite() && f > 0.0 && g.is_finite()).then_some((f, g))
        })
        .collect();
    // Sort and collapse *before* counting. Two hand-pasted rows at one frequency are one knot
    // to every interpolant that reads this curve later, so a baseline that "counts" 2 and
    // interpolates as 1 must not pass the gate — the count and the curve have to be the same
    // object. Last write wins, the rule every other loader here uses.
    let baseline = hypreq::model::sort_and_dedup_pairs(&baseline);
    if baseline.len() < 2 {
        return Err(Error::msg(format!(
            "--target-preset '{}': baseline '{}' holds fewer than two usable points",
            path.display(),
            baseline_path.display()
        )));
    }
    let fixture = match obj.get("fixture") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(text)) => {
            Some(hypreq::fixture::Fixture::parse(text).ok_or_else(|| {
                Error::msg(format!(
                    "--target-preset '{name}' names unknown fixture '{text}' (choose from {})",
                    hypreq::fixture::Fixture::keys_joined()
                ))
            })?)
        }
        Some(other) => {
            return Err(Error::msg(format!(
                "--target-preset '{name}': 'fixture' must be a string, got {other}"
            )))
        }
    };
    let family = obj
        .get("family")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("harman")
        .to_string();
    // A recipe's family is data, not argv. `validate_choice` answers a bad value with
    // `process::exit(2)` — correct for the argument parser, but inside a `Result` function it
    // killed the process mid-call, straight past every caller's cleanup and every unit test
    // that expects a returned error. The recipe failure leaves as an error like every other
    // recipe failure.
    if !["harman", "diffuse", "spatial"].contains(&family.as_str()) {
        return Err(Error::msg(format!(
            "--target-preset '{name}': family '{family}' is not one of \
             'harman', 'diffuse', 'spatial'"
        )));
    }
    let num = |k: &str| -> Result<f64, Error> {
        obj.get("params")
            .and_then(|p| p.get(k))
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| Error::msg(format!("--target-preset params.{k} must be numeric")))
    };
    let params = hypreq::target::TargetParams {
        theta_bass: num("bass")?,
        theta_mid: num("mid")?,
        theta_tilt: num("tilt")?,
    };
    Ok(TargetPreset {
        name,
        baseline,
        fixture,
        family,
        params,
        source,
    })
}

/// Everything target-related the curve derivation needs, after CSV/preset substitution.
struct TargetContext {
    baseline: Vec<(f64, f64)>,
    family: hypreq::target::TargetFamily,
    /// The family actually in force (a recipe may override the CLI default).
    family_label: String,
    params: hypreq::target::TargetParams,
    /// Human-readable attribution of a recipe: name, rig it was defined on, published source.
    preset_info: Option<String>,
    /// The recipe's short name, for report labels that must not print the database's target
    /// name under a curve this run replaced (`None` when no `--target-preset` ran).
    preset_name: Option<String>,
}

fn target_context(model: &Model, args: &Args) -> Result<TargetContext, Error> {
    let mut baseline = model.target_pts.clone();
    // The CLI default is applied here rather than by clap, so that "named `harman`" stays
    // distinguishable from "named nothing" further down, where the preset's own family is
    // considered.
    let mut family = args
        .target_family
        .clone()
        .unwrap_or_else(|| "harman".to_string());
    let mut params = groundwork_target_params(args);
    let mut preset_info: Option<String> = None;
    let mut preset_name: Option<String> = None;
    if let Some(path) = args.target_csv.as_ref() {
        let rows = hypreq::model::load_csv_points(path)
            .map_err(|e| Error::msg(format!("--target-csv '{}': {e}", path.display())))?;
        baseline = rows
            .iter()
            .filter_map(|p| {
                let f = *p.first()?;
                // Mean of the ears on a `[freq, L, R]` row, the same reading `model::point_gain`
                // gives every other curve in this program; `*p.last()` silently ran the right
                // ear as the target.
                let g = hypreq::model::point_gain(p);
                (f.is_finite() && f > 0.0 && g.is_finite()).then_some((f, g))
            })
            .collect();
        // Sorted, last write winning, before anything interpolates it. A target CSV is hand-edited
        // and comes in whatever order the author's spreadsheet had; `Pchip::new` normalizes too, but
        // the count below has to be counted after deduplication or a file of two rows at the same
        // frequency passes as a curve.
        baseline = hypreq::model::sort_and_dedup_pairs(&baseline);
        if baseline.len() < 2 {
            return Err(Error::msg(format!(
                "--target-csv '{}' holds fewer than two usable points",
                path.display()
            )));
        }
    }
    if let Some(path) = effective_target_preset(model, args) {
        // A recipe the *database* recorded can fail on files the user has every right not to have
        // yet (its baseline CSV is user data, out of version control by design). That failure has
        // to name itself as the model's default and offer the opt-out, not masquerade as a bad
        // `--target-preset` the user never typed.
        let from_cli = args.target_preset.as_deref() == Some(path);
        let preset = load_target_preset(path).map_err(|e| {
            if from_cli {
                e
            } else {
                Error::msg(format!(
                    "model '{}' ships with target preset '{}' and it cannot be read: {e} — the \
                     recipe's baseline CSV is user-supplied (targets/README.md; \
                     scripts/import_kb5000_7.py). Pass --no-target-preset to equalise to the \
                     database's built-in curve instead.",
                    model.key,
                    path.display()
                ))
            }
        })?;
        baseline = preset.baseline;
        // The recipe carries the family it was defined with, but only where the user named
        // none. Comparing the *value* against "harman" (as this did while the flag defaulted
        // to "harman" inside clap) made it impossible to take a diffuse/spatial preset back to
        // the flat family — `--target-family harman` was indistinguishable from silence.
        // Naming the flag, even with the default's own word, is the request; the trims below
        // already use that rule for bass/mid/tilt.
        if args.target_family.is_none() {
            family = preset.family;
        }
        // A preset fills in the trims the user did not name. It is not allowed to overwrite one
        // they did, and `0.0` is one they did: a flat shelf is a decision a rig recipe gets to
        // make and a user gets to take back, so "was it named" is carried in the `Option` rather
        // than inferred from the value being indistinguishable from the default.
        if args.target_bass.is_none() {
            params.theta_bass = preset.params.theta_bass;
        }
        if args.target_mid.is_none() {
            params.theta_mid = preset.params.theta_mid;
        }
        if args.target_tilt.is_none() {
            params.theta_tilt = preset.params.theta_tilt;
        }
        preset_name = Some(preset.name.clone());
        preset_info = Some(format!(
            "{} ({}) — {}",
            preset.name,
            preset
                .fixture
                .map(|f| f.key().to_string())
                .unwrap_or_else(|| "rig unstated".to_string()),
            preset.source
        ));
    }
    let family_enum = match family.as_str() {
        "diffuse" => hypreq::target::TargetFamily::DiffuseField,
        "spatial" => hypreq::target::TargetFamily::SpatialDiffuseSlope,
        _ => hypreq::target::TargetFamily::Harman,
    };
    Ok(TargetContext {
        baseline,
        preset_info,
        preset_name,
        family_label: family,
        family: family_enum,
        params,
    })
}

fn groundwork_target_params(args: &Args) -> hypreq::target::TargetParams {
    hypreq::target::TargetParams {
        theta_bass: args.target_bass.unwrap_or(0.0),
        theta_mid: args.target_mid.unwrap_or(0.0),
        theta_tilt: args.target_tilt.unwrap_or(0.0),
    }
}

/// The target recipe this run should use for `model`.
///
/// The order is consent-first, not newest-wins: a recipe named on the command line replaces the
/// model's record outright; a `--target-csv` is a baseline the user brought themselves, and the
/// database's default steps aside for that too (the CLI already refuses `--target-csv` together
/// with `--target-preset`, so the only ambiguity left is between a user's curve and a record);
/// `--no-target-preset` steps it aside by name. With none of the three, the recorded default is
/// what the model ships with — which is the entire point of recording it.
fn effective_target_preset<'a>(model: &'a Model, args: &'a Args) -> Option<&'a Path> {
    if let Some(path) = args.target_preset.as_deref() {
        return Some(path);
    }
    if args.no_target_preset || args.target_csv.is_some() {
        return None;
    }
    model.target_preset.as_deref()
}

/// [`target_extension_active`] for one concrete model: the CLI's own switches, plus whatever
/// target recipe that model resolves to. Keeping the two questions apart is what lets the
/// CLI-only test below keep testing the CLI.
fn target_extension_active_for(model: &Model, args: &Args) -> bool {
    target_extension_active(args) || effective_target_preset(model, args).is_some()
}

/// Did the user ask for the parameterized target rather than the database's own?
///
/// Naming one of the three trims counts even when the number is `0.0` — typing the flag is the
/// request — which is why the flags carry an `Option` and this cannot be written as a value test.
fn target_extension_active(args: &Args) -> bool {
    args.target_family.is_some()
        || args.target_bass.is_some()
        || args.target_mid.is_some()
        || args.target_tilt.is_some()
        || args.iso226_table.is_some()
        || args.target_csv.is_some()
        || args.target_preset.is_some()
}

/// Re-derive the control curve from `raw` against the (possibly edited) target.
/// The control curve derived from the active target, and the report lines describing what
/// `--regularize` did to *that* curve (empty when the flag is off).
///
/// The curve and its own description leave together on purpose: an earlier version returned a curve
/// and let the caller keep regularization lines that had been computed against the model's shipped
/// knots, so with `--target-csv` or `--ensemble-csv` the report described a curve that was thrown
/// away a few statements later.
fn derive_with_target(
    model: &Model,
    args: &Args,
    raw_override: Option<&[(f64, f64)]>,
    ctx: &TargetContext,
) -> Result<CurveAndReport, Error> {
    let (taper, bypass, discount) = derivation_params(model);
    let raw: Vec<Vec<f64>> = match raw_override {
        Some(curve) => curve.iter().map(|&(f, g)| vec![f, g]).collect(),
        None => model.raw_pts.clone(),
    };
    let f_raw: Vec<f64> = raw.iter().map(|p| p[0]).collect();
    // One derivation shared with the report path: see run_target_curve.
    let target = run_target_curve(model, args, ctx, &f_raw)?;

    // `--regularize` acts on the curve this run actually derives, so it cannot be overwritten by the
    // derivation. The unregularized pass is kept as the baseline for the report, which is the only
    // way the numbers printed describe the curve that ships.
    let plain = hypreq::model::derive_ctrl_pts_from_raw(
        &raw,
        &target,
        model.harm_offset,
        taper,
        bypass,
        discount,
    );
    let Some(policy) = regularization_policy(args, &raw) else {
        return Ok((plain, Vec::new()));
    };
    let done = hypreq::model::derive_ctrl_pts_from_raw_regularized(
        &raw,
        &target,
        model.harm_offset,
        taper,
        bypass,
        discount,
        &policy,
    );
    let lines = regularization_lines(
        &policy,
        &plain,
        &done,
        // Wording taken from the type that reports the plain path, so the two can say different
        // things only if the paths genuinely differ.
        hypreq::model::Regularized::Derived(Vec::new()).describe(),
    );
    Ok((done, lines))
}

/// The target curve this run equalises against: the (possibly replaced) baseline sampled on the
/// merged grid with the parameterized deviation -- and the listening-SPL contour when one was
/// requested -- applied to it.
///
/// One function because two report consumers (the audit CSV and the verification plot's target
/// trace) used to read `model.target_pts` -- the database's shipped target -- while the fit read
/// this curve. With `--target-preset` or `--target-csv` in play they then tabulated and drew a
/// target that was not the one the run's own filter had been fitted to. Deriving the curve for
/// the reports from the same code the fit uses is what keeps that from splitting again.
///
/// `harm_offset` is deliberately NOT applied here: the fit's derivation and both report paths add
/// it at their own call sites, so this returns the identical object to what each of them had.
fn run_target_curve(
    model: &Model,
    args: &Args,
    ctx: &TargetContext,
    raw_axis: &[f64],
) -> Result<Vec<(f64, f64)>, Error> {
    let f_t: Vec<f64> = ctx.baseline.iter().map(|p| p.0).collect();
    let g_t: Vec<f64> = ctx.baseline.iter().map(|p| p.1).collect();

    // The target is sampled where the fit will read it, not where the baseline file happens to have
    // rows. `--target-baseline` CSVs are often three rows (20 / 1000 / 20000 Hz), and both things
    // that get added to a baseline are functions of frequency, not of those rows: the parameterized
    // deviation (whose 105 Hz shelf and 2500 Hz mid high-shelf live entirely between them) and an ISO 226
    // contour. Sampled at the baseline's own knots they collapsed to straight lines between three
    // points and the shelf vanished; sampled on the merged grid they survive whatever the file's
    // row spacing is. The grid is a superset of the baseline knots, so nothing is lost -- a PCHIP
    // passes through its own nodes -- and every point added is one the inversion was going to
    // interpolate onto anyway.
    let grid = target_query_grid(&f_t, raw_axis);
    let mut target: Vec<(f64, f64)> = if target_extension_active_for(model, args) {
        let vals = hypreq::target::evaluate_target(&ctx.baseline, &grid, ctx.family, ctx.params);
        grid.iter().copied().zip(vals).collect()
    } else {
        let baseline = hypreq::pchip::Pchip::new(&f_t, &g_t);
        grid.iter().map(|&f| (f, baseline.eval(f))).collect()
    };
    if let (Some(phon), Some(path)) = (args.listen_spl, args.iso226_table.as_ref()) {
        let table = hypreq::target::Iso226Table::from_json_file(path)
            .map_err(|e| Error::msg(format!("--iso226-table: {e}")))?;
        // Read at the target curve's own frequencies: a contour is a function of frequency, and
        // pairing it with a different axis than the one it was evaluated on would shift it.
        let f_target: Vec<f64> = target.iter().map(|p| p.0).collect();
        let delta = table.contour_delta_db(&f_target, phon, 65.0);
        for ((_, g), d) in target.iter_mut().zip(delta) {
            *g += d;
        }
    }
    Ok(target)
}

/// Frequencies to evaluate a target on: both input axes plus a 1/48-octave log grid across their
/// overlap, deduplicated and ascending.
///
/// 48 points per octave is the density the fitting grid already uses, so this samples the target at
/// least as finely as the filter will be built from it, and it stays under a thousand points for any
/// audio band.
#[must_use]
fn target_query_grid(f_t: &[f64], f_raw: &[f64]) -> Vec<f64> {
    const POINTS_PER_OCTAVE: f64 = 48.0;
    const MAX_GRID: usize = 4096;
    let mut grid: Vec<f64> = f_t
        .iter()
        .chain(f_raw.iter())
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();
    if grid.is_empty() {
        return grid;
    }
    grid.sort_by(|a, b| a.total_cmp(b));
    grid.dedup();
    let lo = grid[0];
    let hi = grid[grid.len() - 1];
    if hi > lo {
        let steps = ((hi / lo).log2() * POINTS_PER_OCTAVE).ceil() as usize;
        // One count for both the loop bound and the divisor. Clamping only the loop bound
        // (the old `steps.min(MAX_GRID)` against a `/ steps` denominator) stopped `t` at
        // `MAX_GRID / steps` and left the top slice of the log span unsampled — a >85-octave
        // axis lost its upper octaves silently. Clamping the count itself gives up density,
        // never coverage: the grid still ends exactly at `hi`.
        let n_steps = steps.clamp(1, MAX_GRID);
        for i in 0..=n_steps {
            let t = i as f64 / n_steps as f64;
            grid.push(lo * (hi / lo).powf(t));
        }
        grid.sort_by(|a, b| a.total_cmp(b));
        grid.dedup();
    }
    grid
}

#[cfg(test)]
mod grid_tests {
    use super::target_query_grid;

    /// A span wider than `MAX_GRID` points must thin the grid, not truncate it: the old
    /// code divided the clamped loop index by the unclamped step count, so `t` stopped at
    /// `4096 / steps` and everything above `lo * (hi/lo)^(4096/steps)` silently vanished.
    #[test]
    fn an_overlong_span_gives_up_density_not_coverage() {
        // 1e-24 Hz .. 100 Hz is 86.4 octaves, so the 48-points-per-octave demand exceeds
        // MAX_GRID. The endpoints are seeded into the grid either way — what truncation
        // silently throws away is the *interior* above `lo * (hi/lo)^(MAX_GRID/steps)`,
        // and the largest adjacent ratio is the honest observable: 2.06 (a whole octave
        // of nothing) under the old bound/divisor mismatch, 1.0147 (~1/48 octave) after.
        let f_t = [1e-24, 100.0];
        let grid = target_query_grid(&f_t, &[]);
        assert!(grid.iter().all(|f| f.is_finite() && *f > 0.0));
        let max_ratio = grid.windows(2).map(|w| w[1] / w[0]).fold(0.0f64, f64::max);
        assert!(
            max_ratio < 1.05,
            "the interior lost an octave-sized hole: largest gap x{max_ratio}"
        );
        // An ordinary audio span keeps its full 1/48-octave density (unchanged behaviour;
        // 20 Hz..20 kHz is 9.97 octaves -> 479 steps -> 480 grid points).
        let audio = target_query_grid(&[20.0, 20000.0], &[]);
        assert!(audio.len() >= 479, "audio span thinned: {}", audio.len());
        let adjacent = audio[1] / audio[0];
        assert!(
            adjacent < (2.0f64).powf(1.0 / 47.0),
            "audio span must keep ~1/48-octave spacing, got {adjacent}"
        );
    }
}

/// Result of collapsing repeated placement measurements.
#[derive(Debug)]
struct EnsembleOutcome {
    /// Robust raw curve on the model's own grid, normalised at 1 kHz.
    raw: Vec<(f64, f64)>,
    /// `sigma_r(f)` on that grid, for the global fit's variance propagation.
    sigma: Vec<(f64, f64)>,
    lines: Vec<String>,
}

fn normalize_ensemble_curve(raw: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if raw.len() < 2 {
        return raw.to_vec();
    }
    let (fs, gs): (Vec<f64>, Vec<f64>) = raw.iter().copied().unzip();
    let at_1k = hypreq::pchip::Pchip::new(&fs, &gs).eval(1000.0);
    if at_1k.is_finite() {
        raw.iter().map(|&(f, g)| (f, g - at_1k)).collect()
    } else {
        raw.to_vec()
    }
}

/// Collapse repeated placement measurements into one robust curve.
///
/// The placements come from `--ensemble-csv` when that flag names any file, and otherwise from the
/// curves the selected model records in the database (`raw_ensemble` / `raw_csvs`, which
/// [`hypreq::model`] has already resolved to points). Both sources carry rows that may hold two
/// ears, so both go through [`hypreq::model::point_gain`] — scoring a `[freq, L, R]` row by its
/// last column would fold one ear into the median of the ensemble.
///
/// Spread needs at least two curves. A model recording one placement — or none, which is every
/// database shipped so far — returns `None` and the run proceeds on the single raw curve, exactly
/// as it did before the key existed; a recorded ensemble whose points are all unusable is the same
/// `None`, because nothing the user asked for is being dropped.
fn ensemble_stage(
    args: &Args,
    recorded: &[Vec<hypreq::model::Point>],
) -> Result<Option<EnsembleOutcome>, Error> {
    let usable = |rows: &[hypreq::model::Point]| -> Vec<(f64, f64)> {
        let raw: Vec<(f64, f64)> = rows
            .iter()
            .filter_map(|p| {
                let f = *p.first()?;
                let g = hypreq::model::point_gain(p);
                (f.is_finite() && f > 0.0 && g.is_finite()).then_some((f, g))
            })
            .collect();
        // Sort/dedup and level-align each placement curve at 1 kHz before ensemble statistics so
        // constant SPL/gain offsets between captures never inflate `sigma_r` or bias the median.
        let sorted = hypreq::model::sort_and_dedup_pairs(&raw);
        normalize_ensemble_curve(&sorted)
    };
    let mut curves: Vec<Vec<(f64, f64)>> = Vec::new();
    let from = if args.ensemble_csv.is_empty() {
        for rows in recorded {
            let pts = usable(rows);
            if pts.len() >= 2 {
                curves.push(pts);
            }
        }
        "the model database"
    } else {
        for path in &args.ensemble_csv {
            let rows = hypreq::model::load_csv_points(path)
                .map_err(|e| Error::msg(format!("--ensemble-csv '{}': {e}", path.display())))?;
            let pts = usable(&rows);
            if pts.len() >= 2 {
                curves.push(pts);
            }
        }
        "--ensemble-csv"
    };
    if curves.is_empty() {
        if args.ensemble_csv.is_empty() {
            return Ok(None);
        }
        return Err(Error::msg("--ensemble-csv: no CSV held two usable points"));
    }
    // The paragraph above promises "spread needs at least two curves", and a single curve is
    // exactly what must not reach `analyze_ensemble`: its spread is zero by construction, and
    // the stage would report a smoothed median with a flat `sigma_r = 0` and a zero-width
    // bootstrap around a placement that was never repeated — a measured uncertainty reading of
    // "perfect" for the one thing with no second measurement. One recorded placement proceeds
    // on the single raw curve (`None`), the same as a model that records none; an explicit
    // `--ensemble-csv` that delivered only one usable curve is a request that cannot be
    // honoured, so it says so rather than silently becoming a single-curve "ensemble".
    if curves.len() == 1 {
        if args.ensemble_csv.is_empty() {
            return Ok(None);
        }
        return Err(Error::msg(
            "--ensemble-csv: spread needs at least two placement curves, one was usable",
        ));
    }
    // Build the common log grid over the curves' shared frequency overlap so flat PCHIP
    // endpoint clamping on mismatched bounds does not manufacture artificial edge spread;
    // fall back to the union only if the frequency spans are disjoint.
    let overlap_min = curves
        .iter()
        .filter_map(|c| c.first().map(|p| p.0))
        .fold(f64::NEG_INFINITY, f64::max);
    let overlap_max = curves
        .iter()
        .filter_map(|c| c.last().map(|p| p.0))
        .fold(f64::INFINITY, f64::min);
    let (f_min, f_max) =
        if overlap_min.is_finite() && overlap_max.is_finite() && overlap_min < overlap_max {
            (overlap_min, overlap_max)
        } else {
            let lo = curves
                .iter()
                .flat_map(|c| c.iter().map(|p| p.0))
                .fold(f64::INFINITY, f64::min);
            let hi = curves
                .iter()
                .flat_map(|c| c.iter().map(|p| p.0))
                .fold(f64::NEG_INFINITY, f64::max);
            (lo, hi)
        };
    let grid = hypreq::global_fit::uniform_log_grid(f_min, f_max, 48);
    let mode = if args.ensemble_alpha > 0.0 {
        hypreq::robust_fit::EnsembleMode::UpperVariance {
            alpha: args.ensemble_alpha,
        }
    } else {
        hypreq::robust_fit::EnsembleMode::Median
    };
    let stats = hypreq::robust_fit::analyze_ensemble(&curves, &grid, Some(1.0 / 6.0), mode, 400)
        .ok_or_else(|| Error::msg(format!("{from}: could not resample the placement curves")))?;
    let raw = normalize_ensemble_curve(
        &stats
            .f_hz
            .iter()
            .zip(&stats.safe_db)
            .map(|(&f, &g)| (f, g))
            .collect::<Vec<_>>(),
    );
    let sigma: Vec<(f64, f64)> = stats
        .f_hz
        .iter()
        .zip(&stats.sigma_r_db)
        .map(|(&f, &s)| (f, s))
        .collect();
    let worst = sigma
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|&(f, s)| format!("{s:.2} dB @ {}", hypreq::assess::band_name(f)))
        .unwrap_or_else(|| "n/a".to_string());
    let lines = vec![
        "=== Placement Ensemble ===".to_string(),
        format!(
            "  Curves          : {} resampled to {} bins (1/6-oct smoothed)",
            stats.count,
            grid.len()
        ),
        format!("  Source          : placement curves read from {from}"),
        format!("  Max sigma_r     : {worst}"),
        format!(
            "  CI_95 width     : {:.2} dB (whole-curve bootstrap)",
            stats
                .ci95_hi_db
                .iter()
                .zip(&stats.ci95_lo_db)
                .map(|(hi, lo)| hi - lo)
                .fold(0.0f64, f64::max)
        ),
    ];
    Ok(Some(EnsembleOutcome { raw, sigma, lines }))
}

fn global_fit_stage(
    ctrl: &[(f64, f64)],
    model: &Model,
    args: &Args,
    sigma: Option<&[(f64, f64)]>,
    rig_sigma: Option<&[(f64, f64)]>,
    raw_active: &[(f64, f64)],
) -> Result<CurveAndReport, Error> {
    if !args.global_fit {
        return Ok((ctrl.to_vec(), Vec::new()));
    }
    let f = hypreq::global_fit::uniform_log_grid(20.0, 20000.0, 48);
    if f.len() < 3 || ctrl.len() < 2 {
        return Err(Error::msg(
            "--global-fit: not enough points to build the grid",
        ));
    }
    let (cf, cg): (Vec<f64>, Vec<f64>) = ctrl.iter().copied().unzip();
    let err = hypreq::pchip::Pchip::new(&cf, &cg).eval_grid(&f);
    let k = f.len();
    let sigma_r = match sigma {
        Some(s) if s.len() >= 2 => {
            let (sf, sg): (Vec<f64>, Vec<f64>) = s.iter().copied().unzip();
            hypreq::fft::interp_grid(&f, &sf, &sg)
        }
        _ => vec![0.6; k],
    };
    // With an empirical calibration its delta_sigma_db is the rig term; without one, carry a
    // small conservative allowance rather than silently assuming cross-rig uncertainty is zero.
    //
    // The calibration arrives as knots on the control curve's own axis (19 to 128 rows for the
    // shipped controllers) while the QP grid is 479 bins. The old `rig_sigma.len() == k` check
    // therefore matched only by accident — and the accident path silently replaced a measured
    // `delta_sigma(f)` with a flat 0.5 dB under a report line that said "delta_sigma feeds the
    // QP", while the one case where lengths did agree would have paired two different frequency
    // axes without resampling. Interpolate the knots onto the QP grid exactly as `sigma_r`
    // crosses over; the flat allowance belongs to "no calibration", not to "calibration sampled
    // somewhere else".
    const RIG_SIGMA_DB: f64 = 0.5;
    let rig_term: Vec<f64> = match rig_sigma {
        Some(s) if s.len() >= 2 => {
            let (rf, rs): (Vec<f64>, Vec<f64>) = s.iter().copied().unzip();
            hypreq::fft::interp_grid(&f, &rf, &rs)
        }
        _ => vec![RIG_SIGMA_DB; k],
    };
    let total_sigma: Vec<f64> = sigma_r
        .iter()
        .zip(&rig_term)
        .map(|(&r, &s)| (r * r + s * s).sqrt())
        .collect();
    // The distortion ceiling is stated in absolute SPL, so the calibration term it needs is the
    // headphone's own response R(f) — and it has to be *this run's* R(f): `raw_active` is the
    // ensemble curve when one was collapsed, and it has already been moved by the rig
    // calibration's `+delta_mean` when one is active, because the ceiling compares against the
    // same rig coordinate the correction was built in. Reading `model.raw_pts` here instead —
    // the old behaviour — judged the ceiling of a calibrated or ensemble run on an uncalibrated,
    // single-placement curve. Handing `q_max_bounds` zeros was the earlier failure and is not a
    // neutral default either: it over-allows by R(f) wherever the driver is already sensitive.
    // No raw curve at all stays zeros: there is genuinely nothing to contribute.
    let listen_spl = args.listen_spl.unwrap_or(85.0);
    let h_cal: Vec<f64> = if raw_active.len() >= 2 {
        let (fr, gr): (Vec<f64>, Vec<f64>) = raw_active.iter().copied().unzip();
        hypreq::fft::interp_grid(&f, &fr, &gr)
    } else {
        vec![0.0; k]
    };
    let mut q_max = match model.distortion.as_ref() {
        Some(d) => d.q_max_bounds(&f, &h_cal, listen_spl, 3.0, 12.0),
        None => vec![12.0; k],
    };
    // Counted before `q_max` is clamped to 0.0 at `hf_bypass_hz`: the report has to say how much of the band the
    // distortion ceiling actually binds, rather than mistaking HF bypass zeroing for distortion limits.
    let ceiling_sweeps = model.distortion.as_ref().map_or(0, |d| d.sweeps.len());
    let ceiling_bins = q_max.iter().filter(|q| **q < 12.0 - 1e-9).count();
    let mut q_min = vec![-24.0; k];
    for (i, &hz) in f.iter().enumerate() {
        if hz >= model.hf_bypass_hz {
            q_min[i] = 0.0;
            q_max[i] = 0.0;
        }
    }
    let r_peak = h_cal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let input = hypreq::global_fit::GlobalFitInput {
        f_hz: f.clone(),
        raw_error_db: err,
        total_sigma_db: total_sigma,
        anchor_db: vec![0.0; k],
        q_min_db: q_min,
        q_max_db: q_max,
    };
    let res = hypreq::global_fit::solve_global_qp(
        &input,
        &hypreq::global_fit::GlobalFitConfig::default(),
    )
    .ok_or_else(|| Error::msg("--global-fit: the box-QP could not be factorized"))?;
    let out: Vec<(f64, f64)> = res
        .f_hz
        .iter()
        .copied()
        .zip(res.q_db.iter().copied())
        .collect();
    let mut lines = vec![
        "=== Global Curvature-Constrained Fit ===".to_string(),
        format!(
            "  Grid / iterations: {} bins / {} ({})",
            k,
            res.iterations,
            if res.converged {
                "converged"
            } else {
                "max iterations"
            }
        ),
        "  Weights          : w = 1 / (sigma_r^2 + sigma_rig^2), ERB density, floor 0.30 dB"
            .to_string(),
    ];
    if ceiling_sweeps > 0 {
        lines.push(format!(
            "  Boost ceiling    : {ceiling_sweeps} sweeps, L_max at >3% THD, {ceiling_bins} of {k} bins limited (R(f) peak {r_peak:+.1} dB, listen {listen_spl:.0} dB)",
        ));
    }
    // A database that records `"excursion"` expects an answer about it, which until now it never
    // got: the field was parsed and dropped. It is spent here on a report line rather than on the
    // solver's box, for the reason [`hypreq::driver::ExcursionModel::margins_db`] gives — the
    // 12 dB-per-octave debt is a first-order rule whose rate depends on the alignment, so it can
    // tell the operator that a sub-bass boost is ambitious without silently zeroing a filter. The
    // measured distortion envelope above is what binds. And it is judged against the curve the fit
    // delivered, not the curve it was asked for: a ceiling on demand the filter does not place
    // would be fiction.
    if let Some(exc) = model.excursion.as_ref() {
        let margins = exc.margins_db(&f, &res.q_db, &h_cal, listen_spl);
        let short = margins.iter().filter(|m| **m < 0.0).count();
        let worst = margins
            .iter()
            .zip(&f)
            .filter(|(m, _)| m.is_finite())
            .min_by(|a, b| a.0.total_cmp(b.0));
        let detail = match worst {
            Some((m, &wf)) => format!("worst {m:+.2} dB at {}", hypreq::assess::band_name(wf)),
            None => "no finite margin to quote".to_string(),
        };
        lines.push(format!(
            "  Excursion screen : {short} of {k} bins short of X_max at {listen_spl:.0} dB listen \
             (fs {:.0} Hz, X_max at {:.0} dB SPL) — {detail}; a screen, not a bound on the fit",
            exc.fs_hz, exc.ref_xmax_spl_db
        ));
    }
    Ok((out, lines))
}

/// Synthesize a parametric filter for the curve it is handed.
///
/// `ctrl` must already carry the model's notch ceiling: the export is the curve the user loads
/// into Equalizer APO, and it has to be the same filter the FIR path delivers — same boosts,
/// same safety cap, same preamp reservation. `write_files` is `--measure-only`'s veto over the
/// export (the report line names which happened); `headroom_db` is the convolver's steady-state
/// headroom, mirrored into the preset's Preamp line.
fn synth_stage(
    ctrl: &[(f64, f64)],
    args: &Args,
    fs: u32,
    prefix: &str,
    write_files: bool,
    headroom_db: f64,
) -> Result<CurveAndReport, Error> {
    if args.synth == "fir" || ctrl.len() < 2 {
        return Ok((ctrl.to_vec(), Vec::new()));
    }
    let (cf, cg): (Vec<f64>, Vec<f64>) = ctrl.iter().copied().unzip();
    let grid = hypreq::fft::geomspace(20.0, (0.45 * f64::from(fs)).min(20_000.0), 240);
    let want = hypreq::pchip::Pchip::new(&cf, &cg).eval_grid(&grid);
    let (pred, lines) = match args.synth.as_str() {
        "peq" => {
            let bands = hypreq::synth_alt::fit_peq_cascade(
                &grid,
                &want,
                args.peq_bands,
                f64::from(fs),
                1e-4,
            );
            let pred = hypreq::synth_alt::cascade_mag_db(&bands, &cf, f64::from(fs));
            // APO applies the same filter the convolver would, so it has to keep the same
            // staging: `headroom_db - max boost`, the rule `ir.rs` bakes into every FIR tap.
            // Evaluated on a dense grid spanning the full band (DC to Nyquist) to catch
            // inter-knot peaks and resonance. All-cut filters cap max_boost at 0.0 dB so
            // the preamp never becomes positive.
            let nyq = 0.5 * f64::from(fs);
            let mut eval_grid = hypreq::fft::geomspace(20.0_f64.min(nyq * 0.5).max(1.0), nyq, 512);
            eval_grid.push(0.0);
            eval_grid.push(nyq);
            eval_grid.extend(cf.iter().copied());
            for b in &bands {
                eval_grid.push(b.f0_hz);
            }
            eval_grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            eval_grid.dedup();
            let dense_mag = hypreq::synth_alt::cascade_mag_db(&bands, &eval_grid, f64::from(fs));
            let max_boost = dense_mag
                .into_iter()
                .filter(|v| v.is_finite())
                .fold(0.0_f64, f64::max);
            let preamp_db = headroom_db - max_boost;
            let summary = bands
                .iter()
                .map(|b| {
                    format!(
                        "{}/{:+.1}dB/Q{:.1}",
                        hypreq::assess::band_name(b.f0_hz),
                        b.gain_db,
                        b.q
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            let mut lines = vec![
                "=== Vicanek Matched PEQ Synthesis ===".to_string(),
                format!("  Bands           : {} ({summary})", bands.len()),
            ];
            if write_files {
                let path = format!("{prefix}_peq.txt");
                std::fs::write(
                    &path,
                    hypreq::synth_alt::to_equalizer_apo_text(&bands, preamp_db, f64::from(fs)),
                )
                .map_err(|e| Error::msg(format!("cannot write '{path}': {e}")))?;
                lines.push(format!("  APO preset      : {path}"));
            } else {
                lines.push(format!(
                    "  APO preset      : not written (--measure-only), Preamp {preamp_db:.2} dB"
                ));
            }
            (pred, lines)
        }
        "parallel-iir" => {
            let bank =
                hypreq::synth_alt::BankParallelIir::fit_from_ctrl(ctrl, f64::from(fs), 32, 1e-8)
                    .ok_or_else(|| {
                        Error::msg("--synth parallel-iir: the IIR fit did not converge")
                    })?;
            let pred = bank.eval_mag_db(&cf);
            let nyq = 0.5 * f64::from(fs);
            let mut eval_grid = hypreq::fft::geomspace(20.0_f64.min(nyq * 0.5).max(1.0), nyq, 512);
            eval_grid.push(0.0);
            eval_grid.push(nyq);
            eval_grid.extend(cf.iter().copied());
            eval_grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            eval_grid.dedup();
            let dense_mag = bank.eval_mag_db(&eval_grid);
            let max_boost = dense_mag
                .into_iter()
                .filter(|v| v.is_finite())
                .fold(0.0_f64, f64::max);
            let preamp_db = headroom_db - max_boost;
            let mut export_bank = bank.clone();
            export_bank.scale_gain(10f64.powf(preamp_db / 20.0));
            let mut lines = vec![
                "=== Bank Fixed-Pole Parallel IIR Synthesis ===".to_string(),
                format!(
                    "  Sections        : {} parallel biquads + d0 {:+.3}",
                    bank.sections.len(),
                    bank.d0
                ),
            ];
            if write_files {
                let path = format!("{prefix}_parallel_iir.txt");
                std::fs::write(&path, export_bank.to_text_with_preamp(preamp_db))
                    .map_err(|e| Error::msg(format!("cannot write '{path}': {e}")))?;
                lines.push(format!(
                    "  IIR bank file   : {path} (Preamp {preamp_db:+.2} dB)"
                ));
            } else {
                lines.push(format!(
                    "  IIR bank file   : not written (--measure-only), Preamp {preamp_db:+.2} dB"
                ));
            }
            (pred, lines)
        }
        other => return Err(Error::msg(format!("unknown --synth backend '{other}'"))),
    };
    let out: Vec<(f64, f64)> = cf.iter().copied().zip(pred).collect();
    Ok((out, lines))
}

/// The curve to synthesize, and the lines saying what `--regularize` did to get there.
///
/// Regularize first, cap second, always in that order. The cap is a ceiling on what reaches the
/// filter — the notch range is where a coupler's own resonance lives, so a boost there is a boost
/// the ear will not hear consistently — and `--regularize` is a soft opinion about which bands are
/// worth chasing at all. Inverting the two, or skipping the cap when `--regularize` is on because
/// "the regularizer already softens things", would let a boost through that the model's own
/// metadata forbids: the regularizer shrinks toward a bypass by the *measurement's* deviation, and
/// has no idea what `safe_cap_db` says about this model's notch.
///
/// The groundwork flags are all opt-in: with none of them set — and with no ensemble recorded in
/// the model — this returns the same `(model.ctrl_pts, empty lines)` as before, which is what
/// `regularize_off_hands_back_the_models_own_curve` and `groundwork_flags_are_all_off_by_default`
/// hold shut.
///
/// `fs` is the sample rate the *run* resolved to — the input file's rate in `--input` WAV mode,
/// `--fs` (default 48000) elsewhere — because a PEQ cascade or fixed-pole bank fitted at 48 kHz
/// is the wrong filter at 44.1 or 96 kHz: Vicanek's matched biquads and Bank's pole grid are
/// both functions of `fs`, and only the FIR stage used to see the file's rate. `write_files` is
/// `--measure-only`: a measurement pass exports no preset files.
fn corrected_curve_stages(
    model: &Model,
    args: &Args,
    fs: u32,
    write_files: bool,
) -> Result<FilterStages, Error> {
    let ctx = target_context(model, args)?;
    let target_label: Option<String> = if target_extension_active_for(model, args) {
        Some(match (ctx.preset_name.as_ref(), args.target_csv.as_ref()) {
            (Some(name), _) => format!("Target Curve ({name})"),
            (None, Some(_)) => "Target Curve (csv baseline)".to_string(),
            (None, None) => format!("Target Curve ({} + params)", model.target_name),
        })
    } else {
        None
    };
    let ensemble = ensemble_stage(args, &model.raw_ensemble)?;
    let calibration = match args.rig_calibration.as_ref() {
        Some(path) => Some(
            hypreq::fixture::RigCalibration::from_json_file(path)
                .map_err(|e| Error::msg(format!("--rig-calibration: {e}")))?,
        ),
        None => model.rig_calibration.clone(),
    };

    // The raw curve this run designs against, in target-rig coordinates: the collapsed ensemble
    // when one exists (it *is* the raw curve the fit gets), the database record otherwise, and
    // moved by `+delta_mean` under an active calibration. Both target subtraction, the HF bypass
    // flat-wire taper, regularization, the distortion ceiling, and the excursion screen operate
    // in target-rig coordinates.
    let mut raw_active: Vec<(f64, f64)> = match ensemble.as_ref() {
        Some(e) => e.raw.clone(),
        None => model
            .raw_pts
            .iter()
            .map(|p| (p[0], hypreq::model::point_gain(p)))
            .collect(),
    };
    let mut cal_lines = Vec::new();
    if let Some(cal) = calibration.as_ref() {
        let fr: Vec<f64> = raw_active.iter().map(|p| p.0).collect();
        let (mean_raw, _) = cal.on_grid(&fr);
        for ((_, g), d) in raw_active.iter_mut().zip(&mean_raw) {
            *g += d;
        }
        cal_lines.push("=== Empirical Rig Translation ===".to_string());
        cal_lines.push(format!(
            "  {} -> {}      : raw translated to target rig (+delta_mean), delta_sigma feeds the QP",
            cal.source.key(),
            cal.target.key()
        ));
    }

    let diverged =
        target_extension_active_for(model, args) || ensemble.is_some() || calibration.is_some();

    // Which ever way the curve arrived, the lines that describe it travel with it. The target paths
    // put their own block first and the regularization block after it, because the sentence "knots
    // moved 12 of 480" only means anything once the reader knows which target the knots came from.
    let (mut base_ctrl, mut reg_lines) = if diverged {
        let (curve, regularized) = derive_with_target(model, args, Some(&raw_active), &ctx)?;
        let mut lines = Vec::new();
        if let Some(e) = ensemble.as_ref() {
            lines.extend(e.lines.clone());
        }
        lines.extend(cal_lines);
        if target_extension_active_for(model, args) {
            lines.push("=== Parameterized Target ===".to_string());
            lines.push(format!(
                "  Family          : {} (bass {:+.2}, mid {:+.2}, tilt {:+.2} dB/oct)",
                ctx.family_label,
                ctx.params.theta_bass,
                ctx.params.theta_mid,
                ctx.params.theta_tilt
            ));
            if let Some(info) = ctx.preset_info.as_ref() {
                lines.push(format!("  Preset          : {info}"));
            }
            if args.target_csv.is_some() {
                lines.push("  Baseline        : replaced by --target-csv".to_string());
            }
        }
        lines.extend(regularized);
        (curve, lines)
    } else {
        regularized_ctrl(model, args)
    };

    let rig_sigma: Option<Vec<(f64, f64)>> = calibration.as_ref().map(|cal| {
        let f: Vec<f64> = base_ctrl.iter().map(|p| p.0).collect();
        let (_, sigma) = cal.on_grid(&f);
        f.into_iter().zip(sigma).collect()
    });
    let design_target: Option<Vec<(f64, f64)>> = diverged
        .then(|| {
            let f_raw: Vec<f64> = raw_active.iter().map(|p| p.0).collect();
            run_target_curve(model, args, &ctx, &f_raw)
        })
        .transpose()?;

    if args.global_fit {
        let sigma = ensemble.as_ref().map(|e| e.sigma.as_slice());
        let (fitted, lines) = global_fit_stage(
            &base_ctrl,
            model,
            args,
            sigma,
            rig_sigma.as_deref(),
            &raw_active,
        )?;
        base_ctrl = fitted;
        reg_lines.extend(lines);
    }

    // The ceiling lands here — once, before anything is synthesized from the curve. Capping is
    // not idempotent: `c * tanh(g / c) < g` for every positive `g`, however far below `c`, so a
    // curve that goes through `apply_notch_mode` twice loses level twice inside the notch band.
    // `--stereo split` therefore starts from `pre_cap` below and caps each ear itself rather
    // than starting from this one. It lands *before* the synthesis because the synthesis
    // exports: an Equalizer APO preset cut from the uncapped curve promised the user the boosts
    // the model's own `safe_cap_db` metadata forbids, while the FIR delivered the capped one.
    // Synthesizing the capped curve makes both deliverables the same filter. `--synth fir` is
    // the identity stage, so the default path is numerically untouched by the reordering.
    let pre_cap = base_ctrl;
    let capped_ctrl = apply_notch_mode(
        &pre_cap,
        notch_mode(args),
        model.safe_cap_db,
        model.notch_range_hz,
    )
    .map_err(|e| Error::msg(e.0))?;
    let (active_ctrl, synth_lines) = synth_stage(
        &capped_ctrl,
        args,
        fs,
        &model.file_prefix,
        write_files,
        args.headroom_db,
    )?;
    reg_lines.extend(synth_lines);
    Ok(FilterStages {
        shared: (active_ctrl, reg_lines),
        pre_cap,
        raw_active: diverged.then_some(raw_active),
        target: design_target,
        target_label,
    })
}

/// The delivered filter, and the curves this run was actually designed against.
struct FilterStages {
    /// What the convolver is loaded with: the ceiling-capped curve (`--synth fir`), or the
    /// parametric fit to it (`--synth peq` / `parallel-iir`).
    shared: CurveAndReport,
    /// The capped curve's own input — the derived curve *before* the ceiling. A per-ear trim has
    /// to start here, because the ceiling is the last word per ear and applying it twice is not
    /// applying it once.
    pre_cap: Vec<(f64, f64)>,
    /// This run's raw curve (ensemble-collapsed and rig-translated), when the run diverged from
    /// the database record. `--verify-with` reads this instead of `model.raw_pts`.
    raw_active: Option<Vec<(f64, f64)>>,
    /// This run's resolved target curve (preset/CSV/family/trims/contour), when the run
    /// diverged. `--verify-with` reads this instead of `model.target_pts`.
    target: Option<Vec<(f64, f64)>>,
    /// The legend target label computed when target extension is active.
    target_label: Option<String>,
}

/// The delivered filter alone, for callers that have no use for the pre-ceiling curve.
///
/// Test-only: every production path needs both stages (the report prints the shared filter, `--stereo
/// split` needs the curve it was capped from), so this exists to keep the tests reading as one call
/// rather than as `corrected_curve_stages(..).shared` everywhere.
#[cfg(test)]
fn corrected_curve(model: &Model, args: &Args) -> Result<CurveAndReport, Error> {
    corrected_curve_stages(model, args, fs_hz(args, 48000)?, true).map(|stages| stages.shared)
}

/// The database record's raw curve as plain `(f, gain)` pairs, for tests that call
/// `global_fit_stage` the way `corrected_curve_stages` does on an undiverged run.
#[cfg(test)]
fn raw_curve(model: &Model) -> Vec<(f64, f64)> {
    model
        .raw_pts
        .iter()
        .map(|p| (p[0], hypreq::model::point_gain(p)))
        .collect()
}

/// Apply the experimental mixed-phase stage to a finished filter, keeping the report honest.
///
/// The stage replaces the delivered taps, so the two report numbers that describe those taps — the
/// causal-energy share and the decay floor at truncation — are replaced with the stage's own
/// measurements of the taps it delivered, not left describing a filter that is no longer being
/// written.
///
/// `unwindowed_floor_dbfs` is deliberately untouched: it measures the minimum-phase filter *before*
/// its tail fade, which is exactly what its label claims, and there is no post-reshape equivalent of
/// "before the fade" to substitute. The phase block says which two lines it replaced.
fn phase_stage(
    res: &mut hypreq::ir::IrResult,
    fs: u32,
    label: &str,
    args: &Args,
) -> Result<Option<hypreq::phase::Adjustment>, Error> {
    let Some(crossover) = args.experimental_phase else {
        return Ok(None);
    };
    let mut a = hypreq::phase::linearize_band(&res.ir, fs, crossover).ok_or_else(|| {
        Error::msg(format!(
            "--experimental-phase {:.1} Hz cannot run on the {label} filter: the crossover has to sit \
             above 20 Hz and below Nyquist ({} Hz at {fs} Hz)",
            crossover,
            f64::from(fs) / 2.0
        ))
    })?;
    // The delivered taps are the reshaped ones. Handing back only the diagnostics and leaving the
    // minimum-phase kernel in place is exactly the bug this line prevents, and both of the tests that
    // caught it passed every assertion except the ones that compared bytes.
    res.ir = std::mem::take(&mut a.ir);
    res.energy_pct = a.retained_energy_pct;
    res.windowed_floor_dbfs = a.floor_dbfs;
    // The taps are now in `res`; the caller does not need a second copy of a 65 536-point array.
    a.full = Vec::new();
    Ok(Some(a))
}

/// Print the experimental phase block, once, naming every filter it shaped.
fn print_phase_block(items: &[(&str, hypreq::phase::Adjustment)], is_model_mode: bool) {
    let Some((_, first)) = items.first() else {
        return;
    };
    for line in hypreq::phase::report_lines(first) {
        println!("{line}");
    }
    if items.len() > 1 {
        let parts: Vec<String> = items
            .iter()
            .map(|(label, a)| {
                format!(
                    "{label} {:.3} ms / {:.1} ms latency",
                    a.max_gd_change_ms, a.added_latency_ms
                )
            })
            .collect();
        println!(
            "  Every filter shaped : {} (group delay moved / latency added)",
            parts.join(", ")
        );
    }
    if is_model_mode {
        println!(
            "  Note                : 'Causal Energy' and the '(after half-Kaiser fade)' floor above are "
        );
        println!(
            "                        the reshaped filter's; '(unwindowed)' still describes the "
        );
        println!("                        minimum-phase filter before its tail fade.");
    }
}

/// The written filter's realised magnitude, in dB, on a query grid.
///
/// Not the design's control knots: the filter that exists has truncation ripple, and the half-Kaiser
/// tail pulls the top octave. Verifying against the knots instead would blame the rig for the
/// filter's own deviation, which is the one confusion this section must not produce.
fn realized_gain_db(ir: &[f64], fs: u32, f_query: &[f64]) -> Vec<f64> {
    if ir.is_empty() || fs == 0 {
        return vec![0.0; f_query.len()];
    }
    // Zero-padded past 2L so the transform reads the aperiodic response rather than a circular one
    // — and past a floor of 32,768 points, which is not the same requirement. The magnitude is
    // interpolated *in dB* between bins, and a straight line through two dB samples is a lie
    // wherever a null sits between them: a 2-tap average padded only to its own 2L gets four bins,
    // one of them a null, and a query between them returns a number in the thousands of dB below
    // unity. The convolvers here are 4,096 to 65,536 taps, where bins are well under a hertz apart
    // and the interpolation error is nil; the floor is what stops the helper being quietly wrong
    // for a short filter, which is exactly what a test or a `--taps 256` run hands it.
    const MIN_POINTS: usize = 32768;
    let n = MIN_POINTS.max(hypreq::fft::shl_bit_length(
        ir.len().saturating_mul(2).saturating_sub(1),
    ));
    let mut padded = ir.to_vec();
    padded.resize(n, 0.0);
    let spec = hypreq::fft::rfft(&padded, n);
    let bins = hypreq::fft::rfftfreq(n, f64::from(fs));
    // `+ 1e-300` keeps a null bin finite instead of -inf, which would otherwise poison a band mean.
    let mag: Vec<f64> = spec
        .iter()
        .map(|c| 20.0 * (c.re.hypot(c.im) + 1e-300).log10())
        .collect();
    hypreq::fft::interp_grid(f_query, &bins, &mag)
}

/// Compare this run's filter with a capture someone took after applying it, and say what came of it.
///
/// `dir` is where the per-band table goes; `None` in measure-only mode, which writes no files at
/// all. `note` is the one thing the numbers cannot say about the run itself - which of the several
/// filters this program can write was the one applied.
/// `raw_active` / `target` are the curves **this run** was designed against — the collapsed
/// ensemble and the rig-translated raw, and the preset/CSV/family/contour-resolved target —
/// when the run replaced either. Passing `None` for both (a plain run) keeps the original
/// behaviour of reading `model.raw_pts` and `model.target_pts` verbatim. Feeding a
/// `--target-preset` run its database's shipped target — which is what happened before — scored
/// the post-EQ capture against a curve the run never equalised to: the predicted response used
/// an uncalibrated raw curve, and the shortfall and Harman PPR used the replaced-away target.
#[allow(clippy::too_many_arguments)]
fn run_verification(
    model: &Model,
    raw_active: Option<&[(f64, f64)]>,
    target: Option<&[(f64, f64)]>,
    ir: &[f64],
    fs: u32,
    preamp_db: f64,
    args: &Args,
    dir: Option<&std::path::Path>,
    note: &str,
) -> Result<(), Error> {
    let Some(path) = args.verify_with.as_ref() else {
        return Ok(());
    };
    let rows = hypreq::model::load_csv_points(path).map_err(|e| Error::msg(e.to_string()))?;
    let post = hypreq::verify::read_capture(&rows).ok_or_else(|| {
        Error::msg(format!(
            "--verify-with: '{}' holds no usable (frequency, dB) pair to interpolate",
            path.display()
        ))
    })?;
    let cfg = hypreq::verify::Config {
        pass_db: args.verify_pass_db,
        snr_db: args.verify_snr_db,
    };

    // The prediction is assembled on the model's own axis: the captured response plus the filter
    // that was written, both in dB.
    //
    // The written IR carries this run's preamp - that is what makes it a convolver with headroom -
    // so the constant is taken back out here. Not to be fussy: the verifier removes the *capture's*
    // constant before judging shape, so the preamp would vanish from the residual anyway, but it
    // would stay in the design shortfall, which is never offset-corrected, and that number is meant
    // to describe a spectral decision rather than a headroom choice.
    let (f_raw, g_raw): (Vec<f64>, Vec<f64>) = match raw_active {
        Some(curve) => curve.iter().copied().unzip(),
        None => (
            model.raw_pts.iter().map(|p| p[0]).collect(),
            model
                .raw_pts
                .iter()
                .map(|p| hypreq::model::point_gain(p))
                .collect(),
        ),
    };
    let (f_t, g_t): (Vec<f64>, Vec<f64>) = match target {
        Some(curve) => {
            // `run_target_curve` deliberately does not carry the harmonic-offset constant (see
            // its doc); this is this curve's call site, so it is added here.
            curve
                .iter()
                .map(|&(f, g)| (f, g + model.harm_offset))
                .unzip()
        }
        None => (
            model.target_pts.iter().map(|p| p.0).collect(),
            model
                .target_pts
                .iter()
                .map(|p| p.1 + model.harm_offset)
                .collect(),
        ),
    };

    let f_raw_min = f_raw.iter().cloned().fold(f64::INFINITY, f64::min);
    let f_raw_max = f_raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let f_t_min = f_t.iter().cloned().fold(f64::INFINITY, f64::min);
    let f_t_max = f_t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let top = (f64::from(fs) / 2.0)
        .min(20000.0)
        .min(f_raw_max)
        .min(f_t_max);
    let bottom = 20.0f64.max(f_raw_min).max(f_t_min).min(top);
    let grid = if bottom < top {
        hypreq::fft::geomspace(bottom, top, 1200)
    } else {
        vec![bottom]
    };

    let raw_db = hypreq::pchip::pchip_log_interp(&f_raw, &g_raw, &grid);
    let gain_db = realized_gain_db(ir, fs, &grid);
    let predicted: Vec<(f64, f64)> = grid
        .iter()
        .zip(raw_db.iter().zip(&gain_db))
        .map(|(&f, (&r, &g))| (f, r + g - preamp_db))
        .collect();
    let target_db = hypreq::pchip::pchip_log_interp(&f_t, &g_t, &grid);
    let target: Vec<(f64, f64)> = grid.iter().copied().zip(target_db).collect();

    let Some(rep) = hypreq::verify::verify(&post, &predicted, Some(&target), &cfg) else {
        return Err(Error::msg(format!(
            "--verify-with: '{}' does not overlap the model's measured range ({} - {} Hz), so there is nothing to compare",
            path.display(),
            f_raw
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min)
                .round(),
            f_raw
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                .round(),
        )));
    };
    println!();
    for line in hypreq::verify::report_lines(&rep, Some(note)) {
        println!("{line}");
    }
    // The psychoacoustic read-out is only meaningful once a capture exists, so --verify-with is
    // its gate. It is a second opinion on the same scored bands, not a replacement for the pass
    // counts above.
    let ppr_mode = if args.ppr_mode == "in-ear" {
        hypreq::psycho::PprMode::InEar
    } else {
        hypreq::psycho::PprMode::OverEar
    };
    if let Some(summary) = hypreq::verify::psycho_summary(&rep, ppr_mode) {
        println!();
        for line in hypreq::verify::psycho_report_lines(&summary, ppr_mode) {
            println!("{line}");
        }
    }
    if let Some(d) = dir {
        let out = d.join(format!("{}_verification.csv", model.file_prefix));
        std::fs::write(&out, hypreq::verify::csv_text(&rep))
            .map_err(|e| Error::msg(format!("cannot write '{}': {e}", out.display())))?;
        println!("  Written Verify CSV  : {}", out.display());
    }
    Ok(())
}

/// Print what a correction demands of some content. Reached only behind `--assess`.
///
/// `preamp_db` is the number the FIR synthesis actually applied, not `--headroom-db`: the
/// convolver bakes `headroom_db - max_boost` into every tap, so that preamp *is* the gain
/// staging between the filter and the limiter, and quoting the requested headroom instead
/// overstated the level after the filter by the whole max boost.
fn print_assessment(
    ctrl: &[(f64, f64)],
    planar: &[Vec<f64>],
    fs: u32,
    args: &Args,
    preamp_db: f64,
    limiter_on: bool,
) {
    let run = hypreq::assess::Staging {
        preamp_db,
        ceiling_db: args.tp_ceiling,
        limiter_on,
    };
    let d = hypreq::assess::assess(ctrl, planar, fs, args.assess_budget, &run);
    for line in hypreq::assess::report_lines(&d, args.assess_budget, &run) {
        println!("{line}");
    }
}

/// The `--stereo` mode. [`validate_choice`] over [`hypreq::stereo::Mode::CHOICES`] has already run
/// in the argument phase, so the fallback here is a lookup and not a message nobody would see.
fn stereo_mode(args: &Args) -> hypreq::stereo::Mode {
    hypreq::stereo::Mode::parse(&args.stereo).unwrap_or(hypreq::stereo::Mode::Off)
}

/// The knee the per-ear trim is judged by, and whether it is the default rather than the flag's.
///
/// A trim is a correction like any other, so it is doubted by the same policy `--regularize` resolved
/// to — and that policy is derived from the model's own captured curve, not from the mismatch it is
/// about to cancel. Deriving it from the mismatch would let a pair with one enormous narrow
/// difference raise its own knee and negotiate its way out of being corrected, which is the failure
/// [`hypreq::fit::Policy::auto`] was built to avoid.
///
/// `--regularize off` declines to regularize *the curve*. That is not a claim that a difference
/// standing on a single capture point is trustworthy, so the trim keeps the documented default knee
/// and the report says which one was used.
fn trim_policy(model: &Model, args: &Args) -> (hypreq::fit::Policy, bool) {
    let f: Vec<f64> = model.raw_pts.iter().map(|p| p[0]).collect();
    let g: Vec<f64> = model
        .raw_pts
        .iter()
        .map(|p| hypreq::model::point_gain(p))
        .collect();
    match regularize_arg(&args.regularize).unwrap_or(RegularizeArg::Off) {
        RegularizeArg::Auto => (hypreq::fit::Policy::auto(&f, &g), false),
        RegularizeArg::Fixed(knee) => (hypreq::fit::Policy::fixed(knee), false),
        RegularizeArg::Off => (hypreq::fit::Policy::default(), true),
    }
}

/// The ears block, and the two curves to synthesise from when `--stereo split` asked for a pair.
///
/// `report` computes the same analysis and applies nothing: audio for `report` is audio for `off`,
/// which is the point of separating "look" from "do". `split` builds the pair from the curve the
/// single filter would have been built from, so `--regularize` and the notch ceiling are already in
/// it, and the trim is the last thing to touch the centre curve before it is split.
/// The ears block, and — for [`hypreq::stereo::Mode::Split`] — one filter per ear.
///
/// `base` is the shared curve *before* the notch ceiling (see [`FilterStages`]). The trim is applied
/// to that and the ceiling is then applied once per ear, which is what makes an identical-ears model
/// come out identical: `cap(trim(cap(x)))` compressed every positive knot twice, and so `--stereo
/// split` changed even the shared filter, on a model where both ears measure the same.
fn stereo_pair(
    model: &Model,
    args: &Args,
    base: &hypreq::stereo::Curve,
    mode: hypreq::stereo::Mode,
) -> Result<(Vec<String>, Option<hypreq::stereo::CurvePair>), Error> {
    use hypreq::stereo::{self as st, Mode};
    let Some(ears) = model.ears() else {
        if mode == Mode::Split {
            return Err(Error::msg(format!(
                "model '{}' was measured as a single curve, so there is no per-ear correction to make; --stereo report says what the shared filter leaves, --stereo off says nothing",
                model.key
            )));
        }
        return Ok((
            vec![
                "=== Ears ===".to_string(),
                "  Single capture    : this model is one curve, so there is no pair to compare and the filter is shared".to_string(),
            ],
            None,
        ));
    };
    let m = st::analyse(&ears);
    if mode != Mode::Split {
        return Ok((st::report_lines(&m, ears.len(), None), None));
    }

    let (policy, borrowed) = trim_policy(model, args);
    let t = st::trim(&ears, base, &policy, args.stereo_max_skew);
    let (gl, gr) = t.curves(base);
    // The ceiling is still the last word, ear by ear: `safe_cap_db` says what this model's metadata
    // allows where its concha null lives, and a trim that lifted one ear's fill past it would
    // re-open exactly what the cap keeps shut. Where it bites the pair's middle moves, and the
    // report says so — "the centre is held" is otherwise a sentence this code broke.
    let (gl, bit_l) = cap_ear(&gl, args, model)?;
    let (gr, bit_r) = cap_ear(&gr, args, model)?;
    // Per-ear counts for the line that says which side was held, the union for the line that says
    // how far the middle moved.
    let cap_l = bit_l.iter().filter(|b| **b).count();
    let cap_r = bit_r.iter().filter(|b| **b).count();
    let t = t.with_cap_bites(centre_bites(&bit_l, &bit_r));

    let mut lines = st::report_lines(&m, ears.len(), Some(&t));
    lines.push(format!(
        "  Trim knee         : {:.2} dB of deviation{}",
        policy.lambda_db,
        if borrowed {
            " (the default: --regularize off declines to touch the curve, not to trust a one-sample difference)"
        } else {
            ""
        }
    ));
    if cap_l + cap_r > 0 {
        lines.push(format!(
            "  Notch cap         : held {cap_l} left / {cap_r} right knot(s) after the trim; where it bites the centre moves"
        ));
    }
    if args.assess {
        lines.push(
            "  Assessment        : measured on the shared curve; each ear's filter differs from it by the trim above"
                .to_string(),
        );
    }
    Ok((lines, Some((gl, gr))))
}

/// One ear's curve through the notch ceiling, and a mask marking the knots the ceiling moved.
///
/// A mask and not a count because [`stereo_pair`] needs to know *which* knots bit: a knot the
/// ceiling cut back on both ears moves the centre once, and two counts added together would say twice.
fn cap_ear(
    curve: &hypreq::stereo::Curve,
    args: &Args,
    model: &Model,
) -> Result<(hypreq::stereo::Curve, Vec<bool>), Error> {
    let capped = apply_notch_mode(
        curve,
        notch_mode(args),
        model.safe_cap_db,
        model.notch_range_hz,
    )
    .map_err(|e| Error::msg(e.0))?;
    let bit = curve
        .iter()
        .zip(&capped)
        .map(|(before, after)| before.1 != after.1)
        .collect();
    Ok((capped, bit))
}

/// Knots where the notch ceiling cut at least one ear back, from the two [`cap_ear`] masks.
///
/// The per-ear counts answer "which side was held"; this answers "how many knots did the middle of
/// the image move", and the two are not the same sum. A knot the ceiling clipped on both ears leaves
/// the two ear curves equal there, so the centre sits at the ceiling and has moved by one knot's
/// worth of decision, not two. Adding the counts instead reports a knot twice.
fn centre_bites(left: &[bool], right: &[bool]) -> usize {
    left.iter().zip(right).filter(|(l, r)| **l || **r).count()
}

/// `--apodize` / `--no-apodize` as a tri-state over the model's own default.
///
/// Neither flag is `None`, which is what lets a model in the database keep saying `gauss_apodize:
/// false` while an operator overrides it in one invocation.
fn apodize_flag(args: &Args) -> Option<bool> {
    if args.apodize {
        Some(true)
    } else if args.no_apodize {
        Some(false)
    } else {
        None
    }
}

/// `--notch-mode` as a mode.
///
/// No error path here, and no second wording of the rejection: the argument phase ran
/// [`validate_choice`] over [`NotchMode::CHOICES`], which exits with argparse's message and code 2
/// for anything else, and the test `the_choice_list_and_the_variants_cover_each_other` pins that
/// every member of that list names a mode. What is left is a lookup.
fn notch_mode(args: &Args) -> NotchMode {
    NotchMode::parse(&args.notch_mode).unwrap_or(NotchMode::Safe)
}

/// The path as mpv.conf wants it: absolute, forward slashes, escaped drive colon.
///
/// Lexical, like Python's `os.path.abspath`. `canonicalize` is not interchangeable here:
/// it resolves symlinks and fails outright on the convolver this very run is about to
/// write, and the fallback it would force is a *relative* path, which mpv cannot resolve
/// against its own working directory.
#[must_use]
fn mpv_snippet_path(raw: &str) -> String {
    let abs = std::path::absolute(Path::new(raw)).unwrap_or_else(|_| PathBuf::from(raw));
    let s = abs.display().to_string().replace('\\', "/");
    if s.contains(':') {
        s.replace(':', "\\:")
    } else {
        s
    }
}

/// The name `--input` writes to when no `--output` was given: `<dir>/<stem>[_<model>][_
/// tp_<ratio>].wav`, where `ir_tag` is the model prefix or `"bypass"`.
///
/// `--bypass-eq --no-tp-limiter` leaves both tags empty, and the naive name is then the
/// input's own path — the run would overwrite the file it just read. `_out` keeps the
/// default meaningful and the source intact.
#[must_use]
pub fn default_out_path(input: &str, ir_tag: &str, limiter: Option<&str>) -> String {
    let in_path = PathBuf::from(input);
    let suffix = if ir_tag != "bypass" {
        format!("_{ir_tag}")
    } else {
        String::new()
    };
    let lim_tag = limiter.map_or_else(String::new, |r| format!("_tp_{r}"));
    let tag = if suffix.is_empty() && lim_tag.is_empty() {
        "_out".to_string()
    } else {
        format!("{suffix}{lim_tag}")
    };
    let stem = in_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());
    in_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(format!("{stem}{tag}.wav"))
        .to_string_lossy()
        .to_string()
}

/// Which file `--upgrade-json` rewrites, and the path the line above prints for it.
///
/// The reference's `--data-file` default is an absolute path beside the script, so an
/// untouched default is what its message shows as absolute; an explicit `--data-file` is
/// echoed exactly as it was typed. Reproducing that split is the difference between two
/// byte-identical reports and two that differ in one path.
fn upgrade_target(args: &Args) -> (PathBuf, PathBuf) {
    let (data_file, _) = database_paths(args);
    let explicitly_passed = std::env::args_os()
        .any(|a| a == "--data-file" || a.to_str().is_some_and(|s| s.starts_with("--data-file=")));
    if explicitly_passed {
        (args.data_file.clone(), args.data_file.clone())
    } else if data_file == Path::new(hypreq::model::DEFAULT_DATA_FILE) {
        // Never found anywhere: report the path the user would have to act on, in the
        // form that says where that is.
        let abs = std::path::absolute(&data_file).unwrap_or(data_file.clone());
        (abs.clone(), abs)
    } else {
        // Either an explicit argument, verbatim, or the executable's own directory —
        // which is already absolute.
        (data_file.clone(), data_file)
    }
}

/// Where the database and the extra-model directory are actually looked for.
///
/// Both defaults are relative, so they resolve against the working directory. That is right
/// inside a checkout and wrong for an installed binary run from elsewhere, so an untouched
/// default that is not there is retried beside the executable — which is where the loader's
/// own "place measurements.json next to the binary" advice points.
fn database_paths(args: &Args) -> (PathBuf, PathBuf) {
    let default_data = PathBuf::from(hypreq::model::DEFAULT_DATA_FILE);
    let default_models = PathBuf::from(hypreq::model::DEFAULT_MODELS_DIR);
    if args.data_file != default_data || args.data_file.exists() {
        return (args.data_file.clone(), args.models_dir.clone());
    }
    let Some(exe_dir) = std::env::current_exe().ok().and_then(|p| {
        p.parent()
            .map(|d| d.to_path_buf())
            .filter(|d| !d.as_os_str().is_empty())
    }) else {
        return (args.data_file.clone(), args.models_dir.clone());
    };
    let (data, models) = pick_next_to_exe(&exe_dir, &default_models, |p| p.exists());
    if args.models_dir == default_models {
        (data, models)
    } else {
        (data, args.models_dir.clone())
    }
}

/// The exe-directory candidate for a missing default database, split out so the rule can be
/// tested without moving the real files around.
fn pick_next_to_exe(
    exe_dir: &Path,
    default_models: &Path,
    exists: impl Fn(&Path) -> bool,
) -> (PathBuf, PathBuf) {
    let data = exe_dir.join(hypreq::model::DEFAULT_DATA_FILE);
    let models = exe_dir.join(default_models);
    if exists(&data) {
        (data, models)
    } else {
        // Nothing beside the executable either: report the original relative path, because
        // that is the one the user can actually act on.
        (
            PathBuf::from(hypreq::model::DEFAULT_DATA_FILE),
            PathBuf::from(default_models),
        )
    }
}

/// The value behind a `#[arg(default_value_t = …)]` in a parsed command line.
#[cfg(test)]
fn argv(args: &[&str]) -> Args {
    Args::parse_from(std::iter::once("hypreq").chain(args.iter().copied()))
}

/// `{fs / 1000.0:g}k`: `48k`, `44.1k`, `96k`.
#[must_use]
pub fn format_rate_tag(fs: u32) -> String {
    let khz = f64::from(fs) / 1000.0;
    let tag = if (khz - khz.round()).abs() < 1e-9 {
        format!("{}", khz.round() as i64)
    } else {
        format!("{khz}")
    };
    format!("{tag}k")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_leading_dash_is_a_value_not_an_unknown_flag() {
        // Without `allow_negative_numbers` clap reads `--headroom-db -4.25` as an option
        // named "-4.25" and dies before argparse-style validation can run.
        let a = argv(&[
            "--headroom-db",
            "-4.25",
            "--tp-ceiling",
            "-1.0",
            "--tp-lookahead",
            "-0.5",
            "--tp-attack",
            "-1",
            "--tp-release",
            "-2",
            "--tp-link-transient",
            "-0.25",
            "--tp-link-release",
            "-0.75",
            "--out-bits",
            "-16",
            "--taps",
            "-5",
            "--fs",
            "-1",
        ]);
        assert_eq!(a.headroom_db, -4.25);
        assert_eq!(a.tp_ceiling, -1.0);
        assert_eq!(a.tp_lookahead, -0.5);
        assert_eq!(a.tp_attack, -1.0);
        assert_eq!(a.tp_release, -2.0);
        assert_eq!(a.tp_link_transient, -0.25);
        assert_eq!(a.tp_link_release, -0.75);
        // They arrive as values, so the domain checks are what reject them — `run()`
        // turns each of these into the same exit-2 message argparse produces.
        assert!(!OUT_BITS.contains(&a.out_bits));
        assert!(a.taps < 256);
        assert!(fs_hz(&a, 48000).is_err());
    }

    #[test]
    fn a_non_positive_sample_rate_reads_like_the_reference_guard() {
        // hypreq.py has one rate guard: `sample rate must be positive (got {rate})`.
        for bad in ["0", "-1", "-48000"] {
            let a = argv(&["--fs", bad]);
            let err = fs_hz(&a, 48000).unwrap_err().to_string();
            assert_eq!(err, format!("sample rate must be positive (got {bad})"));
        }
        assert_eq!(fs_hz(&argv(&[]), 48000).unwrap(), 48000);
        assert_eq!(fs_hz(&argv(&["--fs", "44100"]), 48000).unwrap(), 44100);
    }

    #[test]
    fn the_default_output_name_never_points_at_the_input() {
        // `--bypass-eq --no-tp-limiter` used to derive the input's own path and overwrite
        // the file it had just read.
        let p = default_out_path("in/jingle.wav", "bypass", None);
        assert_eq!(PathBuf::from(&p), PathBuf::from("in/jingle_out.wav"));
        assert_ne!(PathBuf::from(&p), PathBuf::from("in/jingle.wav"));
        assert_eq!(
            PathBuf::from(default_out_path("in/jingle.wav", "bypass", Some("x16"))),
            PathBuf::from("in/jingle_tp_x16.wav")
        );
        assert_eq!(
            PathBuf::from(default_out_path(
                "in/jingle.wav",
                "dt770_80ohm",
                Some("x16")
            )),
            PathBuf::from("in/jingle_dt770_80ohm_tp_x16.wav")
        );
        assert_eq!(
            PathBuf::from(default_out_path("in/jingle.wav", "dt770_80ohm", None)),
            PathBuf::from("in/jingle_dt770_80ohm.wav")
        );
        // A bare file name still gets a directory: `Path::parent()` is None there.
        assert_eq!(
            default_out_path("jingle.wav", "bypass", None),
            "jingle_out.wav"
        );
    }

    #[test]
    fn aliases_never_shadow_a_loaded_model() {
        // `-H` resolves aliases before plain keys, so an alias that collides with a model
        // key would make that model unreachable under its own name.
        let shipped = filter_aliases(|n| ["jbl770nc", "dt770_80", "ft1pro", "g733"].contains(&n));
        assert_eq!(shipped.len(), MODEL_ALIASES.len());

        let shadowed =
            filter_aliases(|n| ["dt770", "jbl770nc", "dt770_80", "ft1pro", "g733"].contains(&n));
        assert!(
            !shadowed.iter().any(|(a, _)| *a == "dt770"),
            "a model named 'dt770' is unreachable while its alias is active: {shadowed:?}"
        );
        // The other aliases keep working, and the shadowing key itself stays selectable.
        assert!(shadowed.iter().any(|(a, _)| *a == "dt770_80ohm"));
    }

    #[test]
    fn an_untouched_default_database_is_retried_beside_the_executable() {
        let exe = Path::new("/opt/hypreq/bin");
        let models_default = Path::new("models");
        // Installed layout: the database sits next to the binary.
        let (data, models) = pick_next_to_exe(exe, models_default, |p| {
            p == Path::new("/opt/hypreq/bin/measurements.json") || p == Path::new("models")
        });
        assert_eq!(data, Path::new("/opt/hypreq/bin/measurements.json"));
        assert_eq!(models, Path::new("/opt/hypreq/bin/models"));

        // Neither place has one: the message has to name the path the user typed,
        // not a directory they never mentioned.
        let (data, models) = pick_next_to_exe(exe, models_default, |_| false);
        assert_eq!(data, Path::new("measurements.json"));
        assert_eq!(models, models_default);

        // An explicit --data-file is never second-guessed.
        let a = argv(&["--data-file", "custom/db.json"]);
        let (data, _) = database_paths(&a);
        assert_eq!(data, Path::new("custom/db.json"));
    }

    /// The mpv snippet names a path mpv can actually open: absolute, without the file
    /// having to exist yet, and with the drive colon escaped the way the reference does.
    /// Pinned because `canonicalize` looks like the modern spelling and is wrong here — it
    /// refuses a path to a file that has not been written yet.
    #[test]
    fn the_mpv_snippet_names_a_path_that_does_not_exist_yet() {
        let p = mpv_snippet_path("not/written/yet/ir.wav");
        if cfg!(windows) {
            assert!(
                p.contains("\\:") || p.starts_with('/'),
                "must be absolute on Windows, got {p}"
            );
        } else {
            assert!(p.starts_with('/'), "must be absolute, got {p}");
        }
        assert!(
            p.ends_with("/not/written/yet/ir.wav"),
            "must keep the given path, got {p}"
        );
        assert!(
            !p.replace("\\:", "").contains('\\'),
            "no unescaped backslashes may survive, got {p}"
        );

        // A drive letter is escaped, not rewritten: `C:/x` becomes `C\:/x`.
        let win = mpv_snippet_path("C:/eq/ir.wav");
        assert!(
            win.contains("C\\:"),
            "the colon must be escaped for mpv, got {win}"
        );
        // An absolute input is left alone apart from the separator walk.
        let abs = mpv_snippet_path("/tmp/already/absolute.wav");
        if cfg!(windows) {
            assert!(
                abs.ends_with("/tmp/already/absolute.wav"),
                "expected path ending with /tmp/already/absolute.wav, got {abs}"
            );
        } else {
            assert_eq!(abs.replace('\\', "/"), "/tmp/already/absolute.wav");
        }
    }

    #[test]
    fn the_upgrade_line_names_the_path_the_reference_would_name() {
        // An explicit --data-file is echoed exactly as typed, the way the reference
        // formats its own message.
        let a = argv(&["--data-file", "custom/db.json"]);
        let (target, shown) = upgrade_target(&a);
        assert_eq!(target, Path::new("custom/db.json"));
        assert_eq!(shown, Path::new("custom/db.json"));

        // The untouched default is reported as an absolute path, because that is the form
        // the reference's default has and the form its message prints.
        let a = argv(&[]);
        let (target, shown) = upgrade_target(&a);
        assert!(
            shown.is_absolute(),
            "the default must be reported absolute, got {}",
            shown.display()
        );
        assert_eq!(target, shown, "the line names the file that was written");
        assert_eq!(shown.file_name().unwrap(), hypreq::model::DEFAULT_DATA_FILE);
    }

    /// `--regularize` takes a word or a number, so clap cannot check it and this must.
    #[test]
    fn regularize_accepts_the_three_forms_and_rejects_the_rest() {
        assert_eq!(regularize_arg("off").unwrap(), RegularizeArg::Off);
        assert_eq!(regularize_arg("OFF").unwrap(), RegularizeArg::Off);
        assert_eq!(regularize_arg("  auto ").unwrap(), RegularizeArg::Auto);
        assert_eq!(regularize_arg("6").unwrap(), RegularizeArg::Fixed(6.0));
        assert_eq!(
            regularize_arg("-2.5").unwrap(),
            RegularizeArg::Fixed(-2.5),
            "a leading dash is a value here, not a flag"
        );
        // The point of the check: a number that is not a number, and a number that is not a
        // finite one, both become usage errors rather than a silent no-op.
        for bad in ["yes", "5dB", "autoo", "inf", "NaN", "1e999"] {
            assert!(
                regularize_arg(bad).is_err(),
                "'{bad}' should not parse as a knee"
            );
            assert!(
                regularize_arg(bad)
                    .err()
                    .unwrap_or_default()
                    .contains("--regularize"),
                "the message must name the option"
            );
        }
    }

    /// With the flag off, the curve handed to the filter builder is the model's own — the same
    /// bytes the reference program derives. Nothing else in this PR is allowed to move that.
    /// A hand-tuned model with a +12 dB knot in its notch band, on a measurement smooth enough
    /// that a high knee leaves the curve exactly as written.
    fn notch_cap_fixture() -> Model {
        let raw_pts: Vec<hypreq::model::Point> = (0..24)
            .map(|i| {
                // Flat, and therefore reading as no deviation anywhere: the knee alone decides how
                // much of the boost survives, and `|S|²` is exactly 1 at every knot.
                let hz = 20.0 * 10f64.powf(f64::from(i) / 23.0 * 3.0);
                vec![hz, 0.0]
            })
            .collect();
        let ctrl_pts: Vec<(f64, f64)> = raw_pts
            .iter()
            .map(|p| {
                let hz = p[0];
                // Two octaves wide: broad enough that the half-octave smoothing the load is
                // measured with sees almost no deviation, so the knee — not the shape — decides how
                // much of the +12 dB survives.
                let peak = 12.0 * (-2.0 * ((hz / 4000.0).log2() / 2.0).powi(2)).exp();
                (hz, peak)
            })
            .collect();
        Model {
            derivation: None,
            acoustic_class: hypreq::fixture::AcousticClass::Unknown,
            raw_ensemble: Vec::new(),
            distortion: None,
            excursion: None,
            rig_calibration: None,
            fixture: None,
            fixture_conflict_ok: None,
            target_preset: None,
            key: "capfixture".to_string(),
            name: "Cap Fixture".to_string(),
            short_title: "Cap".to_string(),
            file_prefix: "cap_fixture".to_string(),
            rig_info: String::new(),
            target_name: "flat".to_string(),
            target_pts: vec![(20.0, 0.0), (20000.0, 0.0)],
            harm_offset: 0.0,
            gauss_apodize: false,
            hf_bypass_hz: 20000.0,
            hf_taper_start_hz: None,
            subbass_leak_discount_db: None,
            notch_range_hz: Some((3000.0, 6000.0)),
            notch_ref_hz: Some(4000.0),
            safe_cap_db: 1.5,
            autoeq_geq_str: None,
            autoeq_offset_db: None,
            plot_meta: serde_json::Map::new(),
            raw_pts,
            ctrl_pts,
        }
    }

    /// Regularize first, cap second — in that order, every time.
    ///
    /// `--regularize` decides which bands are worth chasing from how well the measurement supports
    /// them. It has no opinion about `safe_cap_db`, which is this model's own ceiling on the band
    /// where a coupler resonance lives. So a knee that trusts everything must still not put a
    /// +12 dB boost through the notch, and the ceiling has to be what stops it: the same fixture
    /// under `--notch-mode original` keeps the boost, which is the only thing that proves the cap
    /// was the one acting here.
    #[test]
    fn the_notch_cap_still_binds_on_a_regularized_curve() {
        let model = notch_cap_fixture();
        let (lo, hi) = model.notch_range_hz.expect("the fixture has a notch band");
        let peak_in_range = |curve: &[(f64, f64)]| -> f64 {
            curve
                .iter()
                .filter(|(hz, _)| *hz >= lo && *hz <= hi)
                .map(|&(_, g)| g)
                .fold(f64::MIN, f64::max)
        };

        let mut args = Args::try_parse_from(["hypreq", "-H", "capfixture", "--regularize", "1000"])
            .expect("parses");
        let (curve, lines) = corrected_curve(&model, &args).expect("a curve");
        assert!(
            lines.iter().any(|l| l.contains("0 of")),
            "a trusting knee should leave the curve alone: {lines:?}"
        );
        let capped = peak_in_range(&curve);
        assert!(
            capped <= model.safe_cap_db + 1e-9,
            "a +12 dB knot in the notch band went through at {capped:.2} dB; the cap is {}",
            model.safe_cap_db
        );
        assert!(
            capped > 0.5,
            "the safe mode should hold the boost at its cap, not zero it: {capped:.2}"
        );

        // With the cap off, the same curve keeps the demand the knots asked for.
        args.notch_mode = "original".to_string();
        let (uncapped, _) = corrected_curve(&model, &args).expect("a curve");
        assert!(
            peak_in_range(&uncapped) > 6.0,
            "`original` should leave the notch alone: {:.2}",
            peak_in_range(&uncapped)
        );

        args.notch_mode = "safe".to_string();
        // Now the part that says which of the two acts last. A distrustful knee shrinks the +12 dB
        // to roughly +6 dB — still well over the ceiling — and the curve comes back *at* the
        // ceiling. Cap first and regularize second, and the same fixture lands near +0.5 dB: the
        // load would be chewing on a knot the cap had already flattened, softening a boost that
        // was no longer there. Two mechanisms, one order, and the number says which.
        args.regularize = "-0.5".to_string();
        let (shrunk_then_capped, lines) = corrected_curve(&model, &args).expect("a curve");
        let peak = peak_in_range(&shrunk_then_capped);
        assert!(
            (peak - model.safe_cap_db).abs() < 0.1,
            "the ceiling should be the last word, at {peak:.3} dB (cap-first would be ~0.54): \
             {lines:?}"
        );
        args.notch_mode = "original".to_string();
        let (shrunk_only, _) = corrected_curve(&model, &args).expect("a curve");
        let uncapped = peak_in_range(&shrunk_only);
        assert!(
            uncapped > 3.0 && uncapped < 11.0,
            "the knee should have softened the demand without the ceiling erasing it: {uncapped:.2}"
        );
    }

    #[test]
    fn regularize_off_hands_back_the_models_own_curve() {
        let db = hypreq::model::load_measurement_database(
            Path::new("measurements.json"),
            Path::new("models"),
        )
        .expect("shipped database loads");
        let model = db
            .models
            .iter()
            .find(|m| m.key == "dt770_80")
            .expect("dt770_80 ships");
        let mut args = Args::try_parse_from(["hypreq", "-H", "dt770_80"]).expect("parses");
        let (curve, lines) = regularized_ctrl(model, &args);
        assert_eq!(curve, model.ctrl_pts, "off must not touch the curve");
        assert!(lines.is_empty(), "off must not print anything: {lines:?}");

        // A knee high enough to trust everything must also leave the curve alone, exactly.
        args.regularize = "1000".to_string();
        let (curve, lines) = regularized_ctrl(model, &args);
        assert_eq!(
            curve, model.ctrl_pts,
            "lambda far above every deviation must be a no-op"
        );
        // And it has to say "nothing moved" rather than report a largest move of +0.00 dB at
        // 0 Hz — a frequency that is not a knot of this or any curve, and the value the running
        // "worst knot" tracker starts at.
        let text = lines.join("\n");
        assert!(text.contains("0 of"), "{text}");
        assert!(
            !text.contains("@ 0 Hz"),
            "a curve that did not move was given a worst knot it does not have: {text}"
        );
        assert!(
            !text.contains("largest"),
            "nothing moved, so nothing is largest: {text}"
        );
        assert!(text.contains("under the knee"), "{text}");

        // A knee low enough to distrust everything walks the curve to 0 dB, and says so.
        args.regularize = "-40".to_string();
        let (curve, lines) = regularized_ctrl(model, &args);
        assert!(
            curve.iter().all(|&(_, g)| g.abs() < 0.01),
            "a hopeless knee should bypass every band: {:?}",
            &curve[..3]
        );
        assert!(
            lines.iter().any(|l| l.contains("Regularization (knee")),
            "{lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("Knots moved")),
            "the report must quantify the change: {lines:?}"
        );
    }

    /// `--assess` is opt-in, and its budget takes a leading dash like every other dB option.
    #[test]
    fn assessment_is_off_until_it_is_asked_for() {
        let plain = Args::try_parse_from(["hypreq", "-H", "hd6xx"]).expect("plain parse");
        assert!(
            !plain.assess,
            "the default report must not grow a new section"
        );
        assert_eq!(plain.assess_budget, hypreq::assess::DEFAULT_BUDGET_DB);

        let on = Args::try_parse_from(["hypreq", "-H", "hd6xx", "--assess"]).expect("--assess");
        assert!(on.assess);

        let tuned = Args::try_parse_from([
            "hypreq",
            "-H",
            "hd6xx",
            "--assess",
            "--assess-budget",
            "-3.5",
        ])
        .expect("a negative budget is a value, not a flag");
        assert!((tuned.assess_budget + 3.5).abs() < 1e-12);
    }

    #[test]
    fn the_rate_tag_follows_numpys_g_format() {
        assert_eq!(format_rate_tag(48000), "48k");
        assert_eq!(format_rate_tag(44100), "44.1k");
        assert_eq!(format_rate_tag(96000), "96k");
        assert_eq!(format_rate_tag(8000), "8k");
    }

    /// The groundwork flags are additive, not a new default: none of them appears in the
    /// report or changes the curve on a plain run, which is what keeps parity byte-identical.
    #[test]
    fn groundwork_flags_are_all_off_by_default() {
        let a = argv(&["-H", "hd6xx"]);
        assert!(a.ensemble_csv.is_empty());
        assert_eq!(a.ensemble_alpha, 0.0);
        assert!(!a.global_fit);
        assert_eq!(a.synth, "fir");
        assert_eq!(a.peq_bands, 8);
        assert_eq!(a.target_family, None);
        assert_eq!(a.target_bass, None);
        assert_eq!(a.target_mid, None);
        assert_eq!(a.target_tilt, None);
        assert!(a.listen_spl.is_none());
        assert!(a.iso226_table.is_none());
    }

    /// The three trims are `Option` rather than `f64` defaulted to `0.0`, and the difference only
    /// shows when `0.0` is *typed*: an unnamed flag must stay `None` so a `--target-preset` can
    /// fill it in, and a typed `0.0` must stay `Some(0.0)` so the preset cannot talk the user out
    /// of a flat shelf. With a value-typed flag both states are the number `0.0` and the preset
    /// always wins.
    #[test]
    fn naming_a_trim_at_zero_is_not_the_same_as_not_naming_it() {
        let unnamed = argv(&["-H", "hd6xx"]);
        assert_eq!(
            (unnamed.target_bass, unnamed.target_mid, unnamed.target_tilt),
            (None, None, None),
            "an unnamed trim must stay unnamed"
        );

        let typed = argv(&[
            "-H",
            "hd6xx",
            "--target-bass",
            "0",
            "--target-mid",
            "0.0",
            "--target-tilt",
            "0.",
        ]);
        assert_eq!(
            (typed.target_bass, typed.target_mid, typed.target_tilt),
            (Some(0.0), Some(0.0), Some(0.0)),
            "a typed zero must arrive as a decision, not as the default"
        );
        assert!(
            target_extension_active(&typed),
            "naming a trim is the request for the parameterized target, even at 0 dB"
        );
    }

    #[test]
    fn the_recorded_target_recipe_yields_in_consent_order() {
        let mut model = notch_cap_fixture();
        model.target_preset = Some(std::path::PathBuf::from("/db/kb.json"));

        let plain = argv(&["-H", "x"]);
        assert_eq!(
            effective_target_preset(&model, &plain),
            Some(Path::new("/db/kb.json")),
            "the recorded default is the whole point of recording it"
        );
        assert!(
            target_extension_active_for(&model, &plain),
            "a model with a recorded recipe runs its recipe on a plain invocation"
        );
        assert!(
            !target_extension_active(&plain),
            "the CLI-only question must stay CLI-only, or this file tests nothing"
        );

        let cli = argv(&["-H", "x", "--target-preset", "cli.json"]);
        assert_eq!(
            effective_target_preset(&model, &cli),
            Some(Path::new("cli.json")),
            "a recipe named on the command line outranks the record"
        );

        let csv = argv(&["-H", "x", "--target-csv", "mine.csv"]);
        assert_eq!(
            effective_target_preset(&model, &csv),
            None,
            "a baseline the user brought is not silently replaced by the database's default"
        );
        assert!(
            target_extension_active_for(&model, &csv),
            "the csv itself still activates the parameterized path"
        );

        let off = argv(&["-H", "x", "--no-target-preset"]);
        assert_eq!(
            effective_target_preset(&model, &off),
            None,
            "the named opt-out is honoured"
        );
        assert!(
            !target_extension_active_for(&model, &off),
            "--no-target-preset must fall back to the plain database path, not just to another recipe"
        );

        let bare = notch_cap_fixture();
        assert_eq!(effective_target_preset(&bare, &plain), None);
        assert!(!target_extension_active_for(&bare, &plain));

        assert!(
            Args::try_parse_from([
                "hypreq",
                "-H",
                "x",
                "--target-preset",
                "a.json",
                "--no-target-preset"
            ])
            .is_err(),
            "demanding a recipe and refusing recipes at once is a usage error, not a tie"
        );
    }

    #[test]
    fn target_extension_moves_bass_without_touching_the_default_path() {
        let model = notch_cap_fixture();
        let plain = argv(&["-H", "hd6xx"]);
        let (base, lines) = corrected_curve(&model, &plain).expect("plain curve");
        assert!(lines.is_empty(), "a plain run must stay silent: {lines:?}");

        let tuned = argv(&["-H", "hd6xx", "--target-bass", "4.0"]);
        let (changed, lines) = corrected_curve(&model, &tuned).expect("tuned curve");
        assert!(
            lines.iter().any(|l| l.contains("Parameterized Target")),
            "the report must say the target was edited: {lines:?}"
        );
        let max_move = base
            .iter()
            .zip(&changed)
            .map(|((_, a), (_, b))| (b - a).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_move > 2.0,
            "a +4 dB bass shelf barely moved the curve: {max_move:.3} dB"
        );
    }

    #[test]
    fn alternative_synth_backends_change_the_curve_and_report_it() {
        let mut model = notch_cap_fixture();
        model.file_prefix = format!("synth_test_{}", std::process::id());
        let base = corrected_curve(&model, &argv(&["-H", "hd6xx"])).unwrap().0;

        for (backend, banner, apo) in [
            ("peq", "Matched PEQ", true),
            ("parallel-iir", "Parallel IIR", false),
        ] {
            let args = argv(&["-H", "hd6xx", "--synth", backend, "--peq-bands", "6"]);
            let (curve, lines) =
                corrected_curve(&model, &args).unwrap_or_else(|e| panic!("{backend}: {e}"));
            let joined = lines.join("\n");
            assert!(joined.contains(banner), "{backend}: {joined}");
            let moved = base
                .iter()
                .zip(&curve)
                .map(|((_, a), (_, b))| (b - a).abs())
                .fold(0.0f64, f64::max);
            assert!(moved > 0.01, "{backend} left the curve unchanged");
            if apo {
                let path = format!("{}_peq.txt", model.file_prefix);
                assert!(
                    std::path::Path::new(&path).exists(),
                    "{path} was not written"
                );
                let content = std::fs::read_to_string(&path).unwrap();
                assert!(
                    content.contains("Preamp:"),
                    "PEQ must export preamp: {content}"
                );
                std::fs::remove_file(path).ok();
            } else {
                assert!(
                    joined.contains("Preamp"),
                    "parallel-iir report must mention Preamp: {joined}"
                );
                let path = format!("{}_parallel_iir.txt", model.file_prefix);
                assert!(
                    std::path::Path::new(&path).exists(),
                    "{path} was not written"
                );
                let content = std::fs::read_to_string(&path).unwrap();
                assert!(
                    content.contains("# baked_preamp_db:") && !content.contains("\npreamp_db:"),
                    "parallel IIR must comment baked preamp, not emit unbaked key: {content}"
                );
                std::fs::remove_file(path).ok();
            }
        }
    }

    #[test]
    fn iso226_table_and_listening_level_edit_the_target() {
        let model = notch_cap_fixture();
        let dir = std::env::temp_dir().join(format!("hypreq-iso226-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let table = dir.join("table.json");
        std::fs::write(
            &table,
            r#"[
                {"freq_hz": 1000.0, "alpha_f": 0.30, "l_u": 0.0, "t_f": 2.4},
                {"freq_hz": 2000.0, "alpha_f": 0.33, "l_u": 0.0, "t_f": 2.4},
                {"freq_hz": 4000.0, "alpha_f": 0.36, "l_u": 0.0, "t_f": 2.4}
            ]"#,
        )
        .unwrap();
        let args = argv(&[
            "-H",
            "hd6xx",
            "--listen-spl",
            "75",
            "--iso226-table",
            table.to_str().unwrap(),
        ]);
        let (curve, lines) = corrected_curve(&model, &args).expect("iso226 curve");
        std::fs::remove_dir_all(&dir).ok();
        assert!(
            lines.iter().any(|l| l.contains("Parameterized Target")),
            "the contour edit must be reported: {lines:?}"
        );
        assert!(curve.iter().all(|p| p.1.is_finite()));
    }

    #[test]
    fn target_preset_and_csv_replace_the_baseline_curve() {
        let model = notch_cap_fixture();
        let dir = std::env::temp_dir().join(format!("hypreq-preset-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let baseline = dir.join("df.csv");
        std::fs::write(&baseline, "20,0\n1000,0\n20000,0\n").unwrap();
        let preset = dir.join("recipe.json");
        std::fs::write(
            &preset,
            serde_json::json!({
                "name": "test_beta", "fixture": "bk5128", "family": "diffuse",
                "baseline_csv": "df.csv",
                "params": {"bass": 6.4, "mid": -1.4, "tilt": 0.0},
                "source": "unit test"
            })
            .to_string(),
        )
        .unwrap();

        let with_preset = argv(&["-H", "hd6xx", "--target-preset", preset.to_str().unwrap()]);
        let (curve, lines) = corrected_curve(&model, &with_preset).expect("preset curve");
        std::fs::remove_dir_all(&dir).ok();
        let joined = lines.join("\n");
        assert!(joined.contains("test_beta"), "{joined}");
        assert!(joined.contains("bk5128"), "{joined}");
        assert!(joined.contains("Family          : diffuse"), "{joined}");
        assert!(curve.iter().all(|p| p.1.is_finite()));
        let bass = curve.iter().min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1;
        assert!(bass > 5.0, "+6.4 dB shelf disappeared: {bass:.3}");

        // A preset and a raw baseline are mutually exclusive in the argument phase; here we only
        // check that --target-csv alone also changes the curve.
        let dir2 = std::env::temp_dir().join(format!("hypreq-targetcsv-{}", std::process::id()));
        std::fs::create_dir_all(&dir2).unwrap();
        let other = dir2.join("other.csv");
        std::fs::write(&other, "20,4\n1000,0\n20000,0\n").unwrap();
        let with_csv = argv(&["-H", "hd6xx", "--target-csv", other.to_str().unwrap()]);
        let (csv_curve, csv_lines) = corrected_curve(&model, &with_csv).expect("csv curve");
        std::fs::remove_dir_all(&dir2).ok();
        assert!(csv_lines.join("\n").contains("replaced by --target-csv"));
        let csv_bass = csv_curve
            .iter()
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap()
            .1;
        assert!(csv_bass < bass, "baseline CSV should carry its own +4 dB, not the preset shelf: {csv_bass:.3} vs {bass:.3}");
    }

    /// A rig recipe publishes shelf constants and the user is allowed to disagree with them —
    /// including by asking for none of them. `--target-bass 0` used to be indistinguishable from
    /// an unnamed flag (both were the number `0.0`), so the recipe's +6.4 dB shelf survived the
    /// override and the only way to get a flat target out of a shelved recipe was to edit its
    /// JSON. Typed zeros now have to agree with a recipe that says zeros, to the bit, and a
    /// single typed zero has to move exactly the one trim it names.
    #[test]
    fn an_explicit_zero_trim_overrides_the_recipe_that_published_it() {
        let model = notch_cap_fixture();
        let dir = std::env::temp_dir().join(format!("hypreq-trimzero-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("flat.csv"), "20,0\n1000,0\n20000,0\n").unwrap();
        let write_preset = |name: &str, bass: f64, mid: f64, tilt: f64| {
            let path = dir.join(format!("{name}.json"));
            std::fs::write(
                &path,
                serde_json::json!({
                    "name": name, "family": "harman", "baseline_csv": "flat.csv",
                    "params": {"bass": bass, "mid": mid, "tilt": tilt},
                    "source": "unit test"
                })
                .to_string(),
            )
            .unwrap();
            path
        };
        let shelved = write_preset("shelved", 6.4, -1.4, 2.0);
        let all_zero = write_preset("allzero", 0.0, 0.0, 0.0);
        let bass_zero = write_preset("basszero", 0.0, -1.4, 2.0);
        let curve = |args: &[&str]| -> Vec<(f64, f64)> {
            corrected_curve(&model, &argv(args))
                .expect("curve")
                .0
                .into_iter()
                .map(|(f, g)| (f, (g * 1e9).round() / 1e9))
                .collect()
        };

        let p = shelved.to_str().unwrap();
        let from_recipe = curve(&["-H", "hd6xx", "--target-preset", p]);
        let typed_zeros = curve(&[
            "-H",
            "hd6xx",
            "--target-preset",
            p,
            "--target-bass",
            "0",
            "--target-mid",
            "0",
            "--target-tilt",
            "0",
        ]);
        let one_zero = curve(&["-H", "hd6xx", "--target-preset", p, "--target-bass", "0"]);
        let like_all_zero = curve(&["-H", "hd6xx", "--target-preset", all_zero.to_str().unwrap()]);
        let like_bass_zero = curve(&[
            "-H",
            "hd6xx",
            "--target-preset",
            bass_zero.to_str().unwrap(),
        ]);
        std::fs::remove_dir_all(&dir).ok();

        // Precondition: the recipe's trims really are in the first curve, or "the override
        // removed them" would be a statement about nothing.
        assert!(
            (from_recipe[0].1 - like_all_zero[0].1).abs() > 3.0,
            "the recipe's +6.4 dB shelf never reached the curve, so this test would prove \
             nothing: 20 Hz {} vs {}",
            from_recipe[0].1,
            like_all_zero[0].1
        );
        assert_eq!(
            typed_zeros, like_all_zero,
            "typing three zeros kept the recipe's shelves"
        );
        assert_eq!(
            one_zero, like_bass_zero,
            "typing one zero moved something other than the bass shelf"
        );
        assert_ne!(
            one_zero, from_recipe,
            "--target-bass 0 changed nothing at all: {one_zero:?}"
        );
    }

    #[test]
    fn rig_calibration_shifts_the_correction_and_feeds_the_qp() {
        let model = notch_cap_fixture();
        let dir = std::env::temp_dir().join(format!("hypreq-rig-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cal = dir.join("cal.json");
        std::fs::write(
            &cal,
            serde_json::json!({
                "source": "gras_43ag", "target": "bk5128", "acoustic_class": "closed_back",
                "f_hz": [20.0, 20000.0],
                "delta_mean_db": [2.0, 2.0],
                "delta_sigma_db": [0.8, 0.8]
            })
            .to_string(),
        )
        .unwrap();

        // A flat, uncapped model isolates the translation: delta_mean = +2 dB means the
        // correction must move by exactly -2 dB everywhere, with no fit or cap in the way.
        let mut flat = model.clone();
        flat.ctrl_pts = vec![(20.0, 0.0), (20000.0, 0.0)];
        flat.notch_range_hz = None;
        flat.safe_cap_db = 12.0;
        let flat_plain = corrected_curve(&flat, &argv(&["-H", "hd6xx"])).unwrap().0;
        let cal_only_args = argv(&["-H", "hd6xx", "--rig-calibration", cal.to_str().unwrap()]);
        let (flat_cal, _) = corrected_curve(&flat, &cal_only_args).expect("translated curve");
        let shift = flat_plain
            .iter()
            .zip(&flat_cal)
            .map(|((_, a), (_, b))| a - b)
            .sum::<f64>()
            / flat_plain.len() as f64;
        assert!(
            (shift - 2.0).abs() < 0.05,
            "delta_mean +2 dB should lower the correction by 2 dB, got {shift:.3}"
        );

        let plain = corrected_curve(&model, &argv(&["-H", "hd6xx"])).unwrap().0;
        let args = argv(&[
            "-H",
            "hd6xx",
            "--rig-calibration",
            cal.to_str().unwrap(),
            "--global-fit",
        ]);
        let (curve, lines) = corrected_curve(&model, &args).expect("calibrated curve");
        std::fs::remove_dir_all(&dir).ok();
        let joined = lines.join("\n");
        assert!(joined.contains("Empirical Rig Translation"), "{joined}");
        assert!(joined.contains("gras_43ag"), "{joined}");
        assert!(
            joined.contains("Global Curvature-Constrained Fit"),
            "{joined}"
        );
        assert!(curve.iter().all(|p| p.1.is_finite()));
        assert!(!plain.is_empty());
    }

    #[test]
    fn rig_calibration_preserves_hf_bypass_flat_wire_guarantee() {
        let mut model = notch_cap_fixture();
        // Set up an explicit derivation where HF bypass is at 10 kHz (taper starts at 8 kHz).
        model.derivation = Some(hypreq::model::Derivation {
            hf_taper_start_hz: 8000.0,
            hf_bypass_hz: 10000.0,
            subbass_leak_discount_db: 0.0,
        });
        model.notch_range_hz = None;
        model.safe_cap_db = 12.0;

        let dir = std::env::temp_dir().join(format!("hypreq-bypass-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cal = dir.join("cal.json");
        std::fs::write(
            &cal,
            serde_json::json!({
                "source": "gras_43ag", "target": "bk5128", "acoustic_class": "closed_back",
                "f_hz": [20.0, 8000.0, 10000.0, 20000.0],
                "delta_mean_db": [1.0, 2.0, 5.0, 8.0],
                "delta_sigma_db": [0.5, 0.5, 0.5, 0.5]
            })
            .to_string(),
        )
        .unwrap();

        let args = argv(&["-H", "hd6xx", "--rig-calibration", cal.to_str().unwrap()]);
        let (curve, _) = corrected_curve(&model, &args).expect("calibrated curve");
        std::fs::remove_dir_all(&dir).ok();

        // Above bypass_hz (10 kHz), the derived correction curve MUST be 0.0 dB (flat wire).
        // Under the old bug, delta_mean (+5 to +8 dB) was subtracted *after* derivation, injecting
        // -5 to -8 dB into the HF bypass zone.
        for &(hz, gain) in &curve {
            if hz >= 10000.0 {
                assert!(
                    gain.abs() < 1e-6,
                    "at {hz} Hz (>= bypass 10 kHz), gain was {gain} dB instead of 0.0 dB flat-wire"
                );
            }
        }
    }

    #[test]
    fn fixture_guard_allows_cross_rig_when_rig_calibration_is_active() {
        let mut model = notch_cap_fixture();
        model.fixture = Some(hypreq::fixture::Fixture::GrasKb5000);

        let cal = hypreq::fixture::RigCalibration {
            source: hypreq::fixture::Fixture::GrasKb5000,
            target: hypreq::fixture::Fixture::Bk5128,
            acoustic_class: hypreq::fixture::AcousticClass::Unknown,
            f_hz: vec![20.0, 20000.0],
            delta_mean_db: vec![0.0, 0.0],
            delta_sigma_db: vec![0.5, 0.5],
        };

        // When rig calibration bridges GrasKb5000 -> Bk5128, target_fixture is Bk5128.
        // fixture_guard is bypassed, and rig_calibration_guard validates the pairing!
        let target_fixture = Some(hypreq::fixture::Fixture::Bk5128);
        let res = rig_calibration_guard(&model, target_fixture, &cal, false);
        assert!(
            res.is_ok(),
            "rig_calibration_guard must pass matching cross-rig bridge: {res:?}"
        );
    }

    #[test]
    fn fixture_guard_does_not_reject_target_csv_with_built_in_target_fixture() {
        let mut model = notch_cap_fixture();
        model.fixture = Some(hypreq::fixture::Fixture::GrasKb5000);
        model.target_name = "harman_5128".to_string();
        // When a custom target is used (args.target_csv), target_fixture is None,
        // so fixture_guard must not inherit models.target_fixtures("harman_5128") and fail!
        let target_fixture = None;
        let res = fixture_guard(&model, target_fixture, false, false, false);
        assert!(
            res.is_ok(),
            "custom target with unknown fixture must not be rejected: {res:?}"
        );
    }

    #[test]
    fn model_recorded_fixtureless_target_preset_does_not_inherit_replaced_target_fixture() {
        let mut model = notch_cap_fixture();
        model.fixture = Some(hypreq::fixture::Fixture::GrasKb5000);
        model.target_name = "harman_5128".to_string(); // Replaced target has Bk5128 fixture in target_fixtures

        let dir = std::env::temp_dir().join(format!("hypreq-fixless-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let preset_path = dir.join("custom_preset.json");
        std::fs::write(
            &preset_path,
            r#"{"family": "harman", "params": {"theta_bass": 0.0, "theta_mid": 0.0, "theta_tilt": 0.0}}"#,
        )
        .unwrap();

        model.target_preset = Some(preset_path.clone());

        let mut models = Database::default();
        models
            .target_fixtures
            .insert("harman_5128".to_string(), hypreq::fixture::Fixture::Bk5128);
        models.models.push(model.clone());

        // Resolution in main():
        let args = argv(&["-H", &model.name]);
        let cli_preset_fixture = args
            .target_preset
            .as_ref()
            .and_then(|p| preset_fixture_only(p));
        let recipe_fixture = cli_preset_fixture
            .or_else(|| effective_target_preset(&model, &args).and_then(preset_fixture_only));
        let custom_target = args.target_csv.is_some()
            || (effective_target_preset(&model, &args).is_some() && recipe_fixture.is_none());
        let target_fixture = if custom_target {
            recipe_fixture
        } else {
            recipe_fixture.or_else(|| models.target_fixtures.get(&model.target_name).copied())
        };

        assert_eq!(
            target_fixture, None,
            "fixtureless model-recorded target preset must not inherit replaced target's fixture"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `q_max = L_max - listen_spl - R(f)`. Feeding zeros for `R` was not neutral: it widened the
    /// ceiling by exactly the amount the driver is already sensitive, so a bin where the headphone
    /// is +12 dB above its own 1 kHz reference was allowed the full 6 dB of boost even though it
    /// had no headroom left at all.
    #[test]
    fn the_distortion_ceiling_is_lowered_by_the_models_own_sensitivity() {
        let mut model = notch_cap_fixture();
        // 1 kHz-normalised, and +12 dB up top: R(20 kHz) = 12.
        model.raw_pts = vec![vec![20.0, 0.0], vec![1000.0, 0.0], vec![20000.0, 12.0]];
        // Clean everywhere below 3 % at 94 dB, so L_max = 94 dB across the band.
        model.distortion = Some(hypreq::driver::DistortionProfile {
            sweeps: vec![hypreq::driver::DistortionSweep {
                spl_db: 94.0,
                points: vec![(20.0, 1.0), (1000.0, 1.0), (20000.0, 1.0)],
            }],
        });
        // The fit is asked for +8 dB at the top, well past any ceiling.
        let ctrl = vec![(20.0, 0.0), (1000.0, 0.0), (20000.0, 8.0)];
        let args = argv(&["-H", "hd6xx", "--global-fit", "--listen-spl", "85"]);
        let (q, lines) = global_fit_stage(&ctrl, &model, &args, None, None, &raw_curve(&model))
            .expect("global fit");
        let joined = lines.join("\n");
        assert!(joined.contains("Boost ceiling"), "{joined}");
        let at = |probe: f64| {
            q.iter()
                .min_by(|a, b| (a.0 - probe).abs().total_cmp(&(b.0 - probe).abs()))
                .expect("a grid")
        };
        // 94 - 85 - 12 = -3 dB of headroom at the top, floored at 0: no boost at all.
        let top = at(20_000.0);
        assert!(
            top.0 > 19_000.0,
            "the last bin should be at the top of the band"
        );
        assert!(
            top.1 <= 0.05,
            "20 kHz has no headroom once R(f) is counted, got {:+.3} dB",
            top.1
        );
        // Mid-band the ceiling is ~0.5 dB tighter than the 6 dB hard cap, and the fit sits on it
        // (measured: +4.47 dB at 8.0 kHz). With h_cal = 0 the bound there is 9 dB, so the fit runs
        // up to the 6 dB cap and this fails.
        let mid = at(8_000.0).1;
        assert!(
            (3.5..6.0).contains(&mid),
            "8 kHz should sit on its own ceiling (~4.5 dB), not the 6 dB cap, got {mid:+.3}"
        );
    }

    /// A database that records `"excursion"` used to have that record parsed and then dropped: the
    /// `--global-fit` block answered nothing about it. It is answered as a screen now, which makes
    /// two claims worth pinning. The fit must come out identical with the record present or absent
    /// — that equality *is* the "not a bound" claim, and it is what keeps a first-order rule from
    /// turning into a silent sub-bass limiter. And the printed shortfall count and worst bin must be
    /// the numbers an independent recomputation from the delivered curve reports, not a formatted
    /// guess.
    #[test]
    fn a_recorded_excursion_model_is_reported_and_never_binds_the_fit() {
        let mut model = notch_cap_fixture();
        model.excursion = Some(hypreq::driver::ExcursionModel {
            fs_hz: 45.0,
            ref_xmax_spl_db: 96.0,
        });
        let args = argv(&["-H", "hd6xx", "--global-fit", "--listen-spl", "95"]);
        let (q, lines) = global_fit_stage(
            &model.ctrl_pts,
            &model,
            &args,
            None,
            None,
            &raw_curve(&model),
        )
        .expect("fit");
        let joined = lines.join("\n");
        assert!(joined.contains("Excursion screen"), "{joined}");
        assert!(joined.contains("not a bound"), "{joined}");

        let plain = notch_cap_fixture();
        let (solo, solo_lines) = global_fit_stage(
            &plain.ctrl_pts,
            &plain,
            &args,
            None,
            None,
            &raw_curve(&plain),
        )
        .expect("fit");
        assert!(
            !solo_lines.join("\n").contains("Excursion screen"),
            "a model with no excursion record must not print the line"
        );
        assert_eq!(q.len(), solo.len());
        for (i, (a, b)) in q.iter().zip(&solo).enumerate() {
            assert_eq!(
                (a.0, a.1),
                (b.0, b.1),
                "bin {i} moved: the screen bound the fit"
            );
        }

        // The fixture's raw curve is flat, so R(f) = 0 at every bin and the recomputation below is
        // the same arithmetic the report ran, written out here instead of trusted.
        let f: Vec<f64> = q.iter().map(|p| p.0).collect();
        let gains: Vec<f64> = q.iter().map(|p| p.1).collect();
        let zero_cal = vec![0.0; f.len()];
        let margins = model
            .excursion
            .expect("the record this test is about")
            .margins_db(&f, &gains, &zero_cal, 95.0);
        let short = margins.iter().filter(|m| **m < 0.0).count();
        // `min_by` rather than a fold: a fold seeded with `f64::NEG_INFINITY` reports that seed for a
        // free (`min(-inf, x) = -inf`), so an empty selection would masquerade as the worst bin. An
        // empty selection has to be loud here, not plausible.
        let worst = margins
            .iter()
            .copied()
            .filter(|m| m.is_finite())
            .min_by(f64::total_cmp)
            .expect("the fixture must leave at least one finite margin to quote");
        assert!(
            short > 0 && worst < 0.0,
            "the fixture must demand more travel than the record allows ({short}, {worst})"
        );
        assert!(
            joined.contains(&format!("{short} of {} bins", f.len())),
            "the report must count {short} short bins: {joined}"
        );
        assert!(
            joined.contains(&format!("{worst:+.2} dB")),
            "the report must quote a worst margin of {worst:+.4} dB: {joined}"
        );
    }

    /// A baseline CSV with three rows (20 / 1000 / 20000 Hz) is ordinary, and both things added to a
    /// baseline — the 105 Hz shelf and the 2.5 kHz bell, and an ISO 226 contour — live between those
    /// rows. Evaluating them at the baseline's own knots turned them into straight lines and deleted
    /// the bell.

    #[test]
    fn a_three_row_target_baseline_still_carries_the_mid_bell() {
        let model = notch_cap_fixture();
        let args = argv(&["-H", "hd6xx", "--target-mid", "-6"]);
        let params = hypreq::target::TargetParams {
            theta_mid: -6.0,
            ..Default::default()
        };
        let sparse = vec![(20.0f64, 0.0f64), (1000.0, 0.0), (20000.0, 0.0)];
        // The same flat baseline expressed densely. A row count is not a shape, so the derived
        // filter must not care how many rows the file happened to use.
        let sparse_f: Vec<f64> = sparse.iter().map(|p| p.0).collect();
        let no_raw: &[f64] = &[];
        let dense: Vec<(f64, f64)> = target_query_grid(&sparse_f, no_raw)
            .into_iter()
            .map(|f| (f, 0.0))
            .collect();
        let ctx = |baseline: Vec<(f64, f64)>| TargetContext {
            baseline,
            family: hypreq::target::TargetFamily::Harman,
            family_label: "harman".to_string(),
            params,
            preset_info: None,
            preset_name: None,
        };
        let (sparse_curve, lines) =
            derive_with_target(&model, &args, None, &ctx(sparse)).expect("sparse");
        assert!(
            lines.is_empty(),
            "--regularize off must print no block, got {lines:?}"
        );
        let (dense_curve, _) = derive_with_target(&model, &args, None, &ctx(dense)).expect("dense");

        let read = |curve: &[(f64, f64)], hz| {
            let f: Vec<f64> = curve.iter().map(|p| p.0).collect();
            let g: Vec<f64> = curve.iter().map(|p| p.1).collect();
            hypreq::pchip::Pchip::new(&f, &g).eval(hz)
        };
        let dip = read(&sparse_curve, 2500.0);
        assert!(
            dip <= -2.95,
            "the 2.5 kHz shelf survived as {dip:+.2} dB of a requested -6.00 dB"
        );
        let drift = (dip - read(&dense_curve, 2500.0)).abs();
        assert!(drift < 0.05, "row count moved the shelf by {drift:.3} dB");
    }

    /// `--regularize` was computed against the model's shipped knots and then discarded by the target
    /// derivation, so with any `--target-*` flag the knee changed nothing while the report printed
    /// knot counts for a curve that never shipped.
    #[test]
    fn regularize_acts_on_the_target_derived_curve_and_reports_that_curve() {
        let model = notch_cap_fixture();
        // A demand well past the knee, which is where a load-based ceiling has something to say; on
        // a demand inside the knee the correct answer is to change nothing, and this fixture's -6 dB
        // bell measured exactly that (0.0000 dB at a 6 dB knee).
        let plain_args = argv(&["-H", "hd6xx", "--target-mid", "-24"]);
        let reg_args = argv(&["-H", "hd6xx", "--target-mid", "-24", "--regularize", "3"]);
        let ctx = TargetContext {
            baseline: vec![(20.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
            family: hypreq::target::TargetFamily::Harman,
            family_label: "harman".to_string(),
            params: hypreq::target::TargetParams {
                theta_mid: -24.0,
                ..Default::default()
            },
            preset_info: None,
            preset_name: None,
        };
        let (plain, plain_lines) =
            derive_with_target(&model, &plain_args, None, &ctx).expect("plain");
        assert!(
            plain_lines.is_empty(),
            "off must print nothing: {plain_lines:?}"
        );
        let (done, lines) = derive_with_target(&model, &reg_args, None, &ctx).expect("regularized");
        assert!(
            lines.iter().any(|l| l.starts_with("=== Regularization")),
            "no regularization block: {lines:?}"
        );
        let moved = lines
            .iter()
            .find(|l| l.starts_with("  Knots moved"))
            .expect("a knots-moved line");

        // The printed knee effect has to be the effect on the pair of curves that actually exists.
        assert_eq!(
            plain.len(),
            done.len(),
            "the two passes disagree on the grid"
        );
        let worst = plain
            .iter()
            .zip(&done)
            .map(|((_, a), (_, b))| (b - a).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst > 0.05,
            "a 3.00 dB knee did nothing to a -24 dB demand: {worst:.4} dB"
        );
        let printed: f64 = moved
            .split("largest ")
            .nth(1)
            .unwrap_or_else(|| panic!("no largest value in {moved:?}"))
            .split(" dB")
            .next()
            .expect("a dB field")
            .parse()
            .expect("a parsed number");
        assert!(
            (printed.abs() - worst).abs() < 0.005,
            "the report says {printed:+.2} dB, the curves it describes differ by {worst:.4} dB"
        );
    }

    /// The knee is sized from the measurement. Read by its last column, a two-ear model sizes it
    /// from one ear, and the printed knee then describes a curve this run never fitted.
    #[test]
    fn the_regularization_knee_is_sized_from_the_ear_mean_not_from_one_ear() {
        let mut model = notch_cap_fixture();
        // `Policy::auto` sizes the knee from the rough tail of the curve, not from its slope: a
        // smooth tilt in either ear lands on the same floor and tells the two readings apart by
        // nothing (measured: 3.00 dB both ways). So the roughness lives in one ear only — a zigzag
        // on the right, cancelled by its mirror on the left — which leaves the ear mean perfectly
        // smooth and one ear visibly ragged. ±9 dB is the amplitude `fit::AUTO_TAIL_FACTOR` already
        // documents as lifting the knee off its 3 dB floor on this kind of grid; ±3 dB does not.
        model.raw_pts = model
            .raw_pts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let ripple = if i % 2 == 0 { 9.0 } else { -9.0 };
                vec![p[0], -ripple, ripple]
            })
            .collect();
        let args = argv(&["-H", "hd6xx", "--regularize", "auto"]);
        let f: Vec<f64> = model.raw_pts.iter().map(|p| p[0]).collect();
        let mean_g: Vec<f64> = model
            .raw_pts
            .iter()
            .map(|p| hypreq::model::point_gain(p))
            .collect();
        let right_g: Vec<f64> = model.raw_pts.iter().map(|p| p[2]).collect();
        let want = hypreq::fit::Policy::auto(&f, &mean_g).lambda_db;
        let one_ear = hypreq::fit::Policy::auto(&f, &right_g).lambda_db;
        assert!(
            (want - one_ear).abs() > 0.05,
            "the fixture must make the two readings differ: mean {want} vs right {one_ear}"
        );
        let (policy, borrowed) = trim_policy(&model, &args);
        assert!(
            !borrowed,
            "--regularize auto must not report a borrowed knee"
        );
        assert_eq!(
            policy.lambda_db, want,
            "the knee was sized from one ear ({one_ear}), not the ear mean ({want})"
        );
    }

    /// `c * tanh(g / c)` is below `g` for every positive `g`, so passing the already-capped shared
    /// curve into `stereo_pair` and capping each ear again compressed every positive knot in the
    /// notch band twice (measured on the old path: 4.26 dB → 1.4887 → 1.1364) and moved the shared
    /// filter even when the two ears measure identically.
    #[test]
    fn stereo_split_caps_each_ear_once_so_identical_ears_reproduce_the_shared_filter() {
        let mut model = notch_cap_fixture();
        // Per-ear table [freq, left, right, average], both ears identical.
        model.raw_pts = model
            .raw_pts
            .iter()
            .map(|p| vec![p[0], 0.0, 0.0, 0.0])
            .collect();
        model.notch_range_hz = Some((1000.0, 8000.0));
        model.safe_cap_db = 1.5;
        let args = argv(&["-H", "hd6xx", "--stereo", "split", "--notch-mode", "safe"]);
        let stages = corrected_curve_stages(&model, &args, 48000, true).expect("stages");
        let shared = &stages.shared.0;

        // The fixture has to actually exercise the ceiling, or the equality below proves nothing.
        let moved = shared
            .iter()
            .zip(&stages.pre_cap)
            .filter(|((_, a), (_, b))| a != b)
            .count();
        assert!(moved > 0, "the ceiling moved no knot; the fixture is flat");
        let top = shared.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let twice = 1.5f64 * (top / 1.5f64).tanh();
        assert!(
            top - twice > 0.05,
            "a second cap must be distinguishable, else this test cannot fail: {top} vs {twice}"
        );

        let (_, pair) = stereo_pair(&model, &args, &stages.pre_cap, hypreq::stereo::Mode::Split)
            .expect("split");
        let (gl, gr) = pair.expect("a pair from a per-ear model");
        for (ear, curve) in [("left", &gl), ("right", &gr)] {
            assert_eq!(curve.len(), shared.len(), "{ear} ear knot count drifted");
            for (i, ((fh, gh), (fs, gs))) in curve.iter().zip(shared).enumerate() {
                assert_eq!(fh, fs, "{ear} knot {i} frequency");
                assert_eq!(
                    *gh, *gs,
                    "{ear} ear knot {i} at {fh} Hz differs from the shared filter ({gh} vs {gs}) — \
                     the ceiling was applied twice"
                );
            }
        }
    }

    /// The two lines the ceiling writes answer two different questions, so they cannot carry the
    /// same number: "held N left / M right" says which side was cut back, "moved in K knot(s)" says
    /// how many knots of the middle of the image moved. A knot the ceiling clipped on both ears
    /// leaves the ears equal at the ceiling, so it is one knot of movement and two side-holds —
    /// K is the union of the two masks, and `N + M` reports knots twice.
    #[test]
    fn a_knot_the_ceiling_clipped_on_both_ears_moves_the_centre_once() {
        // left, right, bite, expected
        let masks: &[([bool; 4], [bool; 4])] = &[
            ([true, true, false, true], [true, false, true, true]),
            ([true, true, true, true], [true, true, true, true]),
            ([false, false, false, false], [false, false, false, false]),
        ];
        let want = [4usize, 4, 0];
        for ((l, r), w) in masks.iter().zip(want) {
            let held_l = l.iter().filter(|b| **b).count();
            let held_r = r.iter().filter(|b| **b).count();
            let bites = centre_bites(l, r);
            assert_eq!(bites, w, "masks {l:?} {r:?}");
            assert!(
                bites >= held_l.max(held_r),
                "a knot held on either side cannot fail to move the centre: {l:?} {r:?} -> {bites}"
            );
            assert!(
                bites <= held_l + held_r,
                "the union cannot exceed the sum: {l:?} {r:?}"
            );
        }
        // The discriminating case on its own: one shared knot, two side-holds.
        assert_eq!(centre_bites(&[true], &[true]), 1);
        assert_eq!(centre_bites(&[true, false], &[false, true]), 2);
        assert_eq!(centre_bites(&[], &[true, true]), 0);
    }

    /// The same claim at the CLI seam: a model whose two ears differ only by a level offset still
    /// gets its notch band clipped on *both* sides, and the report must not count those knots twice.
    #[test]
    fn the_ears_report_counts_a_shared_ceiling_knot_once_not_twice() {
        let mut model = notch_cap_fixture();
        // [freq, left, right, average]: the ears are 0.4 dB apart in level and identical in shape,
        // so the trim is a pure balance cancel and both curves stay over the ceiling where the
        // shared one was.
        model.raw_pts = model
            .raw_pts
            .iter()
            .map(|p| vec![p[0], 0.2, -0.2, 0.0])
            .collect();
        let args = argv(&["-H", "hd6xx", "--stereo", "split", "--notch-mode", "safe"]);
        let stages = corrected_curve_stages(&model, &args, 48000, true).expect("stages");
        let (lines, pair) =
            stereo_pair(&model, &args, &stages.pre_cap, hypreq::stereo::Mode::Split)
                .expect("split");
        assert!(pair.is_some(), "a per-ear model must yield a pair");
        let text = lines.join("\n");

        let count = |needle: &str| -> usize {
            let tail = text.split(needle).nth(1).unwrap_or_else(|| {
                panic!("no {needle:?} in the ears block; the ceiling bit nowhere:\n{text}")
            });
            tail.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or_else(|_| panic!("no count after {needle:?}:\n{text}"))
        };
        let moved = count("Centre            : moved in ");
        let held_l = count("held ");
        let held_r = count(" left / ");

        // Without both sides biting the fixture proves nothing: one side alone makes union == sum.
        assert!(
            held_l > 0 && held_r > 0,
            "the ceiling held {held_l} left / {held_r} right; the fixture does not exercise both ears:\n{text}"
        );
        assert!(
            moved >= held_l.max(held_r),
            "the centre moved in {moved} knot(s) but the ceiling held {held_l} left / {held_r} \
             right — a knot held on one side moves the centre:\n{text}"
        );
        assert!(
            moved < held_l + held_r,
            "the centre was reported as moving in {moved} knot(s) = {held_l} + {held_r}: the two \
             ears' bites were added instead of unioned, so knots clipped on both sides were \
             counted twice:\n{text}"
        );
    }

    #[test]
    fn ensemble_and_global_fit_flags_run_end_to_end() {
        let model = notch_cap_fixture();
        let dir = std::env::temp_dir().join(format!("hypreq-ensemble-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("placement.csv");
        let mut text = String::new();
        for &(f, g) in &[(20.0, 1.0), (1000.0, 0.0), (4000.0, -5.0), (20000.0, 0.0)] {
            text.push_str(&format!("{f},{g}\n"));
        }
        std::fs::write(&csv, text).unwrap();
        // A spread needs a second placement: one CSV is refused outright now.
        let csv2 = dir.join("second.csv");
        std::fs::write(&csv2, "20,1.6\n1000,0.4\n4000,-4.4\n20000,0.5\n").unwrap();
        let args = argv(&[
            "-H",
            "hd6xx",
            "--ensemble-csv",
            csv.to_str().unwrap(),
            "--ensemble-csv",
            csv2.to_str().unwrap(),
            "--ensemble-alpha",
            "1.5",
            "--global-fit",
        ]);
        let (curve, lines) = corrected_curve(&model, &args).expect("groundwork curve");
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(
            curve.len(),
            hypreq::global_fit::uniform_log_grid(20.0, 20000.0, 48).len()
        );
        let joined = lines.join("\n");
        assert!(joined.contains("Placement Ensemble"), "{joined}");
        assert!(
            joined.contains("Global Curvature-Constrained Fit"),
            "{joined}"
        );
        assert!(curve.iter().all(|p| p.1.is_finite()));
    }

    /// A database may record several placement curves for one model (`raw_ensemble`, inline or as
    /// `raw_csvs`). Those are the same measurement `--ensemble-csv` takes, only already on disk, so
    /// the flag is what overrides them and never the other way round: with no flag the recorded
    /// placements are what the robust fit sees, and the report says where they came from.
    ///
    /// Recorded rows are `[freq, gain]` or `[freq, L, R]`, so this is also where the three-column
    /// form enters through [`hypreq::model::point_gain`]: the run whose third placement carries
    /// `L, R` must land on the run whose third placement was averaged by hand. That is a bit-for-bit
    /// claim about the ear mean, not a tolerance — a reader taking `*p.last()` (the right ear alone)
    /// moves every bin that placement touches.
    #[test]
    fn a_recorded_placement_ensemble_feeds_the_robust_fit_without_any_flag() {
        let placements = |third: Vec<hypreq::model::Point>| {
            vec![
                vec![
                    vec![20.0, 1.0],
                    vec![1000.0, 0.0],
                    vec![4000.0, -5.0],
                    vec![20000.0, 0.0],
                ],
                vec![
                    vec![20.0, 2.0],
                    vec![1000.0, 1.0],
                    vec![4000.0, -3.0],
                    vec![20000.0, 2.0],
                ],
                third,
            ]
        };
        let two_ear = vec![
            vec![20.0, 0.0, 4.0],
            vec![1000.0, 6.0, 10.0],
            vec![4000.0, -8.0, -4.0],
            vec![20000.0, 2.0, 6.0],
        ];
        let hand_averaged = vec![
            vec![20.0, 2.0],
            vec![1000.0, 8.0],
            vec![4000.0, -6.0],
            vec![20000.0, 4.0],
        ];

        let args = argv(&["-H", "hd6xx"]);
        let mut model = notch_cap_fixture();
        model.raw_ensemble = placements(two_ear);
        let (curve, lines) = corrected_curve(&model, &args).expect("recorded ensemble");
        let joined = lines.join("\n");
        assert!(joined.contains("Placement Ensemble"), "{joined}");
        assert!(joined.contains("read from the model database"), "{joined}");
        assert!(joined.contains("3 resampled"), "{joined}");
        assert_eq!(
            curve.len(),
            hypreq::global_fit::uniform_log_grid(20.0, 20000.0, 48).len()
        );
        assert!(curve.iter().all(|p| p.1.is_finite()));

        let mut averaged = notch_cap_fixture();
        averaged.raw_ensemble = placements(hand_averaged);
        let (same, _) = corrected_curve(&averaged, &args).expect("pre-averaged ensemble");
        for (i, (a, b)) in curve.iter().zip(&same).enumerate() {
            assert_eq!(a.0, b.0, "bin {i} frequency drifted");
            assert_eq!(
                a.1, b.1,
                "bin {i} at {} Hz: a [freq, L, R] row must enter as (L + R) / 2, the number the \
                 hand-averaged row carries",
                a.0
            );
        }

        // Same model, ensemble cleared: no block, and a different filter — which is what makes the
        // assertions above about the ensemble rather than about the fixture's own curve.
        let plain = notch_cap_fixture();
        let (solo, solo_lines) = corrected_curve(&plain, &args).expect("no ensemble");
        assert!(
            !solo_lines.join("\n").contains("Placement Ensemble"),
            "an empty raw_ensemble must not print an ensemble block"
        );
        assert!(
            curve
                .iter()
                .zip(&solo)
                .any(|(a, b)| (a.1 - b.1).abs() > 0.5),
            "the recorded placements must move the curve, not pass through"
        );

        // And an explicit --ensemble-csv wins over what the database recorded, saying so.
        let dir = std::env::temp_dir().join(format!("hypreq-rec-ens-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("placement.csv");
        std::fs::write(&csv, "20,7\n1000,3\n4000,-1\n20000,5\n").unwrap();
        // Two placements, because an explicit `--ensemble-csv` with a single usable curve is
        // now refused rather than run as a zero-spread "ensemble".
        let csv2 = dir.join("placement2.csv");
        std::fs::write(&csv2, "20,7.4\n1000,2.6\n4000,-0.6\n20000,5.3\n").unwrap();
        let flagged = argv(&[
            "-H",
            "hd6xx",
            "--ensemble-csv",
            csv.to_str().unwrap(),
            "--ensemble-csv",
            csv2.to_str().unwrap(),
        ]);
        let (_, flag_lines) = corrected_curve(&model, &flagged).expect("flagged ensemble");
        std::fs::remove_dir_all(&dir).ok();
        let flag_joined = flag_lines.join("\n");
        assert!(flag_joined.contains("Placement Ensemble"), "{flag_joined}");
        assert!(
            flag_joined.contains("read from --ensemble-csv"),
            "the flag must replace the recorded placements, not join them: {flag_joined}"
        );
        assert!(flag_joined.contains("2 resampled"), "{flag_joined}");
    }

    /// Placement curves that carry fewer than two usable points are not an error and not an
    /// ensemble: the guard in `ensemble_stage` is what turns them into "this model has no ensemble"
    /// and lets the run fall back to the model's own curve. Without the guard the degenerate list
    /// still counts as a source, `analyze_ensemble` then returns `None` over it, and the whole run
    /// aborts with "could not resample the placement curves" over a model that measures fine.
    #[test]
    fn a_model_recording_unusable_placement_curves_still_derives_its_own_curve() {
        let mut model = notch_cap_fixture();
        model.raw_ensemble = vec![
            Vec::new(),
            vec![vec![1000.0, 3.0]],
            vec![vec![-20.0, 1.0], vec![-1000.0, 2.0]],
            vec![vec![20.0, f64::NAN], vec![1000.0, f64::INFINITY]],
        ];
        let args = argv(&["-H", "hd6xx"]);
        let (curve, lines) = corrected_curve(&model, &args)
            .expect("a model with no usable placement curve must still derive a curve");
        assert!(
            !lines.join("\n").contains("Placement Ensemble"),
            "no usable curve must mean no ensemble block"
        );
        let plain = notch_cap_fixture();
        let (solo, _) = corrected_curve(&plain, &args).expect("fixture curve");
        assert_eq!(curve.len(), solo.len());
        for (i, (a, b)) in curve.iter().zip(&solo).enumerate() {
            assert_eq!(
                (a.0, a.1),
                (b.0, b.1),
                "knot {i} differs from the model's own curve"
            );
        }
    }

    /// The QP runs on 479 bins; a calibration's `delta_sigma` has one value per control knot
    /// (19 to 128 rows). The old `rig_sigma.len() == k` check therefore always lost the
    /// comparison — except by accident, which would have aligned two different axes — and
    /// silently substituted the flat 0.5 dB allowance under a report line that said
    /// "delta_sigma feeds the QP". Measured on this fixture: a 9 dB uncertainty window around
    /// 4 kHz relaxes the delivered boost from +6.0 (the box ceiling) to +1.65; the flat
    /// fallback cannot produce any difference at all, so this test is exactly the one the old
    /// code fails.
    #[test]
    fn calibrated_delta_sigma_survives_the_trip_into_the_qp() {
        let model = notch_cap_fixture();
        let args = argv(&["-H", "hd6xx", "--global-fit"]);
        let rig: Vec<(f64, f64)> = model
            .ctrl_pts
            .iter()
            .map(|(f, _)| {
                (
                    *f,
                    if (f / 4000.0).log2().abs() < 0.5 {
                        9.0
                    } else {
                        0.05
                    },
                )
            })
            .collect();
        let (q_unc, _) = global_fit_stage(
            &model.ctrl_pts,
            &model,
            &args,
            None,
            None,
            &raw_curve(&model),
        )
        .expect("flat rig term");
        let (q_cal, _) = global_fit_stage(
            &model.ctrl_pts,
            &model,
            &args,
            None,
            Some(&rig),
            &raw_curve(&model),
        )
        .expect("calibrated rig term");
        let at = |probe: f64, q: &[(f64, f64)]| {
            q.iter()
                .min_by(|a, b| (a.0 - probe).abs().total_cmp(&(b.0 - probe).abs()))
                .unwrap()
                .1
        };
        let unc = at(4000.0, &q_unc);
        let cal = at(4000.0, &q_cal);
        assert!(
            unc > 5.9,
            "fixture sanity: demand sits on the box, got {unc}"
        );
        assert!(
            unc - cal > 2.0 && cal > 0.5,
            "a 9 dB sigma_rig at the demand changed the fit by {} dB (flat fallback reads 0): \
             {unc} vs {cal}",
            unc - cal
        );
        // Outside the uncertainty window the fit must be untouched.
        assert!(
            (at(2000.0, &q_unc) - at(2000.0, &q_cal)).abs() < 0.2,
            "sigma 0.05 dB elsewhere must not move the 2 kHz demand"
        );
    }

    /// The ceiling is stated in absolute SPL, so its R(f) has to be the curve this run designs
    /// in. A +4 dB rig translation moves the raw curve by +4; the printed `R(f) peak` is the
    /// observable. Measured: `+0.0 dB` before the fix (raw_pts verbatim), `+4.0 dB` after.
    #[test]
    fn the_distortion_ceiling_scores_the_rig_translated_raw_curve() {
        let mut model = notch_cap_fixture();
        model.distortion = Some(hypreq::driver::DistortionProfile {
            sweeps: vec![hypreq::driver::DistortionSweep {
                spl_db: 94.0,
                points: vec![(20.0, 1.0), (1000.0, 1.0), (20000.0, 1.0)],
            }],
        });
        let dir = std::env::temp_dir().join(format!("hypreq-hcal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cal = dir.join("cal.json");
        std::fs::write(
            &cal,
            serde_json::json!({
                "source": "gras_43ag", "target": "bk5128", "acoustic_class": "closed_back",
                "f_hz": [20.0, 20000.0], "delta_mean_db": [4.0, 4.0], "delta_sigma_db": [0.8, 0.8]
            })
            .to_string(),
        )
        .unwrap();
        let peek = |lines: &[String]| -> f64 {
            let l = lines
                .iter()
                .find(|l| l.contains("Boost ceiling"))
                .expect("ceiling line");
            let i = l.find("R(f) peak ").expect("peak quote") + "R(f) peak ".len();
            let tok = l[i..].split(" dB").next().expect("number");
            tok.trim().parse().expect("quoted peak")
        };
        let (_, nocal) = corrected_curve(
            &model,
            &argv(&["-H", "hd6xx", "--global-fit", "--notch-mode", "safe"]),
        )
        .expect("nocal");
        let (_, withcal) = corrected_curve(
            &model,
            &argv(&[
                "-H",
                "hd6xx",
                "--global-fit",
                "--notch-mode",
                "safe",
                "--rig-calibration",
                cal.to_str().unwrap(),
            ]),
        )
        .expect("withcal");
        std::fs::remove_dir_all(&dir).ok();
        assert!(
            peek(&nocal).abs() < 1e-9,
            "uncalibrated run must read the record verbatim, got {}",
            peek(&nocal)
        );
        assert!(
            (peek(&withcal) - 4.0).abs() < 1e-9,
            "the ceiling scored the source rig's R(f), not the translated one: {}",
            peek(&withcal)
        );
    }

    /// Same observable, ensemble side: a model recording placements on a rising raw curve must
    /// have its ceiling scored on the collapsed ensemble, not on the single shipped `raw_pts`.
    #[test]
    fn the_distortion_ceiling_scores_the_ensemble_raw_curve() {
        let mut model = notch_cap_fixture();
        model.distortion = Some(hypreq::driver::DistortionProfile {
            sweeps: vec![hypreq::driver::DistortionSweep {
                spl_db: 94.0,
                points: vec![(20.0, 1.0), (1000.0, 1.0), (20000.0, 1.0)],
            }],
        });
        // Two placements, both +12 dB up top (offset slightly so the median keeps the tilt),
        // while the record's own `raw_pts` stay flat.
        model.raw_ensemble = vec![
            vec![vec![20.0, 0.0], vec![1000.0, 0.0], vec![20000.0, 12.0]],
            vec![vec![20.0, 0.3], vec![1000.0, 0.2], vec![20000.0, 12.2]],
        ];
        let (_, lines) = corrected_curve(
            &model,
            &argv(&["-H", "hd6xx", "--global-fit", "--notch-mode", "safe"]),
        )
        .expect("ensemble fit");
        let l = lines
            .iter()
            .find(|l| l.contains("Boost ceiling"))
            .expect("ceiling line");
        let i = l.find("R(f) peak ").expect("peak quote");
        let v: f64 = l[i + 10..i + 15]
            .trim()
            .replace('+', "")
            .parse()
            .expect("number");
        assert!(
            v > 9.0 && v < 13.0,
            "the ensemble's +12 dB treble must raise the scored R(f) peak, read {v}: {l}"
        );
    }

    /// Three claims about the Equalizer APO export, measured: it must be synthesized at the
    /// sample rate the RUN resolved (the file's rate in WAV mode — the old code refitted at
    /// 48 kHz regardless), from the ceiling-CAPPED curve (the old export shipped the uncapped
    /// +12 dB the model's own `safe_cap_db` forbids), with the convolver's preamp staging
    /// (the old export hardcoded `Preamp: 0.00 dB`).
    #[test]
    fn the_apo_export_follows_the_run_rate_the_ceiling_and_the_preamp_rule() {
        let run = |fs: u32, extra: &[&str]| -> (Vec<(f64, f64)>, String) {
            let mut m = notch_cap_fixture();
            m.file_prefix = format!(
                "apo_{fs}_{}_{}",
                extra.join("-").replace("--", ""),
                std::process::id()
            );
            let mut argv_v = vec!["-H", "hd6xx", "--synth", "peq", "--peq-bands", "4"];
            argv_v.extend_from_slice(extra);
            let (curve, _) = corrected_curve_stages(&m, &argv(&argv_v), fs, true)
                .expect("peq")
                .shared;
            let path = format!("{}_peq.txt", m.file_prefix);
            let text = std::fs::read_to_string(&path).expect("preset written");
            std::fs::remove_file(&path).ok();
            (curve, text)
        };
        let filters = |t: &str| -> String {
            t.lines()
                .filter(|l| l.starts_with("Filter"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let (c44, t44) = run(44100, &["--notch-mode", "safe"]);
        let (c48, t48) = run(48000, &["--notch-mode", "safe"]);
        assert_ne!(
            filters(&t44),
            filters(&t48),
            "the same export at 44.1 kHz and 48 kHz — the biquads ignore the run's rate"
        );
        let _ = (c44, c48);

        // The ceiling must be inside the exported bands, not only inside the FIR: at
        // `--notch-mode original` the 4 kHz demand keeps its +12 dB, at `safe` it is capped
        // to 1.5 dB. (Measured: `original` exports a net +11.9 dB band at the notch centre,
        // `safe` exports its capped shape.)
        let (_, t_off) = run(48000, &["--notch-mode", "original"]);
        assert_ne!(
            filters(&t_off),
            filters(&t48),
            "the exported bands are fitted to the uncapped curve — the preset ships the boost \
             the model's safe_cap_db forbids"
        );

        // Preamp staging mirrors the convolver (`headroom_db - max boost`, ir.rs): the exact
        // delivered curve is the one the export must reserve against.
        let (curve, t) = run(48000, &["--notch-mode", "safe"]);
        let headroom = argv(&["-H", "hd6xx"]).headroom_db;
        let max_boost = curve.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let quoted: f64 = t
            .lines()
            .find(|l| l.starts_with("Preamp: "))
            .unwrap()
            .trim_start_matches("Preamp: ")
            .trim_end_matches(" dB")
            .parse()
            .expect("preamp line");
        assert!(
            quoted <= headroom - max_boost + 0.005,
            "Preamp {quoted} must reserve at least headroom {headroom} - knot max boost {max_boost}"
        );
        assert!(
            quoted < 0.0,
            "a curve that boosts cannot ship 0 dB of preamp: {quoted}"
        );
    }

    /// `--measure-only` is a promise that a measurement pass writes no files; the preset
    /// export used to ignore it. The report line has to say which happened either way.
    #[test]
    fn measure_only_exports_no_preset_but_says_so() {
        let mut m = notch_cap_fixture();
        m.file_prefix = format!("mo_export_{}", std::process::id());
        let path = format!("{}_peq.txt", m.file_prefix);
        std::fs::remove_file(&path).ok();
        let (_, lines) = corrected_curve_stages(
            &m,
            &argv(&["-H", "hd6xx", "--synth", "peq", "--notch-mode", "safe"]),
            48000,
            false,
        )
        .expect("measure-only peq")
        .shared;
        assert!(
            !std::path::Path::new(&path).exists(),
            "{path} written despite write_files = false"
        );
        let line = lines.iter().find(|l| l.contains("APO")).expect("APO line");
        assert!(line.contains("not written"), "{line}");
        let _ = std::fs::remove_file(&path);
    }

    /// `--verify-with` used to read `model.target_pts` even when the run had replaced the
    /// target, scoring the capture against a curve the run never equalised to. With the run's
    /// own target handed in, the written table's `target_db` column moves by the replacement's
    /// exact shape.
    #[test]
    fn verify_with_judges_the_runs_own_target_not_the_shipped_one() {
        let model = notch_cap_fixture(); // shipped target: flat 0 dB, harm_offset 0
        let dir = std::env::temp_dir().join(format!("hypreq-verify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cap = dir.join("post.csv");
        // Capture equal to the predicted response: flat raw + identity filter = 0 dB.
        std::fs::write(&cap, "20,0\n100,0\n1000,0\n10000,0\n20000,0\n").unwrap();
        let args = argv(&["-H", "hd6xx", "--verify-with", cap.to_str().unwrap()]);
        let replaced: Vec<(f64, f64)> = vec![(20.0, 1.0), (1000.0, 2.0), (20000.0, 3.0)];
        let run_to_table = |target: Option<&[(f64, f64)]>| -> Vec<(f64, f64)> {
            run_verification(
                &model,
                None,
                target,
                &[1.0],
                48000,
                0.0,
                &args,
                Some(dir.as_path()),
                "probe",
            )
            .expect("verification");
            let out = dir.join("cap_fixture_verification.csv");
            let text = std::fs::read_to_string(&out).expect("table");
            text.lines()
                .skip(1)
                .filter_map(|l| {
                    let cols: Vec<&str> = l.split(',').collect();
                    Some((cols[1].parse().ok()?, cols[5].parse().ok()?))
                })
                .collect()
        };
        let shipped = run_to_table(None);
        let own = run_to_table(Some(&replaced));
        let at_1k = |t: &[(f64, f64)]| {
            t.iter()
                .min_by(|a, b| (a.0 - 1000.0).abs().total_cmp(&(b.0 - 1000.0).abs()))
                .expect("bands")
                .1
        };
        assert!(
            at_1k(&shipped).abs() < 1e-6,
            "shipped target is flat 0: {}",
            at_1k(&shipped)
        );
        assert!(
            (at_1k(&own) - 2.0).abs() < 0.1,
            "the run's own 1 kHz target (2 dB) never reached the verifier: {}",
            at_1k(&own)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The stage's own doc promises "spread needs at least two curves" — a single curve is an
    /// ensemble with sigma_r = 0 by construction. One recorded placement must fall through to
    /// the single raw curve (`None`); an explicit `--ensemble-csv` that supplied one curve is a
    /// request that cannot be honoured and must say so rather than launder it.
    #[test]
    fn a_single_placement_curve_is_not_an_ensemble() {
        let args = argv(&["-H", "hd6xx"]);
        let one: Vec<hypreq::model::Point> =
            vec![vec![20.0, 0.0], vec![1000.0, 0.5], vec![20000.0, -0.5]];
        let two: Vec<hypreq::model::Point> =
            vec![vec![20.0, 0.2], vec![1000.0, -0.4], vec![20000.0, 0.7]];
        assert!(
            ensemble_stage(&args, std::slice::from_ref(&one))
                .expect("recorded single")
                .is_none(),
            "a single recorded placement entered the ensemble machinery"
        );
        assert!(
            ensemble_stage(&args, &[one.clone(), two.clone()])
                .expect("recorded pair")
                .is_some(),
            "two placements must form an ensemble"
        );
        let dir = std::env::temp_dir().join(format!("hypreq-ens1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f1 = dir.join("one.csv");
        std::fs::write(&f1, "20,0\n1000,0.5\n20000,-0.5\n").unwrap();
        let e = ensemble_stage(
            &argv(&["-H", "hd6xx", "--ensemble-csv", f1.to_str().unwrap()]),
            &[],
        )
        .expect_err("one explicit CSV must be refused, not run as a 1-curve ensemble");
        assert!(e.to_string().contains("at least two"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ensemble_stage_normalises_spl_offsets_and_uses_frequency_overlap() {
        let args = argv(&["-H", "hd6xx"]);
        let base: Vec<hypreq::model::Point> = vec![
            vec![20.0, 2.0],
            vec![1000.0, 0.0],
            vec![4000.0, -3.0],
            vec![18000.0, -5.0],
        ];
        // Identical shape shifted up by +10 dB SPL and extending to 20 kHz.
        let shifted: Vec<hypreq::model::Point> = vec![
            vec![20.0, 12.0],
            vec![1000.0, 10.0],
            vec![4000.0, 7.0],
            vec![18000.0, 5.0],
            // Point extending to 20 kHz whose slope matches PCHIP's endpoint derivative
            // at 18 kHz so the shape on [20, 18000] is identical between both curves:
            vec![20000.0, 4.9200251782271],
        ];
        let out = ensemble_stage(&args, &[base, shifted])
            .expect("valid ensemble")
            .expect("two curves");
        let max_sig = out.sigma.iter().map(|p| p.1).fold(0.0_f64, f64::max);
        assert!(
            max_sig < 1e-6,
            "a pure +10 dB SPL offset must not inflate sigma_r (got {max_sig:.4} dB)"
        );
        let f_hi = out.raw.last().map_or(0.0, |p| p.0);
        assert!(
            f_hi <= 18000.0 + 1e-6,
            "common grid must use the overlap upper bound (18 kHz), got {f_hi:.1} Hz"
        );
    }

    /// `--target-family` used to default to the literal `"harman"`, and the preset-override
    /// check compared the *value* — so naming the flag with the default's own word was
    /// indistinguishable from silence and a diffuse/spatial recipe could never be pulled back.
    #[test]
    fn a_named_target_family_overrides_a_presets_own_family() {
        let dir = std::env::temp_dir().join(format!("hypreq-fam-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("base.csv"), "20,0\n1000,0\n20000,0\n").unwrap();
        let preset = dir.join("p.json");
        std::fs::write(
            &preset,
            serde_json::json!({
                "name": "diffuse-recipe", "baseline_csv": "base.csv", "family": "diffuse",
                "params": {"bass": 0.0, "mid": 0.0, "tilt": 0.0}
            })
            .to_string(),
        )
        .unwrap();
        let model = notch_cap_fixture();
        let ctx = target_context(
            &model,
            &argv(&["-H", "hd6xx", "--target-preset", preset.to_str().unwrap()]),
        )
        .expect("preset ctx");
        assert_eq!(ctx.family, hypreq::target::TargetFamily::DiffuseField);
        let ctx = target_context(
            &model,
            &argv(&[
                "-H",
                "hd6xx",
                "--target-preset",
                preset.to_str().unwrap(),
                "--target-family",
                "harman",
            ]),
        )
        .expect("override ctx");
        assert_eq!(
            ctx.family,
            hypreq::target::TargetFamily::Harman,
            "naming the default cannot override a preset's family"
        );
        // Naming any family — even the default word — is a request for the parameterized path.
        assert!(target_extension_active(&argv(&[
            "-H",
            "hd6xx",
            "--target-family",
            "harman"
        ])));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A recipe is data, and recipe failures must travel as errors: `validate_choice` answers
    /// with `process::exit(2)`, which killed the whole test harness from inside
    /// `load_target_preset`; the baseline count was taken before deduplication, so a two-row
    /// file at one frequency passed as a curve; and a `[freq, L, R]` baseline was read as its
    /// right ear.
    #[test]
    fn recipe_defects_fail_as_errors_not_as_process_exits() {
        let dir = std::env::temp_dir().join(format!("hypreq-recipe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, body: &str| {
            let p = dir.join(name);
            std::fs::write(&p, body).unwrap();
            p
        };
        let recipe = |fam: &str, base: &str| -> std::path::PathBuf {
            write(
                "r.json",
                &serde_json::json!({
                    "name": "r", "baseline_csv": base, "family": fam,
                    "params": {"bass": 0.0, "mid": 0.0, "tilt": 0.0}
                })
                .to_string(),
            )
        };
        // (a) An unknown family returns; it does not exit the process (old code: exit(2)
        // straight out of the test binary).
        write("base.csv", "20,0\n1000,0\n20000,0\n");
        let e = load_target_preset(&recipe("bogus", "base.csv")).expect_err("family must fail");
        assert!(e.to_string().contains("bogus"), "{e}");
        // (b) Two rows at one frequency are one knot: refuse, after the dedup the interpolants
        // themselves apply.
        write("dupe.csv", "1000,1\n1000,2\n");
        let e = load_target_preset(&recipe("harman", "dupe.csv")).expect_err("dupe must fail");
        assert!(e.to_string().contains("fewer than two"), "{e}");
        // (c) A three-column [freq, L, R] baseline means both ears, like every other curve
        // in the program — not the right one.
        write("ears.csv", "20,0,2\n1000,4,0\n20000,2,2\n");
        let p = load_target_preset(&recipe("harman", "ears.csv")).expect("ears preset");
        let gains: Vec<f64> = p.baseline.iter().map(|&(_, g)| g).collect();
        assert_eq!(
            gains,
            vec![1.0, 2.0, 2.0],
            "mean of the ears, not the last column"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
/// The prediction half of `--verify-with`, tested on filters whose magnitude is arithmetic.
///
/// The integration tests cannot reach this: their model corrects nothing, so the realised
/// magnitude and the design knots are the same number there and a mutant that returned the knots
/// instead of `|H|` would pass every one of them. These tests are where "we verify against the
/// filter that exists" is actually pinned.
mod realized_tests {
    use super::realized_gain_db;

    fn at(ir: &[f64], fs: u32, freqs: &[f64]) -> Vec<f64> {
        realized_gain_db(ir, fs, freqs)
    }

    #[test]
    fn a_delta_is_unity_gain_at_every_frequency() {
        let f = [20.0, 1000.0, 20000.0];
        let g = at(&[1.0], 48000, &f);
        for (&q, &v) in f.iter().zip(&g) {
            assert!(
                v.abs() < 1e-9,
                "a delta read {v} dB at {q} Hz, expected 0.00"
            );
        }
    }

    #[test]
    fn a_two_tap_average_is_the_cosine_it_should_be() {
        // H(f) = cos(pi f / fs): exactly -3.010 dB at a quarter of the rate, -8.343 dB at
        // three-eighths. Hand-checked arithmetic, not a golden file, so a change in the transform's
        // scaling or bin mapping shows up as a wrong number rather than a moved curve.
        let g = at(&[0.5, 0.5], 48000, &[12000.0, 18000.0]);
        assert!(
            (g[0] + 3.0103).abs() < 1e-3,
            "fs/4 read {} dB, expected -3.010",
            g[0]
        );
        assert!(
            (g[1] + 8.3432).abs() < 1e-2,
            "3fs/8 read {} dB, expected -8.343",
            g[1]
        );
    }

    #[test]
    fn a_null_bin_is_a_huge_number_and_not_a_nan() {
        // The two-tap average is exactly zero at Nyquist. The band means downstream add these
        // numbers up, so one NaN would silently unscore every band that touched it.
        let g = at(&[0.5, 0.5], 48000, &[24000.0]);
        assert!(
            g[0].is_finite(),
            "a null read {}, which poisons every band mean that touches it",
            g[0]
        );
        assert!(
            g[0] < -100.0,
            "a null read {} dB, which is not a null",
            g[0]
        );
    }

    #[test]
    fn a_scale_in_the_taps_is_measured_rather_than_assumed_away() {
        // The written convolver carries its preamp, so a filter of gain 2.0 measures +6.02 dB. This
        // is the fact that makes `run_verification` take `preamp_db` back out; if the helper ever
        // normalised it away the subtraction would introduce the very constant it removes.
        let g = at(&[2.0], 48000, &[1000.0]);
        assert!(
            (g[0] - 6.0206).abs() < 1e-3,
            "a gain of 2.0 read {} dB, expected +6.021",
            g[0]
        );
    }

    #[test]
    fn nothing_to_measure_is_declared_rather_than_invented() {
        // An empty filter or a zero sample rate has no magnitude. Reporting 0.00 dB keeps the
        // comparison honest about the shape it does have, and above all stays finite.
        assert_eq!(at(&[], 48000, &[1000.0]), vec![0.0]);
        assert_eq!(at(&[1.0, 2.0], 0, &[1000.0, 2000.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn the_estimate_holds_for_a_long_filter_at_every_band_centre() {
        // A 4096-tap filter is the length the real convolvers use, and the query grid is the band
        // centres. Interpolating a 32,768-point spectrum onto them must not drift.
        let fs = 48000u32;
        let taps = 4096usize;
        // A single delayed tap: flat magnitude by construction, so any drift in the estimate is the
        // estimate's, not the filter's. (A modulated window, which an earlier draft of this test
        // used, has sidebands and is not flat at all.)
        let mut ir = vec![0.0; taps];
        ir[taps / 2] = 0.001;
        let centres: Vec<f64> = (0..31)
            .map(|k| 1000.0 * 10f64.powf((k as f64 - 10.0) / 10.0))
            .collect();
        let g = at(&ir, fs, &centres);
        assert!(
            g.iter().all(|v| v.is_finite()),
            "a long filter produced a non-finite gain"
        );
        // A pure tone delayed is flat in magnitude: 0.001 amplitude is -60.0 dB, everywhere.
        for (&c, &v) in centres.iter().zip(&g) {
            if c < 23000.0 {
                assert!(
                    (v + 60.0).abs() < 0.05,
                    "{c} Hz read {v} dB, expected -60.00"
                );
            }
        }
    }
}

#[cfg(test)]
/// Which models a run would actually put a filter on — the set the fixture guard is allowed to
/// refuse on. Wrong here means refusing a run that touches nothing it complained about, or waving
/// through a pairing it never looked at.
mod guard_tests {
    use super::{figure_title, used_models, Database};
    use hypreq::model::normalize_model_entry;
    use std::collections::BTreeMap;
    use std::path::Path;

    fn db_with(keys: &[(&str, Option<&str>)]) -> Database {
        let mut targets: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        targets.insert(
            "harman_2018".to_string(),
            vec![(20.0, 0.0), (1000.0, 0.0), (10000.0, 0.0)],
        );
        let mut db = Database {
            targets: targets.clone(),
            ..Database::default()
        };
        for (key, fixture) in keys {
            let mut cfg = serde_json::json!({
                "name": key,
                "raw_pts": [[20.0, 0.0], [1000.0, 0.0], [10000.0, 0.0]],
                "ctrl_pts": [[20.0, 0.0], [1000.0, 0.0]],
            });
            if let Some(f) = fixture {
                // The loader requires a declared fixture to be named in the model's name too, so a
                // fixture-bearing test model has to be written the way a real one is.
                let label = hypreq::fixture::Fixture::parse(f)
                    .expect("test fixture")
                    .label();
                cfg["fixture"] = serde_json::json!(f);
                cfg["name"] = serde_json::json!(format!("{key} [{label}]"));
            }
            let m = normalize_model_entry(key, cfg.as_object().unwrap(), &targets, Path::new("."))
                .unwrap();
            db.order.push((*key).to_string());
            db.models.push(m);
        }
        db
    }

    fn names(models: Vec<&hypreq::model::Model>) -> Vec<String> {
        models.iter().map(|m| m.key.clone()).collect()
    }

    #[test]
    fn every_model_runs_in_model_mode_and_one_in_audio_mode() {
        let db = db_with(&[("a", None), ("b", None)]);
        assert_eq!(names(used_models(&db, "all", false, false)), ["a", "b"]);
        // `-H all` with `--input` is the odd case the audio path resolves by taking the first model
        // and printing a note about it. The guard has to read that rule the same way, or it refuses
        // a run over a pairing belonging to a model the run never opens.
        assert_eq!(names(used_models(&db, "all", true, false)), ["a"]);
        assert_eq!(names(used_models(&db, "b", true, false)), ["b"]);
    }

    #[test]
    fn a_run_that_applies_no_filter_has_no_pairing_to_refuse() {
        let db = db_with(&[("a", None), ("b", None)]);
        assert!(used_models(&db, "all", false, true).is_empty());
        assert!(used_models(&db, "a", true, true).is_empty());
        // An unresolvable key falls through to the lookup that reports it; the guard must not
        // invent a refusal for a model that does not exist.
        assert!(used_models(&db, "nosuch", false, false).is_empty());
    }

    #[test]
    fn the_figure_title_carries_the_fixture_the_measurement_came_off() {
        // The figure is the one artifact a user prints and pins next to the bench, and the fixture is
        // the fact most likely to be forgotten about it a year later.
        let db = db_with(&[("g", Some("gras_43ag"))]);
        let title = figure_title(db.get("g").unwrap(), super::NotchMode::Safe);
        assert!(
            title.contains("GRAS 43AG"),
            "the figure's title does not say which rig it assumed: {title}"
        );
        // A model with no notch talks about no notch, whatever the mode was asked for.
        assert!(!title.contains("Notch"), "{title}");
        let mut notched = db.get("g").unwrap().clone();
        notched.notch_range_hz = Some((3000.0, 8000.0));
        assert!(
            figure_title(&notched, super::NotchMode::Safe).ends_with("| Notch: safe"),
            "the notch suffix moved, and the parity harness compares this string"
        );
    }

    #[test]
    fn a_declared_fixture_survives_the_loader_with_its_identity() {
        // If the loader silently dropped a fixture it could not match, every pairing below would read
        // as "unknown" and the guard would go quiet without any test noticing.
        let db = db_with(&[("g", Some("gras_43ag")), ("n", None)]);
        assert_eq!(
            db.get("g").and_then(|m| m.fixture.map(|f| f.key())),
            Some("gras_43ag")
        );
        assert_eq!(db.get("n").and_then(|m| m.fixture), None);
    }
}
