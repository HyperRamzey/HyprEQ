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
    apply_notch_mode, export_audit_csv, l1_gain_db, mpv_n_fft, synthesize_minimum_phase_ir,
    NotchMode,
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
    for m in used_models(&models, &selected, args.input.is_some(), args.bypass_eq) {
        fixture_guard(&models, m, args.force)?;
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

/// Refuse a measurement being equalised to a target defined on a different test fixture.
///
/// The check only exists when both sides recorded a fixture; a target with no recorded fixture is
/// unknown, not compatible, and an unknown pairing is not refused — refusing on the absence of a
/// claim would punish a database for being honest about what it does not know. `--force` lets a
/// specific pairing through with a line saying so, because a user who re-measured on another rig may
/// well want it, and it is their call once it is spelled out to them.
fn fixture_guard(models: &Database, model: &Model, force: bool) -> Result<(), Error> {
    use hypreq::fixture::{pair, Pairing};
    let target = models.target_fixtures.get(&model.target_name).copied();
    if pair(model.fixture, target) != Pairing::Different {
        return Ok(());
    }
    // A database can record that it inherited a pairing it cannot fix — which is the honest state of
    // every AutoEq-derived measurement equalised to a Harman target, the shipped three included.
    // The reason is in the file, so a person reading the database sees the decision; the guard's job
    // is to make an *unrecorded* pairing loud rather than to overrule a recorded one.
    if model.fixture_conflict_ok.is_some() {
        return Ok(());
    }
    let measured = model.fixture.expect("a Different pairing has both sides");
    let defined = target.expect("a Different pairing has both sides");
    let message = format!(
        "'{}' was measured on {} (fixture: {}), but the target '{}' was defined on {} \
         (fixture: {}). Different fixtures present different acoustic transfer impedances, so this \
         filter would be fitting one rig's curve onto another's; a target is not portable between \
         them. Choose a target recorded for {}, record the target's real fixture in \
         'target_fixtures', record the pairing on the model with 'fixture_conflict_ok' and its \
         reason, or pass --force to equalise anyway. Note that a published curve for one fixture has \
         no free equivalent for another, which is why the shipped models carry that record.",
        model.key,
        measured,
        measured.key(),
        model.target_name,
        defined,
        defined.key(),
        measured.key(),
    );
    if force {
        println!("  [Note] {message}");
        Ok(())
    } else {
        usage_error(message)
    }
}

/// `--out-bits` accepts exactly these depths.
const OUT_BITS: [i64; 4] = [16, 24, 32, 64];

/// Reject a value outside a choice list with argparse's wording and exit code.
///
/// Quoted value, quoted choices: that is what a current interpreter prints for a list of
/// strings (`repr(str(choice))`), and it is what the oldest ones printed too. CPython changed
/// the quoting of this message twice and reverted one of the changes, so the quote characters
/// here match some patch levels and not others; `scripts/parity.py --selftest` documents the
/// table and the harness ignores quoting inside this one message, so every interpreter from 3.10
/// on compares equal on content. Do not "fix" the quoting to chase a specific patch release.
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

fn limit_ir(ir: &mut Vec<f64>, fs: u32, lim_cfg: &hypreq::limiter::Cfg, headroom_db: f64) -> f64 {
    if !lim_cfg.enable {
        return 0.0;
    }
    let orig_len = ir.len();
    let mut cfg = *lim_cfg;
    cfg.ceiling_db = lim_cfg.ceiling_db.max(headroom_db).min(0.0);
    let mut lim = Limiter::enabled(cfg, fs, 1);
    let lat = lim.latency_samples();
    let mut ch = vec![ir.clone()];
    {
        let mut refs: Vec<&mut [f64]> = ch.iter_mut().map(|c| c.as_mut_slice()).collect();
        lim.process(refs.as_mut_slice());
    }
    let tail = lim.flush();
    if let Some(t) = tail.first() {
        ch[0].extend_from_slice(t);
    }
    let rep = lim.report();
    if ch[0].len() >= lat + orig_len {
        *ir = ch[0][lat..lat + orig_len].to_vec();
    }
    rep.max_gain_reduction_db
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

    let (active_ctrl, reg_lines) = corrected_curve(model, args)?;

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

    let gr_mpv = limit_ir(&mut mpv.ir, fs, lim_cfg, headroom_db);
    mpv.preamp_db -= gr_mpv;
    write_wav_ieee_float(Path::new(&f64_wav), std::slice::from_ref(&mpv.ir), fs, 64)?;
    if copy_default_ir {
        std::fs::copy(Path::new(&f64_wav), Path::new("ir.wav"))
            .map_err(|e| Error::msg(format!("cannot write 'ir.wav': {e}")))?;
    }

    let ir_mpv_meas = measure_signal(std::slice::from_ref(&mpv.ir), fs, lim_cfg.ratio);
    let ir_l1_db = l1_gain_db(&mpv.ir);

    let apod_desc = if use_apodize {
        "Gaussian Apodizer (>20.2 kHz)"
    } else {
        "Non-Apodizing 0.00 dB HF Wire"
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
        let (lines, _) = stereo_pair(model, args, &active_ctrl, stereo_mode(args))?;
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
        "  IR Peak / Meter   : {:+.3} dBFS sample | {:+.3} dBTP ({}) | {:+.3} dBTP (ITU x4) | ||h||_1 bound: {:+.2} dB",
        ir_mpv_meas.sample_peak_dbfs,
        ir_mpv_meas.true_peak_dbtp,
        lim_cfg.ratio.label(),
        ir_mpv_meas.itu_x4_dbtp,
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
    let gr_rt = limit_ir(&mut rt.ir, fs, lim_cfg, headroom_db);
    rt.preamp_db -= gr_rt;
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
        "  IR Peak / Meter   : {:+.3} dBFS sample | {:+.3} dBTP ({}) | {:+.3} dBTP (ITU x4)",
        ir_rt_meas.sample_peak_dbfs,
        ir_rt_meas.true_peak_dbtp,
        lim_cfg.ratio.label(),
        ir_rt_meas.itu_x4_dbtp
    );
    println!("  f32 vs f64 Error  : {:.2} dBFS RMS", err_rms_dbfs);
    println!("  Written WAVs      : {mono_rt}, {stereo_rt}");
    print_phase_block(&phase_done);

    if args.export_csv {
        let csv_path = export_audit_csv(
            &model.key,
            &model.raw_pts,
            &model.target_pts,
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
        print_assessment(
            &active_ctrl,
            &[],
            fs,
            args,
            mpv.preamp_db,
            lim_cfg.enable,
        );
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
        Some(m) if !args.bypass_eq => Some(corrected_curve(m, args)?),
        _ => None,
    };

    // The ears, asked about the curve the filter is about to be built from — so the block prints
    // under the banner with the regularization lines, and so `split`'s pair is derived from exactly
    // the curve `off` would have synthesized. Computed once, before the banner, because printing it
    // later would put the answer above the question.
    let ears_block = match (model_cfg, regularized.as_ref()) {
        (Some(m), Some((ctrl, _))) if stereo_mode(args) != hypreq::stereo::Mode::Off => {
            Some(stereo_pair(m, args, ctrl, stereo_mode(args))?)
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
    if let Some((_, lines)) = regularized.as_ref() {
        for line in lines {
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

    let mut work = wav_in.planar.clone();
    let orig_n = wav_in.planar.first().map_or(0, |c| c.len());
    let mut ir_tag = "bypass".to_string();
    // The filter the output was actually made with. Verification has to run against this one rather
    // than a re-synthesis of the design, because what the listener hears is the taps, not the knots.
    let mut applied_ir: Option<(Vec<f64>, f64)> = None;

    let eq_inputs = match (model_cfg, regularized) {
        (Some(model), Some((active_ctrl, _))) => Some((model, active_ctrl)),
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
        let mut mpv_filter = if per_ear.is_none() || args.assess {
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
                phase_stage(mpv, fs, "applied", args)?
            } else {
                None
            }
        } else {
            None
        };
        if let Some(mpv) = mpv_filter.as_ref() {
            applied_ir = Some((mpv.ir.clone(), mpv.preamp_db));
            if per_ear.is_none() {
                println!(
                    "  EQ FIR ({})   : {} taps @ {} Hz | Preamp {:+.2} dB (Max boost +{:.2} dB, ||h||_1 {:+.2} dB)",
                    model.file_prefix,
                    comma_sep(mpv.ir.len() as u64),
                    comma_sep(u64::from(fs)),
                    mpv.preamp_db,
                    mpv.max_boost_db,
                    l1_gain_db(&mpv.ir)
                );
            }
            if args.assess {
                print_assessment(
                    &active_ctrl,
                    &wav_in.planar,
                    fs,
                    args,
                    mpv.preamp_db,
                    lim_cfg.enable,
                );
            }
            if per_ear.is_none() {
                if let Some(a) = phase.as_ref() {
                    println!("{}", "-".repeat(100));
                    print_phase_block(&[("applied", a.clone())]);
                }
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
                let ear_phase: Vec<(&str, hypreq::phase::Adjustment)> =
                    if let Some(crossover) = args.experimental_phase {
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
                let channels = work.len();
                print_phase_block(&ear_phase);
                let filter_lat = ear_phase.first().map_or(0, |(_, adj)| adj.added_latency_samples);
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
                        let end = (filter_lat + orig_n).min(ch.len());
                        *ch = ch[filter_lat.min(ch.len())..end].to_vec();
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
                        let end = (filter_lat + orig_n).min(ch.len());
                        *ch = ch[filter_lat.min(ch.len())..end].to_vec();
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
    let g: Vec<f64> = model.raw_pts.iter().map(|p| *p.last().unwrap()).collect();
    let policy = match arg {
        RegularizeArg::Auto => hypreq::fit::Policy::auto(&f, &g),
        RegularizeArg::Fixed(knee) => hypreq::fit::Policy::fixed(knee),
        RegularizeArg::Off => return (model.ctrl_pts.clone(), Vec::new()),
    };
    let done = model.regularize(&policy);
    let mut lines = vec![format!(
        "=== Regularization (knee {:.2} dB of deviation, {}) ===",
        policy.lambda_db,
        done.describe()
    )];
    if done.curve().is_empty() {
        lines.push("  Curve           : unchanged".to_string());
        return (model.ctrl_pts.clone(), lines);
    }
    let curve = done.curve().to_vec();
    // The plain curve read at the regularized knot's frequency: the two share frequencies on both
    // paths (the knots themselves, or the measurement grid the knots were derived on), and a PCHIP
    // passes through its own nodes, so this reads the exact old value. One interpolant for the
    // whole curve — building it inside the loop made the report quadratic in the knot count.
    let plain_f: Vec<f64> = model.ctrl_pts.iter().map(|&(x, _)| x).collect();
    let plain_g: Vec<f64> = model.ctrl_pts.iter().map(|&(_, y)| y).collect();
    let plain_curve = hypreq::pchip::Pchip::new(&plain_f, &plain_g);

    let mut worst_db = 0.0f64;
    let mut worst_hz = 0.0f64;
    let mut moved = 0usize;
    let (mut was_at_worst, mut now_at_worst) = (0.0f64, 0.0f64);
    for &(hz, gain) in &curve {
        let plain = plain_curve.eval(hz);
        let delta = gain - plain;
        if delta.abs() > worst_db.abs() {
            worst_db = delta;
            worst_hz = hz;
            was_at_worst = plain;
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
        return (curve, lines);
    }
    lines.push(format!(
        "  Knots moved     : {moved} of {} (largest {worst_db:+.2} dB @ {worst_hz:.0} Hz)",
        curve.len()
    ));
    lines.push(format!(
        "  Extreme shifted : {was_at_worst:+.2} dB -> {now_at_worst:+.2} dB at that knot"
    ));
    (curve, lines)
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
fn corrected_curve(model: &Model, args: &Args) -> Result<CurveAndReport, Error> {
    let (base_ctrl, reg_lines) = regularized_ctrl(model, args);
    let active_ctrl = apply_notch_mode(
        &base_ctrl,
        notch_mode(args),
        model.safe_cap_db,
        model.notch_range_hz,
    )
    .map_err(|e| Error::msg(e.0))?;
    Ok((active_ctrl, reg_lines))
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
fn print_phase_block(items: &[(&str, hypreq::phase::Adjustment)]) {
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
    println!(
        "  Note                : 'Causal Energy' and the '(after half-Kaiser fade)' floor above are "
    );
    println!("                        the reshaped filter's; '(unwindowed)' still describes the ");
    println!("                        minimum-phase filter before its tail fade.");
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
fn run_verification(
    model: &Model,
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
    let f_raw: Vec<f64> = model.raw_pts.iter().map(|p| p[0]).collect();
    let g_raw: Vec<f64> = model.raw_pts.iter().map(|p| *p.last().unwrap()).collect();
    let f_t: Vec<f64> = model.target_pts.iter().map(|p| p.0).collect();
    let g_t: Vec<f64> = model
        .target_pts
        .iter()
        .map(|p| p.1 + model.harm_offset)
        .collect();

    let f_raw_min = f_raw.iter().cloned().fold(f64::INFINITY, f64::min);
    let f_raw_max = f_raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let f_t_min = f_t.iter().cloned().fold(f64::INFINITY, f64::min);
    let f_t_max = f_t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let top = (f64::from(fs) / 2.0).min(20000.0).min(f_raw_max).min(f_t_max);
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
    let g: Vec<f64> = model.raw_pts.iter().map(|p| *p.last().unwrap()).collect();
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
fn stereo_pair(
    model: &Model,
    args: &Args,
    shared: &hypreq::stereo::Curve,
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
    let t = st::trim(&ears, shared, &policy, args.stereo_max_skew);
    let (gl, gr) = t.curves(shared);
    // The ceiling is still the last word, ear by ear: `safe_cap_db` says what this model's metadata
    // allows where its concha null lives, and a trim that lifted one ear's fill past it would
    // re-open exactly what the cap keeps shut. Where it bites the pair's middle moves, and the
    // report says so — "the centre is held" is otherwise a sentence this code broke.
    let (gl, cap_l) = cap_ear(&gl, args, model)?;
    let (gr, cap_r) = cap_ear(&gr, args, model)?;
    let t = t.with_cap_bites(cap_l + cap_r);

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

/// One ear's curve through the notch ceiling, and how many knots the ceiling moved.
fn cap_ear(
    curve: &hypreq::stereo::Curve,
    args: &Args,
    model: &Model,
) -> Result<(hypreq::stereo::Curve, usize), Error> {
    let capped = apply_notch_mode(
        curve,
        notch_mode(args),
        model.safe_cap_db,
        model.notch_range_hz,
    )
    .map_err(|e| Error::msg(e.0))?;
    let moved = curve
        .iter()
        .zip(&capped)
        .filter(|(before, after)| before.1 != after.1)
        .count();
    Ok((capped, moved))
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
    if data_file == Path::new(hypreq::model::DEFAULT_DATA_FILE) {
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
        assert_eq!(p, "in/jingle_out.wav");
        assert_ne!(PathBuf::from(&p), PathBuf::from("in/jingle.wav"));
        assert_eq!(
            default_out_path("in/jingle.wav", "bypass", Some("x16")),
            "in/jingle_tp_x16.wav"
        );
        assert_eq!(
            default_out_path("in/jingle.wav", "dt770_80ohm", Some("x16")),
            "in/jingle_dt770_80ohm_tp_x16.wav"
        );
        assert_eq!(
            default_out_path("in/jingle.wav", "dt770_80ohm", None),
            "in/jingle_dt770_80ohm.wav"
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
        let shipped = filter_aliases(|n| ["jbl770nc", "dt770_80", "ft1pro"].contains(&n));
        assert_eq!(shipped.len(), MODEL_ALIASES.len());

        let shadowed = filter_aliases(|n| ["dt770", "jbl770nc", "dt770_80", "ft1pro"].contains(&n));
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
        assert!(p.starts_with('/'), "must be absolute, got {p}");
        assert!(
            p.ends_with("/not/written/yet/ir.wav"),
            "must keep the given path, got {p}"
        );
        assert!(!p.contains('\\'), "no backslashes may survive, got {p}");

        // A drive letter is escaped, not rewritten: `C:/x` becomes `C\:/x`.
        let win = mpv_snippet_path("C:/eq/ir.wav");
        assert!(
            win.contains("C\\:"),
            "the colon must be escaped for mpv, got {win}"
        );
        // An absolute input is left alone apart from the separator walk.
        let abs = mpv_snippet_path("/tmp/already/absolute.wav");
        assert_eq!(abs, "/tmp/already/absolute.wav");
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
            fixture: None,
            fixture_conflict_ok: None,
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
