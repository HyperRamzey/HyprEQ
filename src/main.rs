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

use hypreq::convolve::convolve_planar_fft;
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
        let path = upgrade_measurements_json(&args.data_file).map_err(|e| Error::msg(e.0))?;
        println!(
            "Upgraded '{}' with plot_meta and callout annotations.",
            path.display()
        );
        return Ok(());
    }

    // The Python loads the database before building the parser, so an unparseable
    // database is reported before an unknown flag is. A pure measurement run needs no
    // database at all, and must not fail when there is none.
    let needs_models = args.measure_wav.is_none() && !(args.input.is_some() && args.bypass_eq);
    let mut models = Database::default();
    if args.data_file.exists() || needs_models {
        models = load_measurement_database(&args.data_file, &args.models_dir)
            .map_err(|e| Error::msg(e.0))?;
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
    validate_choice(&args.notch_mode, &NOTCH_MODES, "--notch-mode");
    validate_out_bits(args.out_bits);

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

/// `-H`/`--notch-mode` choice lists, in the order argparse prints them.
const NOTCH_MODES: [&str; 3] = ["safe", "original", "zero"];

/// `--out-bits` accepts exactly these depths.
const OUT_BITS: [i64; 4] = [16, 24, 32, 64];

/// Reject a value outside a choice list with argparse's wording and exit code.
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

/// `--out-bits`: the choices are integers, so argparse lists them unquoted.
fn validate_out_bits(bits: i64) {
    if !OUT_BITS.contains(&bits) {
        usage_error(format!(
            "argument --out-bits: invalid choice: '{bits}' (choose from 16, 24, 32, 64)"
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

    let mut title = model.name.clone();
    if model.notch_range_hz.is_some() {
        title = format!("{title} | Notch: {}", notch_mode.as_str());
    }

    let active_ctrl = apply_notch_mode(
        &model.ctrl_pts,
        notch_mode,
        model.safe_cap_db,
        model.notch_range_hz,
    )
    .map_err(|e| Error::msg(e.0))?;

    let f64_wav = format!("{prefix}_f64_{rate_tag}.wav");
    let mono_rt = format!("{prefix}_mastering_mono_{rate_tag}.wav");
    let stereo_rt = format!("{prefix}_mastering_stereo_{rate_tag}.wav");

    let n_fft_mpv = mpv_n_fft(taps);
    let kaiser_beta = 18.0;
    let kaiser_form_db = kaiser_beta / 0.1102 + 8.7;
    let mpv = synthesize_minimum_phase_ir(
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
    println!("=== [1] Offline / mpv 64-Bit Double Convolver ({f64_wav}) ===");
    println!(
        "  Format            : 64-bit IEEE Float (pcm_f64le) | {} Hz | Mono",
        comma_sep(u64::from(fs))
    );
    println!(
        "  FFT / Taps        : {} FFT ({:.4} Hz bin) -> {} taps ({:.1} ms)",
        comma_sep(n_fft_mpv as u64),
        fs as f64 / n_fft_mpv as f64,
        comma_sep(taps as u64),
        taps as f64 / f64::from(fs) * 1000.0
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
    // PipeWire/DAW convolvers actually load.
    let n_fft_rt = 32768;
    let rt = synthesize_minimum_phase_ir(
        &active_ctrl,
        fs,
        n_fft_rt,
        16384,
        2048,
        headroom_db,
        use_apodize,
        16.0,
    )
    .map_err(|e| Error::msg(e.0))?;
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
        "  FFT / Taps        : 32,768 FFT ({:.4} Hz bin) -> 16,384 taps ({:.1} ms, Half-Kaiser beta=16.0)",
        fs as f64 / n_fft_rt as f64,
        16384.0 / fs as f64 * 1000.0
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

    // mpv wants an absolute path with escaped separators. `absolute` is the lexical
    // equivalent of Python's `os.path.abspath`: unlike `canonicalize` it neither touches
    // the disk nor hands back a Windows `\\?\` prefix that mpv cannot open.
    let mut abs_ir_path = std::path::absolute(Path::new(&f64_wav))
        .unwrap_or_else(|_| PathBuf::from(&f64_wav))
        .display()
        .to_string()
        .replace('\\', "/");
    if abs_ir_path.contains(':') {
        abs_ir_path = abs_ir_path.replace(':', "\\:");
    }
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

    let m_in = measure_signal(&wav_in.planar, fs, lim_cfg.ratio);
    print_signal_measurements("Input Audio", &m_in);

    let mut work = wav_in.planar.clone();
    let orig_n = wav_in.planar.first().map_or(0, |c| c.len());
    let mut ir_tag = "bypass".to_string();

    if let Some(model) = model_cfg {
        if !args.bypass_eq {
            let use_apodize = match apodize_flag(args) {
                Some(b) => b,
                None => model.gauss_apodize,
            };
            let notch_mode = notch_mode(args);
            let active_ctrl = apply_notch_mode(
                &model.ctrl_pts,
                notch_mode,
                model.safe_cap_db,
                model.notch_range_hz,
            )
            .map_err(|e| Error::msg(e.0))?;
            let taps = taps_count(args);
            let n_fft_mpv = mpv_n_fft(taps);
            let mpv = synthesize_minimum_phase_ir(
                &active_ctrl,
                fs,
                n_fft_mpv,
                taps,
                taps / 8,
                args.headroom_db,
                use_apodize,
                18.0,
            )
            .map_err(|e| Error::msg(e.0))?;
            let l1_gain = l1_gain_db(&mpv.ir);
            println!(
                "  EQ FIR ({})   : {} taps @ {} Hz | Preamp {:+.2} dB (Max boost +{:.2} dB, ||h||_1 {:+.2} dB)",
                model.file_prefix,
                comma_sep(taps as u64),
                comma_sep(u64::from(fs)),
                mpv.preamp_db,
                mpv.max_boost_db,
                l1_gain
            );
            work = convolve_planar_fft(&work, &mpv.ir, args.trim_tail);
            let m_eq = measure_signal(&work, fs, lim_cfg.ratio);
            print_signal_measurements("Post-EQ (Pre-Limiter)", &m_eq);
            ir_tag = model.file_prefix.clone();
        }
    }

    if args.measure_only {
        return Ok(());
    }

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
    Ok(())
}

/// `--apodize` / `--no-apodize` as a tri-state over the model's own default.
fn apodize_flag(args: &Args) -> Option<bool> {
    if args.apodize {
        Some(true)
    } else if args.no_apodize {
        Some(false)
    } else {
        None
    }
}

/// `--notch-mode`, with argparse's invalid-choice message.
fn notch_mode(args: &Args) -> NotchMode {
    match NotchMode::from_str(&args.notch_mode) {
        Ok(m) => m,
        Err(_) => usage_error(format!(
            "argument --notch-mode: invalid choice: '{}' (choose from 'safe', 'original', 'zero')",
            args.notch_mode
        )),
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
        let s = format!("{khz}");
        s
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
    fn the_rate_tag_follows_numpys_g_format() {
        assert_eq!(format_rate_tag(48000), "48k");
        assert_eq!(format_rate_tag(44100), "44.1k");
        assert_eq!(format_rate_tag(96000), "96k");
        assert_eq!(format_rate_tag(8000), "8k");
    }
}
