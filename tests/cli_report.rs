//! The CLI's report, read back from the binary itself.
//!
//! Unit tests call the functions that build each block, so nothing in them can say whether the
//! blocks arrive in an order a person can read. This runs the real program in a temporary
//! directory — the child's working directory, never the test process's — and reads its stdout.

use std::path::{Path, PathBuf};
use std::process::Command;

fn run(tag: &str, args: &[&str]) -> String {
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "hypreq-cli-{}-{tag}",
        std::process::id(),
        tag = tag.replace(|c: char| !c.is_ascii_alphanumeric(), "-")
    ));
    std::fs::create_dir_all(&dir).expect("temporary directory");
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("measurements.json");
    let out = Command::new(env!("CARGO_BIN_EXE_hypreq"))
        .current_dir(&dir)
        .arg("--data-file")
        .arg(&data)
        .args(args)
        .output()
        .expect("the binary runs");
    let stdout = String::from_utf8(out.stdout).expect("stdout is utf-8");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        out.status.success(),
        "exit {:?}, stderr: {stderr}",
        out.status
    );
    stdout
}

/// `--regularize` names knots and a knee, which mean nothing before the reader knows which
/// headphone and which notch policy they belong to, so the block has to follow the model banner
/// rather than lead the report.
#[test]
fn the_regularization_block_sits_under_the_model_it_describes() {
    let stdout = run(
        "auto",
        &["-H", "dt770_80", "--regularize", "auto", "--no-plot"],
    );
    let banner = stdout.find("MODEL: ").expect("the report names the model");
    let block = stdout
        .find("=== Regularization")
        .expect("`--regularize` says what it did");
    assert!(
        banner < block,
        "the regularization block came before the model banner:\n{stdout}"
    );
    // …and before the first section, so it reads as part of the model's heading and not as a
    // result of the filter that came after it.
    let first = stdout
        .find("=== [1] Offline")
        .expect("the report has its sections");
    assert!(
        block < first,
        "the regularization block came after the convolver section:\n{stdout}"
    );
    // It quantifies what it did instead of asserting that it did something.
    let block_text = &stdout[block..first];
    assert!(
        block_text.contains("Knots moved"),
        "no accounting for the change:\n{block_text}"
    );
    assert!(
        !block_text.contains("@ 0 Hz"),
        "a worst knot of 0 Hz is the tracker's initial value, not a knot:\n{block_text}"
    );
}

/// Run in a temporary directory, keep the files it wrote, and do not insist on success: the
/// refusals are as much part of the contract as the reports.
fn run_keep(tag: &str, args: &[&str]) -> (PathBuf, Option<i32>, String, String) {
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "hypreq-cli-{}-{}",
        std::process::id(),
        tag.replace(|c: char| !c.is_ascii_alphanumeric(), "-")
    ));
    std::fs::create_dir_all(&dir).expect("temporary directory");
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("measurements.json");
    let out = Command::new(env!("CARGO_BIN_EXE_hypreq"))
        .current_dir(&dir)
        .arg("--data-file")
        .arg(&data)
        .args(args)
        .output()
        .expect("the binary runs");
    (
        dir,
        out.status.code(),
        String::from_utf8(out.stdout).expect("stdout is utf-8"),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

fn read_bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// `--stereo report` names a band, a level and a knee, all of which belong to the model whose
/// heading the reader has just read, so it follows the banner rather than leading the report.
#[test]
fn the_ears_block_sits_under_the_model_it_describes() {
    let stdout = run(
        "ears",
        &["-H", "dt770_80", "--stereo", "report", "--no-plot"],
    );
    let banner = stdout.find("MODEL: ").expect("the report names the model");
    let block = stdout
        .find("=== Ears")
        .expect("`--stereo report` says what it found");
    let first = stdout
        .find("=== [1] Offline")
        .expect("the report has its sections");
    assert!(
        banner < block,
        "the ears block came before the model:\n{stdout}"
    );
    assert!(
        block < first,
        "the ears block came after the convolver section:\n{stdout}"
    );

    let text = &stdout[block..first];
    for line in [
        "Level difference",
        "Shape mismatch",
        "Reads as          : mismatch",
        // The scope note is the point of the block on this capture: a sixth of an octave is two
        // samples here, so the reason for the difference is not in the data.
        "Resolution",
        "Feature match",
    ] {
        assert!(text.contains(line), "missing {line}:\n{text}");
    }
    // Quantified, not asserted: the numbers the shipped capture actually produces.
    assert!(
        text.contains("+1.27 dB") && text.contains("-7.72 dB @ 15110 Hz"),
        "the shipped DT 770 pair reads differently than measured:\n{text}"
    );
    // A report is not a change of audio: no trim line without `split`.
    assert!(
        !text.contains("Per-ear trim"),
        "`report` claimed to have built a pair:\n{text}"
    );
}

/// The default path is the parity-pinned one, and `--stereo report` must not disturb its bytes.
#[test]
fn a_default_run_says_nothing_about_the_ears_and_report_writes_the_same_filter() {
    let (dir_off, code, stdout, err) = run_keep("stereo-off", &["-H", "dt770_80", "--no-plot"]);
    assert_eq!(code, Some(0), "stderr: {err}");
    assert!(
        !stdout.contains("Ears"),
        "the default report gained a section:\n{stdout}"
    );

    let (dir_rep, code, _, _) = run_keep(
        "stereo-rep",
        &["-H", "dt770_80", "--stereo", "report", "--no-plot"],
    );
    assert_eq!(code, Some(0));
    for name in [
        "dt770_80ohm_f64_48k.wav",
        "dt770_80ohm_mastering_stereo_48k.wav",
        "ir.wav",
    ] {
        assert_eq!(
            read_bytes(&dir_off.join(name)),
            read_bytes(&dir_rep.join(name)),
            "`--stereo report` changed {name}, which is supposed to be eyes-only"
        );
    }
    let _ = std::fs::remove_dir_all(&dir_off);
    let _ = std::fs::remove_dir_all(&dir_rep);
}

/// A model captured as one curve has no pair to compare, and says so in as many words rather than
/// comparing a curve against itself and printing a flattering zero.
#[test]
fn a_model_measured_as_one_curve_says_so() {
    let stdout = run(
        "mono-ears",
        &["-H", "jbl770nc", "--stereo", "report", "--no-plot"],
    );
    assert!(
        stdout.contains("Single capture"),
        "a mono model did not say it was mono:\n{stdout}"
    );
    assert!(
        !stdout.contains("Shape mismatch"),
        "a single curve was compared against itself:\n{stdout}"
    );
}

/// Three refusals, all before any file is written: `split` has nothing to act on without audio,
/// the ceiling cannot be negative, and a mode outside the vocabulary is argparse's exit 2.
#[test]
fn the_ear_flags_refuse_what_they_cannot_do() {
    let (dir, code, _, err) = run_keep("split-no-audio", &["-H", "dt770_80", "--stereo", "split"]);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        code,
        Some(2),
        "`split` without `--input` should be a usage error: {err}"
    );
    assert!(
        err.contains("--stereo split") && err.contains("needs audio to apply them to"),
        "refused without naming the way out: {err}"
    );
    assert!(
        !dir.exists(),
        "the refusal should have happened before anything was written"
    );

    let (dir, code, _, err) = run_keep(
        "skew-negative",
        &[
            "-H",
            "dt770_80",
            "--stereo",
            "split",
            "--stereo-max-skew",
            "-0.5",
        ],
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(code, Some(2), "a negative ceiling should exit 2: {err}");
    assert!(err.contains("--stereo-max-skew"), "{err}");

    for value in ["both", "OFF", ""] {
        let (dir, code, _, err) = run_keep(
            &format!("stereo-bad-{}", value.len()),
            &["-H", "dt770_80", "--stereo", value, "--no-plot"],
        );
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(code, Some(2), "`{value}` should exit 2: {err}");
        assert!(
            err.contains("argument --stereo: invalid choice:")
                && err.contains("choose from 'off', 'report', 'split'"),
            "`{value}` was rejected with: {err}"
        );
    }
}

/// The audio path, end to end: `report` leaves the file alone, `split` changes it, says which two
/// filters it used, and refuses a file with one channel because there is no right ear to correct.
#[test]
fn report_leaves_the_audio_alone_and_split_changes_it() {
    // Model mode writes the pair of mastering test files, which is the stereo material this test
    // needs without committing a fixture.
    let (dir, code, _, err) = run_keep("audio-seed", &["-H", "dt770_80", "--no-plot"]);
    assert_eq!(code, Some(0), "stderr: {err}");
    let stereo = dir
        .join("dt770_80ohm_mastering_stereo_48k.wav")
        .to_string_lossy()
        .to_string();
    let mono = dir
        .join("dt770_80ohm_mastering_mono_48k.wav")
        .to_string_lossy()
        .to_string();

    let audio = |tag: &str, extra: &[&str]| -> (Option<i32>, String, PathBuf) {
        let (dir, code, stdout, _) = run_keep(
            tag,
            &[
                "-H",
                "dt770_80",
                "--no-plot",
                "--input",
                &stereo,
                "--output",
                "out.wav",
            ]
            .iter()
            .chain(extra.iter())
            .copied()
            .collect::<Vec<_>>(),
        );
        (code, stdout, dir.join("out.wav"))
    };

    let (code_off, out_off, wav_off) = audio("audio-off", &[]);
    assert_eq!(code_off, Some(0));
    assert!(
        !out_off.contains("=== Ears"),
        "the default audio run grew an ears block:\n{out_off}"
    );
    let (code_rep, out_rep, wav_rep) = audio("audio-report", &["--stereo", "report"]);
    assert_eq!(code_rep, Some(0));
    assert_eq!(
        read_bytes(&wav_off),
        read_bytes(&wav_rep),
        "`--stereo report` changed the audio"
    );
    assert!(
        out_rep.contains("=== Ears"),
        "no block in audio mode:\n{out_rep}"
    );

    let (code_split, out_split, wav_split) = audio("audio-split", &["--stereo", "split"]);
    assert_eq!(code_split, Some(0));
    assert_ne!(
        read_bytes(&wav_off),
        read_bytes(&wav_split),
        "`--stereo split` wrote the same bytes as the shared filter"
    );
    assert!(
        out_split.contains("EQ FIR left") && out_split.contains("EQ FIR right"),
        "split did not say which two filters it used:\n{out_split}"
    );
    assert!(
        !out_split.contains("EQ FIR (dt770_80ohm)"),
        "split printed a shared filter it did not apply:\n{out_split}"
    );
    // Pinned from the shipped capture: the trim pushes six left knots and nine right knots back
    // over the model's own notch ceiling, and the ceiling is the last word, so the report has to
    // say the centre moved rather than recite the design's promise.
    assert!(
        out_split.contains("Centre            : moved in 15 knot(s)"),
        "split did not account for the ceiling biting:\n{out_split}"
    );
    assert!(
        out_split.contains("Notch cap         : held 6 left / 9 right knot(s)"),
        "the ceiling's accounting is missing:\n{out_split}"
    );

    // A mono file has no second channel to hand a filter to.
    let (dir, code, _, err) = run_keep(
        "audio-mono",
        &[
            "-H",
            "dt770_80",
            "--no-plot",
            "--input",
            &mono,
            "--output",
            "out.wav",
            "--stereo",
            "split",
        ],
    );
    assert_eq!(
        code,
        Some(1),
        "a mono input should be refused, not corrected: {err}"
    );
    assert!(
        err.contains("needs a stereo file") && err.contains("1 channel"),
        "the refusal did not name the channel count: {err}"
    );

    for d in [
        dir,
        wav_off.parent().unwrap().to_path_buf(),
        wav_rep.parent().unwrap().to_path_buf(),
        wav_split.parent().unwrap().to_path_buf(),
    ] {
        let _ = std::fs::remove_dir_all(&d);
    }
}

/// The default path is the parity-pinned one. A flag nobody passed must not add a line.
#[test]
fn a_default_run_says_nothing_about_regularization() {
    let stdout = run("off", &["-H", "dt770_80", "--no-plot"]);
    assert!(
        !stdout.contains("Regularization"),
        "the default report gained a section:\n{stdout}"
    );
    assert!(
        stdout.contains("MODEL: Beyerdynamic DT 770 Pro"),
        "the report lost its heading:\n{stdout}"
    );
}

/// A choice the reference program would reject is a usage error with argparse's wording and exit
/// code, whatever the library thinks of the string. The case matters: the notch-mode vocabulary
/// is matched exactly, so `SAFE` is not `safe`.
#[test]
fn a_notch_mode_outside_the_vocabulary_is_a_usage_error() {
    for value in ["bogus", "SAFE", "Safe", ""] {
        let dir = std::env::temp_dir().join(format!(
            "hypreq-cli-{}-notch-{}",
            std::process::id(),
            value.len()
        ));
        std::fs::create_dir_all(&dir).expect("temporary directory");
        let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("measurements.json");
        let out = Command::new(env!("CARGO_BIN_EXE_hypreq"))
            .current_dir(&dir)
            .arg("--data-file")
            .arg(&data)
            .args(["-H", "dt770_80", "--no-plot", "--notch-mode", value])
            .output()
            .expect("the binary runs");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(out.status.code(), Some(2), "`{value}` should exit 2");
        let message = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(
            message.contains("argument --notch-mode: invalid choice:")
                && message.contains("choose from 'safe', 'original', 'zero'"),
            "`{value}` was rejected with: {message}"
        );
    }
}

/// Run the program in a directory the test has already populated, and keep it for inspection.
///
/// `run_keep` makes its own scratch directory, which is right for a plain run and useless here:
/// verification needs a capture file and often a synthetic model on disk *before* the process
/// starts. The caller cleans up.
fn run_in(dir: &Path, args: &[&str]) -> (Option<i32>, String, String) {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("measurements.json");
    let out = Command::new(env!("CARGO_BIN_EXE_hypreq"))
        .current_dir(dir)
        .arg("--data-file")
        .arg(&data)
        .args(args)
        .output()
        .expect("the binary runs");
    (
        out.status.code(),
        String::from_utf8(out.stdout).expect("stdout is utf-8"),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "hypreq-verify-{}-{}",
        std::process::id(),
        tag.replace(|c: char| !c.is_ascii_alphanumeric(), "-")
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// A model whose correction is 0 dB at every knot, plus two captures of its own raw curve.
///
/// Verification's prediction is `raw + |H|`, and a test cannot recompute `|H|` without linking an
/// FFT of its own. So rather than guess what the filter does, hand it a filter that provably does
/// nothing: then a capture equal to the raw curve *must* pass, and one carrying a 6 dB error at
/// 1 kHz *must* be blamed on the 1 kHz band. Both directions of the acceptance criterion become
/// exact instead of "within whatever the notch decided today".
fn zero_model(dir: &Path) {
    let models = dir.join("models");
    std::fs::create_dir_all(&models).expect("models directory");
    let raw: Vec<String> = (0..=60)
        .map(|i| {
            let f = 20.0_f64 * 1000.0_f64.powf(f64::from(i) / 60.0);
            format!("[{f:.4},{:.4}]", -2.0 + 4.0 * f.log10() / 3.0)
        })
        .collect();
    std::fs::write(
        models.join("zero.json"),
        format!(
            r#"{{"name":"Zero Correction Test Set","file_prefix":"zero","target":"harman_2018","harm_offset":0.0,"hf_bypass_hz":10000.0,"safe_cap_db":6.0,"raw_pts":[{}],"ctrl_pts":[[20.0,0.0],[1000.0,0.0],[20000.0,0.0]]}}"#,
            raw.join(",")
        ),
    )
    .expect("model file");
    let capture = |bump: f64| -> String {
        (0..=600)
            .map(|i| {
                let f = 20.0_f64 * 1000.0_f64.powf(f64::from(i) / 600.0);
                let g = -2.0 + 4.0 * f.log10() / 3.0
                    // A narrow bump, so it belongs to one band rather than smearing over six.
                    + bump * (-0.5 * ((f / 1000.0_f64).ln() / 0.11).powi(2)).exp();
                format!("{f:.4},{g:.6}")
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    std::fs::write(dir.join("post_exact.csv"), capture(0.0)).expect("capture");
    std::fs::write(dir.join("post_bump.csv"), capture(6.0)).expect("capture");
}

#[test]
fn a_capture_that_matches_the_filter_passes_and_writes_its_table() {
    let dir = scratch("pass");
    zero_model(&dir);
    let (code, stdout, stderr) = run_in(
        &dir,
        &[
            "-H",
            "zero",
            "--no-plot",
            "--models-dir",
            "models",
            "--verify-with",
            "post_exact.csv",
        ],
    );
    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert!(
        stdout.contains("Verdict         : PASS"),
        "a capture that reproduces the filter did not pass:\n{stdout}"
    );
    // Nothing to blame, so nothing is blamed: a report that names a worst band while passing every
    // band is describing a band it did not score.
    assert!(
        !stdout.contains("Worst band"),
        "a clean run named a worst band it should not have:\n{stdout}"
    );
    let table = dir.join("zero_verification.csv");
    assert!(table.exists(), "no per-band table was written");
    let csv = std::fs::read_to_string(&table).expect("table");
    let head = csv.lines().next().expect("header");
    assert!(head.starts_with("band_index,centre_hz"), "{head}");
    // The table and the sentence must agree about how many bands there were, or one of them is
    // counting something the other does not.
    let scored_line = stdout
        .lines()
        .find(|l| l.contains("Scored"))
        .expect("the report says how many bands it scored");
    let of: usize = scored_line
        .split(" of ")
        .nth(1)
        .expect("N of M")
        .split(" band")
        .next()
        .expect("M")
        .parse()
        .expect("M is a number");
    assert_eq!(
        csv.lines().filter(|l| !l.is_empty()).count() - 1,
        of,
        "the table's rows and the report's band count disagree"
    );
    assert!(
        csv.lines()
            .skip(1)
            .all(|l| l.ends_with(",pass") || l.ends_with(",uncovered")),
        "a scored band failed against the filter that does nothing:\n{csv}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_known_capture_error_is_named_by_band_and_by_number() {
    let dir = scratch("bump");
    zero_model(&dir);
    let (code, stdout, stderr) = run_in(
        &dir,
        &[
            "-H",
            "zero",
            "--no-plot",
            "--models-dir",
            "models",
            "--verify-with",
            "post_bump.csv",
        ],
    );
    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert!(
        stdout.contains("FAIL"),
        "a 6 dB error at 1 kHz was not failed:\n{stdout}"
    );
    let worst = stdout
        .lines()
        .find(|l| l.contains("Worst band"))
        .expect("the report names a worst band");
    assert!(
        worst.contains("1000 Hz") && worst.contains("band 1 kHz"),
        "the 1 kHz error was blamed elsewhere: {worst}"
    );
    // The number, not just the place: the injected error is 6.00 dB.
    let reported: f64 = worst
        .split(':')
        .next_back()
        .unwrap()
        .trim()
        .trim_start_matches('+')
        .split(' ')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        (reported - 6.0).abs() < 0.5,
        "a 6.00 dB error was reported as {reported} dB: {worst}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Stereo material for the audio tests, produced rather than committed: model mode writes the
/// mastering test pair, and the repository keeps no WAVs in git.
fn stereo_bytes(tag: &str) -> Vec<u8> {
    // The tag is the caller's: these tests run in parallel, and a shared scratch name means one
    // deletes the seed directory while another is still reading it out of.
    let (dir, code, _, err) = run_keep(&format!("stereo-{tag}"), &["-H", "dt770_80", "--no-plot"]);
    assert_eq!(code, Some(0), "stderr: {err}");
    let bytes = read_bytes(&dir.join("dt770_80ohm_mastering_stereo_48k.wav"));
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

#[test]
fn verifying_changes_no_audio_and_says_what_it_compared() {
    let seed = stereo_bytes("verify-audio");
    // Both runs get the same model, the same `-H`, the same everything except the flag under test.
    // An earlier version compared a default-model run with a `zero`-model run and "found" that
    // verification changes the audio, which was the model difference talking.
    let plain = scratch("audio-plain");
    std::fs::write(plain.join("in.wav"), &seed).expect("fixture");
    zero_model(&plain);
    let (c1, o1, e1) = run_in(
        &plain,
        &[
            "--input",
            "in.wav",
            "--no-plot",
            "--output",
            "a.wav",
            "-H",
            "zero",
            "--models-dir",
            "models",
        ],
    );
    assert_eq!(c1, Some(0), "stderr: {e1}");
    let before = read_bytes(&plain.join("a.wav"));
    assert!(
        !o1.contains("Verification (post-EQ"),
        "a run without the flag verified something:\n{o1}"
    );

    let v = scratch("audio-verify");
    std::fs::write(v.join("in.wav"), &seed).expect("fixture");
    zero_model(&v);
    let (c2, o2, e2) = run_in(
        &v,
        &[
            "--input",
            "in.wav",
            "--no-plot",
            "--output",
            "a.wav",
            "-H",
            "zero",
            "--models-dir",
            "models",
            "--verify-with",
            "post_exact.csv",
        ],
    );
    assert_eq!(c2, Some(0), "stderr: {e2}");
    assert_eq!(
        read_bytes(&v.join("a.wav")),
        before,
        "--verify-with changed the audio it only reports on"
    );
    assert!(
        o2.contains("Verification (post-EQ") && o2.contains("predicted from the"),
        "the verification block or its provenance line is missing:\n{o2}"
    );
    assert!(
        v.join("zero_verification.csv").exists(),
        "no table in audio mode"
    );
    let _ = std::fs::remove_dir_all(&plain);
    let _ = std::fs::remove_dir_all(&v);
}

#[test]
fn verification_refuses_before_writing_a_single_file() {
    let seed = stereo_bytes("verify-refuse");
    for (tag, args) in [
        (
            "bypass",
            vec![
                "-H",
                "dt770_80",
                "--no-plot",
                "--bypass-eq",
                "--verify-with",
            ],
        ),
        (
            "split",
            vec![
                "--input",
                "in.wav",
                "--no-plot",
                "--stereo",
                "split",
                "--verify-with",
            ],
        ),
    ] {
        let dir = scratch(tag);
        std::fs::write(dir.join("in.wav"), &seed).expect("fixture");
        std::fs::write(dir.join("cap.csv"), "1000.0,0.0\n2000.0,0.0\n").expect("capture");
        let mut argv = args.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        argv.push("cap.csv".to_string());
        let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
        let (code, stdout, stderr) = run_in(&dir, &refs);
        assert_eq!(code, Some(2), "`{tag}` was not a usage error:\n{stdout}");
        assert!(
            stderr.contains("--verify-with"),
            "`{tag}` was refused for some other reason: {stderr}"
        );
        assert_eq!(
            std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0),
            2,
            "`{tag}` wrote files before refusing"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn a_capture_that_cannot_be_read_fails_before_any_convolver_is_written() {
    let dir = scratch("badpath");
    let (code, stdout, stderr) = run_in(
        &dir,
        &[
            "-H",
            "dt770_80",
            "--no-plot",
            "--verify-with",
            "nowhere.csv",
        ],
    );
    assert_eq!(
        code,
        Some(1),
        "a missing capture is not a usage error:\n{stdout}"
    );
    assert!(
        stderr.contains("nowhere.csv"),
        "the error did not name the file: {stderr}"
    );
    assert_eq!(
        std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0),
        0,
        "a whole convolver set was written for a capture path that does not exist"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn verification_reports_nothing_when_it_was_not_asked_for() {
    let stdout = run("verify-absent", &["-H", "dt770_80", "--no-plot"]);
    assert!(
        !stdout.contains("Verification (post-EQ"),
        "the default report grew a verification section:\n{stdout}"
    );
    assert!(
        !stdout.contains("Written Verify CSV"),
        "the default report wrote a table nobody asked for:\n{stdout}"
    );
}

/// The preamp is a constant the written filter carries; the design shortfall is a claim about
/// spectrum. If the constant leaked into the claim, every "the design fell short by" number would
/// follow the headroom knob, and retargeting a convolver's level would look like the headphone's
/// fit had changed. The default report cannot show this — the level offset absorbs any constant —
/// so it is pinned here, at the one place the constant is visible.
#[test]
fn headroom_does_not_move_what_the_verifier_says_about_the_design() {
    let lines_with = |headroom: &str| -> (String, String) {
        let dir = scratch(&format!("headroom-{headroom}"));
        zero_model(&dir);
        let (code, stdout, stderr) = run_in(
            &dir,
            &[
                "-H",
                "zero",
                "--no-plot",
                "--models-dir",
                "models",
                "--verify-with",
                "post_exact.csv",
                "--headroom-db",
                headroom,
            ],
        );
        assert_eq!(code, Some(0), "stderr: {stderr}");
        let grab = |needle: &str| {
            stdout
                .lines()
                .find(|l| l.contains(needle))
                .unwrap_or_default()
                .to_string()
        };
        let out = (grab("Design shortfall"), grab("Total Preamp"));
        let _ = std::fs::remove_dir_all(&dir);
        out
    };
    let (short_a, pre_a) = lines_with("-1.5");
    let (short_b, pre_b) = lines_with("-9.5");
    // Both halves are load-bearing: a test that only compared the two shortfall lines would pass
    // vacuously if the knob did nothing, or if the line were missing in both runs.
    assert!(
        short_a.contains("Design shortfall") && short_b.contains("Design shortfall"),
        "no shortfall line was printed to compare:\n{short_a:?}\n{short_b:?}"
    );
    assert_ne!(
        pre_a, pre_b,
        "the headroom knob did not move the preamp, so this test proves nothing: {pre_a}"
    );
    assert_eq!(
        short_a, short_b,
        "the design shortfall followed the headroom knob:\n{short_a}\n{short_b}"
    );
}

/// A database holding the three pairings that must come out differently.
///
/// The shipped database cannot show any of this. Its models record a fixture that conflicts with the
/// Harman target's recorded fixture, and each carries a written reason for shipping that way, so
/// every shipped run walks straight past the refusal. Testing the refusal means owning a database.
fn fixture_db(dir: &Path) -> PathBuf {
    let models = [
        // A 5128 measurement against a target defined on a 45CA: the pairing the literature names as
        // the mistake, and the one the refusal exists for.
        (
            "conflict",
            "B&K Rig Set [B&K 5128]",
            "bk5128",
            "harman_2018",
        ),
        // A genuine 43AG capture against a target recorded on a 43AG clone: two units of one fixture
        // design, which is not the difference being guarded.
        (
            "adjacent",
            "Clone Set [GRAS 43AG]",
            "gras_43ag",
            "clone_target",
        ),
        // A measurement whose target records no fixture at all: unknown, so nothing to refuse.
        (
            "unrecorded",
            "Unrecorded Set [GRAS 43AG]",
            "gras_43ag",
            "unrecorded_target",
        ),
    ];
    write_fixture_db(dir, "fixtures.json", &models)
}

/// The same shape with only an unparseable fixture entry in it.
///
/// In a file of its own because a bad fixture name kills the whole load — which is what the test
/// below asserts, and which would otherwise make every pairing test in this file fail for a reason
/// unrelated to what it is checking.
fn broken_fixture_db(dir: &Path) -> PathBuf {
    write_fixture_db(
        dir,
        "broken.json",
        &[("broken", "Broken Set", "sm7422", "harman_2018")],
    )
}

fn write_fixture_db(dir: &Path, name: &str, models: &[(&str, &str, &str, &str)]) -> PathBuf {
    let raw: Vec<String> = (0..500)
        .map(|i| format!("[{:.1},0.0]", 20.0 + 20.0 * i as f64))
        .collect();
    let raw = raw.join(",");
    let entries: Vec<String> = models
        .iter()
        .map(|(key, name, fixture, target)| {
            format!(
                r#""{key}": {{"name":"{name}","fixture":"{fixture}","target":"{target}","raw_pts":[{raw}],"ctrl_pts":[[20.0,0.0],[1000.0,0.0],[20000.0,0.0]]}}"#
            )
        })
        .collect();
    let path = dir.join(name);
    std::fs::write(
        &path,
        format!(
            r#"{{
  "targets": {{
    "harman_2018": [[20.0,0.0],[1000.0,0.0],[20000.0,0.0]],
    "clone_target": [[20.0,0.0],[1000.0,0.0],[20000.0,0.0]],
    "unrecorded_target": [[20.0,0.0],[1000.0,0.0],[20000.0,0.0]]
  }},
  "target_fixtures": {{ "harman_2018": "gras_45ca", "clone_target": "gras_43ag_clone" }},
  "models": {{ {} }}
}}"#,
            entries.join(",")
        ),
    )
    .expect("database");
    path
}

/// Run against a database the test wrote, in a directory the test owns.
fn run_db(dir: &Path, db: &Path, args: &[&str]) -> (Option<i32>, String, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_hypreq"))
        .current_dir(dir)
        .arg("--data-file")
        .arg(db)
        .arg("--models-dir")
        .arg(dir.join("no-such-models-dir"))
        .args(args)
        .output()
        .expect("the binary runs");
    (
        out.status.code(),
        String::from_utf8(out.stdout).expect("stdout is utf-8"),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn a_target_defined_on_another_fixture_is_refused_naming_both() {
    let dir = scratch("fixture-conflict");
    let db = fixture_db(&dir);
    let (code, _, stderr) = run_db(&dir, &db, &["-H", "conflict", "--no-plot"]);
    assert_eq!(
        code,
        Some(2),
        "a 5128 measurement EQ'd to a 45CA target ran:\n{stderr}"
    );
    for want in [
        "B&K 5128",
        "GRAS 45CA",
        "bk5128",
        "gras_45ca",
        "harman_2018",
    ] {
        assert!(
            stderr.contains(want),
            "the refusal did not say {want}: {stderr}"
        );
    }
    // Nothing on disk: a refusal that has already written two convolvers has taught the user
    // nothing about the state of their directory.
    let written = std::fs::read_dir(&dir)
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        })
        .expect("scratch readable")
        .filter(|e| e.file_name() != "fixtures.json")
        .count();
    assert_eq!(written, 0, "the refusal wrote files before refusing");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn force_equalises_anyway_and_says_which_refusal_it_overrode() {
    let dir = scratch("fixture-force");
    let db = fixture_db(&dir);
    let (code, stdout, stderr) = run_db(&dir, &db, &["-H", "conflict", "--no-plot", "--force"]);
    assert_eq!(code, Some(0), "--force did not override: {stderr}");
    assert!(
        stdout.contains("[Note]") && stdout.contains("B&K 5128"),
        "the run overrode the pairing without saying so:\n{stdout}"
    );
    assert!(
        std::fs::read_dir(&dir).expect("scratch readable").any(|e| e
            .map(|e| e.file_name().to_string_lossy().ends_with(".wav"))
            .unwrap_or(false)),
        "--force ran but produced no convolver"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn only_a_real_conflict_is_refused() {
    let dir = scratch("fixture-allowed");
    let db = fixture_db(&dir);
    // Two units of one fixture design are not the difference the literature warns about...
    let (code, _, stderr) = run_db(&dir, &db, &["-H", "adjacent", "--no-plot"]);
    assert_eq!(
        code,
        Some(0),
        "the 43AG/clone adjacency was refused: {stderr}"
    );
    // ...and a target that records no fixture is *unknown*, not universally compatible. Refusing on
    // the absence of a claim would punish a database for being honest about what it does not know.
    let (code, _, stderr) = run_db(&dir, &db, &["-H", "unrecorded", "--no-plot"]);
    assert_eq!(
        code,
        Some(0),
        "an unrecorded target fixture was refused: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_bad_fixture_name_kills_the_database_before_any_run() {
    let dir = scratch("fixture-broken");
    let db = broken_fixture_db(&dir);
    for args in [
        vec!["-H", "broken", "--no-plot"],
        // Listing is not exempt: an unparseable fixture is a broken file, and `--list-models` is
        // exactly how someone would discover it.
        vec!["--list-models"],
        // `--force` overrides a pairing decision, not a database that cannot be read.
        vec!["-H", "broken", "--no-plot", "--force"],
        // Nor is the refusal of some *other* model's pairing allowed to mask it.
        vec!["-H", "conflict", "--no-plot"],
    ] {
        let (code, _, stderr) = run_db(&dir, &db, &args);
        assert_eq!(
            code,
            Some(1),
            "{args:?} treated a broken fixture as fine:\n{stderr}"
        );
        assert!(
            stderr.contains("sm7422") && stderr.contains("gras_45ca"),
            "{args:?} did not report the bad fixture with its choices: {stderr}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_run_that_applies_no_filter_has_no_pairing_to_refuse() {
    let dir = scratch("fixture-bypass");
    let db = fixture_db(&dir);
    let seed = stereo_bytes("fixture-bypass-seed");
    std::fs::write(dir.join("in.wav"), &seed).expect("fixture");
    // --bypass-eq applies no EQ, so there is no target being equalised to and nothing to refuse.
    let (code, _, stderr) = run_db(
        &dir,
        &db,
        &["--input", "in.wav", "--bypass-eq", "--output", "out.wav"],
    );
    assert_eq!(code, Some(0), "a bypass run was refused: {stderr}");
    // Naming one model touches only that model, so an allowed pairing stays allowed even though a
    // conflicting model sits in the same file.
    let (code, _, stderr) = run_db(
        &dir,
        &db,
        &[
            "--input",
            "in.wav",
            "-H",
            "unrecorded",
            "--output",
            "out2.wav",
        ],
    );
    assert_eq!(
        code,
        Some(0),
        "naming one model pulled in another's pairing: {stderr}"
    );
    // `-H all` with `--input` is the odd rule: the audio path takes the *first* model and prints a
    // note about it. `conflict` is first in this file and `unrecorded` is last, so a refusal naming
    // `conflict` proves the guard read the audio path's rule rather than the whole file.
    let (code, _, stderr) = run_db(
        &dir,
        &db,
        &["--input", "in.wav", "-H", "all", "--output", "out3.wav"],
    );
    assert_eq!(
        code,
        Some(2),
        "the audio path's first-model rule changed:\n{stderr}"
    );
    assert!(
        stderr.contains("'conflict'") && !stderr.contains("'unrecorded'"),
        "the guard refused on a model the run would never open: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_shipped_database_records_fixtures_and_the_pairing_it_inherited() {
    // A content check on the shipped data, because the two facts it carries can go missing without
    // any test failing elsewhere: dropping `target_fixtures` silently switches the whole comparison
    // off, and dropping a model's reason would make its own run refuse.
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("measurements.json");
    let text = std::fs::read_to_string(&data).expect("database");
    for fixture in ["gras_43ag", "gras_43ag_clone", "gras_45ca"] {
        assert!(
            text.contains(fixture),
            "no model or target records {fixture}"
        );
    }
    assert!(
        text.contains("\"target_fixtures\"") && text.contains("\"harman_2018\": \"gras_45ca\""),
        "the target's fixture went unrecorded, which turns the refusal off quietly"
    );
    let stdout = run("shipped-fixture-list", &["--list-models"]);
    assert_eq!(
        stdout.matches("[GRAS 43AG]").count(),
        2,
        "two models are genuine 43AG captures and should say so:\n{stdout}"
    );
    assert!(
        stdout.contains("[Clone GRAS 43AG]"),
        "the clone should not be listed as genuine:\n{stdout}"
    );
    let (code, stdout, stderr) = {
        let dir = scratch("shipped-run");
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_hypreq"))
            .current_dir(&dir)
            .arg("--data-file")
            .arg(&data)
            .args(["--no-plot"])
            .output()
            .expect("the binary runs");
        let r = (
            out.status.code(),
            String::from_utf8(out.stdout).expect("stdout"),
            String::from_utf8_lossy(&out.stderr).to_string(),
        );
        let _ = std::fs::remove_dir_all(&dir);
        r
    };
    assert_eq!(
        code,
        Some(0),
        "the shipped pairing stopped running:\n{stderr}"
    );
    assert!(
        !stdout.contains("[Note]"),
        "a shipped pairing printed a refusal it is not supposed to have:\n{stdout}"
    );
}

// ---------------------------------------------------------------------------
// --experimental-phase: the mixed-phase stage, judged on what it promises.
// ---------------------------------------------------------------------------

/// Magnitude in dB on a common grid, so two filters of the same length are compared bin for bin.
fn mag_db(h: &[f64]) -> Vec<f64> {
    let n = hypreq::fft::shl_bit_length(h.len() * 2);
    hypreq::fft::rfft(h, n)
        .iter()
        .map(|c| 20.0 * (c.norm() + 1e-300).log10())
        .collect()
}

/// Group delay in ms, unwrapped then differenced, on the same grid a filter is compared on.
///
/// This is the quantity the experimental stage exists to change, so it is the quantity the test
/// measures. An energy centroid cannot do this job: the bass is a small share of a correction
/// filter's energy, so pulling 10-200 Hz toward linear phase leaves the global centroid sitting at
/// 1.2 samples before and after alike, which is what an earlier draft of this test confidently
/// reported as "nothing moved".
fn group_delay_ms(h: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let fs = 48000.0;
    let n = hypreq::fft::shl_bit_length(h.len() * 2);
    let spec = hypreq::fft::rfft(h, n);
    let freq = hypreq::fft::rfftfreq(n, fs);
    let mut phase = Vec::with_capacity(spec.len());
    let mut carry = 0.0;
    let mut prev = 0.0;
    for (i, c) in spec.iter().enumerate() {
        let p = c.arg();
        if i > 0 {
            let mut d = p - prev;
            while d > std::f64::consts::PI {
                carry -= std::f64::consts::TAU;
                d -= std::f64::consts::TAU;
            }
            while d < -std::f64::consts::PI {
                carry += std::f64::consts::TAU;
                d += std::f64::consts::TAU;
            }
        }
        phase.push(p + carry);
        prev = p;
    }
    let gd: Vec<f64> = (0..phase.len())
        .map(|i| {
            let (a, b) = if i == 0 {
                (0, 1)
            } else if i + 1 == phase.len() {
                (i - 1, i)
            } else {
                (i - 1, i + 1)
            };
            let df = freq[b] - freq[a];
            if df > 0.0 {
                -(phase[b] - phase[a]) / df / (2.0 * std::f64::consts::PI) * 1000.0
            } else {
                0.0
            }
        })
        .collect();
    (freq, gd)
}

/// Spread (max - min) of group delay across a frequency window, ms.
fn gd_spread(h: &[f64], lo: f64, hi: f64) -> f64 {
    let (freq, gd) = group_delay_ms(h);
    let band: Vec<f64> = (0..freq.len())
        .filter(|&i| (lo..=hi).contains(&freq[i]))
        .map(|i| gd[i])
        .collect();
    assert!(band.len() > 10, "the window holds {} bins", band.len());
    let mn = band.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = band.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    mx - mn
}

/// Largest phase difference against another filter, in radians, across a frequency window.
fn phase_difference(a: &[f64], b: &[f64], lo: f64, hi: f64) -> f64 {
    let n = hypreq::fft::shl_bit_length(a.len() * 2);
    let sa = hypreq::fft::rfft(a, n);
    let sb = hypreq::fft::rfft(b, n);
    let freq = hypreq::fft::rfftfreq(n, 48000.0);
    (0..sa.len())
        .filter(|&i| (lo..=hi).contains(&freq[i]))
        .map(|i| {
            let d = (sa[i].arg() - sb[i].arg()).abs();
            d.min((std::f64::consts::TAU - d).abs())
        })
        .fold(0.0f64, f64::max)
}

#[test]
fn a_default_run_says_nothing_about_a_stage_it_did_not_run() {
    let dir = scratch("phase-default");
    let (code, stdout, stderr) = run_in(&dir, &["-H", "dt770_80", "--no-plot"]);
    assert_eq!(code, Some(0), "{stderr}");
    assert!(
        !stdout.contains("Experimental Phase"),
        "the default report talks about the experimental stage:\n{stdout}"
    );
    // And the delivered filter is the minimum-phase one, byte for byte: the stage is opt-in or the
    // parity-pinned artefacts move on their own.
    let bytes = read_bytes(&dir.join("dt770_80ohm_f64_48k.wav"));
    let again = scratch("phase-default-again");
    let (code2, _, stderr2) = run_in(
        &again,
        &["-H", "dt770_80", "--no-plot", "--experimental-phase", "200"],
    );
    assert_eq!(code2, Some(0), "{stderr2}");
    let shaped = read_bytes(&again.join("dt770_80ohm_f64_48k.wav"));
    assert_ne!(
        bytes, shaped,
        "the flag changed nothing, so it is not wired to anything"
    );
}

#[test]
fn the_stage_moves_the_phase_and_leaves_the_curve_alone() {
    // The acceptance criterion, measured end to end on the filter the run writes: the magnitude is
    // the same curve, and the time behaviour is not. Everything else the block prints is commentary
    // on these two facts.
    let dir = scratch("phase-accept");
    let (code, stdout, stderr) = run_in(
        &dir,
        &["-H", "dt770_80", "--no-plot", "--experimental-phase", "200"],
    );
    assert_eq!(code, Some(0), "{stderr}");
    assert!(stdout.contains("Experimental Phase"), "{stdout}");

    let shaped =
        hypreq::wav::read_wav(&dir.join("dt770_80ohm_f64_48k.wav")).expect("shaped filter");
    let plain = scratch("phase-plain");
    assert_eq!(
        run_in(&plain, &["-H", "dt770_80", "--no-plot"]).0,
        Some(0),
        "the reference run works"
    );
    let reference =
        hypreq::wav::read_wav(&plain.join("dt770_80ohm_f64_48k.wav")).expect("reference filter");

    let (a, b) = (&shaped.planar[0], &reference.planar[0]);
    assert!(a.len() >= b.len(), "the stage shrank the tap count");

    let max_len = a.len().max(b.len());
    let mut a_pad = a.to_vec();
    let mut b_pad = b.to_vec();
    a_pad.resize(max_len, 0.0);
    b_pad.resize(max_len, 0.0);

    let ma = mag_db(&a_pad);
    let mb = mag_db(&b_pad);
    let worst = ma
        .iter()
        .zip(&mb)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    // Stated tolerance: a tenth of a dB, which is a fifth of what the verification loop forgives a
    // single third-octave band. The cost is truncation of a response the reshape made longer, not a
    // re-derived curve, and the report prints the number it is holding itself to.
    assert!(
        worst < 0.1,
        "the curve moved by {worst:.4} dB; this stage is supposed to move phase only"
    );

    // Phase moved, where the stage says it moved.
    let moved = phase_difference(a, b, 30.0, 180.0);
    assert!(
        moved > 0.05,
        "the band's phase moved by {moved:.2e} rad, so nothing was reshaped"
    );
    // And it moved the way the stage claims: flatter across the band, not differently flat by accident.
    let (before, after) = (gd_spread(b, 30.0, 180.0), gd_spread(a, 30.0, 180.0));
    assert!(
        after < before,
        "group-delay spread across the band went {before:.3} ms -> {after:.3} ms; it did not flatten"
    );
    // Above the taper the filter's delay *shape* is untouched, with each filter's own median removed.
    // Absolute values are the wrong thing to compare and failed on a correct stage: the delivered kernel
    // arrives later than the minimum-phase one, and a later kernel simply has a taller group delay at
    // every frequency. Raw phase was wronger still and moved by pi — a delay is a linear phase ramp
    // across the whole spectrum. Both of those are the latency, which is the next assertion.
    let (fa, mut ga) = group_delay_ms(&a_pad);
    let (_fb, mut gb) = group_delay_ms(&b_pad);
    let treble: Vec<usize> = (0..fa.len())
        .filter(|&i| (1000.0..20000.0).contains(&fa[i]))
        .collect();
    let centre = |g: &mut Vec<f64>| {
        let mut d: Vec<f64> = treble.iter().map(|&i| g[i]).collect();
        d.sort_by(|x, y| x.total_cmp(y));
        let m = d[d.len() / 2];
        for i in &treble {
            g[*i] -= m;
        }
        m
    };
    let median_a = centre(&mut ga);
    let median_b = centre(&mut gb);
    let above = treble
        .iter()
        .map(|&i| (ga[i] - gb[i]).abs())
        .fold(0.0f64, f64::max);
    assert!(
        above < 0.05,
        "the treble's delay shape moved by {above:.3} ms once each filter's own median was removed; the \n             stage was given 10-200 Hz"
    );
    // And the kernel did arrive later, which is the price. Compared against the number the report
    // prints, so the two cannot drift apart.
    let shift = median_a - median_b;
    assert!(
        shift > 1.0,
        "the delivered kernel is not later than the minimum-phase one ({shift:.3} ms), so the latency the \n         report prints is not the latency the taps carry"
    );
    let quoted = stdout
        .lines()
        .find(|l| l.contains("Kernel"))
        .and_then(|l| l.split('+').nth(1))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<f64>().ok());
    match quoted {
        Some(q) => assert!(
            (q - shift).abs() < 2.0,
            "the block prints {q} ms of latency and the taps carry {shift:.3} ms"
        ),
        None => panic!("the Kernel line does not print a latency the test can read"),
    }
    let block = stdout
        .lines()
        .skip_while(|l| !l.contains("Experimental Phase"))
        // The block is nine lines now that it also prints the budget and the kernel growth; an earlier
        // window of eight clipped off the audibility line and made this test fail on the fact it exists
        // to check.
        .take(12)
        .collect::<Vec<_>>()
        .join("\n");
    for fact in ["Band", "Group delay", "Magnitude", "contested"] {
        assert!(
            block.contains(fact),
            "the block does not state {fact}:\n{block}"
        );
    }
}

#[test]
fn an_impossible_crossover_is_refused_before_a_single_file_is_written() {
    let dir = scratch("phase-refuse");
    // Above Nyquist the band cannot exist.
    let (code, _stdout, stderr) = run_in(
        &dir,
        &[
            "-H",
            "dt770_80",
            "--no-plot",
            "--experimental-phase",
            "30000",
        ],
    );
    assert_eq!(
        code,
        Some(1),
        "a crossover above Nyquist was accepted: {stderr}"
    );
    assert!(stderr.contains("Nyquist"), "{stderr}");
    let leftovers: Vec<String> = std::fs::read_dir(&dir)
        .expect("the scratch dir exists")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    assert!(
        leftovers.is_empty(),
        "a refused run still wrote: {leftovers:?}"
    );

    // A nonsense number is a bad command line, not a failed run.
    let (code2, _, stderr2) = run_in(
        &dir,
        &["-H", "dt770_80", "--no-plot", "--experimental-phase", "0"],
    );
    assert_eq!(code2, Some(2), "zero Hz was accepted as a crossover");
    assert!(stderr2.contains("--experimental-phase"), "{stderr2}");
}

#[test]
fn model_run_applies_true_peak_limiter_to_written_ir_when_enabled() {
    let dir = scratch("limit-ir");
    let (code, _, stderr) = run_in(
        &dir,
        &[
            "-H",
            "dt770_80",
            "--taps",
            "4096",
            "--headroom-db",
            "-10.0",
            "--tp-ceiling",
            "-10.0",
            "--no-plot",
        ],
    );
    assert_eq!(code, Some(0), "run failed: {stderr}");
    let wav_path = dir.join("dt770_80ohm_f64_48k.wav");
    assert!(wav_path.exists(), "IR wav was not written");
    let wav = hypreq::wav::read_wav(&wav_path).expect("read wav");
    let m = hypreq::measure::measure_signal(&wav.planar, 48000, hypreq::tp::Ratio::X16);
    assert!(
        m.true_peak_dbtp <= -10.0 + 1e-4,
        "written IR true peak was not limited to ceiling: got {} dBTP",
        m.true_peak_dbtp
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn bypass_eq_applies_true_peak_limiter_to_music_before_writing() {
    let dir = scratch("bypass-limit-music");
    let in_wav = dir.join("hot_input.wav");
    // Generate a 0 dBFS square/sine signal with peak above -3 dBTP
    let fs = 48000;
    let n = 4800;
    let mut ch0 = vec![0.0f64; n];
    let mut ch1 = vec![0.0f64; n];
    for i in 0..n {
        let s = (i as f64 * 440.0 * std::f64::consts::TAU / fs as f64).sin();
        ch0[i] = s * 0.99;
        ch1[i] = s * 0.99;
    }
    hypreq::wav::write_wav_ieee_float(&in_wav, &[ch0, ch1], fs, 32).expect("write input");
    let out_wav = dir.join("out.wav");
    let (code, _, stderr) = run_in(
        &dir,
        &[
            "--input",
            "hot_input.wav",
            "--bypass-eq",
            "--output",
            "out.wav",
            "--tp-ceiling",
            "-4.0",
        ],
    );
    assert_eq!(code, Some(0), "run failed: {stderr}");
    assert!(out_wav.exists(), "output audio was not written");
    let wav = hypreq::wav::read_wav(&out_wav).expect("read wav");
    let m = hypreq::measure::measure_signal(&wav.planar, fs, hypreq::tp::Ratio::X16);
    assert!(
        m.true_peak_dbtp <= -4.0 + 1e-4,
        "written passthrough music was not limited to ceiling: got {} dBTP",
        m.true_peak_dbtp
    );
    let _ = std::fs::remove_dir_all(&dir);
}
