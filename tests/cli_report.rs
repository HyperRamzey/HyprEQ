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
