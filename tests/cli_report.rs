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
