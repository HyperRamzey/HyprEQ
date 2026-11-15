//! CLI-facing limiter configuration and the limiter's report block.
//!
//! The limiter itself is [`crate::limiter`]; this module is the thin layer the
//! command line needs: parse `--tp-*` into a [`Cfg`], validate it, and render the
//! [`Report`] in the shape the report is printed in.
//!
//! One deliberate difference from the vendored defaults: `Cfg::default()` carries
//! [`Ratio::X8`] (the general-purpose delivery default), while this tool's own
//! default is **x16** — a file being handed to a convolver is a master, and at x8
//! the printed ceiling has the least margin against the interpolator's own
//! under-read on full-band content. That is set *here*, at the call site, so the
//! limiter module keeps its own default and its tests keep meaning what they mean.

use crate::limiter::{
    Cfg, Report, DEFAULT_ATTACK_MS, DEFAULT_CEILING_DB, DEFAULT_LINK_RELEASE,
    DEFAULT_LINK_TRANSIENT, DEFAULT_LOOKAHEAD_MS, DEFAULT_RELEASE_MS, TRANSIENT_RELEASE_MS,
};
use crate::limiter::{MAX_CEILING_DB, MIN_CEILING_DB};
use crate::measure::{sgn, SignalMeasurements};
use crate::tp::Ratio;
use crate::wav::comma_sep;

/// The ratio this tool limits and measures at unless told otherwise.
pub const DEFAULT_RATIO: Ratio = Ratio::X16;

/// Everything the `--tp-*` options select, before it becomes a [`Cfg`].
#[derive(Clone, Debug)]
pub struct TpArgs {
    /// `--tp-ratio`, kept as written so a bad spelling names itself in the error.
    pub ratio: String,
    pub ceiling_db: f64,
    pub lookahead_ms: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    pub link_transient: f64,
    pub link_release: f64,
    /// `--no-tp-limiter`: measure, but never touch a sample.
    pub disable: bool,
}

impl Default for TpArgs {
    fn default() -> Self {
        Self {
            ratio: DEFAULT_RATIO.label().to_string(),
            ceiling_db: DEFAULT_CEILING_DB,
            lookahead_ms: DEFAULT_LOOKAHEAD_MS,
            attack_ms: DEFAULT_ATTACK_MS,
            release_ms: DEFAULT_RELEASE_MS,
            link_transient: DEFAULT_LINK_TRANSIENT,
            link_release: DEFAULT_LINK_RELEASE,
            disable: false,
        }
    }
}

impl TpArgs {
    /// Build and validate the limiter configuration.
    ///
    /// Validation is the limiter's own, so the range messages are the limiter's
    /// (`--tp-ceiling -30` explains the -24 floor rather than a second set of
    /// numbers invented here). A disabled limiter skips validation:
    /// `--no-tp-limiter` with a nonsense ceiling still measures the file.
    pub fn to_cfg(&self) -> Result<Cfg, String> {
        let ratio: Ratio = self
            .ratio
            .parse()
            .map_err(|e: String| format!("--tp-ratio: {e}"))?;
        let cfg = Cfg {
            enable: !self.disable,
            ceiling_db: self.ceiling_db,
            ratio,
            lookahead_ms: self.lookahead_ms,
            attack_ms: self.attack_ms,
            release_ms: self.release_ms,
            link_transient: self.link_transient,
            link_release: self.link_release,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

/// Render the limiter's report and the independent verification of its output.
///
/// The report's own `post_dbtp` is measured by the detector that chose the gain —
/// the control loop grading its own homework — so the same buffer is measured again
/// by [`crate::measure`] and printed next to it. `Within ceiling` is the
/// self-assessment; the independent figure is the evidence.
///
/// Returned as lines rather than printed so the wording can be asserted, and so a
/// caller can redirect the block (e.g. into a log) without intercepting stdout.
#[must_use]
pub fn limiter_report_lines(
    rep: &Report,
    post: &SignalMeasurements,
    cfg: &Cfg,
    fs: u32,
) -> Vec<String> {
    let raised = if cfg.lookahead_was_raised(fs) {
        " [RAISED to filter floor]"
    } else {
        ""
    };
    vec![
        format!(
            "  Side-Chain / Ratio  : {} ({})",
            rep.ratio.label(),
            rep.ratio.chain()
        ),
        format!(
            "  Ceiling / Guard     : {} dBTP (aim guard -{:.2} dB | >0.45 fs band caveat +{:.2} dB)",
            sgn(rep.ceiling_db, 2),
            rep.guard_db,
            rep.band_caveat_db,
        ),
        format!(
            "  Lookahead / Latency : {:.2} ms eff ({:.2} ms req){raised} | {} samples ({:.2} ms), ramp {} frames",
            rep.effective_lookahead_ms,
            rep.requested_lookahead_ms,
            comma_sep(rep.latency_samples as u64),
            rep.latency_ms,
            comma_sep(rep.ramp_frames as u64),
        ),
        format!(
            "  Two-Stage Envelope  : Slow {:.1}/{:.1} ms (link {:.0}%) | Transient {TRANSIENT_RELEASE_MS:.1} ms rel (link {:.0}%)",
            rep.attack_ms,
            rep.release_ms,
            rep.link_release * 100.0,
            rep.link_transient * 100.0,
        ),
        format!(
            "  Limiter Pre -> Post : {} -> {} dBTP ({}) | Independent {}: {} dBTP | ITU x4: {} dBTP | Within ceiling: {}",
            sgn(rep.pre_dbtp, 3),
            sgn(rep.post_dbtp, 3),
            rep.ratio.label(),
            post.ratio.label(),
            sgn(post.true_peak_dbtp, 3),
            sgn(post.itu_x4_dbtp, 3),
            if rep.within_ceiling() { "True" } else { "False" },
        ),
        format!(
            "  Gain Reduction      : Max {:.3} dB (Transient {:.3} dB, Slow {:.3} dB) | Engaged: {} frames (Slow hold: {})",
            rep.max_gain_reduction_db,
            rep.transient_reduction_db,
            rep.release_reduction_db,
            comma_sep(rep.engaged_frames),
            comma_sep(rep.release_frames),
        ),
    ]
}

/// Print [`limiter_report_lines`].
pub fn print_limiter_report(rep: &Report, post: &SignalMeasurements, cfg: &Cfg, fs: u32) {
    for line in limiter_report_lines(rep, post, cfg, fs) {
        println!("{line}");
    }
}

/// The one-line summary printed when no audio file was given.
#[must_use]
pub fn summary_line(cfg: &Cfg, fs: u32) -> String {
    if !cfg.enable {
        return "true-peak limiter disabled".to_string();
    }
    format!(
        "{} limiter, ceiling {:+.2} dBTP (guard {:.2} dB), lookahead {:.2} ms -> {} samples at {} Hz",
        cfg.ratio.label(),
        cfg.ceiling_db,
        cfg.guard_db(),
        cfg.effective_lookahead_ms(fs),
        comma_sep(cfg.latency_samples(fs) as u64),
        comma_sep(u64::from(fs)),
    )
}

/// The `--tp-ceiling` range, for `--help`.
pub const CEILING_RANGE: (f64, f64) = (MIN_CEILING_DB, MAX_CEILING_DB);

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> Report {
        Report {
            pre_dbtp: -0.35,
            post_dbtp: -3.04,
            max_gain_reduction_db: 2.69,
            ceiling_db: -3.0,
            guard_db: 0.04,
            effective_lookahead_ms: 2.0,
            requested_lookahead_ms: 2.0,
            latency_ms: 2.17,
            latency_samples: 104,
            ramp_frames: 30,
            attack_ms: 60.0,
            release_ms: 900.0,
            link_transient: 1.0,
            link_release: 0.75,
            transient_reduction_db: 2.69,
            release_reduction_db: 1.10,
            release_frames: 1234,
            ratio: Ratio::X16,
            engaged_frames: 5678,
            band_caveat_db: 0.30,
        }
    }

    fn post() -> SignalMeasurements {
        SignalMeasurements {
            channels: 2,
            frames: 96000,
            fs: 48000,
            duration_s: 2.0,
            ratio: Ratio::X16,
            sample_peak_lin: 0.7,
            sample_peak_dbfs: -3.1,
            per_channel_sample_dbfs: vec![-3.1, -6.0],
            true_peak_lin: 0.708,
            true_peak_dbtp: -3.04,
            per_channel_dbtp: vec![-3.04, -5.9],
            itu_x4_dbtp: -3.21,
            itu_x4_per_channel_dbtp: vec![-3.21, -6.1],
            rms_dbfs: -14.2,
            per_channel_rms_dbfs: vec![-14.2, -17.2],
            crest_tp_db: 11.16,
            crest_sample_db: 11.1,
            meter_peak_hold_lin: 0.7,
            meter_peak_hold_dbfs: -3.1,
            meter_param: 0.48,
        }
    }

    #[test]
    fn defaults_match_the_reference_cli() {
        let a = TpArgs::default();
        assert_eq!(a.ratio, "x16");
        assert_eq!(a.ceiling_db, -3.0);
        assert_eq!(a.lookahead_ms, 2.0);
        assert_eq!(a.attack_ms, 60.0);
        assert_eq!(a.release_ms, 900.0);
        assert_eq!(a.link_transient, 1.0);
        assert_eq!(a.link_release, 0.75);
        assert!(!a.disable);

        let cfg = a.to_cfg().unwrap();
        assert_eq!(cfg.ratio, Ratio::X16);
        assert!(cfg.enable);
        // The limiter module keeps its own default; only this call site asks for x16.
        assert_eq!(Cfg::default().ratio, Ratio::X8);
    }

    #[test]
    fn every_documented_ratio_spelling_parses() {
        for s in [
            "x2", "x4", "x8", "x16", "x32", "2", "4", "8", "16", "32", "X16", " x8 ",
        ] {
            let a = TpArgs {
                ratio: s.to_string(),
                ..TpArgs::default()
            };
            let cfg = a.to_cfg().unwrap_or_else(|e| panic!("{s}: {e}"));
            let digits = s.trim().to_ascii_lowercase().replace('x', "");
            assert_eq!(cfg.ratio.label(), format!("x{digits}"), "spelling {s}");
        }
        let e = TpArgs {
            ratio: "x64".into(),
            ..TpArgs::default()
        }
        .to_cfg()
        .unwrap_err();
        assert!(e.starts_with("--tp-ratio:"), "got {e}");
    }

    #[test]
    fn out_of_range_values_are_refused_with_the_limiter_s_wording() {
        let e = TpArgs {
            ceiling_db: -30.0,
            ..TpArgs::default()
        }
        .to_cfg()
        .unwrap_err();
        assert!(e.contains("ceiling") && e.contains("-24"), "got {e}");

        let e = TpArgs {
            link_release: 7.5,
            ..TpArgs::default()
        }
        .to_cfg()
        .unwrap_err();
        assert!(e.contains("link") && e.contains("fraction"), "got {e}");

        assert!(TpArgs {
            attack_ms: 0.1,
            ..TpArgs::default()
        }
        .to_cfg()
        .unwrap_err()
        .contains("attack"));
    }

    #[test]
    fn a_disabled_limiter_skips_validation() {
        let cfg = TpArgs {
            disable: true,
            ceiling_db: -30.0,
            ..TpArgs::default()
        }
        .to_cfg()
        .expect("a disabled config skips range checks");
        assert!(!cfg.enable);
        assert_eq!(summary_line(&cfg, 48000), "true-peak limiter disabled");
    }

    /// The expected block is this program's output pasted, not retyped.
    const EXPECTED: &str = "  Side-Chain / Ratio  : x16 (16-phase Kaiser polyphase (51 taps/phase) + peak refinement)\n  Ceiling / Guard     : -3.00 dBTP (aim guard -0.04 dB | >0.45 fs band caveat +0.30 dB)\n  Lookahead / Latency : 2.00 ms eff (2.00 ms req) | 104 samples (2.17 ms), ramp 30 frames\n  Two-Stage Envelope  : Slow 60.0/900.0 ms (link 75%) | Transient 25.0 ms rel (link 100%)\n  Limiter Pre -> Post : -0.350 -> -3.040 dBTP (x16) | Independent x16: -3.040 dBTP | ITU x4: -3.210 dBTP | Within ceiling: True\n  Gain Reduction      : Max 2.690 dB (Transient 2.690 dB, Slow 1.100 dB) | Engaged: 5,678 frames (Slow hold: 1,234)";

    #[test]
    fn report_block_matches_the_reference_wording() {
        let cfg = TpArgs::default().to_cfg().unwrap();
        let lines = limiter_report_lines(&report(), &post(), &cfg, 48000);
        assert_eq!(lines.join("\n"), EXPECTED);
    }

    #[test]
    fn report_says_false_when_the_independent_check_disagrees() {
        let cfg = TpArgs::default().to_cfg().unwrap();
        let mut rep = report();
        rep.post_dbtp = -2.0; // the limiter believes it is done
        let mut post = self::post();
        post.true_peak_dbtp = -2.0; // but the independent pass says otherwise
        let lines = limiter_report_lines(&rep, &post, &cfg, 48000);
        assert!(lines[4].contains("Within ceiling: False"), "{}", lines[4]);
        assert!(
            lines[4].contains("-0.350 -> -2.000 dBTP"),
            "pre/post must both show: {}",
            lines[4]
        );
    }

    #[test]
    fn report_shows_when_lookahead_was_raised_to_the_filter_floor() {
        // x32 at 8 kHz needs 1055/8000 = 132 ms of lookahead, far above the 2 ms
        // requested, so the line must admit the request was raised.
        let cfg = TpArgs {
            ratio: "x32".into(),
            ..TpArgs::default()
        }
        .to_cfg()
        .unwrap();
        let eff = cfg.effective_lookahead_ms(8000);
        // x32's polyphase group delay cannot be realised in fewer than its own
        // filter-delay floor, so at 8 kHz the requested 2 ms is below what the
        // interpolator physically needs and gets raised.
        assert!(
            eff > cfg.lookahead_ms,
            "x32 at 8 kHz needs {eff} ms, requested {}",
            cfg.lookahead_ms
        );
        assert!(cfg.lookahead_was_raised(8000));
        let rep = Report {
            effective_lookahead_ms: eff,
            latency_ms: eff,
            latency_samples: cfg.latency_samples(8000),
            ..report()
        };
        let lines = limiter_report_lines(&rep, &post(), &cfg, 8000);
        assert!(
            lines[2].contains("[RAISED to filter floor]"),
            "{}",
            lines[2]
        );
        assert!(
            lines[2].contains(&format!("{eff:.2} ms eff")),
            "the line must print the raised figure: {}",
            lines[2]
        );
        assert!(lines[2].contains("(2.00 ms req)"), "{}", lines[2]);
    }

    #[test]
    fn summary_line_quotes_the_effective_lookahead() {
        let s = summary_line(&TpArgs::default().to_cfg().unwrap(), 48000);
        assert!(
            s.starts_with("x16 limiter, ceiling -3.00 dBTP (guard 0.04 dB), lookahead 2.00 ms -> "),
            "got {s}"
        );
        assert!(s.contains("samples at 48,000 Hz"), "got {s}");
    }
}
