//! Driver physical limits: Olive & Lelièvre (2026) level-dependent nonlinear distortion
//! envelopes and Thiele–Small sub-resonance diaphragm excursion scaling.

use crate::pchip::Pchip;

/// Measured distortion at a single output SPL sweep.
#[derive(Clone, Debug)]
pub struct DistortionSweep {
    pub spl_db: f64,
    /// `(freq_hz, distortion_pct)` (e.g. THD% or non-coherent %).
    pub points: Vec<(f64, f64)>,
}

/// Collection of multi-SPL distortion sweeps for one headphone model.
#[derive(Clone, Debug, Default)]
pub struct DistortionProfile {
    pub sweeps: Vec<DistortionSweep>,
}

impl DistortionProfile {
    /// Estimate `L_max(f) = max { L : D(f, L) <= threshold_pct }` per frequency in `f_grid`.
    #[must_use]
    pub fn max_clean_output_db(&self, f_grid: &[f64], threshold_pct: f64) -> Vec<f64> {
        if self.sweeps.is_empty() || f_grid.is_empty() {
            return vec![f64::INFINITY; f_grid.len()];
        }
        let mut sorted = self.sweeps.clone();
        sorted.sort_by(|a, b| a.spl_db.total_cmp(&b.spl_db));

        let eval_sweeps: Vec<(f64, Vec<f64>)> = sorted
            .iter()
            .filter_map(|s| {
                let mut pts: Vec<(f64, f64)> = s
                    .points
                    .iter()
                    .copied()
                    .filter(|(f, d)| f.is_finite() && *f > 0.0 && d.is_finite())
                    .collect();
                pts.sort_by(|a, b| a.0.total_cmp(&b.0));
                if pts.len() < 2 {
                    return None;
                }
                let (fs, ds): (Vec<f64>, Vec<f64>) = pts.into_iter().unzip();
                Some((s.spl_db, Pchip::new(&fs, &ds).eval_grid(f_grid)))
            })
            .collect();

        if eval_sweeps.is_empty() {
            return vec![f64::INFINITY; f_grid.len()];
        }

        (0..f_grid.len())
            .map(|bin| {
                let mut clean_max = f64::NEG_INFINITY;
                for (spl, vals) in &eval_sweeps {
                    if vals[bin] <= threshold_pct {
                        clean_max = *spl;
                    } else {
                        break;
                    }
                }
                clean_max
            })
            .collect()
    }

    /// Translate `L_max(f)` into an upper EQ boost bound `q_max(f)` given listening SPL.
    ///
    /// `h_cal_db` is the headphone's own magnitude response `R(f)` in dB, normalised to 0 at its
    /// reference frequency — the calibration function that turns drive level into SPL. With a
    /// flat program at the level that produces `listen_spl_db` where `R = 0`, the acoustic output
    /// at a bin is `L(f) = listen_spl_db + R(f) + q(f)`, so demanding `L(f) <= L_max(f)` gives
    /// `q(f) <= L_max(f) - listen_spl_db - R(f)`. Passing zeros for `h_cal_db` is therefore not a
    /// neutral choice: it is too generous by `R(f)` where the driver is sensitive and too tight
    /// where it is lazy. The target curve does not belong in this bound — substituting the
    /// designed `q = T - R` into the ceiling constrains `T` itself, not the box on `q`.
    ///
    /// Per Olive & Lelièvre (2026), audible music distortion generally requires high SPLs (~104–112 dBA),
    /// so this stays slack at normal listening volumes and tightens dynamically at high SPLs.
    #[must_use]
    pub fn q_max_bounds(
        &self,
        f_grid: &[f64],
        h_cal_db: &[f64],
        listen_spl_db: f64,
        threshold_pct: f64,
        hard_cap_db: f64,
    ) -> Vec<f64> {
        let l_max = self.max_clean_output_db(f_grid, threshold_pct);
        l_max
            .iter()
            .zip(h_cal_db)
            .map(|(&lm, &hc)| {
                // Three different things arrive here and only one of them is "unconstrained":
                //   +INFINITY  no usable sweeps for this bin -> the hard cap is the only bound
                //   -INFINITY  every measured sweep, down to the quietest, is already over
                //              threshold -> this frequency may not be boosted at all
                //   finite     the usual headroom arithmetic
                // Testing them with `!lm.is_finite()` collapsed the second case into the first and
                // handed the most distorted driver the largest allowed boost.
                if lm == f64::INFINITY || lm.is_nan() {
                    hard_cap_db
                } else if lm == f64::NEG_INFINITY {
                    0.0
                } else {
                    (lm - listen_spl_db - hc).min(hard_cap_db).max(0.0)
                }
            })
            .collect()
    }
}

/// Mechanical excursion model below fundamental driver resonance `fs_hz`.
#[derive(Clone, Copy, Debug)]
pub struct ExcursionModel {
    pub fs_hz: f64,
    /// Reference SPL (at `fs_hz`) where the driver reaches 100% linear `X_max`.
    pub ref_xmax_spl_db: f64,
}

impl ExcursionModel {
    /// Estimated excursion margin in dB (`> 0` means within `X_max`, `< 0` means over-driven).
    ///
    /// `band_spl_db + eq_boost_db` is what the diaphragm has to deliver at `f_hz`; `sub_rise_db` is
    /// the extra mechanical travel that costs, below the driver's own resonance: holding a
    /// flat-SPL target down there needs +6 dB of boost per octave, and a compliance-controlled
    /// diaphragm answers a growing boost with `(fs / f)²` of travel, so the debt grows at
    /// `40 · log₁₀(fs / f)` dB — 12.04 dB per octave. **That exponent is ours**, a standing
    /// first-order rule for a compliance-controlled diaphragm, not a published curve: the growth
    /// rate depends on the alignment (a vented box hands the travel to the port below tuning, and a
    /// voltage-driven cone in a sealed box holds roughly constant travel per volt rather than
    /// growing), so `margin_db` is an order-of-magnitude screen, not a limit check. A measured
    /// distortion-versus-SPL envelope ([`DistortionProfile`]) is the number HyprEQ actually acts on;
    /// the true excursion model is still open work (`docs/ROADMAP.md`, "No distortion or excursion
    /// model").
    ///
    /// A non-positive `f_hz` or `fs_hz` returns `0.0`, which reads as exactly-at-`X_max` on purpose:
    /// no opinion is not a safe answer, and a caller with no model has `Option::None`, not this.
    #[must_use]
    pub fn margin_db(&self, f_hz: f64, eq_boost_db: f64, band_spl_db: f64) -> f64 {
        if !f_hz.is_finite()
            || !self.fs_hz.is_finite()
            || !self.ref_xmax_spl_db.is_finite()
            || !eq_boost_db.is_finite()
            || !band_spl_db.is_finite()
        {
            return f64::NAN;
        }
        if f_hz <= 0.0 || self.fs_hz <= 0.0 {
            return 0.0;
        }
        let sub_rise_db = if f_hz < self.fs_hz {
            40.0 * (self.fs_hz / f_hz).log10()
        } else {
            0.0
        };
        self.ref_xmax_spl_db - (band_spl_db + eq_boost_db + sub_rise_db)
    }

    /// Margin at every bin of a designed curve, for a flat program that produces `listen_spl_db`
    /// where the headphone's own response `h_cal_db` is 0 dB.
    ///
    /// Same calibration arithmetic as [`DistortionProfile::q_max_bounds`]: the acoustic output at a
    /// bin is `L(f) = listen_spl_db + h_cal_db[i] + q_db[i]`, so that sum is what the diaphragm has
    /// to deliver, and it goes into [`Self::margin_db`] as `band_spl_db + eq_boost_db`. Dropping the
    /// `h_cal_db` term is not a neutral simplification — it credits the driver with the headroom
    /// that its own sensitivity is already spending, exactly the mistake the distortion ceiling had
    /// to be fixed for.
    ///
    /// A bin whose inputs are non-finite, or whose frequency is not positive, returns `NaN` rather
    /// than [`Self::margin_db`]'s `0.0`: `0.0` reads as "exactly at `X_max`", and a caller counting
    /// shortfalls must not be handed that for a hole in the data. `NaN` is neither a shortfall nor
    /// headroom, so it is skipped by both.
    ///
    /// This is what the CLI's `--global-fit` block reports. It is reported, not enforced on the fit,
    /// for the reason in [`Self::margin_db`]'s doc: the 12 dB-per-octave debt is a first-order rule
    /// whose growth rate depends on the alignment, so it can tell an operator that a sub-bass boost
    /// is ambitious; a measured [`DistortionProfile`] is what bounds the solver.
    #[must_use]
    pub fn margins_db(
        &self,
        f_hz: &[f64],
        q_db: &[f64],
        h_cal_db: &[f64],
        listen_spl_db: f64,
    ) -> Vec<f64> {
        let n = f_hz.len().min(q_db.len()).min(h_cal_db.len());
        (0..n)
            .map(|i| {
                let (f, q, hc) = (f_hz[i], q_db[i], h_cal_db[i]);
                if !self.fs_hz.is_finite()
                    || !self.ref_xmax_spl_db.is_finite()
                    || !listen_spl_db.is_finite()
                    || !f.is_finite()
                    || f <= 0.0
                    || !q.is_finite()
                    || !hc.is_finite()
                {
                    f64::NAN
                } else {
                    self.margin_db(f, q, listen_spl_db + hc)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The margin's own arithmetic, pinned: zero exactly at the stated reference, 12.04 dB of margin
    /// spent per octave below `fs_hz` (`40 · log₁₀2`), nothing spent above it, boost and level
    /// subtracting 1:1, and a garbage frequency returning the documented `0.0` rather than `±inf` or
    /// `NaN` — the guard is the difference between an order-of-magnitude screen and a wild number in
    /// a report.
    #[test]
    fn the_excursion_margin_spends_12_db_per_octave_below_fs() {
        let m = ExcursionModel {
            fs_hz: 60.0,
            ref_xmax_spl_db: 96.0,
        };
        assert_eq!(
            m.margin_db(60.0, 0.0, 96.0),
            0.0,
            "at the reference point the margin is exactly zero"
        );

        let one_octave = 40f64 * 2.0f64.log10();
        assert!(
            (m.margin_db(30.0, 0.0, 96.0) + one_octave).abs() < 1e-12,
            "one octave below fs must cost {one_octave:.4} dB, got {}",
            m.margin_db(30.0, 0.0, 96.0)
        );
        assert!(
            (m.margin_db(15.0, 0.0, 96.0) + 2.0 * one_octave).abs() < 1e-12,
            "the debt must grow linearly in octaves, got {}",
            m.margin_db(15.0, 0.0, 96.0)
        );

        assert_eq!(
            m.margin_db(120.0, 0.0, 96.0),
            0.0,
            "above fs there is no debt"
        );
        assert_eq!(m.margin_db(6000.0, 0.0, 96.0), 0.0);

        assert_eq!(m.margin_db(60.0, 6.0, 96.0), -6.0, "a boost is 1:1");
        assert_eq!(m.margin_db(60.0, 0.0, 90.0), 6.0, "a quieter band is 1:1");
        assert_eq!(
            m.margin_db(60.0, 3.0, 93.0),
            0.0,
            "the two terms must add, not max"
        );

        assert_eq!(
            m.margin_db(0.0, 0.0, 90.0),
            0.0,
            "0 Hz has no excursion to model"
        );
        assert_eq!(m.margin_db(-100.0, 0.0, 90.0), 0.0);
        let no_fs = ExcursionModel {
            fs_hz: 0.0,
            ref_xmax_spl_db: 96.0,
        };
        assert_eq!(
            no_fs.margin_db(1000.0, 0.0, 90.0),
            0.0,
            "an unparsed fs_hz must not report the 6 dB the formula would hand back"
        );
    }

    /// The per-bin form the `--global-fit` report actually calls. Pinned on three things the
    /// one-bin form cannot show: the calibration term is spent 1:1 against the margin (the same
    /// error `q_max_bounds` had), a garbage input poisons its own bin as `NaN` instead of inheriting
    /// [`ExcursionModel::margin_db`]'s `0.0` — which reads as "exactly at `X_max`" — and the vectors
    /// are zipped to their shortest length rather than panicking.
    #[test]
    fn per_bin_margins_spend_the_headphone_sensitivity_and_mark_holes_as_nan() {
        let m = ExcursionModel {
            fs_hz: 60.0,
            ref_xmax_spl_db: 96.0,
        };
        let one_octave = 40f64 * 2.0f64.log10();
        let f = [30.0, 60.0, 120.0];

        let got = m.margins_db(&f, &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 96.0);
        assert_eq!(got.len(), 3);
        assert!(
            (got[0] + one_octave).abs() < 1e-12,
            "one octave below fs at the reference SPL must be {one_octave:.4} dB short, got {:?}",
            got[0]
        );
        assert_eq!(
            (got[1], got[2]),
            (0.0, 0.0),
            "at fs and above there is no debt"
        );

        // Boost and sensitivity each cost 1:1, and they add.
        let boosted = m.margins_db(&f, &[6.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 96.0);
        assert!(
            (boosted[0] + one_octave + 6.0).abs() < 1e-12,
            "a +6 dB boost must cost 6 dB of margin, got {:?}",
            boosted[0]
        );
        let sensitive = m.margins_db(&f, &[0.0, 0.0, 0.0], &[6.0, 0.0, 0.0], 96.0);
        assert!(
            (sensitive[0] + one_octave + 6.0).abs() < 1e-12,
            "a headphone 6 dB hot at 30 Hz must cost 6 dB of margin, got {:?}",
            sensitive[0]
        );
        let both = m.margins_db(&f, &[6.0, 0.0, 0.0], &[6.0, 0.0, 0.0], 96.0);
        assert!(
            (both[0] + one_octave + 12.0).abs() < 1e-12,
            "the two terms must add, got {:?}",
            both[0]
        );

        // A hole marks its own bin and nothing else.
        let holed = m.margins_db(&f, &[f64::NAN, 0.0, 0.0], &[0.0, 0.0, 0.0], 96.0);
        assert!(holed[0].is_nan(), "a NaN boost must not report a margin");
        assert_eq!((holed[1], holed[2]), (0.0, 0.0), "the other bins stand");

        // An unparsed fs_hz is a hole in the model, not 6 dB of headroom.
        let no_fs = ExcursionModel {
            fs_hz: f64::NAN,
            ref_xmax_spl_db: 96.0,
        };
        assert!(
            no_fs.margins_db(&f, &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 90.0)[0].is_nan(),
            "margin_db would have called this +6 dB of headroom"
        );

        // Zero frequency is a hole too; margin_db's 0.0 guard would read as exactly-at-X_max.
        assert!(
            m.margins_db(&[0.0], &[0.0], &[0.0], 90.0)[0].is_nan(),
            "0 Hz must not report a margin"
        );

        // Shortest vector wins; nothing indexes past the end.
        assert_eq!(m.margins_db(&f, &[0.0, 0.0], &[0.0; 9], 96.0).len(), 2);
        assert_eq!(m.margins_db(&f, &[], &[0.0; 3], 96.0).len(), 0);
    }

    #[test]
    fn distortion_envelope_caps_boost_only_at_high_spl() {
        let prof = DistortionProfile {
            sweeps: vec![
                DistortionSweep {
                    spl_db: 94.0,
                    points: vec![(20.0, 1.5), (1000.0, 0.2), (20000.0, 0.3)],
                },
                DistortionSweep {
                    spl_db: 104.0,
                    points: vec![(20.0, 5.0), (1000.0, 0.4), (20000.0, 0.5)],
                },
            ],
        };
        let f = vec![20.0, 1000.0];
        let h = vec![0.0, 0.0];
        let q_quiet = prof.q_max_bounds(&f, &h, 75.0, 3.0, 12.0);
        let q_loud = prof.q_max_bounds(&f, &h, 92.0, 3.0, 12.0);
        assert_eq!(q_quiet[0], 12.0);
        assert!((q_loud[0] - 2.0).abs() < 1e-9);
    }

    /// The two infinities `max_clean_output_db` returns mean opposite things, so they must not
    /// share a branch: `+INFINITY` is "no distortion data, unconstrained", while `-INFINITY` is
    /// "even the quietest sweep I measured is already over threshold" — the driver that may not
    /// be boosted at all. Reading both as "not finite" handed the worst-case driver the full
    /// hard cap, the exact reward the ceiling exists to withhold.
    #[test]
    fn a_driver_already_over_threshold_at_its_quietest_sweep_may_not_be_boosted() {
        let prof = DistortionProfile {
            sweeps: vec![
                // 20 Hz is already 5 % at 94 dB — the threshold is 3 %.
                DistortionSweep {
                    spl_db: 94.0,
                    points: vec![(20.0, 5.0), (1000.0, 0.2)],
                },
                DistortionSweep {
                    spl_db: 104.0,
                    points: vec![(20.0, 8.0), (1000.0, 0.4)],
                },
            ],
        };
        let f = vec![20.0, 1000.0];
        let h = vec![0.0, 0.0];
        assert!(
            prof.max_clean_output_db(&f, 3.0)[0] == f64::NEG_INFINITY,
            "the sentinel under test is -INFINITY, not a finite SPL"
        );
        let q = prof.q_max_bounds(&f, &h, 85.0, 3.0, 12.0);
        assert_eq!(
            q[0], 0.0,
            "over threshold at the quietest sweep must forbid boost, not grant the 12 dB cap"
        );
        // The same driver is still clean at 1 kHz, so the bound there stays open.
        assert_eq!(q[1], 12.0);
    }

    #[test]
    fn no_distortion_data_at_all_stays_unconstrained() {
        let prof = DistortionProfile::default();
        let f = vec![20.0, 1000.0];
        let h = vec![0.0, 0.0];
        assert!(
            prof.max_clean_output_db(&f, 3.0)
                .iter()
                .all(|v| *v == f64::INFINITY),
            "no sweeps must mean unconstrained, on every bin"
        );
        assert_eq!(prof.q_max_bounds(&f, &h, 85.0, 3.0, 12.0), vec![12.0, 12.0]);
    }

    #[test]
    fn margin_db_returns_nan_on_nan_inputs() {
        let em = ExcursionModel {
            fs_hz: 60.0,
            ref_xmax_spl_db: 100.0,
        };
        assert!(em.margin_db(f64::NAN, 0.0, 80.0).is_nan());
        assert!(em.margin_db(40.0, f64::NAN, 80.0).is_nan());
        assert!(em.margin_db(40.0, 0.0, f64::NAN).is_nan());
    }
}
