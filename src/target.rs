//! Parameterized target families, spatial-audio slope (Mülleder et al. 2025), and ISO 226:2023
//! Formula (1) contour deltas.
//!
//! Preference learning is not part of this module: a Bradley–Terry IRLS fit for pairwise preset
//! choices lived here until it was removed, and nothing in the tool ranks presets, so it had no
//! caller and no CLI surface. Re-add it with a caller, not on its own.

use crate::pchip::Pchip;
use std::path::Path;

/// Selectable target family profiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetFamily {
    Harman,
    DiffuseField,
    /// Diffuse field + ~ -1.0 dB/octave spectral slope for spatial audio (Mülleder et al., AES 2025).
    SpatialDiffuseSlope,
}

/// Continuous listener preference modifiers over any baseline target curve.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TargetParams {
    /// Bass shelf gain around 105 Hz (dB).
    pub theta_bass: f64,
    /// Presence / ear-gain high-shelf trim around 2.5 kHz (dB).
    pub theta_mid: f64,
    /// Log-frequency tilt (dB/octave, anchored to 0 dB at 1 kHz).
    pub theta_tilt: f64,
}

impl TargetParams {
    /// The three basis features `[shelf, bell, tilt]` at `f_hz`.
    ///
    /// A finite but non-positive frequency is floored to 1 nHz before the logarithms. `log2(0)` is
    /// `-inf`, and IEEE-754 makes `0.0 * -inf` a NaN — so a baseline CSV with a row at 0 Hz poisoned
    /// every knot of the target curve *even when `theta_tilt` was zero*, which is the usual case and
    /// the one that looks safest. The floor keeps the shelf at its `f -> 0` limit (1), the bell at
    /// its (0), and the tilt at a large but finite slope.
    ///
    /// A non-finite frequency is passed through, not invented: a NaN in the axis is bad input, and
    /// substituting a plausible number for it would hide the row that caused it.
    #[must_use]
    pub fn features(f_hz: f64) -> [f64; 3] {
        // Smooth 2nd-order-like low-shelf basis, 2nd-order high-shelf ear-gain trim, and octave tilt.
        let f = if f_hz.is_finite() {
            f_hz.max(1e-9)
        } else {
            f_hz
        };
        let x_bass = (f / 105.0).powi(2);
        let b = 1.0 / (1.0 + x_bass);
        let x_mid = (f / 2500.0).powi(2);
        let m = x_mid / (1.0 + x_mid);
        let s = (f / 1000.0).log2();
        [b, m, s]
    }

    /// The parameterized deviation from the baseline, in dB. Finite for every finite frequency.
    #[must_use]
    pub fn delta_db(&self, f_hz: f64) -> f64 {
        let [b, m, s] = Self::features(f_hz);
        self.theta_bass * b + self.theta_mid * m + self.theta_tilt * s
    }
}

/// Evaluate `T(f; theta)` over `f_grid` from a baseline curve and optional spatial slope.
#[must_use]
pub fn evaluate_target(
    base_pts: &[(f64, f64)],
    f_grid: &[f64],
    family: TargetFamily,
    params: TargetParams,
) -> Vec<f64> {
    if base_pts.len() < 2 {
        return vec![0.0; f_grid.len()];
    }
    let (fs, gs): (Vec<f64>, Vec<f64>) = base_pts.iter().copied().unzip();
    let base = Pchip::new(&fs, &gs).eval_grid(f_grid);
    let spatial_slope = match family {
        TargetFamily::SpatialDiffuseSlope => -1.0,
        _ => 0.0,
    };
    f_grid
        .iter()
        .zip(base)
        .map(|(&f, b)| {
            let slope = if f > 0.0 {
                spatial_slope * (f / 1000.0).log2()
            } else {
                0.0
            };
            b + slope + params.delta_db(f)
        })
        .collect()
}

/// One row of ISO 226:2023 Table 1 (supplied externally by the user; ISO table data is
/// copyrighted by ISO and is deliberately not embedded in this MIT crate).
#[derive(Clone, Copy, Debug)]
pub struct Iso226Entry {
    pub freq_hz: f64,
    pub alpha_f: f64,
    pub l_u: f64,
    pub t_f: f64,
}

/// External ISO 226:2023 Table 1 loader and Formula (1) contour-delta evaluator.
#[derive(Clone, Debug)]
pub struct Iso226Table {
    entries: Vec<Iso226Entry>,
}

impl Iso226Table {
    pub fn from_entries(mut entries: Vec<Iso226Entry>) -> Option<Self> {
        entries.retain(|e| {
            e.freq_hz.is_finite()
                && e.freq_hz > 0.0
                && e.alpha_f.is_finite()
                && e.alpha_f > 0.0
                && e.l_u.is_finite()
                && e.t_f.is_finite()
        });
        entries.sort_by(|a, b| a.freq_hz.total_cmp(&b.freq_hz));
        // Duplicate frequencies survive a hand-edited table (the same row pasted twice with
        // updated constants). PCHIP needs strictly ascending knots: a repeated frequency is a
        // zero-width interval, and its secant slope — the harmonic-mean tangent and every
        // evaluation that follows — turns to inf/NaN. Collapse runs to their last row, the
        // same "last write wins" the target-CSV loader uses.
        entries.reverse();
        entries.dedup_by(|a, b| a.freq_hz == b.freq_hz);
        entries.reverse();
        (entries.len() >= 2).then_some(Self { entries })
    }

    /// Load user-provided JSON array of `{"freq_hz": ..., "alpha_f": ..., "l_u": ..., "t_f": ...}`.
    pub fn from_json_file(path: &Path) -> crate::error::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let val: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| crate::error::Error::msg(format!("invalid ISO 226 JSON: {e}")))?;
        let arr = val
            .as_array()
            .ok_or_else(|| crate::error::Error::msg("ISO 226 file must be a JSON array"))?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            let f = item.get("freq_hz").and_then(|v| v.as_f64());
            let a = item.get("alpha_f").and_then(|v| v.as_f64());
            let u = item.get("l_u").and_then(|v| v.as_f64());
            let t = item.get("t_f").and_then(|v| v.as_f64());
            if let (Some(freq_hz), Some(alpha_f), Some(l_u), Some(t_f)) = (f, a, u, t) {
                out.push(Iso226Entry {
                    freq_hz,
                    alpha_f,
                    l_u,
                    t_f,
                });
            }
        }
        let table = Self::from_entries(out).ok_or_else(|| {
            crate::error::Error::msg("ISO 226 file holds fewer than 2 valid rows")
        })?;
        // The contour compensation is *defined* relative to 1 kHz — that is the unit-gain
        // point of the equal-loudness construction. A table that does not reach 1 kHz
        // leaves `p.eval(1000.0)` reading PCHIP's flat end clamp, and the subtraction then
        // normalises the whole curve to a fabricated reference taken at the table's edge.
        // Refuse the table instead of shipping the fiction.
        let lo = table.entries[0].freq_hz;
        let hi = table.entries[table.entries.len() - 1].freq_hz;
        if lo > 1000.0 || hi < 1000.0 {
            return Err(crate::error::Error::msg(format!(
                "ISO 226 file spans {lo:.0} Hz .. {hi:.0} Hz and does not cover 1 kHz; the \
                 contour delta is normalised there"
            )));
        }
        Ok(table)
    }

    /// ISO 226:2023 Formula (1) pure-tone SPL (dB) at loudness level `phon`.
    #[must_use]
    pub fn spl_at_entry(e: &Iso226Entry, phon: f64) -> f64 {
        const ALPHA_R: f64 = 0.30;
        const T_R: f64 = 2.4;
        const P_RATIO_SQ: f64 = 4.0e-10; // (20 uPa / 1 Pa)^2
        let term1 = P_RATIO_SQ.powf(ALPHA_R - e.alpha_f)
            * (10f64.powf(ALPHA_R * phon / 10.0) - 10f64.powf(ALPHA_R * T_R / 10.0));
        let term2 = 10f64.powf(e.alpha_f * (e.t_f + e.l_u) / 10.0);
        let a_f = (term1 + term2).max(1e-30);
        (10.0 / e.alpha_f) * a_f.log10() - e.l_u
    }

    /// Contour-delta compensation `(L_p(listen, f) - L_p(listen, 1k)) - (L_p(ref, f) - L_p(ref, 1k))`,
    /// interpolated in `log10(f)` and clamped flat outside the tabulated band.
    #[must_use]
    pub fn contour_delta_db(&self, f_grid: &[f64], listen_phon: f64, ref_phon: f64) -> Vec<f64> {
        let fs: Vec<f64> = self.entries.iter().map(|e| e.freq_hz).collect();
        let diff_at_knots: Vec<f64> = self
            .entries
            .iter()
            .map(|e| {
                let l_cur = Self::spl_at_entry(e, listen_phon);
                let l_ref = Self::spl_at_entry(e, ref_phon);
                l_cur - l_ref
            })
            .collect();
        let p = Pchip::new(&fs, &diff_at_knots);
        let at_1k = p.eval(1000.0);
        p.eval_grid(f_grid).into_iter().map(|v| v - at_1k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `log2(0)` is `-inf`, and IEEE-754 makes `0.0 * -inf` a NaN, so the tilt term poisoned the whole
    /// target curve at a 0 Hz row even when the operator never asked for a tilt.
    #[test]
    fn a_non_positive_frequency_does_not_poison_the_target_deviation() {
        let p = TargetParams {
            theta_bass: 6.4,
            theta_mid: -1.4,
            theta_tilt: 0.0,
        };
        for f in [0.0, -1.0, -20_000.0] {
            let d = p.delta_db(f);
            assert!(
                d.is_finite(),
                "delta_db({f}) = {d}; the shelf must stay finite"
            );
        }
        let [b, m, s] = TargetParams::features(0.0);
        assert!(
            (b - 1.0).abs() < 1e-9,
            "the shelf saturates to 1 as f -> 0, got {b}"
        );
        assert!(
            m.abs() < 1e-12,
            "the high-shelf vanishes as f -> 0, got {m}"
        );
        let [_, m_mid, _] = TargetParams::features(2500.0);
        assert!(
            (m_mid - 0.5).abs() < 1e-12,
            "high-shelf is 0.5 at 2.5 kHz, got {m_mid}"
        );
        let [_, m_inf, _] = TargetParams::features(1e9);
        assert!(
            (m_inf - 1.0).abs() < 1e-6,
            "high-shelf saturates to 1 as f -> inf, got {m_inf}"
        );
        assert!(s < -30.0 && s.is_finite(), "the tilt is finite, got {s}");
        // Bad input stays visible instead of being invented as something plausible.
        assert!(
            p.delta_db(f64::NAN).is_nan(),
            "a NaN frequency must not be silenced"
        );
        // The published anchors are untouched by the guard.
        let tilt = TargetParams {
            theta_bass: 0.0,
            theta_mid: 0.0,
            theta_tilt: 3.0,
        };
        assert!(
            tilt.delta_db(1000.0).abs() < 1e-12,
            "tilt is anchored at 1 kHz"
        );
        assert!(
            (tilt.delta_db(2000.0) - 3.0).abs() < 1e-12,
            "one octave up is one theta, got {}",
            tilt.delta_db(2000.0)
        );
        // And through the curve: a query grid that touches 0 Hz yields a usable target.
        let base = vec![(20.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)];
        let out = evaluate_target(
            &base,
            &[0.0, 20.0, 1000.0, 20000.0],
            TargetFamily::SpatialDiffuseSlope,
            p,
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite target: {out:?}"
        );
    }

    #[test]
    fn iso226_2023_formula1_passes_through_phon_at_1khz() {
        let e = Iso226Entry {
            freq_hz: 1000.0,
            alpha_f: 0.30,
            l_u: 0.0,
            t_f: 2.4,
        };
        for phon in [20.0, 40.0, 65.0, 83.0] {
            let spl = Iso226Table::spl_at_entry(&e, phon);
            assert!((spl - phon).abs() < 1e-12, "{spl} vs {phon}");
        }
    }

    /// A hand-edited table can carry the same frequency twice; PCHIP needs strictly
    /// ascending knots and a repeated one turns every slope, tangent and evaluation that
    /// follows into inf/NaN. Rows collapse last-write-wins, like the CSV loaders.
    #[test]
    fn a_repeated_frequency_row_collapses_instead_of_poisoning_the_interpolant() {
        let mut dupe: Vec<Iso226Entry> = [(20.0f64, 2.0f64), (1000.0, 0.0), (20000.0, 1.5)]
            .iter()
            .map(|&(f, l)| Iso226Entry {
                freq_hz: f,
                alpha_f: 0.3,
                l_u: l,
                t_f: 2.4,
            })
            .collect();
        dupe.push(Iso226Entry {
            freq_hz: 1000.0,
            alpha_f: 0.3,
            l_u: 5.0,
            t_f: 2.4,
        });
        let t = Iso226Table::from_entries(dupe).expect("three distinct frequencies");
        assert_eq!(t.entries.len(), 3, "the duplicate must collapse");
        assert_eq!(t.entries[1].l_u, 5.0, "the later row must win");
        let d = t.contour_delta_db(&[100.0, 1000.0, 4000.0], 65.0, 40.0);
        assert!(d.iter().all(|v| v.is_finite()), "poisoned: {d:?}");
    }

    /// The contour delta is normalised at 1 kHz; a table that does not reach it would
    /// normalise against PCHIP's flat end clamp — a reference the table never measured.
    #[test]
    fn a_table_that_does_not_reach_1khz_is_refused_not_normalised_to_its_edge() {
        let dir = std::env::temp_dir().join(format!("hypreq-iso226-span-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("no-1k.json");
        std::fs::write(
            &path,
            r#"[{"freq_hz":100.0,"alpha_f":0.3,"l_u":2.0,"t_f":2.4},
                {"freq_hz":500.0,"alpha_f":0.3,"l_u":1.0,"t_f":2.4},
                {"freq_hz":900.0,"alpha_f":0.3,"l_u":0.5,"t_f":2.4}]"#,
        )
        .unwrap();
        let e = Iso226Table::from_json_file(&path).unwrap_err();
        assert!(e.to_string().contains("1 kHz"), "{e}");
        std::fs::write(
            &path,
            r#"[{"freq_hz":100.0,"alpha_f":0.3,"l_u":2.0,"t_f":2.4},
                {"freq_hz":1000.0,"alpha_f":0.3,"l_u":0.0,"t_f":2.4}]"#,
        )
        .unwrap();
        assert!(Iso226Table::from_json_file(&path).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
