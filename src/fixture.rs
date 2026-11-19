//! Which test fixture a curve was measured on, and whether two curves may be paired.
//!
//! A headphone response is not a property of the headphone. It is a property of the headphone on a
//! fixture, and the two halves of that sentence matter in different places: the measurement was made
//! on whatever rig the measurer owned, while a target curve was *defined* on one rig and one only.
//! The paper that made this a published problem for target curves states the failure mode exactly:
//!
//! > Olive & Clark, *A Preferred Headphone Target Curve Defined for the B&K 5128 and GRAS 45CA-10
//! > Test Fixtures*, AES 159th Convention (Long Beach, CA, October 2025), eLibrary `23068`, Express Paper 394 — the
//! > Harman around-/on-ear target was defined on a GRAS 45CA fixture with an RA0045 ("711") ear simulator
//! > and a custom pinna, and is "often misused with other fixtures, leading to measurement and
//! > interpretation errors due to differing acoustic transfer impedances".
//!
//! The database this program ships had no field in which to record any of it: `rig_info` described
//! the rig in prose, and prose cannot be compared. This module gives the fixture a key, so a curve
//! measured on one rig and equalised to a target defined on another can be *detected* rather than
//! shipped.
//!
//! # What is claimed here, and by whom
//!
//! The fixture identities are real products with published identities: GRAS 43AG/43AG-7 is a
//! head-and-torso-free simulator fixture with an ear simulator and a swappable pinna; GRAS 45CA
//! (with an RA0045 "711" coupler and custom pinna) is the fixture the original Harman around-ear work
//! used; GRAS 45CA-10 (with an RA0402 coupler and KB5000/KB5001 anthropometric pinna) is the fixture
//! Olive & Clark (2025) defined equivalent targets for; GRAS KB5000/KB5001 is an anthropometric pinna for
//! KEMAR fixtures (45BA/45BB); B&K 5128 is a head-and-torso simulator conforming to ITU-T P.58 and
//! IEC 60318-7 (with Type 4620 ear simulator conforming to ITU-T P.57 Type 4.3); IEC 60318-4 is the
//! "711" ear simulator alone.
//!
//! What is *not* from any standard is the compatibility rule in [`pair`]. No published source says
//! which fixtures may share a target — the literature's finding is that they largely may not, and
//! per-headline translation is an open research question. So the rule here is deliberately narrow:
//! identical fixtures are fine, two units of the 43AG family are treated as the same rig, and every
//! other declared combination is refused rather than judged. Silence in the table below means "no
//! claim is being made", never "compatible".

/// A test fixture a measurement or a target curve can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Fixture {
    /// GRAS 43AG / 43AG-7 fixture: ear simulator plus a swappable pinna. The fixture behind most of
    /// the community measurements that reach AutoEq. The three shipped models were captured on this
    /// fixture family but declare the *acoustic* ear instead - a KB5000-series pinna on a 711-style
    /// ear, per targets/README.md route 1 - which is what their targets are pairable with; the body
    /// stays in their `rig_info` prose.
    Gras43ag,
    /// A non-GRAS build of the 43AG form factor. Named separately because a clone with a different
    /// pinna is a different acoustic object, and a model that says "clone" should not be recorded as
    /// though it were genuine.
    Gras43agClone,
    /// GRAS 45CA with an RA0045 ("711") ear simulator and custom pinna — the fixture the original
    /// Harman around-ear target family was defined on (distinct from GRAS 45CA-10 with RA0402).
    Gras45ca,
    /// GRAS KB5000 anthropometric pinna / KEMAR fixture.
    GrasKb5000,
    /// Brüel & Kjær 5128 head-and-torso simulator (ITU-T P.58 / IEC 60318-7, Type 4620 ear simulator to ITU-T P.57 Type 4.3).
    Bk5128,
    /// An IEC 60318-4 ("711") ear simulator on its own: no pinna, no torso. Common in simple
    /// measurements, and acoustically far from any of the fixture measurements above.
    Iec60318_4,
}

impl Fixture {
    /// Every fixture, in the order a database error should list them.
    pub const ALL: &[Fixture] = &[
        Fixture::Gras43ag,
        Fixture::Gras43agClone,
        Fixture::Gras45ca,
        Fixture::GrasKb5000,
        Fixture::Bk5128,
        Fixture::Iec60318_4,
    ];

    /// The key a database writes: `"gras_43ag"`.
    pub fn key(self) -> &'static str {
        match self {
            Fixture::Gras43ag => "gras_43ag",
            Fixture::Gras43agClone => "gras_43ag_clone",
            Fixture::Gras45ca => "gras_45ca",
            Fixture::GrasKb5000 => "gras_kb5000",
            Fixture::Bk5128 => "bk5128",
            Fixture::Iec60318_4 => "iec60318_4",
        }
    }

    /// The name a human reads, and the string a model's `name` is required to carry when it declares
    /// this fixture — see [`crate::model`] for the check that keeps the two from drifting apart.
    pub fn label(self) -> &'static str {
        match self {
            Fixture::Gras43ag => "GRAS 43AG",
            Fixture::Gras43agClone => "Clone GRAS 43AG",
            Fixture::Gras45ca => "GRAS 45CA",
            Fixture::GrasKb5000 => "GRAS KB5000",
            Fixture::Bk5128 => "B&K 5128",
            Fixture::Iec60318_4 => "IEC 60318-4",
        }
    }

    /// Read a fixture from a database string. Accepts the key or the label, in any case, so that a
    /// person writing `"fixture": "GRAS 45CA"` is not punished for writing the name they know.
    pub fn parse(text: &str) -> Option<Fixture> {
        let low = text.trim().to_lowercase();
        Fixture::ALL
            .iter()
            .copied()
            .find(|f| f.key() == low || f.label().to_lowercase() == low)
    }

    /// The list for an error message: `gras_43ag, gras_43ag_clone, ...`.
    pub fn keys_joined() -> String {
        Fixture::ALL
            .iter()
            .map(|f| f.key())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// True when two fixtures are the same *class* of rig, even if not the same unit.
    ///
    /// Exactly one pair qualifies: a genuine 43AG and a 43AG clone. They are the same fixture design
    /// with the same ear, differing in pinna and build, which is a difference the report cannot
    /// quantify but which is not the difference Olive & Clark warn about — their result is about
    /// transfer impedance between *different fixture designs*. Every other combination, including
    /// `gras_43ag` against `gras_45ca` (both "a GRAS fixture with a 711 ear", and the single most
    /// likely pairing to get shipped by accident), is left to [`pair`] to refuse.
    pub fn same_rig_family(self, other: Fixture) -> bool {
        use Fixture::*;
        matches!(
            (self, other),
            (Gras43ag, Gras43agClone) | (Gras43agClone, Gras43ag)
        )
    }
}

impl std::fmt::Display for Fixture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// What the two declared fixtures say about each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pairing {
    /// Nothing to compare, because one side records no fixture. This is not agreement: it is the
    /// absence of a claim, and no report may read it as a check that passed.
    Unknown,
    /// The same fixture, key for key.
    Same,
    /// Two units of one fixture design ([`Fixture::same_rig_family`]). Paired without complaint,
    /// because the difference between them is not the difference the literature warns about.
    Adjacent,
    /// Two fixtures whose acoustic transfer impedances differ, as far as anyone has published.
    Different,
}

/// Judge a (measurement fixture, target fixture) pair. `None` on either side means "not recorded",
/// which yields [`Pairing::Unknown`] rather than a guess.
pub fn pair(measured: Option<Fixture>, target: Option<Fixture>) -> Pairing {
    let (Some(m), Some(t)) = (measured, target) else {
        return Pairing::Unknown;
    };
    if m == t {
        Pairing::Same
    } else if Fixture::same_rig_family(m, t) {
        Pairing::Adjacent
    } else {
        Pairing::Different
    }
}

/// Byte ranges of every occurrence of `needle` in `hay`, as `(start, end)`.
///
/// Written out rather than using `match_indices` because the caller needs the *ranges*, to tell an
/// occurrence of one label apart from an occurrence of another that merely contains it.
fn ranges(hay: &str, needle: &str) -> Vec<(usize, usize)> {
    let hay_low = hay.to_ascii_lowercase();
    let needle_low = needle.to_ascii_lowercase();
    let hay_bytes = hay.as_bytes();
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = hay_low[from..].find(&needle_low) {
        let s = from + i;
        let e = s + needle_low.len();
        from = s + 1;
        let start_ok =
            s == 0 || (!hay_bytes[s - 1].is_ascii_alphanumeric() && hay_bytes[s - 1] != b'_');
        let end_ok = e == hay.len()
            || (!hay_bytes[e].is_ascii_alphanumeric()
                && hay_bytes[e] != b'-'
                && hay_bytes[e] != b'_');
        if start_ok && end_ok {
            out.push((s, e));
            from = e;
        }
    }
    out
}

/// The second error this module can raise about a database, after an unparseable key.
///
/// A fixture key and the prose name that describes the model are two places where the same fact can
/// be written, and two places is where drift lives: a model re-measured on a 5128 with its key
/// updated and its name left saying "GRAS 43AG" would tell the user one thing and enforce another.
/// So the key is checked against the name in both directions — the declared label has to appear, and
/// no *other* label may appear except as a fragment of the declared one.
pub fn name_conflict(name: &str, declared: Fixture) -> Option<String> {
    let mine = declared.label();
    let mine_ranges = ranges(name, mine);
    if mine_ranges.is_empty() {
        return Some(format!(
            "the name does not mention it ({mine}). Either name the fixture in the model's \
             \"name\", or drop its \"fixture\" key."
        ));
    }
    for other in Fixture::ALL.iter().copied() {
        if other == declared {
            continue;
        }
        for (s, e) in ranges(name, other.label()) {
            // "GRAS 43AG" inside "Clone GRAS 43AG" is one mention of one fixture, not two.
            let covered = mine_ranges.iter().any(|(ms, me)| *ms <= s && e <= *me);
            if !covered {
                return Some(format!(
                    "the name says {} as well ({:?}). A name that mentions two fixtures does not \
                     say which the measurement was made on.",
                    other.label(),
                    name
                ));
            }
        }
    }
    None
}

/// Broad acoustic construction of the headphone, independent of the rig it was measured on.
///
/// This is the stratification that decides whether a paired rig-to-rig translation curve may be
/// borrowed across two headphones at all: the acoustic leakage path of an open-back and the seal
/// of an in-ear are different objects, so a calibration measured on one cannot be assumed to hold
/// for the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AcousticClass {
    OpenBack,
    ClosedBack,
    InEar,
    #[default]
    Unknown,
}

impl AcousticClass {
    /// Read a class from a database string (`"open"`, `"closed"`, `"in_ear"`, …).
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text
            .trim()
            .to_ascii_lowercase()
            .replace(['-', ' '], "_")
            .as_str()
        {
            "open" | "open_back" | "openback" => Some(Self::OpenBack),
            "closed" | "closed_back" | "closedback" | "sealed" => Some(Self::ClosedBack),
            "in_ear" | "inear" | "iem" | "in_ear_monitor" => Some(Self::InEar),
            "unknown" | "unspecified" | "" => Some(Self::Unknown),
            _ => None,
        }
    }
}

/// An empirical, impedance-stratified translation between two measurement fixtures.
///
/// `delta_mean_db` and `delta_sigma_db` are evaluated on `f_hz`. The mean is what the rig
/// translation adds to the response; the sigma is the uncertainty that translation carries, and it
/// is what the global fit must propagate rather than silently treating the translation as exact.
#[derive(Debug, Clone)]
pub struct RigCalibration {
    pub source: Fixture,
    pub target: Fixture,
    pub acoustic_class: AcousticClass,
    pub f_hz: Vec<f64>,
    pub delta_mean_db: Vec<f64>,
    pub delta_sigma_db: Vec<f64>,
}

impl RigCalibration {
    /// Load an empirical rig translation from JSON:
    /// `{"source": "gras_43ag", "target": "bk5128", "acoustic_class": "closed_back",
    ///   "f_hz": [...], "delta_mean_db": [...], "delta_sigma_db": [...]}`.
    ///
    /// `delta_mean_db` is `response_on_target_rig − response_on_source_rig` for the *same*
    /// headphone; `delta_sigma_db` is its uncertainty, including the spread across units and
    /// acoustic classes. Both are arrays on `f_hz`. No data is embedded: this is user-supplied,
    /// just like the ISO 226 table.
    pub fn from_json_file(path: &std::path::Path) -> crate::error::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let val: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| crate::error::Error::msg(format!("invalid rig calibration JSON: {e}")))?;
        Self::from_value(&val)
    }

    /// Parse a rig calibration from an already-decoded JSON value (inline database entries).
    pub fn from_value(val: &serde_json::Value) -> crate::error::Result<Self> {
        let obj = val
            .as_object()
            .ok_or_else(|| crate::error::Error::msg("rig calibration must be a JSON object"))?;
        let fixture_of = |key: &str| -> crate::error::Result<Fixture> {
            let text = obj
                .get(key)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    crate::error::Error::msg(format!("rig calibration needs a string '{key}'"))
                })?;
            Fixture::parse(text).ok_or_else(|| {
                crate::error::Error::msg(format!(
                    "rig calibration '{key}' names unknown fixture '{text}' (choose from {})",
                    Fixture::keys_joined()
                ))
            })
        };
        let source = fixture_of("source")?;
        let target = fixture_of("target")?;
        let acoustic_class = match obj.get("acoustic_class") {
            None | Some(serde_json::Value::Null) => AcousticClass::Unknown,
            Some(serde_json::Value::String(text)) => {
                AcousticClass::parse(text).ok_or_else(|| {
                    crate::error::Error::msg(format!(
                        "rig calibration names unknown acoustic_class '{text}'"
                    ))
                })?
            }
            Some(other) => {
                return Err(crate::error::Error::msg(format!(
                    "rig calibration 'acoustic_class' must be a string, got {other}"
                )))
            }
        };
        let numbers = |key: &str| -> crate::error::Result<Vec<f64>> {
            let arr = obj
                .get(key)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    crate::error::Error::msg(format!("rig calibration needs an array '{key}'"))
                })?;
            arr.iter()
                .map(|v| {
                    v.as_f64().ok_or_else(|| {
                        crate::error::Error::msg(format!(
                            "rig calibration '{key}' holds a non-numeric value: {v}"
                        ))
                    })
                })
                .collect()
        };
        let f_hz = numbers("f_hz")?;
        let delta_mean_db = numbers("delta_mean_db")?;
        let delta_sigma_db = numbers("delta_sigma_db")?;
        if f_hz.len() < 2
            || delta_mean_db.len() != f_hz.len()
            || delta_sigma_db.len() != f_hz.len()
            || f_hz.windows(2).any(|w| w[1] <= w[0])
            || f_hz.iter().any(|&f| !f.is_finite() || f <= 0.0)
            || delta_mean_db
                .iter()
                .chain(&delta_sigma_db)
                .any(|v| !v.is_finite())
        {
            return Err(crate::error::Error::msg(
                "rig calibration needs >= 2 ascending positive frequencies and equal-length finite delta arrays",
            ));
        }
        Ok(Self {
            source,
            target,
            acoustic_class,
            f_hz,
            delta_mean_db,
            delta_sigma_db,
        })
    }

    /// Interpolate the calibration (mean and sigma) onto a query grid, clamped at the ends.
    #[must_use]
    pub fn on_grid(&self, f_grid: &[f64]) -> (Vec<f64>, Vec<f64>) {
        if self.f_hz.len() < 2
            || self.delta_mean_db.len() != self.f_hz.len()
            || self.delta_sigma_db.len() != self.f_hz.len()
        {
            return (vec![0.0; f_grid.len()], vec![0.0; f_grid.len()]);
        }
        (
            crate::fft::interp_grid(f_grid, &self.f_hz, &self.delta_mean_db),
            crate::fft::interp_grid(f_grid, &self.f_hz, &self.delta_sigma_db),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rig_calibration_parses_and_interpolates_its_delta_curves() {
        let val = serde_json::json!({
            "source": "gras_43ag",
            "target": "bk5128",
            "acoustic_class": "closed_back",
            "f_hz": [20.0, 1000.0, 20000.0],
            "delta_mean_db": [3.0, 1.0, -2.0],
            "delta_sigma_db": [0.5, 0.3, 0.6],
        });
        let cal = RigCalibration::from_value(&val).expect("valid calibration");
        assert_eq!(cal.source, Fixture::Gras43ag);
        assert_eq!(cal.target, Fixture::Bk5128);
        assert_eq!(cal.acoustic_class, AcousticClass::ClosedBack);
        let (mean, sigma) = cal.on_grid(&[20.0, 1000.0, 20000.0]);
        assert_eq!(mean, vec![3.0, 1.0, -2.0]);
        assert_eq!(sigma, vec![0.5, 0.3, 0.6]);
        let (mid_mean, _) = cal.on_grid(&[500.0]);
        assert!(mid_mean[0] > 1.0 && mid_mean[0] < 3.0, "{}", mid_mean[0]);

        let bad = serde_json::json!({
            "source": "gras_43ag", "target": "bk5128",
            "f_hz": [20.0, 1000.0], "delta_mean_db": [1.0], "delta_sigma_db": [0.3, 0.4],
        });
        assert!(
            RigCalibration::from_value(&bad).is_err(),
            "length mismatch must be refused"
        );
    }

    #[test]
    fn acoustic_class_parses_the_spellings_a_database_would_use() {
        assert_eq!(
            AcousticClass::parse("open_back"),
            Some(AcousticClass::OpenBack)
        );
        assert_eq!(
            AcousticClass::parse(" OPEN-BACK "),
            Some(AcousticClass::OpenBack)
        );
        assert_eq!(
            AcousticClass::parse("closed"),
            Some(AcousticClass::ClosedBack)
        );
        assert_eq!(AcousticClass::parse("IEM"), Some(AcousticClass::InEar));
        assert_eq!(
            AcousticClass::parse("unspecified"),
            Some(AcousticClass::Unknown)
        );
        assert_eq!(
            AcousticClass::parse("unknown"),
            Some(AcousticClass::Unknown)
        );
        assert_eq!(AcousticClass::parse("gibberish"), None);
        assert_eq!(AcousticClass::default(), AcousticClass::Unknown);
    }

    #[test]
    fn every_fixture_round_trips_through_its_key() {
        for &f in Fixture::ALL {
            assert_eq!(Fixture::parse(f.key()), Some(f), "key {:?}", f.key());
            assert_eq!(Fixture::parse(f.label()), Some(f), "label {:?}", f.label());
        }
    }

    #[test]
    fn a_key_is_read_in_any_case_and_ignoring_surrounding_space() {
        // A database is hand-edited. Refusing "GRAS_45CA " is not a safety check, it is a chore.
        assert_eq!(Fixture::parse("  GRAS_45CA "), Some(Fixture::Gras45ca));
        assert_eq!(
            Fixture::parse("b&k 5128"),
            Some(Fixture::Bk5128),
            "the label is read case-insensitively too, not only the key"
        );
        assert_eq!(
            Fixture::parse("5128"),
            None,
            "a fragment of a label is not a fixture: matching is whole-string"
        );
    }

    #[test]
    fn an_unknown_fixture_is_not_silently_read_as_no_fixture() {
        // The difference matters: "no fixture" means no check is made, and a typo must not buy that.
        assert_eq!(Fixture::parse("gras_45cb"), None);
        assert_eq!(Fixture::parse(""), None);
        assert!(Fixture::keys_joined().contains("gras_45ca"));
    }

    #[test]
    fn no_label_is_a_substring_of_another_except_the_clone() {
        // `name_conflict` reasons about containment; if two labels were unrelated strings the
        // reasoning would be vacuous and the test below would be testing nothing.
        let clone = Fixture::Gras43agClone.label();
        let genuine = Fixture::Gras43ag.label();
        assert!(clone.contains(genuine));
        for &a in Fixture::ALL {
            for &b in Fixture::ALL {
                if a != b && !(a == Fixture::Gras43ag && b == Fixture::Gras43agClone) {
                    assert!(
                        !b.label().contains(a.label()),
                        "{:?} is contained in {:?}",
                        a.label(),
                        b.label()
                    );
                }
            }
        }
    }

    #[test]
    fn a_declared_fixture_must_be_named_in_the_name_too() {
        let name = "Beyerdynamic DT 770 Pro (80 Ω) [GRAS 45CA]";
        assert_eq!(name_conflict(name, Fixture::Gras45ca), None);
        let name_lower = "Beyerdynamic DT 770 Pro (80 Ω) [gras 45ca]";
        assert_eq!(name_conflict(name_lower, Fixture::Gras45ca), None);
        let name_45ca10 = "Beyerdynamic DT 770 Pro (80 Ω) [GRAS 45CA-10]";
        assert!(name_conflict(name_45ca10, Fixture::Gras45ca).is_some());
        let drift = name_conflict("Beyerdynamic DT 770 Pro (80 Ω)", Fixture::Gras45ca);
        assert!(
            drift.as_deref().is_some_and(|m| m.contains("GRAS 45CA")),
            "a key with no matching name was not reported: {drift:?}"
        );
    }

    #[test]
    fn a_name_mentioning_two_fixtures_says_nothing_useful() {
        let both = "FiiO FT1 Pro [GRAS 43AG] [B&K 5128]";
        let msg = name_conflict(both, Fixture::Gras43ag);
        assert!(
            msg.as_deref().is_some_and(|m| m.contains("B&K 5128")),
            "the second fixture was not reported: {msg:?}"
        );
    }

    #[test]
    fn a_clone_mention_is_not_a_second_fixture_for_a_genuine_claims_inverse() {
        // Declaring the clone and mentioning "GRAS 43AG" inside its own label is one mention.
        let clone = "JBL Live 770NC [Clone GRAS 43AG]";
        assert_eq!(name_conflict(clone, Fixture::Gras43agClone), None);
        // But declaring genuine while the name says "Clone" overstates the rig.
        let overstated = name_conflict(clone, Fixture::Gras43ag);
        assert!(
            overstated
                .as_deref()
                .is_some_and(|m| m.contains("Clone GRAS 43AG")),
            "a genuine claim on a clone fixture passed: {overstated:?}"
        );
    }

    #[test]
    fn an_unrecorded_side_is_no_claim_rather_than_agreement() {
        assert_eq!(pair(Some(Fixture::Gras45ca), None), Pairing::Unknown);
        assert_eq!(pair(None, Some(Fixture::Gras45ca)), Pairing::Unknown);
        assert_eq!(pair(None, None), Pairing::Unknown);
    }

    #[test]
    fn the_pairing_that_matters_is_refused_and_the_one_that_does_not_is_not() {
        // The Olive & Clark case: a 45CA-defined target, a 43AG measurement. Both "GRAS with a 711".
        assert_eq!(
            pair(Some(Fixture::Gras43ag), Some(Fixture::Gras45ca)),
            Pairing::Different
        );
        assert_eq!(
            pair(Some(Fixture::Iec60318_4), Some(Fixture::Bk5128)),
            Pairing::Different
        );
        assert_eq!(
            pair(Some(Fixture::Gras45ca), Some(Fixture::Gras45ca)),
            Pairing::Same
        );
        assert_eq!(
            pair(Some(Fixture::Gras43ag), Some(Fixture::Gras43agClone)),
            Pairing::Adjacent
        );
    }

    #[test]
    fn the_family_rule_is_the_one_pair_and_is_symmetric() {
        // If this grows, the claim in `same_rig_family`'s doc ("exactly one pair") is false.
        let mut families = 0;
        for &a in Fixture::ALL {
            for &b in Fixture::ALL {
                if a != b && Fixture::same_rig_family(a, b) {
                    assert_eq!(
                        Fixture::same_rig_family(b, a),
                        Fixture::same_rig_family(a, b)
                    );
                    families += 1;
                }
            }
        }
        assert_eq!(families, 2, "43ag/clone both directions, and nothing else");
    }
}
