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
    /// the community measurements that reach AutoEq, including all three shipped models.
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
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = hay[from..].find(needle) {
        out.push((from + i, from + i + needle.len()));
        from += i + needle.len();
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

#[cfg(test)]
mod tests {
    use super::*;

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
