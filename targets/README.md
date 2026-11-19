# Rig-specific target recipes

This directory holds **recipes, not curves**. Published target tables and measurement-floor
responses are copyrighted or licensed material (ISO, AES/Harman, B&K), so this MIT crate stores
only the numeric constants that are reported in the literature and reads the actual curve from a
CSV that **you** supply beside the recipe.

## Harman Beta 2024 for the B&K 5128

`harman_beta_2024_5128.json` records the constants reported from Sean Olive's CanJam NYC 2024
pilot: a B&K 5128 diffuse-field baseline with a **+6.4 dB shelf below 105 Hz** and a
**−1.4 dB shelf above 2.5 kHz** ([SoundGuys report](https://www.soundguys.com/harman-research-targets-the-bruel-kjaer-5128-112247/)).

The baseline is `bk5128_df.csv` relative to this directory, with two numeric columns:

```
20,0.0
25,0.4
...
20000,-12.0
```

Supply the 5128 DF response **for the exact fixture and pinna you measured on**. Do not fabricate
one from the Harman 45CA curve — the whole point of a rig-specific target is that the two are not
interchangeable.

If you have not measured a 5128 diffuse field of your own, an honest stand-in is the community
5128 diffuse-field response distributed as `Diffuse field 5128.csv` in the AutoEq repository
(jaakkopasanen/AutoEq, MIT; 696 points, 20 Hz - 19.96 kHz, the diffuse-field target behind
headphones.com's and Crinacle's 5128 databases): download it into this directory under the name
`bk5128_df.csv`. It stands in for the pilot's baseline, which was never published, and carries
its own measurement chain; the repo does not track it, like every CSV here.

With the baseline in place:

```sh
hypreq -H your_model --target-preset targets/harman_beta_2024_5128.json
```

The report prints the preset's name, declared rig, and source under the model heading. The
published shelf frequencies coincide with this crate's `TargetParams` bass and presence basis;
the basis *shape* is ours, not a reproduction of the published curve.

## KB5000_7 for the GRAS KB5000 pinna (Olive & Clark 2025)

`harman_kb5000_7.json` points at the curve attached to Sean Olive and Dan Clark's *A Preferred
Headphone Target Curve Defined for the B&K 5128 and GRAS45CA-10 Test Fixtures* (Express Paper 394,
AES 159th Convention, Long Beach, 2025 Oct 23–25, eLibrary `23068`), which the authors distribute
on Head-Fi together with the write-up. The paper's own framing, from its abstract: the Harman
around-/on-ear target was defined "using a GRAS 45CA test fixture with a RA045 ("711") ear
simulator and custom pinna" and is often misused on other fixtures; the 2025 work derives
equivalent targets for the B&K 5128 and the GRAS 45CA-10. The attachment — the **KB5000_7** curve,
the Harman target translated onto the KB5000 pinna — is published *as data* by its authors, but it
is still their measurement data: this recipe keeps the same recipes-not-curves rule and reads it
from `kb5000_7.csv` beside the recipe. Produce that CSV from your own copy of the attachment:

```sh
python3 scripts/import_kb5000_7.py /path/to/your/KB5000_7_export.txt targets/kb5000_7.csv
```

All `params` in the recipe are `0.0`: the published curve already contains its shelf structure, and
`family: "harman"` adds no slope (only `spatial` does, −1 dB/octave). The target is defined on a
GRAS 45CA-10 with the **KB5000** pinna (a 35 Shore OO anthropometric pinna that GRAS sells for the
45BB/45BC KEMAR and the 43AG fixture, mounted on a 711-style IEC 60318-4 ear simulator), so the
recipe declares `fixture: "gras_kb5000"`. Check your reading of the authors' terms at their own
post before redistributing the CSV — the repo neither hosts nor embeds it.

The post's own attachment list (web.archive snapshot of 2026-04-30, checked 2026-11-19 — Head-Fi
refuses direct fetches) carries exactly one curve file, `KB5000_7 any frequency.txt`, and no
licence text with it: that is the only target the authors have published *as data* so far. The
paper's other curves — the single-headphone calibrations and the `_7`/`_20` averaged variants for
both fixtures — exist as figures only, and no third-party data publication for them was found
either. There is therefore no recipe here for them, and digitizing the paper's figures is not an
acceptable route to one: they would be copyrighted curves with unexamined terms, stripped of the
per-headphone scatter the paper itself calls the finding.

## Which target belongs on which rig

The fixture keys are the ones the `fixture` module enforces. "Published target" means a target a
source defines *on that fixture*, not a curve people compare to from that fixture.

| Fixture key | Published target defined there | Defined on | Repo support |
| --- | --- | --- | --- |
| `gras_45ca` | Harman around-/on-ear target family | GRAS 45CA + RA045/RA0045-S1 ("711") + the custom Welti pinna | built-in `harman` family constants; no shipped curve data |
| `gras_kb5000` | **KB5000_7** (Harman equivalent translated to the KB5000 pinna) | GRAS 45CA-10 + KB5000 pinna | `harman_kb5000_7.json` - default recipe of the three shipped models; your CSV |
| `bk5128` | **Harman Beta 2024** (shelf constants over a 5128 diffuse-field baseline); Olive & Clark's 5128 side | B&K 5128 (Type 4.3 ear) | `harman_beta_2024_5128.json`; treat 5128 targets as provisional, below |
| `gras_43ag` | **none** | — | see the three honest routes, below |
| `gras_43ag_clone` | **none** | — | same as `gras_43ag` |
| `iec60318_4` | **none** — a bare 711 coupler has no pinna; comparing it to a Harman curve is the misuse the 2025 paper names | — | no recipe can fix that pairing; re-measure with the pinna your target assumes |

**The shipped models take route 1 (2026-11-19).** `jbl770nc`, `dt770_80` and `ft1pro` declare
`fixture: "gras_kb5000"` — their KB5000-family pinna on a 711-style ear is the acoustic object;
the 43AG or clone body stays in the `rig_info` prose — and they record
`"target_preset": "targets/harman_kb5000_7.json"`, so an ordinary run fits their capture to the
published curve and the pairing is `Same`, no override. Two honest edges. The recipe's baseline
CSV is still user data: without `kb5000_7.csv` a plain run stops with a message naming the path,
and `--no-target-preset` — or a `--target-csv` of your own — falls back to the legacy built-in
curve, which is the pairing the models' `fixture_conflict_ok` now records. And `jbl770nc`'s pinna
is a KB006x *clone*, not a genuine GRAS KB5000: the declaration names the nearest published
acoustic object, not an identity — the paper's scatter figures do not cover clone hardware.

**Why 5128 recipes are provisional.** In the authors' own listening tests (WebMUSHRA, 13
listeners) the 5128-derived targets scattered: their figures show the 7-headphone 5128 target with
a mean ±1 SD spread around 2.5 dB versus roughly 1 dB for KB5000_7, and the write-up states that
"generating a general 5128 target is not possible at this time". A 5128 recipe is a starting
point, not a verdict.

**No published Harman equivalent exists for the 43AG family.** The 43AG is a swappable-pinna
fixture, and community databases' habit of judging 43AG captures against the Harman 2018 curve is
exactly what Paper 394 calls misuse. Three routes that do not lie:

1. **Record the acoustic ear, then pair natively.** A 43AG body carrying a KB5000-series pinna on a
   711-style ear *is* the KB5000 acoustic object — declare `fixture: "gras_kb5000"` on such
   captures (name the declared fixture in the model's `name`, e.g. `[GRAS KB5000]`, and keep the
   43AG body details in the `rig_info` prose; the name fixture check reads the *acoustic* fixture),
   and `harman_kb5000_7.json` pairs with it as `Same`.
2. **Translate with your own data.** A paired measurement of the same unit on two rigs goes in a
   `--rig-calibration` JSON (`source`/`target` fixtures plus `delta_mean_db`/`delta_sigma_db`);
   the fit carries the sigma instead of pretending the translation is exact. That record documents
   the translation; it does not unlock the pairing by itself — the guard below does.
3. **Accept the mismatch knowingly, in the scope each record has.** `fixture_conflict_ok` on the
   model vouches for that model's *shipped* target only — a reason recorded there does not follow
   the model onto a `--target-preset` the database has never seen, which is refused until a one-shot
   `--force`. The guard refuses silent cross-fixture pairing by design: its table tolerates
   identical fixtures and the 43AG/clone pair and nothing else, and it stays that narrow until a
   published source justifies more.

## Rules for adding a recipe

Write a JSON beside these files with `name`, `fixture` (the rig the curve was *defined* on),
`family`, `baseline_csv` (your curve, relative to the recipe), `params`, and `source`, then add a
row to the table above. Keep `bass`/`mid`/`tilt` at `0.0` when the baseline CSV already contains
the published shelf structure — shelves re-added on top of themselves are double shelves. If the
source publishes uncertainty (scatter across headphones or units), record the number in `note`;
a target without its scatter is half a citation. Never embed a copyrighted or paywalled curve:
the constants and the provenance live in the recipe, the data lives in a file the user supplies.
And never widen the fixture compatibility table in `src/fixture.rs` without a published source and
its own discriminating test — `the_family_rule_is_the_one_pair_and_is_symmetric` fails if the
table grows without its claim being re-checked.

## Rules for a recipe file

| Field | Meaning |
| --- | --- |
| `name` | Printed in the report. |
| `fixture` | Fixture key the target was defined on (`bk5128`, `gras_45ca`, …). |
| `family` | `harman`, `diffuse`, or `spatial` — the optional slope modifier applied on top. |
| `baseline_csv` | Path to your curve, relative to the recipe file. |
| `params.bass` / `params.mid` / `params.tilt` | Shelf/tilt constants, in dB and dB/octave. |
| `source` | Citation printed in the report; required for auditability. |

`--target-bass`, `--target-mid` and `--target-tilt` on the command line override a recipe's
corresponding value, so a listener can start from a published recipe and adjust it without
editing the JSON.
