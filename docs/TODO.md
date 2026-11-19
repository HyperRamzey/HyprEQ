# Working TODO (this session's contract + next project)

Order is binding: Phase A fully closes the HyprEQ defect/curve contract (every item shown
against the working tree, per AGENTS.md measure-the-claim), **then** Phase B starts.
Nothing here authorizes a commit or push.

## Phase A — HyprEQ, remaining contract items

- [x] A1. **Per-fixture target recipes ("correct curves per supported pair", future-proof).**
  - Done 2026-11-19: recipe `targets/harman_kb5000_7.json` + importer + README per-fixture table;
    guard smoke 4 cases (native pass prints the preset line; 43AG refused rc=2; `--force` ok;
    preset conflict NOT unlocked by `fixture_conflict_ok` — code-verified scope at main.rs:800).
    That smoke found the refusal text recommending the dead-end route for presets: message is now
    branch-scoped + two new tests (`a_preset_fixture_conflict_is_refused_without_a_dead_end_suggestion`,
    `a_recorded_conflict_ok_unlocks_the_shipped_pairing_only`), both sabotage-proved (fail on
    `if false` variants; main.rs restored md5 `6db7d77b…`). `targets/kb5000_7.csv` generated from
    the user's export (1015 pts, 13.02–19688.4 Hz, +0.163 dB @ 998 Hz) and gitignored
    (`/targets/*.csv` rule).
  - Evidence base (verified this session, none of it embedded): Olive & Clark Express Paper
    394, AES 159th (2025-10-23/25, pp. 394:1-8) defines preferred targets for **B&K 5128 and
    GRAS 45CA-10** fixtures; original Harman OE target defined on GRAS 45CA + RA045("711")/
    RA0045-S1 Type 3.3 + custom **Welti** pinna; Harman Beta 2024 (SoundGuys, CanJam NYC 2024):
    5128-diffuse-field baseline +6.4 dB shelf <105 Hz, −1.4 dB shelf >2.5 kHz; GRAS KB5000
    product page: anthropometric pinna for 45BB/45BC/43AG on a 711-style (IEC 60318-4) ear.
    Authors' Head-Fi figures: 5128 family target scatter ±1 SD ≈ 2.5 dB vs KB5000_7 ≈ 1 dB —
    why they reject a general 5128 target ("not possible at this time"). KB5000_7 curve is
    published **as data** (paper attachment; user has `~/Downloads/KB5000_7_any_frequency.txt`,
    ARTA export, 13 Hz–19.7 kHz) — recipe points at a user-supplied CSV, curve never in repo.
  - [ ] A1a. New recipe `targets/harman_kb5000_7.json`: fixture `gras_kb5000`, family per the
        published curve's shape, `baseline_csv` to the user's exported CSV, `params` all 0.0
        (curve already includes everything), `source` = AES 394 + Head-Fi write-up provenance.
  - [ ] A1b. Rig-guard semantics: decide + document which fixture keys accept this recipe.
        KB5000 mounts on 45CA-10 *and* 43AG (same IEC 60318-4-family ear). If the guard must
        tolerate (capture `gras_43ag`/`gras_43ag_clone`, recipe `gras_kb5000`), that is a
        deliberate guard-table change with its own discriminating test — not a silent widening.
  - [ ] A1c. `targets/README.md`: per-fixture status table for all five fixtures
        (`gras_43ag`, `gras_43ag_clone`, `gras_45ca`, `gras_kb5000`, `bk5128`): what published
        target exists there, on which pinna/coupler, source, licence caution; state plainly
        **no published Harman equivalent exists for the 43AG flat-plate family as such** and
        how a 43AG user should proceed; tighten the "on specific pinnas" sentence to match the
        publisher page wording; keep the recipes-not-curves rule; note the 5128 generalisation
        rejection so future additions carry scatter data, not just a curve.
  - [ ] A1d. Import helper `scripts/import_kb5000_7.py` (ARTA header/summary rows →
        two-column CSV next to the recipe; user runs it on their file; ruff-clean; the script
        embeds no curve data).
  - [ ] A1e. CLI smoke in a temp dir using the user's KB5000_7 export through
        `--target-preset targets/harman_kb5000_7.json` (guard pass + guard refusal cases).
- [x] A2. Done: `a_frame_wider_than_sixteen_bits_is_refused_not_clamped` discriminates — clamp
      sabotage failed it at src/riff.rs:748, width-guard deletion at :763; md5 `f808bbc0…` restore.
- [x] A3. Done: counts measured from the tree (342 lib / 41 bin / 32 cli = 415) and corrected in
      README.md:10 + ROADMAP:27 (SETUP had none). Two new tests + 1 more mid-session → recounted.
- [x] A4. Done: COMPREHENSIVE_AUDIT header no longer claims "All Resolved & Verified"; points at
      the BUGFIX.md COMPLETION AUDIT as the evidence location.
- [x] A5. Done: final gate green — fmt 0, clippy 0 warnings, doc(-D links) 0,
      340+41+32 = 413 executed / 0 failed / 2 ignored; ruff clean (self-tests rerun OK); shellcheck 0.
- [x] A6. Done: all five smokes + calibration refuse/force/unverified cases passed in
      /tmp/hypreq-smoke (shared-preamp −2.30 dB evidence, 44.1k input, global-fit, ensemble+preset).
- [x] A7. Done: COMPLETION AUDIT appended to BUGFIX.md (per-item ran/sab/measured evidence,
      gate numbers, and two honest sabotage-restore incident notes). Two gaps found while auditing
      and closed with new tests: 2.1 had NO test (`measure_only_still_verifies_and_still_writes_nothing`),
      1.4 rate scaling unpinated (`realtime_filter_scales_with_the_sample_rate_it_is_written_for`);
      both sabotage-proved.

## Phase C - Olive & Clark target expansion (2026-11-19, user ask)

- [x] C1. Published-data scope verified against the publisher: the Head-Fi/ASR post (web.archive
      snapshot 2026-04-30; direct fetch 403 even via the local proxy) attaches exactly one data
      file, `KB5000_7 any frequency.txt` - the user's copy sits at the repo root and is now
      gitignored (`/*_any_frequency.txt`). The single-headphone, `_7` and `_20` variants for both
      fixtures are figures only; no third-party data publication exists (searched) and digitizing
      the figures was rejected (copyrighted curves, unexamined terms, scatter lost). Documented in
      targets/README.md; no recipes invented for figure-only targets.
- [x] C2. Beta-5128 baseline gap: filled locally with the AutoEq `Diffuse field 5128.csv` stand-in
      (696 pts, 20 Hz-19.96 kHz, MIT repo; provenance + "stand-in, not the pilot's curve" recorded
      in targets/README.md). Installed as targets/bk5128_df.csv - gitignored like every CSV.
- [x] C3. `--regularize auto` IR set generated: 3 models x {KB5000_7, Beta-5128} = 6 runs,
      `--force` on every cross-fixture pairing with the refusal printed (the 43AG-family captures
      vs KB5000/5128 targets are README route 3: accepted mismatch, documented, not native).
      Knees 4.29 / 4.27 / 3.00 dB - per model, from the capture's own deviation, identical across
      the two targets BY DESIGN of the flag. 18 WAVs in backup/irs/*_regauto_*, 6 figures in
      docs/plots/*_regauto_20261119.png. Audit spot-check: KB5000_7 vs Beta-5128 targets differ
      by -5.1 dB at 8.5 kHz per model.
- [x] C4. Generating the set exposed a report bug: the audit CSV, the figure's target trace and
      its panel-1 legend described the shipped `harman_2018` pairing during preset runs while the
      filter fitted the preset (the fit itself was always right). Fixed by one shared
      `run_target_curve` + `PlotInputs.target_override`/`target_label_override`; full story,
      measurements and sabotage results in BUGFIX.md ("A preset run reported another target").
      Two new discriminating tests; suite 415 -> 417 executed, gate green (fmt/clippy/doc/tests).
- [x] C6. KB5000_7 is the default target of all three shipped models (2026-11-19, user ask
      "make it default then ... for all 3"). Route 1 of targets/README.md is now executed rather
      than merely offered: the models declare `fixture: "gras_kb5000"` (name marker
      `[GRAS KB5000]`; the 43AG/clone bodies stay in `rig_info` prose) and record
      `"target_preset": "targets/harman_kb5000_7.json"`, so a plain run fits the published curve
      and the guard pairs it `Same` with no `--force`. New DB field `target_preset` (path resolved
      against the database file; the recipe is read at run time, so a fresh clone still parses
      without the gitignored CSV and stops with a message naming the import script and the
      opt-out). New `--no-target-preset` flag (clap-conflicts with `--target-preset`); a
      `--target-csv` of the user's own also suppresses the record. Resolution: CLI recipe >
      record, with csv/opt-out yielding the record. The fixture guard resolves the effective
      recipe per model: a recorded recipe that disagrees with its own model's fixture is refused
      (rc 2) and `--force` prints the override. `fixture_conflict_ok` on all three models now
      records exactly the opted-out built-in-curve pairing; `jbl770nc`'s KB006x clone-pinna
      caveat is on file in the DB and targets/README.md (declaration = nearest published
      acoustic object, not identity). Suite 417 -> 421 executed / 0 failed / 2 ignored: one
      model-parse test, one consent-order unit test, four CLI tests (audit follows the record,
      explicit recipe outranks it, csv/opt-out suppress it, wrong-fixture record refused), and a
      real-database check that skips visibly when the user CSV is absent. All four resolution
      and guard paths failed their tests under sabotage and passed on restore (md5-checked);
      27 existing real-model invocations were pinned with `--no-target-preset` so their old
      assertions keep their old meaning. fmt/clippy(-D warnings)/doc -D warnings clean.
      Updated 2026-11-19 after upstream 7ad66f9: `backup/irs/` and both `docs/plots` families
      were regenerated as distinct final/regularized canonical assets, and the README preamp
      values visible in those plot titles were re-pinned. With the authors' published
      `KB5000_7 any frequency.txt` converted to the untracked `targets/kb5000_7.csv`, all nine
      `backup/irs/*.wav` are reproduced byte-for-byte by `-H <model> --target-preset
      targets/harman_kb5000_7.json --regularize auto`; the plain default and regularized runs
      reproduce the docs plot titles. Remaining gap: the CSV is user-supplied, so a fresh clone
      cannot regenerate the KB5000_7 assets without it; the audit removes that CSV again.
- [ ] C5. Loop closure pending: `--verify-with` needs a real capture taken after convolution on
      the target's own fixture - none of the three repo models is a KB5000 or 5128 capture, so
      all six IRs remain cross-fixture estimates until the user measures on one.
