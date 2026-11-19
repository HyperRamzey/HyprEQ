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

## Phase B — after Phase A: continue the Derp project from its latest pi session

- Session (latest by mtime, 2026-11-18 10:11):
  `~/.pi/agent/sessions/--mnt-aosp-Projects-Derp--/2026-11-17T07-58-46-815Z_01a1e883-fe1d-7701-b783-c71e81ccc3e9.jsonl`
  (project: /mnt/aosp/Projects/Derp; next-newest file same dir: `...01a1e9ff...jsonl` 2026-11-17 19:52)
- State as of transcript end:
  - Task given: fix replies to issues in the **a30s device repo** in system, plus new issues,
    plus PRs if any; make the build script build **3 ROMs in 3 folders with per-variant
    ccache** (without building).
  - User instruction inside that session: "once you fix and verify everything, switch to
    **mikrotik job** (mine pi session list to find it)" — the mikrotik job's own session is
    still to be located in the same session tree.
  - Last assistant action: recovery + full restoration of the phone after a thermal test that
    dropped UI to 0.1 fps (load 7.27→1.18, stock config md5, Enforcing). Treat that as a
    standing lesson: no unbounded load loops on production devices.
- First steps when Phase B starts: read the session tail for the unverified remainder
  (which issues were actually fixed and verified vs still open), re-verify against
  /mnt/aosp/Projects/Derp working tree, then continue; locate the mikrotik session the same way.
- Global AGENTS.md timeout rules from that session are already in place
  (~/.pi/agent/AGENTS.md).

### B1 — build_variants.sh --check: diagnosed, fixed, re-verifying (2026-11-19)

The pi session's `--check` looked hung at 240 s; it was never hung, it was ~30 min/variant
and had a silent-failure mode. Measured on the tree, not inferred:

- Cold `lunch` bootstrap for a fresh `out_release`: **625 s**; each `get_build_var`
  afterwards: **134–195 s** — every one a fresh `soong_ui` re-evaluating the product
  config. Old cost per variant = envsetup + lunch + 6 queries ≈ 25–35 min.
- `get_build_var` also returned **empty**: crDroid's patched `build/make/envsetup.sh:1273`
  runs `setup_ccache()` at source time; with no ccache exports it sets
  `CCACHE_DIR="$HOME/.ccache"` — a **symlink** → Soong sandbox bind-mount fails
  (`remountPt(): mount('/home/sanjxz/.ccache', MS_REMOUNT|MS_NOATIME|MS_BIND): Invalid
  argument` → dumpvars exit 255, stderr swallowed by `2>/dev/null`). A/B/C test:
  symlink rc=2 crash / real root dir rc=0 / variant dir rc=0, same kernel
  (linux-cachyos 7.3.4-1 since 2026-11-14, so NOT a kernel regression — yesterday's
  builds worked on it because the derp scripts export the real array path).
- Fixes applied (files were git-**untracked**; nothing reverted, additive edits only):
  - `hm_mod/build_variants.sh config_of`: ONE batched `soong_ui --dumpvars-mode` per
    variant fed via TARGET_* env — no envsetup/lunch sourced at all (avoids the
    setup_ccache trap by construction). ~2–3 min/variant warm.
  - rc-capture bug: `if ! out=$(config_of); then rc=$?` always reported 0 (negation
    overwrites `$?` — measured `inverted-if rc=0 / guarded rc=5`); now `|| rc=$?`.
  - `lib/derp_variants.sh derp_variant_ccache_init`: variant ccache dirs silently ran
    on ccache defaults (measured `ccache -p`: "(default) max_size = 5.0 GiB,
    direct_mode = true" in the variant dir vs 40G/compressed/off at the root);
    root conf is now synced into each variant cache on check/build.
  - `lib/derp_env.sh`: stale Soong citation fixed (wrap is config.mk:556 → ccache.mk →
    CC_WRAPPER → global.go:568, not "global.go:464"), plus the ordering-trap comment.
- Verification DONE: full `--check` on all 3 variants → **OK** (log /tmp/variants_check2.log;
  release/testing/gapps all show their want_* values; ccache conf synced per variant).
  Sabotage proof: release `want_secure` flipped 1→0 → `FAIL: release: ro.adb.secure=1,
  want 0` + FAILED verdict (log /tmp/variants_sab.log); library restored byte-identical
  (md5 7a36015c…). derp_variant_ccache_init unit tests 3/3 (seed / re-sync / no-conf-no-impose).
- Struck oil on the way: `strata_build.sh --detach` DROPPED the module name in its re-exec
  (`module <name>` died rc=2 before parking, printing a supervisor pid for a build that
  never ran — measured). Fixed: tgt_flag carries `module "$MODULE"`, and the guard now
  refuses an option eaten as the name. strata_build.sh is git-TRACKED; change is
  uncommitted working-tree (bash -n + shellcheck clean, negative test rc=2).
- B2 (task 11) mid-flight: `strata_build.sh module recoveryimage` (gapps tree, engine
  handshake, 14 min, rc=0) → fresh out_gapps/target/product/a30s/recovery.img 04:17;
  gate_recovery_usb PASS incl. built-image section (fix rc inside ramdisk under the
  imported init.recovery.exynos7904.rc name, dead usb.rc copy gone) and --selftest
  (gate caught 15 injected v0.8.0-shaped defects). Device root verified (ksu, Enforcing).
  verify_recovery_usb.sh: backup → flash → recovery-boot probe → last_kmsg capture.
- FLASHED + FLASH_VERIFIED (readback md5 21b2f982 = image md5; partition p16).
  flash_recovery_only.sh old basename guard `*/recovery` could never pass on eMMC
  where by-name resolves to mmcblk0p16 — it blocked the first real flash; replaced by
  /dev/block + -b + caller-size cross-check. Old partition backed up on host:
  hm_mod/recovery_backups/recovery_20261119_042052.img (md5 f7d286a9…, --restore puts it back).
- PROBE FINDING (measured: host dmesg + ramdisk-cpio forensics): an unattended plain
  `reboot recovery` enumerates NOTHING by AOSP design — the recipe builds the gadget
  skeleton at `on fs`, but the UDC bind waits for `sys.usb.config=adb`, which only the
  recovery-UI "Enable ADB" tap (recovery.cpp:719, the pi-session fix) or the SIDELOAD
  boot reason (recovery.cpp:158) sets. Zero host connect events in the probe window =
  unbound = designed behavior, NOT a fix regression. Ramdisk proves the chain: init at
  /system/etc/init/hw/init.rc imports /init.recovery.exynos7904.rc at line 1;
  androidboot.hardware=exynos7904 on the recovery cmdline; skeleton + os_desc in place.
- PHONE STATE (04:31+): in recovery UI, charging on USB; AOSP recovery suppresses its
  key timeout while USB is plugged (ui.cpp:610), no adbd/console/network/auto-reboot →
  needs ONE human action: hold VolDown+Power ~10 s, or tap "Reboot system now".
  /data, boot, system, vendor untouched; zero USB events since reboot = stable, no panic loop.
  When the phone is back: `verify_recovery_usb.sh --probe-only --no-backup` now boots
  `adb reboot sideload` (auto sys.usb.config=adb = the issue-#12 scenario itself, headless),
  probes enumeration, then `--capture-only` collects /cache/recovery/last_kmsg.
- Tooling changed (all uncommitted, bash -n + shellcheck clean): flash_recovery_only.sh
  guard rewrite + size arg; verify_recovery_usb.sh sideload default + --menu-boot
  escape hatch + size handoff; strata_build.sh module-name re-exec fix + option-eaten
  guard. flasher guard primitives proven on-device (case/-b/size all pass vs mmcblk0p16).
- ccache plumbing verified in-tree: USE_CCACHE+CCACHE_EXEC → CC_WRAPPER (Soong) and
  BoardConfigKernel.mk:110-113/138 → KERNEL_CC_WRAPPER (kernel), sandbox bind-mounts
  `ccache -k cache_dir` (sandbox_linux.go:341). Chain intact for all 3 variants.

### B queue after B1 (from the pi-session todo snapshot)

- [ ] B2. Task 11: build `recoveryimage`, gate_recovery_usb, backup, flash recovery-only
      (`flash_recovery_only.sh`), prove #12 enumerates adb in recovery. `adb -s RF8M834JT4D`.
- [ ] B3. Task 12: build gapps variant + flash (system/vendor flash = confirm with user
      first despite session grant) + verify #11 VoLTE / #13 doze / #14.1 thermal on device.
- [~] B4. Task 6 (issue #6 round 2): **implemented + host-verified 2026-11-19** —
      full notes + draft reply in Derp `hm_mod/issues_20261116/issue6_uksm_zram_20261119.md`.
      Kernel `mm/uksm.c:565` default `uksm_run = 1 -> UKSM_RUN_STOP` (quiet governor only
      thins scan rounds; STOP parks uksmd in wait_event_freezable, `echo 1 > run` wakes it);
      `CONFIG_UKSM=y` PINNED into full/exynos7885-a30s_defconfig after finding the symbol was
      alive ONLY via stale out-tree .config (defconfig never set it, Kconfig default n —
      fresh out/ would have shipped a different kernel than every warm-tree build);
      new `exynos7885-common/sepolicy-testing/shell.te` (read-only shell rules: sysfs_zram,
      proc_swaps, proc_pressure_mem; uksm knobs deliberately NOT in it — see gate note) gated by
      `ifeq ($(TARGET_PRODUCT),lineage_a30s_testing)` in a30s/BoardConfig.mk (board-var
      gate — SYSTEM_EXT_PRIVATE_SEPOLICY_DIRS is read from BoardConfigs, not product mk).
      Proof: mm/uksm.o compiles rc=0 in warm KERNEL_OBJ (tree is CONFIG_LTO_CLANG => bitcode,
      decoded via `clang -cc1 -x ir -emit-obj`); sabotage cycle through the make path showed
      uksm_run .bss(0) / .data(1) / .bss(0) with source sha256-restored. EACCES root cause:
      SELinux labels (attrs are 0444 DEVICE_ATTR_RO) — measured from genfs_contexts +
      zram_drv.c + public/file.te; neverallow audit clean.
      GATES: `build_variants.sh --check` PASSED all 3 variants after the BoardConfig edit
      (/tmp/variants_check3.log). `sepolicy_neverallows` (testing variant, env-based:
      derp_variant_env exports + strata module build): round 1 ran and CAUGHT a real
      coredomain.te:143 violation in the first draft -> offending line removed (uksm knobs
      stay root-only; unlabeled sysfs is neverallow-protected for shell by design); round 2
      was KILLED by the harness memory-pressure reaper mid-repair before compiling
      (no orphans, engine healthy) — re-run pending explicit ask, use
      `strata_build.sh module sepolicy_neverallows --repair on`. Device checks await phone
      (recovery UI, one button press).
      **All pushed 2026-11-19 (user instruction "commit and push everything"):** kernel
      `e5bce5c0` -> Eureka-Kernel-R24U rom-crdroid16; exynos7885-common `47efb87` ->
      device_samsung_exynos7885-common crdroid-16; a30s `dcf37cf` -> device_samsung_a30s
      crdroid-16 (OTA channel split + CarrierConfig/doze overlays rode along); Derp tooling
      `9c7a9672` -> master (gates, build_variants, issue notes; evidence binaries rec.dump /
      rec_ramdisk.* / twrp_rd/ gitignored, re-acquirable via verify_recovery_usb.sh;
      kernel `KernelSU.old/` deliberately left untracked).
      task 7 (#14.6 Photos): **already fixed in tree** — verified 2026-11-19 that device
      commit 46a8eb9 (Nov 13) replaced the flattened-APK import with base+config splits +
      `derp-gphotos-install.sh` first-boot installer (gphotos/*.apk present; script header
      carries the exact prx.xml hole analysis + INSTALL_FAILED_MISSING_SPLIT). v0.8.0
      predates nothing — 46a8eb9 IS in the 20261115 build window; reporter's crash was h2.
      Remaining: on-device fresh-install proof with the gapps flash (B3). No new code.
      task 8 (#14.5 etspi): **implemented + sabotage-proved 2026-11-19** —
      `drivers/fingerprint/et7xx-spi.c` (a30s builds ET7XX per out-tree .config; ET5XX not
      set, left untouched as unbuildable-here): the 10 s FPSENSOR_DEBUG_TIMER armed
      unconditionally and `etspi_work_func_debug` pr_info'd every tick (82 lines + wake
      source in #14 dmesg). New `debug_verbose` module_param (bool, default off, 0644 at
      /sys/module/et7xx_spi/parameters/) gates `etspi_enable_debug_timer()` (probe + resume
      re-check it). Compile rc=0 in warm KERNEL_OBJ; sabotage cycle: .bss(0) / .data(1) /
      .bss(0), source sha256-restored. #14.4 WifiHAL flood + fs-verity noise: still triage.
      task 13: #7 re-measured on v0.8.0 (jank 60-63%, main-thread measure/layout — trace
      offer open; maintainer asked for root perfetto traces 11-16, awaiting reporter);
      triage #4/#8/#9. No new issues, no open PRs (checked 2026-11-19). Thermal #14.1:
      prefer gated-build/flash evidence, NEVER unbounded load loops on the phone.
- [ ] B5. Task 10: docs + self-review of all device-tree fixes.
- [ ] B6. Task 14: switch to the MikroTik VLESS/nfqws DPI job
      (`docs/projects/ROUTER-DPI.md` + session `...2026-10-28T15-20-29...01a18319...jsonl`).
- Never commit/push unasked. (State 2026-11-19: all device-tree/kernel/Derp fixes committed
  and pushed under the user's explicit instruction — hashes in B4; nothing uncommitted left
  to revert.)
