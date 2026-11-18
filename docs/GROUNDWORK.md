# Groundwork: ensembles, rig-specific targets, and rig translations

This document records the research behind the opt-in modules (`robust_fit`, `global_fit`,
`psycho`, `target`, `driver`, `synth_alt`) and, importantly, separates **published facts** from
**this project's implementation choices**. That separation is required by AGENTS.md §4: prior
art is cited for what it actually says, and our deviations are named as ours.

## 1. Repeated placements — why an ensemble at all

- **Masiero, Fels & Vorländer, DAGA 2011** ([paper](https://pub.dega-akustik.de/DAGA_2011/data/articles/000225.pdf))
  measured two circumaural headphones on a dummy head and on **15 subjects, four comfortable and
  four extreme fittings each**. Below ~4 kHz the HpTF varies mainly in level (leakage); above it,
  cavity and pinna resonances vary strongly. Inverting a single HpTF can create peaks up to
  **10 dB**; averaging several measured HpTFs still leaves high-frequency irregularities.
  Because listeners are more sensitive to peaks than to equivalent valleys, their robust filter
  inverts **the average plus two standard deviations of the HpTF amplitude**, leaving valleys.
- **Wierstorf et al., AES 130th (2011)** ([paper](https://hagenw.github.io/pubs/wierstorf_et_al-2011-a_free_database_of_head-related_impulse_response_measurements.pdf))
  repeatedly removed and replaced AKG K601, AKG K271 MK II and Sennheiser HD25-1 headphones on a
  KEMAR, averaged the repeats into one complex transfer function per model, and used
  frequency-dependent regularization that equalized only parts that did not vary too much.

**What is ours:** `EnsembleMode::UpperVariance{alpha}` computes `median + alpha × 1.4826 × MAD`
in **dB** on **1/6-octave smoothed** curves. The Masiero upper limit is in the **amplitude**
domain and uses `mean + 2σ`. Same intent (push correction toward peaks being avoided), different
domain, estimator and smoothing. The `alpha ≈ 2.0` value is a starting point, not a published
constant for our formulation. The bootstrap CI uses whole curves (not independent bins) to keep
spectral correlation — that is also our design.

### Workflow

PHOnA (Princeton 3D3A, [dataset page](https://3d3a.princeton.edu/princeton-headphone-open-archive-phona),
[1.3 GB SOFA archive](https://www.princeton.edu/3D3A/Downloads/PHOnA_0.3.zip)) contains HpTFs
from multiple labs, headphones, subjects and repositionings. AUR has no `python-sofar` package,
so this repo keeps a local venv:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install sofar
.venv/bin/python scripts/sofa_to_ensemble.py --self-test
.venv/bin/python scripts/sofa_to_ensemble.py PATH/TO/*.sofa --out-dir ensemble/
hypreq -H model --ensemble-csv ensemble/*.csv --ensemble-alpha 2.0
```

The converter normalizes every curve at 1 kHz before writing it, so absolute rig level cannot
be mistaken for placement variation. It supports both `Data.IR` and `Data.Real/Data.Imag` SOFA
conventions and `--ear left|right|all`.

## 2. Targets belong to rigs

- The Harman around-/on-ear target was defined on a **GRAS 45CA with RA0045 ("711") ear
  simulator and custom pinna**; Olive & Clark (AES 159th, 2025, eLibrary `23068`, paper 394)
  show it is often misused on other fixtures because acoustic transfer impedances differ.
- The 711 pinna gives **2–3 dB more high-frequency emphasis** than human ears, and its
  standing-wave resonance near 8 kHz makes >10 kHz unreliable ([Qfiaudio](https://www.qfiaudio.com/learn/why-rigs-differ)).
  Rig differences can exceed product differences, so a target is rig-specific.
- **Harman Beta 2024 for the B&K 5128**, reported from Olive's CanJam NYC 2024 pilot by
  [SoundGuys](https://www.soundguys.com/harman-research-targets-the-bruel-kjaer-5128-112247/),
  is the 5128 diffuse-field response with a **+6.4 dB shelf below 105 Hz** and **−1.4 dB shelf
  above 2.5 kHz**. In that small blind test (13 listeners) the plain 5128 DF response scored
  21.167 versus 64.179 for Beta 2024 and 70.795 for Harman IE 2019.
- A 5128 in-ear study ([AES eLibrary 22928](https://www.aes.org/publications/elibrary-page/?id=22928))
  found preferred tunings spanning **+2…+9 dB bass** and **−5…+2 dB treble** relative to the
  5128 DF response, depending on listener segment.

**What is ours:** `targets/harman_beta_2024_5128.json` stores only the published constants and a
path to a **user-supplied** 5128 DF CSV. We do not embed the DF curve or any Harman table — the
same copyright hygiene that keeps ISO 226 Table 1 external. The realization of the shelves as
`TargetParams`' 105 Hz and 2.5 kHz basis functions is **our** implementation; the published
target's exact filter shape is not reproduced.

```sh
# put your own 5128 DF curve at targets/bk5128_df.csv first
hypreq -H model --target-preset targets/harman_beta_2024_5128.json
# or supply any baseline directly
hypreq -H model --target-csv my_5128_target.csv --target-bass 6.4 --target-mid -1.4
```

`--target-bass`, `--target-mid`, `--target-tilt` override the recipe's constants. `--target-csv`
and `--target-preset` are mutually exclusive; a preset carries its own baseline path.

## 3. Empirical rig translations

No public source surfaced with a ready-made cross-fixture transfer impedance *and* an
uncertainty estimate, so `RigCalibration` is data you derive from same-model cross-rig
measurements (available from headphones.com's database or headphonedatabase.com):

```sh
python3 scripts/derive_rig_calibration.py \
  --source-fixture gras_43ag --target-fixture bk5128 \
  --pair model_a_43ag.csv model_a_5128.csv \
  --pair model_b_43ag.csv model_b_5128.csv \
  --out targets/rig_43ag_to_5128.json

hypreq -H model --rig-calibration targets/rig_43ag_to_5128.json --global-fit
```

Each pair is 1 kHz-normalised before subtraction, so rig SPL calibration cannot masquerade as a
transfer difference. `delta_mean_db` (target minus source) is subtracted from the correction;
`delta_sigma_db` (across-model spread, floored at 0.30 dB) becomes the `sigma_rig` term in
`global_fit`'s weights `w = 1 / (sigma_r² + sigma_rig²)`. With no calibration, a fixed
`0.5 dB` allowance is used instead of pretending the uncertainty is zero.

**What is ours:** the derivation script, the sign convention, the flooring, and the fixed
fallback. None of these are published constants.

## 4. ISO 226 note

`src/target.rs` implements ISO 226:2023 Formula (1) as published: exponent `(α_r − α_f)`, base
`(p₀/p_a)² = 4×10⁻¹⁰`, `T_r = 2.4 dB`, `α_r = 0.30`. The 2003 edition instead uses `α_r = 0.25`
and a different α_f table; the two must never be mixed. Table 1 itself remains external JSON for
the same copyright reason as the target curves above.
