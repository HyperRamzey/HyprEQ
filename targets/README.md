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
interchangeable. With the baseline in place:

```sh
hypreq -H your_model --target-preset targets/harman_beta_2024_5128.json
```

The report prints the preset's name, declared rig, and source under the model heading. The
published shelf frequencies coincide with this crate's `TargetParams` bass and presence basis;
the basis *shape* is ours, not a reproduction of the published curve.

## Other rig targets

The 2025 Olive & Clark paper defines targets for the B&K 5128 and GRAS 45CA-10 on specific pinnas
(paper 394, eLibrary `23068`, AES 159th). Those tables are not reproduced here for the same reason
the ISO 226 table is not: the formula/constants live in code, the data lives in a file you supply.
To use one, write a recipe with its own `name`, `fixture`, `baseline_csv`, and `params`, and point
`--target-preset` at it. Keep `bass`/`mid` at `0.0` if the baseline CSV already includes the
published shelf structure.

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
