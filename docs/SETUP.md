# Setting up HyprEQ

HyprEQ ships as one Rust crate (`hypreq`, a library plus a binary) and one Python
script (`hypreq.py`). The Rust crate is the implementation; `hypreq.py` is kept only as
the **parity oracle** the harness diffs the Rust output against, and it is expected to
drift behind the Rust side as features are added.

```text
Cargo.toml            crate manifest (numerical policy lives in its comments)
src/                  the implementation
hypreq.py             parity oracle — not the thing you build
measurements.json     target curves + headphone definitions, in file order
models/               optional drop-in .json / .csv models (scanned at startup)
scripts/parity.py     Rust-vs-oracle comparison harness
docs/                 this file
```

## 1. Prerequisites

| What | Needed for | Notes |
| --- | --- | --- |
| Rust ≥ 1.87 (`rustc`, `cargo`) | everything | `rust-version` in `Cargo.toml` is 1.87, edition 2021 |
| `pkg-config`, `fontconfig`, `freetype` (C dev files) | the verification PNG | the `ttf` feature of `plotters` goes through `font-kit`, which links against the system fontconfig/freetype and resolves `"sans-serif"` **at runtime** |
| a usable `sans-serif` font | the verification PNG | any family fontconfig can resolve; DejaVu is enough |
| Python ≥ 3.10, `numpy`, `soundfile` | `scripts/parity.py` | `soundfile` wrapslibsndfile for payload decoding |
| Python `matplotlib` | the oracle's PNG only | the Rust figure does not use it |

The font dependencies are not optional if you want the figure: with the `ttf` feature
enabled a machine with no fontconfig configured fails to *link*, and a machine with
fontconfig but no resolvable `sans-serif` panics inside `plotters` while drawing text.
`--no-plot` avoids both paths entirely.

On Arch:

```bash
sudo pacman -S --needed rust fontconfig freetype2 ttf-dejavu python-numpy python-soundfile python-matplotlib
```

## 2. Build

```bash
cargo build --release          # ./target/release/hypreq
cargo build                    # dev profile, faster compile, same numerics
```

The release profile sets `lto = "fat"`, `codegen-units = 1`, `opt-level = 3` and
**nothing else**. Read the comment in `Cargo.toml` before touching it: the DSP must stay
IEEE-754 exact, so no `-C target-cpu=native` with FMA, no `-C llvm-args=--fp-contract=fast`
and no fast-math. A fused multiply-add rounds once where the reference rounds twice, and
the true-peak interpolator, the limiter's gain scheduler and the Hilbert-cepstral
minimum-phase fold would drift off the oracle.

## 3. Run

```bash
# everything in the database: f64 convolver, two f32 real-time convolvers, PNG
./target/release/hypreq --headphone all --export-csv

# one model, lighter filter, different rate
./target/release/hypreq -H dt770_80 --notch-mode safe
./target/release/hypreq -H ft1pro --taps 16384 --fs 44100

# what is loaded, in database order
./target/release/hypreq --list-models

# measure / EQ / true-peak-limit a WAV
./target/release/hypreq -H jbl770nc --input song.wav --out-bits 24
./target/release/hypreq --measure-wav song.wav          # limiter + meters, no EQ, no write
```

Artefacts are written to the **current directory**, named from the model's
`file_prefix` (so `dt770_80` writes `dt770_80ohm_*`):

| File | Contents |
| --- | --- |
| `<prefix>_f64_<rate>.wav` | 64-bit float mono convolver for `mpv`/FFmpeg (`afir=precision=double`) |
| `<prefix>_mastering_mono_<rate>.wav` | 32-bit float, 16,384 taps, PipeWire/DAW |
| `<prefix>_mastering_stereo_<rate>.wav` | the same IR on both channels |
| `<prefix>_final_verification.png` | 1950 × 1725, 4-panel verification figure |
| `<prefix>_audit_data.csv` | frequency, raw, target, raw error, applied EQ (`--export-csv`) |
| `ir.wav` | copy of the single selected model's f64 convolver |
| `<input>_<model>_tp_<ratio>.wav` | audio-mode output |

`<rate>` is `48k`, `96k`, `44.1k`, `88.2k` and so on.

`--data-file` and `--models-dir` point at a different database/drop-in directory;
without them the defaults (`measurements.json`, `models/`) are resolved relative to the
current directory.

## 4. Verification gates

Every one of these has to be clean before a change is pushed:

```bash
cargo fmt --check
cargo build
cargo build --release
cargo test                      # unit + integration, default parallelism
cargo clippy --all-targets      # zero warnings, including tests
python3 scripts/parity.py       # all scenarios

# documentation links must resolve — this is what catches a doc that still names a
# function the code no longer has
env -u RUSTC_WORKSPACE_WRAPPER \
  RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc --no-deps
```

`cargo test` runs a few deliberately expensive tests (full-length convolutions and
figure renders); two are `#[ignore]`d because they take minutes and are covered by the
parity harness instead:

```bash
cargo test -- --include-ignored
```

### The parity harness

```bash
python3 scripts/parity.py                 # everything
python3 scripts/parity.py --only audio    # scenarios whose name contains "audio"
python3 scripts/parity.py --keep          # leave the scratch tree behind
python3 scripts/parity.py --rust ./target/release/hypreq --python python3 --ref hypreq.py
```

Each of the 22 scenarios runs `hypreq.py` and the Rust binary in **separate scratch
directories** seeded with the same `measurements.json` and the same input WAV, then
compares:

* **stdout** byte-for-byte after normalising the absolute paths both tools print;
* **WAV containers** exactly (chunk order, format tag, channels, rate, bits, byte size);
* **WAV payloads** — dithered PCM must be byte-identical, float payloads within a
  per-bit-depth tolerance (a 32-bit payload is judged in f32 ULP; comparing it against an
  f64-scale tolerance is a harness bug, not a divergence);
* **audit CSVs** byte-for-byte;
* **usage errors**: message wording and exit code `2`;
* the **PNG**: the Rust side must write a decodable `1950 × 1725` image **with plotted ink
  inside each of the four panels** — counted strictly inside each panel, inset past the
  frame and the tick marks, so an image whose panels hold nothing but their own frames
  fails. matplotlib and `plotters` are different renderers, so pixels are never compared;
  exact geometry is covered by unit tests that decode the PNG the same way.

The harness exits non-zero if any comparison fails, so it is usable as a CI gate.

## 5. Numerical rules

* All DSP state is `f64`. Only a container may drop to `f32`, and only at write time.
* `+`, `-`, `*` must not be contracted into FMA. Do not enable fast-math anywhere.
* Where the oracle uses `round()` / `np.round()`, Rust uses `.round_ties_even()`:
  Python rounds half to even, and `44.1 kHz` exposes it (`5 ms * 44.1 = 220.5` → `220`).
* `src/fft.rs` wraps `rustfft` to NumPy conventions (`ifft` scales by `1/n`, `rfft`
  returns `n/2+1` bins, `rfftfreq` keeps a positive Nyquist bin). rustfft and NumPy's
  pocketfft factor the same transform differently, so the convolver matches the oracle to
  a couple of ULP rather than bit-for-bit — that residual is transform structure, and it
  is what the f32 real-time convolvers' 1-ULP parity tolerance absorbs.
* PCM output uses a fixed TPDF dither stream whose draw order is part of the byte
  pattern: `(frame, channel)` row-major, one pair of uniform draws per sample. Changing
  the layout changes the bytes.

## 6. Adding headphone measurements

Three ways, none of which touch the DSP:

1. **Drop a CSV in `models/`** — 2 columns (`freq_hz,raw_dBr`) or 4
   (`freq_hz,left_dBr,right_dBr,avg_dBr`); the filename stem becomes the `-H` key. The
   raw curve is normalised at 1 kHz, differenced against the default target, given the
   sub-bass leak discount and tapered to `0.00 dB` at the model's `hf_bypass_hz`.
2. **Drop `models/<key>.json`** — same schema as a database entry; supply `raw_pts` (or
   `raw_csv`/`ctrl_csv`) and either explicit `ctrl_pts` or let them be derived.
3. **Add an entry under `"models"` in `measurements.json`** — order in the file is order
   on screen: `--list-models`, `--headphone all` and the default model all follow it.

Useful keys: `name`, `file_prefix`, `rig_info`, `target`, `harm_offset`, `gauss_apodize`,
`hf_bypass_hz`, `notch_range_hz`, `safe_cap_db`, `subbass_leak_discount_db`,
`autoeq_geq_str`, `plot_meta`, `plot_annotations`.

`--upgrade-json` writes the built-in `plot_meta` and callout annotations back into the
database in place, which is the easiest way to see what a model can carry.

## 7. Troubleshooting

| Symptom | Cause |
| --- | --- |
| build fails linking `fontconfig`/`freetype` | missing C dev packages for the `ttf` feature — install them or build with `--no-plot` runs |
| panic while drawing text | fontconfig cannot resolve `sans-serif`; install a font family |
| `Error: ... measurements.json` | the database is resolved against the current directory — run from the repo root or pass `--data-file` |
| `error: ... exit code 2` | usage error, same shape argparse used; `--help` lists the accepted values |
| parity scenario says "neither side wrote it" | the scenario expects the wrong filename — output files are named from `file_prefix`, not the model key |
