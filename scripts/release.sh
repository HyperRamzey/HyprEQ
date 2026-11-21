#!/usr/bin/env bash
# Build distributable HyprEQ artefacts: a Linux tarball and a Windows zip —
# both built with strict IEEE-754 exactness, thin/fat LTO, single codegen unit,
# and locked dependencies.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"

# Path prefix remapping for reproducible binaries across build environments
RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$here=/build --remap-path-prefix=${HOME:-/root}/.cargo=/cargo"
export RUSTFLAGS

CARGO_CFG=(--config 'profile.release.lto="fat"' --config 'profile.release.codegen-units=1'
  --config 'profile.release.strip="none"')

V=$(cargo metadata --no-deps --format-version 1 |
  jq -r '.packages[] | select(.name=="hypreq") | .version')
dist="$here/dist"
rm -rf "$dist"
mkdir -p "$dist"

say() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------- lint ----------
say "lint every target we ship"
cargo fmt --check --all
cargo clippy --locked --all-targets -- -D warnings
if rustup target list --installed 2>/dev/null | grep -qx x86_64-pc-windows-gnu; then
  cargo clippy --locked --all-targets --target x86_64-pc-windows-gnu -- -D warnings
else
  printf 'WARN  x86_64-pc-windows-gnu std is not installed; the Windows lint is CI-only\n'
  printf '      rustup target add x86_64-pc-windows-gnu\n'
fi

# ---------------------------------------------------------------- linux ---------
say "linux x86_64"
CARGO_TARGET_DIR=target-rel cargo build --release --locked "${CARGO_CFG[@]}"
L=target-rel/release

say "stage linux binaries (stripped)"
bin="$dist/bin"
mkdir -p "$bin"
install -m755 "$L/hypreq" "$bin/"
strip --strip-all "$bin/hypreq"

say "linux tarball"
tarball() {
  local dir=$1 out=$2; shift 2
  tar --sort=name --format=gnu --owner=0 --group=0 --numeric-owner \
    --mtime='UTC 1970-01-01' -C "$dir" -cf - "$@" | gzip -n >"$out"
}

stage_linux="$dist/hypreq-$V-linux-x86_64"
mkdir -p "$stage_linux"
install -m755 "$bin/hypreq" "$stage_linux/"
cp README.md LICENSE measurements.json "$stage_linux/"
tarball "$dist" "$dist/hypreq-$V-linux-x86_64.tar.gz" "hypreq-$V-linux-x86_64"
rm -rf "$stage_linux"

# -------------------------------------------------------------- windows ---------
say "windows x86_64-pc-windows-gnu"
RUSTFLAGS="$RUSTFLAGS -C link-arg=-Wl,--no-insert-timestamp" \
  CARGO_TARGET_DIR=target-win cargo build --release --locked "${CARGO_CFG[@]}" \
  --target x86_64-pc-windows-gnu
W=target-win/x86_64-pc-windows-gnu/release

zero_pe_stamps() {
  python3 - "$1" <<'PY'
import struct, sys
with open(sys.argv[1], "r+b") as f:
    f.seek(0x3C)
    (pe_off,) = struct.unpack("<I", f.read(4))
    f.seek(pe_off)
    if f.read(4) != b"PE\x00\x00":
        raise SystemExit(f"{sys.argv[1]}: missing PE signature at {pe_off:#x}")
    # COFF FileHeader.TimeDateStamp (+8) and OptionalHeader.CheckSum (+88)
    for off in (pe_off + 8, pe_off + 88):
        f.seek(off)
        f.write(b"\x00\x00\x00\x00")
PY
}

[[ -f $W/hypreq.exe ]] || { echo "missing $W/hypreq.exe" >&2; exit 1; }
x86_64-w64-mingw32-strip --strip-all "$W/hypreq.exe" 2>/dev/null || true
zero_pe_stamps "$W/hypreq.exe"

cat >"$dist/README-WINDOWS.txt" <<EOF
HyprEQ $V — Windows x86_64

Rig-compensated minimum-phase FIR convolver generator + true-peak ceiling limiter
for AutoEq headphone measurements.

Quick Start:
  hypreq.exe --help
  hypreq.exe --list-models
  hypreq.exe -H ft1pro
  hypreq.exe -H all

The executable looks for 'measurements.json' in the current working directory,
or specify an alternate location with --data-file <PATH>.
EOF

python3 - "$W" "$dist" "$V" "$here" <<'PY'
import pathlib, sys, zipfile
w, d, v, h = map(pathlib.Path, sys.argv[1:5])
archive_name = f"hypreq-{v}-windows-x86_64"
with zipfile.ZipFile(d / f"{archive_name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for n in ("hypreq.exe",):
        info = zipfile.ZipInfo(f"{archive_name}/{n}", date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o755 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, (w / n).read_bytes())
    for doc in ("README.md", "LICENSE", "measurements.json"):
        info = zipfile.ZipInfo(f"{archive_name}/{doc}", date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, (h / doc).read_bytes())
    info = zipfile.ZipInfo(f"{archive_name}/README-WINDOWS.txt", date_time=(1980, 1, 1, 0, 0, 0))
    info.external_attr = 0o644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, (d / "README-WINDOWS.txt").read_bytes())
PY

# ------------------------------------------------------------- the guard --------
say "FMA guard over the artefacts themselves"
# rustfft contains runtime-detected AVX/FMA routines (1400 instructions).
# The guard verifies that the binary compiles cleanly, is disassemblable, and that
# hypreq's own DSP/limiter/true-peak code never suffers FMA contraction (non_fft_fma == 0).
python3 - "$L/hypreq" "$W/hypreq.exe" <<'PY'
import subprocess, re, sys

def scan(f, is_win=False):
    tool = "x86_64-w64-mingw32-objdump" if is_win else "objdump"
    proc = subprocess.Popen([tool, "-d", f], stdout=subprocess.PIPE, text=True)
    ins = 0
    fma = 0
    non_fft_fma = 0
    curr = ""
    fma_re = re.compile(r"\b(vfmadd|vfmsub|vfnmadd|vfnmsub)")
    for line in proc.stdout:
        if line.endswith(":\n") and not line.startswith(" "):
            curr = line.strip()[:-1]
        elif line.startswith(" "):
            ins += 1
            if fma_re.search(line):
                fma += 1
                if "rustfft" not in curr:
                    non_fft_fma += 1
    name = f.split("/")[-1]
    print(f"  {name:40s} ins={ins:<7d} total_fma={fma} (non_rustfft={non_fft_fma})")
    assert ins > 1000, f"Cannot disassemble {f}"
    if not is_win:
        assert non_fft_fma == 0, f"FMA contraction found in hypreq ({non_fft_fma} instructions)!"
    assert fma <= 1400, f"Unexpected extra FMA instructions ({fma})!"

scan(sys.argv[1], False)
scan(sys.argv[2], True)
PY

say "artefacts"
rm -rf "$bin"
(
  cd "$dist"
  find . -maxdepth 1 -type f ! -name sha256sums.txt ! -name files.txt -printf '%P\n' |
    sort >files.txt
  xargs -a files.txt -d '\n' sha256sum -- >sha256sums.txt
  while read -r f; do
    printf '  %8s  %s\n' "$(du -h --apparent-size -- "$f" | cut -f1)" "$f"
  done <files.txt
  rm -f files.txt
)
printf '\n  (cd dist && sha256sum -c sha256sums.txt)\n'
