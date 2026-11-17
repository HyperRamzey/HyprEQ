#!/usr/bin/env bash
# Copy the project's sources to the clipboard, each one annotated with its path.
#
#   scripts/grab-sources.sh                    # code + docs + build files (default)
#   scripts/grab-sources.sh --stat             # show what would be copied, copy nothing
#   scripts/grab-sources.sh --code             # skip the prose (*.md)
#   scripts/grab-sources.sh --docs             # prose only
#   scripts/grab-sources.sh --data             # also include measurements.json and Cargo.lock
#   scripts/grab-sources.sh 'src/*.rs' 'docs/*.md'   # restrict to path patterns
#   scripts/grab-sources.sh --numbers          # prefix every line with its line number
#   scripts/grab-sources.sh --out bundle.txt   # write a file instead of the clipboard
#
# Selection is from `git ls-files`, so nothing untracked and nothing under target/
# is ever included. Binaries and files over 2 MB are skipped with a note on stderr.
# Order is stable: build files, sources, tests, examples, scripts, docs, then prose,
# so two runs of the same tree paste in the same order.
set -euo pipefail

cd "$(dirname "$0")/.."

max_bytes=$((2 * 1024 * 1024))
out=""
to_clipboard=1
numbers=0
stat_only=0
mode="all"
data=0
patterns=()

while (($#)); do
    case "$1" in
        --stat) stat_only=1 ;;
        --code) mode="code" ;;
        --docs) mode="docs" ;;
        --data) data=1 ;;
        --numbers) numbers=1 ;;
        --out) shift; out="${1:?--out needs a path}"; to_clipboard=0 ;;
        -h | --help)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*)
            echo "unknown option: $1" >&2
            exit 2
            ;;
        *) patterns+=("$1") ;;
    esac
    shift
done

# Extension sets. `sh` is here because the scripts directory is shell, and `json`
# is only data, so it stays out until --data asks for it.
code_exts=(rs py sh toml)
doc_exts=(md)
data_exts=(json lock)
# Files with no extension that are still prose a reviewer wants next to the code.
name_allowlist=(LICENSE THIRD-PARTY-NOTICES NOTICE)

# What a path must match to be selected, given the mode. Built once.
wanted=()
case "$mode" in
    code)
        for e in "${code_exts[@]}"; do wanted+=("*.$e"); done
        ;;
    docs)
        for e in "${doc_exts[@]}"; do wanted+=("*.$e"); done
        ;;
    *)
        for e in "${code_exts[@]}" "${doc_exts[@]}"; do wanted+=("*.$e"); done
        ;;
esac
if ((data)); then
    for e in "${data_exts[@]}"; do wanted+=("*.$e"); done
fi

# Order groups explicitly rather than alphabetically: a reader pastes this into a
# review, and "Cargo.toml, then lib.rs, then the module it registers" reads better
# than "BUGFIX.md, Cargo.toml, README.md, ...".
order_groups=(
    "Cargo.toml"
    "src/"
    "tests/"
    "examples/"
    "scripts/"
    "docs/"
    "hypreq.py"
    "measurements.json"
    "README.md"
    "AGENTS.md"
    "BUGFIX.md"
    "LICENSE"
    "LICENSES/"
    "THIRD-PARTY-NOTICES"
)

mapfile -t tracked < <(git ls-files)

in_group() {
    # Is $1 in the same order group as $2? Exact file, or under the directory.
    local path="$1" group="$2"
    case "$group" in
        */) [[ "${path#"$group"}" != "$path" ]] ;;
        *) [[ "$path" == "$group" ]] ;;
    esac
}
matches_ext() {
    local path="$1" e
    for e in "${wanted[@]}"; do
        # shellcheck disable=SC2053  # right-hand side is a pattern on purpose
        [[ "$path" == $e ]] && return 0
    done
    if [[ "$mode" != "code" ]]; then
        local base="${path##*/}"
        for allowed in "${name_allowlist[@]}"; do
            [[ "$base" == "$allowed" ]] && return 0
        done
    fi
    return 1
}

declare -a picked=()
for group in "${order_groups[@]}"; do
    for f in "${tracked[@]}"; do
        in_group "$f" "$group" || continue
        matches_ext "$f" || continue
        picked+=("$f")
    done
done
# Anything tracked that no group claims still belongs in the bundle; dropping it
# silently would hand a reviewer a tree that looks complete and is not.
for f in "${tracked[@]}"; do
    claimed=0
    for group in "${order_groups[@]}"; do
        if in_group "$f" "$group"; then
            claimed=1
            break
        fi
    done
    ((claimed)) && continue
    matches_ext "$f" && picked+=("$f")
done

# Optional pattern restriction, matched with the shell's own globbing so `src/*.rs`
# behaves the way it looks.
if ((${#patterns[@]})); then
    filtered=()
    for f in "${picked[@]}"; do
        for p in "${patterns[@]}"; do
            # shellcheck disable=SC2053
            [[ "$f" == $p ]] && filtered+=("$f") && break
        done
    done
    picked=("${filtered[@]}")
fi

# Drop unreadable, binary and oversized files, saying so on stderr.
final=()
skipped=0
for f in "${picked[@]}"; do
    if [[ ! -r "$f" ]]; then
        echo "skip (unreadable): $f" >&2
        skipped=$((skipped + 1))
        continue
    fi
    size=$(wc -c <"$f")
    if ((size > max_bytes)); then
        echo "skip (over 2 MB): $f ($size bytes)" >&2
        skipped=$((skipped + 1))
        continue
    fi
    if ! grep -qI '' "$f" 2>/dev/null; then
        echo "skip (binary): $f" >&2
        skipped=$((skipped + 1))
        continue
    fi
    final+=("$f")
done

if ((${#final[@]} == 0)); then
    echo "nothing selected" >&2
    exit 1
fi

total=0
for f in "${final[@]}"; do
    total=$((total + $(wc -c <"$f")))
done

{
    echo "selected: ${#final[@]} file(s), ${total} bytes"
    echo "branch  : $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "tree    : DIRTY - the bundle contains working-tree contents, not HEAD"
    else
        echo "tree    : clean"
    fi
    ((skipped)) && echo "skipped : $skipped"
} >&2

if ((stat_only)); then
    printf '%s\n' "${final[@]}" >&2
    exit 0
fi

emit() {
    local f="$1"
    local lines size rel
    lines=$(wc -l <"$f")
    size=$(wc -c <"$f")
    rel="$f"
    printf '\n'
    printf '=%.0s' {1..78}
    printf '\nFILE: %s  (%s lines, %s bytes)\n' "$rel" "$lines" "$size"
    printf '=%.0s' {1..78}
    printf '\n'
    if ((numbers)); then
        nl -ba "$f"
    else
        cat "$f"
    fi
    # A trailing newline per file keeps the last line of one file from running into
    # the banner of the next when this is pasted into a fixed-width viewer.
    printf '\n'
}

header() {
    printf '=%.0s' {1..78}
    printf '\nHyprEQ sources  (%s @ %s%s)\n' \
        "$(git rev-parse --abbrev-ref HEAD)" \
        "$(git rev-parse --short HEAD)" \
        "$([[ -n "$(git status --porcelain)" ]] && echo ', working tree')"
    printf '%s file(s), %s bytes, generated %s\n' \
        "${#final[@]}" "$total" "$(date '+%Y-%m-%d %H:%M:%S %z')"
    printf 'Each block is one file, introduced by a FILE: line. No file was edited to\n'
    printf 'produce this; the tree is what it is.\n'
    printf '=%.0s' {1..78}
    printf '\n'
}

if [[ -n "$out" ]]; then
    {
        header
        for f in "${final[@]}"; do emit "$f"; done
    } >"$out"
    echo "wrote $out" >&2
    exit 0
fi

if ((to_clipboard)); then
    {
        header
        for f in "${final[@]}"; do emit "$f"; done
    } | if command -v wl-copy >/dev/null 2>&1 && [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
        wl-copy
    elif command -v xclip >/dev/null 2>&1; then
        xclip -selection clipboard
    else
        echo "no wl-copy or xclip found; writing sources-bundle.txt instead" >&2
        cat >sources-bundle.txt
    fi
else
    {
        header
        for f in "${final[@]}"; do emit "$f"; done
    } | cat
fi
