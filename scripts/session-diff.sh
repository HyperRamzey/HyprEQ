#!/usr/bin/env bash
# Copy a working session's diff to the clipboard, with the commits that made it up.
#
#   scripts/session-diff.sh                 # everything since 8 hours ago (default)
#   scripts/session-diff.sh 24h             # since 24 hours ago
#   scripts/session-diff.sh '2026-11-16 20:00'   # since a timestamp
#   scripts/session-diff.sh 5c5bf7d         # since a named commit (that commit excluded)
#   scripts/session-diff.sh --stat 8h       # show what would be copied, copy nothing
#
# The base is the last commit *before* the cut, so the diff contains every commit after
# it and nothing older. wl-copy is used when there is a Wayland session, xclip -selection
# clipboard otherwise; a pipe is fine too (`scripts/session-diff.sh 24h > session.diff`).
set -euo pipefail

stat_only=0
if [[ "${1:-}" == "--stat" ]]; then
    stat_only=1
    shift
fi

arg="${1:-8 hours ago}"
cd "$(dirname "$0")/.."

# A ref wins over a date, so `HEAD~5` and `5c5bf7d` behave like git expects.
if git rev-parse --verify --quiet "$arg^{commit}" >/dev/null; then
    base="$(git rev-parse "$arg^{commit}")"
    how="commit $arg"
else
    # 8h / 90m / 2d -> "8 hours ago", so git's own parser does the arithmetic.
    when="$arg"
    if [[ "$arg" =~ ^([0-9]+)([smhd])$ ]]; then
        n="${BASH_REMATCH[1]}"
        case "${BASH_REMATCH[2]}" in
            s) unit=second ;;
            m) unit=minute ;;
            h) unit=hour ;;
            d) unit=day ;;
        esac
        [[ "$n" == 1 ]] || unit="${unit}s"
        when="$n $unit ago"
    fi
    base="$(git rev-list -1 --before="$when" HEAD)"
    how="since $when"
fi

if [[ -z "$base" ]]; then
    echo "no commit found $how - the repo is younger than that" >&2
    exit 1
fi

short="$(git rev-parse --short "$base")"
count="$(git rev-list --count "$base..HEAD")"

{
    echo "base    : $short  $(git log -1 --format='%s' "$base")"
    echo "head    : $(git rev-parse --short HEAD)  $(git log -1 --format='%s' HEAD)"
    echo "commits : $count  ($how)"
    git diff --shortstat "$base..HEAD"
} >&2

if ((stat_only)); then
    exit 0
fi

{
    echo "Commits in this range (newest first)"
    echo "===================================="
    git log --format='%h %ad %s' --date=format:'%Y-%m-%d %H:%M' "$base..HEAD"
    echo
    echo "Diff $base..HEAD"
    echo "================="
    git diff --find-renames "$base..HEAD"
} | if command -v wl-copy >/dev/null 2>&1 && [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    wl-copy
elif command -v xclip >/dev/null 2>&1; then
    xclip -selection clipboard
else
    echo "no wl-copy or xclip found; writing session.diff instead" >&2
    cat > session.diff
fi
