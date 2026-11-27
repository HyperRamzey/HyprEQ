#!/usr/bin/env bash
# Convenience entry point from inside lean/.  The canonical gate lives at the
# repository root and is what CI runs; this wrapper just saves a `cd`.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$repo_root/scripts/lean-verify.sh" "$@"
