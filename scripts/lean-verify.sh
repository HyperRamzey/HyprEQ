#!/usr/bin/env bash
# Strict axiom gate for the HyprEQ Lean proofs.
#
# Requires elan/lake (Lean 4.36.0, pinned by lean/lean-toolchain).
# Exit 0 only if:
#   * the library builds;
#   * no source under lean/HyprEQ/ contains `sorry` or `admit`;
#   * every theorem declared under lean/HyprEQ/ is audited;
#   * every audited theorem depends only on [propext, Quot.sound] (or nothing).
#
# The audit refuses `Classical.choice` (which Mathlib's derived ordered-field
# instances drag in even for lemmas that print clean in isolation), `sorryAx`,
# `Lean.ofReduceBool` and `Lean.trustCompiler`.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
lean_dir="$repo_root/lean"

if [ -d "$HOME/.elan/bin" ]; then
  export PATH="$HOME/.elan/bin:$PATH"
fi
if ! command -v lake >/dev/null 2>&1; then
  echo "lake not found (install elan: https://github.com/leanprover/elan)" >&2
  exit 1
fi

cd "$lean_dir"

echo "== build =="
lake build

echo "== source scan =="
if grep -RnwE 'sorry|admit' HyprEQ; then
  echo "FAIL: forbidden placeholder in lean/HyprEQ/" >&2
  exit 1
fi

echo "== axiom audit =="
audit_log="$lean_dir/.lake/axiom-audit.log"
lake env lean AxiomAudit.lean > "$audit_log"
cat "$audit_log"

forbidden='Classical\.choice|sorryAx|Lean\.trustCompiler|Lean\.ofReduceBool'
if grep -E "$forbidden" "$audit_log"; then
  echo "FAIL: forbidden axiom in an audited theorem" >&2
  exit 1
fi

allowed="^'HyprEQ\.[A-Za-z0-9_.]+' (depends on axioms: \[(propext(, Quot\.sound)?|Quot\.sound)\]|does not depend on any axioms)$"
if grep -vE "$allowed" "$audit_log"; then
  echo "FAIL: unexpected audit output above" >&2
  exit 1
fi

expected=$(find HyprEQ -name '*.lean' -print0 | xargs -0 grep -hE '^theorem ' \
  | sed -E 's/^theorem ([A-Za-z0-9_]+).*/HyprEQ.\1/' | sort)
actual=$(sed -nE "s/^'HyprEQ\.([A-Za-z0-9_.]+)'.*/HyprEQ.\1/p" "$audit_log" | sort)
if [ "$expected" != "$actual" ]; then
  echo "FAIL: AxiomAudit.lean does not cover exactly the shipped theorems" >&2
  diff <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") >&2 || true
  exit 1
fi

count=$(printf '%s\n' "$actual" | wc -l)
echo "PASS: $count theorems audited; axioms subset of [propext, Quot.sound]"
