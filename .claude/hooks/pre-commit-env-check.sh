#!/usr/bin/env bash
# phase-40.6: pre-commit hook -- validate staged .env-shaped files via
# scripts/qa/env_syntax_check.py before allowing commit.
#
# Scope: ONLY files matching <prefix>/.env or <prefix>/.env.<suffix> in
# the staging area. Other files are ignored entirely (zero-cost when not
# touching env config). The hook reports violations + exits 1 on error,
# exits 0 on clean. See `.claude/hooks/install-pre-commit.md` (if exists)
# for the suggested git-hook wire-in incantation.
#
# Manual install:
#   ln -sf "$(pwd)/.claude/hooks/pre-commit-env-check.sh" .git/hooks/pre-commit
#
# Sourced from project root by the Claude Code session pre-commit pipeline
# (see phase-38.5 pre-commit wiring pattern).
set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "WARN: not in a git repo; skipping env syntax check" >&2
    exit 0
}

cd "$REPO_ROOT" || exit 2

STAGED_ENVS=$(git diff --cached --name-only --diff-filter=ACMR 2>/dev/null \
              | grep -E '(^|/)\.env(\.[^/]+)?$' || true)

if [ -z "$STAGED_ENVS" ]; then
    # Nothing env-shaped is being committed -- zero-cost exit.
    exit 0
fi

if [ ! -x "scripts/qa/env_syntax_check.py" ]; then
    echo "ERROR: scripts/qa/env_syntax_check.py missing or not executable" >&2
    echo "       phase-40.6 invariant broken; restore the file before committing" >&2
    exit 2
fi

exit_code=0
while IFS= read -r f; do
    [ -f "$f" ] || continue
    if ! python3 scripts/qa/env_syntax_check.py "$f"; then
        exit_code=1
    fi
done <<< "$STAGED_ENVS"

if [ "$exit_code" -ne 0 ]; then
    echo "" >&2
    echo "FAIL: env syntax violations in staged file(s) -- commit blocked" >&2
    echo "      Fix the reported lines + re-stage, or pass --no-verify (NOT recommended)" >&2
fi

exit "$exit_code"
