#!/usr/bin/env bash
# phase-4.9 step 4.9.1 -- tag-signed-commit enforcement for
# backend/governance/limits.yaml.
#
# Verifies that any change to the immutable limits file is
# covered by a GPG-signed `limits-rotation-YYYYMMDD` annotated
# tag, signed by an authorized owner, with a meaningful approval
# message.
#
# Usage:
#   verify_limits_tag.sh            # strict, fails hard
#   verify_limits_tag.sh --dry-run  # prints [DRY-RUN] warnings, exits 0
#
# Exit codes:
#   0 = all checks passed (or --dry-run)
#   1 = one or more checks failed (strict mode only)

set -u -o pipefail
# NOTE: intentionally NOT using `set -e` so we can aggregate all
# failures before exiting (helps during the signed-rotation PR).

LIMITS_FILE="backend/governance/limits.yaml"
TAG_PATTERN="limits-rotation-*"
MIN_MSG_LEN=30

# Authorized tag signers. Add a new line per additional approved
# human (or an organizational key) during a rotation ceremony.
ALLOWED_SIGNERS=(
    "peder.bkoppang@hotmail.no"
)

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
fi

FAIL_COUNT=0
fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    if $DRY_RUN; then
        echo "[DRY-RUN] WOULD FAIL: $1" >&2
    else
        echo "ERROR: $1" >&2
    fi
}
info() { echo "INFO: $1" >&2; }

# Locate the repo root so the script can run from any cwd.
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT" || { echo "ERROR: not inside a git repo"; exit 1; }

# -------------------------------------------------------------------
# 1. Find most recent limits-rotation tag. Absent tag is a fail in
#    strict mode; in --dry-run we emit the warning and continue so
#    later checks still get exercised where possible.
# -------------------------------------------------------------------
LATEST_TAG="$(git tag --list "$TAG_PATTERN" --sort=-version:refname \
              2>/dev/null | head -n 1 || true)"

if [ -z "$LATEST_TAG" ]; then
    fail "no $TAG_PATTERN tag found in repo"
    if $DRY_RUN; then
        info "dry-run: skipping downstream tag-object checks"
        echo "[DRY-RUN] verify_limits_tag completed with $FAIL_COUNT would-fails"
        exit 0
    fi
    exit 1
fi
info "latest rotation tag: $LATEST_TAG"

# -------------------------------------------------------------------
# 2. Annotated-tag check. Lightweight tags point directly at a
#    commit object; annotated tags are their own tag object with a
#    tagger + message body. `git verify-tag` only works on the
#    latter.
# -------------------------------------------------------------------
TAG_OBJECT_TYPE="$(git cat-file -t "$LATEST_TAG" 2>/dev/null || echo missing)"
if [ "$TAG_OBJECT_TYPE" != "tag" ]; then
    fail "tag $LATEST_TAG is lightweight (type=$TAG_OBJECT_TYPE); annotated required"
fi

# -------------------------------------------------------------------
# 3. GPG signature: `git verify-tag` exits 0 on valid signature.
#    Public key of the tagger must be present in the local keyring
#    (or GitHub Actions runner -- imported via a prior workflow step).
# -------------------------------------------------------------------
if [ "$TAG_OBJECT_TYPE" = "tag" ]; then
    if ! git verify-tag "$LATEST_TAG" >/dev/null 2>&1; then
        fail "git verify-tag rejected $LATEST_TAG (bad signature or missing key)"
    fi
fi

# -------------------------------------------------------------------
# 4. Tagger identity allow-list. Read the `tagger` header from the
#    raw tag object; compare the email against ALLOWED_SIGNERS.
#    This is separate from (and more authoritative than) the GPG
#    signer identity, which can be manipulated via key-sharing.
# -------------------------------------------------------------------
if [ "$TAG_OBJECT_TYPE" = "tag" ]; then
    TAGGER_LINE="$(git cat-file -p "$LATEST_TAG" 2>/dev/null \
                   | grep '^tagger ' | head -n 1 || true)"
    TAGGER_EMAIL="$(echo "$TAGGER_LINE" \
                    | sed -n 's/.*<\([^>]*\)>.*/\1/p' || true)"
    if [ -z "$TAGGER_EMAIL" ]; then
        fail "could not parse tagger email from $LATEST_TAG"
    else
        AUTHORIZED=false
        for addr in "${ALLOWED_SIGNERS[@]}"; do
            if [ "$TAGGER_EMAIL" = "$addr" ]; then
                AUTHORIZED=true
                break
            fi
        done
        if ! $AUTHORIZED; then
            fail "tagger $TAGGER_EMAIL not in ALLOWED_SIGNERS"
        fi
    fi
fi

# -------------------------------------------------------------------
# 5. Approval message requirement. Annotation body (everything
#    after the first blank line in the tag object) must be >=30
#    chars and contain the word "approved" (case-insensitive).
# -------------------------------------------------------------------
if [ "$TAG_OBJECT_TYPE" = "tag" ]; then
    TAG_MSG="$(git cat-file -p "$LATEST_TAG" 2>/dev/null \
               | awk '/^$/{found=1; next} found{print}' || true)"
    MSG_LEN=${#TAG_MSG}
    if [ "$MSG_LEN" -lt "$MIN_MSG_LEN" ]; then
        fail "tag annotation too short ($MSG_LEN chars; need >=$MIN_MSG_LEN)"
    fi
    if ! echo "$TAG_MSG" | grep -qi 'approved'; then
        fail "tag annotation must contain 'approved' (case-insensitive)"
    fi
fi

# -------------------------------------------------------------------
# 6. Tag-covers-head check. If limits.yaml has commits AFTER the
#    rotation tag, the tag is stale -- the new rotation must be
#    re-signed.
# -------------------------------------------------------------------
if [ "$TAG_OBJECT_TYPE" = "tag" ]; then
    UNCOVERED="$(git log "${LATEST_TAG}..HEAD" --oneline \
                  -- "$LIMITS_FILE" 2>/dev/null || true)"
    if [ -n "$UNCOVERED" ]; then
        fail "limits.yaml changed since $LATEST_TAG:\n$UNCOVERED"
    fi
fi

# -------------------------------------------------------------------
# Exit.
# -------------------------------------------------------------------
if $DRY_RUN; then
    echo "[DRY-RUN] verify_limits_tag completed with $FAIL_COUNT would-fails"
    exit 0
fi
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "ERROR: verify_limits_tag failed ($FAIL_COUNT checks)" >&2
    exit 1
fi
echo "OK: $LATEST_TAG covers HEAD; signer + annotation valid"
exit 0
