#!/usr/bin/env bash
# phase-4.14.27: PreToolUse danger guard.
#
# Blocks the following unless explicitly allowed:
#   - rm -rf ...
#   - git push --force / --force-with-lease / -f
#   - git reset --hard ...
#   - mcp__<server>__execute_sql  (unless MIGRATE token in env)
#
# Exit 2 = block the tool call (PreToolUse convention).
# Exit 0 = allow.
#
# Designed to FAIL OPEN on any internal error -- a broken guard must
# not brick the session. Only explicit pattern matches block.

# Intentionally NOT using `set -e`: we trap every failure mode and
# default to allow.

TOOL="${CLAUDE_TOOL_NAME:-}"
INPUT="${CLAUDE_TOOL_INPUT:-}"

# Parse stdin (Claude Code's documented hook protocol) only when env
# not populated and stdin is actually present + non-interactive.
if [ -z "$TOOL" ] && [ ! -t 0 ]; then
    payload=$(cat 2>/dev/null || true)
    if [ -n "$payload" ]; then
        PY="$(command -v python3 || command -v python || true)"
        if [ -n "$PY" ]; then
            TOOL=$(printf '%s' "$payload" | "$PY" -c 'import sys,json
try:
    d = json.loads(sys.stdin.read() or "{}")
    print(d.get("tool_name", ""))
except Exception:
    pass' 2>/dev/null || true)
            INPUT=$(printf '%s' "$payload" | "$PY" -c 'import sys,json
try:
    d = json.loads(sys.stdin.read() or "{}")
    print(json.dumps(d.get("tool_input", {})))
except Exception:
    print("")' 2>/dev/null || true)
        fi
    fi
fi

# phase-4.16.2: audit JSONL lives in handoff/audit/ (not the root).
AUDIT="${CLAUDE_PROJECT_DIR:-$(pwd)}/handoff/audit/pre_tool_use_audit.jsonl"
mkdir -p "$(dirname "$AUDIT")" 2>/dev/null || true

log_event() {
    local verdict="$1"; local reason="$2"
    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "now")
    printf '{"ts":"%s","tool":"%s","verdict":"%s","reason":"%s"}\n' \
        "$ts" "$TOOL" "$verdict" "$reason" >> "$AUDIT" 2>/dev/null || true
}

# On block, also emit a human-readable stderr message so Claude Code does not
# surface the opaque "No stderr output" line. exit 2 is a PreToolUse block
# per Claude Code hook convention.
block_with_msg() {
    local reason="$1"
    log_event "block" "$reason"
    printf 'pre-tool-use-danger blocked this call: %s\n' "$reason" >&2
    printf 'To override (one-shot): re-run with CLAUDE_ALLOW_DANGER=1 in env.\n' >&2
    printf 'Safer alternatives for __pycache__ cleanup:\n' >&2
    printf '  find <dir> -name "__pycache__" -delete\n' >&2
    printf '  python -c "import shutil,pathlib; [shutil.rmtree(p) for p in pathlib.Path(\"<dir>\").rglob(\"__pycache__\")]"\n' >&2
    exit 2
}

# Escape hatch for the session operator.
if [ "${CLAUDE_ALLOW_DANGER:-}" = "1" ]; then
    log_event "allow" "CLAUDE_ALLOW_DANGER=1"
    exit 0
fi

# ── Bash / shell command patterns ────────────────────────────────
if [ "$TOOL" = "Bash" ]; then
    cmd=""
    # Prefer JSON field; fall back to raw INPUT if not JSON.
    PY="$(command -v python3 || command -v python || true)"
    if [ -n "$INPUT" ] && [ -n "$PY" ]; then
        cmd=$(printf '%s' "$INPUT" | "$PY" -c 'import sys,json
try:
    d = json.loads(sys.stdin.read() or "{}")
    print(d.get("command", ""))
except Exception:
    pass' 2>/dev/null || true)
    fi
    if [ -z "$cmd" ] && [ -n "$INPUT" ]; then
        first="${INPUT:0:1}"
        if [ "$first" != "{" ]; then
            cmd="$INPUT"
        fi
    fi
    # Target-aware `rm -rf` gate. Uses Python shell-word parsing so legitimate
    # scoped cleanup (find -exec rm -rf, pycache, node_modules, explicit project
    # subpaths) is allowed while catastrophic targets are blocked.
    if [ -n "$PY" ] && printf '%s' "$cmd" | grep -qE '(^|[[:space:]])rm[[:space:]]+(-[^[:space:]]*r[^[:space:]]*|--recursive)'; then
        verdict=$(printf '%s' "$cmd" | "$PY" -c '
import sys, shlex
DANGEROUS = {"/", "~", "$HOME", ".", "..", "*", "/*", "/**"}
try:
    tokens = shlex.split(sys.stdin.read(), posix=True)
except Exception:
    print("allow")
    sys.exit(0)
i = 0
while i < len(tokens):
    t = tokens[i]
    if t == "rm" or t.endswith("/rm"):
        j = i + 1
        has_recursive = False
        targets = []
        while j < len(tokens) and tokens[j] not in (";", "&&", "||", "|", "&"):
            tok = tokens[j]
            if tok.startswith("-") and tok != "-":
                if "r" in tok.lower() or tok == "--recursive":
                    has_recursive = True
            else:
                targets.append(tok)
            j += 1
        if has_recursive:
            for tgt in targets:
                if tgt in DANGEROUS or tgt == "~" or tgt.startswith("~/") or tgt.startswith("$HOME"):
                    print("block:" + tgt)
                    sys.exit(0)
        i = j
    else:
        i += 1
print("allow")
' 2>/dev/null || printf 'allow')
        case "$verdict" in
            block:*)
                block_with_msg "rm -rf on dangerous target: ${verdict#block:}"
                ;;
        esac
    fi
    # Static git guards.
    case "$cmd" in
        *"git push --force"*|*"git push -f "*|*"git push -f"|*"git push --force-with-lease"*)
            block_with_msg "git push --force detected" ;;
        *"git reset --hard"*)
            block_with_msg "git reset --hard detected" ;;
    esac
    # File-level checkout/restore guard. Patterns matched (word-boundary
    # regex, so 'git checkout -- ' appearing INSIDE a grep/rg/awk quoted
    # argument is not blocked -- that is just searching for the string,
    # not running git):
    #   git checkout -- <path>
    #   git restore <path>
    # Word boundary = start-of-line, whitespace, `;`, `&&`, or `||`.
    if [[ "$cmd" =~ (^|[[:space:]]|\;|\&\&|\|\|)git[[:space:]]+checkout[[:space:]]+(--|HEAD) ]]; then
        block_with_msg "file-level git checkout detected -- silently discards working-tree edits"
    fi
    if [[ "$cmd" =~ (^|[[:space:]]|\;|\&\&|\|\|)git[[:space:]]+restore[[:space:]] ]]; then
        block_with_msg "git restore detected -- silently discards working-tree edits"
    fi
fi

# ── MCP execute_sql gate ─────────────────────────────────────────
case "$TOOL" in
    mcp__*__execute_sql)
        if [ "${MCP_MIGRATE_TOKEN:-}" != "granted" ]; then
            block_with_msg "execute_sql without MCP_MIGRATE_TOKEN=granted"
        fi
        ;;
esac

log_event "allow" "ok"
exit 0
