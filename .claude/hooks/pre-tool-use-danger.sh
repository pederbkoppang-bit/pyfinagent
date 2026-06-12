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
    # phase-62.0 (away-ops rails 3+9): robust force-push + launchd removal,
    # scoped PER COMMAND SEGMENT (split on ; && || |) so prose inside a
    # commit message / heredoc in ANOTHER segment cannot false-positive
    # (live-discovered: the 62.0 commit message mentioning the guards
    # blocked its own commit). Flags are position-free and "+refspec"
    # forces with no flag at all -- both missed by the static case above.
    if [ -n "$PY" ] && { [[ "$cmd" == *"git push"* ]] || [[ "$cmd" == *launchctl* ]]; }; then
        seg_verdict=$(printf '%s' "$cmd" | "$PY" -c '
import sys, re
cmd = sys.stdin.read()
for seg in re.split(r";|&&|\|\||\|", cmd):
    if re.search(r"(^|\s)git\s+push(\s|$)", seg):
        if re.search(r"\s(--force|--force-with-lease|--force-if-includes|-f)(\s|$|=)", seg):
            print("block:force-push flag"); sys.exit(0)
        if re.search(r"git\s+push\s+\S+\s+\+[A-Za-z0-9_./-]+", seg):
            print("block:+refspec force-push"); sys.exit(0)
    if re.search(r"(^|\s)launchctl\s+(bootout|unload|remove|disable)\s", seg) and "com.pyfinagent." in seg:
        print("block:launchctl removal on pyfinagent agent"); sys.exit(0)
print("allow")
' 2>/dev/null || printf 'allow')
        case "$seg_verdict" in
            block:*force*)
                block_with_msg "force-push variant detected (${seg_verdict#block:}) -- away-ops rail 3" ;;
            block:*launchctl*)
                block_with_msg "launchctl removal verb on a pyfinagent agent -- away-ops rail 9 (kickstart is the allowed restart path)" ;;
        esac
    fi
    # phase-62.0 (away-ops rail 1): backend/.env write tripwire. Shapes per
    # BashFAQ/050 (a complete parser is impossible; the 62.4 sentinel
    # reconciliation is the backstop): >>/> redirects, sed -i, tee, perl -i.
    # Gate: handoff/away_ops/tokens_cursor fresh (mtime < 6h) = an operator
    # token was just applied = the write is authorized.
    if [[ "$cmd" =~ (\>\>|\>)[[:space:]]*([A-Za-z0-9_./\"\x27-]*/)?backend/\.env ]] \
       || [[ "$cmd" =~ sed[[:space:]]+(-[A-Za-z]*i|--in-place)[^\;\&\|]*backend/\.env ]] \
       || [[ "$cmd" =~ tee[[:space:]]+(-a[[:space:]]+)?[^\;\&\|]*backend/\.env ]] \
       || [[ "$cmd" =~ perl[[:space:]]+[^\;\&\|]*-[A-Za-z]*i[^\;\&\|]*backend/\.env ]]; then
        CURSOR="${CLAUDE_PROJECT_DIR:-$(pwd)}/handoff/away_ops/tokens_cursor"
        fresh=0
        if [ -f "$CURSOR" ]; then
            now_s=$(date +%s 2>/dev/null || echo 0)
            cur_s=$(stat -f %m "$CURSOR" 2>/dev/null || echo 0)
            if [ "$now_s" -gt 0 ] && [ "$cur_s" -gt 0 ] && [ $((now_s - cur_s)) -lt 21600 ]; then
                fresh=1
            fi
        fi
        if [ "$fresh" != "1" ]; then
            log_event "block" "backend/.env write without fresh token cursor"
            printf 'pre-tool-use-danger blocked this call: backend/.env write without a fresh operator token (away-ops rail 1).\n' >&2
            printf 'Do NOT retry. Record the ask in handoff/away_ops/pending_tokens.json and move on to the next calendar item.\n' >&2
            printf 'The gate opens automatically when a session applies a matching operator token (tokens_cursor mtime < 6h).\n' >&2
            exit 2
        fi
        log_event "allow" "backend/.env write with fresh token cursor"
    fi
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

# ── phase-62.0: Edit/Write tool coverage for backend/.env ────────
# A Bash-only tripwire is bypassable via the Edit/Write tools
# (researcher finding); same token-cursor gate applies.
case "$TOOL" in
    Edit|Write|NotebookEdit)
        fp=""
        PY="$(command -v python3 || command -v python || true)"
        if [ -n "$INPUT" ] && [ -n "$PY" ]; then
            fp=$(printf '%s' "$INPUT" | "$PY" -c 'import sys,json
try:
    d = json.loads(sys.stdin.read() or "{}")
    print(d.get("file_path", ""))
except Exception:
    pass' 2>/dev/null || true)
        fi
        case "$fp" in
            *backend/.env)
                CURSOR="${CLAUDE_PROJECT_DIR:-$(pwd)}/handoff/away_ops/tokens_cursor"
                fresh=0
                if [ -f "$CURSOR" ]; then
                    now_s=$(date +%s 2>/dev/null || echo 0)
                    cur_s=$(stat -f %m "$CURSOR" 2>/dev/null || echo 0)
                    if [ "$now_s" -gt 0 ] && [ "$cur_s" -gt 0 ] && [ $((now_s - cur_s)) -lt 21600 ]; then
                        fresh=1
                    fi
                fi
                if [ "$fresh" != "1" ]; then
                    log_event "block" "$TOOL on backend/.env without fresh token cursor"
                    printf 'pre-tool-use-danger blocked this call: %s on backend/.env without a fresh operator token (away-ops rail 1).\n' "$TOOL" >&2
                    printf 'Do NOT retry. Record the ask in handoff/away_ops/pending_tokens.json and move on.\n' >&2
                    exit 2
                fi
                log_event "allow" "$TOOL on backend/.env with fresh token cursor"
                ;;
        esac
        ;;
esac

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
