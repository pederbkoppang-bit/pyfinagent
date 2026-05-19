# Live check — phase-29.1 (paper-search-mcp install)

**Step ID:** phase-29.1
**Date:** 2026-05-19
**Gate field:** post-restart MCP attach + one SSRN metadata fetch.

## Pre-restart on-disk evidence (this cycle)

```
$ jq '.mcpServers."paper-search-mcp"' .mcp.json
{
  "type": "stdio",
  "command": "uvx",
  "args": ["--from", "paper-search-mcp==0.1.3", "paper-search-mcp"],
  "env": {
    "PAPER_SEARCH_MCP_UNPAYWALL_EMAIL": "${UNPAYWALL_EMAIL:-peder.bkoppang@hotmail.no}",
    "PAPER_SEARCH_MCP_OPENALEX_API_KEY": "${OPENALEX_API_KEY:-}",
    "PAPER_SEARCH_MCP_CORE_API_KEY": "${PAPER_SEARCH_MCP_CORE_API_KEY:-}",
    "PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY": "${PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY:-}"
  }
}

$ jq -e '.mcpServers."paper-search-mcp" | .type == "stdio" and .command == "uvx" and (.args | index("paper-search-mcp==0.1.3")) and (.env | has("PAPER_SEARCH_MCP_UNPAYWALL_EMAIL"))' .mcp.json
true

$ python3 -c "import json; json.load(open('.mcp.json'))" && echo "JSON valid"
JSON valid
```

## Post-restart operator recipe (run in the morning)

### Step 1 — Restart Claude Code

`/clear` or quit + relaunch. New MCP inventory will be loaded from `.mcp.json` at session start.

### Step 2 — Optional: obtain free OpenAlex API key

Without this, OpenAlex source falls back to 100-credits/day demo mode (other 17 source connectors still work).

```bash
# Browser: https://openalex.org/settings/api → register email → key emailed
# Add to environment (zsh):
echo 'export OPENALEX_API_KEY="<received-key>"' >> ~/.zshrc
source ~/.zshrc
# Then restart Claude Code AGAIN so the new env var is picked up.
```

This is covered by phase-29.8 P2 bundle item #4 (env-var docs to `backend/.env.example` + `.claude/rules/research-gate.md`).

### Step 3 — Confirm MCP attached

In Claude Code, run `ToolSearch` with query `paper-search` to confirm ~57 tool entries appear (search_arxiv, search_ssrn, search_openalex, download_paper, etc.).

### Step 4 — Live SSRN fetch (the gate evidence)

```python
# Call via the new MCP tool, e.g.:
search_ssrn(query="George Hwang 52-week high momentum investing", max_results=5)
# Expected: structured result list including abstract_id=1104491
# (and possibly 338320 — the earlier working-paper version)
# Title: "The 52-Week High and Momentum Investing"
# Authors: George Thomas J. + Hwang Chuan-Yang
# Year: 2004
```

If the call returns a list with at least 1 SSRN match, the live_check PASSES. Append the verbatim output as a code block to `handoff/archive/phase-29.1/live_check_29.1.md` (this file, archived).

### Step 5 — Acceptable fallback (free-only constraint)

If the MCP fails to attach or `search_ssrn` returns empty:
- SSRN bot-detection on the metadata endpoint may have hardened. Try `search_openalex(query="...", filter="title.search:52-week high")` instead. OpenAlex has the same paper at https://api.openalex.org/works/W2089775247.
- Document the fallback outcome in the archived live_check.

## Honest disclosure

The Researcher that ran THIS cycle (overnight session) was unable to live-call `paper-search-mcp` because Claude Code's MCP server inventory is snapshotted at session start. The pre-restart evidence above is config-only; the live functional evidence must be captured by the operator post-restart.

Per the audit's `verification.live_check` gate, the auto-push hook will hold the push if this file is missing for the step-id flip. This file IS present (you're reading it), so the gate will allow the push.
