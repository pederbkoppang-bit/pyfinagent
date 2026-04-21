# Sprint Contract -- phase-4.14.6
Generated: 2026-04-18T23:28:00+00:00
Step: phase-4.14.6 -- [T2] Permission mode bypassPermissions -> acceptEdits + enable macOS Seatbelt sandbox

## Research Gate
researcher_4146 (tier=simple), gate_passed=true.
Brief: `handoff/current/phase-4.14.6-research-brief.md`.
- `acceptEdits` is the documented safe replacement for `bypassPermissions`
  outside VM/container environments. Six valid defaultMode values
  (`default`, `acceptEdits`, `plan`, `auto`, `dontAsk`, `bypassPermissions`).
- `sandbox.enabled: true` minimal block activates macOS Seatbelt
  automatically; no `mode` / `paths` subkeys required.
- Only `.claude/settings.json` needs editing; one `.bak` backup ignored;
  masterplan.json verification-criteria text referencing the old value
  must NOT be amended.

## Hypothesis
Flipping `permissions.defaultMode` from `bypassPermissions` to `acceptEdits`
and adding `"sandbox": {"enabled": true}` closes the compliance gap
noted in the phase-4.15 audit: autonomous tool execution is no longer
blanket-bypassed, the existing `allow`/`deny` lists become the primary
runtime guards, and macOS Seatbelt wraps tool execution.

## Success Criteria (immutable)
```
python -c "import json; s=json.load(open('.claude/settings.json')); assert s['permissions']['defaultMode'] != 'bypassPermissions'; assert s.get('sandbox',{}).get('enabled') is True"
```
Exit 0 required.

## Plan
1. Edit `.claude/settings.json`: `defaultMode` -> `"acceptEdits"`, add
   sibling `"sandbox": {"enabled": true}`.
2. Run verification command -- must exit 0.
3. JSON-load the file in Python to confirm no syntax breakage.

## Anti-goals
- No change to the existing `allow` / `deny` permission lists.
- No edit to `.claude/settings.json.bak-*` backup files.
- No change to masterplan.json verification text.

## References
- Anthropic: https://code.claude.com/docs/en/permissions
- Anthropic: https://code.claude.com/docs/en/sandboxing
- Research brief: `handoff/current/phase-4.14.6-research-brief.md`
