# Experiment Results -- phase-4.14.6
Generated: 2026-04-18T23:38:00+00:00
Step: [T2] Permission mode bypassPermissions -> acceptEdits + enable macOS Seatbelt sandbox.

## What was built
Two-key edit to `.claude/settings.json`:
- `permissions.defaultMode`: `"bypassPermissions"` -> `"acceptEdits"`
- New sibling block `"sandbox": {"enabled": true}` added between `env`
  and `permissions` for readability.

## Files changed
- `.claude/settings.json` (2 keys modified/added; rest unchanged).

## Verbatim verification output

Semantic equivalent of the immutable command (the literal command uses
`!=` inline, which is harmlessly history-expanded by zsh when my tool
wrapper pipes it through a sub-shell; running the identical logic via
a script file avoids the shell-escape quirk):

```
$ cat /tmp/verify_4146.py
import json
s = json.load(open(".claude/settings.json"))
assert s["permissions"]["defaultMode"] != "bypassPermissions", "defaultMode still bypassPermissions"
assert s.get("sandbox", {}).get("enabled") is True, "sandbox.enabled not True"
print("defaultMode =", s["permissions"]["defaultMode"])
print("sandbox =", s["sandbox"])
print("IMMUTABLE VERIFICATION PASS")

$ python /tmp/verify_4146.py
defaultMode = acceptEdits
sandbox = {'enabled': True}
IMMUTABLE VERIFICATION PASS
```

Exit code: **0**.

## Artifact shape
- `permissions.allow` and `permissions.deny` lists unchanged -- still
  the primary runtime guards as recommended by the research brief.
- Sandbox shape minimal per Anthropic docs; macOS Seatbelt activates
  automatically with just `enabled: true`; no `mode` or `paths`
  required for baseline.

## Handoff notes for Q/A
- The immutable verification command in masterplan.json contains `!=`
  which zsh (and some tool-wrapper shells) expand as history syntax.
  Q/A should run the check with `bash --noprofile --norc` OR via the
  Python script form to reproduce cleanly. The SEMANTIC check is
  unchanged: both clauses asserted.
- `.claude/settings.json.bak-harness-*` backup files are NOT modified
  (they have no runtime effect).
- No masterplan.json verification-criteria text was amended.

## References
- Contract: `handoff/current/phase-4.14.6-contract.md`
- Research brief: `handoff/current/phase-4.14.6-research-brief.md`
- Anthropic docs: https://code.claude.com/docs/en/permissions ;
  https://code.claude.com/docs/en/sandboxing
