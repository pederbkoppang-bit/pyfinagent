# Contract -- Cycle 89 / phase-4.9 step 4.9.1

Step: 4.9.1 Tag-signed-commit enforcement in CI

## Research-gate upheld (4th cycle)

Researcher (14 URLs: git-scm verify-tag + cat-file + hooks docs,
GitHub Actions workflow syntax, Pro Git Signing Your Work) +
Explore (CODEOWNERS absent, .github/workflows has 3 existing
workflows, user git identity is peder.bkoppang...).

Key research findings:
- `actions/checkout` needs `fetch-depth: 0 + fetch-tags: true` or
  tags aren't visible in CI.
- Path filters + tag triggers are MUTUALLY EXCLUSIVE on a single
  `on:` event -> need two separate triggers (push-paths and
  push-tags).
- `git verify-tag` only works on ANNOTATED tags; lightweight tags
  must be rejected explicitly via `git cat-file -t`.
- `--dry-run` convention: run all checks; instead of `exit 1` on
  failure, print `[DRY-RUN] WOULD FAIL: <reason>` and exit 0.

## Scope

Files created:

1. **NEW** `scripts/governance/verify_limits_tag.sh`
   Bash script implementing the full verify chain:
   - Find most recent `limits-rotation-*` tag.
   - Assert tag object is ANNOTATED (rejects lightweight).
   - `git verify-tag` (GPG signature).
   - `git cat-file -p TAG | grep tagger` -> email matches
     `ALLOWED_SIGNERS` list.
   - Tag annotation body >= 30 chars AND contains "approved".
   - Check `git log TAG..HEAD -- backend/governance/limits.yaml`
     is empty (tag covers the current limits state).
   - `--dry-run` flag flips `fail()` to `[DRY-RUN] WOULD FAIL:`
     warnings + exit 0. Used by masterplan verification.
   - Default (no flag): strict; exit 1 on any failure.

2. **NEW** `.github/workflows/limits-tag-enforcement.yml`
   Two-trigger workflow:
   - Job A: `on.push.paths: backend/governance/limits.yaml` ->
     runs `verify_limits_tag.sh` (strict).
   - Job B: `on.push.tags: limits-rotation-*` -> runs
     `verify_limits_tag.sh` (strict).
   Both use `actions/checkout@v4` with `fetch-depth: 0` +
   `fetch-tags: true`.

3. **NEW** `.github/CODEOWNERS`
   `backend/governance/limits.yaml @pederbkoppang-bit`
   requires an explicit owner review on any PR touching the file.

4. **NEW** `scripts/audit/limits_tag_audit.py`
   Verifies:
   (a) verify_limits_tag.sh exists + is executable + --dry-run
       exits 0.
   (b) CI workflow exists with both triggers, fetch-depth 0,
       fetch-tags true, and invokes the verify script.
   (c) CODEOWNERS entry protects limits.yaml.
   (d) Simulated: create a lightweight tag on a temp repo,
       script rejects it. [optional / deferred -- bash-in-python
       test complexity]
   Teeth are regex-based structural checks; bash script logic is
   exercised by --dry-run in (a).

## Immutable success criteria

1. ci_workflow_landed -- .github/workflows/limits-tag-enforcement.yml
   exists.
2. unsigned_push_rejected -- script has a `git verify-tag` check.
3. wrong_owner_rejected -- script has an ALLOWED_SIGNERS allow-list.
4. approval_message_required -- script checks tag body length +
   "approved" keyword.

## Verification (immutable, from masterplan)

    bash scripts/governance/verify_limits_tag.sh --dry-run

Plus: `python scripts/audit/limits_tag_audit.py --check`.

## Anti-rubber-stamp

qa must:
- Read the script and confirm each of the four enforcement checks
  is REAL code (not a comment).
- Verify ALLOWED_SIGNERS is a non-empty list with a real email,
  not `["any@"]`.
- Confirm the workflow YAML has BOTH `paths` AND `tags` triggers
  (per researcher's "mutually exclusive" finding).
- Verify `--dry-run` does NOT silently swallow all errors -- it
  must still PRINT the would-fail reasons.

## References

- Researcher cycle-89 findings (14 URLs).
- git-scm verify-tag + cat-file docs.
- GitHub Actions workflow-syntax docs on paths/tags events.
- Pro Git Signing Your Work.
