# live_check 76.9.3 — verbatim live evidence (2026-07-25)

## 1. BEFORE — the broken state, reproduced first-hand

`ddgs` absent from the venv:

```
$ .venv/bin/pip show ddgs
WARNING: Package(s) not found: ddgs
$ .venv/bin/pip show duckduckgo_search | head -2
Name: duckduckgo_search
Version: 8.1.1          <- the FROZEN pre-rename package the dist METADATA still declares
```

**The step's own immutable command passes in the fully broken state** (measured, not
assumed) — `check_pkg('ddgs')` runs inside `__init__`, not at module import:

```
$ .venv/bin/python -c "from gpt_researcher.retrievers.duckduckgo.duckduckgo import Duckduckgo; print('MODULE IMPORT: OK')"
MODULE IMPORT: OK (immutable cmd would PASS)
immutable-import exit=0

$ .venv/bin/python -c "... Duckduckgo('guard probe') ..."
CONSTRUCT: FAILED -> ImportError Unable to import ddgs. Please install with `pip install -U ddgs`
```

That ImportError is character-for-character the string in the live nightly memo logs.
This is why the guards below CONSTRUCT rather than import: an import-only guard here
is a guard that cannot fail.

## 2. Install — targeted, not `-r` (langchain-core drift avoidance)

```
$ .venv/bin/pip show langchain-core | head -2      # BEFORE
Name: langchain-core
Version: 1.4.8

$ .venv/bin/pip install "ddgs==9.14.4"
Successfully installed ddgs-9.14.4 fake-useragent-2.2.0 h2-4.4.0 hpack-4.2.0 hyperframe-6.1.0 primp-1.3.1 socksio-1.0.0

$ .venv/bin/pip show langchain-core | head -2      # AFTER
Name: langchain-core
Version: 1.4.8                                     <- NO drift
```

Disclosed, pre-existing and NOT caused by this step: pip warns
`gpt-researcher 0.14.8 requires numpy<2.3.0,>=2.0.0, but you have numpy 2.4.4`.
The install touched no numpy (absent from the "Successfully installed" list); the
conflict predates this step and is left untouched as out-of-boundary.

## 3. AFTER — criterion 1 (construction) + criterion 2 (one live retrieval)

```
$ .venv/bin/python -c "... Duckduckgo('probe') ..."
CONSTRUCT OK -> ddgs.ddgs.DDGS
ddgs version: 9.14.4

$ .venv/bin/python -c "... Duckduckgo('post-earnings announcement drift equity returns').search(max_results=3) ..."
type: list | n = 3
  [1] title: A review of the Post-Earnings-Announcement Drift - ScienceDirect
      href : https://www.sciencedirect.com/science/article/pii/S2214635020303750
      body : Abstract The "Post-Earnings-Announcement Drift" refers to an anomaly in financ
  [2] title: Is Post-Earnings Announcement Drift a Thing? Again?
      href : https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-a
      body : It concludes that earnings drift remains a major market factor. Subrahmanyam s
  [3] title: Post earnings announcement drift: A simple earnings surprise measure ...
      href : https://www.sciencedirect.com/science/article/pii/S1057521924003922
      body : The "post-earnings-announcement drift" (PEAD), which describes the phenomenon
```

Real network results through the real `gpt_researcher` retriever class — the third
retriever leg (`run_memo.py:283` `RETRIEVER=semantic_scholar,arxiv,duckduckgo`) is
live again.

## 4. Immutable verification command (verbatim)

```
$ .venv/bin/python -c "from gpt_researcher.retrievers.duckduckgo.duckduckgo import Duckduckgo" && .venv/bin/python -m pytest backend/tests/test_phase_75_deps.py -q
16 passed, 1 warning in 2.39s
IMMUTABLE exit=0
```

(12 pre-existing + 4 new; derived, not asserted: `git show HEAD:backend/tests/
test_phase_75_deps.py | grep -c '^def test_'` = 12, worktree = 16. Regenerated after
the cycle-2 fourth guard was added — the earlier capture in this file read
`15 passed ... 2.45s` and was stale. Per §1 the first leg is NOT evidence of the fix;
the pytest leg carries the weight.)

## 5. Mutation matrix — criterion 3 (every guard killed)

| # | Mutation | Target guard | Result |
|---|----------|--------------|--------|
| N1 | manifest pin loosened `ddgs==9.14.4` → `ddgs>=0.0.1` | manifest pin test | `1 failed, 14 passed` → `test_autoresearch_requirements_manifest_pins_ddgs` **RED** |
| N2 | **ENV**: real `pip uninstall -y ddgs` (the criterion's "uninstall") | behavioral construct guard | `1 failed, 14 passed` → `test_ddg_retriever_constructs_with_real_ddgs_installed` **RED**; manifest test stayed GREEN (guards are independent, not co-firing) |
| N3 | **STUB**: the loud-failure guard's own rename-sim neutered (`return None` → `pass`) | loud-failure counterpart | `1 failed, 14 passed` → `test_ddg_retriever_fails_loud_when_ddgs_missing` **RED** (the negative test is load-bearing, not self-satisfying) |
| N4 | **anti-tautology**: construct guard vs. the PRE-install state | behavioral construct guard | §1 above — failed with the verbatim ImportError before the fix existed |
| N5 | manifest pin `9.14.4` → `9.13.0` while the venv still holds 9.14.4 | version-agreement guard (added in cycle-2) | `2 failed, 14 passed` → `test_installed_ddgs_version_matches_the_manifest_pin` **RED** (co-fires with the manifest guard — disclosed below) |

**Cycle-2 addition (from the Q/A CONDITIONAL).** Guards 1–3 share a vacuity: they all
pass on a venv holding *any* ddgs (e.g. 9.0.0) against a 9.14.4 pin, because
construction and the error string are version-independent. A fourth guard,
`test_installed_ddgs_version_matches_the_manifest_pin`, asserts
`importlib.metadata.version("ddgs") == the parsed pin`, so the pin means something at
runtime and not only on paper.

N5 turns **two** tests red, not one — the manifest guard and the version-agreement
guard both read the same pin line, so they co-fire when the *pin* is mutated.

**N6 (added cycle-3) settles whether that co-firing means the new guard is redundant.**
It does not. The isolating mutation is to patch the **installed** side and leave the pin
alone — the one direction only the new guard can see:

```
=== N6 ISOLATING mutation: patch the INSTALLED side, leave the pin alone ===
version-agreement guard: RED -> installed ddgs 9.0.0 != manifest pin 9.14.4 -- the venv and
                                the autoresearch manifest have diverged;
manifest guard:            GREEN (correct - pin file untouched)
```

So the guard is **independently load-bearing**, not a co-firing shadow of the manifest
test. Credit where due: the 76.9.3 Q/A ran this isolating mutation before I did, and it
corrects my cycle-2 disclosure, which had left the weaker impression that co-firing was
all N5 showed.

Cycle-1 matrix run (N1-N4), verbatim as executed when the suite held 15 tests:

```
=== BASELINE ===      15 passed, 1 warning in 2.51s
=== RESTORE ===       ['Successfully installed ddgs-9.14.4']
POST-RESTORE:         15 passed, 1 warning in 2.45s  reds=[]
SHA identical: True   (requirements-autoresearch.txt + test_phase_75_deps.py)
```

Left at its historical values on purpose — that run really did execute against 15
tests, and rewriting it to today's count would falsify the record. The cycle-2 re-run
below carries the current numbers and the N5 restore/SHA proof that cycle-1 could not
have contained.

Every mutation killed exactly its intended guard and nothing else; source files
byte-identical after revert; venv restored to the fixed state.

### Cycle-2/3 matrix re-run (current suite: 16 tests) + the restore/SHA proof cycle-1 lacked

```
=== CYCLE-2 BASELINE ===                      16 passed, 1 warning in 2.41s

=== N1 manifest pin loosened -> >=0.0.1 ===   2 failed, 14 passed
  reds: ['test_autoresearch_requirements_manifest_pins_ddgs',
         'test_installed_ddgs_version_matches_the_manifest_pin']
  KILLED test_autoresearch_requirements_manifest_pins_ddgs: True

=== N3 STUB: loud-guard rename-sim neutered ===  1 failed, 15 passed
  reds: ['test_ddg_retriever_fails_loud_when_ddgs_missing']
  KILLED test_ddg_retriever_fails_loud_when_ddgs_missing: True

=== N5 pin 9.14.4 -> 9.13.0 (venv still 9.14.4) ===  2 failed, 14 passed
  reds: ['test_autoresearch_requirements_manifest_pins_ddgs',
         'test_installed_ddgs_version_matches_the_manifest_pin']
  KILLED test_installed_ddgs_version_matches_the_manifest_pin: True

=== POST-REVERT ===                           16 passed, 1 warning in 2.40s  reds: []
SHA identical (requirements + test file): True
  requirements sha: 0988b78604d7edbd | test sha: 6b33ae164d944185
```

This block supplies what the 76.9.3 Q/A correctly flagged as missing: cycle-1's
restore/SHA proof covered only N1–N4 and predated N5, so no restore evidence existed
for the pin mutation. N2 (the real `pip uninstall`) is deliberately **not** re-run here
— it mutates the shared venv, and re-running it would add risk without adding
information over the cycle-1 result recorded above.
