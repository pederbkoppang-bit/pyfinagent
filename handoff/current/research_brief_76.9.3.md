# Research Brief — Step 76.9.3: ddgs pin for gpt-researcher 0.14.8 DDG retriever

Tier: simple. Audit-class: false. Started: 2026-07-24 (write-first skeleton).

## Search queries run (3-variant discipline)
1. Current-year: `ddgs duckduckgo_search renamed PyPI package 2026`
2. Last-2-year: `ddgs python package rate limit DDGS text backend 2025`
3. Year-less canonical: `duckduckgo_search python package renamed ddgs`

## Sources read in full (>=5 required; counts toward gate)
| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://pypi.org/project/duckduckgo-search/ | 2026-07-25 | official PyPI | WebFetch full | Latest 8.1.1 (2025-07-06, FROZEN): "This package (`duckduckgo_search`) has been renamed to `ddgs`! Use `pip install ddgs` instead." |
| https://github.com/deedy5/ddgs | 2026-07-25 | official README | WebFetch full | `from ddgs import DDGS; DDGS().text("...", max_results=5)`; `text(query, region="us-en", safesearch="moderate", timelimit=None, max_results=10, page=1, backend="auto")`; ctor `(proxy, timeout=5, verify=True)`; backends: bing, brave, duckduckgo, google, grokipedia, mojeek, startpage, yandex, yahoo, wikipedia; Python >=3.10 |
| https://pypi.org/pypi/ddgs/json | 2026-07-25 | official PyPI JSON API | curl + parse (full metadata, per gcloud-docs curl convention) | ddgs 9.14.4 (2026-05-15) latest of 43 releases; first release 9.0.0 = 2025-07-06 (rename day); requires-python >=3.10; deps: click>=8.1.8, primp>=1.2.3, lxml>=4.9.4, httpx[brotli,http2,socks]>=0.28.1, fake-useragent>=2.2.0 |
| https://github.com/deedy5/ddgs/issues/272 | 2026-07-25 | issue thread | WebFetch full | `backend='auto'` 202/403 Ratelimit reports on v7.1.1 (Jan 2025, PRE-metasearch era); closed without maintainer recipe; proxies did not help the reporter |
| https://raw.githubusercontent.com/assafelovic/gpt-researcher/master/requirements.txt | 2026-07-25 | upstream source | WebFetch full | upstream gpt-researcher now pins **`ddgs>=9.0.0`** (exactly once; no duckduckgo-search anywhere) — upstream's own compat spec for the identical retriever code |
| https://github.com/deedy5/ddgs/releases | 2026-07-25 | official changelog | WebFetch full | 9.11.4→9.14.4 (Mar–May 2026): "No explicit breaking changes... incremental improvements rather than structural API modifications to the DDGS class or core method signatures"; 9.14.4 = "DuckDuckGo engine update"; 9.14.1 disabled Google engine + primp bump |

## Identified but snippet-only (does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|---|---|---|
| https://pypi.org/project/ddgs/ | official PyPI | JS-walled ("A required part of this site couldn't load") — replaced by the PyPI JSON API fetch above |
| https://github.com/langchain-ai/langchain/issues/31892 | issue | rename-fallout corroboration only; DDG retriever here is gpt-researcher's, not langchain's |
| https://github.com/pewdiepie-archdaemon/odysseus/issues/3696 | issue | third-party migration ticket; confirms "duckduckgo-search is frozen" framing |
| https://github.com/open-webui/open-webui/discussions/6624 | discussion | 202 Ratelimit anecdotes, old package era |
| https://github.com/FoundationAgents/MetaGPT/issues/1567 | issue | rate-limit anecdote, wrapper-specific |
| https://github.com/langchain-ai/opengpts/issues/247 | issue | rate-limit anecdote, old package |
| https://grokipedia.com/page/ddgs_Python_library | tertiary | encyclopedia summary; primary sources already read |
| https://tessl.io/registry/tessl/pypi-duckduckgo-search/8.1.0 | mirror | registry mirror of PyPI data |
| https://cloudsmith.com/navigator/pypi/duckduckgo-search | mirror | registry mirror |
| https://iproyal.com/blog/duckduckgo-api/ | vendor blog | proxy-vendor marketing; low weight |
| https://aur.archlinux.org/packages/python-duckduckgo-search | distro pkg | not relevant to venv pip |
| https://pypi.org/project/AsyncDDGS/ | PyPI | unrelated fork |
| https://github.com/trose/ddgs-python | mirror repo | unofficial mirror of deedy5/ddgs |
| https://hermes-agent.nousresearch.com/docs/user-guide/skills/optional/research/research-duckduckgo-search | doc | confirms keyless ddgs use pattern only |
| https://raw.githubusercontent.com/assafelovic/gpt-researcher/v0.14.8/gpt_researcher/retrievers/duckduckgo/duckduckgo.py | upstream source | 404 (tag not named v0.14.8); installed site-packages file quoted below is ground truth anyway |

URLs collected (unique): 21 (6 read in full + 15 snippet/attempted).

## Recency scan (2024-2026)
Performed (queries #1 and #2 above are year-scoped). Findings, all inside the window: (a) `duckduckgo_search` FROZEN at 8.1.1 on 2025-07-06 and renamed to `ddgs` 9.0.0 the same day — the frozen package gets no further anti-bot fixes, so old installs silently degrade; (b) ddgs evolved from a DDG scraper into a METAsearch library ("Dux Distributed Global Search") rotating across ~10 backends with `backend="auto"` — this materially changes the 2024-era rate-limit picture (single-endpoint 202 Ratelimit storms) because auto-rotation spreads load; (c) active release cadence through 2026-05-15 (9.14.4, "DuckDuckGo engine update"); Google engine disabled in 9.14.1 (2026-04-20); (d) gpt-researcher upstream moved its own pin to `ddgs>=9.0.0` on master. No finding supersedes the step's premise; all confirm it.

## Key findings (external)
1. **Rename is total and dated**: `duckduckgo_search` last release 8.1.1 = 2025-07-06 with the verbatim banner "This package (`duckduckgo_search`) has been renamed to `ddgs`! Use `pip install ddgs` instead." (PyPI duckduckgo-search page). ddgs 9.0.0 shipped the same day (PyPI JSON API).
2. **The import surface gpt-researcher 0.14.8 needs exists in ALL ddgs 9.x**: `from ddgs import DDGS`, zero-arg `DDGS()`, `.text(query_positional, region=..., max_results=...)` (README + release notes: no structural API changes 9.0→9.14.4). Upstream gpt-researcher itself specifies `ddgs>=9.0.0` (master requirements.txt) for byte-identical retriever code.
3. **The 75.13 pin-set predates the rename by construction**: gpt-researcher 0.14.8's own dist METADATA still declares `Requires-Dist: duckduckgo-search>=4.1.1` (stale metadata; code imports `ddgs`) — that is exactly why pip considers the env satisfied while runtime is broken. Installing `ddgs` fixes runtime without disturbing the metadata dep.
4. **Python/dep compatibility is clean**: ddgs 9.14.4 requires Python >=3.10 (venv is 3.14 OK); click>=8.1.8 (have 8.3.1), lxml>=4.9.4 (have 6.0.2), httpx>=0.28.1 (have exactly 0.28.1). Deltas pip will make: NEW ddgs, fake-useragent, h2, socksio (httpx http2/socks extras — both absent today); UPGRADE primp 1.2.2 → >=1.2.3 (duckduckgo_search 8.1.1's `primp>=0.15.0` floor stays satisfied — no conflict possible; nothing else in the venv or any requirements file references ddgs/duckduckgo).
5. **Rate limiting in 2026**: unauthenticated nightly use remains viable; historical 202-Ratelimit storms (issue #272, v7.1.1 era) predate the metasearch rewrite; `backend="auto"` (the default — gpt-researcher passes no backend kwarg) rotates across ~10 engines. gpt-researcher's retriever already swallows per-call failures into an empty list (`except Exception ... search_response = []`), matching the tolerant sub-query fan-out. No retry/backoff work needed in this step.
6. **Alternatives (informational only, no scope creep)**: the retriever stack already has semantic_scholar + arxiv; ddgs successor candidates (SearXNG, paid SERP APIs) all require infra or keys — ddgs is the only keyless drop-in the gpt-researcher retriever class supports.

## Internal code inventory
| File | Lines | Role | Status |
|---|---|---|---|
| `scripts/autoresearch/requirements-autoresearch.txt` | 15-22 | 75.13 pin manifest (gpt-researcher==0.14.8, langchain-huggingface==1.2.1, sentence-transformers==5.5.1) | NO ddgs pin — the gap this step closes |
| `.venv/lib/python3.14/site-packages/gpt_researcher/retrievers/duckduckgo/duckduckgo.py` | 1-30 | Installed retriever (GROUND TRUTH) | imports `ddgs`, absent from venv → ctor raises |
| `.venv/lib/python3.14/site-packages/gpt_researcher/retrievers/utils.py` | 44-60 | `check_pkg()` — `importlib.util.find_spec(pkg)` else ImportError with the exact logged message | source of the verbatim error in the step |
| `scripts/autoresearch/run_memo.py` | 283 | `"RETRIEVER": "semantic_scholar,arxiv,duckduckgo"` (post-76.9 order, env_defaults setdefault) | duckduckgo slot silently dead |
| `scripts/autoresearch/run_memo.py` | 197-217 | `_gpt_researcher_guard()` — returns str\|None; main() prints to stderr + returns 1 (`:297-299`) | the 75.13 pattern to mirror |
| `backend/tests/test_phase_75_deps.py` | 206-212 | manifest pin test (`_parse_requirements` → assert `== 0.14.8`) | extend for ddgs pin |
| `backend/tests/test_phase_75_deps.py` | 218-286 | behavioral guard tests: monkeypatch `importlib.util.find_spec` → assert rc=1 + `_embedding_preflight` NOT reached; sanity counterpart asserts pass-through | the mutation-killable shape to copy |
| `.venv/lib/python3.14/site-packages/gpt_researcher-0.14.8.dist-info/METADATA` | Requires-Dist | `duckduckgo-search>=4.1.1` — metadata still requires the OLD name while code imports `ddgs` | root cause of the silent gap: pip resolver satisfied, runtime broken |
| `backend/tests/test_phase_76_9_launchd_fixes.py` | 127-151 | retriever-order test (source-scan of the RETRIEVER string) | unaffected by this step |

### Environment probes (2026-07-25, read-only)
- `gpt-researcher 0.14.8` installed; `__version__` attr = n/a.
- `pip show ddgs` → **Package(s) not found** (the defect).
- `duckduckgo_search 8.1.1` IS installed (Requires: click, lxml, primp; Required-by: gpt-researcher) — satisfies the stale `duckduckgo-search>=4.1.1` metadata dep but is NOT what the code imports.
- `python -c "from gpt_researcher.retrievers.duckduckgo.duckduckgo import Duckduckgo"` → **import SUCCEEDS** (only warnings + unrelated `Failed to import MCPRetriever: No module named 'langchain_mcp_adapters'`). The failure is at **construction time**: `check_pkg('ddgs')` runs inside `__init__`, not at module import. ⇒ a pure module-import guard would be a guard that cannot fail; the guard MUST exercise `find_spec("ddgs")`/instantiation.
- Resolver-conflict check: only gpt-researcher requires `duckduckgo_search`; nothing in the venv or any requirements file pins `ddgs` or `duckduckgo` (grep of `requirements*.txt`, `scripts/autoresearch/*.txt`, `backend/requirements*.txt` = zero hits). Installing `ddgs` adds a NEW distribution, conflicts impossible from the repo side.

## Verbatim installed-retriever import/call quote
`.venv/lib/python3.14/site-packages/gpt_researcher/retrievers/duckduckgo/duckduckgo.py` (entire file, installed 0.14.8):

```python
from itertools import islice
from ..utils import check_pkg


class Duckduckgo:
    """
    Duckduckgo API Retriever
    """
    def __init__(self, query, query_domains=None):
        check_pkg('ddgs')
        from ddgs import DDGS
        self.ddg = DDGS()
        self.query = query
        self.query_domains = query_domains or None

    def search(self, max_results=5):
        """
        Performs the search
        :param query:
        :param max_results:
        :return:
        """
        # TODO: Add support for query domains
        try:
            search_response = self.ddg.text(self.query, region='wt-wt', max_results=max_results)
        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            search_response = []
        return search_response
```

Required import surface from `ddgs`: `from ddgs import DDGS`; `DDGS()` zero-arg ctor; `.text(query, region=..., max_results=...)` with positional query + those two kwargs. `check_pkg` (`utils.py:44-60`) produces the exact logged error: `Unable to import ddgs. Please install with `pip install -U ddgs``.

## Recommended pin
**`ddgs==9.14.4`** appended to `scripts/autoresearch/requirements-autoresearch.txt`, following the file's exact-pin + consumer-anchor comment convention (`:15` shape), e.g.:

```
ddgs==9.14.4                  # gpt_researcher/retrievers/duckduckgo/duckduckgo.py -- `from ddgs import DDGS` (0.14.8 code imports ddgs; its dist METADATA still declares the FROZEN duckduckgo-search>=4.1.1, so pip is satisfied while runtime breaks -- phase-76.9.3)
```

Why 9.14.4 exactly: (a) latest stable (2026-05-15) and that release is itself a "DuckDuckGo engine update" — anti-bot freshness is the package's whole value; (b) satisfies upstream's own `ddgs>=9.0.0` spec for identical retriever code; (c) required import surface (`from ddgs import DDGS`, `DDGS()`, `.text(q, region=, max_results=)`) is unchanged across all of 9.x per release notes; (d) manifest convention is `==` pins (requirements-autoresearch.txt:12-13 "Pinned at the versions measured installed"). Install with a targeted `pip install ddgs==9.14.4` (or the -r file re-run WITH the standing langchain-core constraints discipline — see Risks).

Expected env delta on install: +ddgs, +fake-useragent(>=2.2.0), +h2, +socksio (httpx http2/socks extras); primp 1.2.2 → >=1.2.3. No removals, no conflicts (`duckduckgo_search` 8.1.1 stays, floors all still met).

## Recommended guard-test shape + mutation
Boundary is `requirements-autoresearch.txt + install + tests` — run_memo.py is NOT in scope, so the guard is pytest-side, mirroring `backend/tests/test_phase_75_deps.py` (manifest test :206-212, behavioral tests :218-286).

**Critical design fact (measured 2026-07-25)**: `from gpt_researcher.retrievers.duckduckgo.duckduckgo import Duckduckgo` **succeeds today with ddgs absent** — `check_pkg('ddgs')` runs inside `__init__`, not at module import. A module-import-only guard is a guard that cannot fail (feedback_mutation_test_guards_and_fixtures). The guard MUST construct.

1. **Manifest pin test** (mirror :206-212): parse `requirements-autoresearch.txt` via the existing `_parse_requirements`, assert `("==", "9.14.4")` for `ddgs`. Mutation: delete/loosen the pin line → red.
2. **Behavioral construct guard** (the criterion-3 guard): `Duckduckgo("guard probe query")` constructs and `hasattr(r.ddg, "text")` — exercises the REAL `check_pkg('ddgs')` + `from ddgs import DDGS` + `DDGS()` chain, no network (DDGS ctor builds a client; requests fire only on `.text()`). Mutation that turns it red: `pip uninstall ddgs` OR rename-sim (monkeypatch `importlib.util.find_spec` to return None for `'ddgs'` — check_pkg calls `importlib.util.find_spec` at call time, so the patch bites) → ImportError → red.
3. **Loud-failure counterpart** (mirror the :228-251 negative shape): with find_spec('ddgs') monkeypatched to None, assert `pytest.raises(ImportError, match="pip install -U ddgs")` on construction — proves the failure mode is the LOUD verbatim message seen in the live logs, not a silent empty result.

Live proof (criterion 2) is GENERATE's job, not the pytest guard: one sub-query-shaped call `Duckduckgo("<query>").search(max_results=3)` recorded verbatim (network, run once, goes in experiment_results/live_check — keep the pytest guard offline-deterministic).

## Risks / unknowns
- **`region='wt-wt'`**: installed retriever passes the OLD worldwide region code; ddgs 9.x default is `us-en`. Upstream ships this exact combination against `ddgs>=9.0.0`, so it is upstream-vetted; worst case is region fallback, not an exception. Low.
- **primp upgrade side-effect**: duckduckgo_search 8.1.1 (staying in the venv as a stale metadata dep) keeps working with newer primp (`>=0.15.0` floor); nothing else imports it. Low.
- **Full `-r` re-install risk**: re-running the whole requirements file without the standing constraints discipline can silently bump langchain-core (documented incident, cron-maintenance memory; requirements-autoresearch.txt itself carries no langchain-core pin) — prefer targeted `pip install ddgs==9.14.4`.
- **DDGS() ctor network assumption**: guard assumes construction is offline (README pattern constructs eagerly, queries on `.text()`); if a future ddgs version changes this, the guard test gains a network dependency — note in the test docstring.
- **Rate limiting**: ddgs remains unauthenticated scraping; a nightly batch may still see per-backend throttles. `backend="auto"` rotation + the retriever's swallow-to-empty behavior bound the blast radius; no in-step mitigation needed.
- **Upstream tag 404**: gpt-researcher's GitHub tag for 0.14.8 is not `v0.14.8` (raw fetch 404'd); the installed site-packages file was used as ground truth instead — stronger evidence anyway.

## Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL (6: 5 WebFetch + 1 PyPI JSON API via curl per the established curl-counts convention)
- [x] 10+ unique URLs total (21 incl. snippet-only)
- [x] Recency scan (last 2 years) performed + reported (rename 2025-07-06; releases through 2026-05-15)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] 3-variant search discipline visible (queries listed)
- [x] Internal exploration covered every relevant module (manifest, installed retriever+utils, run_memo, both test homes, dist METADATA, env probes)

## JSON envelope
```json
{"tier":"simple","external_sources_read_in_full":6,"snippet_only_sources":15,"urls_collected":21,"recency_scan_performed":true,"internal_files_inspected":9,"gate_passed":true}
```
