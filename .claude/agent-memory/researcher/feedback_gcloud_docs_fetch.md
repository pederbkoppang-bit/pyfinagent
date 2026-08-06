---
name: gcloud-docs-webfetch-nav-only
description: WebFetch on JS-rendered vendor doc sites (cloud.google.com, docs.slack.dev concept pages) returns nav/homepage only; use curl+tag-strip, or read the vendor's own tests/installed source instead
metadata:
  type: feedback
---

WebFetch on Google Cloud documentation pages (cloud.google.com and its
docs.cloud.google.com redirect target) returns only the navigation tree —
the article body is JS-rendered and absent from the markdown conversion.
Observed 2026-07-08 on three BigQuery standard-SQL reference pages
(conversion_rules, aggregate_functions, conversion_functions), including
after retries.

**Why:** the research gate counts only sources read IN FULL; a nav-only
fetch silently fails the floor while looking like a completed call.
**How to apply:** for any cloud.google.com reference page, go straight to
`curl -sL <url>` + Python tag-strip extraction (re.sub scripts/styles/tags,
html.unescape) in the scratchpad — full text comes through cleanly and
still counts as read-in-full. Note "curl + text extraction" in the
Fetched-how column.

**Same class, other vendors (2026-08-06, step 82.59):** `docs.slack.dev`
*concept* pages (`/tools/bolt-python/concepts/...`) return the docs
HOMEPAGE, twice, including after following the 301 from `tools.slack.dev`.
Their `/reference/events/...` pages fetch fine, so the failure is
per-section, not per-host — don't conclude "the host is fine" from one
good fetch.
**Better substitute than curl for an SDK question:** read the vendor's OWN
tests from `raw.githubusercontent.com` plus the INSTALLED package under
`.venv/` — both are stronger evidence than prose docs (they're what
actually executes), and the installed source settles version questions the
docs can't. Find the vendor test's real path with
`gh api "search/code?q=<symbol>+repo:<org>/<repo>+extension:py"` rather
than guessing filenames (guessing cost a 404 on this step).
