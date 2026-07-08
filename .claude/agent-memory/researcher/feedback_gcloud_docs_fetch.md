---
name: gcloud-docs-webfetch-nav-only
description: WebFetch on cloud.google.com / docs.cloud.google.com reference pages returns nav-only; use curl + tag-strip extraction
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
