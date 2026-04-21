# Virtual-Fund Postmortem (phase-8.5.9 seed doc)

**Date:** 2026-04-20
**Status:** Initial seed; to be updated as paper-trading incidents accumulate.

---

## Purpose

This document enumerates **known failure buckets** from pyfinagent's virtual-fund paper
trading (phase-3.7, phase-4.5, phase-4.8 shadow runs). phase-8.5.9 uses it to prioritize
new autoresearch trials toward fixing these buckets **before** exploring novel search
directions.

## Failure Bucket 1 -- Plateau after parameter saturation
- 27+ harness cycles ended with `CONDITIONAL -- kept with warning`
  because all parameters hit `SATURATED` status.
- **Seed target for 8.5.9:** candidates that alter feature bundles
  (not just params). Bundle `mda_plus_ensemble_blend` is the first try.

## Failure Bucket 2 -- Live CDN / API reliability
- phase-7.3 FINRA CDN 403s; phase-7.1 S3 bucket drift.
- **Seed target:** candidates that are source-agnostic -- rely only on
  phase-1 OHLCV + phase-6 news + phase-7.2 13F (most reliable feeds).

## Failure Bucket 3 -- Zero-shot transformer underperformance
- phase-8.4 REJECT: zero-shot TimesFM/Chronos underperform AR(1).
- **Seed target:** candidates using `ensemble_blend` only when
  `transformer_shadow` IC has been measured > 0.05 over 60+ days.
  Until that data exists, stick to MDA + alt-data bundles.

## Failure Bucket 4 -- Single-chamber coverage (congress)
- phase-7.1 Senate-only; House deferred.
- **Seed target:** candidates weighted so Congress features contribute
  at most 20% of signal until both chambers are ingested.

## Novel-search Secondary Buckets
After the above four failure buckets have each been seeded with at least one
candidate, the proposer explores novel search (new param combinations not
previously tested). This is the "secondary" priority.

---

## How 8.5.9 consumes this doc

`scripts/harness/autoresearch_seed_from_postmortem.py` parses each `## Failure Bucket N`
heading + its sub-bullets and emits a seed candidate dict per bucket. The seed list is
written to the proposer's queue before any novel-search candidate.
