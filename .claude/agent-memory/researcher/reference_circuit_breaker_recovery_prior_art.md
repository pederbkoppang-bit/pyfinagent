---
name: circuit-breaker-recovery-prior-art
description: The canonical external answer for "a safety latch that cannot clear itself" — resilience4j's automaticTransitionFromOpenToHalfOpenEnabled, Azure's manual-reset + health-check-probe guidance, and the date-keyed anchor rule
metadata:
  type: reference
---

Prior art to reach for whenever a pyfinagent safety gate latches and its clearing
condition sits behind the thing that failed.

- **resilience4j CircuitBreaker docs** —
  https://resilience4j.readme.io/docs/circuitbreaker — names the bug as a shipped
  DEFAULT: `automaticTransitionFromOpenToHalfOpenEnabled=false` ⇒ *"the transition
  to HALF_OPEN only happens if a call is made, even after `waitDurationInOpenState`
  is passed."* Setting it true creates *"a thread ... to monitor all the instances
  of CircuitBreakers"*. This is the citation for "drive recovery from a scheduler,
  not from the protected path". Also documents `DISABLED` / `FORCED_OPEN` /
  `METRICS_ONLY` as first-class states.
- **Azure Architecture Center, Circuit Breaker pattern** —
  https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker —
  three quotes that keep earning their keep: *"Failed operations testing"* (probe
  with a special health-check operation rather than re-invoking the failed
  operation); *"Manual override: ... you should provide a manual reset option that
  enables an administrator to close a circuit breaker"*; *"Recoverability: ... if
  the circuit breaker remains in the Open state for a long period, it can raise
  exceptions even if the reason for the failure is resolved."*
- **Google SRE, Addressing Cascading Failures** —
  https://sre.google/sre-book/addressing-cascading-failures/ — *"the code path you
  never use is the code path that (often) doesn't work"* (the argument against
  trusting an unexercised "Verified:" comment), and *"Stop Health Check
  Failures"* when health-checking itself makes the service unhealthy.
- **Daily-loss anchors must be DATE-keyed, never event-keyed** — MQL5 blog
  2026-08-02, https://www.mql5.com/en/blogs/post/773545 : re-anchoring on an init
  event means *"From every anchor's point of view the limit was never breached."*
  Fix = key on account+date so the anchor is self-expiring. This is the decisive
  argument against "re-anchor on backend startup".
- **sec.gov 403s WebFetch.** Use `curl -sL -A "<name> <email>" <url>` + a
  regex tag-strip; that counts as read-in-full. The Small Entity Compliance Guide
  (`.../34-63241-secg.htm`) is HTML and carries the load-bearing 15c3-5 language
  ("prevent the entry of orders unless there has been compliance ... on a
  pre-order entry basis"; "direct and exclusive control"). RTS 6 Art. 12/14 text
  is fetchable from legislation.gov.uk; the FCA Handbook URL returns the landing
  page instead of the article.
