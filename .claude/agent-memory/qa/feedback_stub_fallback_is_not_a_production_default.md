---
name: stub-fallback-is-not-a-production-default
description: A getattr fallback that fires only because a test stub omits the attribute gets written up as "the production default" -- read settings.py, not the test comment
metadata:
  type: feedback
---

When a test comment or commit message explains why a setting is passed
explicitly, verify the stated default against the real `Settings` model, not
against the value the code falls back to. `getattr(settings, "X", 0)` returns 0
only because a `SimpleNamespace` stub OMITS `X`; the production default can be
anything.

**Why:** 86.74 cycle-3 documented `paper_swap_max_per_cycle` as "defaults to 0
and short-circuits the whole function". Production defaults it to **2**
(`backend/config/settings.py:378`) with `paper_swap_enabled` **True** (`:368`) --
the swap path is LIVE by default. The 0 was the fallback at
`portfolio_manager.py:719` for an attribute the test's `_settings()` never sets.
The error ran in the dangerous direction: a reader concludes a money path is dark
when it is live. The same check also answered the over-configuration question in
the author's favour -- all five kwargs matched production exactly, which made the
harm production-representative rather than hypothetical.

**How to apply:** whenever a harness sets flags explicitly, diff that kwarg list
against the `Field(...)` defaults. It costs one `sed -n` and decides two things
at once: whether the scenario is production-representative, and whether the
stated provenance is real. Related: [[rederive-the-label-not-just-the-number]],
[[count-the-class-not-your-list]].
