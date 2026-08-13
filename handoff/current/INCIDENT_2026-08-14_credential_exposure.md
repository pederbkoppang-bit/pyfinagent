# INCIDENT — Anthropic OAuth token published to a public remote

**Opened:** 2026-08-14 ~10:30 CEST
**Severity:** HIGH — a live credential on a public GitHub remote for six days
**Status:** **OPEN. Operator action required. NOTHING has been remediated.**
**Relates to:** operator ask **06-2** (credential rotation), masterplan step **86.67**

> **NO TOKEN VALUE APPEARS IN THIS FILE.** Only the `sk-ant-oat01` prefix, which
> establishes the shape without disclosing the secret. Do not paste the value into any
> artifact, commit message, or chat when remediating.

---

## 1. What is exposed

**Five tracked files**, each containing one `sk-ant-oat01…` Anthropic OAuth token in the
**`result`** field — an Authorization header echoed into session telemetry.

| File | First commit | Date |
|---|---|---|
| `handoff/away_ops/session_pm_20260808T200008Z.json` | `8aa3f52e` | 2026-08-08 |
| `handoff/away_ops/session_am_20260809T053008Z.json` | `5d0e462c` | 2026-08-09 |
| `handoff/away_ops/session_pm_20260809T200008Z.json` | `6763f10f` | 2026-08-09 |
| `handoff/away_ops/session_am_20260810T053009Z.json` | `cad38647` | 2026-08-10 |
| `handoff/away_ops/session_pm_20260810T200010Z.json` | `630fa95b` | 2026-08-11 |

**All five are tracked, committed, and pushed to `origin/main`** at
`https://github.com/pederbkoppang-bit/pyfinagent.git`. **Public since 2026-08-08.**

## 2. The leak is CLOSED, not ongoing

Re-scanned all **76** `handoff/away_ops/session_*.json` with a corrected pattern:

```
files WITH a token   :  5   dates 2026-08-08 .. 2026-08-10
files WITHOUT        : 71   dates 2026-06-12 .. 2026-08-13
newest file overall  : 2026-08-13   (clean)
clean files written AFTER the last leak: 6  (08-11 x2, 08-12 x2, 08-13 x2)
```

**Six consecutive clean files after 2026-08-10** indicate the producer stopped emitting
the token. **This narrows the incident to a bounded 3-day window — it does not reduce
the urgency of rotation**, because the token was public for six days regardless.

**Not verified:** *why* it stopped. I did not find the producer change that fixed it, so
"fixed" is inferred from six clean outputs, not demonstrated. Treat as unconfirmed.

## 3. Why my earlier scan reported CLEAN — the failure is instructive

At ~05:40 today I scanned these files and reported **"zero credential-shaped values,
positive control passed."** That was a **false negative**.

```
MY regex        sk-[A-Za-z0-9]{20,}      vs  sk-ant-oat01-…   ->  0 matches
CORRECT regex   sk-ant-[A-Za-z0-9_-]{10,} vs  sk-ant-oat01-…   ->  1 match
```

`[A-Za-z0-9]` **cannot cross a hyphen**, and an Anthropic OAuth token is hyphenated
throughout.

**The deeper failure is the control, not the regex.** My "positive control" was a
synthetic `{"api_key":"sk-abcdefghijklmnopqrstuvwxyz123456"}` — **hyphen-free**. So it
exercised exactly the case my regex could already handle and returned 1, and I read that
as proof the probe was sound.

**I built a control that could only ever agree with me.**

Every other control I ran during this session was derived from the **real artifact's
shape** and several of them caught me. This one I invented from my own pattern, and it
confirmed my own pattern. That is the distinction:

> **A control built from the artifact tests the probe. A control built from your own
> pattern tests nothing.**

## 4. What has NOT been done, deliberately

- **No token rotated** — operator credential action.
- **No git history rewritten, no file removed, no `.gitignore` edited.** 86.67
  criterion 4 forbids rewriting published history; it is operator-gated under ask 06-2.
- **No force-push, no remote change.**

I stopped at measurement because remediation here is destructive and outward-facing.

## 5. Recommended order

1. **ROTATE THE TOKEN FIRST.** It has been public for six days. Removing it from history
   does **not** un-publish it — GitHub retains unreachable objects, forks and clones keep
   copies, and scrapers index public repos continuously. **Rotation is the only action
   that actually revokes the exposure.**
2. **Then** decide history remediation under ask 06-2, knowing it is cosmetic relative to
   step 1 and carries its own risk on a repo with a concurrent session pushing to `main`.
3. **Confirm the producer fix.** Six clean files is evidence, not proof. Find the change
   that stopped the echo, or add a guard so a token cannot reach a telemetry `result`
   field again. **`.gitignore` alone would not have prevented this** — these files were
   *tracked*, so an ignore rule would not have applied.
4. **Re-scan with the corrected pattern across the whole repo**, not just `away_ops` — my
   broken regex may have produced false negatives anywhere it was used today.

## 6. Provenance

Surfaced by the **86.67 research gate** (`wf_b9a52054-ddd`), which flagged
*"five TRACKED files, all on origin/main, carry an `sk-ant-oat01` token … the given
`sk-<20+>` regex cannot cross hyphens."* I then verified it independently: the regex
comparison above, the tracked status of all five files, the commit SHAs, and the
containing JSON key.

**The gate returned `gate_passed: false`** on an unrelated 38-vs-37 URL over-claim. The
security finding is in its summary and is verified regardless of that verdict — a failed
gate does not make its observations false, and this one is the most important thing the
session produced.

---

## 7. ADDENDA (2026-08-14 ~12:10) — peer session `95794`, each item re-verified by me

### 7a. It is ONE token, not five — and the repo is PUBLIC WITH A FORK

```
gh repo view  ->  visibility: PUBLIC   isPrivate: false   forkCount: 1   stargazerCount: 2
distinct credentials across the 5 files: 1   (sha256-identical match, length 92)
```

**One credential, exposed five times.** One revoke closes all five.

**THE FORK CHANGES THE REMEDIATION CALCULUS.** A fork is a **separate GitHub-hosted
copy**. History rewriting on `origin` **does not reach it**. Combined with GitHub
retaining unreachable objects and any clones already taken, this makes §5's ordering
stronger, not weaker: **rotation is not merely the priority — it is the only action that
actually revokes the credential.** History remediation is cosmetic by comparison.

*Honest note on the digest:* the peer reported `916320a93a02…`, I computed
`32fd30514637…`. **The digests differ because we hashed different substrings**, not
because we disagree — both independently conclude **one distinct credential**. Recorded
rather than smoothed, because two people reporting "the same hash" when they hashed
different things is exactly how a false agreement gets into a record.

### 7b. The window is bounded on BOTH sides — I had only established the trailing edge

```
session_pm_20260807T200011Z   clean
session_am_20260808T053009Z   clean   <- LAST CLEAN BEFORE
session_pm_20260808T200008Z   LEAK    <- FIRST LEAK
  … exactly 5 consecutive runs …
session_pm_20260810T200010Z   LEAK    <- LAST LEAK
session_am_20260811T053009Z   clean   <- FIRST CLEAN AFTER
  … 6 consecutive clean, through 2026-08-13 PM
```

Across all 77 session files back to June. **This converts the producer hunt from
"closed but unexplained" into two ~14-hour brackets to diff:**

- something changed between **2026-08-08 05:30Z and 08-08 20:00Z**
- and reverted between **2026-08-10 20:00Z and 08-11 05:30Z**

That is a far better starting point than §2's open-ended "why it stopped is unverified",
and it supersedes that framing.

### 7c. My broken regex did NOT hide exposure elsewhere — that worry is CLOSED

The peer re-scanned the **entire tracked tree** at `origin/main` with
`sk-ant-[A-Za-z0-9_-]{10,}`, not just `away_ops`: **18 files match the vendor prefix;
only these 5 are real.** The other 13 are placeholders (FAKE/TEST markers in the value)
or 17–43-character stubs.

§5 item 4 asked for exactly this re-scan. **It is done, by an independent session, and
it came back negative.** The residual risk from my false-negative pattern is therefore
bounded to what is already recorded here.

### 7d. No fresh exposure is pending

The **3 newest session files are untracked and clean**, so a `git add -A` by either
session — and both our hooks stage tree-wide — will not publish a new token.
