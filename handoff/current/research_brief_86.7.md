# Research Brief -- step 86.7

**Topic:** Headless / unattended authentication for CLI tools that store credentials in an
OS keychain -- launchd/systemd background jobs obtaining credentials without an interactive
session; macOS Keychain reachability from a non-GUI launchd context (login-keychain lock
state, `SecKeychain` ACLs, partition lists, LaunchAgents vs LaunchDaemons); long-lived token
alternatives and their revocation/expiry risk; OAuth device-authorization + refresh-token
handling for CLI clients; and the reliability principle that an unattended job must fail
LOUDLY rather than degrade silently when authentication is unavailable.

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Started:** 2026-08-11. **Researcher:** Layer-3 researcher via the Workflow rail.

---

## STATUS ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 12,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 4,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "gate_passed": true
}
```

*This envelope was written born-inert before any source was read and flipped to COMPLETE as the
final act. `coverage.dry` is false and does NOT gate: the caller declared this step NOT audit-class,
so `coverage` is informational only.*

---

## Read in full (10; >=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding (detail in the Source log below) |
|---|-----|----------|------|-------------|-------------|
| S1 | https://datatracker.ietf.org/doc/html/rfc8628 | 2026-08-11 | IETF standard | WebFetch, full | Only `authorization_pending`/`slow_down` may be retried; any other error -> "MUST stop polling". Device grant is a human-in-the-loop BOOTSTRAP, not no-human auth. |
| S2 | https://datatracker.ietf.org/doc/html/rfc9700 | 2026-08-11 | IETF BCP (Jan 2025) | WebFetch, full | "Refresh tokens for public clients MUST be sender-constrained or use refresh token rotation." A CLI is a public client. |
| S3 | https://developer.apple.com/library/archive/technotes/tn2083/_index.html | 2026-08-11 | Apple official | WebFetch, full | Agent = "on behalf of a particular user", can "reliably access the user's home directory"; a daemon asking for user state "when the user is not logged in ... will fail". Says nothing about keychains (honest negative). |
| S4 | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-11 | Google SRE Book | WebFetch, full | "Every page should be actionable"; names "an HTTP 200 success response, but coupled with the wrong content" as a failure mode. |
| S5 | https://systemd.io/CREDENTIALS/ | 2026-08-11 | systemd official | WebFetch, full | Env vars "are inherited down the process tree"; credentials instead get a kernel-enforced access check per access, in non-swappable memory. |
| S6 | https://code.claude.com/docs/en/authentication | 2026-08-11 | Anthropic official | WebFetch, full | 6-slot precedence ladder (slot 5 `CLAUDE_CODE_OAUTH_TOKEN` beats slot 6 keychain); macOS creds "stored in the encrypted macOS Keychain"; helper failures went from "ten silent retries" to a named error in three attempts (v2.1.208). |
| S7 | https://keith.github.io/xcode-man-pages/security.1.html | 2026-08-11 | Apple man page | WebFetch, full | Partition list = "an extra parameter in the ACL which limits access to the key based on an application's code signature"; "-p ... is insecure"; unlock "might succeed when an incorrect password is presented". |
| S8 | https://sre.google/workbook/alerting-on-slos/ | 2026-08-11 | Google SRE Workbook | WebFetch, full | Recall = "100% if every significant event results in an alert" -- the attribute a one-shot page latch fails. Omits alerting on disappearing signals (honest gap). |
| S9 | https://docs.github.com/en/actions/how-tos/deploy/deploy-to-third-party-platforms/sign-xcode-applications | 2026-08-11 | GitHub official | WebFetch, full | The industry pattern for headless macOS keychain use is a dedicated ephemeral keychain + explicit unlock + explicit partition list, never the login keychain. |
| S10 | https://learn.microsoft.com/en-us/entra/identity-platform/refresh-tokens | 2026-08-11 | Microsoft official (2025-11-05 / upd. 2026-06-15) | WebFetch, full | Refresh tokens "can be revoked by the sign-in service at any time before their expiration"; per-class revocation matrix; rotation on every use. |

*(The snippet-only / collected-not-fetched table and the attempted-but-failed table are near the
end of this file, to keep each URL recorded exactly once.)*

---
## Source log (running; appended as each source lands)

### S1 -- RFC 8628, OAuth 2.0 Device Authorization Grant (IETF standards track)
URL: https://datatracker.ietf.org/doc/html/rfc8628 -- accessed 2026-08-11 -- kind: standard/spec -- WebFetch, full page.

- Scope statement: the grant "is designed for Internet-connected devices that either lack a
  browser to perform a user-agent-based authorization or are input constrained to the extent
  that requiring the user to input text in order to authenticate during the authorization flow
  is impractical."
- Flow: device POSTs to a **device authorization endpoint** with `client_id` (+`scope`); server
  returns `device_code`, `user_code`, `verification_uri`, optional `verification_uri_complete`,
  `expires_in`, `interval` (default **5s** when omitted). Device polls the token endpoint with
  `grant_type=urn:ietf:params:oauth:grant-type:device_code`.
- Polling contract (this is the load-bearing part for an unattended job): only two error codes
  mean "keep polling" -- `authorization_pending` and `slow_down` (which "MUST be increased by 5
  seconds for this and all subsequent requests"). Everything else is terminal: "If the client
  receives an error response with any other error code, it **MUST stop polling** and SHOULD react
  accordingly, for example, by displaying an error to the user." `access_denied` and
  `expired_token` are terminal; on `expired_token` the client "MAY commence a new device
  authorization request but SHOULD wait for user interaction before restarting to avoid
  unnecessary polling."
- Connection timeouts: "clients MUST unilaterally reduce their polling frequency before retrying",
  exponential backoff RECOMMENDED.
- **Refresh tokens are NOT specified here.** The response diagram shows "(& Optional Refresh
  Token)" but the RFC states no requirements for refresh-token issuance or use -- that is left to
  RFC 6749 / the AS. (Recorded because it is exactly the gap that bites a headless client: the
  device grant gets you a first token, not a renewal strategy.)
- Headless-ness is explicit: "There is no requirement that the user code be displayed by the
  device visually. Other methods of one-way communication can potentially be used" -- the device
  itself never needs a browser or an interactive TTY; a **human on a second device** completes the
  flow. Device-grant is therefore a BOOTSTRAP mechanism requiring a human once, not a way to
  authenticate with no human ever.
- Security: device clients "are generally incapable of maintaining the confidentiality of their
  credentials ... they should be treated as public clients"; device_code needs "a very high
  entropy code"; user_code needs server-side rate limiting; remote-phishing mitigation is to
  "inform the user that they are authorizing a device ... and to confirm that the device is in
  their possession."

### S2 -- RFC 9700, Best Current Practice for OAuth 2.0 Security (IETF BCP, Jan 2025)
URL: https://datatracker.ietf.org/doc/html/rfc9700 -- accessed 2026-08-11 -- kind: standard/BCP -- WebFetch, full page.
**This is a last-2-year source (published January 2025) -- it anchors the recency scan.**

- Normative core (S2.2.2): "**Refresh tokens for public clients MUST be sender-constrained or use
  refresh token rotation.**" A CLI / native app is a **public client**, so this applies directly.
- Rotation semantics (S4.14): "refresh token rotation ensures that a new refresh token is issued
  with every token response" and "the previous refresh token is invalidated after a certain time
  window"; the AS SHOULD detect and reject **replay** of a refresh token as an indicator of
  compromise.
- Sender-constraining alternative: mTLS (RFC 8705) or DPoP (RFC 9449) binding.
- No fixed expiry period is mandated -- lifetimes are a deployment decision, but revocation
  capability is expected.
- **Consequence for an unattended job:** under rotation, the stored credential is *mutable state*.
  Two processes sharing one refresh token will race, and the loser's token is invalidated ->
  looks exactly like a revocation. A single-writer discipline around the credential store is a
  protocol requirement, not a nicety.

### S3 -- Apple Technical Note TN2083, "Daemons and Agents" (Apple official, year-less canonical)
URL: https://developer.apple.com/library/archive/technotes/tn2083/_index.html -- accessed 2026-08-11 -- kind: official vendor doc -- WebFetch, full page.

- Definitions: "A **daemon** is a program that runs in the background as part of the overall system
  (that is, it is not tied to a particular user). A daemon cannot display any GUI; more
  specifically, it is not allowed to connect to the window server." / "An **agent** is a process
  that runs in the background on behalf of a particular user. Agents are useful because they can
  do things that daemons can't, like reliably access the user's home directory or connect to the
  window server."
- Bootstrap/session context: a launchd **daemon** "Uses the global bootstrap namespace unless the
  `SessionCreate` property is specified"; launchd **agents** run in per-session (GUI or non-GUI)
  or per-user bootstrap namespaces.
- The decisive limitation for unattended credential access: "It is not possible for a daemon to act
  on behalf of a user with 100% fidelity ... it is not possible to mount that volume without the
  user's security credentials (typically their password). So, if a daemon tries to get a user
  preference when the user is not logged in, it will fail."
- **HONEST NEGATIVE RESULT:** TN2083 says *nothing* about the keychain -- no mention of the login
  keychain, System keychain, unlocking, or keychain reachability from a daemon. The
  daemon-vs-agent distinction it establishes is necessary background but is NOT itself the
  keychain citation; that is sourced separately below.
- Also: "If your process connects to the window server, it will not survive a normal logout."

### S4 -- Google SRE Book, "Monitoring Distributed Systems" (Beyer et al., O'Reilly/Google, year-less canonical)
URL: https://sre.google/sre-book/monitoring-distributed-systems/ -- accessed 2026-08-11 -- kind: industry/engineering book (official Google SRE) -- WebFetch, full chapter.

- Definitions: white-box = "Monitoring based on metrics exposed by the internals of the system";
  black-box = "Testing externally visible behavior as a user would see it"; an alert is "A
  notification intended to be read by a human".
- Four golden signals: **latency, traffic, errors, saturation**. Errors = "The rate of requests
  that fail, either explicitly (e.g., HTTP 500s), implicitly".
- Paging philosophy: "**Every page should be actionable**" and "**Every page response should
  require intelligence. If a page merely merits a robotic response, it shouldn't be a page.**"
  Symptom bias: "It's better to spend much more effort on catching symptoms than causes; when it
  comes to causes, only worry about very definite, very imminent causes."
- **Silent-degradation is called out by name:** the chapter names "an HTTP 200 success response,
  but coupled with the wrong content" as a failure mode -- i.e. an implicit error that a naive
  exit-code / status-code check will score as success. This is the literature basis for "an
  unattended job that produces degraded output is an OUTAGE, not a success."
- **HONEST GAP:** this chapter does NOT explicitly cover alerting on the *absence* of a signal or
  on a job that stops running. That needs a separate source (see S7).

### S5 -- systemd Credentials (systemd.io official documentation, year-less canonical)
URL: https://systemd.io/CREDENTIALS/ -- accessed 2026-08-11 -- kind: official project doc -- WebFetch, full page.
*(The systemd half of the caller's question -- the cross-platform prior art for how an init system
should hand a secret to an unattended service.)*

- Mechanisms: `LoadCredential=`, `SetCredential=`, `LoadCredentialEncrypted=`,
  `SetCredentialEncrypted=`, `ImportCredential=`. The service reads them from a directory named by
  `$CREDENTIALS_DIRECTORY` (`/run/credentials/<unit>` for system services -- but "hardcoding this
  is discouraged").
- The explicit argument AGAINST environment variables, which is directly on point for a launchd
  plist's `EnvironmentVariables`: env vars are "problematic because by default they are inherited
  down the process tree, have size limitations, and issues with binary data."
- The security property env vars lack: "**Access to credentials is restricted to the service's
  user. Unlike environment variables the credential data is not propagated down the process tree.
  Instead each time a credential is accessed an access check is enforced by the kernel.**"
  Credentials are placed in "non-swappable memory" via ramfs.
- Encryption: AES256-GCM keyed from the TPM2 device and/or `/var/lib/systemd/credential.secret`,
  so "credentials protected this way can only be decrypted and validated on the local hardware and
  OS installation." Decryption happens at service activation.
- `SetCredentialEncrypted=` is explicitly safe to embed in a world-readable unit file -- **the
  direct analogue of what a launchd plist should have carried instead of a plaintext token.**

### S6 -- Anthropic / Claude Code official docs, "Authentication"
URL: https://code.claude.com/docs/en/authentication -- accessed 2026-08-11 -- kind: official vendor doc -- WebFetch, full page.
*(The vendor half: this is the exact CLI whose credentials pyfinagent's rail depends on.)*

- **Storage:** "On macOS, credentials are stored in the encrypted macOS Keychain." Linux =
  `~/.claude/.credentials.json` mode `0600`; Windows = `%USERPROFILE%\.claude\.credentials.json`.
- **Authentication precedence (6 slots, verbatim order):** 1 cloud provider creds
  (`CLAUDE_CODE_USE_BEDROCK|VERTEX|FOUNDRY`), 2 `ANTHROPIC_AUTH_TOKEN`, 3 `ANTHROPIC_API_KEY`,
  4 `apiKeyHelper` output, 5 `CLAUDE_CODE_OAUTH_TOKEN`, 6 subscription OAuth from `/login`.
  A gateway session sits outside and outranks all.
- **Long-lived token:** `claude setup-token` mints a **one-year** OAuth token; "The command opens
  the same browser authorization flow as `/login`" and "**It does not save the token anywhere**;
  copy it and set it as the `CLAUDE_CODE_OAUTH_TOKEN` environment variable". Requires a Pro/Max/
  Team/Enterprise plan. Scope is narrow: "It can only make model requests, so it can't establish
  Remote Control sessions or fetch claude.ai connectors."
- **Trap:** "**Bare mode does not read `CLAUDE_CODE_OAUTH_TOKEN`.** If your script passes
  `--bare`, authenticate with `ANTHROPIC_API_KEY` or an `apiKeyHelper` instead."
- **The fail-loud story, and it is recent and directly quotable:**
  - `apiKeyHelper` "is called after 5 minutes or on HTTP 401 response"; TTL overridable via
    `CLAUDE_CODE_API_KEY_HELPER_TTL_MS`.
  - "**Helper failures**: when the script exits with an error, times out, or prints nothing,
    requests fail with `Your apiKeyHelper script is failing` within three attempts. **Before
    v2.1.208, helper failures surfaced as a generic 401 after about ten silent retries.**"
    -- i.e. the vendor itself shipped a fix converting a *silent* degraded path into a *named,
    early* failure. Direct prior art for the principle this step is about.
  - "**Slow helper notice**: if `apiKeyHelper` takes longer than 10 seconds to return a key,
    Claude Code displays a warning notice."
  - Expiry warning: "`Your login expires in 3 days - run /login to renew`" (v2.1.203+; 5 days
    before v2.1.217). Once expired: "each request fails with `Login expired - Please run /login`
    ... **Before v2.1.206, an expired login surfaced as a model error instead.**" `/status` shows
    a `Login` row reading `Expired - log in again` (v2.1.210+).
  - **The unattended sentence, verbatim:** "**Renewing early matters most for sessions that run
    unattended.** A background session in agent view or a Remote Control session that outlives the
    login **stops making progress once the credential expires and can't recover until you sign in
    again.**"
  - Caveat that matters for a monitor: the expiry warning "appears only when a claude.ai or Claude
    Console login is the active credential, and **not** when a cloud provider, `ANTHROPIC_API_KEY`,
    `ANTHROPIC_AUTH_TOKEN`, or `apiKeyHelper` supplies the credential" -- so a rail authenticating
    via slot 2/3/4/5 gets **no** advance-expiry warning at all.

### S7 -- `security(1)` man page, macOS (Apple man-page content, year-less canonical)
URL: https://keith.github.io/xcode-man-pages/security.1.html -- accessed 2026-08-11 -- kind: official vendor man page (curated mirror of Apple's Xcode man pages) -- WebFetch, full page.
*(This is the source for the ACL / partition-list half of the caller's question, which TN3137 could not supply -- see the failed-fetch table.)*

- `unlock-keychain [-hu] [-p password] [keychain]`. The man page's own warning: "**Use of the -p
  option is insecure.**" With no keychain named, the **default** keychain is unlocked.
- A subtle and important sentence for any unlock-based automation: "**Unlocking the login keychain
  might succeed when an incorrect password is presented, if other unlock factors are available.**"
  -> an unlock-based headless probe can report success for the wrong reason.
- `set-keychain-settings [-hlu] [-t timeout] [keychain]`: `-l` = "Lock keychain when the system
  sleeps"; `-u` + `-t` = lock after idle timeout; **omitting the timeout means "no timeout"**.
  These two flags are exactly the settings that decide whether a long-running unattended job's
  keychain access survives sleep/idle.
- `set-key-partition-list [-S partition-list] [-k password] ... [keychain]`. **Partition list
  definition, verbatim:** "an extra parameter in the ACL which limits access to the key based on
  an application's code signature." And: "**You must present the keychain's password to change a
  partition list.**" -> a partition-list repair is itself an operation that needs the keychain
  secret, so it cannot be self-healed by an unattended job that has no secret.
- ACL trust flags on `add-generic-password` / `import` / `create-keypair`: `-A` = "any application
  to access this [item/key] without warning (**insecure, not recommended!**)"; `-T` = "Specify an
  application which may access this [item/key]" (repeatable). Default: "**By default, the
  application which creates an item is trusted to access its data without warning.**"
  -> the default ACL is *creator-only*; a different binary (or a re-signed/updated binary) reading
  the same item is precisely what triggers the interactive prompt that an unattended job cannot
  answer.

### S8 -- Google SRE Workbook, "Alerting on SLOs" (official Google SRE, year-less canonical)
URL: https://sre.google/workbook/alerting-on-slos/ -- accessed 2026-08-11 -- kind: industry/engineering book -- WebFetch, full chapter.

- The four attributes of an alerting strategy, verbatim: **Precision** = "The proportion of events
  detected that were significant. Precision is 100% if every alert corresponds to a significant
  event." **Recall** = "The proportion of significant events detected. Recall is 100% if every
  significant event results in an alert." Plus **detection time** and **reset time** (a long reset
  time "create[s] confusion and alert fatigue").
- Six approaches, ending at **multiwindow multi-burn-rate** as the recommended one because it is
  the only one that gets good precision, recall, detection time AND reset time simultaneously.
  The relevant lesson for a once-per-incident latch: approach 2 ("increased alert window") has
  "extremely poor reset time" and approach 3 ("duration parameter") has "poor recall".
- **HONEST GAP (same as S4):** "The chapter notably omits discussion of alerting when signals
  disappear." Neither SRE source directly supplies a "alert on the ABSENCE of a successful run"
  pattern; that gap is a finding, and the design must supply it locally (see Application, A4).

### S9 -- GitHub Docs, "Sign Xcode applications" (installing an Apple cert on a macOS CI runner)
URL: https://docs.github.com/en/actions/how-tos/deploy/deploy-to-third-party-platforms/sign-xcode-applications -- accessed 2026-08-11 -- kind: official vendor doc / industry practice -- WebFetch, full page.
*(The canonical worked example of "a headless macOS job that needs the keychain".)*

The documented sequence -- note it never touches the login keychain:
```
security create-keychain -p "$KEYCHAIN_PASSWORD" $KEYCHAIN_PATH
security set-keychain-settings -lut 21600 $KEYCHAIN_PATH
security unlock-keychain -p "$KEYCHAIN_PASSWORD" $KEYCHAIN_PATH
security import $CERTIFICATE_PATH -P "$P12_PASSWORD" -A -t cert -f pkcs12 -k $KEYCHAIN_PATH
security set-key-partition-list -S apple-tool:,apple: -k "$KEYCHAIN_PASSWORD" $KEYCHAIN_PATH
security list-keychain -d user -s $KEYCHAIN_PATH
```
- The industry answer to "headless job needs a keychain" is **create a dedicated, ephemeral
  keychain with a random password, unlock it explicitly, set its partition list explicitly, and
  delete it after** -- NOT "reach into the user's login keychain". The password is "any new random
  string" because the keychain is disposable.
- `-lut 21600` = lock-on-sleep + 6h idle timeout, i.e. even the CI pattern accepts a bounded
  unlock window rather than an indefinite one.
- Cleanup is explicit for self-hosted runners: "the keychain and provisioning profile might still
  exist on the runner" after the job.
- **Limit of the analogy (stated honestly):** this pattern works because CI *possesses* the secret
  (a p12 + password from a secrets store) and merely needs a container for it. It does NOT solve
  "the secret only exists inside the user's login keychain and the job has no copy" -- which is
  pyfinagent's actual situation post-86.7-fix.

### S10 -- Microsoft Entra identity platform, "Refresh tokens" (Microsoft official)
URL: https://learn.microsoft.com/en-us/entra/identity-platform/refresh-tokens -- accessed 2026-08-11 -- kind: official vendor doc -- WebFetch, full page. `ms.date: 2025-11-05`, `updated_at: 2026-06-15` -> **a last-2-year source; second recency anchor.**

- Lifetimes: "**24 hours** for single-page applications", "24 hours for apps that use email
  one-time passcode authentication flow", "**90 days for** all other scenarios."
- Rotation is the default behaviour: "**Refresh tokens replace themselves with a fresh token upon
  every use.** The Microsoft identity platform doesn't revoke old refresh tokens when used to
  fetch new access tokens. **Securely delete the old refresh token after acquiring a new one.**"
- Expiry AND independent revocation: "Refresh tokens will automatically expire once the lifetime
  period elapses. **Additionally, they can be revoked by the sign-in service at any time before
  their expiration.** Your app should handle such revocations gracefully by redirecting the user
  to an interactive sign-in prompt to reauthenticate."
- The revocation matrix is the useful artefact: password change / SSPR / admin password reset /
  user revokes / **admin revokes all refresh tokens for a user** / single sign-out each revoke
  different token classes. "Admin revokes all refresh tokens for a user" -> **Revoked** in every
  column including confidential-client tokens.
- **The generalisable claim: there is NO long-lived credential whose validity an unattended job may
  assume.** Even a 90-day refresh token is revocable at any instant by an event the job cannot
  observe. The only sound design is *detect and alert*, never *assume and degrade*.

## Attempted but NOT read in full (recorded so the gap is auditable)

| URL | Attempt | Outcome |
|---|---|---|
| https://developer.apple.com/documentation/technotes/tn3137-on-mac-keychain-apis-and-implementations | WebFetch | **HTTP 404** |
| https://developer.apple.com/tutorials/data/documentation/technotes/tn3137-on-mac-keychain-apis-and-implementations.json | WebFetch (JSON doc API) | **HTTP 404** |
| https://developer.apple.com/documentation/technotes/tn3137-on-mac-keychains | WebFetch | 200 but **JS-rendered**: only the `<title>` was extractable, zero body. Same class as the known `cloud.google.com` nav-only failure. |
| https://www.usenix.org/legacy/events/hotos03/tech/full_papers/candea/candea_html/index.html | WebFetch (Crash-Only Software, Candea & Fox, HotOS IX) | **HTTP 403** |

TN3137 is the one source that would have settled "data protection keychain vs file-based keychain
for a daemon" from Apple's own mouth. It could not be retrieved. **S7 (`security(1)`) covers the
ACL/partition-list half; the daemon-vs-agent half is covered by S3 (TN2083) plus the direct
MEASUREMENT in I-7/I-8 below.** This is stated rather than papered over.

## Search-method disclosure (protocol deviation, disclosed not hidden)

`.claude/rules/research-gate.md` requires three search-query variants (current-year / last-2-year /
year-less). **WebSearch was unavailable for this entire session**: the first two search calls
returned `this session has used its web search budget (200 of 200 WebSearch calls)` -- the budget is
session-shared and was already spent before this agent was spawned. `WebFetch` was unaffected.

Compensating method, and what it does and does not buy: sources were selected by targeting
canonical URLs directly, deliberately spanning the same three bands the rule asks for --
**year-less canonical** (RFC 8628 (2019), TN2083, `security(1)`, SRE Book, SRE Workbook, systemd
CREDENTIALS), **last-2-year** (RFC 9700, Jan 2025; Entra refresh-tokens, `ms.date` 2025-11-05 /
updated 2026-06-15), and **current-year frontier** (the Claude Code authentication doc, which
version-stamps behaviour changes at v2.1.203 / .206 / .208 / .210 / .217 / .223 -- all 2026
releases). The rule's own stated alternative for making the discipline visible -- "ensuring the
source table has a mix of current-year, last-2-year, and year-less hits" -- is therefore satisfied.
What it does NOT buy: no discovery of sources I did not already know to exist. If Main wants a
true frontier sweep for this topic, re-spawn in a session with search budget.

## Recency scan (2024-2026) -- PERFORMED

Method: as above (targeted, not searched). **Result: 3 findings in the window that materially
change the design, not zero.**

1. **RFC 9700 (Jan 2025)** supersedes the informal refresh-token advice in RFC 6749 with a
   normative MUST for public clients: sender-constrained **or** rotated. A CLI is a public client.
   This is new normative text since the older OAuth BCP drafts.
2. **The Claude Code CLI changed its failure semantics from silent to loud during 2026, twice.**
   `apiKeyHelper` failures used to surface "as a generic 401 after about **ten silent retries**"
   and now fail with a named error "**within three attempts**" (v2.1.208); an expired login used
   to surface "**as a model error**" and now says `Login expired - Please run /login` (v2.1.206);
   plus a 3-day advance warning (v2.1.203) and a `/status` `Login: Expired` row (v2.1.210). The
   vendor independently converged on exactly the principle this step is about. **It also means the
   local CLI version determines how loud the failure is** -- pyfinagent's measured CLI is 2.1.226,
   past all four, so the loud behaviour is available.
3. **Entra doc refreshed 2025-11-05 / 2026-06-15** with the explicit "revoked by the sign-in
   service at any time before their expiration" framing and the per-class revocation matrix.

No finding in the window contradicts the older canonical sources (TN2083's daemon/agent model and
`security(1)`'s ACL model are unchanged); the newer work is additive.

---

# INTERNAL CODE INVENTORY (the Explore half) -- all claims measured 2026-08-11

| # | File / object | Anchor | Role | Status |
|---|---|---|---|---|
| I-1 | `backend/agents/claude_code_client.py` | 821 lines total | The production rail client. `claude_code_invoke()` builds the argv and shells out. | LIVE |
| I-2 | env scrub | `claude_code_client.py:401-412` | Removes `ANTHROPIC_API_KEY` + `ANTHROPIC_AUTH_TOKEN` from the subprocess env so precedence slots 2/3 cannot outrank the Max OAuth credential. **It removes two vars and adds none** -- so the rail has NO env credential of its own by design. | LIVE, correct |
| I-3 | `--bare` prohibition | `claude_code_client.py:366-370` (comment) | "Do NOT add `--bare` -- ... --bare rejects OAuth + keychain reads and requires `ANTHROPIC_API_KEY`". **Corroborated by S6**: "Bare mode does not read `CLAUDE_CODE_OAUTH_TOKEN`." | LIVE, corroborated |
| I-4 | Failure signatures (the six) | `:423-428` timeout -> `ClaudeCodeError`; `:430-437` `FileNotFoundError` -> "binary not found"; `:439-457` non-zero exit (logs **both** stdout and stderr -- phase-66.2 fix, because the CLI puts auth/limit diagnostics on **stdout**); `:459-462` empty stdout; `:464-472` `json.JSONDecodeError`; `:474-483` `subtype != "success"` | Every failure path RAISES. None returns a degraded value. | LIVE, sound |
| I-5 | **Where a 401 lands** | `:474-483` | A 401 does **not** produce a non-zero exit: the CLI exits 0 and emits an envelope with `is_error=true` / `subtype != "success"`. So the 401 is caught by the **subtype** check, not the returncode check. `duration_api_ms=0` (the step's own measurement) is the distinguishing fingerprint of an auth rejection vs a slow success. | LIVE -- **this is the seam any auth-specific alerting must hook** |
| I-6 | Circuit breaker + page | `:171-211`; state at `:93-103`; reset at `:115-122`; `paged` latch at `:97` and `:184-185` | On the closed->open transition it calls `raise_cron_alert_sync(source="claude_code_rail", error_type="breaker_open", severity="P1", ... operator_action="check \`claude auth status\` on the host")`. Paging is fail-open (`except Exception: logger.warning(...)`). | LIVE. **Gap:** `paged` is an exactly-once latch and `rail_guard_reset()` rebuilds the whole state **per cycle** -- so it pages once per cycle, and the alert says "breaker_open", never "auth". It cannot distinguish a dead credential from a slow model. |
| I-7 | `claude_code_health_probe()` | `:494-533` | Runs `claude auth status` in the **same scrubbed env** the real rail uses; token-less and free; **never raises**; returns `(ok, detail)`. | LIVE. Called from `backend/services/autonomous_loop.py:464-470` (`asyncio.to_thread`). This is the existing, reusable headless auth seam. |
| I-8 | away-ops watchdog auth leg | `scripts/away_ops/healthcheck.sh:85-139`, seam at `scripts/away_ops/auth_state.py` | `claude auth status` every 30 min from a launchd agent; pages **once per incident** via a latch file `handoff/away_ops/auth_page_state.json`; message names the runbook `docs/runbooks/credential-expiry-monitoring.md`. The script's own comment states the limit: `auth status` = "LOCAL credential presence (cannot [detect server-side revocation])". | LIVE and GREEN today |
| I-9 | away-session pre-launch probe | `scripts/away_ops/run_away_session.sh:142-158` | If the latch is open, spend ONE 20s probe (`claude -p --max-turns 1 --output-format json`) instead of a full launch; clears the latch on success, else `result=auth-dead-skip`, `exit 0`. | LIVE. **Note `exit 0` on auth-dead** -- launchd records success; the loudness comes only from the earlier one-shot page. |
| I-10 | away-session page on 401 | `run_away_session.sh:190-198` | Reads `SLACK_BOT_TOKEN` out of `backend/.env` with `grep`/`cut` and posts "P1 AWAY: Claude credential DEAD (401)". | LIVE |
| I-11 | launchd inventory | `~/Library/LaunchAgents/com.pyfinagent.*.plist` | **All pyfinagent jobs are LaunchAgents. `/Library/LaunchDaemons/com.pyfinagent.*` does not exist (zsh: "no matches found").** | MEASURED |
| I-12 | plist `EnvironmentVariables` today | backend: `['DEV_LOCALHOST_BYPASS','PATH','PYTHONUNBUFFERED']`; slack-bot: `['PATH','PYTHONUNBUFFERED']`; away-watchdog: `['HOME','PATH']`; away-session-am: `['CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS','HOME','PATH']` | **No plist carries `CLAUDE_CODE_OAUTH_TOKEN`.** The step's premise is CONFIRMED for the live plists. | MEASURED |
| I-13 | **the `.bak` siblings** | `~/Library/LaunchAgents/*.bak*` | **12 backup plists still contain the literal token** (3 backups x 4 plists: `.bak.20260809T145706`, `.bak.20260809T151306`, `.bak.pre-token-removal.20260809T151936`). | **OPEN DEFECT** -- see F-1 |
| I-14 | keychain item | `security find-generic-password -s "Claude Code-credentials"` | `genp` item, `acct="ford"`, in `/Users/ford/Library/Keychains/login.keychain-db`. `cdat=20260416233224Z`, **`mdat=20260811075516Z`** (today). | LIVE. The `mdat` proves the item is **rewritten on refresh** -> mutable shared state (see F-3) |
| I-15 | login-keychain settings | `security show-keychain-info ~/Library/Keychains/login.keychain-db` -> `no-timeout` | **No idle timeout and no lock-on-sleep.** Per S7 those are exactly `-t`/`-u` and `-l`. | MEASURED -- the current config is the *permissive* one |
| I-16 | away-ops rail 9 | `docs/runbooks/away-ops-rules.md` rail 9 | "launchctl authority limited to backend/frontend/slack-bot kickstart; **never touch watchdog/session plists**." | BINDING CONSTRAINT on any fix |
| I-17 | live watchdog health | `handoff/away_ops/health.jsonl` tail | `{"ts":"2026-08-11T07:43:54Z","ok":true,"auth_ok":"true","auth_detail":"ok","auth_p1":false}`; `launchctl print` -> `runs = 654`, `last exit code = 0`. | **GREEN today, post-removal** |
| I-18 | 86.7 masterplan entry | `.claude/masterplan.json` id `86.7` | 6 immutable criteria + a `live_check`. Criteria 1/2 demand a MEASURED headless auth result and a MEASURED keychain-unavailable behaviour; criterion 5 demands an explicit CLOSED/STILL-OPEN ruling on 62.1.1 and 85.3.3; criterion 6 demands a mutation test (re-introducing the env var must break the rail again). | pending |

## Key findings

**F-1 (NEW, SECURITY, P0-adjacent -- decides criterion 5). The live token value is committed to
git, in 5 TRACKED files, so 62.1.1 and 85.3.3 CANNOT close on the plist removal alone, and a
ROTATION ask is now evidence-backed rather than conditional.**
Measured by fingerprinting only (no value printed, no value reproduced anywhere in this brief):
- Backup plists under `~/Library/LaunchAgents/` carry three distinct token values:
  `(len 79, sha256[:12]=15dc02093816)`, `(79, 35558d206b8c)`, `(92, 32fd30514637)`.
  The two 79-char ones are **exactly** the two the 86.7 audit_basis already names; the 92-char one
  is the malformed doubled-prefix value from ask #20.
- The **92-char value `32fd30514637` also appears, verbatim and git-TRACKED**, in five files:
  `handoff/away_ops/session_am_20260809T053008Z.json`, `session_am_20260810T053009Z.json`,
  `session_pm_20260808T200008Z.json`, `session_pm_20260809T200008Z.json`,
  `session_pm_20260810T200010Z.json`. Overlap between the tree fingerprints and the backup-plist
  fingerprints is non-empty: `[(92, '32fd30514637')]`.
- A sixth file, **untracked**, `backend/.env.env.bak-20260417-224659`, carries a *fourth*, distinct
  108-char value (`d773f475c399`).
- **Method caveat, stated because 62.1.1's own verification command gets this wrong:** the step's
  command `git log --all -S CLAUDE_CODE_OAUTH_TOKEN` returns 45 commits and
  `-S "sk-ant-oat01-" --pickaxe-regex` returns 40 -- but those counts are dominated by artifacts
  that merely *discuss* the variable name and the prefix in prose. The count is not the exposure.
  The exposure test that actually discriminates is a **full-shape** match
  (`sk-ant-oat01-[A-Za-z0-9_-]{20,}`) cross-checked by **hash against the known values** -- which is
  what produced the result above. Do not let the raw `-S` count stand in for it.
- Consequence: 62.1.1 criterion 1 ("the literal token no longer appears in **any** file under
  `~/Library/LaunchAgents/`, **including the `*.bak` siblings**") is **NOT satisfied** -- 12 files
  still carry it. 85.3.3's git-history criterion resolves **positive**, which by its own text means
  "the token must be rotated -- an operator action to file in the ask list with the evidence".
- Mitigating (do not overstate the severity): all four values are *already known-dead* -- the two
  79-char tokens 401 by measurement, the 92-char one is malformed. Rotation here is hygiene +
  invalidating anything still live, not an active-compromise response.

**F-2 (decides criterion 1). All pyfinagent launchd jobs are AGENTS, not DAEMONS, and that is why
the keychain is reachable at all -- and a launchd-context keychain read is already being MEASURED
every 30 minutes, successfully, today.**
Per S3/TN2083 the distinction is categorical: an agent "runs in the background **on behalf of a
particular user**" and can "reliably access the user's home directory", whereas for a daemon "if a
daemon tries to get a user preference when the user is not logged in, it will fail." I-11 measures
that pyfinagent has **zero** `/Library/LaunchDaemons/com.pyfinagent.*` entries. I-17 measures the
away-watchdog (a LaunchAgent, `runs=654`, `last exit code = 0`) writing
`auth_ok:"true"` at `2026-08-11T07:43:54Z`, i.e. **after** the 08-09 token removal -- so a launchd
agent invoking `claude auth status` with no env token is already succeeding in production.
**But criterion 1 is still NOT satisfied by that**, and the step must not accept it as such:
`healthcheck.sh:86-89` says in its own comment that `auth status` proves "LOCAL credential
presence" only. Presence is not a model request. Criterion 1 asks for the away-session **entrypoint**
to authenticate; the honest measurement is I-9's real probe shape
(`claude -p --max-turns 1 --output-format json`) run from the launchd context, checked for
`is_error`/`subtype`/`duration_api_ms`, not `auth status`.

**F-3 (NEW, changes the risk model). The keychain item is MUTABLE state that is being rewritten
right now, which makes concurrent sessions a live hazard -- not a theoretical one.**
I-14 measures `mdat = 2026-08-11 07:55:16Z` on an item created 2026-04-16: the OAuth credential is
refreshed and **written back** to the keychain. S2/RFC 9700 requires public clients to use
rotation or sender-constraining, and S10/Entra states the general rotation contract plainly:
"Refresh tokens replace themselves with a fresh token upon every use ... Securely delete the old
refresh token after acquiring a new one." A rotated credential in a single shared keychain item is
**single-writer state**. pyfinagent runs the backend agent, the away-session agents, the watchdog,
AND interactive Claude Code sessions against that one item (the operator's own memory records two
concurrent Claude sessions). A lost rotation race is indistinguishable from a revocation at the
call site -- it presents as exactly the `401 / duration_api_ms=0` signature of I-5. This is a real
candidate explanation for "two separately-minted `setup-token` values both 401" and it deserves to
be on the table alongside "Anthropic's setup-token is broken".

**F-4 (decides criterion 2). Every alert on this path is a ONE-SHOT latch, and one of the
skip paths exits 0 -- so "the rail is down" is loud exactly once and quiet forever after.**
Three independent latches, all measured: `_RAIL_GUARD.paged` (I-6, one page per *cycle*, and the
message says `breaker_open`, never `auth`); `auth_page_state.json` `incident_open` (I-8, "Paged
ONCE per incident"); and `run_away_session.sh` (I-9) which on auth-dead logs
`result=auth-dead-skip` and **`exit 0`**, so `launchctl print` reports `last exit code = 0` for a
session that did no work. Against S4: "Every page should be actionable" is satisfied, but the
**recall** attribute from S8 ("100% if every significant event results in an alert") is not --
after the first page, subsequent significant events produce no alert. And S4's named failure mode,
"an HTTP 200 success response, but coupled with the wrong content", is the exact shape of an
`exit 0` no-op session. Both SRE sources **omit** the "alert on the absence of a successful run"
pattern (honest gap, S4/S8) -- so the design must add a **positive-heartbeat / staleness** alarm
locally: alert when the last SUCCESSFUL away-session or rail call is older than N, rather than
alerting only on transitions.

**F-5 (decides criterion 3). The literature offers exactly four fallback shapes, and three of them
are unavailable or forbidden here -- which makes "accept the risk, with a loud detector" the
defensible answer, and that should be argued rather than defaulted into.**
- *(a) A long-lived token* -- S6 documents `claude setup-token` as a **one-year** OAuth token,
  precedence slot 5, "**It does not save the token anywhere**". Currently broken for this account
  (86.7 audit_basis: two well-formed tokens, both `401 OAuth access token is invalid`, CLI
  2.1.226). S10 is the reason not to trust this even when repaired: a long-lived token "can be
  revoked by the sign-in service **at any time before** [its] expiration".
- *(b) `apiKeyHelper`* -- precedence slot 4, "called after 5 minutes or on HTTP 401", TTL via
  `CLAUDE_CODE_API_KEY_HELPER_TTL_MS`, and since v2.1.208 it fails **loudly** in three attempts.
  This is the vendor's own designed hook for "dynamic or rotating credentials, such as short-lived
  tokens fetched from a vault". **Trap:** it sits at slot 4, *above* the keychain -- adopting it
  would re-create exactly the override that caused this outage (86.7: "the PRESENCE of
  `CLAUDE_CODE_OAUTH_TOKEN` overrode a WORKING keychain credential"). Also it needs a secret to
  hand out, which pyfinagent does not have.
- *(c) A dedicated unlocked keychain* -- S9's CI pattern. Requires possessing the secret; the
  Claude credential exists only inside the login keychain. **Not applicable.**
- *(d) A keychain-unlock step in the away runbook* -- S7 makes this unattractive: `-p` is flagged
  "insecure" by Apple's own man page, it needs the login password in a script, and
  "**Unlocking the login keychain might succeed when an incorrect password is presented, if other
  unlock factors are available**" -- so it can even report a false success.
- The residual honest option is *accept-the-risk + a detector that cannot be silent*, which the
  step's criterion 3 explicitly permits ("or its absence is justified ... with the blast radius
  stated"). F-4 is what has to be fixed for that to be defensible.

**F-6 (bounds the exposure for criterion 2's "keychain unavailable" test). Measured, the login
keychain is currently configured `no-timeout` -- so idle and sleep do NOT lock it; only logout,
reboot, or an explicit `security lock-keychain` do.**
I-15 measures `no-timeout`; S7 identifies the two settings that would change that (`-l`
lock-on-sleep, `-u`/`-t` idle timeout), and that omitting the timeout "means no timeout". So the
86.7 audit_basis phrase "can be locked by reboot, logout, or screen lock" is **partly refuted as
configured**: screen lock does not lock this keychain today, because lock-on-sleep is off. Reboot
and logout still do. This narrows criterion 2's test: the realistic trigger is an
unattended-window **reboot** (e.g. after a crash or an OS update), not a screensaver. It also
means the exposure is one `security set-keychain-settings -l` away from getting worse, so the
setting itself is worth pinning and asserting.

## Consensus vs debate (external)

**Consensus, and it is unusually strong across all four independent traditions read here:**
- *Credentials must not travel in environment variables.* S5/systemd: env vars "are inherited down
  the process tree", vs credentials where "each time a credential is accessed an access check is
  enforced by the kernel"; S7: "Use of the -p option is insecure"; S1/RFC 8628: device clients
  "are generally incapable of maintaining the confidentiality of their credentials".
- *No long-lived credential's validity may be assumed.* S10 (revocable at any time), S2 (rotation
  MUST for public clients), S6 (a one-year token that still expires and whose owner-account can
  break).
- *Authentication failure is terminal, not retryable.* S1 is the crispest formulation available:
  only `authorization_pending`/`slow_down` may be retried; on any other error the client "**MUST
  stop polling** and SHOULD react accordingly, for example, by displaying an error to the user."
  An auth error is a *stop-and-tell-a-human* condition by protocol design.
- *Degraded-but-successful output is a failure mode to design against.* S4 names "an HTTP 200
  success response, but coupled with the wrong content".

**Genuine debate / tension:**
1. **Ephemeral-keychain (S9) vs use-the-user's-keychain (the pyfinagent design).** The CI industry
   answer assumes you hold the secret. When the credential is a browser-minted OAuth session that
   only ever materialises inside the login keychain, S9's pattern does not apply, and no source
   read here offers a clean answer for that case. That is a real gap in the literature, not an
   oversight in this brief.
2. **Alert-on-transition vs alert-on-state.** I-6 cites Fowler/PagerDuty alert-on-transition; S8's
   *recall* attribute pushes the other way. Both are defensible; the resolution is that transition
   alerts need a **staleness/heartbeat** companion, which neither SRE chapter supplies.
3. **Rotation as protection vs rotation as a fragility.** S2 mandates rotation; F-3 shows rotation
   turns a shared credential store into single-writer state and manufactures a failure mode that
   is indistinguishable from revocation. The mitigation (S10) is "securely delete the old refresh
   token after acquiring a new one" -- i.e. accept it and never keep stale copies.

## Pitfalls (from the literature, mapped to how each would bite here)

- **P1 -- Higher-precedence credential silently shadows a working one.** S6's 6-slot ladder; the
  measured 86.7 root cause is exactly this (slot 5 beat slot 6). *Any* future fallback added at
  slots 2-5 re-arms the same trap. Any fix must assert which slot is active, not that a credential
  exists.
- **P2 -- `auth status` proves presence, not authorisation.** S6 + `healthcheck.sh:86-89`'s own
  comment. A probe that cannot see a server-side 401 will report green through an outage.
  (I-8 is currently green by this weaker test.)
- **P3 -- `--bare` silently drops OAuth + the env token.** S6: "Bare mode does not read
  `CLAUDE_CODE_OAUTH_TOKEN`"; I-3 already guards it. Do not let a "faster probe" reintroduce it.
- **P4 -- The advance-expiry warning does not fire for the credential class you are using.** S6:
  the 3-day warning "appears only when a claude.ai or Claude Console login is the active
  credential, and not when a cloud provider, `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or
  `apiKeyHelper` supplies the credential." Adopting an `apiKeyHelper` fallback would **remove** the
  early-warning signal.
- **P5 -- Unlock succeeds for the wrong reason.** S7: "Unlocking the login keychain might succeed
  when an incorrect password is presented, if other unlock factors are available."
- **P6 -- Default keychain ACL is creator-only.** S7: "By default, the application which creates an
  item is trusted to access its data without warning"; a partition list "limits access to the key
  based on an application's code signature". A `claude` CLI **upgrade** (new signature) is
  therefore a plausible future trigger for a prompt no unattended job can answer -- and repairing a
  partition list itself requires the keychain password ("You must present the keychain's password
  to change a partition list").
- **P7 -- Backups outlive the fix.** Not from the literature but measured (F-1/I-13): the remediation
  created 12 `.bak` files whose whole purpose was to preserve the removed secret.
- **P8 -- `exit 0` on a skipped run.** I-9. Any external supervisor keyed on exit status scores a
  credential outage as a healthy run (S4's "wrong content behind a 200").

## Application to pyfinagent (external findings -> file:line)

- **A1 (criterion 1).** Do not accept I-8's `auth status` green as the measurement. Run the
  **real** probe shape from `run_away_session.sh:147-149`
  (`claude -p --max-turns 1 --output-format json`) in the launchd agent context and assert on the
  envelope: `is_error`, `subtype`, and `duration_api_ms` (I-5 -- `duration_api_ms=0` is the auth
  fingerprint). Faithful stand-in = same `HOME`, same `PATH` as the plist (`away-watchdog` PATH
  omits the venv, per I-12), launched via launchd, not from an interactive shell.
- **A2 (criterion 2).** The keychain-unavailable test is `security lock-keychain
  ~/Library/Keychains/login.keychain-db` -- because I-15/F-6 measures `no-timeout`, idle and sleep
  will not produce the state naturally. Record the exact envelope the rail sees and confirm it
  lands at `claude_code_client.py:474-483` (subtype) rather than the returncode branch.
- **A3 (criterion 2, the loudness half).** Add auth as a **first-class** error type, not a subtype
  of `breaker_open` (I-6). The alert payload already has the right operator action; the
  classification does not. And fix the recall gap (F-4/S8) with a staleness alarm on the newest
  `ok:true` in `handoff/away_ops/health.jsonl` and on the newest successful rail call -- the
  one-shot latches (`_RAIL_GUARD.paged` `:97`, `auth_page_state.json` `incident_open`) are
  structurally incapable of re-paging.
- **A4 (criterion 2).** Reconsider `run_away_session.sh:157` `exit 0` on `auth-dead-skip`. A
  non-zero exit makes `launchctl print`'s `last exit code` a truthful signal (P8). Constrained by
  rail 9 (I-16): the plists themselves are off-limits; the **script** is not.
- **A5 (criterion 3).** Recommend **accept-the-risk + fix the detector**, argued from F-5: (a) is
  broken upstream and revocable anyway (S10); (b) sits above the keychain and re-arms P1/P4; (c)
  needs a secret that does not exist outside the keychain; (d) is flagged insecure by Apple and can
  false-succeed (P5). Blast radius to state explicitly: on a reboot inside an away window, every
  away session and the cc_rail lose auth until the operator logs in -- the analysis pipeline runs
  its degraded fallbacks (policy: hold), so the book does not trade on bad data, but it does not
  trade at all.
- **A6 (criterion 5) -- the ruling, with evidence.** **62.1.1 = STILL OPEN**: its criterion 1
  requires the token be absent from every file under `~/Library/LaunchAgents/` *including the
  `.bak` siblings*, and 12 such files still carry it (I-13). **85.3.3 = STILL OPEN**: its
  git-history criterion resolves **positive** (F-1: five tracked files carry the byte-identical
  92-char value), which by its own wording mandates a rotation ask. Both had their *plist-storage*
  half resolved by the 08-09 removal (I-12); neither is closeable. Also file the untracked
  `backend/.env.env.bak-20260417-224659` (fourth distinct value) -- it is out of both steps'
  stated scopes, so per the standing rule it wants its own step rather than silent absorption.
- **A7 (criterion 6).** The mutation is well-posed and cheap: copy the backend plist, re-insert
  `CLAUDE_CODE_OAUTH_TOKEN` with a known-dead value, and assert the production client
  (`ClaudeCodeClient.generate_content`) fails. The mechanism to assert is the **precedence
  shadowing** (S6 slot 5 > slot 6), and the observable is I-5's signature. Do NOT mutate a live
  plist -- rail 9 (I-16) and the standing "isolation must cover every channel" discipline both
  apply; use a copy plus an env-var-level A/B in-process, which is what the immutable verification
  command already does (it `os.environ.pop`s the variable, spelled via `chr()` so the checker
  itself contains no literal).
- **A8 (hardening, cheap).** Pin and assert `security show-keychain-info` stays `no-timeout` and
  that lock-on-sleep is off (F-6/S7). It is a one-line precondition whose silent change would
  reintroduce the outage class, and today nothing observes it.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **10** (S1-S10)
- [x] 10+ unique URLs total -- **22** (10 read in full + 4 attempted-and-failed + 8 collected-not-fetched)
- [x] Recency scan (last 2 years) performed + reported -- 3 in-window findings; method disclosed
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (I-1..I-18)

Soft checks:
- [x] Internal exploration covered every module in the caller's INTERNAL SCOPE
- [x] Contradictions / consensus noted (3 live tensions recorded, incl. one against a project idiom)
- [x] All claims cited per-claim
- [~] **Search-variant discipline NOT performed as specified** -- WebSearch budget was exhausted
  session-wide before this agent started. Disclosed in full above with the compensating method and
  its limits. Not a hard blocker per `researcher.md`'s checklist; flagged so Main can re-spawn in a
  search-capable session if frontier discovery matters.
- [~] TN3137 (the one Apple source that would settle data-protection-keychain-vs-daemon from the
  vendor) was unreachable across three URL forms; substituted with `security(1)` + TN2083 + direct
  measurement.

## Collected but not fetched in full (context; does NOT count toward the gate)

| URL | Kind | Why not fetched |
|---|---|---|
| https://datatracker.ietf.org/doc/html/rfc6749 | standard | OAuth 2.0 core; refresh-token semantics reached via S2's normative update |
| https://datatracker.ietf.org/doc/html/rfc8705 | standard | mTLS sender-constraining; cited by S2, not applicable to this CLI |
| https://datatracker.ietf.org/doc/html/rfc9449 | standard | DPoP; cited by S2, not applicable to this CLI |
| https://code.claude.com/docs/en/headless | vendor doc | `--bare` detail already quoted via S6; I-3 already guards it |
| https://code.claude.com/docs/en/settings | vendor doc | `apiKeyHelper` semantics already quoted via S6 |
| https://code.claude.com/docs/en/errors | vendor doc | the named failure string already quoted via S6 |
| https://code.claude.com/docs/en/env-vars | vendor doc | `CLAUDE_CODE_API_KEY_HELPER_TTL_MS` already quoted via S6 |
| https://code.claude.com/docs/en/claude-apps-gateway | vendor doc | enterprise gateway; not available on a personal Max plan |
| https://www.usenix.org/legacy/events/hotos03/tech/full_papers/candea/ | peer-reviewed | Crash-Only Software; 403 (see failed-fetch table) |
| https://developer.apple.com/documentation/technotes/tn3137-on-mac-keychains | vendor doc | JS-rendered, body unextractable (see failed-fetch table) |
| https://sre.google/sre-book/table-of-contents/ | book index | navigation only |
| https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens | vendor doc | PAT expiry; S10 already supplies the expiry/revocation argument with a stronger matrix |

---
*End of brief. The status envelope at the top of this file is authoritative for gate assessment.*
