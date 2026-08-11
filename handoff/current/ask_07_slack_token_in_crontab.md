# ASK #7 -- a live Slack bot token is inlined in the crontab

**Found 2026-08-11 ~18:5x CEST** while verifying that the 20:00 book cycle was
actually scheduled. Incidental to 86.33; not part of any step.

**No value is printed anywhere in this file, and nothing was touched, rotated or
deleted.**

## What it is

`crontab -l` line 1:

```
*/2 * * * * export SLACK_BOT_TOKEN='<59-char xoxb- value>' && scripts/slack_mention_checker.sh ...
```

- one distinct token, **59 chars**, `sha256[:16] = dc0f762af1c68ebe`
- fires **every 2 minutes**

## Why the shape matters more than the storage

`export TOKEN=... && script` puts the secret **into the command line of the spawned
shell**, so it is visible in `ps` output to any process on this machine for the
lifetime of each run -- 720 times a day. A secret in a file is readable by whoever
can read the file; a secret on a command line is readable by everyone, briefly, on
a schedule.

## WHAT THIS IS NOT -- I checked before raising it

My first probe found **35 git-tracked files containing an `xoxb-` string** and I
was about to report a second published-credential incident alongside ASK #2. It is
not one:

```
distinct token VALUES in git : 5
  86c686a7417fdea0  len=15  in 20 files
  1f98e9c1097e3ef4  len=33  in 1
  f9a545a0535545fc  len=20  in 1
  5b39dfa8182c2ebf  len=16  in 1
  717f1f393204cfc6  len=15  in 1
```

**All 15-33 chars -- far too short to be real Slack tokens -- and NONE matches the
crontab value's hash.** They are placeholders in tests and archived briefs. The real
token appears in **zero** git-tracked files, so unlike ASK #2's `sk-ant-*` it has
**not** reached origin.

## The remedy looks cheap, and I have not applied it

`backend/.env` already carries `SLACK_BOT_TOKEN`, so the crontab inlining appears
unnecessary -- the job could source the env file instead of exporting the literal.

**I have not made that change.** It touches a credential and a scheduled job, the
standing goal forbids `.env` writes outright, and "looks equivalent" is not
"verified equivalent" for something that runs 720 times a day next to a live book.

## NEEDS

1. Rotate this token, or accept the `ps` exposure as tolerable on a single-user Mac?
2. May the crontab job be rewritten to source `backend/.env` rather than inline the
   literal? That is a crontab edit, not a `.env` write, but it changes how a live
   job authenticates.

## Related, and deliberately kept separate

**ASK #2** is a different credential (`sk-ant-*`, 92 chars, `32fd305146379e49`) in
**5 git-tracked files that ARE on origin** -- roughly 3 days of remote exposure.
That one is published; this one is local. Conflating them would misstate both.
