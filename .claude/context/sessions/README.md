# Session Logs (Episodic Memory)

This directory stores session summaries from remote agent runs.
Each run writes a log file before pushing, and reads previous logs at startup.

Format: `YYYY-MM-DD-HHMM.md`

The lead agent MUST:
1. At startup: read the last 3 session logs for context
2. At end: write a session summary before pushing
