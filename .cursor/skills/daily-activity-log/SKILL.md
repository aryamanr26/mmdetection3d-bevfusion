---
name: daily-activity-log
description: >-
  Creates a concise daily activity log of important work by the user and the AI
  agent: date, major tasks, commands, code changes, configuration or environment
  changes, problems encountered, troubleshooting steps and solutions, decisions,
  and the current state or outcome. Use at the end of a working session, or when
  the user asks to log the day, write a daily recap, session notes, document
  today's work, wrap up, write a changelog of the conversation, or continue later
  without rereading the chat. Writes docs/daily/YYYY-MM-DD.md and links that
  document from the project's main README.md.
---

# Daily Activity Log

At the end of each working session, create a concise daily activity log documenting the important work completed by the user and the AI agent. Capture the date, major tasks, commands or code changes that were made, important configuration or environment changes, problems encountered, troubleshooting steps and their solutions, decisions made, and the current state or outcome of the work. Focus on meaningful technical changes and discoveries rather than documenting every message in the conversation. The log should be chronological where useful and detailed enough that the user can understand what changed that day and continue the work later without rereading the entire conversation. Include exact commands, file names, configuration values, errors, or implementation details when they are important for reproducing or understanding the work. Clearly distinguish between changes actually completed and ideas or suggestions that were discussed but not implemented. Link that document to main README.md.

## When to write

Write or update the log when the user wraps up, asks to document the day, or this skill is attached. Do not invent a session that did not happen. If today's log already exists, merge new work into it instead of creating a second file.

## Gather facts first

Collect only what you can verify. Do not guess commands, paths, errors, or outcomes.

1. This conversation: tasks, decisions, failures, and results.
2. Workspace changes: `git status`, `git diff`, and recent `git log` when the directory is a git repo.
3. Terminal history in the current session: commands that were actually run and whether they succeeded.
4. Prior agent transcripts for the same calendar day if this chat is missing earlier work.
5. The existing file at `docs/daily/YYYY-MM-DD.md` if present.

Skip secrets, credentials, tokens, and full dumps of logs or datasets.

## Log location

- One file per calendar day: `docs/daily/YYYY-MM-DD.md` (use the user's local date).
- Index: `docs/daily/README.md` (create if missing).
- Create `docs/daily/` if it does not exist.

## Log template

```markdown
# Daily activity — YYYY-MM-DD

## Summary
[One short paragraph: what the user and agent set out to do, and where things stand now.]

## Major tasks
- [Task and outcome]

## Commands and code changes
- `exact command` — why it was run, result
- `path/to/file` — what changed and why

## Configuration and environment
- [Setting, path, version, or env change that matters for reproducing the work]

## Problems and troubleshooting
- **Problem:** [error or symptom, include the exact message when it matters]
  - **Tried:** [step]
  - **Solution:** [what fixed it, or "unresolved"]

## Decisions
- [Choice made and the reason]

## Completed
- [Work that is actually done]

## Discussed but not implemented
- [Ideas or suggestions only; do not mix these into Completed]

## Current state
[What works, what is left, and the next concrete step to resume.]
```

Omit empty sections. Keep the writing chronological inside a section when order matters (especially troubleshooting).

## Link from README.md

After writing the day's log:

1. Ensure `docs/daily/README.md` lists days newest-first, each linking to `YYYY-MM-DD.md` with a one-line summary.
2. In the project root `README.md`, add or update a **Daily activity logs** section immediately after the existing **Changelog** section. Keep the rest of README.md unchanged.
3. That section must link the day's document and the index, newest day first:

```markdown
## Daily activity logs

Session notes for work in this checkout (newest first):

- [YYYY-MM-DD](docs/daily/YYYY-MM-DD.md) — [one-line summary]
- Index: [docs/daily/](docs/daily/README.md)
```

4. If the section already exists, prepend today's bullet if it is new, or refresh that day's one-line summary. Do not duplicate the same date.

## Style

- Prefer facts over narrative. One bullet per change or finding.
- Quote exact commands, file names, config values, and errors when someone would need them to reproduce or resume the work.
- Mark incomplete or failed work as such.
- Do not document every chat turn, tool call, or file open.

## Example

See [example.md](example.md) for tone and density.
