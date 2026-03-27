# Supervisor: Parallel Submission Management

How to manage multiple parallel worker agents building Claw4S submissions — launch, monitor, review, iterate, and report. The "manager" role that orchestrates autonomous workers.

This is a **coordination framework**, not a recipe. The supervisor makes judgment calls about batching, viability, and when to escalate.

---

## Setup Phase

1. **Assess viability** of each topic before launching:
   - High viability: can use published/hardcoded data only (no model downloads)
   - Medium viability: needs small CPU training or small dataset downloads
   - Needs redesign: original idea requires model weights → must pivot to published data
   - Batch topics by viability tier (3 at a time is practical)

2. **Create git worktrees** from the cloned repo:
   ```bash
   mkdir -p .worktrees
   git worktree add .worktrees/<topic-name> -b feat/<topic-name> main
   ```
   - Each worker gets its own worktree and branch
   - All branch from `main` (not from each other)
   - Worktrees share `.git` — concurrent commits can cause lock contention, but in practice it works with 3-4 agents

3. **Launch workers in a SINGLE message** — 3 parallel `Agent` calls:
   - `model: opus` (these are complex, multi-phase tasks)
   - `run_in_background: true` (don't block the conversation)
   - Each gets a comprehensive prompt with:
     - The topic and scientific question
     - Working directory path (the specific worktree)
     - The full 5-phase workflow (see worker-submission-workflow.md)
     - Reference patterns from existing submissions
     - Python version note (`/opt/homebrew/bin/python3.13` for venv)
     - All critical rules (no GPU, no model downloads, pin deps, etc.)
     - What to report when done

## Monitoring Phase

4. **Set up recurring monitoring** with CronCreate (every 10 minutes):
   ```
   CronCreate: */10 * * * *, prompt: "Check progress of parallel agents..."
   ```

5. **What to check at each interval:**
   - `git -C .worktrees/<name> log --oneline -5` — new commits = progress
   - `ls .worktrees/<name>/submissions/*/src/*.py` — source files appearing
   - Compare commit count to previous check — no change in 20+ min = potential stall

6. **Detecting problems:**
   - **No commits after 20+ min:** Check the agent's output file for errors or blocks
   - **Agent completed but no submission files:** Agent may have worked in wrong directory or failed silently
   - **Git lock contention:** If an agent reports git errors, wait and retry — worktrees share `.git`

7. **Report to user** with a concise status table:
   ```
   | Agent | Phase | Commits | Modules | Status |
   |-------|-------|---------|---------|--------|
   ```

## Completion & Review Phase

8. **When a worker completes**, verify (don't trust the report):
   - `git log --oneline feat/<name> --not main` — all commits present
   - `cd .worktrees/<name>/submissions/<name> && .venv/bin/python -m pytest tests/ -v` — tests pass
   - `.venv/bin/python validate.py` — validation passes
   - If any of these fail, dispatch a fix agent to the worktree

9. **Dispatch parallel rubric reviews** (same pattern as scaling-laws):
   - 4 agents in parallel: Rubric Scorer, Code Quality, Silent Failure Hunter, SKILL.md Clarity
   - Each reads from the specific worktree path
   - Consolidate findings into must-fix / should-fix / nice-to-have

10. **Fix and iterate:**
    - Dispatch a fix agent to each worktree with consolidated issues
    - Re-review with rubric scorer
    - Target: weighted score ≥ 8.5
    - Max 3 iteration cycles per submission

11. **Final report to user:**
    ```
    | Submission | Branch | Tests | Score | Key Finding |
    |------------|--------|-------|-------|-------------|
    ```

## Key Supervisor Principles

- **Never do the workers' work** — if a worker fails, dispatch a new agent to fix it, don't fix it inline (preserves context separation)
- **Verify, don't trust** — workers can claim DONE without actually completing. Always `git log` and `pytest` to verify.
- **Batch by viability** — don't launch topics that need redesign alongside ready-to-go topics
- **3 parallel is practical** — more than 3 increases git lock contention and makes monitoring harder
- **Report progress without being asked** — the cron job handles this, but also report at natural milestones (all specs done, first worker completes, all workers complete)
- **Escalate genuinely blocked workers** — if a worker is stuck on a design decision (e.g., "data doesn't exist"), surface to the user rather than guessing

## Lifecycle of a Batch

```
User: "Run batch 1 (topics 3, 9, 10)"
  ↓
Supervisor: Create 3 worktrees
  ↓
Supervisor: Launch 3 background agents (single message)
  ↓
Supervisor: Set up 10-min cron monitoring
  ↓
[Agents work autonomously: research → design → plan → implement → self-review]
  ↓
Supervisor: Report progress at each cron tick
  ↓
Agent completes → Supervisor verifies (git log, pytest, validate.py)
  ↓
Supervisor: Dispatch 4 parallel review agents per completed submission
  ↓
Supervisor: Consolidate review findings, dispatch fix agents
  ↓
Supervisor: Re-review until score ≥ 8.5
  ↓
Supervisor: Report final status to user
  ↓
User: "Run batch 2 (topics 2, 4)"
```
