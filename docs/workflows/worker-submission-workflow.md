# Worker: Autonomous Submission Workflow

End-to-end workflow for taking a topic idea and producing a complete, reviewed Claw4S submission autonomously — from research through implementation and rubric-scored review iteration.

This is a **framework, not a script**. Each topic needs different data sources, methodologies, and design decisions. The worker adapts the framework based on what it discovers during research.

---

## Phase 1: Design (Sub-agent reviewed)

1. **Research the topic** — Dispatch a research sub-agent (opus) to:
   - Search for existing literature, key papers, open questions
   - Identify what public data/artifacts are available (no GPU, no auth, runs in minutes)
   - Assess novelty and feasibility for Claw4S constraints

2. **Draft the design spec** — Write to `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`:
   - Scientific question and thesis
   - Data sources (with exact citations — hardcode published values, no downloads preferred)
   - Analysis methodology
   - Module structure (follow existing submission pattern: data.py, src/, tests/, run.py, validate.py)
   - Key figures
   - Dependencies (keep minimal — numpy/scipy/matplotlib/pytest)
   - Validation criteria
   - How to Extend section

3. **Spec review loop** — Dispatch opus sub-agent to review the spec for:
   - Data availability (are exact values actually published? verify URLs/tables)
   - Methodological soundness
   - Claw4S feasibility (runtime, deps, no auth)
   - Module structure appropriateness
   - Fix all issues, re-review until APPROVED

## Phase 2: Planning (Sub-agent reviewed)

4. **Write implementation plan** — Use superpowers:writing-plans skill:
   - Follow existing submission patterns (tokenizer-analysis)
   - Bite-sized TDD tasks (test first → verify fail → implement → verify pass → commit)
   - Complete code in plan (not "add validation")
   - Exact file paths and commands

5. **Plan review loop** — Dispatch opus sub-agent to review plan:
   - Spec coverage (every spec feature mapped to a task)
   - Task ordering (dependencies respected)
   - TDD compliance
   - Code completeness
   - Fix all issues, re-review until APPROVED

## Phase 3: Implementation (Sub-agent driven)

6. **Set up workspace** — Clone repo or create worktree on a new branch (`feat/<topic>`)

7. **Execute tasks via sub-agents** — Use subagent-driven-development:
   - Dispatch one sub-agent per task (sonnet for mechanical tasks, opus for complex ones)
   - Verify each commit actually landed (sub-agents can claim DONE without doing work — always `git log` to verify)
   - Run full test suite after each task to catch regressions
   - For simple scaffolding/data tasks: skip formal review, verify via tests
   - For complex tasks (fitting, analysis): dispatch spec + code quality reviewers
   - Parallelize independent tasks (e.g., plots + report)

   **Model selection heuristic:**
   - Scaffolding, data entry, config: sonnet
   - Mathematical functions, fitting, analysis pipelines: opus
   - Plots, reports, SKILL.md: sonnet
   - Research note (LaTeX): opus

8. **Key implementation patterns to follow:**
   - `run.py`: thin orchestrator (~10 lines), adds `os.makedirs`, calls analysis/plots/report
   - `validate.py`: accumulates `errors = []`, prints checks, `sys.exit(1)` if errors
   - `conftest.py`: `sys.path.insert(0, os.path.dirname(__file__))`
   - Use `/opt/homebrew/bin/python3.13` for venv (system python3 is 3.9)
   - Working-directory guard in run.py: `assert os.path.exists("src/data.py")`
   - All data hardcoded in `src/data.py` with inline citations
   - Seed pinned to 42, passed explicitly everywhere
   - Error handling: specific exception types + traceback logging, never bare `except Exception: pass`

## Phase 4: Review & Iterate (Parallel sub-agents)

9. **Dispatch 4 parallel review agents:**

   a. **Claw4S Rubric Scorer** (opus) — Score each of the 5 criteria 1-10:
      - Executability (25%): Can Claw run it cold?
      - Reproducibility (25%): Pinned deps, deterministic, cited data?
      - Scientific Rigor (20%): Sound methodology, appropriate caveats?
      - Generalizability (15%): Easy to extend, modular?
      - Clarity for Agents (15%): Unambiguous SKILL.md, expected outputs?

   b. **Code Quality Reviewer** (opus, superpowers:code-reviewer) — Clean code, test coverage, maintainability

   c. **Silent Failure Hunter** (opus, pr-review-toolkit:silent-failure-hunter) — Swallowed exceptions, overly broad catches, missing error messages

   d. **SKILL.md Agent Clarity** (opus) — Simulate executing SKILL.md step by step, flag ambiguity

10. **Consolidate findings** — Group issues by:
    - Must-fix (affects scoring or correctness)
    - Should-fix (code quality)
    - Nice-to-have

11. **Fix all must-fix and should-fix items** — Dispatch opus fix agent with all issues in one batch

12. **Re-review** — Dispatch rubric scorer again to verify score improved. Target: weighted ≥ 8.5

13. **Iterate** — If score < 8.5 or must-fix items remain, fix and re-review (max 3 iterations)

## Phase 5: Completion

14. **Final E2E cold-start** — Delete .venv and results, follow SKILL.md from scratch
15. **Do NOT submit to clawRxiv** unless user explicitly asks
16. **Report final state** — Branch name, commit count, test count, weighted score, key scientific findings

## Key Lessons Learned

- **Sub-agents can claim DONE without doing work** — always verify commits with `git log` and test with `pytest`
- **Data availability is the #1 blocker** — verify exact published values exist before designing around them. Pythia training losses are NOT published as exact values (only wandb). Cerebras-GPT IS the gold standard for published loss + benchmark data.
- **CPU inference is too slow for Claw4S** — never download and run transformer models. Hardcode published data instead.
- **n=7 data points require statistical humility** — use parametric bootstrap (not nonparametric), adjusted R² (not R²), cautious language ("no significant evidence" not "confirms"), and explicit degree-of-freedom caveats for 5-parameter models
- **The review cycle catches real issues** — the first implementation scored 8.2; after fixing review findings it scored 8.8. The silent-failure hunter found 3 critical patterns.
- **Parallelize where possible** — research agents, independent tasks (plots + report), and all 4 review agents can run simultaneously
