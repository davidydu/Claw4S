# Claw4S Submission Project

## Project Overview

This repo contains submissions to the **Claw4S 2026 conference** (Stanford + Princeton AI4Science Catalyst Institute). Each submission is an executable "skill" — a SKILL.md that an AI agent can run to reproduce scientific results — plus a 1-4 page LaTeX research note.

- **Deadline:** April 5, 2026
- **Agent name:** `the-mad-lobster`
- **Human authors:** Yun Du, Lina Ji
- **Website:** https://claw4s.github.io/
- **Submission platform:** clawRxiv (http://18.118.210.52)

## Repo Structure

```
Claw4S/
├── CLAUDE.md                          # This file — shared project rules
├── README.md
├── .gitignore
├── .claude/
│   ├── skills/                        # Shared skills (usable across all submissions)
│   │   ├── clawrxiv.md                # clawRxiv API reference
│   │   ├── claw4s-submit.md           # Submission format, criteria, checklist
│   │   ├── claw4s-review-iterate.md   # Review & iteration workflow
│   │   └── claw4s-end-to-end.md       # Full brainstorm-to-submit workflow
│   └── memory/                        # Shared memory
├── docs/superpowers/plans/            # Implementation plans
└── submissions/                       # One folder per submission
    ├── tokenizer-analysis/            # Submission 1 (clawRxiv #101)
    │   ├── SKILL.md                   # Executable skill (primary deliverable)
    │   ├── run.py                     # Analysis runner
    │   ├── validate.py                # Results validator
    │   ├── requirements.txt           # Pinned deps (per-submission)
    │   ├── conftest.py                # Pytest config
    │   ├── src/                       # Source modules
    │   ├── tests/                     # Unit tests
    │   ├── research_note/             # LaTeX paper
    │   ├── .venv/                     # Virtual env (gitignored, per-submission)
    │   └── results/                   # Generated output (gitignored)
    └── <next-submission>/             # Future submissions follow same structure
```

Each submission is **self-contained** — its own venv, requirements, source, tests, and SKILL.md. All commands in a SKILL.md run from that submission's directory.

## Evaluation Criteria

Every submission is judged on these 5 criteria. Optimize in order of weight:

| Criterion | Weight | Key Question | What Scores High |
|-----------|--------|-------------|------------------|
| **Executability** | 25% | Can Claw run it start to finish? | All commands work, deps install, no auth needed, no ambiguity |
| **Reproducibility** | 25% | Same results independently? | Pinned versions (`==`), pinned dataset/model revisions, deterministic |
| **Scientific Rigor** | 20% | Sound methodology? | Well-defined metrics, statistical variance reported, limitations acknowledged |
| **Generalizability** | 15% | Adaptable to other domains? | "How to Extend" section, modular code, configurable parameters |
| **Clarity for Agents** | 15% | Written clearly for AI? | Expected outputs per step, no fragile `-c` quoting, prerequisites stated |

## Critical Rules

1. **Always populate `skill_md`** when submitting via clawRxiv API — 50% of eval weight depends on it.
2. **Never use `source .venv/bin/activate`** in SKILL.md — use `.venv/bin/python` directly.
3. **Pin everything** — dependency versions (`==`), dataset revisions, model revisions.
4. **Never use multiline `python -c`** in SKILL.md — create `run.py` / `validate.py` scripts.
5. **Claw must be co-author** on every submission.
6. **Test with a fresh agent** before submitting — delete `.venv/` and `results/`, follow SKILL.md cold.
7. **Each submission is self-contained** — all paths in SKILL.md are relative to the submission folder.

## Available Skills (.claude/skills/)

| Skill | When to Use |
|-------|-------------|
| `clawrxiv` | Register agent, publish paper, browse/vote on clawRxiv |
| `claw4s-submit` | Check submission requirements, format, prizes, checklist |
| `claw4s-review-iterate` | Evaluate submission against 5 criteria, prioritize fixes, iterate |
| `claw4s-end-to-end` | Full workflow from brainstorm to submission (7 phases) |

## Creating a New Submission

1. Create folder: `submissions/<submission-name>/`
2. Follow the `claw4s-end-to-end` skill (7 phases)
3. Each submission gets its own: SKILL.md, run.py, validate.py, requirements.txt, src/, tests/, research_note/
4. Use the `claw4s-review-iterate` skill before submitting
5. Submit via the `clawrxiv` skill with `skill_md` populated

## Current Submissions

| # | Folder | Title | clawRxiv | Status |
|---|--------|-------|----------|--------|
| 1 | `submissions/tokenizer-analysis/` | Cross-Lingual Tokenizer Equity | #101 | Submitted |
| 2 | `submissions/scaling-laws/` | (Approach A: Scaling Laws) | — | Planned |

## Submission Checklist

- [ ] SKILL.md with executable steps (`.venv/bin/python`, not `source activate`)
- [ ] `run.py` and `validate.py` scripts (no fragile `-c` commands)
- [ ] `requirements.txt` with pinned versions (`==`)
- [ ] Dataset and model revisions pinned in source code
- [ ] Unit tests passing
- [ ] Research note (1-4 pages LaTeX) with Claw as co-author
- [ ] E2E test from clean state passes (from submission directory)
- [ ] Review-iterate cycle completed (target: weighted score >= 8.0)
- [ ] Published to clawRxiv with `skill_md` field populated
- [ ] `human_names` includes all human authors
