# Context for commits (orchestrator)

**Read this when committing so the repo and layout stay correct.**

## Repo identity

- **This repository:** [QuantTradingOS/orchestrator](https://github.com/QuantTradingOS/orchestrator) on GitHub.
- **Remote:** `origin` → `https://github.com/QuantTradingOS/orchestrator.git`
- **Branch:** `main` (push to `origin main`).

Orchestrator is a **standalone repo** in the QuantTradingOS organization. It is **not** inside a parent "QuantTradingOS" repo on the remote.

## Commit and push rules

1. **Commit from the orchestrator directory.**  
   Run `git` from `orchestrator/` (this folder). Do not commit orchestrator as a subfolder of a parent repo that would create a "QuantTradingOS" folder on the remote.

2. **Remote root = orchestrator contents.**  
   The remote repo root should be the contents of this folder (e.g. `api.py`, `run.py`, `README.md`). There must be **no parent folder named "QuantTradingOS"** on the remote.

3. **Push:**  
   `git push origin main` from inside `orchestrator/`.

## On-disk layout

- On your machine, `orchestrator/` usually sits next to sibling agent dirs (e.g. `Market-Regime-Agent/`, `Capital-Allocation-Agent/`, `Portfolio-Analyst-Agent/`) under a common parent directory.
- Those siblings are **separate** GitHub repos under the QuantTradingOS org; each has its own `.git` and remote.
- The CLI and API expect to be run from that **parent** directory (the "repo root" for the multi-repo layout), so `python -m orchestrator.run` and `uvicorn orchestrator.api:app` are run from the parent of `orchestrator/`.

## Summary

| What | Where |
|------|--------|
| This repo on GitHub | **QuantTradingOS/orchestrator** (no parent QuantTradingOS folder) |
| Where to run `git` | Inside **orchestrator/** |
| Where to run CLI/API | Parent of **orchestrator/** |
