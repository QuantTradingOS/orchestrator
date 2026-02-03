# QuantTradingOS Orchestrator

**Context for commits:** See [CONTEXT.md](CONTEXT.md) for repo identity, remote, and commit rules (no parent QuantTradingOS folder on remote).

Orchestration layer for **QuantTradingOS**: one pipeline (regime → portfolio → execution-discipline → allocation → optional guardian) and a **FastAPI** API with Swagger that exposes the pipeline plus all agents (sentiment, insider, trade-journal, guardian, execution-discipline, portfolio-report). Pass API keys in the request body or via environment variables; no keys are stored in the repo.

## What this repo does

- **CLI:** `python -m orchestrator.run` — Load prices and holdings, run Market-Regime-Agent, Portfolio-Analyst-Agent (real), and Capital-Allocation-Agent. Uses cached execution-discipline score when trades+plan are not provided.
- **API:** `uvicorn orchestrator.api:app` — **POST/GET /decision** (full pipeline), **POST/GET /backtest** (Phase 3: agent-triggered backtest), plus **POST /execution-discipline**, **POST /guardian**, **POST /sentiment-alert**, **POST /insider-report**, **POST /trade-journal**, **POST /portfolio-report**. Swagger at `/docs`.
- **Scheduler:** Run the pipeline on an interval or cron (standalone or with the API).

## Layout

```
orchestrator/
├── __init__.py
├── adapters.py    # Map regime/portfolio outputs → Capital Allocation inputs
├── run.py         # CLI: load prices, run regime, portfolio, (optional) discipline, allocation, (optional) guardian
├── backtest_runner.py  # Phase 3: run qtos-core backtest; data from CSV or data service
├── scheduler.py   # APScheduler: run pipeline on interval or cron (standalone or with API)
├── api.py         # FastAPI app: /decision + /backtest + agent endpoints; Swagger at /docs
├── BACKTEST-PHASE3.md   # Agent-triggered backtest and gating pattern
├── Dockerfile     # Build from workspace root: docker build -f orchestrator/Dockerfile -t qtos-api .
├── docker-compose.yml   # From workspace root: docker-compose -f orchestrator/docker-compose.yml up --build
├── .dockerignore  # Copy to workspace root when building to reduce image size
├── requirements.txt
├── README.md
├── env.example    # Copy to .env and set FINNHUB_API_KEY, OPENAI_API_KEY, optional ORCHESTRATOR_SCHEDULE_*
└── state/         # Created at run time: regime_memory.json, discipline_memory.json, discipline_last_score.json
```

## Getting Started

### Run your first pipeline (CLI)

**Prerequisites:** Python 3.10+, pandas, numpy, yfinance, fastapi, uvicorn, apscheduler. Your workspace must contain `orchestrator/` plus sibling agent repos (Market-Regime-Agent, Portfolio-Analyst-Agent, Capital-Allocation-Agent).

**Steps:**

1. **Set up workspace:** Clone a workspace that includes the orchestrator and required agents, or ensure they're siblings under a common parent directory:
   ```bash
   # Example: clone orchestrator and agents into a workspace
   mkdir QuantTradingOS-workspace && cd QuantTradingOS-workspace
   git clone https://github.com/QuantTradingOS/orchestrator.git
   git clone https://github.com/QuantTradingOS/Market-Regime-Agent.git
   git clone https://github.com/QuantTradingOS/Portfolio-Analyst-Agent.git
   git clone https://github.com/QuantTradingOS/Capital-Allocation-Agent.git
   # ... other agents as needed
   ```

2. **Install dependencies:** From the workspace root (parent of `orchestrator/`):
   ```bash
   pip install -r orchestrator/requirements.txt
   ```

3. **Run the pipeline:**
   ```bash
   python -m orchestrator.run
   ```

   This runs: Market-Regime-Agent → Portfolio-Analyst-Agent → Capital-Allocation-Agent and prints the allocation decision. Uses default paths: `Market-Regime-Agent/data/sample_prices.csv` and `Portfolio-Analyst-Agent/portfolio.csv`.

4. **What happened:** The pipeline loaded prices, detected market regime, computed portfolio metrics, and produced a capital allocation decision (allow_trade, position_size_multiplier, risk_mode, etc.).

5. **Next:** Customize prices/holdings with `--prices` and `--holdings`, or run the API (see below).

### Run your first API (paper run)

**Steps:**

1. **Start the API** (from workspace root):
   ```bash
   uvicorn orchestrator.api:app --host 0.0.0.0 --port 8000
   ```

2. **Open Swagger UI:** http://localhost:8000/docs

3. **Test health:**
   ```bash
   curl http://localhost:8000/health
   ```

4. **Run the pipeline via API:**
   ```bash
   curl -X POST http://localhost:8000/decision -H "Content-Type: application/json" -d '{}'
   ```

   Returns JSON with `regime`, `portfolio`, `decision`, `execution_discipline_score`.

5. **Try agent endpoints:** In Swagger (`/docs`), try `/sentiment-alert`, `/insider-report`, `/trade-journal`, etc. (Note: some require `FINNHUB_API_KEY` and `OPENAI_API_KEY`; see "API keys" section below.)

### Run your first backtest (Phase 3)

**Prerequisites:** qtos-core cloned as sibling of `orchestrator/` (same workspace root). Optional: data-ingestion-service running and `DATA_SERVICE_URL` set for live data.

**Steps:**

1. **From workspace root**, ensure qtos-core is present: `ls qtos-core/examples/data/sample_ohlcv.csv`
2. **Start the API** (if not already): `uvicorn orchestrator.api:app --host 0.0.0.0 --port 8000`
3. **Run backtest with sample data:**
   ```bash
   curl -X POST http://localhost:8000/backtest -H "Content-Type: application/json" -d '{"symbol":"SPY","data_source":"csv","quantity":50}'
   ```
   Returns `metrics` (e.g. total_return_pct, sharpe_ratio, max_drawdown_pct), `num_trades`, `status`.
4. **With data service:** Set `DATA_SERVICE_URL=http://localhost:8001` and use `"data_source":"data_service"` to backtest on data from the data-ingestion-service.

Agents can call `POST /backtest` and gate alerts on metrics (e.g. only alert if `sharpe_ratio > 0.5` and `max_drawdown_pct < 20`). See [BACKTEST-PHASE3.md](BACKTEST-PHASE3.md).

## Run

From the **QuantTradingOS repo root** (parent of `orchestrator/`):

```bash
# Default: uses Market-Regime-Agent/data/sample_prices.csv, Portfolio-Analyst-Agent/portfolio.csv
python -m orchestrator.run

# Custom prices and/or holdings CSV
python -m orchestrator.run --prices path/to/prices.csv --holdings path/to/holdings.csv

# Optional: peak equity for drawdown, config path, state dir
python -m orchestrator.run --peak-equity 500000 --config Capital-Allocation-Agent/config.yaml
```

## Dependencies

- **Python 3.10+**
- **pandas**, **numpy**, **yfinance**, **fastapi**, **uvicorn**, **apscheduler** (see `requirements.txt`)
- Sibling agent dirs on disk (same repo root): **Market-Regime-Agent**, **Capital-Allocation-Agent**, **Portfolio-Analyst-Agent**, **Execution-Discipline-Agent**, **Capital-Guardian-Agent**, **Sentiment-Shift-Alert-Agent**, **Equity-Insider-Intelligence-Agent**, **Trade-Journal-Coach-Agent**. They are imported by adding their paths to `sys.path`; no need to `pip install` each agent.

## Current behavior

1. **Load prices** from CSV (default: `Market-Regime-Agent/data/sample_prices.csv`).
2. **Run Market-Regime-Agent** → get `RegimeDecision` (label, confidence, etc.).
3. **Run Portfolio-Analyst-Agent**: load holdings from CSV (default: `Portfolio-Analyst-Agent/portfolio.csv`), build `prices_df` (from regime CSV if it contains all holding symbols, else fetch via **yfinance**), call `compute_portfolio(holdings_df, prices_df)` → total_value, vol_annual, ret_total, open_positions, etc.
4. **Build allocation input**: regime + portfolio outputs are adapted to Capital-Allocation format. **execution_discipline_score**: if you pass `execution_trades_path` + `execution_plan_path`, the pipeline runs Execution-Discipline-Agent and uses its compliance score; otherwise it uses the **last cached score** (from a previous run of Execution-Discipline via `/decision` or `/execution-discipline`). If no cache exists, default is 0.88.
5. **Run Capital-Allocation-Agent** → get decision (allow_trade, position_size_multiplier, risk_mode, reason, etc.).
6. Print the decision.

## REST API (FastAPI + Swagger)

Run the API from repo root:

```bash
uvicorn orchestrator.api:app --reload --host 0.0.0.0 --port 8000
```

- **Swagger UI:** http://localhost:8000/docs  
- **OpenAPI JSON:** http://localhost:8000/openapi.json  
- **Health:** GET http://localhost:8000/health  

**Pipeline**

- **POST/GET /decision** — Full pipeline: regime → portfolio → [execution-discipline] → allocation → [guardian].  
  Optional body: `prices_path`, `holdings_path`, `peak_equity`, `execution_score`, `execution_trades_path`, `execution_plan_path`, `include_guardian`, `guardian_price`, `guardian_atr`.  
  Returns `{ regime, portfolio, decision, execution_discipline_score [, guardian_guardrails ] }`.

**Agent endpoints**

- **POST /execution-discipline** — Run Execution-Discipline-Agent (trades + plan + regime_label). Returns compliance score and violations.  
- **POST /guardian** — Run Capital-Guardian-Agent (account_size, drawdown_pct, regime, etc.). Returns decision + optional stop and position size.  
- **POST /sentiment-alert** — Run Sentiment-Shift-Alert-Agent (symbol; requires FINNHUB_API_KEY + OPENAI_API_KEY or body keys). Returns sentiment score and explanation.  
- **POST /insider-report** — Run Equity-Insider-Intelligence-Agent (symbol; same keys). Returns LLM report text.  
- **POST /trade-journal** — Run Trade-Journal-Coach-Agent (trades CSV upload or trades_json; OPENAI_API_KEY). Returns metrics + coaching report.  
- **POST /portfolio-report** — Standalone portfolio analytics (holdings_path [, prices_path ]). Returns total_value, vol_annual, holdings, etc.  

**Backtest (Phase 3: agent-integrated backtesting)**

- **POST/GET /backtest** — Run a backtest using qtos-core. Data from CSV (default: qtos-core sample) or from data-ingestion-service (set `DATA_SERVICE_URL`). Body/params: `symbol`, `data_source` (csv | data_service), `csv_path`, `initial_cash`, `quantity`, `strategy_type` (buy_and_hold), `period` (for data_service). Returns `metrics` (initial_value, final_value, total_pnl, total_return_pct, cagr, sharpe_ratio, max_drawdown, max_drawdown_pct), `num_trades`, `status`. Requires **qtos-core** as sibling of orchestrator. Agents can call this to validate a signal before alerting (e.g. only alert if `sharpe_ratio` > threshold). See [BACKTEST-PHASE3.md](BACKTEST-PHASE3.md) for the gating pattern.

No Swagger file is checked in—FastAPI generates OpenAPI at runtime.

**Quick test (API running):**

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/decision -H "Content-Type: application/json" -d '{}'
```

## Scheduler

The pipeline can run on a **timer** (interval or cron) in two ways.

**1. Standalone** (from repo root):

```bash
# Every 60 minutes (config from env or CLI)
python -m orchestrator.scheduler --minutes 60

# Cron: 9:00 weekdays (same as ORCHESTRATOR_SCHEDULE_CRON)
python -m orchestrator.scheduler --cron "0 9 * * 1-5"

# Config from .env: ORCHESTRATOR_SCHEDULE_MINUTES=60 or ORCHESTRATOR_SCHEDULE_CRON=0 9 * * 1-5
python -m orchestrator.scheduler
```

**2. With the API** (scheduler runs in the same process as uvicorn):

Set one of these in the environment before starting the API:

- `ORCHESTRATOR_SCHEDULE_MINUTES=60` — run pipeline every 60 minutes
- `ORCHESTRATOR_SCHEDULE_CRON=0 9 * * 1-5` — run at 9:00 on weekdays (cron expression)

Example:

```bash
ORCHESTRATOR_SCHEDULE_MINUTES=60 uvicorn orchestrator.api:app --host 0.0.0.0 --port 8000
```

The scheduler uses **default paths** (same as `python -m orchestrator.run`): Market-Regime-Agent data, Portfolio-Analyst-Agent portfolio, Capital-Allocation-Agent config. No execution-discipline or guardian in the scheduled run unless you extend `scheduler._run_scheduled_pipeline`.

## Running with Docker

You can build and run the API in a container. The **build context must be the workspace root** (the directory that contains `orchestrator/` and the sibling agent repos).

**Prerequisites:** Docker (and Docker Compose). Your workspace must contain `orchestrator/` plus the agent repos the pipeline needs (e.g. Market-Regime-Agent, Portfolio-Analyst-Agent, Capital-Allocation-Agent).

**Steps:**

1. **Optional:** Copy `orchestrator/.dockerignore` to your workspace root as `.dockerignore` to keep the image smaller (excludes `.git`, `.github-repo`, etc.).
2. **Optional:** Copy `orchestrator/env.example` to `.env` in the workspace root and set `FINNHUB_API_KEY`, `OPENAI_API_KEY` if you use sentiment/insider/trade-journal endpoints.
3. From the **workspace root** (parent of `orchestrator/`):
   ```bash
   docker-compose -f orchestrator/docker-compose.yml up --build
   ```
   Or build and run without Compose:
   ```bash
   docker build -f orchestrator/Dockerfile -t qtos-api .
   docker run -p 8000:8000 --env-file .env qtos-api
   ```
4. API: **http://localhost:8000** — Swagger at **http://localhost:8000/docs**. Health: `curl http://localhost:8000/health`.

State (e.g. `orchestrator/state/`) is persisted via the volume in `docker-compose.yml` so it survives container restarts.

### API keys (Finnhub + OpenAI)

Endpoints that need external APIs (**sentiment-alert**, **insider-report**, **trade-journal**) require:

| Key | Used by | Get one at |
|-----|--------|------------|
| **FINNHUB_API_KEY** | sentiment-alert, insider-report | [Finnhub](https://finnhub.io/docs/api) |
| **OPENAI_API_KEY** | sentiment-alert, insider-report, trade-journal | [OpenAI](https://platform.openai.com/api-keys) |

**Never commit real keys to GitHub.** Users can send keys in either of these ways:

1. **Request body** — Include `finnhub_key` and/or `openai_key` in the JSON body for each call. Example:
   ```json
   { "symbol": "AAPL", "finnhub_key": "your_finnhub_key", "openai_key": "your_openai_key" }
   ```
2. **Environment variables** — Set `FINNHUB_API_KEY` and `OPENAI_API_KEY` in the environment where the API runs. For local dev: copy `orchestrator/env.example` to `orchestrator/.env`, fill in your keys, and load them (e.g. `export $(grep -v '^#' .env | xargs)` before starting the server). The `.env` file is in `.gitignore` and will not be committed.

Swagger UI at `/docs` shows the optional body fields for each endpoint so users can pass keys per request.

## Execution-discipline score and cache

- **Real score:** Pass `execution_trades_path` and `execution_plan_path` to `/decision` (or call `POST /execution-discipline`) to run Execution-Discipline-Agent; the compliance score is used and **cached** in `state/discipline_last_score.json`.
- **Cached score:** When `/decision` is called without trades+plan, the pipeline uses the last cached score if present; otherwise it uses 0.88. So run Execution-Discipline (or `/execution-discipline`) periodically with your latest trades; subsequent `/decision` calls use that score until the next run.

## Next steps

- Optionally expose **OpenAPI JSON** as a static file for external tooling.
