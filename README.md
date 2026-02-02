# QuantTradingOS Orchestrator

Single pipeline: **regime → portfolio (mock) → allocation**. Runs Market-Regime-Agent and Capital-Allocation-Agent in sequence and produces one regime-aware allocation decision.

## Layout

```
orchestrator/
├── __init__.py
├── adapters.py    # Map regime/portfolio outputs → Capital Allocation inputs
├── run.py         # CLI entrypoint: load prices, run regime, run portfolio, run allocation
├── api.py         # FastAPI app: POST/GET /decision, auto Swagger at /docs
├── requirements.txt
├── README.md
├── env.example    # Copy to .env and set FINNHUB_API_KEY, OPENAI_API_KEY (never commit .env)
└── state/         # Created at run time (regime memory)
```

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
- **pandas**, **numpy**, **yfinance** (see `requirements.txt`; yfinance used when portfolio symbols are not in the prices CSV)
- Sibling agents on disk: **Market-Regime-Agent**, **Capital-Allocation-Agent**, **Portfolio-Analyst-Agent** (imported by adding their paths to `sys.path`).

## Current behavior

1. **Load prices** from CSV (default: `Market-Regime-Agent/data/sample_prices.csv`).
2. **Run Market-Regime-Agent** → get `RegimeDecision` (label, confidence, etc.).
3. **Run Portfolio-Analyst-Agent**: load holdings from CSV (default: `Portfolio-Analyst-Agent/portfolio.csv`), build `prices_df` (from regime CSV if it contains all holding symbols, else fetch via **yfinance**), call `compute_portfolio(holdings_df, prices_df)` → total_value, vol_annual, ret_total, open_positions, etc.
4. **Build allocation input**: regime + portfolio outputs are adapted to Capital-Allocation format; **execution_discipline_score** is still fixed (0.88). TODO: wire Execution-Discipline-Agent.
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

No Swagger file is checked in—FastAPI generates OpenAPI at runtime.

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

## Next steps

- Wire **Execution-Discipline-Agent** for real execution_discipline_score (or cache last score).
- Add **scheduling** (cron or APScheduler) to run the pipeline on a timer.
