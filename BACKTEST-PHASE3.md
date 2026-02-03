# Phase 3: Agent-Triggered Backtesting

The orchestrator exposes **POST/GET /backtest** so agents (or any client) can run a backtest using qtos-core and use the results to gate decisions—e.g. "only alert if this signal would have been profitable in a backtest."

## What’s implemented

- **Backtest API:** `POST /backtest` and `GET /backtest` with symbol, data source (CSV or data-ingestion-service), initial_cash, quantity, strategy_type (currently `buy_and_hold`), and optional period for data_service.
- **Runner:** `orchestrator.backtest_runner` loads OHLCV from a CSV path or from the data-ingestion-service (`DATA_SERVICE_URL`), runs qtos-core’s `BacktestEngine` with `BuyAndHoldStrategy`, and returns metrics (PnL, CAGR, Sharpe, max drawdown, etc.) and num_trades.
- **Requirements:** qtos-core must be present as a **sibling** of the orchestrator (same workspace root). Optional: data-ingestion-service for live data.

## Agent gating pattern

1. **Agent produces a signal** (e.g. sentiment shift, insider activity).
2. **Agent (or orchestrator) calls** `POST /backtest` with the relevant symbol and parameters (e.g. quantity, period).
3. **Client checks metrics** in the response: `sharpe_ratio`, `max_drawdown_pct`, `total_return_pct`, etc.
4. **Gate the alert:** Only notify or act if metrics pass thresholds (e.g. `sharpe_ratio > 0.5` and `max_drawdown_pct < 20`).

Example (pseudo-code in an agent or wrapper):

```python
response = requests.post(
    "http://localhost:8000/backtest",
    json={"symbol": "AAPL", "data_source": "data_service", "period": "1y", "quantity": 10},
)
data = response.json()
metrics = data.get("metrics", {})
if metrics.get("sharpe_ratio", 0) > 0.5 and metrics.get("max_drawdown_pct", 100) < 20:
    send_alert("Signal passed backtest", data)
else:
    log("Signal did not pass backtest; no alert")
```

## Data sources

- **csv** — Use `csv_path` (or default qtos-core sample at `qtos-core/examples/data/sample_ohlcv.csv`). Good for demos and CI.
- **data_service** — Use data-ingestion-service. Set `DATA_SERVICE_URL` (e.g. `http://localhost:8001`). Requires the data service to have ingested prices for the symbol.

## Extending

- **More strategies:** Extend `backtest_runner.run_backtest()` to accept other `strategy_type` values and instantiate the corresponding qtos-core strategy (or wrap VectorBT/Backtrader later).
- **Advisors/validators/observers:** The qtos-core `BacktestEngine` already supports advisors, validators, and observers. The runner can be extended to accept optional agent hooks (e.g. regime filter, guardian validator) and pass them into the engine.
- **Richer metrics:** The API already returns standard metrics; add custom metrics in the runner or in the client.

## References

- qtos-core backtesting: [qtos-core README](https://github.com/QuantTradingOS/qtos-core) and `backtesting/` module.
- Orchestrator API: Swagger at `http://localhost:8000/docs` when the API is running.
