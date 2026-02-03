"""
Run qtos-core backtest and return metrics. Used by POST /backtest.

Expects qtos-core as sibling of orchestrator (workspace root). Adds qtos-core to sys.path
so backtesting and qtos_core can be imported. Data can come from CSV path or from
data-ingestion-service (DATA_SERVICE_URL).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("orchestrator.backtest_runner")


def _ensure_qtos_core_path() -> None:
    """Add qtos-core to sys.path so backtesting and qtos_core can be imported."""
    import sys
    qtos_core_root = ROOT / "qtos-core"
    if not qtos_core_root.exists():
        raise ImportError("qtos-core not found as sibling of orchestrator. Clone qtos-core into the workspace.")
    if str(qtos_core_root) not in sys.path:
        sys.path.insert(0, str(qtos_core_root))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def load_ohlcv_from_csv(csv_path: Path, symbol: str) -> Any:
    """Load OHLCV DataFrame from CSV. Returns pandas DataFrame with datetime index."""
    _ensure_qtos_core_path()
    from backtesting.data_loader import load_csv
    return load_csv(csv_path, symbol=symbol)


def load_ohlcv_from_data_service(symbol: str, period: str = "1y") -> Any:
    """Load OHLCV from data-ingestion-service. Returns pandas DataFrame with datetime index."""
    import pandas as pd
    import requests
    from datetime import datetime, timedelta
    base = os.environ.get("DATA_SERVICE_URL", "").rstrip("/")
    if not base:
        raise ValueError("DATA_SERVICE_URL not set; cannot fetch data from data service")
    # Map period to end_date and start_date (data service uses start_date, end_date, limit)
    end_date = datetime.utcnow()
    if period == "1y" or period == "1Y":
        start_date = end_date - timedelta(days=365)
    elif period == "6mo":
        start_date = end_date - timedelta(days=180)
    elif period == "2y":
        start_date = end_date - timedelta(days=730)
    else:
        start_date = end_date - timedelta(days=365)
    url = f"{base}/prices/{symbol}"
    params = {
        "start_date": start_date.isoformat() + "Z",
        "end_date": end_date.isoformat() + "Z",
        "limit": 10000,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"No price data returned for {symbol}")
    # API returns list of {symbol, timestamp, open, high, low, close, volume}
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns=lambda c: c.lower())
    df.attrs["symbol"] = symbol
    return df


def run_backtest(
    data: Any,
    symbol: str,
    initial_cash: float = 100_000.0,
    quantity: float = 50.0,
    strategy_type: str = "buy_and_hold",
) -> dict[str, Any]:
    """
    Run backtest using qtos-core. Returns dict with metrics and summary.

    Parameters
    ----------
    data : DataFrame
        OHLCV DataFrame with datetime index (from load_ohlcv_from_csv or load_ohlcv_from_data_service).
    symbol : str
        Symbol for the strategy.
    initial_cash : float
        Starting portfolio value.
    quantity : float
        Shares to buy (for buy_and_hold).
    strategy_type : str
        Currently only "buy_and_hold" is supported.

    Returns
    -------
    dict
        metrics (initial_value, final_value, total_pnl, total_return_pct, cagr, sharpe_ratio,
        max_drawdown, max_drawdown_pct), num_trades, symbol, strategy_type, status.
    """
    _ensure_qtos_core_path()
    from backtesting import BacktestEngine, compute_metrics
    from backtesting.engine import PassThroughRiskManager
    from qtos_core import Portfolio
    from qtos_core.examples.buy_and_hold import BuyAndHoldStrategy

    if strategy_type != "buy_and_hold":
        raise ValueError(f"Unsupported strategy_type: {strategy_type}. Only buy_and_hold is supported.")

    portfolio = Portfolio(cash=initial_cash)
    strategy = BuyAndHoldStrategy(symbol=symbol, quantity=quantity)
    risk_manager = PassThroughRiskManager()
    engine = BacktestEngine(
        strategy=strategy,
        risk_manager=risk_manager,
        portfolio=portfolio,
        advisors=(),
        validators=(),
        observers=(),
    )
    result = engine.run(data, symbol=symbol)
    metrics = compute_metrics(initial_cash, result.equity_curve)

    return {
        "status": "ok",
        "symbol": symbol,
        "strategy_type": strategy_type,
        "num_trades": len(result.trades),
        "metrics": {
            "initial_value": metrics.initial_value,
            "final_value": metrics.final_value,
            "total_pnl": metrics.total_pnl,
            "total_return_pct": metrics.total_return_pct,
            "cagr": metrics.cagr,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "max_drawdown_pct": metrics.max_drawdown_pct,
        },
    }


def run_backtest_from_request(
    csv_path: Path | None = None,
    symbol: str = "SPY",
    data_source: str = "csv",
    initial_cash: float = 100_000.0,
    quantity: float = 50.0,
    strategy_type: str = "buy_and_hold",
    period: str = "1y",
) -> dict[str, Any]:
    """
    Load data (from CSV or data service), run backtest, return result dict.

    Raises
    ------
    FileNotFoundError
        If data_source is csv and csv_path is missing or file not found.
    ValueError
        If data_source is data_service and DATA_SERVICE_URL is not set, or data load fails.
    """
    if data_source == "data_service":
        data = load_ohlcv_from_data_service(symbol, period=period)
    else:
        if not csv_path or not csv_path.exists():
            # Default sample data in qtos-core
            default_csv = ROOT / "qtos-core" / "examples" / "data" / "sample_ohlcv.csv"
            if default_csv.exists():
                csv_path = default_csv
            else:
                raise FileNotFoundError(f"CSV not found: {csv_path}")
        data = load_ohlcv_from_csv(csv_path, symbol)

    return run_backtest(
        data=data,
        symbol=symbol,
        initial_cash=initial_cash,
        quantity=quantity,
        strategy_type=strategy_type,
    )
