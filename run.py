"""
Orchestrator entrypoint: run regime → portfolio → allocation in one shot.

Usage (from QuantTradingOS repo root):
  python -m orchestrator.run
  python -m orchestrator.run --prices path/to/prices.csv --holdings path/to/holdings.csv

Expects sibling agent dirs: Market-Regime-Agent, Capital-Allocation-Agent, Portfolio-Analyst-Agent.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

DISCIPLINE_CACHE_FILENAME = "discipline_last_score.json"

# Repo root = parent of orchestrator/
ROOT = Path(__file__).resolve().parent.parent

# Add agent roots to path so we can import them
def _setup_paths(extra: list[str] | None = None) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    names = ["Capital-Allocation-Agent", "Market-Regime-Agent", "Portfolio-Analyst-Agent"]
    if extra:
        names = list(names) + list(extra)
    for name in names:
        agent_root = ROOT / name
        if agent_root.exists() and str(agent_root) not in sys.path:
            sys.path.insert(0, str(agent_root))


def load_prices(csv_path: Path) -> pd.DataFrame:
    """Load wide-format CSV with date column and ticker columns (e.g. SPY, QQQ, TLT)."""
    df = pd.read_csv(csv_path)
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def prices_for_portfolio(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert regime-style prices (date column + ticker columns) to index-by-date for compute_portfolio."""
    out = prices_wide.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.set_index("date").sort_index()
    return out


def run_portfolio(holdings_path: Path, prices_df: pd.DataFrame) -> dict:
    """
    Run Portfolio-Analyst-Agent compute_portfolio(holdings_df, prices_df).
    If holdings symbols are not all in prices_df, fetch portfolio prices via yfinance.
    """
    _setup_paths()
    from app import compute_portfolio, fetch_prices  # Portfolio-Analyst-Agent

    holdings_df = pd.read_csv(holdings_path)
    if "symbol" not in holdings_df.columns or "shares" not in holdings_df.columns:
        raise ValueError("Holdings CSV must have columns: symbol, shares (optionally avg_price)")
    symbols = holdings_df["symbol"].tolist()
    missing = [s for s in symbols if s not in prices_df.columns]
    if missing:
        # Fetch portfolio symbols' prices (e.g. when regime CSV has SPY/QQQ/TLT but portfolio has AAPL/NVDA)
        prices_df = fetch_prices(symbols)
        if prices_df.empty:
            raise ValueError("Could not fetch prices for portfolio symbols")
    else:
        # Use subset of regime prices for portfolio symbols only
        prices_df = prices_df[[c for c in symbols if c in prices_df.columns]].copy()
        if prices_df.empty:
            prices_df = fetch_prices(symbols)

    result = compute_portfolio(holdings_df, prices_df)
    # Build dict expected by adapters
    holdings = result["holdings"]
    total_value = result["total_value"]
    vol_annual = result.get("vol_annual") or 0.0
    ret_total = result.get("ret_total") or 0.0
    return {
        "total_value": total_value,
        "vol_annual": vol_annual,
        "ret_total": ret_total,
        "holdings": holdings,
        "open_positions": len(holdings),
        "total_exposure": total_value,
        "portfolio_heat": min(1.0, vol_annual * 0.6) if vol_annual and not pd.isna(vol_annual) else 0.0,
    }


def run_regime(prices: pd.DataFrame, state_dir: Path):
    """Run Market-Regime-Agent; return RegimeDecision as dict."""
    _setup_paths()
    from src.agent import MarketRegimeAgent  # Market-Regime-Agent

    memory_path = str(state_dir / "regime_memory.json")
    state_dir.mkdir(parents=True, exist_ok=True)
    agent = MarketRegimeAgent(memory_path=memory_path)
    decision, action = agent.run(prices)
    return dataclasses.asdict(decision)


def run_allocation(inputs: dict, config_path: Path) -> dict:
    """Run Capital-Allocation-Agent; return decision dict."""
    _setup_paths()
    from agent import CapitalAllocationAgent  # Capital-Allocation-Agent

    agent = CapitalAllocationAgent(config_path=str(config_path))
    return agent.decide(inputs)


def get_cached_discipline_score(state_dir: Path) -> float | None:
    """Return last cached compliance score (0..1) if present, else None."""
    path = state_dir / DISCIPLINE_CACHE_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return float(data.get("compliance_score"))
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def set_cached_discipline_score(state_dir: Path, score: float) -> None:
    """Persist compliance score so pipeline can use it when trades+plan not provided."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / DISCIPLINE_CACHE_FILENAME
    path.write_text(json.dumps({
        "compliance_score": score,
        "asof": datetime.utcnow().isoformat() + "Z",
    }, indent=2))


def _run_execution_discipline(trades_path: Path, plan_path: Path, regime_label: str, state_dir: Path) -> float:
    """Run Execution-Discipline-Agent; return compliance_score (0..1)."""
    _setup_paths(extra=["Execution-Discipline-Agent"])
    import pandas as pd
    import json
    from src.agent import ExecutionDisciplineAgent  # Execution-Discipline-Agent

    trades = pd.read_csv(trades_path)
    plan = json.loads(plan_path.read_text())
    memory_path = str(state_dir / "discipline_memory.json")
    state_dir.mkdir(parents=True, exist_ok=True)
    agent = ExecutionDisciplineAgent(memory_path=memory_path)
    report = agent.run(trades, plan, regime_label)
    return report.compliance_score


def _run_guardian(
    total_value: float,
    drawdown_pct: float,
    regime_label: str,
    volatility: str = "normal",
    trade_type: str = "swing",
    recent_losses: int = 0,
    price: float | None = None,
    atr: float | None = None,
) -> dict:
    """Run Capital-Guardian-Agent; return decision + optional stop_distance and position_size."""
    _setup_paths(extra=["Capital-Guardian-Agent"])
    from schemas import TradeInput  # Capital-Guardian-Agent
    from risk_engine import CapitalGuardian
    from stop_loss import calculate_stop_loss
    from position_sizer import calculate_position_size

    # Map regime label to guardian regime (trend, range, choppy)
    r = regime_label.upper()
    if "HIGH VOL" in r or "CHOP" in r or "RANGING" in r:
        regime = "choppy"
    elif "UPTREND" in r or "DOWNTREND" in r:
        regime = "trend"
    else:
        regime = "range"

    trade = TradeInput(
        account_size=total_value,
        drawdown_pct=drawdown_pct,
        volatility=volatility,
        regime=regime,
        trade_type=trade_type,
        recent_losses=recent_losses,
    )
    guardian = CapitalGuardian()
    decision = guardian.evaluate(trade)
    out = {"guardian_decision": decision}
    if price is not None and atr is not None and decision.get("trade_allowed"):
        stop = calculate_stop_loss(atr, regime, trade_type)
        size = calculate_position_size(total_value, decision["allowed_risk_pct"], stop, price)
        out["stop_distance"] = stop
        out["position_size"] = size
    return out


def run_pipeline(
    prices_path: Path,
    holdings_path: Path,
    config_path: Path,
    state_dir: Path,
    peak_equity: float | None = None,
    execution_score: float = 0.88,
    execution_trades_path: Path | None = None,
    execution_plan_path: Path | None = None,
    include_guardian: bool = False,
    guardian_price: float | None = None,
    guardian_atr: float | None = None,
) -> dict:
    """
    Run full pipeline: regime → portfolio → [execution-discipline] → allocation → [guardian].
    Returns dict with keys: regime, portfolio, decision, and optionally execution_discipline_score, guardian_guardrails.
    """
    from orchestrator.adapters import build_capital_allocation_inputs

    prices = load_prices(prices_path)
    prices_df = prices_for_portfolio(prices)
    regime_output = run_regime(prices, state_dir)
    portfolio_output = run_portfolio(holdings_path, prices_df)
    regime_label = regime_output.get("label", "Unknown")

    # Execution discipline: real score if trades+plan provided; else use cached score if any
    if execution_trades_path and execution_plan_path and execution_trades_path.exists() and execution_plan_path.exists():
        execution_score = _run_execution_discipline(
            execution_trades_path, execution_plan_path, regime_label, state_dir
        )
        set_cached_discipline_score(state_dir, execution_score)
    else:
        cached = get_cached_discipline_score(state_dir)
        if cached is not None:
            execution_score = cached

    peak = peak_equity or (portfolio_output["total_value"] * 1.1)
    inputs = build_capital_allocation_inputs(
        market_regime_output=regime_output,
        portfolio_analyst_output=portfolio_output,
        execution_discipline_score=execution_score,
        peak_equity=peak,
    )
    decision = run_allocation(inputs, config_path)

    result = {
        "regime": regime_output,
        "portfolio": {
            "total_value": portfolio_output["total_value"],
            "vol_annual": portfolio_output["vol_annual"],
            "open_positions": portfolio_output["open_positions"],
            "portfolio_heat": portfolio_output["portfolio_heat"],
        },
        "decision": decision,
        "execution_discipline_score": execution_score,
    }

    if include_guardian:
        drawdown_pct = (peak - portfolio_output["total_value"]) / peak * 100.0 if peak > 0 else 0.0
        guardian_result = _run_guardian(
            total_value=portfolio_output["total_value"],
            drawdown_pct=drawdown_pct,
            regime_label=regime_label,
            price=guardian_price,
            atr=guardian_atr,
        )
        result["guardian_guardrails"] = guardian_result

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run regime → allocation pipeline")
    parser.add_argument(
        "--prices",
        type=Path,
        default=ROOT / "Market-Regime-Agent" / "data" / "sample_prices.csv",
        help="Path to prices CSV (date + ticker columns)",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=ROOT / "orchestrator" / "state",
        help="Directory for regime memory etc.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "Capital-Allocation-Agent" / "config.yaml",
        help="Path to Capital-Allocation-Agent config",
    )
    parser.add_argument(
        "--holdings",
        type=Path,
        default=ROOT / "Portfolio-Analyst-Agent" / "portfolio.csv",
        help="Path to holdings CSV (symbol, shares [, avg_price])",
    )
    parser.add_argument(
        "--peak-equity",
        type=float,
        default=None,
        help="Peak equity for drawdown (default: 1.1 * total_value)",
    )
    args = parser.parse_args()

    if not args.prices.exists():
        print(f"Error: prices file not found: {args.prices}", file=sys.stderr)
        return 1
    if not args.config.exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        return 1
    if not args.holdings.exists():
        print(f"Error: holdings file not found: {args.holdings}", file=sys.stderr)
        return 1

    print("Running Market-Regime-Agent...")
    result = run_pipeline(
        prices_path=args.prices,
        holdings_path=args.holdings,
        config_path=args.config,
        state_dir=args.state_dir,
        peak_equity=args.peak_equity,
    )
    regime_output = result["regime"]
    portfolio_output = result["portfolio"]
    decision = result["decision"]
    print(f"  regime label: {regime_output.get('label', '?')}")
    print(f"  confidence:   {regime_output.get('confidence', 0):.2f}")
    print("\nRunning Portfolio-Analyst-Agent (compute_portfolio)...")
    print(f"  total_value:  {portfolio_output['total_value']:,.0f}")
    print(f"  vol_annual:   {portfolio_output['vol_annual']:.2f}")
    print(f"  open_positions: {portfolio_output['open_positions']}")
    print("\nRunning Capital-Allocation-Agent...")
    print("\n--- Capital Allocation Decision ---")
    for k, v in decision.items():
        print(f"  {k}: {v}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
