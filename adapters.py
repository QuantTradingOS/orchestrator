"""
Adapters: map upstream agent outputs to Capital Allocation Agent inputs.

Contracts (in spirit):
- Market-Regime-Agent outputs RegimeDecision (asof, label, confidence, drivers, implications, signals).
- Portfolio-Analyst-Agent (compute_portfolio) returns total_value, holdings, vol_annual, ret_total, etc.
- Capital-Allocation-Agent expects market_regime {regime, confidence}, portfolio_metrics {…}, execution_discipline_score.
"""

from __future__ import annotations

from typing import Any


def _map_regime_label_to_capital_allocation(label: str) -> str:
    """
    Map Market Regime Agent label to Capital Allocation Agent regime key.

    Label format: "Risk-On/Off / High Vol|Low Vol / Uptrend|Downtrend / Healthy|Weak Breadth"
    Maps to: trend, trending, range, ranging, chop, high_vol, volatile, unknown
    """
    label_upper = label.upper()
    if "HIGH VOL" in label_upper:
        return "high_vol"
    if "LOW VOL" in label_upper or "NORMAL VOL" in label_upper:
        if "UPTREND" in label_upper or "DOWNTREND" in label_upper:
            return "trending" if "UPTREND" in label_upper else "ranging"
        if "WEAK BREADTH" in label_upper:
            return "chop"
        return "trending"
    if "CHOP" in label_upper or "RANGING" in label_upper:
        return "ranging"
    return "unknown"


def adapt_market_regime_output(regime_decision: dict[str, Any]) -> dict[str, Any]:
    """
    Adapt Market Regime Agent output to Capital Allocation Agent market_regime input.

    Args:
        regime_decision: Output from MarketRegimeAgent.run() -> RegimeDecision (as dict).
            Expected keys: label, confidence; may include drivers, implications, signals.

    Returns:
        market_regime dict for Capital Allocation Agent inputs.
    """
    label = regime_decision.get("label", "Unknown")
    confidence = float(regime_decision.get("confidence", 0.5))
    regime_key = _map_regime_label_to_capital_allocation(label)
    return {
        "regime": regime_key,
        "confidence": confidence,
    }


def adapt_portfolio_analyst_output(
    portfolio_result: dict[str, Any],
    peak_equity: float | None = None,
) -> dict[str, Any]:
    """
    Adapt Portfolio Analyst Agent output to Capital Allocation Agent portfolio_metrics input.

    Args:
        portfolio_result: Output from compute_portfolio() or equivalent.
            Expected keys: total_value, holdings (or open_positions), vol_annual;
            optional: ret_total, portfolio_heat, total_exposure.
        peak_equity: Peak equity for drawdown. If None, assumes no drawdown.

    Returns:
        portfolio_metrics dict for Capital Allocation Agent inputs.
    """
    total_value = float(portfolio_result.get("total_value", 0.0))
    holdings = portfolio_result.get("holdings")
    if holdings is not None and hasattr(holdings, "__len__") and not isinstance(holdings, dict):
        open_positions = len(holdings) if hasattr(holdings, "__len__") else int(portfolio_result.get("open_positions", 0))
    else:
        open_positions = int(portfolio_result.get("open_positions", 0))
    vol_annual = float(portfolio_result.get("vol_annual", 0.0))

    portfolio_heat = portfolio_result.get("portfolio_heat")
    if portfolio_heat is None:
        portfolio_heat = min(1.0, vol_annual * 0.6) if vol_annual else 0.0
    else:
        portfolio_heat = float(portfolio_heat)

    if peak_equity is not None and peak_equity > 0:
        drawdown = max(0.0, (peak_equity - total_value) / peak_equity)
    else:
        drawdown = 0.0

    total_exposure = float(portfolio_result.get("total_exposure", total_value))
    ret_total = portfolio_result.get("ret_total", 0.0)
    current_pnl = float(ret_total * total_value) if ret_total else 0.0

    return {
        "current_pnl": current_pnl,
        "portfolio_heat": portfolio_heat,
        "drawdown": drawdown,
        "open_positions": open_positions,
        "total_exposure": total_exposure,
    }


def build_capital_allocation_inputs(
    market_regime_output: dict[str, Any],
    portfolio_analyst_output: dict[str, Any],
    execution_discipline_score: float = 0.9,
    peak_equity: float | None = None,
) -> dict[str, Any]:
    """
    Build Capital Allocation Agent inputs from upstream agent outputs.

    Returns:
        Inputs dict for CapitalAllocationAgent.decide().
    """
    market_regime = adapt_market_regime_output(market_regime_output)
    portfolio_metrics = adapt_portfolio_analyst_output(
        portfolio_analyst_output,
        peak_equity=peak_equity,
    )
    return {
        "market_regime": market_regime,
        "portfolio_metrics": portfolio_metrics,
        "execution_discipline_score": execution_discipline_score,
    }
