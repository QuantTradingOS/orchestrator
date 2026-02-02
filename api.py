"""
FastAPI app for the orchestrator pipeline and all agents.

Run from QuantTradingOS repo root:
  uvicorn orchestrator.api:app --reload --host 0.0.0.0 --port 8000

Swagger UI:  http://localhost:8000/docs
OpenAPI JSON: http://localhost:8000/openapi.json
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

# Repo root = parent of orchestrator/
ROOT = Path(__file__).resolve().parent.parent

# Default paths (relative to repo root)
DEFAULT_PRICES = ROOT / "Market-Regime-Agent" / "data" / "sample_prices.csv"
DEFAULT_HOLDINGS = ROOT / "Portfolio-Analyst-Agent" / "portfolio.csv"
DEFAULT_CONFIG = ROOT / "Capital-Allocation-Agent" / "config.yaml"
DEFAULT_STATE_DIR = ROOT / "orchestrator" / "state"
DEFAULT_PLAN = ROOT / "Execution-Discipline-Agent" / "data" / "plan.example.json"


class DecisionRequest(BaseModel):
    """Optional overrides for pipeline inputs. Paths are relative to repo root or absolute."""

    prices_path: Optional[str] = Field(None, description="Path to prices CSV (date + ticker columns)")
    holdings_path: Optional[str] = Field(None, description="Path to holdings CSV (symbol, shares [, avg_price])")
    peak_equity: Optional[float] = Field(None, description="Peak equity for drawdown calculation")
    execution_score: float = Field(0.88, ge=0.0, le=1.0, description="Execution discipline score (0-1); ignored if execution_trades_path+plan provided")
    execution_trades_path: Optional[str] = Field(None, description="Path to trades CSV for Execution-Discipline (with plan_path)")
    execution_plan_path: Optional[str] = Field(None, description="Path to plan JSON for Execution-Discipline")
    include_guardian: bool = Field(False, description="Include Capital-Guardian guardrails in response")
    guardian_price: Optional[float] = Field(None, description="Price for guardian position size (optional)")
    guardian_atr: Optional[float] = Field(None, description="ATR for guardian stop (optional)")


class GuardianRequest(BaseModel):
    """Input for Capital-Guardian-Agent."""

    account_size: float = Field(..., gt=0)
    drawdown_pct: float = Field(0.0, ge=0.0, le=100.0)
    volatility: str = Field("normal", description="low | normal | high")
    regime: str = Field("range", description="trend | range | choppy")
    trade_type: str = Field("swing", description="scalp | swing | earnings")
    recent_losses: int = Field(0, ge=0)
    price: Optional[float] = Field(None)
    atr: Optional[float] = Field(None)


class SentimentRequest(BaseModel):
    """Input for Sentiment-Shift-Alert-Agent. Pass API keys in body or set FINNHUB_API_KEY and OPENAI_API_KEY in env."""

    symbol: str = Field(..., description="Ticker symbol")
    finnhub_key: Optional[str] = Field(None, description="Finnhub API key (or set FINNHUB_API_KEY in environment)")
    openai_key: Optional[str] = Field(None, description="OpenAI API key (or set OPENAI_API_KEY in environment)")


class InsiderReportRequest(BaseModel):
    """Input for Equity-Insider-Intelligence-Agent. Pass API keys in body or set FINNHUB_API_KEY and OPENAI_API_KEY in env."""

    symbol: str = Field(..., description="Ticker symbol")
    finnhub_key: Optional[str] = Field(None, description="Finnhub API key (or set FINNHUB_API_KEY in environment)")
    openai_key: Optional[str] = Field(None, description="OpenAI API key (or set OPENAI_API_KEY in environment)")


class PortfolioReportRequest(BaseModel):
    """Input for standalone portfolio report."""

    holdings_path: Optional[str] = Field(None, description="Path to holdings CSV")
    prices_path: Optional[str] = Field(None, description="Path to prices CSV (optional; yfinance used if missing)")


class TradeJournalRequest(BaseModel):
    """Input for Trade-Journal-Coach-Agent. Pass openai_key in body or set OPENAI_API_KEY in env."""

    openai_key: Optional[str] = Field(None, description="OpenAI API key (or set OPENAI_API_KEY in environment)")
    trades_json: Optional[str] = Field(None, description="JSON array of trades: [{date, symbol, side, qty, entry, exit, fees?}, ...]")


def _resolve_path(value: Optional[str], default: Path) -> Path:
    if value is None:
        return default
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    return p


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start optional pipeline scheduler on startup if env is set; shutdown on exit."""
    scheduler = None
    try:
        from orchestrator.scheduler import get_scheduler_config, start_scheduler
        minutes, cron = get_scheduler_config()
        if minutes or cron:
            scheduler = start_scheduler(minutes=minutes, cron=cron)
            logging.getLogger("orchestrator.api").info(
                "Pipeline scheduler started (minutes=%s, cron=%s)", minutes, cron
            )
    except Exception as e:
        logging.getLogger("orchestrator.api").warning("Scheduler not started: %s", e)
    yield
    if scheduler:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="QuantTradingOS API",
    description="Orchestrator pipeline (regime → portfolio → allocation) plus agent endpoints: execution-discipline, guardian, sentiment, insider, trade-journal, portfolio-report.",
    version="0.2.0",
    lifespan=_lifespan,
)


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/decision", response_model=dict)
def run_decision(body: Optional[DecisionRequest] = None):
    """
    Run the full pipeline: Regime → Portfolio → [Execution-Discipline] → Allocation → [Guardian].
    Optionally pass execution_trades_path + execution_plan_path for real discipline score; set include_guardian=True for guardian guardrails.
    """
    req = body or DecisionRequest()
    prices_path = _resolve_path(req.prices_path, DEFAULT_PRICES)
    holdings_path = _resolve_path(req.holdings_path, DEFAULT_HOLDINGS)
    config_path = _resolve_path(None, DEFAULT_CONFIG)
    state_dir = DEFAULT_STATE_DIR
    exec_trades = _resolve_path(req.execution_trades_path, Path()) if req.execution_trades_path else None
    exec_plan = _resolve_path(req.execution_plan_path, DEFAULT_PLAN) if req.execution_plan_path else None
    if exec_trades and not exec_trades.exists():
        exec_trades = None
    if exec_plan and not exec_plan.exists():
        exec_plan = None

    if not prices_path.exists():
        raise HTTPException(status_code=400, detail=f"Prices file not found: {prices_path}")
    if not holdings_path.exists():
        raise HTTPException(status_code=400, detail=f"Holdings file not found: {holdings_path}")
    if not config_path.exists():
        raise HTTPException(status_code=500, detail=f"Config not found: {config_path}")

    from orchestrator.run import run_pipeline

    try:
        result = run_pipeline(
            prices_path=prices_path,
            holdings_path=holdings_path,
            config_path=config_path,
            state_dir=state_dir,
            peak_equity=req.peak_equity,
            execution_score=req.execution_score,
            execution_trades_path=exec_trades if (exec_trades and exec_plan) else None,
            execution_plan_path=exec_plan if (exec_trades and exec_plan) else None,
            include_guardian=req.include_guardian,
            guardian_price=req.guardian_price,
            guardian_atr=req.guardian_atr,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/decision", response_model=dict)
def run_decision_get(
    prices_path: Optional[str] = None,
    holdings_path: Optional[str] = None,
    peak_equity: Optional[float] = None,
    execution_score: float = 0.88,
    include_guardian: bool = False,
):
    """Same as POST /decision but with query params."""
    req = DecisionRequest(
        prices_path=prices_path,
        holdings_path=holdings_path,
        peak_equity=peak_equity,
        execution_score=execution_score,
        include_guardian=include_guardian,
    )
    return run_decision(body=req)


# ---------- Agent endpoints ----------


@app.post("/execution-discipline", response_model=dict)
def run_execution_discipline(
    regime_label: str,
    trades_path: Optional[str] = None,
    plan_path: Optional[str] = None,
    trades_file: Optional[UploadFile] = File(default=None),
    plan_json: Optional[str] = None,
):
    """
    Run Execution-Discipline-Agent: evaluate trades vs plan, return compliance score and violations.
    Provide either (trades_path + plan_path) or (trades_file CSV upload + plan_json body).
    """
    import json
    import pandas as pd
    from orchestrator.run import _setup_paths

    _setup_paths(extra=["Execution-Discipline-Agent"])
    from src.agent import ExecutionDisciplineAgent  # Execution-Discipline-Agent

    if trades_path and plan_path:
        tpath = _resolve_path(trades_path, Path())
        ppath = _resolve_path(plan_path, DEFAULT_PLAN)
        if not tpath.exists() or not ppath.exists():
            raise HTTPException(status_code=400, detail="trades_path and plan_path must exist")
        trades_df = pd.read_csv(tpath)
        plan = json.loads(ppath.read_text())
    elif trades_file and plan_json:
        try:
            content = trades_file.file.read()
            trades_df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid trades CSV: {e}")
        try:
            plan = json.loads(plan_json)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid plan JSON: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide (trades_path + plan_path) or (trades_file + plan_json)")

    state_dir = DEFAULT_STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    agent = ExecutionDisciplineAgent(memory_path=str(state_dir / "discipline_memory.json"))
    report = agent.run(trades_df, plan, regime_label)
    # Cache score so next /decision (without trades+plan) uses it
    from orchestrator.run import set_cached_discipline_score
    set_cached_discipline_score(state_dir, report.compliance_score)
    return {
        "compliance_score": report.compliance_score,
        "violations": [{"trade_index": v.trade_index, "violation_type": v.violation_type, "detail": v.detail} for v in report.violations],
        "regime_mismatch_rate": report.regime_mismatch_rate,
        "violation_summary": report.violation_summary,
        "compliance_trend": report.compliance_trend,
    }


@app.post("/guardian", response_model=dict)
def run_guardian(body: GuardianRequest):
    """Run Capital-Guardian-Agent: evaluate trade, return decision + optional stop and position size."""
    # Pass regime as "regime_label" so _run_guardian maps it; for API we already have trend/range/choppy so use a fake label that maps correctly
    label_map = {"trend": "Uptrend", "range": "Range", "choppy": "Choppy"}
    regime_label = label_map.get(body.regime.lower(), body.regime)
    from orchestrator.run import _run_guardian

    result = _run_guardian(
        total_value=body.account_size,
        drawdown_pct=body.drawdown_pct,
        regime_label=regime_label,
        volatility=body.volatility,
        trade_type=body.trade_type,
        recent_losses=body.recent_losses,
        price=body.price,
        atr=body.atr,
    )
    return result


@app.post("/sentiment-alert", response_model=dict)
def run_sentiment_alert(body: SentimentRequest):
    """Run Sentiment-Shift-Alert-Agent: fetch news for symbol, infer sentiment (score, confidence, explanation)."""
    finnhub_key = body.finnhub_key or os.environ.get("FINNHUB_API_KEY")
    openai_key = body.openai_key or os.environ.get("OPENAI_API_KEY")
    if not finnhub_key or not openai_key:
        raise HTTPException(status_code=400, detail="Set finnhub_key and openai_key in body or FINNHUB_API_KEY and OPENAI_API_KEY in env")

    _setup_paths_agent("Sentiment-Shift-Alert-Agent")
    from app import get_company_news, infer_sentiment  # Sentiment-Shift-Alert-Agent

    news = get_company_news(body.symbol, finnhub_key)
    result = infer_sentiment(openai_key, body.symbol, news)
    return result


@app.post("/insider-report", response_model=dict)
def run_insider_report(body: InsiderReportRequest):
    """Run Equity-Insider-Intelligence-Agent: insider + price + news, then LLM report. Returns report text."""
    finnhub_key = body.finnhub_key or os.environ.get("FINNHUB_API_KEY")
    openai_key = body.openai_key or os.environ.get("OPENAI_API_KEY")
    if not finnhub_key or not openai_key:
        raise HTTPException(status_code=400, detail="Set finnhub_key and openai_key in body or env")

    _setup_paths_agent("Equity-Insider-Intelligence-Agent")
    from app import finnhub_get_insider_transactions, finnhub_get_company_news, get_price_snapshot, build_report_with_openai
    import pandas as pd

    symbol = body.symbol.upper()
    insider = finnhub_get_insider_transactions(symbol, finnhub_key)
    news = finnhub_get_company_news(symbol, finnhub_key)
    price = get_price_snapshot(symbol)
    insider_df = pd.DataFrame(insider.get("data", [])[:50])
    news_df = pd.DataFrame(news[:20]) if news else pd.DataFrame()
    report = build_report_with_openai(openai_key, symbol, price, insider_df, news_df)
    return {"symbol": symbol, "report": report}


def _setup_paths_agent(name: str) -> None:
    if str(ROOT) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(ROOT))
    agent_root = ROOT / name
    if agent_root.exists() and str(agent_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(agent_root))


@app.post("/trade-journal", response_model=dict)
def run_trade_journal(
    body: Optional[TradeJournalRequest] = None,
    trades_file: Optional[UploadFile] = File(default=None),
    openai_key_form: Optional[str] = Form(default=None, alias="openai_key", description="OpenAI key (when using file upload)"),
):
    """
    Run Trade-Journal-Coach-Agent: compute metrics from trades CSV/JSON, then LLM coaching report.
    Pass openai_key in JSON body, or as form field when uploading trades_file, or set OPENAI_API_KEY in env.
    Provide trades_file (CSV upload) or body.trades_json (JSON array in body).
    """
    import json
    import pandas as pd
    import numpy as np

    openai_key = (body.openai_key if body else None) or openai_key_form or os.environ.get("OPENAI_API_KEY")
    trades_json = body.trades_json if body else None
    if not openai_key:
        raise HTTPException(status_code=400, detail="Set openai_key or OPENAI_API_KEY")

    _setup_paths_agent("Trade-Journal-Coach-Agent")
    from app import load_trades_df, coach_report  # Trade-Journal-Coach-Agent

    # Build payload for coach: metrics dict
    if trades_file:
        content = trades_file.file.read()
        df_raw = pd.read_csv(io.BytesIO(content))
    elif trades_json:
        try:
            data = json.loads(trades_json)
            df_raw = pd.DataFrame(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid trades_json: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide trades_file (CSV upload) or body.trades_json (JSON array in request body)")

    df = load_trades_df(df_raw)
    n = len(df)
    if n == 0:
        raise HTTPException(status_code=400, detail="No valid trades after parsing")

    # Minimal metrics payload for coach
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    payload = {
        "n_trades": n,
        "win_rate": float(len(wins) / n) if n else 0,
        "avg_win": float(wins["pnl"].mean()) if len(wins) else 0,
        "avg_loss": float(losses["pnl"].mean()) if len(losses) else 0,
        "total_pnl": float(df["pnl"].sum()),
        "profit_factor": float(wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and losses["pnl"].sum() != 0 else 0,
        "expectancy": float(df["pnl"].mean()),
    }
    report = coach_report(openai_key, payload)
    return {"metrics": payload, "coaching_report": report}


@app.post("/portfolio-report", response_model=dict)
def run_portfolio_report(body: PortfolioReportRequest):
    """Standalone portfolio analytics (Portfolio-Analyst-Agent compute_portfolio). No allocation."""
    holdings_path = _resolve_path(body.holdings_path, DEFAULT_HOLDINGS)
    prices_path = _resolve_path(body.prices_path, DEFAULT_PRICES) if body.prices_path else None
    if not holdings_path.exists():
        raise HTTPException(status_code=400, detail=f"Holdings file not found: {holdings_path}")

    from orchestrator.run import load_prices, prices_for_portfolio, run_portfolio

    prices_df = prices_for_portfolio(load_prices(prices_path)) if prices_path and prices_path.exists() else None
    if prices_df is None or prices_df.empty:
        # Need to fetch from holdings symbols
        import pandas as pd
        h = pd.read_csv(holdings_path)
        symbols = h["symbol"].tolist()
        _setup_paths_agent("Portfolio-Analyst-Agent")
        from app import fetch_prices
        prices_df = fetch_prices(symbols)
    result = run_portfolio(holdings_path, prices_df)
    # Make JSON-serializable (holdings is a DataFrame)
    if "holdings" in result and hasattr(result["holdings"], "to_dict"):
        result = dict(result)
        result["holdings"] = result["holdings"].to_dict(orient="records")
    return result
