"""
Shared data client for QuantTradingOS agents.
All agents use this module to fetch market data from the internal data service.
Replaces all direct yfinance and finnhub imports across the codebase.
"""
import os
from typing import List

import httpx

DATA_SERVICE_URL = os.environ.get("DATA_SERVICE_URL", "http://localhost:8001").rstrip("/")
DEFAULT_TIMEOUT = 10.0  # seconds


def get_prices(symbol: str, days: int = 30, limit: int = 100) -> List[dict]:
    """
    Fetch OHLCV price bars for a symbol.
    Returns list of dicts with keys: timestamp, open, high, low, close, volume, source.
    Sorted newest first.
    """
    try:
        response = httpx.get(
            f"{DATA_SERVICE_URL}/prices/{symbol.upper()}",
            params={"days": days, "limit": limit},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["data"]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return []
        raise
    except Exception as e:
        raise RuntimeError(f"Data service unavailable for prices/{symbol}: {e}") from e


def get_news(symbol: str, days: int = 7, limit: int = 20) -> List[dict]:
    """
    Fetch recent news articles for a symbol.
    Returns list of dicts with keys: timestamp, headline, summary, source, url.
    Sorted newest first.
    """
    try:
        response = httpx.get(
            f"{DATA_SERVICE_URL}/news/{symbol.upper()}",
            params={"days": days, "limit": limit},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["data"]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return []
        raise
    except Exception as e:
        raise RuntimeError(f"Data service unavailable for news/{symbol}: {e}") from e


def get_insider(symbol: str, days: int = 90, limit: int = 50) -> List[dict]:
    """
    Fetch insider transactions for a symbol.
    Returns list of dicts with keys: transaction_date, type, shares, price, value, insider_name.
    Sorted newest first.
    """
    try:
        response = httpx.get(
            f"{DATA_SERVICE_URL}/insider/{symbol.upper()}",
            params={"days": days, "limit": limit},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["data"]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return []
        raise
    except Exception as e:
        raise RuntimeError(f"Data service unavailable for insider/{symbol}: {e}") from e


def health_check() -> bool:
    """Check if the data service is reachable."""
    try:
        response = httpx.get(f"{DATA_SERVICE_URL}/health", timeout=3.0)
        return response.status_code == 200
    except Exception:
        return False
