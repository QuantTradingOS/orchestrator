"""
Scheduled pipeline runs: regime → portfolio → allocation (and optional discipline/guardian).

Run standalone (from QuantTradingOS repo root):
  python -m orchestrator.scheduler
  python -m orchestrator.scheduler --minutes 60
  python -m orchestrator.scheduler --cron "0 9 * * 1-5"

Or run the API with scheduling enabled via env:
  ORCHESTRATOR_SCHEDULE_MINUTES=60 uvicorn orchestrator.api:app --host 0.0.0.0 --port 8000
  ORCHESTRATOR_SCHEDULE_CRON="0 9 * * 1-5" uvicorn orchestrator.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Repo root = parent of orchestrator/
ROOT = Path(__file__).resolve().parent.parent

# Default paths (same as run.py)
DEFAULT_PRICES = ROOT / "Market-Regime-Agent" / "data" / "sample_prices.csv"
DEFAULT_HOLDINGS = ROOT / "Portfolio-Analyst-Agent" / "portfolio.csv"
DEFAULT_CONFIG = ROOT / "Capital-Allocation-Agent" / "config.yaml"
DEFAULT_STATE_DIR = ROOT / "orchestrator" / "state"

LOG = logging.getLogger("orchestrator.scheduler")


def _run_scheduled_pipeline() -> None:
    """Run the full pipeline with default paths. Called by APScheduler."""
    from orchestrator.run import run_pipeline

    if not DEFAULT_PRICES.exists():
        LOG.warning("Skipping scheduled run: prices file not found %s", DEFAULT_PRICES)
        return
    if not DEFAULT_HOLDINGS.exists():
        LOG.warning("Skipping scheduled run: holdings file not found %s", DEFAULT_HOLDINGS)
        return
    if not DEFAULT_CONFIG.exists():
        LOG.warning("Skipping scheduled run: config not found %s", DEFAULT_CONFIG)
        return

    try:
        result = run_pipeline(
            prices_path=DEFAULT_PRICES,
            holdings_path=DEFAULT_HOLDINGS,
            config_path=DEFAULT_CONFIG,
            state_dir=DEFAULT_STATE_DIR,
        )
        regime = result.get("regime", {})
        decision = result.get("decision", {})
        LOG.info(
            "Scheduled pipeline run: regime=%s decision=%s",
            regime.get("label", "?"),
            decision,
        )
    except Exception as e:
        LOG.exception("Scheduled pipeline run failed: %s", e)


def get_scheduler_config() -> tuple[int | None, str | None]:
    """
    Read schedule from env: ORCHESTRATOR_SCHEDULE_MINUTES (interval) or
    ORCHESTRATOR_SCHEDULE_CRON (cron expression). Returns (minutes, cron).
    At most one should be set; if both, cron wins.
    """
    cron = os.environ.get("ORCHESTRATOR_SCHEDULE_CRON", "").strip() or None
    minutes_str = os.environ.get("ORCHESTRATOR_SCHEDULE_MINUTES", "").strip()
    minutes = None
    if minutes_str:
        try:
            minutes = int(minutes_str)
            if minutes < 1:
                minutes = None
        except ValueError:
            pass
    return minutes, cron


def start_scheduler(minutes: int | None = None, cron: str | None = None):
    """
    Start APScheduler: run pipeline on interval (minutes) or cron expression.
    Returns the scheduler instance so the caller can shutdown() on app exit.
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    if not minutes and not cron:
        raise ValueError("Set either minutes (interval) or cron expression")

    scheduler = BackgroundScheduler()
    if cron:
        scheduler.add_job(
            _run_scheduled_pipeline,
            CronTrigger.from_crontab(cron),
            id="orchestrator_pipeline",
            name="Orchestrator pipeline",
        )
        LOG.info("Scheduler: pipeline will run on cron %s", cron)
    else:
        scheduler.add_job(
            _run_scheduled_pipeline,
            IntervalTrigger(minutes=minutes),
            id="orchestrator_pipeline",
            name="Orchestrator pipeline",
        )
        LOG.info("Scheduler: pipeline will run every %s minute(s)", minutes)

    scheduler.start()
    return scheduler


def main() -> int:
    """CLI: run scheduler standalone. Config from env or --minutes / --cron."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run orchestrator pipeline on a schedule (interval or cron)"
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=None,
        help="Run pipeline every N minutes (overrides ORCHESTRATOR_SCHEDULE_MINUTES)",
    )
    parser.add_argument(
        "--cron",
        type=str,
        default=None,
        help='Cron expression (e.g. "0 9 * * 1-5" for 9am weekdays; overrides ORCHESTRATOR_SCHEDULE_CRON)',
    )
    args = parser.parse_args()

    minutes = args.minutes
    cron = args.cron
    if minutes is None and not cron:
        minutes, cron = get_scheduler_config()
    if not minutes and not cron:
        print(
            "Set schedule via --minutes N or --cron 'expr', or env ORCHESTRATOR_SCHEDULE_MINUTES / ORCHESTRATOR_SCHEDULE_CRON",
            file=sys.stderr,
        )
        return 1

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    scheduler = start_scheduler(minutes=minutes, cron=cron)
    try:
        # Run one time immediately (optional; keeps process alive with interval)
        _run_scheduled_pipeline()
        # Block forever (scheduler runs in background)
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        scheduler.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
