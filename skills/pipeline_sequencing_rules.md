# Pipeline Sequencing Rules

## Purpose
Defines the order and dependencies of agents in the decision pipeline so the orchestrator runs them correctly and passes outputs as inputs. Load when configuring or executing the pipeline.

## Key Knowledge
- **Typical sequence:** Regime → Portfolio (analyst) / Sentiment / Insider (intelligence) → Execution-Discipline (optional) → Allocation → Guardian. Guardian is pre-trade gate; order may vary (e.g., allocation then guardian check).
- **Data flow:** Regime writes to shared `current_regime`; Portfolio writes to `portfolio_state`; Orchestrator writes to `recent_decisions`. Intelligence agents feed allocation; allocation and guardian feed execution.
- **Trigger:** Pipeline can run on schedule (e.g., interval/cron), on API call (/decision), or on event; sequencing is same.

## Decision Criteria
- Step 1: Run Market-Regime-Agent; persist output to shared [current_regime.md](../../shared-skills/current_regime.md).
- Step 2: Run intelligence agents (Sentiment, Insider) and optionally Portfolio-Analyst (for state); persist or cache outputs.
- Step 3: Run Capital-Allocation-Agent with regime + intelligence + portfolio state; produce allocation plan.
- Step 4: Run Capital-Guardian-Agent with plan + portfolio state + regime; approve or block/modify.
- Step 5: If execution path: run Execution-Discipline-Agent on executed trades (post-trade) or skip if no execution.
- On error: apply [error_handling_patterns.md](error_handling_patterns.md) and [fallback_behavior.md](fallback_behavior.md).

## Related Nodes
- [agent_weighting_logic.md](agent_weighting_logic.md) — how to combine agent outputs when multiple inform decision
- [error_handling_patterns.md](error_handling_patterns.md) — when a step fails
- [fallback_behavior.md](fallback_behavior.md) — what to do on failure
- Shared: [current_regime.md](../../../shared-skills/current_regime.md), [portfolio_state.md](../../../shared-skills/portfolio_state.md), [recent_decisions.md](../../../shared-skills/recent_decisions.md)

## Memory Hook
After using: write pipeline run id, sequence executed, timestamps per step, and final decision summary to [recent_decisions.md](../../../shared-skills/recent_decisions.md).
