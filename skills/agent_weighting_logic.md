# Agent Weighting Logic

## Purpose
Defines how the orchestrator combines or weights outputs from multiple agents (e.g., regime, sentiment, insider, allocation) when forming a single decision or recommendation. Load when aggregating agent outputs.

## Key Knowledge
- **Regime:** Often primary filter; allocation and guardian depend on it; weight is implicit (downstream agents consume regime directly).
- **Intelligence agents:** Sentiment and insider can boost or reduce conviction for a name; weight by historical accuracy (e.g., from [verified_performance/](../../shared-skills/verified_performance/)) or fixed weights.
- **Guardian:** Veto power; not weighted—if guardian blocks, decision is blocked.
- **Execution-Discipline:** Post-trade; can influence future weighting (e.g., lower weight to allocation if discipline score is low).

## Decision Criteria
- Regime: used as context by allocation and guardian; no numeric weight—binary or categorical consumption.
- Sentiment and insider: if both have signals on same symbol, combine (e.g., average or max of scores) or use hierarchy (e.g., insider over sentiment when both present). Optional: scale by verified_performance score (Sigmodx) when available.
- Allocation: produces the plan; guardian can block but not override allocation logic—guardian is gate.
- When verified_performance (Sigmodx) scores exist: weight agent contribution by score (e.g., higher score → more weight to that agent’s signal). Placeholder files in shared-skills/ hold these scores.
- Document which agents contributed and at what effective weight for [recent_decisions.md](../../shared-skills/recent_decisions.md).

## Related Nodes
- [pipeline_sequencing_rules.md](pipeline_sequencing_rules.md) — order of execution
- [error_handling_patterns.md](error_handling_patterns.md) — when an agent fails, weight may be 0 or fallback
- Shared: [verified_performance/](../../shared-skills/verified_performance/) — regime_agent, sentiment_agent, insider_agent placeholders for scores
- Shared: [recent_decisions.md](../../shared-skills/recent_decisions.md) — write decision and weights used

## Memory Hook
After using: write effective weights or contribution per agent for the decision to recent_decisions for audit and tuning.
