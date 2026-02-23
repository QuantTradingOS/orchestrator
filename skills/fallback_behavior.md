# Fallback Behavior

## Purpose
Defines safe fallback values or actions when an agent or data source fails so the system does not make unsafe decisions. Load when implementing error handling and default behavior.

## Key Knowledge
- **Regime fallback:** If Market-Regime agent fails, assume conservative: e.g., treat as "sideways" or "high_vol" so allocation and guardian tighten rather than relax.
- **Portfolio state fallback:** If portfolio state unavailable, do not add risk; optionally block allocation until state is available or use last-known with a staleness check.
- **Intelligence fallback:** If Sentiment or Insider agent fails, proceed with null signal (no boost/reduction from that agent); allocation uses remaining signals.
- **Guardian fallback:** If guardian fails to respond, do not assume pass—default to block new risk (fail closed) until guardian is available.

## Decision Criteria
- Regime: on failure or timeout → set current_regime to { label: "sideways", confidence: 0, high_vol: true } or similar conservative state; log "regime fallback."
- Portfolio state: on failure → do not execute new orders that increase risk; or use last cached state if fresh (< N minutes) and log.
- Guardian: on failure → block pipeline from sending new orders; return "guardian unavailable, no new risk."
- Sentiment/Insider: on failure → set their output to null/neutral; allocation proceeds with other inputs.
- Document every fallback in [recent_decisions.md](../../shared-skills/recent_decisions.md) or audit log.

## Related Nodes
- [error_handling_patterns.md](error_handling_patterns.md) — when to trigger fallback
- [pipeline_sequencing_rules.md](pipeline_sequencing_rules.md) — which step uses which fallback
- Shared: [current_regime.md](../../shared-skills/current_regime.md), [portfolio_state.md](../../shared-skills/portfolio_state.md) — fallback writes to these
- Capital-Guardian: [circuit_breaker_logic.md](../../Capital-Guardian-Agent/skills/circuit_breaker_logic.md) — guardian-down is similar to breaker (block)

## Memory Hook
After using: write fallback type (e.g., "regime_conservative"), reason (e.g., "timeout"), and resulting state to recent_decisions for audit.
