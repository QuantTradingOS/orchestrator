# Error Handling Patterns

## Purpose
Defines how the orchestrator handles failures (agent timeout, API error, bad data) so the pipeline degrades gracefully and logs consistently. Load when implementing or debugging pipeline error handling.

## Key Knowledge
- **Per-agent failure:** If one agent fails (e.g., regime agent timeout), do not necessarily fail entire pipeline; use fallback (e.g., last known regime, or conservative default) per [fallback_behavior.md](fallback_behavior.md).
- **Critical vs non-critical:** Guardian and allocation are critical; intelligence agents may be optional (proceed with reduced signal).
- **Retry:** Optional retry with backoff for transient failures (network, rate limit); limit retries to avoid long blocks.
- **Logging:** Log error type, agent, timestamp, and fallback used; do not expose internal details to client.

## Decision Criteria
- On agent timeout: retry once if idempotent; else use fallback for that agent and continue pipeline.
- On agent returning error status: log; if critical agent (e.g., guardian), abort or use safe default (e.g., block all new risk); if non-critical (e.g., sentiment), proceed with null or last-known for that input.
- On data validation failure (e.g., malformed regime output): use fallback value and log; alert if repeated.
- Always write pipeline outcome (success, partial, failed) and which step failed to [recent_decisions.md](../../shared-skills/recent_decisions.md) or audit log.

## Related Nodes
- [fallback_behavior.md](fallback_behavior.md) — what to use when an agent fails
- [pipeline_sequencing_rules.md](pipeline_sequencing_rules.md) — which step failed
- [agent_weighting_logic.md](agent_weighting_logic.md) — failed agent gets weight 0 or fallback input
- Shared: [recent_decisions.md](../../shared-skills/recent_decisions.md) — log errors and outcome

## Memory Hook
After using: log error code, agent name, fallback applied, and pipeline result (success/partial/fail) for ops and debugging.
