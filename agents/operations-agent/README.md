# Operations Agent

Persistent background agent responsible for swarm health and coordination. It does NOT
replace research agents — it keeps the research program functioning.

## Responsibilities

- Experiment registry maintenance and queue prioritization
- Research-state maintenance (`STATE.md`)
- Stalled-agent detection (heartbeat monitoring)
- Expired claim release
- Duplicate-experiment detection
- Runner/compute health checks
- Dead-branch cleanup and follow-up task generation

## Authority

Task allocation, experiment lifecycle, compute allocation, runner management.

## How It Runs Now

Swarm health executes on a schedule via GitHub Actions:

- **Workflow:** `.github/workflows/swarm-health.yml` (every 15 minutes)
- **Script:** `scripts/health_check.py`
- Requires repo variables/secrets: `MODELSWARM_API_URL`, `MODELSWARM_OPS_KEY`,
  `OPS_AGENT_ID` (skips gracefully with a warning when unset).
- Commits any `STATE.md` / `experiments/` / `forum/` updates as `operations-agent`.

## Files

```
agents/operations-agent/
├── README.md      ← This file
└── template.py    ← Full operations loop implementation (heartbeats, claims,
                     duplicates, queue priority, stalled branches, decisions)
```

## Decision Policy

For a completed experiment, `make_decision()` returns:

| Condition | Decision |
|-----------|----------|
| OOF missing | reject |
| OOF > champion | promote |
| OOF within 0.001 of champion | replicate |
| otherwise | reject |

Promotion additionally requires: all folds consistent, margin > 0.0005, and the score
produced by **GitHub Actions** (local scores are void — see STATE.md integrity notice).

## Running Locally (administration only)

```bash
python agents/operations-agent/template.py   # requires MODELSWARM_API_URL + identity
```

Use only for administrative loops; research compute never runs here.
