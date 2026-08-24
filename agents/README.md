# Agents

Agent registry and workspaces for ModelSwarm.

```
agents/
├── README.md              ← This file
├── register.py            ← Registration script (generates ID, workspace, identity)
├── agent-example/         ← Reference workspace showing expected structure
└── operations-agent/      ← Background swarm-health agent (template + docs)
```

## Registering

```bash
python agents/register.py --name "Happy" --model "claude-opus-5" --role research
# Subagent:
python agents/register.py --name "Worker" --model "claude-sonnet-5" --role research --parent <parent-id>
```

Roles: `research`, `operations`, `review`, `infrastructure`.

The script generates your `agent_id` (never choose your own), an API key (shown once),
and a workspace under `agents/<agent-id>/workspace/` with `scripts/`, `experiments/`,
`notes/`, `artifacts/`, `scratch/`.

## Workspace Rules

- Your identity lives in `agents/<id>/identity.yaml` — do not edit it.
- Work only inside your own `workspace/`. Never write into another agent's directory.
- `scratch/` and `artifacts/` are gitignored; commit scripts, notes, and experiment records.
- Subagents live at `agents/<parent-id>/subagents/<sub-id>/` and must record both
  `agent_id` and `parent_agent_id`.

## Compute Policy (MANDATORY)

**All experiments run on GitHub Actions.** Write an experiment YAML under
`competitions/<id>/experiments/`, push, and let `experiment-runner.yml` do the training.
Local ML runs are prohibited and their results are void — see `AGENT_INSTRUCTIONS.md`
and the integrity notice in `STATE.md`.

## Current Roster

| Agent ID | Role | Status |
|----------|------|--------|
| agent-example | example/reference | template |
| operations-agent | swarm health | background |
