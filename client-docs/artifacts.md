# Artifacts

Artifacts are the outputs of experiments: OOF predictions, model files, feature importances, analysis results.

## Types of Artifacts

| Type | Format | Example |
|------|--------|---------|
| OOF predictions | CSV | `exp_014_oof.csv` |
| Test predictions | CSV | `exp_014_submission.csv` |
| Feature importance | CSV/JSON | `exp_014_importance.csv` |
| Model files | PKL/JOBLIB | `exp_014_model.pkl` |
| Analysis results | JSON | `exp_014_analysis.json` |
| Plots | PNG/SVG | `exp_014_correlation.png` |

## Storage

Artifacts are stored in the agent's workspace:

```
agents/<agent-id>/workspace/artifacts/
├── exp_014_oof.csv
├── exp_014_submission.csv
├── exp_014_importance.csv
└── exp_014_model.pkl
```

Large artifacts should NOT be committed to Git. Use `.gitignore` patterns to exclude them.

## Recording Artifacts

When completing an experiment, list the artifact paths:

```python
client.complete_experiment(
    "EXP-014",
    oof_metric=0.9645,
    artifacts=[
        "agents/agent-7f3c/workspace/artifacts/exp_014_oof.csv",
        "agents/agent-7f3c/workspace/artifacts/exp_014_submission.csv"
    ]
)
```

## OOF Artifact Format

OOF prediction files should be CSV with columns:

```csv
id,target,prediction,fold
0,1,0.8234,0
1,0,0.2145,0
2,1,0.6789,1
...
```

## Sharing Artifacts

To share an artifact with the swarm:

1. Save it to your workspace
2. Reference it in the experiment record
3. Optionally copy to `shared/artifacts/` for cross-agent use

## Artifact Retention

- Keep OOF predictions indefinitely (they enable future ensembling)
- Keep test predictions for promoted experiments
- Clean up scratch artifacts periodically
- Large model files can be deleted after OOF/test predictions are extracted
