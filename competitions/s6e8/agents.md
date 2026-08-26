# S6E8 — Agent Onboarding

> You are joining the S6E8 research swarm. This is a real Kaggle competition with real data. Your job is to do real research, not fabricate results.

## Competition

- **Name:** Kaggle Playground Series S6E8 — Smartphone Addiction Prediction
- **Target:** `addicted_label` (binary)
- **Metric:** ROC-AUC
- **Current Champion:** EXP-119 — LightGBM ensemble + free_time_slack + 1-D target encoding (OOF 0.967959; LB 0.97050 as part of blend v2). See REGISTRY.md and HISTORY.md
- **Compute:** PUBLIC-LB OVERFITTING POLICY (2026-08-26): treat the leaderboard as CONFIRMATION, not an
optimizer. Classify every submission: genuine model improvement / OOF-validated blend /
public-LB-only / repeated-probe-of-same-hypothesis. NO brute-force weight sweeps on the LB;
choose weights via OOF/held-out first. Flag overfit risk after ~2 successive probes of the
same hypothesis family. A standalone model that jumps substantially WITHOUT probe-tuning
deserves far more confidence than a blend tuned by repeated probing. Final selection:
prefer genuinely distinct, independently validated submissions over multiple variants of
one public-fit lineage. Tally kept in GHA.md.

PRIORITIZATION RULE (2026-08-25): when a prepared score-producing pipeline (blend,
submission, promotion) becomes unblocked by a finished experiment, EXECUTE IT IMMEDIATELY
before creating additional tooling or infrastructure. Infrastructure only when it blocks
execution.

OVERNIGHT AUTONOMOUS OPERATION (2026-08-25): during unattended windows run the loop
check state -> consume results -> research -> hypothesize -> prepare -> queue -> monitor ->
update docs -> repeat, for the entire window. Never sleep-wait minutes for a result; do
independent work between checks. Protect the best validated LB/OOF: promote only on the
>0.0005 OOF bar or a genuinely better OOS blend. End-of-window deliverable: concise report
(completed / running / best OOF+LB+rank / discoveries / promoted / rejected / next frontier /
blockers). Keep GHA.md, REGISTRY.md, STATE.md, HISTORY.md current throughout.

GHA.md = CANONICAL ACTIONS BOARD (2026-08-25): every pushed experiment/analysis gets a row
(root GHA.md) at push time; record run ID when GitHub creates it; verify status before calling
anything finished; read GHA.md at the start of every cycle and re-check RUNNING/QUEUED rows.
Never rely on memory or notifications for Actions state.

STANDING REMINDER (2026-08-25): while any experiment runs, sweep Kaggle for NEW
discussions, recently published notebooks, leaderboard movement, and strong competitors
(Krishanki-style jumps). Extract mechanisms we have not tested; compare every discovery
against REGISTRY.md before queueing. Idle time = research time.

HARD CONTINUOUS-WORK RULE (2026-08-25, strengthened): ending a turn while useful
independent research remains available is PROHIBITED. Waiting-for-results is not a terminal
state. Mandatory idle-loop while any compute runs: CHECK runs/agents/leaderboard -> RESEARCH
new mechanisms -> HYPOTHESIZE -> VERIFY against experiment history -> PREPARE yamls/scripts ->
CHECK AGAIN. Maintain a never-empty frontier of queued hypotheses; prepare follow-up work
before results arrive; re-check GHA/agents/LB periodically within the turn instead of stopping.

CONTINUOUS-RESEARCH MANDATE (2026-08-25): never idle while remote compute runs - mine
discussions/leaderboards, maintain a ranked hypothesis frontier, prepare follow-up scripts and
YAMLs BEFORE results arrive, consume results immediately on arrival, update REGISTRY.md.
A failed implementation closes only that implementation, never the hypothesis family.

LAPTOP = CONTROLLER ONLY (2026-08-25): no heavy or prolonged computation locally EVEN IF it is
not a model - no large weight searches, repeated AUC over huge matrices, exhaustive correlation
scans, long optimizations, or multi-minute scripts. If an analysis may exceed ~30-60s of real
CPU: prepare the script locally, commit it, execute on GHA (experiment yaml or dispatchable
analysis workflow), consume artifacts. Inspecting already-produced artifacts is fine.

GitHub Actions ONLY — local training is prohibited and its results are void

## Data Location

```
competitions/s6e8/data/
+-- train.csv           <- Training data (REAL -- download from Kaggle)
+-- test.csv            <- Test data (REAL -- download from Kaggle)
+-- sample_submission.csv <- Submission format
```

## CRITICAL: Real Data Only

**DO NOT FABRICATE DATA.** Before any experiment:

1. **Verify data exists:**
   ```bash
   ls competitions/s6e8/data/
   ```

2. **Validate data integrity:**
   ```bash
   python competitions/s6e8/validate_data.py
   ```

3. **If data is missing**, download from Kaggle:
   ```bash
   kaggle competitions download -c playground-series-s6e8 -p competitions/s6e8/data/
   unzip competitions/s6e8/data/playground-series-s6e8.zip -d competitions/s6e8/data/
   ```

4. **Never create synthetic data** and claim it is real. This invalidates the entire research program.

## Research Workflow

### Every Session

```
1. git pull origin master
2. python competitions/s6e8/validate_data.py   <- VERIFY DATA FIRST
3. Read competitions/s6e8/README.md and STATE.md
4. Check recent forum: modelswarm feed
5. Check existing experiments: modelswarm experiments
6. Design hypothesis; check it hasn't been tested
7. Write config: competitions/s6e8/experiments/EXP-0XX.yaml
   (training.compute: github_actions — see EXP-007.yaml as template)
8. git pull origin master
9. git add -A && git commit -m "exp: queue EXP-0XX <hypothesis>"
10. git push origin master                      <- triggers GitHub Actions run
11. Monitor: gh run watch or the Actions tab
12. Review results auto-committed into the YAML by the runner
13. Record decision/reasoning; update STATE.md if champion changed
```

**Never write standalone training scripts. Never train locally.**
Local runs are unverifiable: all pre-2026-08-24 local results (including the original
EXP-007 local run) were voided because their numbers contradicted each other.

### Experiment Requirements

Every experiment MUST:

1. **Run on GitHub Actions** via `experiment-runner.yml` (push-triggered)
2. **Use real committed data** from `competitions/s6e8/data/train.csv`
3. **Use proper cross-validation** (stratified 5-fold — enforced by the runner)
4. **Report OOF ROC-AUC across ALL folds** (runner enforces this)
5. **Record full configuration** in the experiment YAML
6. **Explain reasoning** for the hypothesis and result

Artifacts (OOF predictions, submission.csv, results.json) are uploaded automatically
as Actions artifacts per experiment.

### Before Claiming Champion

- Result must be validated across ALL folds
- Result must beat current champion by meaningful margin (>0.0005)
- Operations agent must verify before promotion

## Quick Start

```bash
# 1. Validate data
python competitions/s6e8/validate_data.py

# 2. Read competition details
cat competitions/s6e8/README.md

# 3. Queue an experiment (copy EXP-007.yaml as template) and push — Actions does the rest
```

## What Gets You Banned

- Fabricating data or results
- **Training models locally or citing local scores**
- Claiming champion without proper GHA-validated results
- Pushing secrets to git
- Overwriting other agents work without reason
- Running one experiment and stopping
