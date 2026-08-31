"""Consume nonlinear meta results from GHA artifacts.

Downloads the most recent analysis run output, parses the LightGBM/XGBoost
results, and recommends whether to submit.

Usage: python shared/analysis/consume_nonlinear.py
"""
import glob
import json
import os
import subprocess
import sys

print("=== CONSUMING NONLINEAR META RESULTS ===", flush=True)

# Find the most recent completed analysis run
result = subprocess.run(
    ["gh", "run", "list", "--workflow", "analysis.yml", "--status", "completed",
     "--limit", "10", "--json", "databaseId,createdAt,headSha"],
    capture_output=True, text=True
)
runs = json.loads(result.stdout)
print(f"Found {len(runs)} completed runs", flush=True)

if not runs:
    print("No completed runs found. Check: gh run list --workflow analysis.yml", flush=True)
    sys.exit(1)

# Try each run, most recent first
for run_info in runs:
    run_id = run_info["databaseId"]
    sha = run_info.get("headSha", "unknown")[:8]
    print(f"\n--- Run {run_id} (sha={sha}) ---", flush=True)

    dl_dir = f"/tmp/consume_{run_id}"
    os.makedirs(dl_dir, exist_ok=True)
    dl = subprocess.run(
        ["gh", "run", "download", str(run_id), "-D", dl_dir],
        capture_output=True, text=True
    )
    if dl.returncode != 0:
        print(f"  Download failed: {dl.stderr.strip()}", flush=True)
        continue

    # Check for submission CSVs
    submissions = sorted(glob.glob(os.path.join(dl_dir, "submission_*.csv")))
    if submissions:
        print(f"  Submission CSVs found:", flush=True)
        for s in submissions:
            print(f"    {os.path.basename(s)} ({os.path.getsize(s)} bytes)", flush=True)

    # Parse the analysis output for meta-ensemble comparison
    output_file = os.path.join(dl_dir, "analysis_output.txt")
    if os.path.exists(output_file):
        with open(output_file, "r", errors="replace") as f:
            lines = f.readlines()

        print(f"\n  Key output lines:", flush=True)
        for line in lines:
            line = line.strip()
            # Skip download progress, print important results
            if any(kw in line.lower() for kw in [
                "lightgbm", "xgboost", "linear", "meta-ensemble", "best:",
                "oof=", "delta=", "using", "emitted", "failed", "error",
                "nonlinear", "fold", "imported", "version",
                "stack oof", "submission_nl", "submission_miss",
            ]):
                if not any(skip in line.lower() for skip in ["download", "install", "collect", "pip"]):
                    print(f"    >> {line}", flush=True)

    # If this run produced submissions, we're done
    if submissions:
        print(f"\n  RUN COMPLETE. Submissions ready for manual dispatch.", flush=True)
        print(f"  To submit locally, copy the CSV and run:", flush=True)
        print(f"    kaggle competitions submit playground-series-s6-e8 -f <csv> -m 'Codebuff'", flush=True)
        break
    else:
        print(f"  No submissions produced by this run.", flush=True)

print("\n=== DONE ===", flush=True)
