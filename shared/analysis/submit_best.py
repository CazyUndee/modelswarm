"""Submit the best candidate from the latest GHA analysis run.

This script runs on GHA where kaggle credentials may not exist.
It prints the recommended submission for the user to run locally.
To actually submit, run locally: python shared/analysis/submit_best.py --submit

Usage: gh workflow run analysis.yml -f script_path=shared/analysis/submit_best.py
"""
import glob
import os
import subprocess
import sys

DO_SUBMIT = "--submit" in sys.argv

print("=== SUBMIT BEST CANDIDATE ===", flush=True)
print(f"Mode: {'SUBMIT' if DO_SUBMIT else 'DRY RUN (add --submit to actually submit)'}", flush=True)

# Find all submission CSVs in the working directory
csvs = sorted(glob.glob("submission_*.csv"))
print(f"Found {len(csvs)} submission CSVs", flush=True)
for c in csvs:
    size = os.path.getsize(c)
    print(f"  {c} ({size} bytes)", flush=True)

if not csvs:
    print("No submission CSVs found. Nothing to do.", flush=True)
    sys.exit(0)

# Submit only the highest-priority candidate (don't burn submission slots)
priority_patterns = [
    "submission_nl_w0.35.csv",    # nonlinear meta, balanced weight
    "submission_nl_w0.30.csv",
    "submission_nl_w0.40.csv",
    "submission_miss_w0.35.csv",   # missingness features
    "submission_miss_w0.30.csv",
    "submission_rgs_w0.35.csv",    # rank-gauss stack (safe fallback)
]

best = None
for pattern in priority_patterns:
    for c in csvs:
        if os.path.basename(c) == pattern:
            best = c
            break
    if best:
        break

if not best:
    # Last resort: pick the first submission CSV
    best = csvs[0]
    print(f"WARNING: No priority match, using {best}", flush=True)

print(f"\nRecommended submission: {best}", flush=True)

if DO_SUBMIT:
    result = subprocess.run(
        ["kaggle", "competitions", "submit", "playground-series-s6-e8",
         "-f", best, "-m", f"Codebuff {os.path.basename(best)}"],
        capture_output=True, text=True
    )
    print(f"Result: {result.stdout}", flush=True)
    if result.returncode != 0:
        print(f"Error (exit {result.returncode}): {result.stderr}", flush=True)
else:
    print(f"\nTo submit locally, run:", flush=True)
    print(f"  kaggle competitions submit playground-series-s6-e8 -f {best} -m 'Codebuff {os.path.basename(best)}'", flush=True)

print("\n=== DONE ===", flush=True)
