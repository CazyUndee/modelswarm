"""Submit the best candidate from the latest GHA analysis run.

Checks the most recent analysis output, finds the best submission CSV,
and submits it to Kaggle. Run after any analysis GHA run completes.

Usage: gh workflow run analysis.yml -f script_path=shared/analysis/submit_best.py
"""
import glob
import os
import subprocess
import sys

print("=== SUBMIT BEST CANDIDATE ===", flush=True)

# Find all submission CSVs in the working directory
csvs = sorted(glob.glob("submission_*.csv"))
print(f"Found {len(csvs)} submission CSVs", flush=True)
for c in csvs:
    size = os.path.getsize(c)
    print(f"  {c} ({size} bytes)", flush=True)

# Submit the most relevant candidates
# Priority: nonlinear > missingness > rgs > blend
priority_patterns = [
    "submission_nl_w0.35.csv",   # nonlinear meta, balanced weight
    "submission_nl_w0.30.csv",
    "submission_nl_w0.40.csv",
    "submission_miss_w0.35.csv",  # missingness features
    "submission_miss_w0.30.csv",
    "submission_rgs_w0.35.csv",   # rank-gauss stack
]

submitted = 0
for pattern in priority_patterns:
    matches = [c for c in csvs if os.path.basename(c) == pattern]
    if not matches:
        # Try partial match
        base = pattern.replace(".csv", "")
        matches = [c for c in csvs if base in c]
    
    for csv_path in matches:
        print(f"\nSubmitting {csv_path}...", flush=True)
        result = subprocess.run(
            ["kaggle", "competitions", "submit", "playground-series-s6-e8",
             "-f", csv_path, "-m", f"Codebuff {os.path.basename(csv_path)}"],
            capture_output=True, text=True
        )
        print(f"  stdout: {result.stdout}", flush=True)
        print(f"  stderr: {result.stderr}", flush=True)
        submitted += 1
        
        if submitted >= 2:
            print(f"\nSubmitted {submitted} candidates. Stopping.", flush=True)
            break
    
    if submitted >= 2:
        break

if submitted == 0:
    print("\nNo submission candidates found!", flush=True)
    # Try to submit any CSV found
    for csv_path in csvs:
        if "submission" in csv_path and csv_path.endswith(".csv"):
            print(f"Attempting: {csv_path}", flush=True)
            result = subprocess.run(
                ["kaggle", "competitions", "submit", "playground-series-s6-e8",
                 "-f", csv_path, "-m", f"Codebuff fallback {os.path.basename(csv_path)}"],
                capture_output=True, text=True
            )
            print(f"  stdout: {result.stdout}", flush=True)
            print(f"  stderr: {result.stderr}", flush=True)
            submitted += 1
            break

print(f"\n=== DONE: {submitted} submissions made ===", flush=True)
