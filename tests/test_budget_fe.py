"""budget_constraint + impute_median FE op tests — dataframe only, no models."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_experiment import apply_feature_engineering  # noqa: E402


def test_budget_constraint_bounds_and_flags():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "daily": [10.0, 8.0, np.nan, 5.0],
        "social": [2.0, np.nan, 1.0, 4.9],
        "gaming": [3.0, 3.0, np.nan, 0.2],
        "work": [1.0, 1.0, 1.0, np.nan],
    })
    cfg = {"feature_engineering": [{
        "op": "budget_constraint",
        "daily": "daily", "components": ["social", "gaming", "work"],
        "day_hours": 24.0, "prefix": "bud_",
    }]}
    out = apply_feature_engineering(df, cfg)
    r = out.iloc[0]
    assert r["bud_slack"] == 4.0                      # 10 - (2+3+1)
    assert r["bud_n_observed"] == 4
    assert r["bud_viol_social"] == 0.0                # nothing missing -> no violation
    r2 = out.iloc[1]
    assert r2["bud_n_missing"] == 1
    # social missing: room = daily - other observed components = 8 - (3+1) = 4
    assert r2["bud_viol_social"] == pytest_approx(4.0 - df["social"].median())
    assert r2["bud_slack"] == pytest_approx(8.0 - (0.0 + 3.0 + 1.0))


def pytest_approx(v):
    return v


def test_impute_median_fills_and_keeps_other_frames():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", None]})
    cfg = {"feature_engineering": [{"op": "impute_median", "columns": ["a"]}]}
    out = apply_feature_engineering(df, cfg)
    assert out["a"].tolist() == [1.0, 2.0, 3.0]
    assert out["b"].iloc[2] is None or (isinstance(out["b"].iloc[2], float) and np.isnan(out["b"].iloc[2]))
