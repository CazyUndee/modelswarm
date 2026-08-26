"""digit_split FE op tests — dataframe only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_experiment import apply_feature_engineering  # noqa: E402


def test_digit_split_basic_values():
    df = pd.DataFrame({"v": [7.89, 7.00, 12.34, np.nan]})
    cfg = {"feature_engineering": [{"op": "digit_split", "columns": ["v"]}]}
    out = apply_feature_engineering(df, cfg)
    assert out.loc[0, "d1_v"] == 8          # first decimal digit of 7.89
    assert out.loc[1, "d1_v"] == 0          # .00 -> 0
    assert out.loc[2, "d1_v"] == 3          # 12.34 -> 3
    assert np.isnan(out.loc[3, "d1_v"])     # NaN preserved
    assert abs(out.loc[0, "frac_v"] - 0.89) < 1e-9
    assert out.loc[1, "frac_v"] == 0.0


def test_weekend_slack_diff():
    df = pd.DataFrame({
        "weekend_screen_time": [5.0, 6.0],
        "social_media_hours": [2.0, 2.0],
        "gaming_hours": [1.0, 3.0],
    })
    cfg = {"feature_engineering": [{
        "op": "diff", "name": "weekend_slack",
        "terms": ["weekend_screen_time", "social_media_hours", "gaming_hours"],
    }]}
    out = apply_feature_engineering(df, cfg)
    assert out["weekend_slack"].tolist() == [2.0, 1.0]
