"""pair_grid FE op tests — pure dataframe transforms, no model execution."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_experiment import apply_feature_engineering  # noqa: E402


def test_pair_grid_codes_are_deterministic_and_nan_safe():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "a": [1.23, 4.56, np.nan],
        "b": [7.89, 0.01, 2.22],
        "c": ["M", "F", "F"],
    })
    cfg = {"feature_engineering": [{"op": "pair_grid", "max_pairs": 10}]}
    out = apply_feature_engineering(df, cfg)
    assert "pair_a__b" in out.columns
    # deterministic integer codes; NaN side maps to sentinel bucket -1*1e7+code
    r0 = out.loc[0, "pair_a__b"]
    assert r0 == int(round(1.23 * 100) * 1e7 + round(7.89 * 100))
    # row with NaN 'a': sentinel = -1e7 + code(b)
    assert out.loc[2, "pair_a__b"] == int(-1e7 + round(2.22 * 100))
    # original columns untouched
    assert pd.isna(out.loc[2, "a"])


def test_pair_grid_max_pairs_and_exclude():
    df = pd.DataFrame({
        "id": range(5),
        "n1": np.arange(5) * 0.01,
        "n2": np.arange(5) * 0.02,
        "n3": np.arange(5) * 0.03,
        "g": list("abcab"),
    })
    cfg = {"feature_engineering": [{"op": "pair_grid", "exclude": ["n3"]}]}
    out = apply_feature_engineering(df, cfg)
    pairs = [c for c in out.columns if c.startswith("pair_")]
    assert pairs == ["pair_n1__n2"]
