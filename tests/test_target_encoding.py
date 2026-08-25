"""Unit tests for leak-free target encoding helpers in run_experiment.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_experiment import _target_encode_apply, _target_encode_fit  # noqa: E402


def test_exact_formula_and_prior_fallback():
    df = pd.DataFrame({"v": [1, 1, 2, 2, 2]})
    y = pd.Series([0, 1, 1, 1, 0])
    stats = _target_encode_fit(df, y, ["v"], smoothing=50.0)
    assert stats["prior"] == pytest.approx(0.6)
    enc = stats["maps"]["v"]
    # group 1: (1 + 50*0.6)/(2+50); group 2: (2 + 30)/(3+50)
    assert enc.loc[1] == pytest.approx((1 + 50 * 0.6) / 52)
    assert enc.loc[2] == pytest.approx(32 / 53)
    # unseen value -> prior
    appl = _target_encode_apply(pd.DataFrame({"v": [1, 999]}), stats, ["v"])
    assert appl["te_v"].iloc[1] == pytest.approx(0.6)


def test_nan_group_is_explicit_not_dropped():
    # pandas>=3 groupby(dropna=True) default would silently drop NaN rows;
    # our fit must keep them as their own group.
    df = pd.DataFrame({"v": [1.0, 1.0, np.nan, np.nan]})
    y = pd.Series([0, 0, 1, 1])
    stats = _target_encode_fit(df, y, ["v"], smoothing=10.0)
    enc = stats["maps"]["v"]
    assert np.nan in enc.index or enc.index.isna().any()
    nan_val = float(enc.iloc[enc.index.get_indexer([np.nan])[0]])
    assert nan_val == pytest.approx((2 + 10 * 0.5) / 12)
    appl = _target_encode_apply(pd.DataFrame({"v": [np.nan]}), stats, ["v"])
    assert appl["te_v"].iloc[0] == pytest.approx(nan_val)


def test_no_leakage_stats_come_from_fit_frame_only():
    # A value present only in the APPLY frame must resolve to prior,
    # proving no apply-frame statistics flowed into the encoding.
    df_fit = pd.DataFrame({"v": ["a", "a", "b"]})
    y_fit = pd.Series([0, 0, 1])
    stats = _target_encode_fit(df_fit, y_fit, ["v"], smoothing=20.0)
    df_apply = pd.DataFrame({"v": ["a", "SECRET"]})
    out = _target_encode_apply(df_apply, stats, ["v"])
    assert out["te_v"].iloc[0] == pytest.approx((0 + 20 * (1 / 3)) / 22)
    assert out["te_v"].iloc[1] == pytest.approx(1 / 3)


def test_apply_does_not_mutate_input():
    df = pd.DataFrame({"v": [1, 2]})
    y = pd.Series([0, 1])
    stats = _target_encode_fit(df, y, ["v"], smoothing=5)
    appl = _target_encode_apply(df, stats, ["v"])
    assert list(df.columns) == ["v"]
    assert "te_v" in appl.columns
