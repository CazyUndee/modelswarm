"""Nested (inner-CV out-of-fold) target encoding tests — no model execution."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_experiment import _nested_target_encode_train  # noqa: E402


def test_nested_te_excludes_own_label_for_lonely_levels():
    # A level with exactly ONE row must be encoded WITHOUT that row's own y.
    # With y=1 for the lonely row, its TE must equal the smoothed prior of the
    # OTHER rows only -- if own-label leaked in, it would be materially higher.
    n = 200
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "v": [f"lonely_{i}" if i < 5 else f"common_{i % 3}" for i in range(n)],
        "x": rng.rand(n),
    })
    y = pd.Series(rng.rand(n) < 0.3, dtype=int)
    y.iloc[:5] = 1  # lonely levels all positive
    enc, stats = _nested_target_encode_train(df, y, ["v", "x"], smoothing=10.0,
                                             n_inner=5, seed=42)
    lonely = enc[enc["v"].str.startswith("lonely")]
    base = float(np.mean(y.iloc[5:]))
    expected = (1 + 10 * base) / (11 + 10)  # sum from other inner-train rows ~0 + prior pull
    # each lonely row's TE must be near the no-own-label estimate, NOT near 1.0
    assert (lonely["te_v"] < 0.5).all(), f"own-label leak suspected: {lonely['te_v'].tolist()}"
    assert abs(lonely["te_v"].iloc[0] - expected) < 0.15


def test_nested_te_full_fit_stats_transform_val():
    df = pd.DataFrame({"v": ["a"] * 50 + ["b"] * 50})
    y = pd.Series([1] * 50 + [0] * 50)
    enc, stats = _nested_target_encode_train(df, y, ["v"], smoothing=10.0,
                                             n_inner=5, seed=42)
    assert not enc["te_v"].isna().any()
    assert stats["maps"]["v"].loc["a"] == pytest.approx((50 + 10 * 0.5) / 60)
    # freq column present and log1p-shaped
    assert enc["freq_v"].between(0, np.log1p(60)).all()
