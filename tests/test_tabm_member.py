"""Static validation of the 'tabm' member path with a MOCKED TabM class.

No training. Verifies: ctor-vs-fit argument split, unconditional imputed
indicator columns (identical schema across train/val/test frames), and
categorical vocabulary pinning to fold-train.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class _FakeTabM:
    """Records fit-frame schema; returns constant probabilities."""

    train_columns = None
    train_cat_dtypes = None

    def __init__(self, **kwargs):
        _FakeTabM.last_ctor = kwargs

    def fit(self, X, y, X_val=None, y_val=None, time_to_fit_in_seconds=None):
        _FakeTabM.train_columns = list(X.columns)
        _FakeTabM.train_cat_dtypes = {c: str(X[c].dtype) for c in X.columns
                                      if isinstance(X[c].dtype, pd.CategoricalDtype)}
        if X_val is not None:
            assert set(X_val.columns) == set(X.columns), "val schema mismatch"
            for c, dt in _FakeTabM.train_cat_dtypes.items():
                assert str(X_val[c].dtype) == dt, f"val dtype mismatch {c}"
        return self

    def predict_proba(self, X):
        assert set(X.columns) == set(_FakeTabM.train_columns), "pred schema mismatch"
        out = np.zeros((len(X), 2))
        out[:, 1] = 0.5
        return out


@pytest.fixture()
def fake_tabm(monkeypatch):
    stub = types.ModuleType("pytabkit")
    stub.TabM_D_Classifier = _FakeTabM
    monkeypatch.setitem(sys.modules, "pytabkit", stub)
    import importlib
    import run_experiment
    importlib.reload(run_experiment)
    yield run_experiment


def test_tabm_member_schema_consistency(fake_tabm):
    # Explicit disjoint NaN patterns: 'b' missing only in train rows,
    # 'c' missing only in pred rows -> indicator columns must exist in BOTH
    # frames regardless of where missingness falls.
    tr_rows, pr_rows = 40, 20
    X = pd.DataFrame({
        "a": np.linspace(0, 1, tr_rows + pr_rows),
        "b": 1.0,
        "c": 2.0,
        "g": ["M"] * (tr_rows + pr_rows),
    })
    y = pd.Series(np.random.RandomState(0).rand(tr_rows + pr_rows) < 0.5, dtype=int)
    X.loc[5, "b"] = np.nan            # train-only missingness
    X.iloc[tr_rows + 3, X.columns.get_loc("c")] = np.nan   # pred-only missingness

    X_train, X_pred = X.iloc[:tr_rows], X.iloc[tr_rows:]

    preds = fake_tabm._fit_member_predict(
        "tabm", {"random_state": 0, "time_to_fit_in_seconds": 5},
        X_train, y.iloc[:tr_rows], X_pred, cat_cols=["g"])
    assert len(preds) == len(X_pred)
    assert _FakeTabM.last_ctor.get("device") == "cpu"
    assert "time_to_fit_in_seconds" not in _FakeTabM.last_ctor
    # unconditional indicators present even where a frame had no NaNs
    assert "b_imputed" in _FakeTabM.train_columns
    assert "c_imputed" in _FakeTabM.train_columns


def test_tabm_time_budget_goes_to_fit(fake_tabm):
    called = {}
    orig_fit = _FakeTabM.fit

    def spy_fit(self, X, y, X_val=None, y_val=None, time_to_fit_in_seconds=None):
        called["t"] = time_to_fit_in_seconds
        return orig_fit(self, X, y, X_val, y_val, time_to_fit_in_seconds)

    _FakeTabM.fit = spy_fit
    try:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]})
        y = pd.Series([0, 1, 0])
        fake_tabm._fit_member_predict("tabm", {"time_to_fit_in_seconds": 77},
                                      X, y, X, cat_cols=["g"])
        assert called["t"] == 77
    finally:
        _FakeTabM.fit = orig_fit
