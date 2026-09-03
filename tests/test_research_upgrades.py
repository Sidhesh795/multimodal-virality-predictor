"""
Unit tests for research-credibility target construction and leakage-safe splitters.
"""

import numpy as np
import pandas as pd

from src.data.splitters import LOCOSplitter, TemporalSplitter
from src.data.target_builder import (
    compute_virality_v1,
    compute_virality_v2,
    discretize_virality,
)


def test_compute_virality_v1_floor():
    views = np.array([1000.0, 500.0])
    followers = np.array([50.0, 200.0])
    v1 = compute_virality_v1(views, followers)
    np.testing.assert_allclose(v1, [1000.0 / 100.0, 500.0 / 200.0])


def test_compute_virality_v2_smoothing():
    views = np.array([2000.0])
    followers = np.array([500.0])
    v2 = compute_virality_v2(views, followers, k=1000)
    np.testing.assert_allclose(v2, [2000.0 / 1500.0])


def test_discretize_virality_percentile_bins():
    values = pd.Series(np.arange(1, 101, dtype=float))
    labels = discretize_virality(values)
    assert set(labels.unique()) == {0, 1, 2, 3}
    assert (labels.iloc[:50] == 0).all()
    assert (labels.iloc[50:75] == 1).all()
    assert (labels.iloc[75:90] == 2).all()
    assert (labels.iloc[90:] == 3).all()


def test_temporal_splitter_no_future_leakage():
    df = pd.DataFrame(
        {
            "created_at": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": np.arange(10),
        }
    )
    splitter = TemporalSplitter(df, timestamp_col="created_at", train_ratio=0.8)
    train_idx, test_idx = splitter.split()
    train_times = pd.to_datetime(df.iloc[train_idx]["created_at"])
    test_times = pd.to_datetime(df.iloc[test_idx]["created_at"])
    assert train_times.max() <= test_times.min()
    assert len(train_idx) == 8
    assert len(test_idx) == 2
    assert set(train_idx).isdisjoint(set(test_idx))


def test_loco_splitter_holds_out_each_category():
    df = pd.DataFrame(
        {
            "category_id": [0, 0, 1, 1, 2, 2],
            "x": np.arange(6),
        }
    )
    folds = list(LOCOSplitter(df, category_col="category_id"))
    held_out = {fold[2] for fold in folds}
    assert held_out == {0, 1, 2}
    for train_idx, test_idx, category in folds:
        assert set(df.iloc[test_idx]["category_id"].unique()) == {category}
        assert category not in set(df.iloc[train_idx]["category_id"].unique())
        assert set(train_idx).isdisjoint(set(test_idx))
