"""
Virality target construction utilities.

Defines follower-normalized engagement ratios (V1, V2) and percentile-based
discretization into Low / Medium / High / Viral classes.
"""

from typing import Union
import numpy as np
import pandas as pd

ArrayLike = Union[pd.Series, np.ndarray, list]


def compute_virality_v1(views: ArrayLike, followers: ArrayLike) -> np.ndarray:
    """
    Follower-normalized virality with a floor on audience size.

    V1 = views / max(followers, 100)
    """
    views_arr = np.asarray(views, dtype=float)
    followers_arr = np.asarray(followers, dtype=float)
    return views_arr / np.maximum(followers_arr, 100.0)


def compute_virality_v2(
    views: ArrayLike, followers: ArrayLike, k: float = 1000
) -> np.ndarray:
    """
    Additive-smoothed virality that dampens small-account noise.

    V2 = views / (followers + k)
    """
    views_arr = np.asarray(views, dtype=float)
    followers_arr = np.asarray(followers, dtype=float)
    return views_arr / (followers_arr + float(k))


def discretize_virality(v_series: ArrayLike) -> pd.Series:
    """
    Map a continuous virality series to 4 classes using percentile cuts.

    Thresholds:
        <= P50 -> 0 Low
        <= P75 -> 1 Medium
        <= P90 -> 2 High
         > P90 -> 3 Viral
    """
    series = pd.Series(v_series, dtype=float)
    p50 = float(series.quantile(0.50))
    p75 = float(series.quantile(0.75))
    p90 = float(series.quantile(0.90))

    labels = np.zeros(len(series), dtype=int)
    values = series.to_numpy()
    labels[values > p50] = 1
    labels[values > p75] = 2
    labels[values > p90] = 3

    return pd.Series(labels, index=series.index, name="virality_class")
