"""
Leakage-safe dataset splitters for virality experiments.

TemporalSplitter: chronological hold-out.
LOCOSplitter: Leave-One-Category-Out cross-validation.
"""

from typing import Generator, Iterator, Tuple
import numpy as np
import pandas as pd

IndexArray = np.ndarray


class TemporalSplitter:
    """
    Chronological train/test split on a timestamp column.

    Earlier records go to train; later records go to test. This avoids
    leakage from future posts into the training window.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "created_at",
        train_ratio: float = 0.8,
    ):
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

        self.df = df
        self.timestamp_col = timestamp_col
        self.train_ratio = train_ratio

        timestamps = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        if timestamps.isna().any():
            raise ValueError(f"Column '{timestamp_col}' contains unparseable timestamps.")

        order = np.argsort(timestamps.to_numpy(), kind="mergesort")
        n_train = int(len(df) * train_ratio)
        if n_train == 0 or n_train == len(df):
            raise ValueError("Temporal split produced an empty train or test partition.")

        self.train_idx: IndexArray = order[:n_train]
        self.test_idx: IndexArray = order[n_train:]

    def split(self) -> Tuple[IndexArray, IndexArray]:
        """Return positional train and test indices (for DataFrame.iloc)."""
        return self.train_idx, self.test_idx

    def split_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return chronologically ordered train and test DataFrames."""
        train_idx, test_idx = self.split()
        return self.df.iloc[train_idx].copy(), self.df.iloc[test_idx].copy()


class LOCOSplitter:
    """
    Leave-One-Category-Out splitter.

    Yields (train_idx, test_idx, held_out_category) for each unique category
    so generalization to unseen content categories can be measured.
    """

    def __init__(self, df: pd.DataFrame, category_col: str = "category_id"):
        if category_col not in df.columns:
            raise ValueError(f"Category column '{category_col}' not found in DataFrame.")
        self.df = df
        self.category_col = category_col
        self.categories = np.array(sorted(df[category_col].dropna().unique().tolist()))

    def split(self) -> Generator[Tuple[IndexArray, IndexArray, object], None, None]:
        """Generator of positional train/test indices per held-out category."""
        values = self.df[self.category_col].to_numpy()
        all_idx = np.arange(len(self.df))
        for held_out in self.categories:
            test_mask = values == held_out
            test_idx = all_idx[test_mask]
            train_idx = all_idx[~test_mask]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx, held_out

    def __iter__(self) -> Iterator[Tuple[IndexArray, IndexArray, object]]:
        return self.split()
