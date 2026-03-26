from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .structure_features import STRUCTURE_FEATURES


@dataclass(frozen=True)
class FeatureStats:
    low: float
    high: float
    median: float
    iqr: float


class StructureProfileBank:
    def __init__(
        self,
        csv_path: Path,
        quantile_low: float,
        quantile_high: float,
    ) -> None:
        self.csv_path = csv_path
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.by_split_label: dict[tuple[str, str], dict[str, FeatureStats]] = {}
        self.by_split: dict[str, dict[str, FeatureStats]] = {}
        self.global_stats: dict[str, FeatureStats] | None = None
        self._load()

    def _compute_stats(self, df: pd.DataFrame) -> dict[str, FeatureStats]:
        stats: dict[str, FeatureStats] = {}
        for name in STRUCTURE_FEATURES:
            if name not in df.columns:
                continue
            x = pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(x) < 8:
                continue
            low = float(np.quantile(x, self.quantile_low))
            high = float(np.quantile(x, self.quantile_high))
            median = float(np.quantile(x, 0.50))
            q1 = float(np.quantile(x, 0.25))
            q3 = float(np.quantile(x, 0.75))
            iqr = max(q3 - q1, 1e-4)
            stats[name] = FeatureStats(low=low, high=high, median=median, iqr=iqr)
        return stats

    def _load(self) -> None:
        if not self.csv_path.exists():
            return
        df = pd.read_csv(self.csv_path)
        required = {"split", "label"}
        if not required.issubset(df.columns):
            return

        for split, split_df in df.groupby("split"):
            split_stats = self._compute_stats(split_df)
            if split_stats:
                self.by_split[str(split)] = split_stats
            for label, part in split_df.groupby("label"):
                key = (str(split), str(label))
                stats = self._compute_stats(part)
                if stats:
                    self.by_split_label[key] = stats

        self.global_stats = self._compute_stats(df)

    def get_target(self, split: str, label: str) -> dict[str, FeatureStats] | None:
        stats = self.by_split_label.get((split, label))
        if stats is not None:
            return stats
        stats = self.by_split.get(split)
        if stats is not None:
            return stats
        return self.global_stats

    def is_ready(self) -> bool:
        return bool(self.by_split_label or self.by_split or self.global_stats)

