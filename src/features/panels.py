"""Wide (date x hub) calendar panels.

Reindexing onto a *complete daily calendar* is what makes ``shift``/``rolling``
mean "N calendar days", not "N rows". A hub with a missing block of dates gets
NaN there instead of silently pulling values from before the gap.
"""
from __future__ import annotations

import pandas as pd

from src.data.schema import DATE_COL, ENTITY_COL


def build_panel(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wide = df.pivot(index=DATE_COL, columns=ENTITY_COL, values=value_col)
    full_idx = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    wide = wide.reindex(full_idx)
    wide.index.name = DATE_COL
    return wide
