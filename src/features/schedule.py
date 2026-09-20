"""Known-in-advance schedule features (open / promo / holiday / school closure).

Unlike demand, the schedule for the whole forecast window is given up front, so
"is a promo starting tomorrow?" can be built on the *target date itself* without
using any unknown outcome. These features never touch the target column.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.data.schema import DATE_COL, ENTITY_COL, SCHEDULE_COLS


def build_schedule_panels(schedule: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """``schedule``: long frame with HubID, Date and SCHEDULE_COLS (history + future)."""
    full_idx = pd.date_range(schedule[DATE_COL].min(), schedule[DATE_COL].max(), freq="D")
    panels = {}
    for col in SCHEDULE_COLS:
        wide = schedule.pivot(index=DATE_COL, columns=ENTITY_COL, values=col)
        panels[col] = wide.reindex(full_idx)
    return panels


def compute_schedule_neighbor_features(panels: Dict[str, pd.DataFrame], shifts: List[int]) -> pd.DataFrame:
    """Neighbouring-day schedule values. shift -1 => ``next1`` (tomorrow), +1 => ``prev1``."""
    long_frames = []
    for col, panel in panels.items():
        for shift in shifts:
            label = f"next{abs(shift)}" if shift < 0 else f"prev{shift}"
            s = panel.shift(shift).stack(future_stack=True)
            s.name = f"{col}_{label}"
            long_frames.append(s)
    out = pd.concat(long_frames, axis=1)
    out.index.names = ["Date", "HubID"]
    return out.reset_index()
