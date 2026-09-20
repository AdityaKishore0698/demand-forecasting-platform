"""Static hub attributes and features derived from (attribute, target date)."""
from __future__ import annotations

import pandas as pd

MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def add_hub_metadata_features(df: pd.DataFrame, hub_meta: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.merge(hub_meta, on="HubID", how="left", validate="many_to_one")

    months_since = (
        (out[date_col].dt.year - out["CompetitorOpenSinceYear"]) * 12
        + (out[date_col].dt.month - out["CompetitorOpenSinceMonth"])
    )
    out["MonthsSinceCompetitorOpen"] = months_since.clip(lower=0)
    out["HasKnownCompetitorOpenDate"] = out["CompetitorOpenSinceYear"].notna().astype(int)

    iso = out[date_col].dt.isocalendar()
    current_yearweek = iso["year"].astype("Int64") * 100 + iso["week"].astype("Int64")
    loyalty_start = (
        out["LoyaltyProgramSinceYear"].astype("Int64") * 100
        + out["LoyaltyProgramSinceWeek"].astype("Int64")
    )
    is_loyalty_active = (
        (out["LoyaltyProgram"] == 1) & loyalty_start.notna() & (current_yearweek >= loyalty_start)
    )
    out["IsLoyaltyActive"] = is_loyalty_active.astype(int)

    month_abbr = out[date_col].dt.month.map(MONTH_ABBR)
    interval = out["LoyaltyProgramInterval"].fillna("")
    in_interval = pd.Series(
        [m in iv.split(",") for m, iv in zip(month_abbr, interval)], index=out.index
    )
    out["IsLoyaltyPromoMonth"] = (is_loyalty_active & in_interval).astype(int)
    return out
