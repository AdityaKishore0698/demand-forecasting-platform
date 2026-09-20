"""Plain-English names, groups and descriptions for every model feature.

The dashboard shows these instead of raw column names, so a non-ML reader can
understand what the model is using.
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

GROUPS = {
    "schedule": "Operating schedule",
    "promo": "Promotions & holidays",
    "calendar": "Calendar",
    "recent": "Recent demand",
    "baseline": "Store baseline",
    "hub": "Store attributes",
    "horizon": "Forecast horizon",
}

_STATIC: Dict[str, Tuple[str, str, str]] = {
    "IsOpen": ("Store open that day", "schedule",
               "Whether the store is scheduled to be open. Closed days always have zero orders, so this is the single strongest signal."),
    "IsWeekend": ("Weekend", "calendar", "Saturday or Sunday."),
    "Weekday": ("Day of week", "calendar", "Demand follows a strong weekly rhythm; most stores are closed on Sundays."),
    "Month": ("Month", "calendar", "Seasonal level (e.g. December peaks)."),
    "DayOfMonth": ("Day of month", "calendar", "Captures pay-day / month-start and month-end effects."),
    "WeekOfYear": ("Week of year", "calendar", "A smooth annual seasonality signal."),
    "PromoActive": ("Promotion running", "promo", "Promotion days lift demand noticeably on average."),
    "RegionalHoliday": ("Regional holiday", "promo", "Public/regional holiday code for the day."),
    "SchoolClosureFlag": ("School closure", "promo", "Schools closed that day (changes household routines)."),
    "h": ("Forecast horizon (days ahead)", "horizon",
          "How far past the last known day the target is. Lets the model trust recent history less for later days."),
    "HubID": ("Store identity", "hub", "Lets the model learn store-specific levels beyond the numeric features."),
    "HubFormat": ("Store format", "hub", "Format/type of the store."),
    "AssortmentTier": ("Assortment tier", "hub", "Breadth of the product range the store carries."),
    "CompetitorDistance": ("Distance to nearest competitor", "hub", "Closer competitors can cap demand."),
    "MonthsSinceCompetitorOpen": ("Months since competitor opened", "hub", "How long the nearest competitor has been trading."),
    "HasKnownCompetitorOpenDate": ("Competitor opening date known", "hub", "Whether the competitor's opening date is recorded."),
    "LoyaltyProgram": ("Loyalty programme store", "hub", "Whether the store participates in the recurring loyalty programme."),
    "IsLoyaltyActive": ("Loyalty programme active", "hub", "Whether the programme has started by this date."),
    "IsLoyaltyPromoMonth": ("Loyalty promo month", "hub", "Whether this month is one of the store's recurring loyalty-promo months."),
    "OV_hubweekday_mean": ("Store's typical demand on this weekday", "baseline",
                           "Average demand this specific store has historically had on this weekday. Combines the weekly rhythm with store-level differences."),
    "OV_expanding_mean": ("Store's long-run average demand", "baseline", "Average demand over the store's entire known history."),
}

_NEIGHBOR = re.compile(r"^(IsOpen|PromoActive|RegionalHoliday|SchoolClosureFlag)_(next|prev)(\d+)$")
_LAG = re.compile(r"^OV_lag(\d+)$")
_ROLLMEAN = re.compile(r"^OV_rollmean(\d+)$")
_ROLLSTD = re.compile(r"^OV_rollstd(\d+)$")

_NEIGHBOR_NAMES = {
    "IsOpen": "Open", "PromoActive": "Promotion",
    "RegionalHoliday": "Holiday", "SchoolClosureFlag": "School closure",
}


def describe_feature(name: str) -> Dict[str, str]:
    if name in _STATIC:
        label, group, desc = _STATIC[name]
    elif _LAG.match(name):
        k = int(_LAG.match(name).group(1))
        label = "Last known day's demand" if k == 1 else f"Demand {k - 1} days before the last known day"
        group, desc = "recent", "Demand on a specific past day, measured from the last day with known data."
    elif _ROLLMEAN.match(name):
        w = _ROLLMEAN.match(name).group(1)
        label, group = f"Average demand, last {w} days", "recent"
        desc = f"Smoothed recent demand level over the {w} days up to the last known day."
    elif _ROLLSTD.match(name):
        w = _ROLLSTD.match(name).group(1)
        label, group = f"Demand volatility, last {w} days", "recent"
        desc = f"How much the store's demand has fluctuated over the last {w} days."
    elif _NEIGHBOR.match(name):
        col, side, n = _NEIGHBOR.match(name).groups()
        when = "tomorrow" if side == "next" else "yesterday"
        label = f"{_NEIGHBOR_NAMES[col]} {when}"
        group = "schedule" if col == "IsOpen" else "promo"
        desc = f"Known schedule value for the day {'after' if side == 'next' else 'before'} the target date (e.g. promotion starting/ending)."
    else:
        label, group, desc = name, "hub", ""
    return {"feature": name, "label": label, "group": group, "group_label": GROUPS[group], "description": desc}
