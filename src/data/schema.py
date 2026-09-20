"""Column names and dataset contracts shared across the pipeline."""

ENTITY_COL = "HubID"        # one fulfilment hub / store
DATE_COL = "Date"
TARGET_COL = "OrderVolume"  # daily orders at a hub (non-negative integer)

# Operational attributes that are known IN ADVANCE for every future day
# (a planned schedule), so they are legitimate inputs at inference time.
SCHEDULE_COLS = ["IsOpen", "PromoActive", "RegionalHoliday", "SchoolClosureFlag"]

DAILY_ATTR_COLS = [ENTITY_COL, DATE_COL, "Weekday"] + SCHEDULE_COLS

TRAIN_REQUIRED = DAILY_ATTR_COLS + [TARGET_COL]
FUTURE_REQUIRED = DAILY_ATTR_COLS               # future calendar: no target
HUB_META_REQUIRED = [
    ENTITY_COL, "HubFormat", "AssortmentTier", "CompetitorDistance",
    "CompetitorOpenSinceMonth", "CompetitorOpenSinceYear", "LoyaltyProgram",
    "LoyaltyProgramSinceWeek", "LoyaltyProgramSinceYear", "LoyaltyProgramInterval",
]
