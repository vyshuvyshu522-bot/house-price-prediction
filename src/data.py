"""Loading, cleaning and splitting the King County house-sales data."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src import config

MAX_PLAUSIBLE_BEDROOMS = 15  # the raw data contains a 33-bedroom row (typo)


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the CSV, failing with a clear message if it is missing."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found at {path}. See data/README.md.") from exc


def clean_data(df: pd.DataFrame):
    """Return (clean_df, report). The report counts rows removed by each rule."""
    required = {config.TARGET, config.GROUP_COLUMN, *config.FEATURES}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_cols)}")

    report = {"rows_raw": int(len(df))}
    df = df[[config.GROUP_COLUMN, config.TARGET, *config.FEATURES]].copy()

    # Force numeric columns to numbers; unparseable values become NaN and are
    # handled later by the imputer inside the pipeline (fit on training data only).
    for col in [config.TARGET, *config.NUMERIC_FEATURES]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n = len(df)
    df = df.drop_duplicates()
    report["duplicate_rows_removed"] = n - len(df)

    # A row without a target cannot be used for supervised learning.
    n = len(df)
    df = df[df[config.TARGET].notna() & (df[config.TARGET] > 0)]
    report["missing_or_nonpositive_price_removed"] = n - len(df)

    n = len(df)
    df = df[~(df["sqft_living"] <= 0)]
    report["nonpositive_living_area_removed"] = n - len(df)

    n = len(df)
    df = df[~(df["bedrooms"] > MAX_PLAUSIBLE_BEDROOMS)]
    report["implausible_bedrooms_removed"] = n - len(df)

    # ZIP codes are categories, not quantities: store as strings.
    df["zipcode"] = df["zipcode"].map(lambda z: str(int(float(z))) if pd.notna(z) else np.nan)

    df = df.reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    report["missing_values_in_features"] = int(df[config.FEATURES].isna().sum().sum())
    return df, report


def split_train_test(df: pd.DataFrame, test_size=config.TEST_SIZE, seed=config.RANDOM_STATE):
    """Group-aware split: all sales of the same house id land on the same side.

    Without this, a house sold twice could appear in both train and test, which
    would leak information and inflate the test scores.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[config.GROUP_COLUMN]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
