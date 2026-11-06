import pandas as pd

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    local_time = result["utc_timestamp"].dt.tz_convert("Europe/Berlin")

    result["hour"] = local_time.dt.hour
    result["day_of_week"] = local_time.dt.day_of_week
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)

    return result

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["load_lag_24"] = result["load_mw"].shift(24)
    result["load_lag_168"] = result["load_mw"].shift(168)

    return result

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["rolling_mean_24"] = result["load_mw"].rolling(24).mean()
    result["rolling_mean_24_safe"] = result["load_mw"].shift(24).rolling(24).mean()

    return result