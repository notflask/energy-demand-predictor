import pandas as pd
from pathlib import Path

def load_demand_data(csv_path: str | Path) -> pd.DataFrame:
    if isinstance(csv_path, str):
        csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise ValueError(f"CSV doesnt exist by path '{csv_path}'")

    df = pd.read_csv(
        csv_path,
        usecols=["utc_timestamp", "DE_load_actual_entsoe_transparency"]
    )

    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    df = df.sort_values(by="utc_timestamp")

    df = df.reset_index(drop=True)

    duplicates = df["utc_timestamp"].duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"Data set has {duplicates} duplicates")

    missng_hours = (df["utc_timestamp"].diff().iloc[1:] != pd.Timedelta(hours=1)).sum()
    if missng_hours > 0:
        raise ValueError(f"Data set has {missng_hours} missing hours")

    df = df.rename(columns={"DE_load_actual_entsoe_transparency": "load_mw"})

    return df


