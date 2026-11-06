from pathlib import Path
import pandas as pd
import pytest

from energy_demand.preprocessing import load_demand_data


def generate_test_csv(
        csv_path: str | Path,
        utc_timestamps: list[str] = [
                "2020-01-01T00:00:00Z",
                "2020-01-01T01:00:00Z",
                "2020-01-01T02:00:00Z"
        ],
        load_mw: list[float | int] = [
            40000, 42000, 41000
        ]
) -> None:
    data = pd.DataFrame(
        {
            "utc_timestamp": utc_timestamps,
            "DE_load_actual_entsoe_transparency": load_mw
        }
    )

    data.to_csv(csv_path, index=False)



def test_load_valid_csv(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    generate_test_csv(csv_path)

    df = load_demand_data(csv_path)

    assert len(df) == 3
    assert list(df.columns) == ["utc_timestamp", "load_mw"]
    assert df["load_mw"].tolist() == [40000, 42000, 41000]
    assert str(df["utc_timestamp"].dtype) == "datetime64[us, UTC]"

def test_load_unsorted_csv(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    generate_test_csv(
        csv_path,
        utc_timestamps=[
                "2020-01-01T02:00:00Z",
                "2020-01-01T00:00:00Z",
                "2020-01-01T01:00:00Z"
        ],
    )

    df = load_demand_data(csv_path)

    assert df["utc_timestamp"].is_monotonic_increasing
    assert df["load_mw"].tolist() == [42000, 41000, 40000]
    assert df.index.tolist() == [0, 1, 2]

def test_load_duplicate_timestamps(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    generate_test_csv(
        csv_path,
        utc_timestamps=[
            "2020-01-01T00:00:00Z",
            "2020-01-01T01:00:00Z",
            "2020-01-01T01:00:00Z"
        ],
    )

    with pytest.raises(ValueError):
        load_demand_data(csv_path)

def test_load_missing_hour(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    generate_test_csv(
        csv_path,
        utc_timestamps=[
            "2020-01-01T00:00:00Z",
            "2020-01-01T01:00:00Z",
            "2020-01-01T03:00:00Z"
        ],
    )

    with pytest.raises(ValueError):
        load_demand_data(csv_path)

def test_load_missing_demand(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    generate_test_csv(
        csv_path,
        load_mw=[40000,None,41000]
    )

    df = load_demand_data(csv_path)

    assert df["load_mw"].isna().sum() == 1
