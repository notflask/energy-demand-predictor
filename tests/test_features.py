import pandas as pd
import pytest

from energy_demand.features import add_calendar_features, add_lag_features, add_rolling_features


def test_calendar_features_weekday():
    df = pd.DataFrame({
        "utc_timestamp": pd.to_datetime(
            ["2020-07-01T12:00:00Z"],
            utc=True
        ),
        "load_mw": [50000]
    })

    result = add_calendar_features(df)

    assert result["hour"].iloc[0] == 14
    assert result["day_of_week"].iloc[0] == 2
    assert result["is_weekend"].iloc[0] == 0

def test_calendar_features_weekend():
    df = pd.DataFrame({
        "utc_timestamp": pd.to_datetime(
            ["2020-07-04T22:00:00Z"],
            utc=True
        ),
        "load_mw": [50000]
    })

    result = add_calendar_features(df)

    assert result["hour"].iloc[0] == 0
    assert result["day_of_week"].iloc[0] == 6
    assert result["is_weekend"].iloc[0] == 1

def test_calendar_features_no_mutation():
    df = pd.DataFrame({
        "utc_timestamp": pd.to_datetime(
            ["2020-07-04T22:00:00Z"],
            utc=True
        ),
        "load_mw": [50000]
    })
    copy = df.copy()

    add_calendar_features(df)

    assert copy.equals(df)

def test_lag_features_values():
    df = pd.DataFrame({
        "utc_timestamp": pd.date_range(
            start="2020-01-01",
            periods=200,
            freq="h",
            tz="UTC"
        ),
        "load_mw": [40000 + i * 100 for i in range(200)]
    })

    result = add_lag_features(df)

    assert result["load_lag_24"].iloc[24] == 40000
    assert result["load_lag_24"].iloc[25] == 40100
    assert result["load_lag_24"].iloc[168] == 54400
    assert result["load_lag_168"].iloc[168] == 40000
    assert result["load_lag_168"].iloc[169] == 40100
    assert result["load_lag_168"].iloc[199] == 43100

def test_lag_features_nan():
    df = pd.DataFrame({
        "utc_timestamp": pd.date_range(
            start="2020-01-01",
            periods=200,
            freq="h",
            tz="UTC"
        ),
        "load_mw": [40000 + i * 100 for i in range(200)]
    })

    result = add_lag_features(df)

    assert result["load_lag_24"].iloc[0:24].isna().sum() == 24
    assert result["load_lag_168"].iloc[0:168].isna().sum() == 168

def test_lag_features_no_mutation():
    df = pd.DataFrame({
        "utc_timestamp": pd.date_range(
            start="2020-01-01",
            periods=200,
            freq="h",
            tz="UTC"
        ),
        "load_mw": [40000 + i * 100 for i in range(200)]
    })
    copy = df.copy()

    add_lag_features(df)

    assert copy.equals(df)

def test_rolling_features_values():
    df = pd.DataFrame({
        "utc_timestamp": pd.date_range(
            start="2020-01-01",
            periods=200,
            freq="h",
            tz="UTC"
        ),
        "load_mw": [40000 + i * 100 for i in range(200)]
    })

    result = add_rolling_features(df)

    assert result["rolling_mean_24_safe"].iloc[47] == pytest.approx(41150)
    assert result["rolling_mean_24_safe"].iloc[48] == pytest.approx(41250)

def test_rolling_features_nan():
    df = pd.DataFrame({
        "utc_timestamp": pd.date_range(
            start="2020-01-01",
            periods=200,
            freq="h",
            tz="UTC"
        ),
        "load_mw": [40000 + i * 100 for i in range(200)]
    })

    result = add_rolling_features(df)

    assert result["rolling_mean_24_safe"].iloc[:47].isna().sum() == 47

def test_rolling_features_no_mutation():
    df = pd.DataFrame({
        "utc_timestamp": pd.date_range(
            start="2020-01-01",
            periods=200,
            freq="h",
            tz="UTC"
        ),
        "load_mw": [40000 + i * 100 for i in range(200)]
    })

    original = df.copy()
    add_rolling_features(df)
    pd.testing.assert_frame_equal(original, df)