import numpy as np
import pytest

from energy_demand.statistics import analyze_demand, normalize_demand, demand_changes, moving_average

# analyze_demand
def test_analyze_demand():
    result = analyze_demand([40, 50, 60, 70])
    assert result["mean"] == 55.0

def test_analyze_empty():
    with pytest.raises(ValueError):
        analyze_demand([])

# normalize_demand
def test_normalize_demand():
    result = normalize_demand([40, 50, 60, 70])

    assert np.mean(result) == pytest.approx(0.0)
    assert np.std(result) == pytest.approx(1.0)

def test_normalize_constant():
    result = normalize_demand([50, 50, 50, 50])
    assert np.array_equal(result, np.zeros(4))

def test_normalize_empty():
    with pytest.raises(ValueError):
        normalize_demand([])

# demand_changes
def test_demand_changes():
    result = demand_changes([40, 45, 60, 55, 70])
    np.testing.assert_array_equal(result, [5, 15, -5, 15])

def test_demand_exactly_two_values():
    result = demand_changes([40, 60])
    np.testing.assert_array_equal(result, [20])

def test_demand_invalid_input():
    with pytest.raises(ValueError):
        demand_changes([40])

# moving_average
def test_moving_average():
    result = moving_average([40, 50, 60, 70, 80], window=3)
    np.testing.assert_array_equal(result, [50, 60, 70])

def test_moving_average_small_window():
    result = moving_average([40, 50, 60], window=1)
    np.testing.assert_array_equal(result, [40, 50, 60])

def test_moving_average_whole_array():
    result = moving_average([10, 20, 30, 40], window=4)
    np.testing.assert_array_equal(result, [25])

def test_moving_average_invalid_array():
    with pytest.raises(ValueError):
        moving_average([40, 50, 60], window=4)
