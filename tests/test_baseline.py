import numpy as np
import pytest

from energy_demand.baseline import predict_next_demand

def test_predict_next_demand():
    result = predict_next_demand([40, 50, 60, 70])
    assert result == 70

def test_predict_decreasing_demand():
    result = predict_next_demand([50, 45, 35])
    assert result == 35

def test_predict_single_value():
    result = predict_next_demand([80])
    assert result == 80

def test_predict_empty():
    with pytest.raises(ValueError):
        predict_next_demand([])
