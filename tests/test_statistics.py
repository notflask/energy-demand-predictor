import numpy as np
import pytest

from energy_demand.statistics import analyze_demand, normalize_demand


def test_analyze_demand():
    result = analyze_demand([40, 50, 60, 70])
    assert result["mean"] == 55.0

def test_analyze_empty():
    with pytest.raises(ValueError):
        analyze_demand([])

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


