import numpy as np

def analyze_demand(data: list[float]) -> dict[str, float]:
    """
    Calculates descriptive statistics for electricity demand
    """
    if not data:
        raise ValueError("Demand data cannot be empty")

    return {
        "mean": np.mean(data),
        "min": np.min(data),
        "max": np.max(data),
        "std": np.std(data)
    }

def normalize_demand(data: list[float]) -> np.ndarray:
    """
    Standardize electricity demand using Z-Scores
    """
    if not data:
        raise ValueError("Demand data cannot be empty")

    values = np.asarray(data, dtype=float)

    mean = np.mean(values)
    std = np.std(values, ddof=0)

    if std == 0:
        return np.zeros_like(values)

    return (values - mean) / std

def demand_changes(data: list[float]) -> np.ndarray:
    if len(data) < 2:
        raise ValueError("Demand data should have more than two milestones")

    values = np.asarray(data, dtype=float)

    previous_values = values[:-1]
    next_values = values[1:]

    return next_values - previous_values

def moving_average(
        data: list[float],
        window: int = 3
) -> np.ndarray:
    if window < 1 or window > len(data):
        raise ValueError("Window should be bigger than one and less than the length of the array")

    result = []

    for index in range(0, len(data) - window + 1):
        values = data[index:index+window]
        result.append(np.mean(values))

    return np.asarray(result, dtype=float)