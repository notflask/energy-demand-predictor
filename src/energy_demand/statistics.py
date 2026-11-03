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