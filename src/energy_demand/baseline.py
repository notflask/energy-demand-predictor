import numpy as np

def predict_next_demand(data: list[float]) -> float:
    if not data:
        raise ValueError("Demand data should not be empty")

    return data[-1]