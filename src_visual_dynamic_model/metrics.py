import numpy as np


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)
    covariance = np.nanmean((x - x_mean) * (y - y_mean))
    x_var = np.nansum((x - x_mean) ** 2) / max(len(x) - 1, 1)
    y_var = np.nansum((y - y_mean) ** 2) / max(len(y) - 1, 1)
    return float((2.0 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2 + 1e-12))

