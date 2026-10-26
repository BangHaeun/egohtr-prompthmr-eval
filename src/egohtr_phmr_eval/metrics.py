from __future__ import annotations

import numpy as np


def compute_mpjpe(prediction: np.ndarray, ground_truth: np.ndarray):
    prediction = np.asarray(prediction, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if prediction.shape != (25, 3) or ground_truth.shape != (25, 3):
        raise ValueError(f"Expected matching BODY_25 arrays (25,3), got {prediction.shape} and {ground_truth.shape}")
    error_mm = np.linalg.norm(prediction - ground_truth, axis=1) * 1000.0
    return error_mm, float(error_mm.mean())
