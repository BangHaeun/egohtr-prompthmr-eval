from __future__ import annotations

import numpy as np


def compute_mpjpe(prediction: np.ndarray, ground_truth: np.ndarray):
    prediction = np.asarray(prediction, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if prediction.shape != (25, 3) or ground_truth.shape != (25, 3):
        raise ValueError(f"Expected matching BODY_25 arrays (25,3), got {prediction.shape} and {ground_truth.shape}")
    error_mm = np.linalg.norm(prediction - ground_truth, axis=1) * 1000.0
    return error_mm, float(error_mm.mean())


def compute_pa_mpjpe(prediction: np.ndarray, ground_truth: np.ndarray):
    """Compute BODY_25 PA-MPJPE after a proper similarity Procrustes alignment.

    Prediction is aligned to ground truth with one global rotation, translation,
    and uniform scale. Reflection is disallowed. Inputs and returned errors use
    meters and millimeters respectively.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if prediction.shape != (25, 3) or ground_truth.shape != (25, 3):
        raise ValueError(
            f"Expected matching BODY_25 arrays (25,3), got {prediction.shape} and {ground_truth.shape}"
        )
    if not np.isfinite(prediction).all() or not np.isfinite(ground_truth).all():
        raise ValueError("PA-MPJPE inputs must contain only finite values")

    pred_mean = prediction.mean(axis=0)
    gt_mean = ground_truth.mean(axis=0)
    pred_centered = prediction - pred_mean
    gt_centered = ground_truth - gt_mean
    pred_variance = float(np.sum(pred_centered * pred_centered))
    if pred_variance <= np.finfo(np.float64).eps:
        raise ValueError("Cannot Procrustes-align a prediction with zero spatial variance")

    # For row-vector points, R = U D V^T minimizes ||X R - Y||_F.
    u, singular_values, vt = np.linalg.svd(pred_centered.T @ gt_centered)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    scale = float(np.sum(singular_values * np.diag(correction)) / pred_variance)
    aligned = scale * (pred_centered @ rotation) + gt_mean
    error_mm = np.linalg.norm(aligned - ground_truth, axis=1) * 1000.0
    return error_mm, float(error_mm.mean())
