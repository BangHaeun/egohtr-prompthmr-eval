"""Sequence-level world-coordinate joint metrics.

Inputs must be temporally synchronized, identity-matched, and expressed in
compatible world coordinate systems. Arrays have shape (T, J, 3), in meters.
This module does not perform camera/SLAM alignment or time synchronization.
"""
from __future__ import annotations

import numpy as np


def _validate(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("pred and gt must both have shape (T, J, 3)")
    if pred.shape[0] < 2 or pred.shape[1] < 3:
        raise ValueError("At least 2 frames and 3 joints are required")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("All joints must be finite; filter invalid frames first")
    return pred, gt


def _rigid_fit(source, target):
    """Fit one proper rotation and translation; no reflection or scaling."""
    x = source.reshape(-1, 3)
    y = target.reshape(-1, 3)
    xc, yc = x.mean(axis=0), y.mean(axis=0)
    u, _, vt = np.linalg.svd((x - xc).T @ (y - yc))
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(u @ vt)
    r = u @ d @ vt
    t = yc - xc @ r
    return r, t


def evaluate_world_mpjpe(pred, gt, window_size=100, initial_frames=2,
                         include_partial=False):
    """Return WA-MPJPE and W-MPJPE (millimeters) over nonoverlapping windows.

    WA: one rigid alignment fitted using the entire window.
    W: one rigid alignment fitted using only the first initial_frames.
    Both alignments are applied to every frame in their respective window.
    No per-frame or per-joint alignment is performed.
    """
    pred, gt = _validate(pred, gt)
    if window_size < 2 or not 1 <= initial_frames <= window_size:
        raise ValueError("Require window_size >= 2 and 1 <= initial_frames <= window_size")
    records = []
    for start in range(0, len(pred), window_size):
        end = min(start + window_size, len(pred))
        if end - start < window_size and not include_partial:
            break
        if end - start < initial_frames:
            break
        p, g = pred[start:end], gt[start:end]
        wa_r, wa_t = _rigid_fit(p, g)
        w_r, w_t = _rigid_fit(p[:initial_frames], g[:initial_frames])
        wa = np.linalg.norm(p @ wa_r + wa_t - g, axis=-1) * 1000
        w = np.linalg.norm(p @ w_r + w_t - g, axis=-1) * 1000
        records.append({
            "start": start, "end": end, "frames": end - start,
            "wa_mpjpe_mm": float(wa.mean()),
            "w_mpjpe_mm": float(w.mean()),
        })
    if not records:
        raise ValueError("No evaluable windows; check length and partial-window policy")
    total = sum(r["frames"] for r in records)
    return {
        "wa_mpjpe_mm": sum(r["wa_mpjpe_mm"] * r["frames"] for r in records) / total,
        "w_mpjpe_mm": sum(r["w_mpjpe_mm"] * r["frames"] for r in records) / total,
        "windows": records,
        "protocol": {
            "window_size": window_size, "initial_frames": initial_frames,
            "include_partial": include_partial, "alignment": "rigid_SE3_no_scale",
            "unit": "mm",
        },
    }
