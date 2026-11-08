from __future__ import annotations

import numpy as np


def _rigid_align(source: np.ndarray, target: np.ndarray):
    """Return row-vector rigid transform source @ R + t -> target (no scale)."""
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("Rigid alignment needs matching point arrays with at least 3 points")
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ d @ vt
    translation = b - a @ rotation
    return rotation, translation


def compute_window_metrics(prediction, ground_truth, source_indices, window_frames=100,
                          initial_alignment_frames=2):
    """Compute WA/W-MPJPE on non-overlapping contiguous evaluation windows.

    Arrays are (F,25,3), in meters; source_indices are original NPZ frame ids.
    Windows never cross missing-frame gaps. Rigid alignment has no scale.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    ids = np.asarray(source_indices, dtype=np.int64)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[1:] != (25, 3):
        raise ValueError(f"Expected matching (F,25,3) arrays, got {pred.shape}, {gt.shape}")
    if len(ids) != len(pred) or window_frames <= 0 or initial_alignment_frames < 1:
        raise ValueError("Invalid source indices or window configuration")
    if initial_alignment_frames > window_frames:
        raise ValueError("initial_alignment_frames cannot exceed window_frames")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("Window metric inputs must be finite")

    # Consecutive-source runs are evaluated independently; skipped frames are never stitched.
    cuts = np.flatnonzero(np.diff(ids) != 1) + 1
    segments = np.split(np.arange(len(ids)), cuts)
    windows = []
    for segment_id, segment in enumerate(segments):
        for start in range(0, len(segment), window_frames):
            ix = segment[start:start + window_frames]
            if len(ix) < initial_alignment_frames:
                continue
            p, g = pred[ix], gt[ix]
            R_all, t_all = _rigid_align(p.reshape(-1, 3), g.reshape(-1, 3))
            wa = np.linalg.norm(p.reshape(-1, 3) @ R_all + t_all - g.reshape(-1, 3), axis=1).mean() * 1000
            n_init = min(initial_alignment_frames, len(ix))
            R_first, t_first = _rigid_align(p[:n_init].reshape(-1, 3), g[:n_init].reshape(-1, 3))
            w = np.linalg.norm(p.reshape(-1, 3) @ R_first + t_first - g.reshape(-1, 3), axis=1).mean() * 1000
            windows.append({
                "segment_id": int(segment_id), "start_frame": int(ids[ix[0]]),
                "end_frame": int(ids[ix[-1]]), "num_frames": int(len(ix)),
                "wa_mpjpe_mm": float(wa), "w_mpjpe_mm": float(w),
                "wa_rotation": R_all.tolist(), "wa_translation": t_all.tolist(),
                "w_rotation": R_first.tolist(), "w_translation": t_first.tolist(),
            })
    return windows


def aggregate_window_metric(windows, key):
    if not windows:
        return float("nan")
    weights = np.asarray([w["num_frames"] * 25 for w in windows], dtype=np.float64)
    values = np.asarray([w[key] for w in windows], dtype=np.float64)
    return float(np.average(values, weights=weights))
