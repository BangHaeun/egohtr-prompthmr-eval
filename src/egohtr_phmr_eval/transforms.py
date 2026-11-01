from __future__ import annotations

import numpy as np


def make_transform_chain(sequence_npz, index: int):
    """Return the sole evaluation chain, from the RGB camera to SLAM Origin."""
    with np.load(sequence_npz, allow_pickle=False) as data:
        required = ("aria_obs_T_origin_device", "aria_obs_T_device_camera")
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"full_sequence.npz lacks {missing}")
        origin_device = np.asarray(data[required[0]], dtype=np.float64)
        device_camera = np.asarray(data[required[1]], dtype=np.float64)
        T_origin_device = origin_device[index] if origin_device.ndim == 3 else origin_device
        T_device_camera = device_camera[index] if device_camera.ndim == 3 else device_camera
    # T_target_source convention: device->origin composed with camera->device.
    T_origin_camera = T_origin_device @ T_device_camera
    return {
        "T_origin_device": T_origin_device,
        "T_device_camera": T_device_camera,
        "T_origin_camera": T_origin_camera,
    }


def apply_transform(T_target_source: np.ndarray, points_source: np.ndarray) -> np.ndarray:
    points = np.asarray(points_source)
    return points @ T_target_source[:3, :3].T + T_target_source[:3, 3]
