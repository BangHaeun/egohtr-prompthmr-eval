from __future__ import annotations

import numpy as np

from .dataset import read_npz_rgb_frame


def extract_sample(sequence_npz, index, timestamp, output_dir):
    from pathlib import Path
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(sequence_npz, allow_pickle=False) as data:
        keys = ("smplx_exomc_joints", "smplx_exomc_body_pose", "smplx_exomc_global_orient",
                "smplx_exomc_betas", "smplx_exomc_transl")
        missing = [k for k in keys if k not in data]
        if missing:
            raise KeyError(f"full_sequence.npz lacks GT fields: {missing}")
        sample = {k: np.asarray(data[k][index]) for k in keys}
        sample.update(index=np.asarray(index), timestamp=np.asarray(timestamp, dtype=np.float64))
        for name in ("aria_timestamp", "exomc_timestamp"):
            if name in data:
                sample[name] = np.asarray(data[name][index], dtype=np.float64)
        if "smplx_exomc_faces" in data:
            # Small, sequence-static topology needed to reproduce the GT mesh
            # after the large full_sequence.npz is removed.
            sample["smplx_exomc_faces"] = np.asarray(data["smplx_exomc_faces"], dtype=np.int32)
    np.savez_compressed(output_dir / "eval_sample.npz", **sample)
    rgb = read_npz_rgb_frame(sequence_npz, "aria_obs_rgb_imgs", index)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 observer RGB frame, got {rgb.shape}")
    # NPZ stores RGB; OpenCV writer expects BGR.
    if not cv2.imwrite(str(output_dir / "rgb.png"), rgb[:, :, ::-1]):
        raise IOError("Failed to write extracted RGB image")
    return sample, rgb
