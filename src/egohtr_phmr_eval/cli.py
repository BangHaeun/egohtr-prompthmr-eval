from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import yaml

from .dataset import ensure_sequence_files, select_index
from .gt import BODY25_NAMES, generate_gt
from .inference import run_prompthmr
from .metrics import compute_mpjpe
from .sample import extract_sample
from .transforms import apply_transform, make_transform_chain


def _expand(value):
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def _load_aria_rgb_intrinsics(sequence_npz, rgb_shape):
    """Build the pinhole K from EgoHTR's saved observer RGB calibration scalars."""
    height, width = rgb_shape[:2]
    with np.load(sequence_npz, allow_pickle=False) as data:
        required = ("aria_obs_rgb_focal_length", "aria_obs_rgb_img_width", "aria_obs_rgb_img_height")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"full_sequence.npz lacks RGB calibration fields {missing}")

        def scalar_at(key):
            value = np.asarray(data[key], dtype=np.float64)
            return float(value.reshape(-1)[0])

        focal = scalar_at("aria_obs_rgb_focal_length")
        calibrated_width = int(scalar_at("aria_obs_rgb_img_width"))
        calibrated_height = int(scalar_at("aria_obs_rgb_img_height"))
    if (width, height) != (calibrated_width, calibrated_height):
        raise ValueError(
            f"RGB frame is {width}x{height}, but EgoHTR calibration says "
            f"{calibrated_width}x{calibrated_height}"
        )
    if focal <= 0:
        raise ValueError(f"Invalid Aria RGB focal length {focal}")
    return np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def main():
    parser = argparse.ArgumentParser(description="Run one EgoHTR–PromptHMR frame evaluation")
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--index", type=int)
    parser.add_argument("--timestamp", type=float)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    config_path = _expand(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    dataset_root = _expand(cfg["dataset_root"])
    sequence = str(cfg["sequence"])
    archive, sequence_config = ensure_sequence_files(dataset_root, sequence, cfg.get("huggingface_repo_id"))
    sequence_cfg = yaml.safe_load(sequence_config.read_text(encoding="utf-8")) or {}
    smplx_gender = (sequence_cfg.get("smplx_config") or {}).get("gender")
    if not smplx_gender:
        raise KeyError(f"{sequence_config} lacks smplx_config.gender; set the correct EgoHTR model gender")
    index, timestamp = select_index(archive,
        args.index if args.index is not None else cfg.get("index"),
        args.timestamp if args.timestamp is not None else cfg.get("timestamp_seconds"))

    output_dir = _expand(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sequence: {sequence}\nFrame: {index}\nTimestamp: {timestamp:.10f} s")
    _, rgb_image = extract_sample(archive, index, timestamp, output_dir)

    repo = _expand(cfg["prompthmr_repo"])
    smplx_folder = None
    if cfg.get("smplx_model_folder"):
        smplx_folder = Path(os.path.expandvars(os.path.expanduser(str(cfg["smplx_model_folder"]))))
        if not smplx_folder.is_absolute():
            smplx_folder = (repo / smplx_folder).resolve()
    gt_joints, gt_vertices, gt_faces, gt_joint_indices, joint_reconstruction_max_abs = generate_gt(
        output_dir / "eval_sample.npz", smplx_folder, repo, smplx_gender, cfg.get("device", "cuda"))
    # Joints are selected from EgoHTR's stored array; only the mesh is regenerated.
    np.savez_compressed(output_dir / "gt_body25_mesh.npz", joints=gt_joints,
                        joint_names=np.asarray(BODY25_NAMES), joint_indices=gt_joint_indices,
                        joint_source=np.asarray("smplx_exomc_joints; coordinates assumed SLAM Origin"),
                        smplx_gender=np.asarray(smplx_gender),
                        vertices=gt_vertices, faces=gt_faces)

    cam_int = _load_aria_rgb_intrinsics(archive, rgb_image.shape)
    pred, pred_faces, _ = run_prompthmr(
        output_dir / "rgb.png", repo, cfg["model_folder"], cfg["yolo_weights"], smplx_folder,
        output_dir, cfg.get("device", "cuda"), float(cfg.get("detect_conf", .3)),
        int(cfg.get("image_size", 896)), cam_int=cam_int)
    pred_joints_camera = np.asarray(pred["body_joints"]).squeeze()
    pred_vertices_camera = np.asarray(pred["vertices"]).squeeze()

    fit_dir = output_dir / "gender_fit"
    from .gender_fit import fit_prediction_gender
    fit_report = fit_prediction_gender(
        output_dir / "prediction.npz", fit_dir, repo, smplx_folder,
        target_gender=smplx_gender,
        device=cfg.get("device", "cuda"),
        max_iter=int(cfg.get("fit_max_iter", 120)),
        pose_prior_weight=float(cfg.get("fit_pose_prior_weight", 1e-3)),
        shape_prior_weight=float(cfg.get("fit_shape_prior_weight", 1e-4)),
    )
    fitted_path = fit_dir / f"fitted_{smplx_gender}_prediction.npz"
    with np.load(fitted_path, allow_pickle=False) as z:
        fitted_vertices_camera = np.asarray(z["vertices"])
        fitted_faces = np.asarray(z["faces"], dtype=np.int32)
    if not np.array_equal(pred_faces, fitted_faces) or not np.array_equal(pred_faces, gt_faces):
        raise ValueError("PromptHMR, fitted SMPL-X, and GT mesh topology do not match")

    transforms = make_transform_chain(archive, index)
    T = transforms["T_origin_camera"]
    prediction_joints_origin = apply_transform(T, pred_joints_camera)
    fitted_vertices_origin = apply_transform(T, fitted_vertices_camera)
    # GT joints remain the direct BODY_25 selection from EgoHTR; only the
    # original, unfitted PromptHMR joints are used for the metric.
    errors, mean_error = compute_mpjpe(prediction_joints_origin, gt_joints)

    names = BODY25_NAMES
    with (output_dir / "mpjpe_body25.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "joint_name", "error_mm"])
        writer.writerows((i, names[i], f"{errors[i]:.3f}") for i in range(25))
    np.savez_compressed(output_dir / "evaluation.npz", index=np.asarray(index), timestamp=np.asarray(timestamp),
        target_coordinate_frame=np.asarray("SLAM Origin"), T_camera_to_origin=T,
        camera_intrinsics=cam_int, prediction_body_joints_camera=pred_joints_camera,
        prediction_body_joints_origin=prediction_joints_origin, gt_body_joints_origin=gt_joints,
        joint_names=np.asarray(names), gt_body25_indices=gt_joint_indices,
        gt_smplx_gender=np.asarray(smplx_gender),
        smplx_joint_reconstruction_max_abs_m=np.asarray(
            np.nan if joint_reconstruction_max_abs is None else joint_reconstruction_max_abs),
        joint_error_mm=errors, mean_mpjpe_mm=np.asarray(mean_error),
        prediction_vertices_camera=pred_vertices_camera,
        prediction_vertices_fitted_origin=fitted_vertices_origin, gt_vertices_origin=gt_vertices,
        faces=gt_faces, **transforms)
    manifest = {
        "sequence": sequence, "index": index, "timestamp_seconds": timestamp,
        "pipeline_steps": [
            "extract 2nd Aria RGB and EgoHTR GT sample",
            "run PromptHMR with EgoHTR Aria RGB intrinsics",
            "fit neutral PromptHMR mesh to the sequence-gender SMPL-X model",
            "transform camera-frame prediction and camera pose into SLAM Origin",
            "compute absolute BODY_25 MPJPE from original prediction and GT joints; open Viser",
        ],
        "metric": "absolute 3D BODY_25 MPJPE in SLAM Origin; original unfitted PromptHMR joints; no root subtraction, Procrustes, or scale alignment",
        "evaluation_coordinate_frame": "SLAM Origin",
        "ground_truth_coordinate_assumption": "GT coordinates are treated as already expressed in SLAM Origin",
        "ground_truth_joint_source": "direct BODY_25 selection from EgoHTR smplx_exomc_joints",
        "ground_truth_body25_indices": gt_joint_indices.tolist(),
        "ground_truth_smplx_gender": smplx_gender,
        "ground_truth_mesh": "regenerated from EgoHTR SMPL-X parameters using configured gender; PromptHMR body-joint corrections are not used for GT joints or vertices",
        "smplx_joint_reconstruction_max_abs_m": joint_reconstruction_max_abs,
        "prediction_frame": "PromptHMR output RGB camera frame",
        "aria_rgb_intrinsics_input": cam_int.tolist(),
        "prediction_gender": "neutral",
        "fitted_mesh_gender": smplx_gender,
        "fitting_report": str((fit_dir / "fit_report.json").relative_to(output_dir)),
        "mpjpe_prediction_joint_source": "original PromptHMR prediction.npz body_joints before SMPL-X gender fitting",
        "viewer_prediction_mesh_source": str(fitted_path.relative_to(output_dir)),
        "assumed_camera_extrinsic_key": "aria_obs_T_device_camera",
        "camera_extrinsic_semantics": cfg.get("camera_extrinsic_semantics", "unverified"),
        "transform_chain": "T_origin_camera = aria_obs_T_origin_device[index] @ aria_obs_T_device_camera",
        "mean_mpjpe_mm": mean_error,
        "camera_to_origin_rotation": T[:3, :3].tolist(),
        "transforms": {k: v.tolist() for k, v in transforms.items()},
        "camera_frame_semantics_note": "NPZ key name alone does not establish RGB-vs-SLAM frame semantics; verify calibration provenance.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "mpjpe_body25.txt").write_text(
        f"Sequence = {sequence}\nIndex = {index}\nTimestamp = {timestamp:.10f} s\n"
        "Evaluation = absolute BODY_25 MPJPE in SLAM Origin (no root subtraction or alignment)\n"
        f"GT joints = direct selection from smplx_exomc_joints; SMPL-X gender = {smplx_gender}\n"
        "Prediction joints = original PromptHMR joints before gender fitting\n"
        f"Mean MPJPE = {mean_error:.3f} mm\n"
        "GT coordinate assumption = already in SLAM Origin\n"
        "Transform = aria_obs_T_origin_device[index] @ aria_obs_T_device_camera\n",
        encoding="utf-8")
    print(f"Mean absolute MPJPE: {mean_error:.3f} mm")
    print(f"Camera → SLAM Origin rotation:\n{T[:3, :3]}")
    print(f"Gender-fit vertex RMSE: {fit_report[f'{smplx_gender}_after_fit']['vertices']['rmse_m'] * 1000:.3f} mm")
    print(f"Saved outputs: {output_dir}")
    if cfg.get("viewer", True) and not args.no_viewer:
        from .viewer import show_meshes
        show_meshes(
            fitted_vertices_origin,
            gt_vertices,
            gt_faces,
            port=int(cfg.get("viewer_port", 8080)),
            T_camera_to_origin=T,
            rgb_image=rgb_image,
            camera_intrinsics=cam_int,
            network_input_size=int(cfg.get("image_size", 896)),
            world_frame_name="SLAM Origin",
        )


if __name__ == "__main__":
    main()
