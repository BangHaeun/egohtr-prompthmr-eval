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
    sample, _ = extract_sample(archive, index, timestamp, output_dir)

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
                        joint_source=np.asarray("smplx_exomc_joints"),
                        smplx_gender=np.asarray(smplx_gender),
                        vertices=gt_vertices, faces=gt_faces)

    pred, pred_faces, _ = run_prompthmr(
        output_dir / "rgb.png", repo, cfg["model_folder"], cfg["yolo_weights"], smplx_folder,
        output_dir, cfg.get("device", "cuda"), float(cfg.get("detect_conf", .3)),
        int(cfg.get("image_size", 896)))
    pred_joints_camera = np.asarray(pred["body_joints"]).squeeze()
    pred_vertices_camera = np.asarray(pred["vertices"]).squeeze()
    transforms = make_transform_chain(sequence_config, archive, index)
    T = transforms["T_exomc_camera"]
    pred_joints_exomc = apply_transform(T, pred_joints_camera)
    pred_vertices_exomc = apply_transform(T, pred_vertices_camera)
    errors, mean_error = compute_mpjpe(pred_joints_exomc, gt_joints)

    names = BODY25_NAMES
    with (output_dir / "mpjpe_body25.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "joint_name", "error_mm"])
        writer.writerows((i, names[i], f"{errors[i]:.3f}") for i in range(25))
    np.savez_compressed(output_dir / "evaluation.npz", index=np.asarray(index), timestamp=np.asarray(timestamp),
        T_camera_to_exomc=T, prediction_body_joints_camera=pred_joints_camera,
        prediction_body_joints_exomc=pred_joints_exomc, gt_body_joints_exomc=gt_joints,
        joint_names=np.asarray(names), gt_body25_indices=gt_joint_indices,
        gt_smplx_gender=np.asarray(smplx_gender),
        smplx_joint_reconstruction_max_abs_m=np.asarray(
            np.nan if joint_reconstruction_max_abs is None else joint_reconstruction_max_abs),
        joint_error_mm=errors, mean_mpjpe_mm=np.asarray(mean_error),
        prediction_vertices_exomc=pred_vertices_exomc, gt_vertices_exomc=gt_vertices,
        faces=gt_faces, **transforms)
    manifest = {
        "sequence": sequence, "index": index, "timestamp_seconds": timestamp,
        "metric": "absolute 3D BODY_25 MPJPE in ExoMC; no root subtraction, Procrustes, or scale alignment",
        "ground_truth_joint_source": "direct BODY_25 selection from EgoHTR smplx_exomc_joints",
        "ground_truth_body25_indices": gt_joint_indices.tolist(),
        "ground_truth_smplx_gender": smplx_gender,
        "ground_truth_mesh": "regenerated from EgoHTR SMPL-X parameters using configured gender; PromptHMR body-joint corrections are not used for GT joints or vertices",
        "smplx_joint_reconstruction_max_abs_m": joint_reconstruction_max_abs,
        "prediction_frame": "PromptHMR output camera frame",
        "assumed_camera_extrinsic_key": "aria_obs_T_device_camera",
        "camera_extrinsic_semantics": cfg.get("camera_extrinsic_semantics", "unverified"),
        "transform_chain": "inverse(T_origin_exoW) @ aria_obs_T_origin_device[index] @ aria_obs_T_device_camera",
        "mean_mpjpe_mm": mean_error,
        "camera_to_exomc_rotation": T[:3, :3].tolist(),
        "transforms": {k: v.tolist() for k, v in transforms.items()},
        "camera_frame_semantics_note": "NPZ key name alone does not establish RGB-vs-SLAM frame semantics; verify calibration provenance.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "mpjpe_body25.txt").write_text(
        f"Sequence = {sequence}\nIndex = {index}\nTimestamp = {timestamp:.10f} s\n"
        f"Evaluation = absolute BODY_25 MPJPE in ExoMC (no root subtraction or alignment)\n"
        f"GT joints = direct selection from smplx_exomc_joints; SMPL-X gender = {smplx_gender}\n"
        f"Mean MPJPE = {mean_error:.3f} mm\n"
        "Transform = inverse(T_origin_exoW) @ aria_obs_T_origin_device[index] @ aria_obs_T_device_camera\n",
        encoding="utf-8")
    print(f"Mean absolute MPJPE: {mean_error:.3f} mm")
    print(f"Camera → ExoMC rotation:\n{T[:3, :3]}")
    print(f"Saved outputs: {output_dir}")
    if cfg.get("viewer", True) and not args.no_viewer:
        from .viewer import show_meshes
        show_meshes(pred_vertices_exomc, gt_vertices, gt_faces, int(cfg.get("viewer_port", 8080)))


if __name__ == "__main__":
    main()
