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
    parser = argparse.ArgumentParser(description="Run one EgoHTRâ€“PromptHMR frame evaluation")
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
    print(f"Camera â†’ SLAM Origin rotation:\n{T[:3, :3]}")
    print(f"Gender-fit vertex RMSE: {fit_report[f'{smplx_gender}_after_fit']['vertices']['rmse_m'] * 1000:.3f} mm")
    print(f"Saved outputs: {output_dir}")
    if cfg.get("viewer", True) and not args.no_viewer:
        from .viewer import show_meshes
        show_meshes(
    ×žº¶‰žËkºwµç]Ù˜XÙ\ÊN‚ˆ˜Z\ÙH˜[YQ\œ›ÜŠ•\™Ù]Y\ÚÜÛÙÞHÙ\È›ÝX]ÚH›Û\TˆÓTV[Ù[ŠB‚ˆ\HHÜ˜Ú™›Ø]Ì‚ˆ\™Ù]Ý™\XÙ\ÈHÜ˜Ú˜\×Ý[œÛÜŠ\™Ù]Ý™\XÙ\×Ûœ]šXÙOY]šXÙK\OY\JK[œÜ]YY^™J
Bˆ\™Ù]Ú›Ú[ÈHÜ˜Ú˜\×Ý[œÛÜŠ\™Ù]Ú›Ú[×Ûœ]šXÙOY]šXÙK\OY\JK[œÜ]YY^™J
BˆÛØ˜[ÛÜšY[HÜ˜Ú˜\×Ý[œÛÜŠÜšYÚ[˜[Ü›ÝX]ÛœÎŒWK]šXÙOY]šXÙK\OY\JK[œÜ]YY^™J
Bˆ[š]X[Ø›ÙWÜ›ÝX]HÜ˜Ú˜\×Ý[œÛÜŠÜšYÚ[˜[Ü›ÝX]ÛœÌN—K]šXÙOY]šXÙK\OY\JK[œÜ]YY^™J
Bˆ[š]X[Ø™]\ÈHÜ˜Ú˜\×Ý[œÛÜŠÜšYÚ[˜[Ø™]\×Ûœ]šXÙOY]šXÙK\OY\JK[œÜ]YY^™J
Bˆš^YÝ˜[œÛHÜ˜Ú˜\×Ý[œÛÜŠÜšYÚ[˜[Ý˜[œÛÛœ]šXÙOY]šXÙK\OY\JK[œÜ]YY^™J
Bˆ[™ÛYX[ˆH[Ù[š[™×ÛYX[‹Ê]šXÙOY]šXÙK\OY\JBˆYÚ[™ÜÜÙHH[™ÛYX[–Î‹ŒMWBˆšYÚÚ[™ÜÜÙHH[™ÛYX[–Î‹MN—B‚ˆYˆ›ÜØ\™
™]\Ë›ÙWÜ›ÝX]
N‚ˆ™]\›ˆ[Ù[
ˆ™]\ÏX™]\ËˆÛØ˜[ÛÜšY[YÛØ˜[ÛÜšY[ˆ›ÙWÜÜÙOX›ÙWÜ›ÝX]ˆ˜[œÛYš^YÝ˜[œÛˆYÚ[™ÜÜÙO[YÚ[™ÜÜÙKˆšYÚÚ[™ÜÜÙO\šYÚÚ[™ÜÜÙKˆ
B‚ˆš[
–Ùš]KÍH™XÛÛœÝXÝ[™ÈHØ]™Y›Û\TˆY\ÚÚ]™]]˜[ÓTV‹‹ˆ‹›\ÚUYJBˆ™]]˜[HÓT
ÝŠÛ\Û[Ù[Ù›Û\ŠKÙ[™\H›™]]˜[ŠKÊ]šXÙJK™]˜[

Bˆ™]]˜[Ú[™ÛYX[ˆH™]]˜[š[™×ÛYX[‹Ê]šXÙOY]šXÙK\OY\JBˆÚ]Ü˜Ú››×ÙÜ˜Y

N‚ˆ™]]˜[ÛÝ]H™]]˜[
ˆ™]\ÏZ[š]X[Ø™]\ËˆÛØ˜[ÛÜšY[YÛØ˜[ÛÜšY[ˆ›ÙWÜÜÙOZ[š]X[Ø›ÙWÜ›ÝX]ˆ˜[œÛYš^YÝ˜[œÛˆYÚ[™ÜÜÙO[™]]˜[Ú[™ÛYX[–Î‹ŒMWKˆšYÚÚ[™ÜÜÙO[™]]˜[Ú[™ÛYX[–Î‹MN—Kˆ
Bˆ™]]˜[Ý™\XÙ\×ÛœH™]]˜[ÛÝ]™\XÙ\ÖÌK™]XÚ

K˜ÜJ
K›[\J
Bˆ™]]˜[Ú›Ú[×ÛœH™]]˜[ÛÝ]˜›ÙWÚ›Ú[ÖÌK™]XÚ

K˜ÜJ
K›[\J
Bˆ™]]˜[Ý™\^Ù\œ›ÜˆHÙ\œ›Ü—ÜÝ[[X\žJ™]]˜[Ý™\XÙ\×Ûœ\™Ù]Ý™\XÙ\×Ûœ
Bˆ™]]˜[Ú›Ú[Ù\œ›ÜˆHÙ\œ›Ü—ÜÝ[[X\žJ™]]˜[Ú›Ú[×Ûœ\™Ù]Ú›Ú[×Ûœ
Bˆš[
ˆˆ™]]˜[™\^“TÑO^Û™]]˜[Ý™\^Ù\œ›Ü–ÉÜ›\ÙWÛI×N‹ŽŸHH‹›\ÚUYJBˆYˆ™]]˜[Ý™\^Ù\œ›Ü–Èœ›\ÙWÛH—HˆX^Ú[š]X[Ü™XÛÛœÝXÝ[Û—Ü›\ÙWÛN‚ˆ˜Z\ÙH[[YQ\œ›ÜŠˆ“™]]˜[ÓTV\˜[Y]\œÈÈ›Ý™\›ÙXÙHHØ]™Y›Û\TˆY\Ú‚ˆˆŠ“TÑHÛ™]]˜[Ý™\^Ù\œ›Ü–ÉÜ›\ÙWÛI×N‹ŽßHH^ÙYYÈ‚ˆˆžÛX^Ú[š]X[Ü™XÛÛœÝXÝ[Û—Ü›\ÙWÛN™ßHJNÈ™Y\Ú[™ÈÈš]HZ\ÛX]ÚY\™Ù]ˆ‚ˆ
Bˆ[™]]˜[™]]˜[ÛÝ]‚ˆš[
–Ùš]‹ÍHYX\Ý\š[™È[š]X[X[HÓTVš]‹‹ˆ‹›\ÚUYJBˆÚ]Ü˜Ú››×ÙÜ˜Y

N‚ˆX[WÚ[š]X[H›ÜØ\™
[š]X[Ø™]\Ë[š]X[Ø›ÙWÜ›ÝX]
BˆX[WÚ[š]X[Ý™\XÙ\ÈHX[WÚ[š]X[™\XÙ\Ë™]XÚ

BˆX[WÚ[š]X[Ú›Ú[ÈHX[WÚ[š]X[˜›ÙWÚ›Ú[Ë™]XÚ

Bˆ[š]X[Ý™\^Ù\œ›ÜˆHÙ\œ›Ü—ÜÝ[[X\žJX[WÚ[š]X[Ý™\XÙ\ÖÌK˜ÜJ
K›[\J
K\™Ù]Ý™\XÙ\×Ûœ
Bˆ[š]X[Ú›Ú[Ù\œ›ÜˆHÙ\œ›Ü—ÜÝ[[X\žJX[WÚ[š]X[Ú›Ú[ÖÌK˜ÜJ
K›[\J
K\™Ù]Ú›Ú[×Ûœ
B‚ˆš[
–Ùš]ËÍHš][™ÈX[H›ÙHÜÙH[™™]\ÎÈÛØ˜[ÜšY[][Ûˆ[™˜[œÛ][Ûˆ\™Hš^Y‹‹ˆ‹›\ÚUYJBˆÈÛX[[™Ù[\ÜXÙHÜÙH\]\È™\Ù\™H˜[Y›Ý][ÛœÈ]]™\žHÝ\‚ˆÜÙWÙ[HHÜ˜Ú››‹”\˜[Y]\ŠÜ˜Úž™\›ÜÊ
KŒKÊK]šXÙOY]šXÙK\OY\JJBˆš]YØ™]\ÈHÜ˜Ú››‹”\˜[Y]\Š[š]X[Ø™]\Ë™]XÚ

K˜ÛÛ™J
JBˆÜ[Z^™\ˆHÜ˜Ú›Ü[K“‘‘ÔÊˆÜÜÙWÙ[Kš]YØ™]\×KˆLŽˆX^Ú]\Z[
X^Ú]\ŠKˆ\ÝÜžWÜÚ^™OLÌˆÛ\˜[˜ÙWÙÜ˜YLYKNKˆÛ\˜[˜ÙWØÚ[™ÙOLYKLL‹ˆ[™WÜÙX\˜ÚÙ›HœÝ›Û™×ÝÛÛ™H‹ˆ
BˆÛÜÝ\™WØØ[ÈH‚ˆYˆÛÜÝ\™J
N‚ˆ›Û›ØØ[ÛÜÝ\™WØØ[ÂˆÜ[Z^™\‹ž™\›×ÙÜ˜Y
Ù]Ý×Û›Û™OUYJBˆ[WÜ›ÝX]H^\×Ø[™ÛWÝ×ÛX]š^
ÜÙWÙ[JBˆ›ÙWÜ›ÝX]HÜ˜Ú›X]][
[WÜ›ÝX][š]X[Ø›ÙWÜ›ÝX]
BˆÝ]H›ÜØ\™
š]YØ™]\Ë›ÙWÜ›ÝX]
Bˆ™\^ÛÜÜÈHÜ˜Ú›YX[Š
Ý]™\XÙ\ÈH\™Ù]Ý™\XÙ\ÊH
ŠˆŠBˆÜÙWÜš[ÜˆHÜ˜Ú›YX[ŠÜÙWÙ[H
ŠˆŠBˆÚ\WÜš[ÜˆHÜ˜Ú›YX[Š
š]YØ™]\ÈH[š]X[Ø™]\ÊH
ŠˆŠBˆÜÜÈH™\^ÛÜÜÈ
ÈÜÙWÜš[Ü—ÝÙZYÚ
ˆÜÙWÜš[Üˆ
ÈÚ\WÜš[Ü—ÝÙZYÚ
ˆÚ\WÜš[Ü‚ˆÜÜË˜˜XÚÝØ\™

BˆÛÜÝ\™WØØ[È
ÏHBˆYˆÛÜÝ\™WØØ[ÈOHHÜˆÛÜÝ\™WØØ[È	HHOH‚ˆš[
ˆˆˆÛÜÝ\™HØÛÜÝ\™WØØ[ßNˆ™\^Û\ÙO^Ý™\^ÛÜÜË™]XÚ

Kš][J
N‹ŽßHWŒ‹‚ˆˆÝ[^ÛÜÜË™]XÚ

Kš][J
N‹ŽßH‹›\ÚUYBˆ
Bˆ™]\›ˆÜÜÂ‚ˆÜ[Z^™\‹œÝ\
ÛÜÝ\™JB‚ˆš[
–Ùš]ÍHØ]š[™È\˜[Y]\œÈ[™™\šYžZ[™È™\^Ú›Ú[™XÛÛœÝXÝ[Û‹‹‹ˆ‹›\ÚUYJBˆÚ]Ü˜Ú››×ÙÜ˜Y

N‚ˆš]YØ›ÙWÜ›ÝX]HÜ˜Ú›X]][
^\×Ø[™ÛWÝ×ÛX]š^
ÜÙWÙ[JK[š]X[Ø›ÙWÜ›ÝX]
Bˆš]YÛÝ]H›ÜØ\™
š]YØ™]\Ëš]YØ›ÙWÜ›ÝX]
Bˆš]YÝ™\XÙ\×ÛœHš]YÛÝ]™\XÙ\ÖÌK™]XÚ

K˜ÜJ
K›[\J
K˜\Ý\Jœ™›Ø]ÌŠBˆš]YÚ›Ú[×ÛœHš]YÛÝ]˜›ÙWÚ›Ú[ÖÌK™]XÚ

K˜ÜJ
K›[\J
K˜\Ý\Jœ™›Ø]ÌŠBˆš]YØ›ÙWÜ›ÝX]ÛœHš]YØ›ÙWÜ›ÝX]ÌK™]XÚ

K˜ÜJ
K›[\J
K˜\Ý\Jœ™›Ø]ÌŠBˆš]YÙÛØ˜[Ü›ÝX]ÛœHÛØ˜[ÛÜšY[ÌK™]XÚ

K˜ÜJ
K›[\J
K˜\Ý\Jœ™›Ø]ÌŠBˆš]YØ™]\×ÛœHš]YØ™]\ÖÌK™]XÚ

K˜ÜJ
K›[\J
K˜\Ý\Jœ™›Ø]ÌŠBˆš^YÝ˜[œÛÛœHš^YÝ˜[œÛÌK™]XÚ

K˜ÜJ
K›[\J
K˜\Ý\Jœ™›Ø]ÌŠBˆš]YØ›ÙWØXWÛœHX]š^Ý×Ø^\×Ø[™ÛJš]YØ›ÙWÜ›ÝX]ÌJK™]XÚ

K˜ÜJ
K›[\J
K˜\Ý\Jœ™›Ø]ÌŠBˆš]YÙÛØ˜[ØXWÛœHX]š^Ý×Ø^\×Ø[™ÛJÛØ˜[ÛÜšY[ÌJK™]XÚ

K˜ÜJ
K›[\J
K˜\Ý\Jœ™›Ø]ÌŠBˆš]YÝ™\^Ù\œ›ÜˆHÙ\œ›Ü—ÜÝ[[X\žJš]YÝ™\XÙ\×Ûœ\™Ù]Ý™\XÙ\×Ûœ
Bˆš]YÚ›Ú[Ù\œ›ÜˆHÙ\œ›Ü—ÜÝ[[X\žJš]YÚ›Ú[×Ûœ\™Ù]Ú›Ú[×Ûœ
BˆÜÙWÙ[WÛœHÜÙWÙ[K™]XÚ

K˜ÜJ
K›[\J
VÌBˆÜÙWÙ[WÚ›Ú[Ø[™ÛWÙYÈHœ™YÜ™Y\Êœ›[˜[Ë››Ü›JÜÙWÙ[WÛœ^\ÏKLJJB‚ˆš]YÜ^[ØYHXÝ
ˆÙ[™\[œ˜\Ø\œ˜^J\™Ù]ÙÙ[™\ŠKˆ™\XÙ\ÏYš]YÝ™\XÙ\×Ûœˆ›ÙWÚ›Ú[ÏYš]YÚ›Ú[×Ûœˆ˜XÙ\Ï]\™Ù]Ù˜XÙ\ËˆÛØ˜[ÛÜšY[Ü›ÝX]Yš]YÙÛØ˜[Ü›ÝX]ÛœˆÛØ˜[ÛÜšY[Ø^\×Ø[™ÛOYš]YÙÛØ˜[ØXWÛœˆ›ÙWÜÜÙWÜ›ÝX]Yš]YØ›ÙWÜ›ÝX]Ûœˆ›ÙWÜÜÙWØ^\×Ø[™ÛOYš]YØ›ÙWØXWÛœˆ™]\ÏYš]YØ™]\×Ûœˆ˜[œÛYš^YÝ˜[œÛÛœˆYÚ[™ÜÜÙWÜ›ÝX][YÚ[™ÜÜÙVÌK™]XÚ

K˜ÜJ
K›[\J
K˜\Ý\Jœ™›Ø]ÌŠKˆšYÚÚ[™ÜÜÙWÜ›ÝX]\šYÚÚ[™ÜÜÙVÌK™]XÚ

K˜ÜJ
K›[\J
K˜\Ý\Jœ™›Ø]ÌŠKˆÜÙWÙ[WØ^\×Ø[™ÛO\ÜÙWÙ[WÛœˆÛÝ\˜ÙWÜ™YXÝ[Û[œ˜\Ø\œ˜^JÝŠ™YXÝ[Û—Ü]
JKˆ
BˆYˆØ[WÚ[Ûœ\È›Ý›Û™N‚ˆš]YÜ^[ØYÈ˜Ø[WÚ[—HHØ[WÚ[ÛœˆœœØ]™^—ØÛÛ\™\ÜÙY
Ý]]Ù\ˆÈˆ™š]YÞÝ\™Ù]ÙÙ[™\ŸWÜ™YXÝ[Û‹›œˆ‹
Š™š]YÜ^[ØY
Bˆ™\ÜHÂˆœÛÝ\˜ÙWÜ™YXÝ[ÛˆŽˆÝŠ™YXÝ[Û—Ü]
Kˆ›Ý]]Ü™YXÝ[ÛˆŽˆÝŠÝ]]Ù\ˆÈˆ™š]YÞÝ\™Ù]ÙÙ[™\ŸWÜ™YXÝ[Û‹›œˆŠKˆ\™Ù]ÙÙ[™\ˆŽˆ\™Ù]ÙÙ[™\‹ˆ˜˜XÚÙ[™Žˆ”›Û\T‰ÜÈ[œÝ[YY™™\™[XX›HÓTV^Y\ŽÈÛÜœ™\ÜÛ™[™Ë]™\^P‘‘ÔÈ‹ˆ™ÛØ˜[ÛÜšY[Ùš^YŽˆYKˆ˜[œÛ][Û—Ùš^YŽˆYKˆ˜[œÛÜ™\Ù\™YÙ^XÝHŽˆ›ÛÛ
œ˜\œ˜^WÙ\]X[
š^YÝ˜[œÛÛœÜšYÚ[˜[Ý˜[œÛÛœ
JKˆ™ÛØ˜[ÛÜšY[Ü™\Ù\™YÙ^XÝHŽˆ›ÛÛ
œ˜\œ˜^WÙ\]X[
š]YÙÛØ˜[Ü›ÝX]ÛœÜšYÚ[˜[Ü›ÝX]ÛœÌJJKˆ›ÜšYÚ[˜[ÙÛØ˜[ÛÜšY[Ü›ÝX]ŽˆÜšYÚ[˜[Ü›ÝX]ÛœÌKÛ\Ý

Kˆ˜[œÛ][Û—ÛHŽˆš^YÝ˜[œÛÛœÛ\Ý

KˆœÜÙWÜš[Ü—ÝÙZYÚŽˆ›Ø]
ÜÙWÜš[Ü—ÝÙZYÚ
KˆœÚ\WÜš[Ü—ÝÙZYÚŽˆ›Ø]
Ú\WÜš[Ü—ÝÙZYÚ
Kˆ›™™Ü×ÛX^Ú]\ˆŽˆ[
X^Ú]\ŠKˆ›™™Ü×ØÛÜÝ\™WØØ[ÈŽˆ[
ÛÜÝ\™WØØ[ÊKˆ›™]]˜[Ü™XÛÛœÝXÝ[ÛˆŽˆÂˆ™\XÙ\ÈŽˆ™]]˜[Ý™\^Ù\œ›Ü‹ˆš›Ú[ÈŽˆ™]]˜[Ú›Ú[Ù\œ›Ü‹ˆKˆˆžÝ\™Ù]ÙÙ[™\ŸWØ™Y›Ü™WÙš]ŽˆÂˆ™\XÙ\ÈŽˆ[š]X[Ý™\^Ù\œ›Ü‹ˆš›Ú[ÈŽˆ[š]X[Ú›Ú[Ù\œ›Ü‹ˆKˆˆžÝ\™Ù]ÙÙ[™\ŸWØY\—Ùš]ŽˆÂˆ™\XÙ\ÈŽˆš]YÝ™\^Ù\œ›Ü‹ˆš›Ú[ÈŽˆš]YÚ›Ú[Ù\œ›Ü‹ˆKˆ™š]YØ™]\ÈŽˆš]YØ™]\×ÛœÛ\Ý

Kˆ›ÜšYÚ[˜[Ø™]\ÈŽˆÜšYÚ[˜[Ø™]\×ÛœÛ\Ý

Kˆ˜™]WÙ[WÛˆŽˆ›Ø]
œ›[˜[Ë››Ü›Jš]YØ™]\×ÛœHÜšYÚ[˜[Ø™]\×Ûœ
JKˆœÜÙWÙ[WØ^\×Ø[™ÛWÜ›\×Ü˜YŽˆ›Ø]
œœÜ\
œ›YX[ŠÜÙWÙ[WÛœ
ŠˆŠJJKˆ˜›ÙWÜÜÙWÙ[WÚ›Ú[Ø[™ÛWÛYX[—ÙYÈŽˆ›Ø]
ÜÙWÙ[WÚ›Ú[Ø[™ÛWÙYË›YX[Š
JKˆ˜›ÙWÜÜÙWÙ[WÚ›Ú[Ø[™ÛWÛX^ÙYÈŽˆ›Ø]
ÜÙWÙ[WÚ›Ú[Ø[™ÛWÙYË›X^

JKˆ˜›ÙWÜÜÙWÙ[WÚ›Ú[Ø[™ÛWÙYÈŽˆÜÙWÙ[WÚ›Ú[Ø[™ÛWÙYËÛ\Ý

Kˆ™\^Ù\œ›Ü—ÛY]šXÈŽˆœ\‹]™\^]XÛYX[ˆ\Ý[˜ÙH™]ÙY[ˆš]YY\Ú[™ÜšYÚ[˜[›Û\Tˆ\™Ù]Y\ÚY]\œÈ‹ˆš›Ú[Ù\œ›Ü—ÛY]šXÈŽˆœ\‹Z›Ú[]XÛYX[ˆ\Ý[˜ÙH™]ÙY[ˆš]Y“ÑWÌH[™ÜšYÚ[˜[›Û\Tˆ“ÑWÌKY]\œÈ‹ˆBˆ
Ý]]Ù\ˆÈ™š]Ü™\ÜšœÛÛˆŠKÜš]WÝ^
œÛÛ‹™[\Ê™\Ü[™[LŠK[˜ÛÙ[™ÏH]‹NŠBˆ™]\›ˆ™\Üˆš[˜[N‚ˆÜË˜Ú\ŠÛØÝÙ
B‚‚™YˆXZ[Š
HOˆ›Û™N‚ˆ\œÙ\ˆH\™Ü\œÙK\™Ý[Y[\œÙ\Š\ØÜš\[ÛH‘š]Ø]™Y›Û\TˆÓTVY\ÚÈ[›Ý\ˆ[Ù[Ù[™\ˆŠBˆ\œÙ\‹˜YØ\™Ý[Y[
‹K\™YXÝ[Ûˆ‹™\]Z\™YUYK[H”›Û\Tˆ™YXÝ[Û‹›œˆŠBˆ\œÙ\‹˜YØ\™Ý[Y[
‹K[Ý]]Y\ˆ‹™\]Z\™YUYJBˆ\œÙ\‹˜YØ\™Ý[Y[
‹K\›Û\\‹\™\È‹™\]Z\™YUYJBˆ\œÙ\‹˜YØ\™Ý[Y[
‹K\Û\[[Ù[Y›Û\ˆ‹™\]Z\™YUYJBˆ\œÙ\‹˜YØ\™Ý[Y[
‹KYÙ[™\ˆ‹Y˜][H›X[H‹ÚÚXÙ\ÏJ›X[H‹™™[X[H‹›™]]˜[ŠJBˆ\œÙ\‹˜YØ\™Ý[Y[
‹KY]šXÙH‹Y˜][H˜ÝYHŠBˆ\œÙ\‹˜YØ\™Ý[Y[
‹K[X^Z]\ˆ‹\OZ[Y˜][LLŒ
Bˆ\œÙ\‹˜YØ\™Ý[Y[
‹K\ÜÙK\š[Ü‹]ÙZYÚ‹\OY›Ø]Y˜][LYKLÊBˆ\œÙ\‹˜YØ\™Ý[Y[
‹K\Ú\K\š[Ü‹]ÙZYÚ‹\OY›Ø]Y˜][LYKM
Bˆ\™ÜÈH\œÙ\‹œ\œÙWØ\™ÜÊ
Bˆ™\ÜHš]Ü™YXÝ[Û—ÙÙ[™\Šˆ\™ÜËœ™YXÝ[Û‹ˆ\™ÜË›Ý]]Ù\‹ˆ\™ÜËœ›Û\\—Ü™\Ëˆ\™ÜËœÛ\Û[Ù[Ù›Û\‹ˆ\™Ù]ÙÙ[™\X\™ÜË™Ù[™\‹ˆ]šXÙOX\™ÜË™]šXÙKˆX^Ú]\X\™ÜË›X^Ú]\‹ˆÜÙWÜš[Ü—ÝÙZYÚX\™ÜËœÜÙWÜš[Ü—ÝÙZYÚˆÚ\WÜš[Ü—ÝÙZYÚX\™ÜËœÚ\WÜš[Ü—ÝÙZYÚˆ
Bˆš[
œÛÛ‹™[\Ê™\Ü[™[LŠJB‚‚šYˆ×Û˜[YW×ÈOH—×ÛXZ[—×ÈŽ‚ˆXZ[Š
B