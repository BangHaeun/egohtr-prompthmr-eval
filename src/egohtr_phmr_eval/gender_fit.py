"""Fit a PromptHMR SMPL-X mesh to another SMPL-X gender model.

The fitter uses PromptHMR's installed differentiable SMPL-X layer and direct
vertex correspondences. Global orientation and translation stay fixed; body
pose and shape are optimized. No camera/world alignment is performed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _error_summary(fitted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    d = np.asarray(fitted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    e = np.linalg.norm(d, axis=-1)
    return {
        "mean_m": float(e.mean()),
        "rmse_m": float(np.sqrt(np.mean(e ** 2))),
        "median_m": float(np.median(e)),
        "p95_m": float(np.percentile(e, 95)),
        "max_m": float(e.max()),
    }


def fit_prediction_gender(
    prediction_path: str | Path,
    output_dir: str | Path,
    prompthmr_repo: str | Path,
    smplx_model_folder: str | Path,
    target_gender: str = "male",
    device: str = "cuda",
    max_iter: int = 120,
    pose_prior_weight: float = 1e-3,
    shape_prior_weight: float = 1e-4,
    max_initial_reconstruction_rmse_m: float = 1e-4,
) -> dict:
    """Fit target-gender SMPL-X to saved PromptHMR vertices and save separately.

    ``transl`` and ``global_orient`` are fixed to the original prediction.
    The objective is corresponding-vertex least squares with small priors to
    the original pose and betas. Body-pose updates are represented as local
    axis-angle deltas composed with the original rotation matrices.
    """
    import torch

    prediction_path = Path(prediction_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    prompthmr_repo = Path(prompthmr_repo).expanduser().resolve()
    smplx_model_folder = Path(smplx_model_folder).expanduser().resolve()
    if target_gender not in {"male", "female", "neutral"}:
        raise ValueError("target_gender must be male, female, or neutral")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(prediction_path, allow_pickle=False) as z:
        required = ("vertices", "body_joints", "rotmat", "betas", "transl", "faces")
        missing = [k for k in required if k not in z]
        if missing:
            raise KeyError(f"Prediction is missing required fields: {missing}")
        target_vertices_np = np.asarray(z["vertices"], dtype=np.float32).squeeze()
        target_joints_np = np.asarray(z["body_joints"], dtype=np.float32).squeeze()
        original_rotmat_np = np.asarray(z["rotmat"], dtype=np.float32).squeeze()
        original_betas_np = np.asarray(z["betas"], dtype=np.float32).reshape(-1, 10)[0]
        original_transl_np = np.asarray(z["transl"], dtype=np.float32).reshape(-1, 3)[0]
        target_faces = np.asarray(z["faces"], dtype=np.int32)
        cam_int_np = np.asarray(z["cam_int"], dtype=np.float32) if "cam_int" in z else None

    if target_vertices_np.ndim != 2 or target_vertices_np.shape[1] != 3:
        raise ValueError(f"Expected PromptHMR vertices (V,3), got {target_vertices_np.shape}")
    if target_joints_np.shape != (25, 3):
        raise ValueError(f"Expected PromptHMR BODY_25 (25,3), got {target_joints_np.shape}")
    if original_rotmat_np.shape != (22, 3, 3):
        raise ValueError(f"Expected PromptHMR rotmat (22,3,3), got {original_rotmat_np.shape}")

    old_cwd = Path.cwd()
    sys.path.insert(0, str(prompthmr_repo))
    try:
        os.chdir(prompthmr_repo)
        from prompt_hmr.smpl_family import SMPLX
        from prompt_hmr.utils.rotation_utils import axis_angle_to_matrix, matrix_to_axis_angle

        model = SMPLX(str(smplx_model_folder), gender=target_gender).to(device).eval()
        if int(model.v_template.shape[0]) != target_vertices_np.shape[0]:
            raise ValueError(
                f"Target has {target_vertices_np.shape[0]} vertices but SMPL-X has {model.v_template.shape[0]}"
            )
        model_faces = np.asarray(model.faces, dtype=np.int32)
        if model_faces.shape != target_faces.shape or not np.array_equal(model_faces, target_faces):
            raise ValueError("Target mesh topology does not match the PromptHMR SMPL-X model")

        dtype = torch.float32
        target_vertices = torch.as_tensor(target_vertices_np, device=device, dtype=dtype).unsqueeze(0)
        target_joints = torch.as_tensor(target_joints_np, device=device, dtype=dtype).unsqueeze(0)
        global_orient = torch.as_tensor(original_rotmat_np[:1], device=device, dtype=dtype).unsqueeze(0)
        initial_body_rotmat = torch.as_tensor(original_rotmat_np[1:], device=device, dtype=dtype).unsqueeze(0)
        initial_betas = torch.as_tensor(original_betas_np, device=device, dtype=dtype).unsqueeze(0)
        fixed_transl = torch.as_tensor(original_transl_np, device=device, dtype=dtype).unsqueeze(0)
        hand_mean = model.hands_mean.to(device=device, dtype=dtype)
        left_hand_pose = hand_mean[:, :15]
        right_hand_pose = hand_mean[:, 15:]

        def forward(betas, body_rotmat):
            return model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_rotmat,
                transl=fixed_transl,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
            )

        print("[fit 1/4] Reconstructing the saved PromptHMR mesh with Neutral SMPL-X...", flush=True)
        neutral = SMPLX(str(smplx_model_folder), gender="neutral").to(device).eval()
        neutral_hand_mean = neutral.hands_mean.to(device=device, dtype=dtype)
        with torch.no_grad():
            neutral_out = neutral(
                betas=initial_betas,
                global_orient=global_orient,
                body_pose=initial_body_rotmat,
                transl=fixed_transl,
                left_hand_pose=neutral_hand_mean[:, :15],
                right_hand_pose=neutral_hand_mean[:, 15:],
            )
        neutral_vertices_np = neutral_out.vertices[0].detach().cpu().numpy()
        neutral_joints_np = neutral_out.body_joints[0].detach().cpu().numpy()
        neutral_vertex_error = _error_summary(neutral_vertices_np, target_vertices_np)
        neutral_joint_error = _error_summary(neutral_joints_np, target_joints_np)
        print(f"  neutral vertex RMSE={neutral_vertex_error['rmse_m']:.8f} m", flush=True)
        if neutral_vertex_error["rmse_m"] > max_initial_reconstruction_rmse_m:
            raise RuntimeError(
                "Neutral SMPL-X parameters do not reproduce the saved PromptHMR mesh "
                f"(RMSE {neutral_vertex_error['rmse_m']:.8g} m exceeds "
                f"{max_initial_reconstruction_rmse_m:g} m); refusing to fit a mismatched target."
            )
        del neutral, neutral_out

        print("[fit 2/4] Measuring initial Male SMPL-X fit...", flush=True)
        with torch.no_grad():
            male_initial = forward(initial_betas, initial_body_rotmat)
        male_initial_vertices = male_initial.vertices.detach()
        male_initial_joints = male_initial.body_joints.detach()
        initial_vertex_error = _error_summary(male_initial_vertices[0].cpu().numpy(), target_vertices_np)
        initial_joint_error = _error_summary(male_initial_joints[0].cpu().numpy(), target_joints_np)

        print("[fit 3/4] Fitting Male body pose and betas; global orientation and translation are fixed...", flush=True)
        # Small tangent-space pose updates preserve valid rotations at every step.
        pose_delta = torch.nn.Parameter(torch.zeros((1, 21, 3), device=device, dtype=dtype))
        fitted_betas = torch.nn.Parameter(initial_betas.detach().clone())
        optimizer = torch.optim.LBFGS(
            [pose_delta, fitted_betas],
            lr=0.8,
            max_iter=int(max_iter),
            history_size=30,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0

        def closure():
            nonlocal closure_calls
            optimizer.zero_grad(set_to_none=True)
            delta_rotmat = axis_angle_to_matrix(pose_delta)
            body_rotmat = torch.matmul(delta_rotmat, initial_body_rotmat)
            out = forward(fitted_betas, body_rotmat)
            vertex_loss = torch.mean((out.vertices - target_vertices) ** 2)
            pose_prior = torch.mean(pose_delta ** 2)
            shape_prior = torch.mean((fitted_betas - initial_betas) ** 2)
            loss = vertex_loss + pose_prior_weight * pose_prior + shape_prior_weight * shape_prior
            loss.backward()
            closure_calls += 1
            if closure_calls == 1 or closure_calls % 25 == 0:
                print(
                    f"  closure {closure_calls}: vertex_mse={vertex_loss.detach().item():.8g} m^2, "
                    f"total={loss.detach().item():.8g}", flush=True
                )
            return loss

        optimizer.step(closure)

        print("[fit 4/4] Saving parameters and verifying vertex/joint reconstruction...", flush=True)
        with torch.no_grad():
            fitted_body_rotmat = torch.matmul(axis_angle_to_matrix(pose_delta), initial_body_rotmat)
            fitted_out = forward(fitted_betas, fitted_body_rotmat)
        fitted_vertices_np = fitted_out.vertices[0].detach().cpu().numpy().astype(np.float32)
        fitted_joints_np = fitted_out.body_joints[0].detach().cpu().numpy().astype(np.float32)
        fitted_body_rotmat_np = fitted_body_rotmat[0].detach().cpu().numpy().astype(np.float32)
        fitted_global_rotmat_np = global_orient[0, 0].detach().cpu().numpy().astype(np.float32)
        fitted_betas_np = fitted_betas[0].detach().cpu().numpy().astype(np.float32)
        fixed_transl_np = fixed_transl[0].detach().cpu().numpy().astype(np.float32)
        fitted_body_aa_np = matrix_to_axis_angle(fitted_body_rotmat[0]).detach().cpu().numpy().astype(np.float32)
        fitted_global_aa_np = matrix_to_axis_angle(global_orient[0, 0]).detach().cpu().numpy().astype(np.float32)
        fitted_vertex_error = _error_summary(fitted_vertices_np, target_vertices_np)
        fitted_joint_error = _error_summary(fitted_joints_np, target_joints_np)
        pose_delta_np = pose_delta.detach().cpu().numpy()[0]
        pose_delta_joint_angle_deg = np.degrees(np.linalg.norm(pose_delta_np, axis=-1))

        fitted_payload = dict(
            gender=np.asarray(target_gender),
            vertices=fitted_vertices_np,
            body_joints=fitted_joints_np,
            faces=target_faces,
            global_orient_rotmat=fitted_global_rotmat_np,
            global_orient_axis_angle=fitted_global_aa_np,
            body_pose_rotmat=fitted_body_rotmat_np,
            body_pose_axis_angle=fitted_body_aa_np,
            betas=fitted_betas_np,
            transl=fixed_transl_np,
            left_hand_pose_rotmat=left_hand_pose[0].detach().cpu().numpy().astype(np.float32),
            right_hand_pose_rotmat=right_hand_pose[0].detach().cpu().numpy().astype(np.float32),
            pose_delta_axis_angle=pose_delta_np,
            source_prediction=np.asarray(str(prediction_path)),
        )
        if cam_int_np is not None:
            fitted_payload["cam_int"] = cam_int_np
        np.savez_compressed(output_dir / f"fitted_{target_gender}_prediction.npz", **fitted_payload)
        report = {
            "source_prediction": str(prediction_path),
            "output_prediction": str(output_dir / f"fitted_{target_gender}_prediction.npz"),
            "target_gender": target_gender,
            "backend": "PromptHMR's installed differentiable SMPL-X layer; corresponding-vertex L-BFGS",
            "global_orient_fixed": True,
            "translation_fixed": True,
            "transl_preserved_exactly": bool(np.array_equal(fixed_transl_np, original_transl_np)),
            "global_orient_preserved_exactly": bool(np.array_equal(fitted_global_rotmat_np, original_rotmat_np[0])),
            "original_global_orient_rotmat": original_rotmat_np[0].tolist(),
            "translation_m": fixed_transl_np.tolist(),
            "pose_prior_weight": float(pose_prior_weight),
            "shape_prior_weight": float(shape_prior_weight),
            "lbfgs_max_iter": int(max_iter),
            "lbfgs_closure_calls": int(closure_calls),
            "neutral_reconstruction": {
                "vertices": neutral_vertex_error,
                "joints": neutral_joint_error,
            },
            f"{target_gender}_before_fit": {
                "vertices": initial_vertex_error,
                "joints": initial_joint_error,
            },
            f"{target_gender}_after_fit": {
                "vertices": fitted_vertex_error,
                "joints": fitted_joint_error,
            },
            "fitted_betas": fitted_betas_np.tolist(),
            "original_betas": original_betas_np.tolist(),
            "beta_delta_l2": float(np.linalg.norm(fitted_betas_np - original_betas_np)),
            "pose_delta_axis_angle_rms_rad": float(np.sqrt(np.mean(pose_delta_np ** 2))),
            "body_pose_delta_joint_angle_mean_deg": float(pose_delta_joint_angle_deg.mean()),
            "body_pose_delta_joint_angle_max_deg": float(pose_delta_joint_angle_deg.max()),
            "body_pose_delta_joint_angle_deg": pose_delta_joint_angle_deg.tolist(),
            "vertex_error_metric": "per-vertex Euclidean distance between fitted mesh and original PromptHMR target mesh, meters",
            "joint_error_metric": "per-joint Euclidean distance between fitted BODY_25 and original PromptHMR BODY_25, meters",
        }
        (output_dir / "fit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit saved PromptHMR SMPL-X mesh to another model gender")
    parser.add_argument("--prediction", required=True, help="PromptHMR prediction.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompthmr-repo", required=True)
    parser.add_argument("--smplx-model-folder", required=True)
    parser.add_argument("--gender", default="male", choices=("male", "female", "neutral"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--pose-prior-weight", type=float, default=1e-3)
    parser.add_argument("--shape-prior-weight", type=float, default=1e-4)
    args = parser.parse_args()
    report = fit_prediction_gender(
        args.prediction,
        args.output_dir,
        args.prompthmr_repo,
        args.smplx_model_folder,
        target_gender=args.gender,
        device=args.device,
        max_iter=args.max_iter,
        pose_prior_weight=args.pose_prior_weight,
        shape_prior_weight=args.shape_prior_weight,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
