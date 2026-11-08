from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .dataset import ensure_sequence_files, iter_npz_rgb_frames
from .gt import BODY25_NAMES
from .metrics import compute_mpjpe, compute_pa_mpjpe
from .transforms import apply_transform
from .video_metrics import aggregate_window_metric, compute_window_metrics


def _expand(value):
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def _load_sequence_arrays(archive: Path, start: int, stop: int):
    with np.load(archive, allow_pickle=False) as data:
        total = len(data["aria_timestamp"])
        stop = total if stop is None else min(int(stop), total)
        if not (0 <= start < stop <= total):
            raise ValueError(f"Invalid video range [{start}, {stop}) for {total} frames")
        aria_ts = np.asarray(data["aria_timestamp"][start:stop], dtype=np.float64)
        exo_ts = np.asarray(data["exomc_timestamp"][start:stop], dtype=np.float64)
        if not np.allclose(aria_ts, exo_ts, rtol=0, atol=1e-5):
            raise ValueError("aria_timestamp and exomc_timestamp do not share the selected time grid")
        device = np.asarray(data["aria_obs_T_origin_device"][start:stop], dtype=np.float64)
        device_camera = np.asarray(data["aria_obs_T_device_camera"], dtype=np.float64)
        transforms = device @ device_camera
        gt = np.asarray(data["smplx_exomc_joints"][start:stop], dtype=np.float32)
        gt_params = {
            "body_pose": np.asarray(data["smplx_exomc_body_pose"][start:stop], dtype=np.float32),
            "global_orient": np.asarray(data["smplx_exomc_global_orient"][start:stop], dtype=np.float32),
            "betas": np.asarray(data["smplx_exomc_betas"][start:stop], dtype=np.float32),
            "transl": np.asarray(data["smplx_exomc_transl"][start:stop], dtype=np.float32),
        }
        required_intrinsics = ("aria_obs_rgb_focal_length", "aria_obs_rgb_img_width", "aria_obs_rgb_img_height")
        missing = [key for key in required_intrinsics if key not in data]
        if missing:
            raise KeyError(f"full_sequence.npz lacks RGB calibration fields {missing}")
        focal = float(np.asarray(data[required_intrinsics[0]]).reshape(-1)[0])
        width = int(np.asarray(data[required_intrinsics[1]]).reshape(-1)[0])
        height = int(np.asarray(data[required_intrinsics[2]]).reshape(-1)[0])
    if transforms.shape != (stop - start, 4, 4) or gt.ndim != 3 or gt.shape[1] <= 65:
        raise ValueError(f"Unexpected pose/joint shapes: T={transforms.shape}, joints={gt.shape}")
    return np.arange(start, stop, dtype=np.int64), aria_ts, transforms, gt, gt_params, (focal, width, height)


def _write_npz_rgb_mp4(archive: Path, indices: np.ndarray, timestamps: np.ndarray, path: Path):
    """Encode the contiguous EgoHTR RGB range as MP4 for PromptHMR's official video loader."""
    if len(indices) < 2 or not np.all(np.diff(indices) == 1):
        raise ValueError("PromptHMR MP4 input requires at least two contiguous source frames")
    dt = np.diff(timestamps)
    fps = float(1.0 / np.median(dt))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid source video frame rate inferred from timestamps: {fps}")
    # A constant-frame-rate MP4 can preserve the source-frame mapping only when
    # the EgoHTR samples are regularly spaced. Keep original timestamps in the
    # result archive; reject irregular grids instead of silently retiming them.
    max_dt_error = float(np.max(np.abs(dt - np.median(dt))))
    if max_dt_error > max(1e-5, 1e-3 / fps):
        raise ValueError(f"EgoHTR timestamps are not a regular video grid (max interval deviation {max_dt_error:.3g}s)")

    writer = None
    count = 0
    try:
        for source_i, frame in iter_npz_rgb_frames(
            archive, "aria_obs_rgb_imgs", int(indices[0]), int(indices[-1]) + 1
        ):
            if source_i != indices[count]:
                raise RuntimeError(f"RGB source index mismatch: got {source_i}, expected {indices[count]}")
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"Expected uint8 RGB frame, got {frame.dtype} {frame.shape}")
            height, width = frame.shape[:2]
            if height % 2 or width % 2:
                raise ValueError(f"MP4 requires even image dimensions; got {width}x{height}")
            if writer is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create EgoHTR MP4 input at {path}")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            count += 1
            if count == 1 or count % 100 == 0 or count == len(indices):
                print(f"[video] MP4 frames encoded: {count}/{len(indices)}", flush=True)
    finally:
        if writer is not None:
            writer.release()
    if count != len(indices):
        raise RuntimeError(f"Encoded {count} frames for requested range of {len(indices)}")
    return fps


def _project_gt_box(joints_origin, T_origin_camera, K):
    # Row point convention: camera -> origin is T_origin_camera.
    points_camera = (joints_origin - T_origin_camera[:3, 3]) @ T_origin_camera[:3, :3]
    z = points_camera[:, 2]
    valid = np.isfinite(points_camera).all(axis=1) & (z > 1e-4)
    pixels = np.full((len(points_camera), 2), np.nan)
    pixels[valid, 0] = K[0, 0] * points_camera[valid, 0] / z[valid] + K[0, 2]
    pixels[valid, 1] = K[1, 1] * points_camera[valid, 1] / z[valid] + K[1, 2]
    h, w = int(2 * K[1, 2]), int(2 * K[0, 2])
    inside = valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    if inside.sum() < 2:
        return None, int(inside.sum())
    xy = pixels[inside]
    return np.asarray([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()]), int(inside.sum())


def _box_iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def _choose_track(people, source_indices, gt_body25, transforms, K, configured_track=None):
    if not people:
        raise RuntimeError("PromptHMR tracking found no people in the selected video frames")
    if configured_track is not None:
        for key in people:
            if str(key) == str(configured_track):
                return key, {str(key): {"mean_gt_box_iou": None, "selection": "configured"}}
        raise KeyError(f"Configured video.track_id={configured_track!r} not in PromptHMR tracks {list(people)}")

    scores = {}
    for key, track in people.items():
        frames = np.asarray(track["frames"], dtype=np.int64)
        boxes = np.asarray(track["bboxes"], dtype=np.float64)
        ious = []
        for local_i, box in zip(frames, boxes):
            gt_box, _ = _project_gt_box(gt_body25[local_i], transforms[local_i], K)
            if gt_box is not None:
                ious.append(_box_iou(gt_box, box))
        scores[str(key)] = {
            "mean_gt_box_iou": float(np.mean(ious)) if ious else None,
            "frames": int(len(frames)), "selection": "projected_gt_bbox_iou",
        }
    with_iou = [(float(v["mean_gt_box_iou"]), key) for key, v in scores.items()
                if v["mean_gt_box_iou"] is not None]
    if with_iou and max(v[0] for v in with_iou) > 0:
        selected = max(with_iou)[1]
    else:
        selected = max(people, key=lambda key: len(people[key]["frames"]))
        scores[str(selected)]["selection"] = "longest_track_fallback_no_bbox_overlap"
    actual_key = next(key for key in people if str(key) == str(selected))
    return actual_key, scores


def _run_video_hps_with_camera(pipeline, camera_intrinsic, T_origin_camera, device):
    """Reuse official PromptHMR image/video heads, replacing only identity cam motion."""
    from scipy.ndimage import gaussian_filter1d
    from scipy import signal
    from torch.amp import autocast
    from torch.utils.data import DataLoader
    from pipeline.phmr_vid import PromptHMR_Video, PromptHMRVideoDataset
    from pipeline.gvhmr.hmr4d.utils.geo_transform import compute_cam_angvel
    from pipeline.gvhmr.hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, normalize_kp2d
    from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d

    if not str(device).startswith("cuda"):
        raise ValueError("PromptHMR official video inference currently requires CUDA")
    people = pipeline.results["people"]
    phmr = PromptHMR_Video()
    cam_int = torch.as_tensor(camera_intrinsic, dtype=torch.float32)
    dataset = PromptHMRVideoDataset(pipeline.images, people, cam_int)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=lambda x: x)
    for track in people.values():
        track["smplx_pose"], track["smplx_transl"], track["smplx_betas"], track["prhmr_img_feats"] = [], [], [], []

    print("[video] Running official PromptHMR per-frame image head...", flush=True)
    for batch_i, batch in enumerate(dataloader):
        with torch.no_grad(), autocast("cuda"):
            output = phmr.model(batch, mask_prompt=pipeline.cfg.tracker == "sam2")
        for bid, item in enumerate(batch):
            for tid in item["track_ids"]:
                btid = item["track_ids"].index(tid)
                people[tid]["smplx_pose"].append(output[bid]["rotmat"][btid].detach())
                people[tid]["smplx_transl"].append(output[bid]["transl"][btid].detach())
                people[tid]["smplx_betas"].append(output[bid]["betas"][btid].detach())
                people[tid]["prhmr_img_feats"].append(output[bid]["features"][btid].detach())
        if batch_i == 0 or (batch_i + 1) % 25 == 0 or batch_i + 1 == len(dataloader):
            print(f"[video] image head batches: {batch_i + 1}/{len(dataloader)}", flush=True)

    for track in people.values():
        for key in ("smplx_pose", "smplx_transl", "smplx_betas", "prhmr_img_feats"):
            track[key] = torch.stack(track[key]).float()

    vid_head = phmr.vid_head
    print("[video] Running official temporal head with EgoHTR camera angular velocity...", flush=True)
    for pid, track in people.items():
        frames = np.asarray(track["frames"], dtype=np.int64)
        starts = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1]
        stops = np.r_[starts[1:], len(frames)]
        accum = {k: [] for k in ("rotmat", "pose", "shape", "trans", "contact", "static_conf_logits")}
        for begin, end in zip(starts, stops):
            sl = slice(int(begin), int(end))
            track_frames = frames[sl]
            seqlen = len(track_frames)
            R_w2c = torch.as_tensor(T_origin_camera[track_frames, :3, :3].transpose(0, 2, 1), dtype=torch.float32)
            if seqlen > 1:
                cam_angvel = compute_cam_angvel(R_w2c)
            else:
                cam_angvel = matrix_to_rotation_6d(torch.eye(3))[None]
            bbx_xys = get_bbx_xys_from_xyxy(torch.from_numpy(np.asarray(track["bboxes"])[sl])).numpy()
            smoothed = np.array([signal.medfilt(param, 11) for param in bbx_xys.T]).T
            bbx_xys = np.array([gaussian_filter1d(traj, 3) for traj in smoothed.T]).T
            bbx_xys = torch.from_numpy(bbx_xys).float()
            vitpose = torch.from_numpy(np.asarray(track["vitpose"])[sl])
            vitpose = normalize_kp2d(vitpose, bbx_xys).float()
            batch = {
                "length": torch.tensor([seqlen]), "obs": vitpose[None],
                "bbx_xys": bbx_xys[None],
                "K_fullimg": cam_int[None, None].repeat_interleave(seqlen, 1),
                "cam_angvel": cam_angvel[None],
                "f_imgseq": track["prhmr_img_feats"][sl][None],
            }
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad(), autocast("cuda"):
                output_w_kpts = vid_head.pipeline.forward(batch, train=False, postproc=False, static_cam=True)
                batch["obs"] = torch.zeros_like(batch["obs"])
                output = vid_head.pipeline.forward(batch, train=False, postproc=False, static_cam=True)
            contact = output["model_output"]["static_conf_logits"].sigmoid() > 0.8
            params = output["pred_smpl_params_incam"]
            pose_aa = torch.cat([params["global_orient"][0], params["body_pose"][0],
                                 torch.zeros(seqlen, 9, device=device)], dim=-1)
            rotmat = axis_angle_to_matrix(pose_aa.reshape(-1, 3)).reshape(-1, 25, 3, 3)
            hands = torch.zeros(rotmat.shape[0], 30, 3, 3, device=rotmat.device, dtype=rotmat.dtype)
            rotmat = torch.cat([rotmat, hands], dim=1)
            values = {
                "rotmat": rotmat.cpu().float().numpy(), "pose": pose_aa.cpu().numpy(),
                "shape": params["betas"][0].cpu().float().numpy(),
                "trans": output_w_kpts["pred_smpl_params_incam"]["transl"][0].cpu().float().numpy(),
                "contact": contact[0].cpu().numpy(),
                "static_conf_logits": output["model_output"]["static_conf_logits"][0].cpu().numpy(),
            }
            for key, value in values.items():
                accum[key].append(value)
        track["smplx_cam"] = {key: np.concatenate(value, axis=0) for key, value in accum.items()}
        track["smplx_cam"]["rotmat"] = np.concatenate(accum["rotmat"], axis=0)
        print(f"[video] temporal segments complete for track {pid} ({len(starts)} segment(s))", flush=True)
    pipeline.results["contact_joint_ids"] = [7, 10, 8, 11, 20, 21]
    pipeline.results["has_hps_cam"] = True
    return pipeline.results


def _make_origin_model(model_folder, repo, device):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from prompt_hmr.smpl_family import SMPLX
    return SMPLX(str(model_folder), gender="neutral").to(device).eval()


def _fit_display_meshes(target_vertices, pose_aa, betas, transl, repo, model_folder, gender, device,
                        max_iter, pose_prior_weight, shape_prior_weight, batch_size):
    """Apply image-mode's regularized gender fit to video meshes for display only."""
    from prompt_hmr.smpl_family import SMPLX
    from prompt_hmr.utils.rotation_utils import axis_angle_to_matrix, matrix_to_axis_angle

    fitted_vertices, fitted_joints, fitted_pose, fitted_betas = [], [], [], []
    max_source_rmse = 0.0
    model = SMPLX(str(model_folder), gender=gender).to(device).eval()
    neutral = SMPLX(str(model_folder), gender="neutral").to(device).eval()
    dtype = torch.float32
    n = len(target_vertices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        B = end - start
        target_np = np.asarray(target_vertices[start:end], dtype=np.float32)
        pose_np = np.asarray(pose_aa[start:end], dtype=np.float32)
        if pose_np.ndim == 2 and pose_np.shape[1] == 165:
            pose_np = pose_np.reshape(B, 55, 3)
        beta_np = np.asarray(betas[start:end], dtype=np.float32)
        trans_np = np.asarray(transl[start:end], dtype=np.float32)
        if pose_np.shape != (B, 55, 3):
            raise ValueError(f"Expected video SMPL-X pose (B,55,3), got {pose_np.shape}")
        pose_rot = axis_angle_to_matrix(torch.as_tensor(pose_np, device=device, dtype=dtype))
        beta0 = torch.as_tensor(beta_np, device=device, dtype=dtype)
        trans = torch.as_tensor(trans_np, device=device, dtype=dtype)
        target = torch.as_tensor(target_np, device=device, dtype=dtype)
        hands_l, hands_r = pose_rot[:, 25:40], pose_rot[:, 40:55]
        fixed_global = pose_rot[:, :1]
        body0 = pose_rot[:, 1:22]
        jaw_eye = torch.eye(3, device=device, dtype=dtype)[None].expand(B, 3, 3)
        with torch.no_grad():
            neutral_out = neutral(
                betas=beta0, global_orient=fixed_global, body_pose=body0,
                left_hand_pose=hands_l, right_hand_pose=hands_r, transl=trans,
                jaw_pose=jaw_eye, leye_pose=jaw_eye, reye_pose=jaw_eye,
                expression=torch.zeros(B, 10, device=device), pose2rot=False,
            )
            source_rmse = torch.mean((neutral_out.vertices - target) ** 2).sqrt().item()
        max_source_rmse = max(max_source_rmse, source_rmse)
        if source_rmse > 1e-4:
            raise RuntimeError(f"Neutral SMPL-X did not reproduce video target mesh (RMSE={source_rmse:.6g}m)")

        pose_delta = torch.nn.Parameter(torch.zeros((B, 21, 3), device=device, dtype=dtype))
        fitted_beta = torch.nn.Parameter(beta0.detach().clone())
        optimizer = torch.optim.LBFGS(
            [pose_delta, fitted_beta], lr=0.8, max_iter=int(max_iter), history_size=30,
            tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            body = torch.matmul(axis_angle_to_matrix(pose_delta), body0)
            out = model(
                betas=fitted_beta, global_orient=fixed_global, body_pose=body,
                left_hand_pose=hands_l, right_hand_pose=hands_r, transl=trans,
                jaw_pose=jaw_eye, leye_pose=jaw_eye, reye_pose=jaw_eye,
                expression=torch.zeros(B, 10, device=device), pose2rot=False,
            )
            vertex_loss = torch.mean((out.vertices - target) ** 2)
            loss = (vertex_loss + pose_prior_weight * torch.mean(pose_delta ** 2)
                    + shape_prior_weight * torch.mean((fitted_beta - beta0) ** 2))
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            body = torch.matmul(axis_angle_to_matrix(pose_delta), body0)
            out = model(
                betas=fitted_beta, global_orient=fixed_global, body_pose=body,
                left_hand_pose=hands_l, right_hand_pose=hands_r, transl=trans,
                jaw_pose=jaw_eye, leye_pose=jaw_eye, reye_pose=jaw_eye,
                expression=torch.zeros(B, 10, device=device), pose2rot=False,
            )
        fitted_vertices.append(out.vertices.cpu().numpy().astype(np.float32))
        fitted_joints.append(out.body_joints[:, :25].cpu().numpy().astype(np.float32))
        fitted_pose.append(matrix_to_axis_angle(torch.cat([fixed_global, body, pose_rot[:, 22:]], dim=1)).cpu().numpy().astype(np.float32))
        fitted_betas.append(fitted_beta.detach().cpu().numpy().astype(np.float32))
        print(f"[video] {gender} display mesh fitting: {end}/{n} frames; neutral reconstruction RMSE={source_rmse:.3g}m", flush=True)
    return (np.concatenate(fitted_vertices), np.concatenate(fitted_joints),
            np.concatenate(fitted_pose), np.concatenate(fitted_betas), max_source_rmse)


def run_video(cfg, config_path, *, start_index=None, stop_index=None, no_viewer=False):
    dataset_root = _expand(cfg["dataset_root"])
    sequence = str(cfg["sequence"])
    archive, seq_config = ensure_sequence_files(dataset_root, sequence, cfg.get("huggingface_repo_id"))
    sequence_cfg = yaml.safe_load(seq_config.read_text(encoding="utf-8")) or {}
    gender = (sequence_cfg.get("smplx_config") or {}).get("gender")
    if not gender:
        raise KeyError(f"{seq_config} lacks smplx_config.gender")
    video_cfg = dict(cfg.get("video") or {})
    start = int(start_index if start_index is not None else video_cfg.get("start_index", 0))
    stop = stop_index if stop_index is not None else video_cfg.get("stop_index")
    source_indices, timestamps, T_origin_camera, gt_joints_raw, gt_params, (focal, rgb_w, rgb_h) = _load_sequence_arrays(archive, start, stop)
    image_size = int(video_cfg.get("image_size", 896))
    if image_size != 896:
        raise ValueError("PromptHMR's official Pipeline.load_frames currently fixes max_height=896; set video.image_size=896")
    K_full = np.asarray([[focal, 0, rgb_w / 2], [0, focal, rgb_h / 2], [0, 0, 1]], dtype=np.float32)
    if abs(rgb_w - rgb_h) > 0:
        raise ValueError("Official PromptHMR video preprocessing uses a square padded image; current EgoHTR RGB is not square")
    scale = image_size / max(rgb_w, rgb_h)
    K = K_full.copy()
    K[:2] *= scale
    K[:2, 2] += (image_size - scale * np.asarray([rgb_w, rgb_h], dtype=np.float32)) / 2
    output_dir = _expand(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[video] Sequence {sequence}; source frames [{source_indices[0]}, {source_indices[-1] + 1}); "
          f"{len(source_indices)} frames at {timestamps[0]:.6f}–{timestamps[-1]:.6f}s", flush=True)
    print(f"[video] Intrinsics source f={focal:.4f}px at {rgb_w}x{rgb_h}; processed K={K.tolist()}", flush=True)
    repo = _expand(cfg["prompthmr_repo"])
    model_folder = Path(os.path.expandvars(os.path.expanduser(str(cfg.get("smplx_model_folder", "data/body_models/smplx")))))
    if not model_folder.is_absolute():
        model_folder = (repo / model_folder).resolve()
    device = str(cfg.get("device", "cuda"))
    previous_cwd = Path.cwd()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(repo)
    try:
        print("[video] Initializing official PromptHMR video pipeline...", flush=True)
        from pipeline.pipeline import Pipeline
        from pipeline.world import world_hps_estimation
        from pipeline.postprocessing import post_optimization
        from prompt_hmr.smpl_family.vertex_ids import smplx_to_openpose

        body25_indices = np.asarray(smplx_to_openpose, dtype=np.int64)
        if body25_indices.shape != (25,):
            raise ValueError(f"PromptHMR mapping is not BODY_25: {body25_indices.shape}")
        gt_body25 = gt_joints_raw[:, body25_indices].copy()
        pipeline = Pipeline()
        pipeline.cfg.tracker = str(video_cfg.get("tracker", "sam2"))
        pipeline.cfg.num_max_people = int(video_cfg.get("num_max_people", pipeline.cfg.num_max_people))
        pipeline.cfg.det_thresh = float(video_cfg.get("det_thresh", pipeline.cfg.det_thresh))
        pipeline.cfg.det_score_thresh = float(video_cfg.get("det_score_thresh", pipeline.cfg.det_score_thresh))
        pipeline.cfg.det_height_thresh = float(video_cfg.get("det_height_thresh", pipeline.cfg.det_height_thresh))
        pipeline.cfg.bbox_interp = bool(video_cfg.get("bbox_interp", pipeline.cfg.bbox_interp))
        pipeline.cfg.smooth_cam = False
        pipeline.cfg.use_spec_calib = False
        pipeline.cfg.use_floor_rectify = False
        pipeline.cfg.run_post_opt = bool(video_cfg.get("run_post_opt", True))
        # The official optimizer has a calibration correction for its estimated camera.
        # Here the per-frame EgoHTR trajectory is authoritative; optimize body translation only.
        pipeline.cfg.run_post_opt_cam = False
        pipeline.cfg.postopt_lr = float(video_cfg.get("postopt_lr", pipeline.cfg.postopt_lr))
        work_dir = output_dir / "prompthmr_video_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        input_mp4 = work_dir / f"egohtr_rgb_{int(source_indices[0]):06d}_{int(source_indices[-1]) + 1:06d}.mp4"
        print(f"[video] Encoding EgoHTR RGB range to MP4 for PromptHMR: {input_mp4}", flush=True)
        source_fps = _write_npz_rgb_mp4(archive, source_indices, timestamps, input_mp4)
        print("[video] Decoding MP4 through official PromptHMR Pipeline.load_frames()...", flush=True)
        import pipeline.utils as prompthmr_video_utils
        original_codec_probe = prompthmr_video_utils.get_video_codec
        if shutil.which("ffprobe") is None:
            # PromptHMR's loader uses ffprobe only to detect AV1. Our locally
            # generated MP4V file is OpenCV-readable; retain its official
            # load_video_frames/video2frames decoder without editing that repo.
            print("[video] ffprobe unavailable; bypassing codec probe and using PromptHMR's OpenCV MP4 decoder", flush=True)
            prompthmr_video_utils.get_video_codec = lambda _path: "mp4v"
        try:
            pipeline.images, seq_folder = pipeline.load_frames(str(input_mp4), str(work_dir), read_frames=True)
        finally:
            prompthmr_video_utils.get_video_codec = original_codec_probe
        pipeline.images = np.asarray(pipeline.images)
        if len(pipeline.images) != len(source_indices):
            raise RuntimeError(
                f"PromptHMR MP4 decoder returned {len(pipeline.images)} frames for {len(source_indices)} EgoHTR frames; "
                "source-index mapping would be ambiguous"
            )
        if pipeline.images.ndim != 4 or pipeline.images.shape[1:3] != (image_size, image_size):
            raise ValueError(f"Unexpected official PromptHMR decoded frame shape {pipeline.images.shape}")
        pipeline.fps = float(source_fps)
        pipeline.cfg.fps = float(source_fps)
        pipeline.seq_folder = str(seq_folder)
        pipeline.cfg.seq_folder = str(seq_folder)
        print(f"[video] MP4 mapping verified: {len(pipeline.images)} decoded frames ↔ "
              f"EgoHTR indices [{source_indices[0]}, {source_indices[-1]}], fps={source_fps:.6f}", flush=True)
        pipeline.results = {
            "camera": {"img_focal": float(K[0, 0]), "img_center": K[:2, 2].copy(),
                       "pred_cam_R": T_origin_camera[:, :3, :3],
                       "pred_cam_T": T_origin_camera[:, :3, 3]},
            "people": {}, "timings": {}, "masks": None,
            "has_tracks": False, "has_hps_cam": False, "has_hps_world": False,
            "has_slam": True, "has_hands": False, "has_2d_kpts": False, "has_post_opt": False,
        }
        pipeline.results["masks"] = None
        print("[video] Detecting, segmenting, and tracking with official PromptHMR stages...", flush=True)
        pipeline.run_detect_track()
        print(f"[video] Tracks found: {[(k, len(v['frames'])) for k, v in pipeline.results['people'].items()]}", flush=True)
        print("[video] Estimating official ViTPose 2D keypoints...", flush=True)
        pipeline.estimate_2d_keypoints()
        pipeline.results = _run_video_hps_with_camera(pipeline, K, T_origin_camera, device)

        track_id, association = _choose_track(
            pipeline.results["people"], source_indices, gt_body25, T_origin_camera, K,
            video_cfg.get("track_id"),
        )
        print(f"[video] Evaluation track: {track_id}; association={association[str(track_id)]}", flush=True)

        # world_hps_estimation builds CPU tensors internally; post_optimization
        # moves this same module to CUDA itself when that stage runs.
        pipeline.smplx = pipeline.smplx.eval()
        print("[video] Applying official world_hps_estimation in its recoverable canonical frame...", flush=True)
        with torch.no_grad():
            pipeline.results = world_hps_estimation(pipeline.cfg, pipeline.results, pipeline.smplx)
        # Official world.py uses Q=diag(1,-1,-1), then subtracts a floor offset.
        Q = np.diag([1.0, -1.0, -1.0])
        canonical_camera = pipeline.results["camera_world"]
        offsets = np.einsum("ij,fj->fi", Q, T_origin_camera[:, :3, 3]) - canonical_camera["Twc"]
        offset = np.median(offsets, axis=0)
        rotation_residual = np.max(np.abs(canonical_camera["Rwc"] - Q[None] @ T_origin_camera[:, :3, :3]))
        offset_residual = np.max(np.abs(offsets - offset[None]))
        if rotation_residual > 2e-4 or offset_residual > 2e-3:
            raise RuntimeError("Could not recover PromptHMR world.py canonical transform from EgoHTR camera poses: "
                               f"rotation residual={rotation_residual:.3g}, offset residual={offset_residual:.3g}")
        pipeline.results["canonicalization"] = {
            "rotation": Q.tolist(), "offset": offset.tolist(),
            "rotation_max_residual": float(rotation_residual), "offset_max_residual_m": float(offset_residual),
        }
        if pipeline.cfg.run_post_opt:
            print("[video] Running official post_optimization with camera pose optimization disabled...", flush=True)
            pipeline.results = post_optimization(pipeline.cfg, pipeline.results, pipeline.images, pipeline.smplx)
        else:
            print("[video] Official post_optimization disabled by video.run_post_opt=false", flush=True)

        track = pipeline.results["people"][track_id]
        local_frames = np.asarray(track["frames"], dtype=np.int64)
        eval_source_indices = source_indices[local_frames]
        min_visible = int(video_cfg.get("min_gt_visible_joints", 8))
        visible_counts = []
        exclusions = {}
        keep = []
        for pos, local_i in enumerate(local_frames):
            _, count = _project_gt_box(gt_body25[local_i], T_origin_camera[local_i], K)
            visible_counts.append(count)
            if count < min_visible:
                exclusions[int(source_indices[local_i])] = f"GT visible joints {count} < configured minimum {min_visible}"
            elif not np.isfinite(gt_body25[local_i]).all():
                exclusions[int(source_indices[local_i])] = "GT BODY_25 contains non-finite coordinates"
            else:
                keep.append(pos)
        for local_i in range(len(source_indices)):
            if local_i not in set(local_frames.tolist()):
                exclusions[int(source_indices[local_i])] = "selected PromptHMR track missing in this frame or tracking gap"
        if len(keep) < 2:
            raise RuntimeError(f"Only {len(keep)} valid GT/prediction frames remain; at least 2 required")
        keep = np.asarray(keep, dtype=np.int64)
        kept_local = local_frames[keep]
        kept_source = source_indices[kept_local]

        # Preserve raw camera-frame video parameters before official world conversion mutates them.
        cam_raw = track["smplx_cam"]
        raw_cam_pose = np.asarray(cam_raw["pose"]).copy()
        raw_cam_shape = np.asarray(cam_raw["shape"]).copy()
        raw_cam_trans = np.asarray(cam_raw["trans"]).copy()
        raw_cam_rotmat = np.asarray(cam_raw["rotmat"]).copy()
        track["smplx_world"]["pose"] = np.asarray(track["smplx_world"]["pose"])
        track["smplx_world"]["shape"] = np.asarray(track["smplx_world"]["shape"])
        track["smplx_world"]["trans"] = np.asarray(track["smplx_world"]["trans"])

        print("[video] Reconstructing selected-track mesh/joints and reversing world.py's canonical frame...", flush=True)
        model = _make_origin_model(model_folder, repo, device)
        # Invert the recoverable canonicalization at the SMPL-X parameter level,
        # then regenerate vertices/joints directly in SLAM Origin.
        from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
        from pipeline.world import transform_smpl_params
        pose_world = torch.as_tensor(track["smplx_world"]["pose"], dtype=torch.float32, device=device)
        beta_world = torch.as_tensor(track["smplx_world"]["shape"], dtype=torch.float32, device=device)
        trans_world = torch.as_tensor(track["smplx_world"]["trans"], dtype=torch.float32, device=device)
        mean_beta = beta_world.mean(0, keepdim=True)
        with torch.no_grad():
            tpose_pelvis = model(
                global_orient=torch.eye(3, device=device)[None],
                body_pose=torch.eye(3, device=device)[None, None].expand(1, 21, 3, 3),
                betas=mean_beta, pose2rot=False,
            ).joints[0, 0]
        Q_inv = torch.as_tensor(Q.T, dtype=torch.float32, device=device)
        inverse_translation = Q_inv @ torch.as_tensor(offset, dtype=torch.float32, device=device)
        root_origin, trans_origin = transform_smpl_params(
            axis_angle_to_matrix(pose_world[:, :3]), trans_world,
            Q_inv.expand(len(local_frames), -1, -1),
            inverse_translation.expand(len(local_frames), -1), tpose_pelvis,
        )
        pose_origin = pose_world.clone()
        pose_origin[:, :3] = matrix_to_axis_angle(root_origin)
        origin_pose_np = pose_origin.detach().cpu().numpy()
        origin_trans_np = trans_origin.detach().cpu().numpy()
        vertices_batches, joints_batches = [], []
        for batch_start in range(0, len(local_frames), int(video_cfg.get("mesh_batch_size", 32))):
            sl = slice(batch_start, min(batch_start + int(video_cfg.get("mesh_batch_size", 32)), len(local_frames)))
            B = sl.stop - sl.start
            pose_rotmat = axis_angle_to_matrix(pose_origin[sl].reshape(-1, 3)).reshape(B, 55, 3, 3)
            with torch.no_grad():
                out = model(
                    global_orient=pose_rotmat[:, 0], body_pose=pose_rotmat[:, 1:22],
                    left_hand_pose=pose_rotmat[:, 25:40], right_hand_pose=pose_rotmat[:, 40:55],
                    betas=beta_world[sl], transl=trans_origin[sl],
                    jaw_pose=torch.eye(3, device=device)[None].expand(B, 3, 3),
                    leye_pose=torch.eye(3, device=device)[None].expand(B, 3, 3),
                    reye_pose=torch.eye(3, device=device)[None].expand(B, 3, 3),
                    expression=torch.zeros(B, 10, device=device), pose2rot=False,
                )
            vertices_batches.append(out.vertices.detach().cpu().numpy().astype(np.float32))
            joints_batches.append(out.body_joints[:, :25].detach().cpu().numpy().astype(np.float32))
            if sl.stop == len(local_frames) or sl.stop % 500 == 0:
                print(f"[video] mesh batches through {sl.stop}/{len(local_frames)} frames", flush=True)
        vertices_all = np.concatenate(vertices_batches)
        pred_joints_all = np.concatenate(joints_batches)

        fit_display = bool(video_cfg.get("fit_display_meshes", True))
        fitted_vertices_kept = None
        fit_report = None
        if fit_display:
            print(f"[video] Fitting only display meshes to sequence gender ({gender}); metrics remain on original joints...", flush=True)
            (fitted_vertices_kept, fitted_joints_kept, fitted_pose_kept,
             fitted_betas_kept, fit_source_rmse_max) = _fit_display_meshes(
                vertices_all[keep], origin_pose_np[kept_local], beta_world[kept_local].detach().cpu().numpy(),
                origin_trans_np[kept_local], repo, model_folder, str(gender), device,
                max_iter=int(video_cfg.get("fit_max_iter", cfg.get("fit_max_iter", 120))),
                pose_prior_weight=float(video_cfg.get("fit_pose_prior_weight", cfg.get("fit_pose_prior_weight", 1e-3))),
                shape_prior_weight=float(video_cfg.get("fit_shape_prior_weight", cfg.get("fit_shape_prior_weight", 1e-4))),
                batch_size=int(video_cfg.get("fit_batch_size", 4)),
            )
            fitted_faces = np.asarray(model.faces, dtype=np.int32)
            np.save(output_dir / f"prediction_vertices_fitted_{gender}_slam_origin.npy", fitted_vertices_kept)
            np.savez_compressed(
                output_dir / f"prediction_fitted_{gender}_video.npz",
                source_frame_indices=kept_source, timestamps=timestamps[kept_local],
                gender=np.asarray(str(gender)), vertices_path=np.asarray(f"prediction_vertices_fitted_{gender}_slam_origin.npy"),
                body_joints=fitted_joints_kept, pose_axis_angle=fitted_pose_kept,
                betas=fitted_betas_kept, transl=origin_trans_np[kept_local], faces=fitted_faces,
            )
            vertex_delta = np.linalg.norm(fitted_vertices_kept - vertices_all[keep], axis=-1)
            joint_delta = np.linalg.norm(fitted_joints_kept - pred_joints_all[keep], axis=-1)
            fit_report = {
                "target_gender": str(gender), "purpose": "Viser/display meshes only; excluded from MPJPE and PA-MPJPE",
                "method": "same regularized corresponding-vertex L-BFGS fit as image mode; global orientation and translation fixed",
                "max_iter": int(video_cfg.get("fit_max_iter", cfg.get("fit_max_iter", 120))),
                "pose_prior_weight": float(video_cfg.get("fit_pose_prior_weight", cfg.get("fit_pose_prior_weight", 1e-3))),
                "shape_prior_weight": float(video_cfg.get("fit_shape_prior_weight", cfg.get("fit_shape_prior_weight", 1e-4))),
                "source_mesh_neutral_reconstruction_rmse_max_m": float(fit_source_rmse_max),
                "fitted_vs_original_vertex_distance_mean_mm": float(vertex_delta.mean() * 1000),
                "fitted_vs_original_vertex_distance_rmse_mm": float(np.sqrt(np.mean(vertex_delta ** 2)) * 1000),
                "fitted_vs_original_body_joint_distance_mean_mm": float(joint_delta.mean() * 1000),
                "fitted_global_orientation_and_translation_fixed": True,
                "num_fitted_frames": int(len(kept_source)),
            }
            (output_dir / f"video_gender_fit_{gender}_report.json").write_text(
                json.dumps(fit_report, indent=2, allow_nan=False), encoding="utf-8")
        else:
            print("[video] Display gender fitting disabled by video.fit_display_meshes=false", flush=True)

        gt_eval = gt_body25[kept_local]
        pred_eval = pred_joints_all[keep]
        errors = np.stack([compute_mpjpe(p, g)[0] for p, g in zip(pred_eval, gt_eval)])
        pa_errors = np.stack([compute_pa_mpjpe(p, g)[0] for p, g in zip(pred_eval, gt_eval)])
        frame_mpjpe = errors.mean(axis=1)
        frame_pa = pa_errors.mean(axis=1)
        windows = compute_window_metrics(
            pred_eval, gt_eval, kept_source,
            window_frames=int(video_cfg.get("window_frames", 100)),
            initial_alignment_frames=int(video_cfg.get("initial_alignment_frames", 2)),
        )
        scored_window_frames = {frame for window in windows
                                for frame in range(window["start_frame"], window["end_frame"] + 1)}
        unscored_window_frames = [
            {"source_index": int(index), "reason": "contiguous segment shorter than initial_alignment_frames"}
            for index in kept_source if int(index) not in scored_window_frames
        ]

        # Preserve parameters for every detected PromptHMR track. This permits later
        # inspection of ID changes without rerunning inference; only the GT-associated
        # track contributes to the reported metrics and saved dense vertex sequence.
        all_track_data = {}
        all_track_frames, all_track_ids = [], []
        all_camera_pose, all_camera_rotmat, all_camera_shape, all_camera_trans = [], [], [], []
        all_canonical_pose, all_canonical_shape, all_canonical_trans = [], [], []
        for other_id, other in pipeline.results["people"].items():
            other_frames = np.asarray(other["frames"], dtype=np.int64)
            camera_params = other["smplx_cam"]
            world_params = other["smplx_world"]

            def as_cpu_array(value):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().numpy()
                return np.asarray(value)

            all_track_frames.append(source_indices[other_frames])
            all_track_ids.extend([str(other_id)] * len(other_frames))
            all_camera_pose.append(as_cpu_array(camera_params["pose"]))
            all_camera_rotmat.append(as_cpu_array(camera_params["rotmat"]))
            all_camera_shape.append(as_cpu_array(camera_params["shape"]))
            all_camera_trans.append(as_cpu_array(camera_params["trans"]))
            all_canonical_pose.append(as_cpu_array(world_params["pose"]))
            all_canonical_shape.append(as_cpu_array(world_params["shape"]))
            all_canonical_trans.append(as_cpu_array(world_params["trans"]))
        all_track_data = {
            "all_track_source_frame_indices": np.concatenate(all_track_frames),
            "all_track_ids": np.asarray(all_track_ids),
            "all_prediction_smplx_pose_camera": np.concatenate(all_camera_pose),
            "all_prediction_smplx_rotmat_camera": np.concatenate(all_camera_rotmat),
            "all_prediction_smplx_shape_camera": np.concatenate(all_camera_shape),
            "all_prediction_smplx_transl_camera": np.concatenate(all_camera_trans),
            "all_prediction_smplx_pose_canonical_world": np.concatenate(all_canonical_pose),
            "all_prediction_smplx_shape_canonical_world": np.concatenate(all_canonical_shape),
            "all_prediction_smplx_transl_canonical_world": np.concatenate(all_canonical_trans),
        }

        # Save selected-track outputs in original-frame order; large meshes use .npy to avoid
        # duplicating them in the compressed metrics archive.
        np.save(output_dir / "prediction_vertices_slam_origin.npy", vertices_all[keep])
        output_npz = output_dir / "video_evaluation.npz"
        np.savez_compressed(
            output_npz, source_frame_indices=kept_source, timestamps=timestamps[kept_local],
            all_source_frame_indices=source_indices, all_source_timestamps=timestamps,
            all_T_origin_camera=T_origin_camera,
            gt_body25_all_source_frames_slam_origin=gt_body25,
            gt_smplx_body_pose_all_source_frames=gt_params["body_pose"],
            gt_smplx_global_orient_all_source_frames=gt_params["global_orient"],
            gt_smplx_betas_all_source_frames=gt_params["betas"],
            gt_smplx_transl_all_source_frames=gt_params["transl"],
            track_source_frame_indices=source_indices[local_frames],
            track_timestamps=timestamps[local_frames],
            track_T_origin_camera=T_origin_camera[local_frames],
            track_id=np.asarray(str(track_id)), joint_names=np.asarray(BODY25_NAMES),
            prediction_body25_slam_origin=pred_eval, gt_body25_slam_origin=gt_eval,
            prediction_body25_all_track_slam_origin=pred_joints_all,
            prediction_smplx_pose_camera=raw_cam_pose, prediction_smplx_rotmat_camera=raw_cam_rotmat,
            prediction_smplx_shape_camera=raw_cam_shape, prediction_smplx_transl_camera=raw_cam_trans,
            prediction_smplx_pose_canonical_world=track["smplx_world"]["pose"],
            prediction_smplx_shape_world=track["smplx_world"]["shape"],
            prediction_smplx_transl_canonical_world=track["smplx_world"]["trans"],
            prediction_smplx_pose_slam_origin=origin_pose_np,
            prediction_smplx_shape_slam_origin=track["smplx_world"]["shape"],
            prediction_smplx_transl_slam_origin=origin_trans_np,
            gt_smplx_body_pose=gt_params["body_pose"][kept_local],
            gt_smplx_global_orient=gt_params["global_orient"][kept_local],
            gt_smplx_betas=gt_params["betas"][kept_local],
            gt_smplx_transl=gt_params["transl"][kept_local],
            T_origin_camera=T_origin_camera[kept_local], camera_intrinsics_processed=K,
            joint_error_mm=errors, frame_mpjpe_mm=frame_mpjpe, frame_pa_mpjpe_mm=frame_pa,
            visible_gt_joints=np.asarray(visible_counts)[keep],
            **all_track_data,
        )
        with (output_dir / "video_frame_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source_index", "timestamp_seconds", "track_id", "mpjpe_mm", "pa_mpjpe_mm"])
            writer.writerows(zip(kept_source, timestamps[kept_local], [str(track_id)] * len(keep), frame_mpjpe, frame_pa))
        (output_dir / "video_excluded_frames.json").write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
        (output_dir / "video_windows.json").write_text(json.dumps(windows, indent=2), encoding="utf-8")
        summary = {
            "sequence": sequence, "mode": "video", "source_frame_range": [int(source_indices[0]), int(source_indices[-1]) + 1],
            "num_source_frames": int(len(source_indices)), "num_evaluated_frames": int(len(kept_source)),
            "prompthmr_input": "MP4 decoded by official Pipeline.load_frames",
            "mp4_path": str(input_mp4), "mp4_fps": float(source_fps),
            "mp4_decoded_frame_count": int(len(pipeline.images)),
            "num_excluded_source_frames": int(len(exclusions)), "track_id": str(track_id),
            "gt_smplx_gender": str(gender), "prediction_smplx_gender": "neutral",
            "gender_fitting": "display meshes fitted to sequence gender; all metrics use original PromptHMR video joints",
            "display_mesh_gender_fitting": (fit_report if fit_report is not None else "disabled"),
            "window_metric_unscored_frames": unscored_window_frames,
            "track_association": association, "timestamp_start_seconds": float(timestamps[0]),
            "timestamp_end_seconds": float(timestamps[-1]),
            "mean_mpjpe_mm": float(frame_mpjpe.mean()), "mean_pa_mpjpe_mm": float(frame_pa.mean()),
            "wa_mpjpe_mm": aggregate_window_metric(windows, "wa_mpjpe_mm"),
            "w_mpjpe_mm": aggregate_window_metric(windows, "w_mpjpe_mm"),
            "window_frames": int(video_cfg.get("window_frames", 100)),
            "initial_alignment_frames": int(video_cfg.get("initial_alignment_frames", 2)),
            "window_alignment": "proper rigid rotation plus translation, no scale; non-overlapping source-contiguous windows",
            "coordinate_frame": "SLAM Origin", "T_origin_camera": "aria_obs_T_origin_device[i] @ aria_obs_T_device_camera",
            "camera_intrinsics_source": {"focal_px": focal, "width": rgb_w, "height": rgb_h, "processed_K": K.tolist()},
            "body25_source_indices": body25_indices.tolist(),
            "world_postprocessing": {
                "world_hps_estimation": "official PromptHMR function; EgoHTR per-frame camera poses supplied; smooth_cam and SPEC disabled",
                "post_optimization": bool(pipeline.cfg.run_post_opt),
                "post_optimization_camera_parameters": "disabled; EgoHTR camera pose held fixed, person translation refinement retained",
                "reversible_canonicalization": pipeline.results["canonicalization"],
                "inverse_applied_to_saved_mesh_and_joints": True,
                "reason": "world.py axis rotation and floor offset form one rigid transform recovered from its output camera poses",
            },
            "prompt_hmr_adjustments": [
                "Official detector/tracker, ViTPose, frame-level PromptHMR, temporal head, bbox smoothing, and contact head retained",
                "Temporal camera angular velocity computed from EgoHTR camera-to-SLAM rotations",
                "Temporal inference split only at tracking gaps; metric windows are applied after inference",
            ],
            "viewer_prediction_mesh_source": (
                f"prediction_vertices_fitted_{gender}_slam_origin.npy" if fit_display
                else "prediction_vertices_slam_origin.npy"
            ),
            "outputs": ["video_evaluation.npz", "prediction_vertices_slam_origin.npy", "video_frame_metrics.csv",
                        "video_windows.json", "video_excluded_frames.json"] + (
                            [f"prediction_vertices_fitted_{gender}_slam_origin.npy",
                             f"prediction_fitted_{gender}_video.npz",
                             f"video_gender_fit_{gender}_report.json"] if fit_display else []
                        ),
        }
        (output_dir / "video_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[video] MPJPE {summary['mean_mpjpe_mm']:.3f} mm; PA-MPJPE {summary['mean_pa_mpjpe_mm']:.3f} mm; "
              f"WA-MPJPE {summary['wa_mpjpe_mm']:.3f} mm; W-MPJPE {summary['w_mpjpe_mm']:.3f} mm", flush=True)
        print(f"[video] Results saved: {output_dir}", flush=True)
        if bool(cfg.get("viewer", True)) and not no_viewer:
            print("[video] Timeline viewer is not wired to this output yet; saved SMPL-X parameters/vertices can be loaded by Viser.", flush=True)
        return summary
    finally:
        os.chdir(previous_cwd)
