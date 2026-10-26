from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast


def run_prompthmr(image_path, repo, model_folder, yolo_weights, smplx_model_folder,
                  output_dir, device="cuda", detect_conf=0.3, image_size=896):
    repo = Path(repo).expanduser().resolve()
    sys.path.insert(0, str(repo))
    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        from ultralytics import YOLO
        from data_config import SMPLX_PATH
        from prompt_hmr import load_model_from_folder
        from prompt_hmr.models.inference import prepare_batch
        from prompt_hmr.smpl_family import SMPLX

        actual_smplx = str(Path(smplx_model_folder).expanduser()) if smplx_model_folder else SMPLX_PATH
        smplx = SMPLX(actual_smplx).to(device).eval()
        yolo = YOLO(str((repo / yolo_weights).resolve()) if not Path(yolo_weights).is_absolute() else yolo_weights)
        phmr = load_model_from_folder(str((repo / model_folder).resolve()) if not Path(model_folder).is_absolute() else model_folder)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image {image_path}")
        image_rgb = image_bgr[:, :, ::-1].copy()
        detections = yolo(str(image_path), verbose=False, conf=detect_conf, classes=0)
        boxes = detections[0].boxes.data.cpu()
        if len(boxes) == 0:
            raise RuntimeError("YOLO did not detect a person in the selected RGB frame")
        batch = prepare_batch([{"image_cv": image_rgb, "boxes": boxes, "text": None, "masks": None}],
                              img_size=image_size, interaction=False)
        with torch.no_grad(), autocast("cuda", enabled=str(device).startswith("cuda")):
            output = phmr(batch, use_mean_hands=True)[0]
        keys = ("pose", "betas", "transl", "rotmat", "vertices", "body_joints", "cam_int")
        saved = {k: output[k].detach().cpu() for k in keys}
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        torch.save(saved, outdir / "output.pt")
        np.savez_compressed(outdir / "prediction.npz", **{k: v.float().numpy() for k, v in saved.items()}, faces=np.asarray(smplx.faces, np.int32))
        return {k: v.numpy() for k, v in saved.items()}, np.asarray(smplx.faces, np.int32), image_rgb
    finally:
        os.chdir(old_cwd)
