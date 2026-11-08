# EgoHTR–PromptHMR evaluation

This repository runs image-frame and video EgoHTR–PromptHMR evaluations using SLAM Origin as the common coordinate frame. PromptHMR remains an existing local checkout; this repository contains the extraction, transformation, scoring, and visualization pipeline.

## Pipeline

1. Extract the selected 2nd Aria RGB image, original EgoHTR SMPL-X parameters, and stored joints from `full_sequence.npz`.
2. Read the observer RGB image size and focal length from EgoHTR, construct the pinhole intrinsic matrix, and pass it explicitly to PromptHMR for Neutral SMPL-X inference.
3. Fit the Neutral PromptHMR mesh to the sequence's configured SMPL-X gender (Male for the included high-object frame) with pose and shape regularization. Save the fitted model separately.
4. Transform the PromptHMR camera pose, prediction, and fitted mesh into SLAM Origin. Keep the GT joint coordinates as stored and interpret them as SLAM Origin. The GT mesh is regenerated from the saved GT parameters and displayed in the same assumed frame.
5. Compute absolute MPJPE and PA-MPJPE from the **original PromptHMR joints before fitting** and the original EgoHTR joints, then open Viser to compare the fitted prediction mesh, GT mesh, RGB image, and 2nd Aria RGB camera.

## Coordinate frames

The single supported evaluation frame is SLAM Origin. Homogeneous transforms use the `T_target_source` convention:

```text
RGB camera -> Aria device:      aria_obs_T_device_camera
Aria device -> SLAM Origin:     aria_obs_T_origin_device[index]

T_origin_camera = aria_obs_T_origin_device[index] @ aria_obs_T_device_camera
```

Apply `T_origin_camera` to PromptHMR's original camera-frame BODY_25 joints and the fitted prediction mesh. Use the same matrix for the Viser camera pose and image frustum. EgoHTR's original `smplx_exomc_joints` are mapped to BODY_25 with PromptHMR's `smplx_to_openpose` indices and used directly as GT in this SLAM Origin hypothesis. The pipeline does not use `T_origin_exoW`, axis permutations, root subtraction, Procrustes alignment, or scale alignment.

The stored `aria_obs_T_device_camera` is treated as the 2nd Aria RGB-camera extrinsic. Local calibration comparison and RGB projection support this interpretation; the high-object sequence's own VRS calibration was not available for direct comparison, so that provenance limitation remains. EgoHTR's saved RGB focal length and image dimensions are used directly; for the included k=2150 frame they produce a 1328×1328 image, 608.4 px focal length, and principal point (664, 664).

## Install

Use the Python/CUDA environment already configured for the local PromptHMR checkout:

```bash
conda activate phmr_pt2.4
cd ~/Codes/egohtr-prompthmr-eval
python -m pip install -e .
```

PromptHMR's checkpoint, YOLO weights, and SMPL-X assets must already be installed in that checkout. The EgoHTR files must exist locally, or the dataset repository ID must be configured for the pipeline's minimal-file download path.

## Run

Set paths and frame selection in [`configs/example.yaml`](configs/example.yaml). The included config reproduces `exo-high-object_s4_xhall_01`, index 2150, timestamp 71.6666666667 s:

```bash
egohtr-phmr-eval --config configs/example.yaml
```

Use `--no-viewer` to finish after saving evaluation artifacts. `--index` or `--timestamp` can override the frame selected in the config. The pipeline selects the nearest timestamp and checks the Aria and ExoMC time grids at that index.

### Video mode

Video mode streams the selected 2nd Aria RGB frames from `full_sequence.npz` into a contiguous MP4, then passes that MP4 path to PromptHMR's official `Pipeline.load_frames()` decoder. PromptHMR's SAM2/ViTPose/image-head/video-head stages consume the decoded video frames; no JPG sequence is used as model input. The encoded frame count is checked against the EgoHTR source range before inference, and the original frame indices and timestamps are retained for GT matching. Since `full_sequence.npz` stores RGB frames rather than an MP4, the MP4 is generated locally under the run output's `prompthmr_video_work/` folder. By default it processes the full sequence; use `video.start_index` and exclusive `video.stop_index` in [`configs/example_video.yaml`](configs/example_video.yaml) to select a contiguous source range. These range options are useful for smoke tests; WA/W-MPJPE windows are applied only after inference.

```bash
egohtr-phmr-eval --config configs/example_video.yaml
# Or override the configured range from the command line:
egohtr-phmr-eval --config configs/example_video.yaml --start-index 2140 --stop-index 2160 --no-viewer
```

PromptHMR's temporal camera-angular-velocity input is derived from `aria_obs_T_origin_device[i] @ aria_obs_T_device_camera` instead of identity rotations. Its official `world_hps_estimation` stage is retained. The stage's fixed axis rotation and floor offset are a rigid canonicalization; the code recovers that transform from the before/after camera poses and applies its inverse to saved meshes and joints. Official `post_optimization` is retained by default in this canonical gravity/floor frame. Its camera pose adjustment is disabled because EgoHTR camera poses are the fixed reference; person translation refinement, reprojection, temporal, and contact terms remain enabled. PromptHMR camera smoothing and SPEC rotation are disabled because they would modify the supplied EgoHTR trajectory before canonicalization. These deviations and the recovered transform are recorded in `video_summary.json`.

The selected evaluation track is associated using overlap between projected GT BODY_25 bounds and PromptHMR track boxes. Set `video.track_id` to override association. Inference is separated at track gaps so the temporal model does not bridge missing intervals. Frames with missing selected-track detections or too few GT joints projected inside the image are recorded in `video_excluded_frames.json`.

Video mode reports absolute MPJPE and framewise PA-MPJPE, plus WA-MPJPE and W-MPJPE over non-overlapping, source-contiguous windows (defaults: 100 frames; first 2 frames for W alignment). Both window metrics use a proper rigid rotation and translation with no scale. This follows the 100-frame/first-two-versus-whole-window alignment protocol described in TRAM/WHAM; the papers do not specify a scale term in those descriptions, so this implementation explicitly uses no scale. Results are in millimeters.

Video metrics use the original PromptHMR video joints. Separately, the selected-track display meshes are fitted frame by frame to the gender in the sequence config using the same regularized corresponding-vertex L-BFGS fitting method as image mode: global orientation and translation stay fixed, while body pose and shape are optimized with priors. This fitted mesh is only for visualization and never enters MPJPE/PA-MPJPE/WA-MPJPE/W-MPJPE. Configure it under `video.fit_display_meshes` and the related fit settings. The EgoHTR GT SMPL-X parameters and joints are saved for later GT mesh regeneration.

Video outputs include:

- `video_evaluation.npz`: evaluated GT/prediction BODY_25, selected-track and GT SMPL-X parameters, camera poses, timestamps, and frame errors.
- `prediction_vertices_slam_origin.npy`: selected-track prediction mesh vertices for evaluated frames, without image or mesh alignment.
- `prediction_vertices_fitted_<gender>_slam_origin.npy` and `prediction_fitted_<gender>_video.npz`: sequence-gender fitted display meshes and parameters, stored separately from the original mesh and metrics.
- `video_gender_fit_<gender>_report.json`: fit settings and reconstruction/shape diagnostics.
- `video_frame_metrics.csv`: source-indexed MPJPE and PA-MPJPE.
- `video_windows.json`: per-window WA/W-MPJPE and rigid transforms.
- `video_excluded_frames.json`: source frame IDs and exclusion reasons.
- `video_summary.json`: aggregate metrics, track association, camera/intrinsic details, and post-processing provenance.

The current video output stores prediction meshes and the full GT parameters/joints needed to regenerate GT meshes. A time-synchronized Viser timeline is not yet connected to video mode.

## Evaluation definition

For each BODY_25 joint `j`, the pipeline calculates:

```text
error_j_mm = 1000 * || T_origin_camera * prediction_joint_camera[j]
                         - original_gt_joint_origin[j] ||_2
MPJPE_mm = mean(error_j_mm for all 25 joints)
```

PA-MPJPE applies a single similarity Procrustes alignment to the prediction BODY_25 joints before measuring the same 25 Euclidean errors. The alignment allows global rotation, translation, and uniform scale, and disallows reflection. It is reported alongside absolute MPJPE; it does not replace or modify the absolute metric.

Only the original PromptHMR `body_joints` and EgoHTR's original stored joints contribute to the metrics. The gender-fitted mesh and its joints are excluded. Absolute MPJPE uses the unaligned SLAM Origin coordinates; PA-MPJPE aligns a copy of the prediction joints to GT and does not change saved joints, meshes, or the absolute MPJPE result.

## Outputs

The configured output directory contains:

- `rgb.png`: selected 2nd Aria RGB frame.
- `eval_sample.npz`: original GT joints and SMPL-X parameters for the selected frame.
- `gt_body25_mesh.npz`: direct GT BODY_25 mapping plus regenerated GT mesh and faces.
- `prediction.npz` and `output.pt`: original Neutral PromptHMR prediction, including the model's processed camera intrinsics, mesh, joints, and parameters. The original full-resolution Aria RGB intrinsics are saved in `evaluation.npz` and `manifest.json`.
- `gender_fit/fitted_male_prediction.npz` and `gender_fit/fit_report.json`: fitted model parameters and vertex/joint reconstruction diagnostics.
- `evaluation.npz`: original prediction joints, direct GT joints, transformed fitted mesh, camera-to-Origin transform, and per-joint/mean MPJPE and PA-MPJPE.
- `mpjpe_body25.csv` and `mpjpe_body25.txt`: per-joint and mean absolute MPJPE in millimeters; the text summary also reports PA-MPJPE.
- `pa_mpjpe_body25.csv`: per-joint Procrustes-aligned errors in millimeters.
- `manifest.json`: selected frame, coordinate assumptions, transform matrices, fitting provenance, and metric definition.

Viser shows the fitted prediction mesh in red, regenerated GT mesh in blue, the SLAM Origin axes, the 2nd Aria RGB camera pose, and the RGB image frustum. Camera axes follow PromptHMR/OpenCV (+X right, +Y down, +Z forward); the scene uses +Z as up.

## Render a qualitative comparison figure

After an evaluation run has produced its artifacts, render a publication-style PNG without rerunning inference:

```bash
egohtr-phmr-figure \
  --evaluation-dir runs/exo-high-object-k3700-qualitative \
  --output runs/exo-high-object-k3700-qualitative/qualitative_comparison.png
```

The figure contains two SLAM Origin side views with the 2nd Aria camera frustum, a calibrated RGB mesh projection, and matched GT-defined front/right/left/rear pose views. The lower GT and fitted-prediction panels use the same GT-pelvis camera target, camera directions, orthographic projection, and scale; mesh coordinates are left untouched. It uses the fitted Male mesh for qualitative display and evaluation metrics from the original, unfitted prediction joints.

To crop horizontal whitespace from the already-rendered lower mesh panels while keeping the upper panels unchanged, run:

```bash
egohtr-phmr-crop-figure \
  --figure runs/exo-high-object-k3700-qualitative/qualitative_comparison.png \
  --rgb runs/exo-high-object-k3700-qualitative/rgb.png
```

This is an image-only post-process. It preserves the GT and PromptHMR crop aspect ratios and common native pixel scale, without rerendering or transforming the meshes.

## Fit an existing prediction separately

To fit a previously saved Neutral PromptHMR prediction without rerunning the evaluation, use:

```bash
egohtr-phmr-fit-gender \
  --prediction runs/exo-high-object-k2150-final/prediction.npz \
  --output-dir runs/exo-high-object-k2150-final/gender_fit \
  --prompthmr-repo ~/Codes/PromptHMR \
  --smplx-model-folder ~/Codes/PromptHMR/data/body_models/smplx \
  --gender male --device cuda --pose-prior-weight 1e-3
```

The fitter first checks that the installed Neutral model reproduces the saved source mesh. It fixes global orientation and translation and fits body pose and shape with priors toward their original values. Source prediction and GT files are not modified.
