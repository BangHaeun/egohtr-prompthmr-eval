"""Render a reproducible qualitative comparison from one pipeline run.

The source meshes and their coordinates are never edited. For the pose/shape
views, each mesh is viewed with the same orthographic camera settings relative
to its own pelvis so that the body remains in frame; the body-aligned view
directions are derived once from the GT BODY_25 and reused for both meshes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


GT_BLUE = "#2864C5"
PRED_RED = "#D6423A"
CAMERA_GOLD = "#E69A19"
VIEWS = (("Front", "front"), ("Right", "right"), ("Left", "left"), ("Rear", "rear"))


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return fallback.astype(np.float64, copy=True)
    return np.asarray(vector, dtype=np.float64) / norm


def _gt_body_directions(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return GT-derived right, up, and forward directions in SLAM Origin."""
    # BODY_25: R/L shoulder = 2/5, nose/neck = 0/1, R/L hip = 9/12.
    up = _unit(joints[[2, 5]].mean(axis=0) - joints[[9, 12]].mean(axis=0), np.array([0, 0, 1.]))
    right = joints[2] - joints[5]
    right = right - np.dot(right, up) * up
    right = _unit(right, np.array([1., 0, 0]))
    forward = _unit(np.cross(right, up), np.array([0., 1, 0]))
    # Orient the torso normal toward the face when the nose direction is usable.
    face = joints[0] - joints[1]
    face = face - np.dot(face, up) * up
    if np.linalg.norm(face) > 1e-6 and np.dot(forward, face) < 0:
        forward = -forward
        right = -right
    return right, up, forward


def _azimuth(direction: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def _mesh_collection(ax, vertices, faces, color, alpha=1.0, light_direction=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import to_rgb

    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    base = np.asarray(to_rgb(color), dtype=np.float64)
    if light_direction is None:
        shade = np.ones(len(triangles), dtype=np.float64)
    else:
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        norms = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(norms[:, None], 1e-12)
        light = _unit(np.asarray(light_direction), np.array([0., 0., 1.]))
        shade = 0.48 + 0.52 * np.maximum(0.0, normals @ light)
    colors = np.concatenate((np.clip(base[None, :] * shade[:, None], 0, 1),
                             np.full((len(triangles), 1), alpha)), axis=1)
    collection = Poly3DCollection(triangles, facecolors=colors, edgecolors="none", linewidths=0,
                                  zsort="average")
    ax.add_collection3d(collection)
    return collection


def _draw_frustum(ax, T_camera_to_origin, K, image_shape, depth=0.7):
    height, width = image_shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pixels = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)
    corners_camera = np.column_stack(((pixels[:, 0] - cx) * depth / fx,
                                      (pixels[:, 1] - cy) * depth / fy,
                                      np.full(4, depth)))
    R = T_camera_to_origin[:3, :3]
    t = T_camera_to_origin[:3, 3]
    corners_world = corners_camera @ R.T + t
    for i in range(4):
        a, b = corners_world[i], corners_world[(i + 1) % 4]
        ax.plot([t[0], a[0]], [t[1], a[1]], [t[2], a[2]], color=CAMERA_GOLD, lw=1.4, alpha=.9)
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=CAMERA_GOLD, lw=1.4, alpha=.9)
    forward = R[:, 2]
    ax.scatter(*t, color=CAMERA_GOLD, s=38, marker="o", depthshade=False, label="2nd Aria RGB camera")
    ax.quiver(*t, *forward, length=.35, color=CAMERA_GOLD, linewidth=1.6, arrow_length_ratio=.2)
    return np.vstack((corners_world, t[None, :]))


def _equal_limits_3d(ax, points, pad=0.04):
    points = np.asarray(points, dtype=np.float64)
    low = np.nanmin(points, axis=0)
    high = np.nanmax(points, axis=0)
    center = (low + high) / 2
    span = max(float(np.max(high - low)), .5) * (1 + pad)
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
    ax.set_box_aspect((1, 1, 1))


def _project_mesh_overlay(image, meshes, faces, T_camera_to_origin, K):
    """Painter-style translucent triangle overlay using the supplied RGB K."""
    from PIL import Image, ImageDraw

    base = Image.fromarray(np.asarray(image, dtype=np.uint8).copy()).convert("RGBA")
    width, height = base.size
    R = T_camera_to_origin[:3, :3]
    t = T_camera_to_origin[:3, 3]
    triangles_to_draw = []
    for vertices, color in meshes:
        camera = (np.asarray(vertices, dtype=np.float64) - t) @ R
        z = camera[:, 2]
        uv = np.empty((len(camera), 2), dtype=np.float64)
        uv[:, 0] = K[0, 0] * camera[:, 0] / np.maximum(z, 1e-12) + K[0, 2]
        uv[:, 1] = K[1, 1] * camera[:, 1] / np.maximum(z, 1e-12) + K[1, 2]
        for face in np.asarray(faces, dtype=np.int64):
            depths = z[face]
            if np.any(depths <= .03) or not np.all(np.isfinite(uv[face])):
                continue
            pts = uv[face]
            if (pts[:, 0].max() < -width or pts[:, 0].min() > 2 * width or
                    pts[:, 1].max() < -height or pts[:, 1].min() > 2 * height):
                continue
            triangles_to_draw.append((float(depths.mean()), pts, color))
    # Far-to-near painter order approximates the camera depth buffer.
    triangles_to_draw.sort(key=lambda item: item[0], reverse=True)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(layer, "RGBA")
    from matplotlib.colors import to_rgb
    for _, points, color in triangles_to_draw:
        rgb = tuple(int(c * 255) for c in to_rgb(color))
        polygon = [(int(round(x)), int(round(y))) for x, y in points]
        drawer.polygon(polygon, fill=(*rgb, 105))
    return Image.alpha_composite(base, layer).convert("RGB")


def render_qualitative_figure(evaluation_dir: str | Path, output_path: str | Path,
                              dpi: int = 180) -> Path:
    """Render location, RGB projection, and matched-view pose/shape panels."""
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    evaluation_dir = Path(evaluation_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(evaluation_dir / "evaluation.npz", allow_pickle=False) as data:
        index = int(data["index"])
        timestamp = float(data["timestamp"])
        gt_vertices = np.asarray(data["gt_vertices_origin"], dtype=np.float32)
        pred_vertices = np.asarray(data["prediction_vertices_fitted_origin"], dtype=np.float32)
        gt_joints = np.asarray(data["gt_body_joints_origin"], dtype=np.float32)
        faces = np.asarray(data["faces"], dtype=np.int32)
        T = np.asarray(data["T_camera_to_origin"], dtype=np.float64)
        K = np.asarray(data["camera_intrinsics"], dtype=np.float64)
        mean_mpjpe_mm = float(data["mean_mpjpe_mm"])
        mean_pa_mpjpe_mm = (float(data["mean_pa_mpjpe_mm"])
                            if "mean_pa_mpjpe_mm" in data else None)
    bgr = cv2.imread(str(evaluation_dir / "rgb.png"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(evaluation_dir / "rgb.png")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with np.load(evaluation_dir / "gender_fit" / "fitted_male_prediction.npz", allow_pickle=False) as fitted:
        fitted_faces = np.asarray(fitted["faces"], dtype=np.int32)
    if gt_vertices.shape != pred_vertices.shape or not np.array_equal(faces, fitted_faces):
        raise ValueError("GT and fitted prediction mesh topology must match")

    right, _up, forward = _gt_body_directions(gt_joints)
    scene_points = np.vstack((gt_vertices, pred_vertices))
    frustum_points = _draw_frustum_points(T, K, image.shape, depth=.7)
    all_points = np.vstack((scene_points, frustum_points))
    overlay = _project_mesh_overlay(image, ((gt_vertices, GT_BLUE), (pred_vertices, PRED_RED)),
                                    faces, T, K)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    fig = plt.figure(figsize=(18, 18), constrained_layout=False, facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=(.88, 1.52), left=.045, right=.985,
                             top=.91, bottom=.035, hspace=.15)
    top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=.12)
    lower = GridSpecFromSubplotSpec(4, 3, subplot_spec=outer[1],
                                    width_ratios=(1.05, 1, 1), wspace=.1, hspace=.13)
    ax_right = fig.add_subplot(top[0, 0], projection="3d")
    ax_left = fig.add_subplot(top[0, 1], projection="3d")
    ax_overlay = fig.add_subplot(top[0, 2])

    for ax, direction, title in ((ax_right, right, "Right-side location"),
                                 (ax_left, -right, "Left-side location")):
        light = _unit(direction + np.array([0., 0., .7]), np.array([1., 0, 1.]))
        _mesh_collection(ax, gt_vertices, faces, GT_BLUE, alpha=.83, light_direction=light)
        _mesh_collection(ax, pred_vertices, faces, PRED_RED, alpha=.83, light_direction=light)
        frustum_corners = _draw_frustum(ax, T, K, image.shape, depth=.7)
        _equal_limits_3d(ax, all_points, pad=.08)
        ax.view_init(elev=7, azim=_azimuth(direction), roll=0)
        ax.set_proj_type("ortho")
        ax.set_title(title, pad=1, loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.grid(False)
        ax.set_facecolor("white")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("#dddddd")
        ax.tick_params(labelsize=6, pad=0)

    ax_overlay.imshow(overlay)
    ax_overlay.set_title("RGB mesh projection", loc="left", pad=4)
    ax_overlay.axis("off")
    fig.suptitle(f"EgoHTR × PromptHMR qualitative comparison  |  k={index}, t={timestamp:.3f} s",
                 x=.045, y=.975, ha="left", fontsize=17, fontweight="bold")
    handles = [Line2D([0], [0], color=GT_BLUE, lw=5, label="EgoHTR GT (Male SMPL-X)"),
               Line2D([0], [0], color=PRED_RED, lw=5, label="PromptHMR fitted (Male SMPL-X)"),
               Line2D([0], [0], color=CAMERA_GOLD, marker="o", lw=1.5,
                      label="2nd Aria RGB camera + frustum")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .945),
               ncol=3, frameon=False, fontsize=9)
    fig.text(.045, .645, "3D LOCATION  ·  SLAM ORIGIN (m)", fontsize=10, fontweight="bold", color="#333333")
    fig.text(.045, .615, "POSE & SHAPE  ·  GT-defined front / right / left / rear views",
             fontsize=10, fontweight="bold", color="#333333")

    rgb_ax = fig.add_subplot(lower[:, 0])
    rgb_ax.imshow(image)
    rgb_ax.set_title("Input 2nd Aria RGB", loc="left", pad=5)
    rgb_ax.axis("off")
    fig.text(.385, .59, "EgoHTR GT · Male SMPL-X", ha="center", fontsize=10, fontweight="bold", color=GT_BLUE)
    fig.text(.72, .59, "PromptHMR fit · Male SMPL-X", ha="center", fontsize=10, fontweight="bold", color=PRED_RED)

    # Pose views share the exact same GT-pelvis target, orientation, projection,
    # and limits for both meshes. Neither mesh is recentered or otherwise edited.
    view_vectors = {"front": forward, "right": right, "left": -right, "rear": -forward}
    gt_root = gt_joints[8]
    vertical_span = float(np.ptp(np.concatenate((gt_vertices[:, 2], pred_vertices[:, 2]))))
    half_z = max(vertical_span * .54, .9)
    half_xy = max(vertical_span * .43, .68)
    for row, (view_label, key) in enumerate(VIEWS):
        for col, (vertices, color) in enumerate(
                ((gt_vertices, GT_BLUE), (pred_vertices, PRED_RED)), start=1):
            ax = fig.add_subplot(lower[row, col], projection="3d")
            direction = view_vectors[key]
            light = _unit(direction + np.array([0., 0., .6]), np.array([0., 0., 1.]))
            _mesh_collection(ax, vertices, faces, color, alpha=1., light_direction=light)
            ax.set_xlim(gt_root[0] - half_xy, gt_root[0] + half_xy)
            ax.set_ylim(gt_root[1] - half_xy, gt_root[1] + half_xy)
            ax.set_zlim(gt_root[2] - half_z, gt_root[2] + half_z)
            # Use metric-equal scene units. A 2x vertical box aspect made the
            # rasterized person unnaturally tall/narrow before whitespace crop.
            ax.set_box_aspect((1, 1, 1))
            ax.set_proj_type("ortho")
            ax.view_init(elev=0, azim=_azimuth(direction), roll=0)
            ax.set_axis_off()
            if col == 1:
                ax.set_title(view_label, loc="left", fontsize=9, pad=0)

    metric_text = f"MPJPE: {mean_mpjpe_mm:.1f} mm"
    if mean_pa_mpjpe_mm is not None:
        metric_text += f"  |  PA-MPJPE: {mean_pa_mpjpe_mm:.1f} mm"
    fig.text(.965, .012, metric_text,
             ha="right", fontsize=8, color="#555555")
    fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output_path


def _draw_frustum_points(T_camera_to_origin, K, image_shape, depth=.7):
    h, w = image_shape[:2]
    pix = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    c = np.column_stack(((pix[:, 0] - K[0, 2]) * depth / K[0, 0],
                         (pix[:, 1] - K[1, 2]) * depth / K[1, 1], np.full(4, depth)))
    return np.vstack((c @ T_camera_to_origin[:3, :3].T + T_camera_to_origin[:3, 3],
                      T_camera_to_origin[:3, 3]))


def main():
    parser = argparse.ArgumentParser(description="Render a qualitative EgoHTR–PromptHMR comparison figure")
    parser.add_argument("--evaluation-dir", required=True, help="Output directory created by egohtr-phmr-eval")
    parser.add_argument("--output", help="PNG output path; defaults to <evaluation-dir>/qualitative_comparison.png")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    evaluation_dir = Path(args.evaluation_dir).expanduser()
    output_path = Path(args.output).expanduser() if args.output else evaluation_dir / "qualitative_comparison.png"
    print(f"Saved: {render_qualitative_figure(evaluation_dir, output_path, args.dpi)}")


if __name__ == "__main__":
    main()
