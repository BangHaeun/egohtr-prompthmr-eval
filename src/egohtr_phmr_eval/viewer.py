from __future__ import annotations

import numpy as np


def show_meshes(
    pred_vertices,
    gt_vertices,
    faces,
    port=8080,
    T_camera_to_exomc=None,
    rgb_image=None,
    camera_intrinsics=None,
    network_input_size=896,
):
    import viser
    from scipy.spatial.transform import Rotation

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = True
    server.scene.add_mesh_simple("/prediction", vertices=np.asarray(pred_vertices), faces=faces,
                                 color=(230, 80, 60), flat_shading=False)
    server.scene.add_mesh_simple("/ground_truth", vertices=np.asarray(gt_vertices), faces=faces,
                                 color=(60, 130, 235), flat_shading=False)
    server.scene.add_frame("/exomc_origin", wxyz=(1, 0, 0, 0), position=(0, 0, 0), axes_length=0.4)

    if T_camera_to_exomc is not None:
        T_world_cv = np.asarray(T_camera_to_exomc, dtype=np.float64)
        if T_world_cv.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 camera-to-ExoMC transform, got {T_world_cv.shape}")

        # Viser camera frustums use the same OpenCV convention as PromptHMR:
        # +X right, +Y down, +Z forward. Do not add an axis flip here.
        quat_xyzw = Rotation.from_matrix(T_world_cv[:3, :3]).as_quat()
        quat_wxyz = np.concatenate((quat_xyzw[3:], quat_xyzw[:3]))
        camera_position = T_world_cv[:3, 3]

        server.scene.add_frame(
            "/aria_2nd_rgb_camera_pose",
            wxyz=quat_wxyz,
            position=camera_position,
            show_axes=True,
            axes_length=0.35,
            axes_radius=0.015,
        )

        if rgb_image is not None and camera_intrinsics is not None:
            image = np.asarray(rgb_image)
            K = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
            height, width = image.shape[:2]
            resize_scale = float(network_input_size) / max(width, height)
            fx = K[0, 0] / resize_scale
            fy = K[1, 1] / resize_scale
            fov_y = 2.0 * np.arctan(height / (2.0 * fy))
            server.scene.add_camera_frustum(
                "/aria_2nd_rgb_camera_rgb",
                fov=float(fov_y),
                aspect=float(width / height),
                scale=0.7,
                line_width=2.0,
                color=(255, 170, 35),
                image=image,
                wxyz=quat_wxyz,
                position=camera_position,
            )

    print(f"Viser viewer: http://localhost:{port} (Ctrl+C to stop)")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
