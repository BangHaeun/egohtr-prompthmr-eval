from __future__ import annotations

import numpy as np


def show_meshes(pred_vertices, gt_vertices, faces, port=8080):
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
    print(f"Viser viewer: http://localhost:{port} (Ctrl+C to stop)")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
