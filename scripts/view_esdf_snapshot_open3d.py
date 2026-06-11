#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import open3d as o3d
except Exception as e:  # pragma: no cover
    raise ImportError("Open3D is required for this viewer. Install `open3d`.") from e

try:
    from skimage.measure import marching_cubes
except Exception as e:  # pragma: no cover
    raise ImportError("scikit-image is required for marching cubes. Install `scikit-image`.") from e

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mast3r_slam.esdf_snapshot import load_esdf_snapshot


def _as_R_t(T_WC: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    M = np.asarray(T_WC, dtype=np.float64).reshape(4, 4)
    A = M[:3, :3]
    detA = float(np.linalg.det(A))
    s = float(np.cbrt(max(1e-12, abs(detA))))
    R = A / s
    t = M[:3, 3]
    return R, t


def _frustum_lineset(T_WC: np.ndarray, fov_deg: float = 60.0, scale: float = 0.5) -> o3d.geometry.LineSet:
    R_wc, t_wc = _as_R_t(T_WC)
    f = float(np.deg2rad(fov_deg))
    z = float(scale)
    x = z * np.tan(0.5 * f)
    y = x
    pts_c = np.array(
        [[0.0, 0.0, 0.0], [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]],
        dtype=np.float32,
    )
    pts_w = pts_c @ R_wc.T + t_wc.reshape(1, 3)
    lines = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_w.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.75, 0.0]] * len(lines), dtype=np.float64))
    return ls


def _bbox_lineset_world(*, origin_w: np.ndarray, dims: Sequence[int], voxel_m: float) -> o3d.geometry.LineSet:
    origin_w = np.asarray(origin_w, dtype=np.float64).reshape(3)
    nx, ny, nz = (int(dims[0]), int(dims[1]), int(dims[2]))
    v = float(voxel_m)
    max_w = origin_w + np.array([(nx - 1) * v, (ny - 1) * v, (nz - 1) * v], dtype=np.float64)
    corners = np.array(
        [
            [origin_w[0], origin_w[1], origin_w[2]],
            [max_w[0], origin_w[1], origin_w[2]],
            [max_w[0], max_w[1], origin_w[2]],
            [origin_w[0], max_w[1], origin_w[2]],
            [origin_w[0], origin_w[1], max_w[2]],
            [max_w[0], origin_w[1], max_w[2]],
            [max_w[0], max_w[1], max_w[2]],
            [origin_w[0], max_w[1], max_w[2]],
        ],
        dtype=np.float64,
    )
    lines = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]],
        dtype=np.int32,
    )
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.array([[0.15, 0.15, 0.15]] * len(lines), dtype=np.float64))
    return ls


def _set_view_from_pose(vis: o3d.visualization.Visualizer, T_WC: np.ndarray, look_ahead: float = 1.0) -> None:
    R_wc, t_wc = _as_R_t(T_WC)
    forward = (R_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float64)).reshape(3)
    up = (R_wc @ np.array([0.0, -1.0, 0.0], dtype=np.float64)).reshape(3)
    f_n = forward / max(1e-12, np.linalg.norm(forward))
    u_n = up / max(1e-12, np.linalg.norm(up))
    lookat = t_wc + float(look_ahead) * f_n
    vc = vis.get_view_control()
    vc.set_front((-f_n).tolist())
    vc.set_up(u_n.tolist())
    vc.set_lookat(lookat.tolist())
    try:
        vc.set_zoom(0.7)
    except Exception:
        pass


def _label_to_color(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    colors = np.zeros((labels.shape[0], 3), dtype=np.float64)
    valid = labels >= 0
    if not np.any(valid):
        colors[:] = np.array([0.7, 0.7, 0.72], dtype=np.float64)
        return colors
    vals = labels[valid]
    colors_valid = np.stack(
        [
            ((vals * 37) % 255) / 255.0,
            ((vals * 91 + 53) % 255) / 255.0,
            ((vals * 17 + 127) % 255) / 255.0,
        ],
        axis=1,
    )
    colors[:] = np.array([0.7, 0.7, 0.72], dtype=np.float64)
    colors[valid] = colors_valid
    return colors


def _build_mesh(snapshot, mesh_style: str) -> o3d.geometry.TriangleMesh:
    verts, faces, normals, _ = marching_cubes(snapshot.esdf, level=0.0, spacing=(1.0, 1.0, 1.0))
    verts_w = np.asarray(snapshot.origin_w, dtype=np.float32).reshape(1, 3) + verts.astype(np.float32) * float(snapshot.voxel_m)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_w.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

    if mesh_style == "semantic":
        ijk = np.rint(verts).astype(np.int32)
        ijk[:, 0] = np.clip(ijk[:, 0], 0, snapshot.sem_label.shape[0] - 1)
        ijk[:, 1] = np.clip(ijk[:, 1], 0, snapshot.sem_label.shape[1] - 1)
        ijk[:, 2] = np.clip(ijk[:, 2], 0, snapshot.sem_label.shape[2] - 1)
        labels = snapshot.sem_label[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(_label_to_color(labels))
    else:
        mesh.paint_uniform_color([0.72, 0.72, 0.75])

    mesh.compute_vertex_normals()
    return mesh


def main() -> int:
    p = argparse.ArgumentParser(description="Offline Open3D viewer for a saved MASt3R ESDF snapshot (.npz).")
    p.add_argument("--snapshot", type=Path, required=True, help="Path to final_esdf_snapshot.npz")
    p.add_argument("--show-frustum", action="store_true", help="Show current camera frustum from snapshot pose.")
    p.add_argument("--show-bounds", action="store_true", help="Show ESDF volume bounds.")
    p.add_argument("--mesh-style", choices=["solid", "semantic"], default="solid", help="Surface coloring mode.")
    p.add_argument("--fov-deg", type=float, default=60.0, help="Frustum horizontal FOV in degrees.")
    p.add_argument("--frustum-scale", type=float, default=0.5, help="Frustum size in meters.")
    args = p.parse_args()

    snapshot = load_esdf_snapshot(args.snapshot)
    mesh = _build_mesh(snapshot, mesh_style=args.mesh_style)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"ESDF Snapshot: {args.snapshot.name}", width=1600, height=1000)
    vis.add_geometry(mesh)

    if args.show_bounds:
        vis.add_geometry(
            _bbox_lineset_world(origin_w=snapshot.origin_w, dims=snapshot.dims, voxel_m=float(snapshot.voxel_m))
        )
    if args.show_frustum:
        vis.add_geometry(_frustum_lineset(snapshot.curr_T_WC, fov_deg=float(args.fov_deg), scale=float(args.frustum_scale)))

    ro = vis.get_render_option()
    ro.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    ro.light_on = True
    ro.mesh_show_back_face = True
    _set_view_from_pose(vis, snapshot.curr_T_WC)

    print(
        f"snapshot={args.snapshot} frame_id={snapshot.frame_id} "
        f"dims={tuple(int(x) for x in snapshot.dims)} voxel_m={float(snapshot.voxel_m):.4f}"
    )
    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
