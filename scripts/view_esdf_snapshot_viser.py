#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
import viser.transforms as tf

try:
    from skimage.measure import marching_cubes
except Exception as e:  # pragma: no cover
    raise ImportError("scikit-image is required for marching cubes.") from e

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mast3r_slam.esdf_snapshot import load_esdf_snapshot


def _label_to_color(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    valid = labels >= 0
    colors[:] = np.array([180, 180, 184], dtype=np.uint8)
    if np.any(valid):
        vals = labels[valid]
        colors[valid] = np.stack(
            [
                (vals * 37) % 255,
                (vals * 91 + 53) % 255,
                (vals * 17 + 127) % 255,
            ],
            axis=1,
        ).astype(np.uint8)
    return colors


def _extract_surface(snapshot) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    verts, faces, _normals, _ = marching_cubes(
        snapshot.esdf,
        level=0.0,
        spacing=(float(snapshot.voxel_m), float(snapshot.voxel_m), float(snapshot.voxel_m)),
    )
    verts = verts.astype(np.float32) + np.asarray(snapshot.origin_w, dtype=np.float32).reshape(1, 3)
    faces = faces.astype(np.int32)

    ijk = np.rint(
        (verts - np.asarray(snapshot.origin_w, dtype=np.float32).reshape(1, 3)) / float(snapshot.voxel_m)
    ).astype(np.int32)
    ijk[:, 0] = np.clip(ijk[:, 0], 0, snapshot.sem_label.shape[0] - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, snapshot.sem_label.shape[1] - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, snapshot.sem_label.shape[2] - 1)
    labels = snapshot.sem_label[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    colors = _label_to_color(labels)
    return verts, faces, colors


def _bbox_corners(origin_w: np.ndarray, dims: np.ndarray, voxel_m: float) -> np.ndarray:
    origin_w = np.asarray(origin_w, dtype=np.float32).reshape(3)
    dims = np.asarray(dims, dtype=np.int32).reshape(3)
    max_w = origin_w + (dims.astype(np.float32) - 1.0) * float(voxel_m)
    return np.array(
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
        dtype=np.float32,
    )


def _camera_pose_parts(T_WC: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T_WC = np.asarray(T_WC, dtype=np.float64).reshape(4, 4)
    R = T_WC[:3, :3]
    t = T_WC[:3, 3]
    wxyz = tf.SO3.from_matrix(R).wxyz
    return wxyz.astype(np.float32), t.astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description="Visualize a MASt3R ESDF snapshot in Viser.")
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=18090)
    p.add_argument("--mesh-style", choices=["solid", "semantic"], default="semantic")
    p.add_argument("--mesh-opacity", type=float, default=0.9)
    p.add_argument("--show-bounds", action="store_true")
    p.add_argument("--show-frustum", action="store_true")
    p.add_argument("--frustum-scale", type=float, default=0.25)
    p.add_argument("--fov-deg", type=float, default=60.0)
    p.add_argument("--no-loop", action="store_true")
    args = p.parse_args()

    snapshot = load_esdf_snapshot(args.snapshot)
    verts, faces, colors = _extract_surface(snapshot)

    server = viser.ViserServer(host=args.host, port=int(args.port))
    print(f"Viser running at http://{args.host}:{args.port}")
    print(
        f"snapshot={args.snapshot} frame_id={snapshot.frame_id} "
        f"dims={tuple(int(x) for x in snapshot.dims)} voxel_m={float(snapshot.voxel_m):.4f}"
    )

    if args.mesh_style == "semantic":
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual.vertex_colors = colors
        handle = server.scene.add_mesh_trimesh("/esdf_surface", mesh=mesh, wxyz=tf.SO3.identity().wxyz)
        try:
            handle.opacity = float(args.mesh_opacity)
        except Exception:
            pass
    else:
        server.scene.add_mesh_simple(
            "/esdf_surface",
            vertices=verts.astype(np.float32),
            faces=faces.astype(np.int32),
            color=(160, 190, 220),
            opacity=float(args.mesh_opacity),
            wxyz=tf.SO3.identity().wxyz,
        )

    if args.show_bounds:
        corners = _bbox_corners(snapshot.origin_w, snapshot.dims, float(snapshot.voxel_m))
        edges = np.array(
            [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
            ],
            dtype=np.int32,
        )
        for i, (a, b) in enumerate(edges):
            server.scene.add_line_segments(
                f"/bounds/{i}",
                points=np.stack([corners[a], corners[b]], axis=0)[None, ...],
                colors=np.array([[[40, 40, 40], [40, 40, 40]]], dtype=np.uint8),
                line_width=2.0,
            )

    if args.show_frustum:
        wxyz, position = _camera_pose_parts(snapshot.curr_T_WC)
        server.scene.add_camera_frustum(
            "/camera",
            fov=float(np.deg2rad(args.fov_deg)),
            aspect=1.0,
            scale=float(args.frustum_scale),
            color=(40, 200, 40),
            wxyz=wxyz,
            position=position,
        )

    if args.no_loop:
        return 0

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    raise SystemExit(main())
