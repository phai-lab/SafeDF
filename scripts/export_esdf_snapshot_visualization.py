#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EFFICIENTVIT_ROOT = REPO_ROOT / "efficientvit"
if str(EFFICIENTVIT_ROOT) not in sys.path:
    sys.path.append(str(EFFICIENTVIT_ROOT))

from applications.efficientvit_seg.eval_efficientvit_seg_model import ADE20KDataset  # type: ignore
from mast3r_slam.esdf_snapshot import load_esdf_snapshot


def _palette() -> tuple[list[str], np.ndarray, dict[int, int]]:
    names = list(ADE20KDataset.classes)
    colors = np.asarray(ADE20KDataset.class_colors, dtype=np.uint8)
    color_to_id: dict[int, int] = {}
    for idx, c in enumerate(colors.tolist()):
        code = (int(c[0]) << 16) | (int(c[1]) << 8) | int(c[2])
        color_to_id.setdefault(code, idx)
    return names, colors, color_to_id


def _decode_sem_ids(labels: np.ndarray, colors: np.ndarray, color_to_id: dict[int, int]) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return labels
    if int(labels.max(initial=0)) < len(colors):
        return labels
    out = np.full(labels.shape, -1, dtype=np.int64)
    for code, class_id in color_to_id.items():
        mask = (labels == int(code)) & (out < 0)
        if np.any(mask):
            out[mask] = int(class_id)
    return np.where(out >= 0, out, labels)


def _class_color(class_id: int, colors: np.ndarray) -> np.ndarray:
    if 0 <= int(class_id) < len(colors):
        return colors[int(class_id)].astype(np.float32) / 255.0
    x = int(class_id)
    return np.asarray(
        [((x * 37 + 17) & 255), ((x * 67 + 29) & 255), ((x * 97 + 43) & 255)],
        dtype=np.float32,
    ) / 255.0


def _class_name(class_id: int, names: list[str]) -> str:
    if 0 <= int(class_id) < len(names):
        return names[int(class_id)]
    return f"unknown-{int(class_id)}"


def _extract_mesh(snapshot):
    verts_ijk, faces, _normals, _ = marching_cubes(
        snapshot.esdf,
        level=0.0,
        spacing=(1.0, 1.0, 1.0),
    )
    verts_world = np.asarray(snapshot.origin_w, dtype=np.float32).reshape(1, 3) + verts_ijk.astype(np.float32) * float(
        snapshot.voxel_m
    )
    return verts_ijk.astype(np.float32), verts_world.astype(np.float32), faces.astype(np.int32)


def _mesh_class_ids(snapshot, verts_ijk: np.ndarray, colors: np.ndarray, color_to_id: dict[int, int]) -> np.ndarray:
    ijk = np.rint(verts_ijk).astype(np.int32)
    ijk[:, 0] = np.clip(ijk[:, 0], 0, snapshot.sem_label.shape[0] - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, snapshot.sem_label.shape[1] - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, snapshot.sem_label.shape[2] - 1)
    labels = snapshot.sem_label[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return _decode_sem_ids(labels, colors, color_to_id)


def _set_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins)) * 0.56
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _bbox_lines(origin: np.ndarray, dims: np.ndarray, voxel_m: float) -> list[tuple[np.ndarray, np.ndarray]]:
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    max_w = origin + (np.asarray(dims, dtype=np.float32).reshape(3) - 1.0) * float(voxel_m)
    c = np.array(
        [
            [origin[0], origin[1], origin[2]],
            [max_w[0], origin[1], origin[2]],
            [max_w[0], max_w[1], origin[2]],
            [origin[0], max_w[1], origin[2]],
            [origin[0], origin[1], max_w[2]],
            [max_w[0], origin[1], max_w[2]],
            [max_w[0], max_w[1], max_w[2]],
            [origin[0], max_w[1], max_w[2]],
        ],
        dtype=np.float32,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(c[a], c[b]) for a, b in edges]


def _render_surface_png(
    *,
    snapshot,
    verts_world: np.ndarray,
    faces: np.ndarray,
    vertex_ids: np.ndarray,
    names: list[str],
    colors: np.ndarray,
    out_path: pathlib.Path,
) -> None:
    face_ids = []
    for f in faces:
        vals = [int(vertex_ids[i]) for i in f if int(vertex_ids[i]) >= 0]
        face_ids.append(Counter(vals).most_common(1)[0][0] if vals else -1)
    face_ids_arr = np.asarray(face_ids, dtype=np.int64)
    face_colors = np.asarray([_class_color(int(x), colors) for x in face_ids_arr], dtype=np.float32)

    fig = plt.figure(figsize=(16, 12), dpi=160)
    views = [
        ("isometric", 28, -45),
        ("top", 90, -90),
        ("front", 5, -90),
        ("side", 5, 0),
    ]
    tri_verts = verts_world[faces]
    for idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        mesh = Poly3DCollection(tri_verts, facecolors=face_colors, edgecolor="none", linewidths=0.0, alpha=0.96)
        ax.add_collection3d(mesh)
        for a, b in _bbox_lines(snapshot.origin_w, snapshot.dims, float(snapshot.voxel_m)):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="black", linewidth=0.7, alpha=0.6)
        _set_equal_3d(ax, verts_world)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")
        ax.set_zlabel("world z (m)")
        ax.grid(False)

    used_ids = [int(x) for x in sorted(set(face_ids_arr.tolist())) if int(x) >= 0]
    legend_items = []
    for class_id in used_ids:
        color = _class_color(class_id, colors)
        legend_items.append(plt.Line2D([0], [0], marker="s", color=color, label=f"{class_id}: {_class_name(class_id, names)}", linestyle=""))
    if legend_items:
        fig.legend(handles=legend_items, loc="lower center", ncol=min(4, len(legend_items)), frameon=False, fontsize=9)

    fig.suptitle(
        f"Global ESDF zero surface | frame {snapshot.frame_id} | dims={tuple(int(x) for x in snapshot.dims)} | "
        f"voxel={float(snapshot.voxel_m):.3f} m",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def _render_slices_png(snapshot, out_path: pathlib.Path) -> None:
    esdf = np.asarray(snapshot.esdf, dtype=np.float32)
    mid = [s // 2 for s in esdf.shape]
    slices = [
        ("x middle slice", esdf[mid[0], :, :].T),
        ("y middle slice", esdf[:, mid[1], :].T),
        ("z middle slice", esdf[:, :, mid[2]].T),
    ]
    vmax = float(np.nanpercentile(np.abs(esdf), 98))
    vmax = max(vmax, float(snapshot.voxel_m))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=160)
    for ax, (title, img) in zip(axes, slices):
        im = ax.imshow(img, origin="lower", cmap="RdBu_r", norm=norm)
        ax.contour(img, levels=[0.0], colors="black", linewidths=0.8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="ESDF signed distance (m)")
    fig.suptitle("Global ESDF middle slices (black contour = zero level)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_plotly_html(
    *,
    snapshot,
    verts_world: np.ndarray,
    faces: np.ndarray,
    vertex_ids: np.ndarray,
    colors: np.ndarray,
    out_path: pathlib.Path,
) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False

    vertex_rgb = []
    for class_id in vertex_ids.tolist():
        c = (_class_color(int(class_id), colors) * 255.0).clip(0, 255).astype(np.uint8)
        vertex_rgb.append(f"rgb({int(c[0])},{int(c[1])},{int(c[2])})")

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=verts_world[:, 0],
            y=verts_world[:, 1],
            z=verts_world[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            vertexcolor=vertex_rgb,
            opacity=0.96,
            name="ESDF zero surface",
        )
    )
    for idx, (a, b) in enumerate(_bbox_lines(snapshot.origin_w, snapshot.dims, float(snapshot.voxel_m))):
        fig.add_trace(
            go.Scatter3d(
                x=[a[0], b[0]],
                y=[a[1], b[1]],
                z=[a[2], b[2]],
                mode="lines",
                line=dict(color="black", width=3),
                showlegend=False,
                name=f"bounds_{idx}",
            )
        )
    fig.update_layout(
        title=(
            f"Global ESDF zero surface | frame {snapshot.frame_id} | "
            f"dims={tuple(int(x) for x in snapshot.dims)} | voxel={float(snapshot.voxel_m):.3f}m"
        ),
        scene=dict(aspectmode="data", xaxis_title="world x (m)", yaxis_title="world y (m)", zaxis_title="world z (m)"),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Export static and interactive visualizations for a saved ESDF snapshot.")
    parser.add_argument("--snapshot", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    snapshot = load_esdf_snapshot(args.snapshot)
    names, colors, color_to_id = _palette()
    verts_ijk, verts_world, faces = _extract_mesh(snapshot)
    vertex_ids = _mesh_class_ids(snapshot, verts_ijk, colors, color_to_id)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    surface_png = args.out_dir / "global_esdf_surface_semantic_views.png"
    slices_png = args.out_dir / "global_esdf_middle_slices.png"
    html_path = args.out_dir / "global_esdf_surface_interactive.html"
    summary_path = args.out_dir / "summary.txt"

    _render_surface_png(
        snapshot=snapshot,
        verts_world=verts_world,
        faces=faces,
        vertex_ids=vertex_ids,
        names=names,
        colors=colors,
        out_path=surface_png,
    )
    _render_slices_png(snapshot, slices_png)
    html_ok = _write_plotly_html(
        snapshot=snapshot,
        verts_world=verts_world,
        faces=faces,
        vertex_ids=vertex_ids,
        colors=colors,
        out_path=html_path,
    )

    used_ids = [int(x) for x in sorted(set(vertex_ids.tolist())) if int(x) >= 0]
    summary = [
        f"snapshot: {args.snapshot}",
        f"frame_id: {snapshot.frame_id}",
        f"dims: {tuple(int(x) for x in snapshot.dims)}",
        f"voxel_m: {float(snapshot.voxel_m):.6f}",
        f"origin_w: {np.asarray(snapshot.origin_w).tolist()}",
        f"surface_vertices: {len(verts_world)}",
        f"surface_faces: {len(faces)}",
        "surface_semantic_classes:",
    ]
    for class_id in used_ids:
        summary.append(f"  {class_id}: {_class_name(class_id, names)}")
    summary.append(f"interactive_html: {'yes' if html_ok else 'no'}")
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"[done] {surface_png}")
    print(f"[done] {slices_png}")
    if html_ok:
        print(f"[done] {html_path}")
    print(f"[done] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
