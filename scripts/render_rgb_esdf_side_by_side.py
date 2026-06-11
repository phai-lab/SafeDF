#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from scipy.spatial.transform import Rotation, Slerp
from skimage.measure import marching_cubes

from mast3r_slam.esdf_snapshot import load_esdf_snapshot


REPO_ROOT = Path(__file__).resolve().parents[1]
EFFICIENTVIT_ROOT = REPO_ROOT / "efficientvit"
if str(EFFICIENTVIT_ROOT) not in sys.path:
    sys.path.append(str(EFFICIENTVIT_ROOT))


def _load_ade20k_metadata():
    from applications.efficientvit_seg.eval_efficientvit_seg_model import ADE20KDataset  # type: ignore

    return tuple(ADE20KDataset.classes), np.asarray(ADE20KDataset.class_colors, dtype=np.uint8)


def _center_crop_square_bgr(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img_bgr[y0 : y0 + s, x0 : x0 + s]


def _center_crop_square_label(label_hw: np.ndarray) -> np.ndarray:
    h, w = label_hw.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return label_hw[y0 : y0 + s, x0 : x0 + s]


def _stable_hash_color(label: int) -> tuple[int, int, int]:
    v = int(label) & 0xFFFFFFFF
    return (
        int((v * 37 + 17) & 255),
        int((v * 57 + 29) & 255),
        int((v * 97 + 53) & 255),
    )


def _load_label_metadata(path: Path | None) -> tuple[dict[int, str], dict[int, tuple[int, int, int]]]:
    if path is None:
        return {}, {}
    obj = json.loads(path.read_text())
    names = {int(k): str(v) for k, v in obj.get("label_names", {}).items()}
    colors = {
        int(k): (int(v[0]), int(v[1]), int(v[2]))
        for k, v in obj.get("label_colors", {}).items()
        if isinstance(v, (list, tuple)) and len(v) >= 3
    }
    return names, colors


def _parse_rgb01(value: str) -> list[float]:
    parts = [float(x.strip()) for x in str(value).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected R,G,B")
    if max(parts) > 1.0:
        parts = [x / 255.0 for x in parts]
    return [float(np.clip(x, 0.0, 1.0)) for x in parts]


def _label_to_color(
    labels: np.ndarray,
    palette_u8: np.ndarray,
    label_colors: dict[int, tuple[int, int, int]] | None = None,
    hash_labels: bool = False,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    colors = np.zeros((labels.shape[0], 3), dtype=np.float64)
    colors[:] = np.array([0.75, 0.75, 0.78], dtype=np.float64)
    valid = labels >= 0
    if np.any(valid):
        vals = labels[valid]
        colors_valid = np.array([0.75, 0.75, 0.78], dtype=np.float64).reshape(1, 3).repeat(len(vals), axis=0)
        if label_colors:
            for lab, color in label_colors.items():
                m = vals == int(lab)
                if np.any(m):
                    colors_valid[m] = np.asarray(color, dtype=np.float64) / 255.0
        else:
            valid_ids = (vals >= 0) & (vals < len(palette_u8))
            if np.any(valid_ids):
                colors_valid[valid_ids] = palette_u8[vals[valid_ids]].astype(np.float64) / 255.0
        if hash_labels:
            for lab in np.unique(vals).tolist():
                if label_colors and int(lab) in label_colors:
                    continue
                if 0 <= int(lab) < len(palette_u8) and not label_colors:
                    continue
                colors_valid[vals == int(lab)] = np.asarray(_stable_hash_color(int(lab)), dtype=np.float64) / 255.0
        colors[valid] = colors_valid
    return colors


def _colorize_label_hw(
    label_hw: np.ndarray,
    palette_u8: np.ndarray,
    label_colors: dict[int, tuple[int, int, int]] | None = None,
    hash_labels: bool = False,
) -> np.ndarray:
    labels = np.asarray(label_hw, dtype=np.int64)
    flat_rgb = _label_to_color(
        labels.reshape(-1),
        palette_u8,
        label_colors=label_colors,
        hash_labels=hash_labels,
    )
    return np.clip(flat_rgb.reshape(labels.shape[0], labels.shape[1], 3) * 255.0, 0, 255).astype(np.uint8)


def _load_frame_semantic(
    sem_dir: Path,
    pattern: str,
    sem_frame_id: int,
    size: int,
) -> np.ndarray | None:
    path = sem_dir / pattern.format(frame_id=int(sem_frame_id))
    if not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        label_hw = np.load(str(path))
    else:
        label_hw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label_hw is None:
        return None
    label_hw = np.asarray(label_hw)
    if label_hw.ndim == 3:
        label_hw = label_hw[..., 0]
    if label_hw.ndim != 2:
        return None
    label_crop = _center_crop_square_label(label_hw)
    return cv2.resize(label_crop.astype(np.int32, copy=False), (int(size), int(size)), interpolation=cv2.INTER_NEAREST)


def _semantic_overlay_bgr(
    rgb_bgr: np.ndarray,
    label_hw: np.ndarray | None,
    palette_u8: np.ndarray,
    label_colors: dict[int, tuple[int, int, int]] | None,
    hash_labels: bool,
    alpha: float,
) -> np.ndarray:
    if label_hw is None:
        return rgb_bgr.copy()
    sem_rgb = _colorize_label_hw(label_hw, palette_u8, label_colors=label_colors, hash_labels=hash_labels)
    sem_bgr = cv2.cvtColor(sem_rgb, cv2.COLOR_RGB2BGR)
    valid = np.asarray(label_hw) >= 0
    out = rgb_bgr.copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    if np.any(valid):
        blended = cv2.addWeighted(rgb_bgr, 1.0 - a, sem_bgr, a, 0.0)
        out[valid] = blended[valid]
    return out


def _extract_mesh(
    snapshot,
    semantic: bool,
    palette_u8: np.ndarray,
    label_colors: dict[int, tuple[int, int, int]] | None = None,
    hash_labels: bool = False,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    verts, faces, normals, _ = marching_cubes(
        snapshot.esdf,
        level=0.0,
        spacing=(float(snapshot.voxel_m), float(snapshot.voxel_m), float(snapshot.voxel_m)),
    )
    verts = verts.astype(np.float32) + np.asarray(snapshot.origin_w, dtype=np.float32).reshape(1, 3)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    labels = np.full((verts.shape[0],), -1, dtype=np.int32)
    if semantic:
        ijk = np.rint(
            (verts - np.asarray(snapshot.origin_w, dtype=np.float32).reshape(1, 3)) / float(snapshot.voxel_m)
        ).astype(np.int32)
        ijk[:, 0] = np.clip(ijk[:, 0], 0, snapshot.sem_label.shape[0] - 1)
        ijk[:, 1] = np.clip(ijk[:, 1], 0, snapshot.sem_label.shape[1] - 1)
        ijk[:, 2] = np.clip(ijk[:, 2], 0, snapshot.sem_label.shape[2] - 1)
        labels = snapshot.sem_label[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            _label_to_color(labels, palette_u8, label_colors=label_colors, hash_labels=hash_labels)
        )
    else:
        mesh.paint_uniform_color([0.70, 0.76, 0.84])
    mesh.compute_vertex_normals()
    return mesh, labels


def _approx_intrinsics(size: int, fov_deg: float) -> np.ndarray:
    fov = float(fov_deg) * (np.pi / 180.0)
    fx = 0.5 * float(size) / max(1e-6, np.tan(0.5 * fov))
    fy = fx
    cx = 0.5 * (float(size) - 1.0)
    cy = 0.5 * (float(size) - 1.0)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _load_traj(path: Path):
    times = []
    xyz = []
    quat_xyzw = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) != 8:
            continue
        t, x, y, z, qx, qy, qz, qw = vals
        times.append(t)
        xyz.append([x, y, z])
        quat_xyzw.append([qx, qy, qz, qw])
    times = np.asarray(times, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    if len(times) < 1:
        raise ValueError(f"No poses found in {path}")
    return times, xyz, quat_xyzw


class PoseInterpolator:
    def __init__(self, times: np.ndarray, xyz: np.ndarray, quat_xyzw: np.ndarray):
        keep = np.ones(len(times), dtype=bool)
        if len(times) >= 2:
            keep[1:] = np.diff(times) > 1e-12
        self.times = times[keep]
        self.xyz = xyz[keep]
        self.rots = Rotation.from_quat(quat_xyzw[keep])
        self.slerp = None
        if len(self.times) >= 2:
            self.slerp = Slerp(self.times, self.rots)

    def pose_at(self, t: float) -> np.ndarray:
        if len(self.times) == 1:
            R = self.rots.as_matrix()[0]
            p = self.xyz[0]
        else:
            t = float(np.clip(t, self.times[0], self.times[-1]))
            p = np.empty(3, dtype=np.float64)
            for k in range(3):
                p[k] = np.interp(t, self.times, self.xyz[:, k])
            R = self.slerp([t]).as_matrix()[0]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = p
        return T


def _render_frame(renderer: rendering.OffscreenRenderer, K: np.ndarray, T_WC: np.ndarray, size: int) -> np.ndarray:
    T_CW = np.linalg.inv(T_WC)
    renderer.setup_camera(K, T_CW, size, size)
    img = np.asarray(renderer.render_to_image())
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    return img[:, :, :3]


def _legend_entries(
    labels: np.ndarray,
    class_names: tuple[str, ...],
    palette_u8: np.ndarray,
    label_names: dict[int, str] | None = None,
    label_colors: dict[int, tuple[int, int, int]] | None = None,
    hash_labels: bool = False,
) -> list[tuple[str, tuple[int, int, int]]]:
    uniq = np.unique(np.asarray(labels, dtype=np.int32))
    uniq = uniq[uniq >= 0]
    entries: list[tuple[str, tuple[int, int, int]]] = []
    for lab in uniq.tolist():
        if label_names and int(lab) in label_names:
            name = label_names[int(lab)]
        elif 0 <= lab < len(class_names):
            name = class_names[lab]
        else:
            name = f"label_{lab}"

        if label_colors and int(lab) in label_colors:
            color = tuple(int(x) for x in label_colors[int(lab)])
            entries.append((name, color))
        elif 0 <= lab < len(palette_u8):
            color = tuple(int(x) for x in palette_u8[lab].tolist())
            entries.append((name, color))
        elif hash_labels:
            entries.append((name, _stable_hash_color(int(lab))))
    return entries


def _layout_legend(entries: list[tuple[str, tuple[int, int, int]]], width: int) -> tuple[list[list[tuple[str, tuple[int, int, int], int]]], int]:
    if not entries:
        return [], 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    left_pad = 16
    right_pad = 16
    gap_x = 16
    swatch = 18
    line_h = 28
    rows: list[list[tuple[str, tuple[int, int, int], int]]] = []
    current: list[tuple[str, tuple[int, int, int], int]] = []
    used = left_pad
    max_w = max(80, width - right_pad)
    for name, color in entries:
        (tw, _th), _ = cv2.getTextSize(name, font, font_scale, thickness)
        item_w = swatch + 8 + tw
        extra_gap = gap_x if current else 0
        if used + extra_gap + item_w > max_w:
            rows.append(current)
            current = []
            used = left_pad
            extra_gap = 0
        if current:
            used += gap_x
        current.append((name, color, item_w))
        used += item_w
    if current:
        rows.append(current)
    legend_h = 18 + len(rows) * line_h + 10
    return rows, legend_h


def _draw_legend(frame_bgr: np.ndarray, entries: list[tuple[str, tuple[int, int, int]]], legend_h: int) -> np.ndarray:
    if legend_h <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    canvas = np.full((h + legend_h, w, 3), 245, dtype=np.uint8)
    canvas[:h] = frame_bgr
    cv2.rectangle(canvas, (0, h), (w - 1, h + legend_h - 1), (238, 238, 238), thickness=-1)
    cv2.line(canvas, (0, h), (w - 1, h), (180, 180, 180), 1, cv2.LINE_AA)
    rows, _ = _layout_legend(entries, w)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    left_pad = 16
    gap_x = 16
    swatch = 18
    y = h + 26
    for row in rows:
        x = left_pad
        for name, color, item_w in row:
            color_bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.rectangle(canvas, (x, y - 13), (x + swatch, y + 5), color_bgr, thickness=-1)
            cv2.rectangle(canvas, (x, y - 13), (x + swatch, y + 5), (60, 60, 60), thickness=1)
            cv2.putText(canvas, name, (x + swatch + 8, y), font, font_scale, (30, 30, 30), thickness, cv2.LINE_AA)
            x += item_w + gap_x
        y += 28
    return canvas


def main() -> int:
    p = argparse.ArgumentParser(description="Render side-by-side RGB and ESDF video with matched camera viewpoint.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--traj", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=480, help="Per-panel square size.")
    p.add_argument("--fov-deg", type=float, default=60.0, help="Approximate square-camera FOV for ESDF rendering.")
    p.add_argument("--semantic-mesh", action="store_true", help="Color ESDF mesh by semantic labels.")
    p.add_argument(
        "--label-metadata",
        type=Path,
        default=None,
        help="Optional JSON with label_names and label_colors maps for non-ADE semantic labels.",
    )
    p.add_argument(
        "--hash-labels",
        action="store_true",
        help="Assign deterministic colors to labels outside the ADE20K palette.",
    )
    p.add_argument("--unlit", action="store_true", help="Render ESDF mesh without lighting so colors exactly match the semantic palette.")
    p.add_argument("--fps", type=float, default=0.0, help="Override output FPS. Default: use input video FPS.")
    p.add_argument(
        "--source-fps",
        type=float,
        default=0.0,
        help="Override input-frame indexing FPS. Useful for ScanNet++ rgb.mkv when the SLAM image-folder timestamps use 30 FPS.",
    )
    p.add_argument(
        "--frame-sem-dir",
        type=Path,
        default=None,
        help="Optional per-frame semantic label-id maps for an RGB+semantic overlay middle panel.",
    )
    p.add_argument(
        "--frame-sem-pattern",
        type=str,
        default="{frame_id:06d}.npy",
        help="Filename pattern inside --frame-sem-dir. Uses processed semantic frame id.",
    )
    p.add_argument(
        "--frame-sem-subsample",
        type=int,
        default=10,
        help="Original RGB frame stride represented by each semantic label map.",
    )
    p.add_argument("--semantic-alpha", type=float, default=0.45, help="Alpha for RGB+semantic overlay.")
    p.add_argument(
        "--background",
        type=_parse_rgb01,
        default=[1.0, 1.0, 1.0],
        help="ESDF renderer background color as R,G,B in 0..1 or 0..255.",
    )
    p.add_argument(
        "--base-color",
        type=_parse_rgb01,
        default=None,
        help="Override non-semantic mesh base color as R,G,B in 0..1 or 0..255.",
    )
    p.add_argument("--roughness", type=float, default=0.85, help="Lit material roughness.")
    p.add_argument("--metallic", type=float, default=0.0, help="Lit material metallic value.")
    p.add_argument(
        "--lighting-profile",
        choices=["hard", "medium", "soft", "no-shadows"],
        default="medium",
        help="Open3D scene lighting profile for lit rendering.",
    )
    p.add_argument("--sun-intensity", type=float, default=45000.0, help="Sun intensity for lit rendering.")
    p.add_argument("--ibl-intensity", type=float, default=25000.0, help="Indirect light intensity for lit rendering.")
    p.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Multiply trajectory timestamps by this factor before interpolation. Use 10.0 for config/base_subsample10.yaml videos.",
    )
    p.add_argument(
        "--match-video-length",
        action="store_true",
        help="Render until the full input video ends; after the final pose, hold the last camera pose.",
    )
    p.add_argument("--start-time", type=float, default=0.0, help="Start rendering at this timestamp in seconds.")
    p.add_argument(
        "--max-duration",
        type=float,
        default=0.0,
        help="Maximum rendered duration in seconds. 0 means render until the selected end time.",
    )
    args = p.parse_args()

    class_names, palette_u8 = _load_ade20k_metadata()
    label_names, label_colors = _load_label_metadata(args.label_metadata)
    snapshot = load_esdf_snapshot(args.snapshot)
    mesh, mesh_labels = _extract_mesh(
        snapshot,
        semantic=bool(args.semantic_mesh),
        palette_u8=palette_u8,
        label_colors=label_colors,
        hash_labels=bool(args.hash_labels),
    )
    legend_entries = (
        _legend_entries(
            mesh_labels,
            class_names,
            palette_u8,
            label_names=label_names,
            label_colors=label_colors,
            hash_labels=bool(args.hash_labels),
        )
        if bool(args.semantic_mesh)
        else []
    )
    num_panels = 3 if args.frame_sem_dir is not None else 2
    _legend_rows, legend_h = _layout_legend(legend_entries, int(args.size) * num_panels)

    times, xyz, quat_xyzw = _load_traj(args.traj)
    times = times * float(args.time_scale)
    pose_interp = PoseInterpolator(times, xyz, quat_xyzw)
    t_end = float(times[-1])

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    native_src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    src_fps = float(args.source_fps) if float(args.source_fps) > 0 else native_src_fps
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_duration = float(src_frames) / src_fps if src_fps > 0 else 0.0
    out_fps = float(args.fps) if float(args.fps) > 0 else src_fps
    if out_fps <= 0:
        out_fps = 20.0
    sequence_end = src_duration if bool(args.match_video_length) else t_end
    start_time = max(0.0, float(args.start_time))
    render_duration = max(0.0, sequence_end - start_time)
    if float(args.max_duration) > 0.0:
        render_duration = min(render_duration, float(args.max_duration))

    K = _approx_intrinsics(int(args.size), float(args.fov_deg))
    renderer = rendering.OffscreenRenderer(int(args.size), int(args.size))
    renderer.scene.set_background([*args.background, 1.0])
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit" if bool(args.unlit) else "defaultLit"
    if args.base_color is not None:
        material.base_color = [*args.base_color, 1.0]
    material.base_roughness = float(np.clip(args.roughness, 0.0, 1.0))
    material.base_metallic = float(np.clip(args.metallic, 0.0, 1.0))
    if not bool(args.unlit):
        lighting_profiles = {
            "hard": rendering.Open3DScene.LightingProfile.HARD_SHADOWS,
            "medium": rendering.Open3DScene.LightingProfile.MED_SHADOWS,
            "soft": rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
            "no-shadows": rendering.Open3DScene.LightingProfile.NO_SHADOWS,
        }
        renderer.scene.scene.set_indirect_light_intensity(float(args.ibl_intensity))
        renderer.scene.scene.set_sun_light(
            [-0.5, -0.7, -0.5],
            [1.0, 1.0, 1.0],
            float(args.sun_intensity),
        )
        renderer.scene.set_lighting(lighting_profiles[str(args.lighting_profile)], [-0.5, -0.7, -0.5])
    renderer.scene.add_geometry("esdf_mesh", mesh, material)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (int(args.size) * num_panels, int(args.size) + int(legend_h)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {args.out}")

    frame_idx = 0
    while True:
        rel_t = frame_idx / out_fps
        if rel_t > render_duration + 1e-6:
            break
        t = start_time + rel_t

        src_idx = int(round(t * src_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break

        rgb_left = _center_crop_square_bgr(frame_bgr)
        rgb_left = cv2.resize(rgb_left, (int(args.size), int(args.size)), interpolation=cv2.INTER_AREA)

        overlay_panel = None
        if args.frame_sem_dir is not None:
            sem_frame_id = int(round(float(src_idx) / float(max(1, int(args.frame_sem_subsample)))))
            label_hw = _load_frame_semantic(
                args.frame_sem_dir,
                str(args.frame_sem_pattern),
                sem_frame_id,
                int(args.size),
            )
            overlay_panel = _semantic_overlay_bgr(
                rgb_left,
                label_hw,
                palette_u8,
                label_colors=label_colors,
                hash_labels=bool(args.hash_labels),
                alpha=float(args.semantic_alpha),
            )

        T_WC = pose_interp.pose_at(t)
        esdf_rgb = _render_frame(renderer, K, T_WC, int(args.size))
        esdf_bgr = cv2.cvtColor(esdf_rgb, cv2.COLOR_RGB2BGR)

        panels = [rgb_left]
        if overlay_panel is not None:
            panels.append(overlay_panel)
        panels.append(esdf_bgr)
        combo = np.concatenate(panels, axis=1)
        cv2.putText(combo, "RGB", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 230, 30), 2, cv2.LINE_AA)
        if overlay_panel is not None:
            cv2.putText(
                combo,
                "RGB+Semantic",
                (int(args.size) + 18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 230, 30),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            combo,
            "ESDF",
            ((num_panels - 1) * int(args.size) + 18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 230, 30),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            combo,
            f"t={t:.2f}s",
            (18, int(args.size) - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        combo = _draw_legend(combo, legend_entries, int(legend_h))
        writer.write(combo)
        frame_idx += 1

    writer.release()
    cap.release()
    renderer.scene.remove_geometry("esdf_mesh")
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
