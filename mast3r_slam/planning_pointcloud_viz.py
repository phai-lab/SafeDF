"""
CPU-side semantic point cloud visualization + frame overlay recorder (separate process).

Why this module exists
----------------------
The user wants a lightweight, planning-oriented "semantic point cloud" output that:
  - is driven by the SLAM reconstruction (keyframes),
  - can run ONLINE without blocking SLAM tracking,
  - does NOT depend on Open3D (so it can be used in minimal environments),
  - can visualize the result as an image sequence saved to disk.

This module is intentionally simple:
  - It runs in a separate `multiprocessing.Process` spawned by `main_semantic.py` / `main_semantic_limo.py`.
  - It reads shared-memory keyframes/states (already used by the official visualization).
  - It builds a *downsampled* point set from a configurable window of recent keyframes.
  - It projects the semantic point cloud into the current camera image using the current pose.
  - It writes overlay frames to disk and optionally shows an OpenCV window.

Important constraints
---------------------
  - This is a debugging / integration utility. It MUST NOT change SLAM behavior.
  - It must not add backpressure to the SLAM main loop (no blocking queues, no heavy locks).
  - It intentionally uses only hard-label semantics already available in the pipeline.
    For the "no voting" version requested by the user, we color points from `keyframe.semantic_label`
    (the legacy RGB semantic mask). V3 semantic pointmap voting (`keyframe.sem_label`) is NOT used here.
"""

from __future__ import annotations

import json
import os
import signal
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import lietorch
import torch

from mast3r_slam.config import config, set_global_config
from mast3r_slam.frame import Mode
from mast3r_slam.planning_pointcloud_buffer import LatestSharedKeyframeMapBuffer
from mast3r_slam.planning_pointcloud_buffer import LatestSharedTsdfVolumeBuffer
from mast3r_slam.planning_tsdf import RollingTsdfSemanticVolume
from mast3r_slam.planning_tsdf import RollingTsdfSemanticVolumeTorch


def _approx_intrinsics(h: int, w: int, fov_deg: float = 60.0) -> np.ndarray:
    """
    Build a simple pinhole intrinsics matrix K when calibration is not available.

    This is only a fallback for visualization. For accurate projection, enable real calibration
    (`config["use_calib"] == True`) and provide intrinsics through the dataset/stream.
    """

    fov = float(fov_deg) * (np.pi / 180.0)
    fx = 0.5 * float(w) / max(1e-6, np.tan(0.5 * fov))
    fy = fx
    cx = 0.5 * (float(w) - 1.0)
    cy = 0.5 * (float(h) - 1.0)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def _compute_grid_centers(h: int, w: int, grid: int) -> list[tuple[int, int]]:
    """
    Compute (y,x) centers for an evenly spaced grid (e.g., 3x3).

    This mirrors the idea used in `mast3r_slam/streaming/debug_viz.py`:
      - place points at i/(grid+1) fractions to avoid borders.

    Inputs:
      h, w: image height/width
      grid: number of samples per axis

    Output:
      centers: list of (cy, cx) integer coordinates
    """

    g = int(max(1, grid))
    centers: list[tuple[int, int]] = []
    for iy in range(1, g + 1):
        for ix in range(1, g + 1):
            cy = int(round((iy / float(g + 1)) * float(h - 1)))
            cx = int(round((ix / float(g + 1)) * float(w - 1)))
            centers.append((cy, cx))
    return centers


def _patch_mean_depth(
    depth_hw: np.ndarray,
    valid_hw: np.ndarray,
    *,
    cy: int,
    cx: int,
    patch: int,
) -> float:
    """
    Compute the mean depth in a small square patch around (cy,cx).

    Inputs:
      depth_hw: (H,W) float32 depth (meters), NaN where invalid
      valid_hw: (H,W) bool valid mask
      cy,cx: patch center
      patch: odd patch size (e.g., 3,5,7). If even, we treat it as patch+1.

    Output:
      mean_depth_m: float (NaN if no valid pixels)
    """

    h, w = int(depth_hw.shape[0]), int(depth_hw.shape[1])
    p = int(max(1, patch))
    if p % 2 == 0:
        p += 1
    r = p // 2
    y0 = max(0, int(cy) - r)
    y1 = min(h, int(cy) + r + 1)
    x0 = max(0, int(cx) - r)
    x1 = min(w, int(cx) + r + 1)

    d = depth_hw[y0:y1, x0:x1]
    m = valid_hw[y0:y1, x0:x1]
    if d.size == 0 or not np.any(m):
        return float("nan")
    return float(np.nanmean(d[m]))


def _rgb_to_label_id(rgb_hw3: np.ndarray) -> np.ndarray:
    """
    Convert an RGB mask to a label-id map using a palette-agnostic 24-bit packing:
      id = (R << 16) | (G << 8) | B

    Notes:
      - If the RGB mask is truly palette-based, this still yields a deterministic id per color.
      - If the RGB mask is bit-packed (as used in this repo for hard labels), this exactly recovers ids.
    """

    arr = np.asarray(rgb_hw3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3, got {arr.shape}")
    if arr.dtype != np.uint8:
        maxv = float(arr.max()) if arr.size else 0.0
        if maxv <= 1.0 + 1e-6:
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    r = arr[..., 0].astype(np.uint32)
    g = arr[..., 1].astype(np.uint32)
    b = arr[..., 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


def _auto_save_final_esdf_snapshot(
    *,
    out_dir: str,
    tsdf_vol: RollingTsdfSemanticVolume | RollingTsdfSemanticVolumeTorch | None,
    frame_id: int,
    timestamp_s: float,
    curr_T_WC_f32: np.ndarray | None,
    use_semantic: bool,
    w_min: float = 1.0,
    sem_w_min: float = 1.0,
    dilate_iters: int = 1,
) -> Path | None:
    """
    Persist the final TSDF-derived ESDF snapshot to disk on shutdown.

    This avoids depending on shared-memory lifetime after the producer exits.
    """

    if tsdf_vol is None or int(frame_id) < 0:
        return None

    try:
        from mast3r_slam.esdf_snapshot import EsdfSnapshot, compute_esdf_from_tsdf, save_esdf_snapshot
    except Exception as e:
        print(f"[PlanningTSDF] final snapshot import failed: {e!r}")
        return None

    try:
        if isinstance(tsdf_vol, RollingTsdfSemanticVolumeTorch):
            snap = tsdf_vol.snapshot_cpu()
            radius_m = float(tsdf_vol.radius_m)
        else:
            snap = tsdf_vol.snapshot()
            radius_m = float(tsdf_vol.radius_m)

        esdf, occ = compute_esdf_from_tsdf(
            tsdf=np.asarray(snap.tsdf, dtype=np.float32),
            weight=np.asarray(snap.weight, dtype=np.float32),
            sem_label=np.asarray(snap.sem_label, dtype=np.int32),
            sem_weight=np.asarray(snap.sem_weight, dtype=np.float32),
            voxel_m=float(snap.voxel_m),
            w_min=float(w_min),
            use_semantic=bool(use_semantic),
            obstacle_labels=None,
            sem_w_min=float(sem_w_min),
            dilate_iters=int(dilate_iters),
        )

        out_path = Path(out_dir) / "final_esdf_snapshot.npz"
        snapshot = EsdfSnapshot(
            schema="mast3r_esdf_snapshot_v1",
            frame_id=int(frame_id),
            timestamp_s=float(timestamp_s),
            scene_id="",
            radius_m=radius_m,
            voxel_m=float(snap.voxel_m),
            dims=np.asarray(esdf.shape, dtype=np.int32),
            origin_w=np.asarray(snap.origin_w, dtype=np.float32),
            curr_T_WC=np.asarray(
                np.eye(4, dtype=np.float32) if curr_T_WC_f32 is None else curr_T_WC_f32,
                dtype=np.float32,
            ).reshape(4, 4),
            esdf=np.asarray(esdf, dtype=np.float32),
            occ=np.asarray(occ, dtype=np.uint8),
            tsdf=np.asarray(snap.tsdf, dtype=np.float32),
            weight=np.asarray(snap.weight, dtype=np.float32),
            sem_label=np.asarray(snap.sem_label, dtype=np.int32),
            sem_weight=np.asarray(snap.sem_weight, dtype=np.float32),
            use_semantic=bool(use_semantic),
            obstacle_labels=np.asarray([], dtype=np.int32),
            w_min=float(w_min),
            sem_w_min=float(sem_w_min),
            dilate_iters=int(dilate_iters),
        )
        save_esdf_snapshot(snapshot, out_path)
        print(f"[PlanningTSDF] saved final ESDF snapshot: {out_path}")
        return out_path
    except Exception as e:
        print(f"[PlanningTSDF] failed to save final ESDF snapshot: {e!r}")
        print(traceback.format_exc())
        return None


def _load_external_semantic_label_hw(
    *,
    sem_dir: str,
    pattern: str,
    frame_id: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """
    Load an external per-frame semantic label-id map from disk and map it to (out_h,out_w).

    This is used by the rolling TSDF publisher when the user provides an offline segmentation
    directory (e.g., InternImage results stored as `000123.npy`).

    Supported input formats:
      - `.npy`: NumPy array shaped (H,W) with integer labels.
      - `.png`: single-channel image shaped (H,W) with integer labels.

    Alignment policy:
      - If the source resolution does not match (out_h,out_w), we center-crop to the target
        aspect ratio and then nearest-neighbor resize. This avoids geometric distortion of
        discrete labels while still matching the SLAM/TSDF observation resolution.
    """

    from pathlib import Path

    sem_dir_p = Path(str(sem_dir))
    name = str(pattern).format(frame_id=int(frame_id))
    path = sem_dir_p / name
    if not path.exists():
        raise FileNotFoundError(f"External semantic not found: {path}")

    if str(path).lower().endswith(".npy"):
        arr = np.load(str(path))
    else:
        import cv2  # type: ignore

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"Failed to read external semantic image: {path}")

    lab = np.asarray(arr)
    if lab.ndim == 3:
        # Fail fast: a color overlay (RGB) cannot be safely decoded into label ids here.
        raise ValueError(f"Expected label-id map (HxW), got {lab.shape} from {path}")
    if lab.ndim != 2:
        raise ValueError(f"Expected label-id map (HxW), got {lab.shape} from {path}")

    H0, W0 = int(lab.shape[0]), int(lab.shape[1])
    Ht, Wt = int(out_h), int(out_w)
    if H0 <= 0 or W0 <= 0 or Ht <= 0 or Wt <= 0:
        raise ValueError(f"Invalid semantic shapes: src={lab.shape} dst={(Ht, Wt)}")

    target_aspect = float(Wt) / float(Ht)
    src_aspect = float(W0) / float(H0)
    y0, y1, x0, x1 = 0, H0, 0, W0
    if src_aspect > target_aspect + 1e-6:
        new_w = int(round(float(H0) * target_aspect))
        new_w = max(1, min(new_w, W0))
        x0 = int((W0 - new_w) // 2)
        x1 = int(x0 + new_w)
    elif src_aspect < target_aspect - 1e-6:
        new_h = int(round(float(W0) / target_aspect))
        new_h = max(1, min(new_h, H0))
        y0 = int((H0 - new_h) // 2)
        y1 = int(y0 + new_h)
    cropped = lab[y0:y1, x0:x1]
    if cropped.shape[0] != Ht or cropped.shape[1] != Wt:
        import cv2  # type: ignore

        cropped = cv2.resize(cropped, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
    return np.asarray(cropped, dtype=np.int64)


def _load_external_pose_table(pose_json: str | None) -> dict | None:
    if pose_json is None or str(pose_json).strip() == "":
        return None
    path = Path(str(pose_json))
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected pose JSON object, got {type(data).__name__}: {path}")
    return data


def _lookup_external_pose_matrix(
    pose_table: dict | None,
    *,
    frame_id: int,
    stride: int,
    pattern: str,
    pose_key: str,
) -> np.ndarray | None:
    if pose_table is None:
        return None

    source_frame_id = int(frame_id) * max(1, int(stride))
    key = str(pattern).format(frame_id=source_frame_id)
    meta = pose_table.get(key)
    if meta is None:
        # Useful fallback for small custom JSONs keyed by integer strings.
        meta = pose_table.get(str(source_frame_id))
    if meta is None:
        raise KeyError(f"Pose frame not found: frame_id={frame_id}, source_frame_id={source_frame_id}, key={key!r}")
    if str(pose_key) not in meta:
        raise KeyError(f"Pose key {pose_key!r} not found for frame {key!r}")

    T = np.asarray(meta[str(pose_key)], dtype=np.float32)
    return T.reshape(4, 4)


def _hash_colorize_label_id(label_hw: np.ndarray) -> np.ndarray:
    """
    Deterministic pseudo-color palette (same style as `debug_viz.py`).

    Input:
      label_hw: (H,W) integer label ids
    Output:
      rgb_u8: (H,W,3) uint8
    """

    v = np.asarray(label_hw, dtype=np.uint32)
    r = (v * 37 + 17) & 255
    g = (v * 17 + 59) & 255
    b = (v * 97 + 101) & 255
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _sim3_matrix(T: lietorch.Sim3) -> np.ndarray:
    """
    Convert a lietorch Sim3 into a 4x4 float32 numpy matrix.

    The returned matrix maps points from the camera frame to the world frame:
      X_world = T_WC * X_cam
    """

    m = T.matrix()
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 3:
        m = m[0]
    return m


class _KalmanVectorRW:
    """
    Very small random-walk Kalman filter for a vector observation.

    Why this exists
    ---------------
    The user wants to reduce pose jitter for downstream planning/visualization.
    We implement an optional pose smoother inside the *publisher process* so that the
    shared-memory `T_WC` seen by downstream planning can be a smoothed pose.

    Model (per component)
    ---------------------
    Random walk:
      x_t = x_{t-1} + w,   w ~ N(0, Q)
      z_t = x_t     + v,   v ~ N(0, R)

    Update:
      P <- P + Q
      K <- P / (P + R)
      x <- x + K * (z - x)
      P <- (1 - K) * P

    IMPORTANT
    ---------
    - This filter only affects published/debug pose.
    - It must never feed back into SLAM estimation.
    """

    def __init__(self, *, q: float, r: float, p0: float = 1.0) -> None:
        self.q = float(max(q, 0.0))
        self.r = float(max(r, 1e-12))
        self.p0 = float(max(p0, 1e-12))
        self.x: np.ndarray | None = None
        self.p: np.ndarray | None = None

    def reset(self) -> None:
        self.x = None
        self.p = None

    def step(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        if self.x is None or self.p is None or self.x.shape != z.shape:
            self.x = z.copy()
            self.p = np.full_like(z, self.p0, dtype=np.float32)
            return self.x

        p = self.p + float(self.q)
        k = p / (p + float(self.r))
        x = self.x + k * (z - self.x)
        p = (1.0 - k) * p

        self.x = x
        self.p = p
        return self.x


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a unit quaternion [w,x,y,z].

    Notes:
      - Standard branch-based conversion.
      - Sufficient for publishing/debug filtering (no heavy deps).
    """

    M = np.asarray(R, dtype=np.float32).reshape(3, 3)
    tr = float(M[0, 0] + M[1, 1] + M[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (M[2, 1] - M[1, 2]) / s
        y = (M[0, 2] - M[2, 0]) / s
        z = (M[1, 0] - M[0, 1]) / s
    else:
        if (M[0, 0] > M[1, 1]) and (M[0, 0] > M[2, 2]):
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2.0
            w = (M[2, 1] - M[1, 2]) / s
            x = 0.25 * s
            y = (M[0, 1] + M[1, 0]) / s
            z = (M[0, 2] + M[2, 0]) / s
        elif M[1, 1] > M[2, 2]:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2.0
            w = (M[0, 2] - M[2, 0]) / s
            x = (M[0, 1] + M[1, 0]) / s
            y = 0.25 * s
            z = (M[1, 2] + M[2, 1]) / s
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2.0
            w = (M[1, 0] - M[0, 1]) / s
            x = (M[0, 2] + M[2, 0]) / s
            y = (M[1, 2] + M[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion [w,x,y,z] to a 3x3 rotation matrix.
    """

    qw, qx, qy, qz = (np.asarray(q, dtype=np.float32).reshape(4)).tolist()
    n = float(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz))
    if not np.isfinite(n) or n < 1e-12:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        n = 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


class _PoseKalmanFilter:
    """
    Pose smoother for publishing: translation Kalman + quaternion Kalman (approximate).

    Rotation filtering strategy (pragmatic + common in real-time)
    -------------------------------------------------------------
    We filter quaternions by:
      1) converting rotation matrix -> quaternion
      2) hemisphere correction: ensure dot(q_prev, q_meas) >= 0 to avoid sign flips
      3) component-wise Kalman update
      4) renormalize quaternion to unit length

    This avoids the "pi flip" symptom where quaternions jump sign even if rotation is continuous.

    IMPORTANT:
      This filter only affects published/visualized pose. It must never affect SLAM.
    """

    def __init__(self, *, pos_strength: float, rot_strength: float) -> None:
        self.pos_strength = float(pos_strength)
        self.rot_strength = float(rot_strength)

        # We expose only "strength" knobs (R). Larger R => trust measurements less => smoother.
        # Q is fixed small (random walk).
        self._kf_pos = _KalmanVectorRW(q=1e-6, r=max(self.pos_strength, 1e-12), p0=1.0)
        self._kf_rot = _KalmanVectorRW(q=1e-6, r=max(self.rot_strength, 1e-12), p0=1.0)

    def reset(self) -> None:
        self._kf_pos.reset()
        self._kf_rot.reset()

    def step(self, T_WC: np.ndarray) -> np.ndarray:
        M = np.asarray(T_WC, dtype=np.float32).reshape(4, 4)
        # ------------------------------------------------------------------
        # IMPORTANT: `T_WC` is a lietorch Sim3 (not guaranteed to be pure SE3).
        #
        # In Sim3, the upper-left 3x3 block is:
        #   A = s * R
        # where:
        #   - s is a uniform scale (scalar)
        #   - R is a proper rotation matrix (det(R)=+1)
        #
        # Our original implementation treated `A` as a rotation matrix and fed it
        # directly into quaternion conversion + filtering. That is WRONG when s!=1:
        #   - quaternion conversion assumes an orthonormal rotation matrix
        #   - filtering would "strip" the scale and effectively turn Sim3 into SE3
        #   - downstream, this breaks consistency between:
        #       * keyframe poses (still Sim3, updated by loop closure)
        #       * current pose (accidentally converted to SE3)
        #     causing the classic symptom: "current and keyframe trajectory don't align".
        #
        # Fix:
        #   - Estimate scale s from det(A)^(1/3)
        #   - Normalize: R = A / s (so R is orthonormal)
        #   - Filter translation and rotation on (t, R)
        #   - Recompose: A_f = s * R_f (preserve Sim3 scale)
        # ------------------------------------------------------------------
        A = M[:3, :3]
        detA = float(np.linalg.det(A))
        if not np.isfinite(detA) or detA == 0.0:
            s = 1.0
        else:
            # For Sim3, det(A) ~= s^3. We use abs() to be robust to tiny negative values
            # from numerical noise, but we keep s positive.
            s = float(np.cbrt(abs(detA)))
            if not np.isfinite(s) or s < 1e-12:
                s = 1.0
        R = (A / s).astype(np.float32, copy=False)
        t = M[:3, 3]

        # Translation.
        t_f = t
        if self.pos_strength > 0.0:
            t_f = self._kf_pos.step(t.astype(np.float32)).reshape(3)

        # Rotation.
        R_f = R
        if self.rot_strength > 0.0:
            q_meas = _rotmat_to_quat_wxyz(R)
            q_prev = self._kf_rot.x
            if q_prev is not None:
                q_prev = np.asarray(q_prev, dtype=np.float32).reshape(4)
                if float(np.dot(q_prev, q_meas)) < 0.0:
                    q_meas = -q_meas
            q_f = self._kf_rot.step(q_meas.astype(np.float32)).reshape(4)
            n = float(np.linalg.norm(q_f))
            if np.isfinite(n) and n > 1e-12:
                q_f = (q_f / n).astype(np.float32)
            else:
                q_f = q_meas
            R_f = _quat_wxyz_to_rotmat(q_f)

        out = np.array(M, copy=True)
        # Recompose Sim3: keep the *original* scale `s` and only smooth rotation.
        out[:3, :3] = (float(s) * R_f).astype(np.float32, copy=False)
        out[:3, 3] = t_f
        out[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return out


@dataclass
class _KfBlock:
    """
    Cached per-keyframe block in keyframe camera coordinates.

    We store points in the keyframe camera frame so we can apply the latest keyframe pose and the
    latest current pose at visualization time (avoids stale world coordinates if backend updates poses).
    """

    frame_id: int
    n_updates: int
    h: int
    w: int
    Xk: np.ndarray  # (N,3) float32, keyframe camera coordinates
    Ck: np.ndarray  # (N,) float32, average confidence (optional gating)
    # Semantic is represented as an integer ID per point for planning/control.
    # We ALSO keep a per-point RGB color for visualization only (computed from label_id).
    label_id: np.ndarray  # (N,) int32, semantic class id per point
    rgb_sem: np.ndarray  # (N,3) uint8, semantic colors (visualization-only)
    # Per-point appearance color (from keyframe unnormalized RGB image).
    # This enables consumer-side reprojection to render an "RGB-like" view from point cloud + pose,
    # without passing full images across processes.
    rgb_u8: np.ndarray  # (N,3) uint8, keyframe RGB colors aligned with Xk


def _build_keyframe_block(
    *,
    keyframe,
    stride: int,
    conf_threshold: float,
    max_points: int | None = None,
) -> Optional[_KfBlock]:
    """
    Extract a downsampled point set and semantic colors from a keyframe.

    IMPORTANT:
      - This uses `keyframe.semantic_label` (legacy RGB mask) as the semantic source,
        per user request ("do not use voting for now").
      - If semantic_label is missing, we fall back to a constant color.
    """

    fid = int(getattr(keyframe, "frame_id", -1))
    n_updates = int(getattr(keyframe, "N_updates", 0))

    # Resolve image shape (H,W) for this keyframe.
    h, w = (int(x) for x in keyframe.img_shape.flatten().tolist())
    if h <= 0 or w <= 0:
        return None

    # Points: use the already-fused pointmap.
    # In this codebase, X_canon is stored as a flattened (H*W,3) tensor.
    if keyframe.X_canon is None:
        return None
    s = int(max(1, stride))

    # -------------------------------------------------------------------------
    # Performance critical: avoid copying the full dense pointmap to CPU.
    #
    # The fused pointmap `X_canon` lives on GPU in shared memory and is (H*W,3).
    # A naive `.cpu().numpy()` would copy ~H*W*3 floats per keyframe (e.g., ~3MB for 512^2),
    # which can:
    #   - stall the GPU (synchronization)
    #   - contend with the main SLAM process (MASt3R inference)
    #
    # Instead, we slice on GPU first (stride downsample) and only transfer the smaller array:
    #   (H/s * W/s) points, then we do filtering on CPU.
    # -------------------------------------------------------------------------
    X = keyframe.X_canon
    if isinstance(X, torch.Tensor):
        X_t = X.reshape(h, w, 3)
        X_ds_t = X_t[::s, ::s].reshape(-1, 3)
        Xs = X_ds_t.detach().to("cpu").to(torch.float32).numpy()
    else:
        X_np = np.asarray(X, dtype=np.float32).reshape(h, w, 3)
        Xs = X_np[::s, ::s].reshape(-1, 3).astype(np.float32, copy=False)

    # Confidence: average confidence is used by the official visualization to gate points.
    C = keyframe.get_average_conf()
    if isinstance(C, torch.Tensor):
        C_t = C.reshape(h, w)
        C_ds_t = C_t[::s, ::s].reshape(-1)
        Cs = C_ds_t.detach().to("cpu").to(torch.float32).numpy()
    else:
        C_np = np.asarray(C, dtype=np.float32).reshape(h, w)
        Cs = C_np[::s, ::s].reshape(-1).astype(np.float32, copy=False)

    # Filter invalid points and low-confidence points.
    valid = np.isfinite(Xs).all(axis=1) & (Xs[:, 2] > 0.0) & np.isfinite(Cs)
    if conf_threshold > 0:
        valid = valid & (Cs >= float(conf_threshold))
    if not np.any(valid):
        return None
    Xs = Xs[valid]
    Cs = Cs[valid]

    # Optional cap on the number of points per keyframe.
    #
    # Motivation (shared-memory publisher):
    #   The keyframe-centric shared-memory schema stores a fixed number of point slots per keyframe
    #   (e.g., 8192). We therefore cap the extracted list to keep:
    #     - memory writes bounded and predictable
    #     - consumer-side sampling simple (fixed layout)
    #
    # IMPORTANT:
    #   We intentionally do NOT sort by confidence here (sorting can be expensive and was discouraged
    #   in previous real-time constraints). We simply take the first `max_points`.
    mp = None if max_points is None else int(max(0, int(max_points)))
    if mp is not None and mp > 0 and Xs.shape[0] > mp:
        Xs = Xs[:mp]
        Cs = Cs[:mp]

    # -------------------------------------------------------------------------
    # Per-point RGB appearance (for consumer-side visualization)
    # -------------------------------------------------------------------------
    #
    # We sample RGB from `keyframe.uimg` at the same pixels used for X/C.
    #
    # IMPORTANT:
    #   - This is NOT semantic. This is appearance color (texture).
    #   - It is used only to render a point-cloud reprojected "RGB" panel on the consumer side.
    #   - If uimg is missing, we fall back to a constant mid-gray.
    rgb_u8 = None
    try:
        if hasattr(keyframe, "uimg") and keyframe.uimg is not None:
            rgb = keyframe.uimg
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.detach().cpu().numpy()
            rgb = np.asarray(rgb)
            rgb = rgb.reshape(h, w, 3)
            rgb_ds = rgb[::s, ::s].reshape(-1, 3)[valid]

            # `uimg` is typically float in [0,1]. We convert to uint8 [0,255] for compactness.
            if rgb_ds.dtype != np.uint8:
                maxv = float(np.nanmax(rgb_ds)) if rgb_ds.size else 0.0
                if maxv <= 1.0 + 1e-6:
                    rgb_ds = np.clip(rgb_ds * 255.0, 0.0, 255.0).astype(np.uint8)
                else:
                    rgb_ds = np.clip(rgb_ds, 0.0, 255.0).astype(np.uint8)
            rgb_u8 = rgb_ds
    except Exception:
        rgb_u8 = None
    if rgb_u8 is None:
        rgb_u8 = np.full((Xs.shape[0], 3), 127, dtype=np.uint8)

    # Semantic: from `keyframe.semantic_label` (RGB mask).
    #
    # IMPORTANT:
    #   - Downstream planning wants semantic as an integer class ID, NOT an RGB color.
    #   - For visualization, we derive a deterministic pseudo-color from that ID.
    label_id_i32 = None
    rgb_sem_u8 = None
    if hasattr(keyframe, "semantic_label") and keyframe.semantic_label is not None:
        sem = keyframe.semantic_label
        if isinstance(sem, torch.Tensor):
            sem = sem.detach().cpu().numpy()
        sem = np.asarray(sem)
        # semantic_label is stored as (H,W,3) float in [0,1] in this repo.
        sem = sem.reshape(h, w, 3)
        sem_ds = sem[::s, ::s]
        try:
            lab = _rgb_to_label_id(sem_ds)
            # Per-point semantic ID (planning/control).
            label_id_i32 = lab.reshape(-1)[valid].astype(np.int32, copy=False)
            # Per-point semantic RGB (visualization-only).
            rgb_sem_u8 = _hash_colorize_label_id(lab).reshape(-1, 3)[valid]
        except Exception:
            # If anything fails, fall back to a constant color.
            label_id_i32 = None
            rgb_sem_u8 = None

    if label_id_i32 is None:
        label_id_i32 = np.full((Xs.shape[0],), np.int32(-1), dtype=np.int32)
    if rgb_sem_u8 is None:
        rgb_sem_u8 = np.full((Xs.shape[0], 3), (0, 255, 255), dtype=np.uint8)  # cyan

    return _KfBlock(
        frame_id=fid,
        n_updates=n_updates,
        h=h,
        w=w,
        Xk=Xs,
        Ck=Cs,
        label_id=label_id_i32,
        rgb_sem=rgb_sem_u8,
        rgb_u8=rgb_u8,
    )


def _project_points_to_image(
    *,
    X_cam: np.ndarray,
    rgb_u8: np.ndarray,
    K: np.ndarray,
    colors_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points (in CURRENT camera coordinates) into the RGB image.

    Implementation:
      - O(N) z-buffer using a 64-bit key trick (no sorting required):
          key = (z_int << 32) | idx
          best_key[pix] = min(best_key[pix], key)
        This yields the nearest point per pixel and its index.

    Inputs:
      X_cam: (N,3) float32, points in the current camera frame
      rgb_u8: (H,W,3) uint8, background image in RGB
      K: (3,3) float32, intrinsics
      colors_u8: (N,3) uint8, per-point color (semantic)

    Output:
      overlay_rgb_u8: (H,W,3) uint8
      depth_hw_m:     (H,W) float32, z-buffer depth in meters (NaN where invalid)
      valid_hw:       (H,W) bool, valid mask for the z-buffer depth
    """

    img = np.asarray(rgb_u8, dtype=np.uint8)
    h, w = int(img.shape[0]), int(img.shape[1])
    K = np.asarray(K, dtype=np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    X = np.asarray(X_cam, dtype=np.float32)
    z = X[:, 2]
    valid = np.isfinite(X).all(axis=1) & (z > 1e-6)
    if not np.any(valid):
        depth_hw = np.full((h, w), np.nan, dtype=np.float32)
        valid_hw = np.zeros((h, w), dtype=bool)
        return img, depth_hw, valid_hw
    X = X[valid]
    z = X[:, 2]
    col = np.asarray(colors_u8, dtype=np.uint8)[valid]

    u = (fx * (X[:, 0] / z) + cx).astype(np.int32)
    v = (fy * (X[:, 1] / z) + cy).astype(np.int32)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h) & np.isfinite(z)
    if not np.any(inside):
        depth_hw = np.full((h, w), np.nan, dtype=np.float32)
        valid_hw = np.zeros((h, w), dtype=bool)
        return img, depth_hw, valid_hw
    u = u[inside]
    v = v[inside]
    z = z[inside]
    col = col[inside]

    pix = (v.astype(np.int64) * int(w) + u.astype(np.int64)).astype(np.int64)
    idx = np.arange(col.shape[0], dtype=np.uint64)
    z_int = np.clip(z * 10000.0, 0.0, float(2**31 - 1)).astype(np.uint64)
    key = (z_int << 32) | idx

    sentinel = np.uint64(0xFFFFFFFFFFFFFFFF)
    best_key = np.full((h * w,), sentinel, dtype=np.uint64)
    np.minimum.at(best_key, pix, key)

    chosen = best_key != sentinel
    if not np.any(chosen):
        depth_hw = np.full((h, w), np.nan, dtype=np.float32)
        valid_hw = np.zeros((h, w), dtype=bool)
        return img, depth_hw, valid_hw
    best_idx = (best_key[chosen] & np.uint64(0xFFFFFFFF)).astype(np.int64)

    out = img.copy()
    out_flat = out.reshape(-1, 3)
    chosen_flat_idx = np.where(chosen)[0]
    out_flat[chosen_flat_idx] = col[best_idx]

    # Build a depth buffer aligned with the RGB image.
    #
    # Notes:
    #   - This depth comes from the projected keyframe point cloud (z-buffer) and is therefore
    #     typically more stable than per-frame monocular depth.
    #   - It can contain holes where no points project.
    depth_flat = np.full((h * w,), np.nan, dtype=np.float32)
    depth_flat[chosen_flat_idx] = z[best_idx].astype(np.float32)
    depth_hw = depth_flat.reshape(h, w)
    valid_hw = chosen.reshape(h, w)
    return out, depth_hw, valid_hw


def run_planning_pointcloud_viz(
    cfg: dict,
    states,
    keyframes,
    *,
    out_dir: str = "logs/planning_pointcloud",
    fps: float = 5.0,
    max_keyframes: int = 30,
    stride: int = 4,
    conf_threshold: float = 1.5,
    show_window: bool = True,
    window_name: str = "Planning Semantic PointCloud (CPU)",
    save_images: bool = True,
    publish_shm: bool = True,
    shm_info_path: str | None = None,
    shm_max_points: int = 200_000,
    shm_max_keyframes: int = 1024,
    shm_points_per_kf: int = 8192,
    # Pose smoothing (optional, publishing/debug only):
    #   - 0.0 disables filtering
    #   - larger values mean stronger smoothing (trust measurements less)
    pose_kalman_pos: float = 0.0,
    pose_kalman_rot: float = 0.0,
    # -----------------------------------------------------------------
    # Rolling TSDF (+ optional semantic) publish (optional)
    #
    # This is intended for downstream CBF/planning:
    #   - TSDF provides a smooth implicit surface
    #   - semantics are fused at voxel level (hard label + weight)
    #
    # IMPORTANT:
    #   This is output-only (runs in this aux process). It must never affect SLAM.
    # -----------------------------------------------------------------
    publish_tsdf: bool = False,
    tsdf_shm_info_path: str | None = None,
    tsdf_radius_m: float = 2.0,
    tsdf_voxel_m: float = 0.1,
    tsdf_trunc_m: float | None = None,
    tsdf_max_weight: float = 100.0,
    tsdf_use_semantic: bool = True,
    tsdf_semantic_band_m: float | None = None,
    tsdf_frame_sem_dir: str | None = None,
    tsdf_frame_sem_pattern: str = "{frame_id:06d}.npy",
    tsdf_pose_json: str | None = None,
    tsdf_pose_key: str = "aligned_pose",
    tsdf_pose_frame_stride: int = 1,
    tsdf_pose_frame_pattern: str = "frame_{frame_id:06d}",
    tsdf_backend: str = "numpy",
    tsdf_torch_device: str = "cuda",
    tsdf_torch_dtype: str = "float32",
) -> None:
    """
    Entry point to be spawned as a separate process.

    Inputs:
      cfg: global config dict (passed from main process)
      states: SharedStates
      keyframes: SharedKeyframes

    Output:
      - Saves overlay images to `out_dir` (if enabled)
      - Shows an OpenCV window (if enabled)

    Important:
      This process must be robust and must NOT throw unhandled exceptions that could bring down SLAM.
    """

    set_global_config(cfg)

    # ---------------------------------------------------------------------
    # Graceful shutdown (IMPORTANT for shared_memory cleanup)
    # ---------------------------------------------------------------------
    # This function runs in a separate `multiprocessing.Process`.
    #
    # The SLAM main process stops us using `Process.terminate()`, which sends SIGTERM on Linux.
    # If we do not handle SIGTERM, Python may exit immediately without executing `finally`,
    # leaving shared-memory segments (meta/pose/points) leaked. This shows up as:
    #   "resource_tracker: There appear to be N leaked shared_memory objects..."
    #
    # To ensure buffers are always `close(unlink=True)`'d, we trap SIGTERM/SIGINT and break out
    # of the main loop cleanly so our `finally` block runs.
    _should_stop = {"stop": False}  # mutable cell for signal handler closure

    def _handle_stop_signal(signum: int, frame) -> None:  # type: ignore[no-untyped-def]
        _should_stop["stop"] = True

    try:
        signal.signal(signal.SIGTERM, _handle_stop_signal)
        signal.signal(signal.SIGINT, _handle_stop_signal)
    except Exception:
        # Best-effort only. If signal registration fails (rare), we still run without it.
        pass

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Optional latest-only shared-memory publisher.
    #
    # This enables a downstream planning process (separate OS process) to consume:
    #   - a semantic point cloud (world coordinates + semantic colors),
    #   - the current pose (T_WC),
    # without requiring any GUI window or file I/O.
    #
    # The buffer is fixed-size (`shm_max_points`) and overwritten each update.
    shm_buf: LatestSharedKeyframeMapBuffer | None = None
    if publish_shm:
        if shm_info_path is None or str(shm_info_path).strip() == "":
            shm_info_path = str(Path(out_dir) / "shm_info.json")
        # NOTE:
        #   We intentionally publish a KEYFRAME-CENTRIC map (not a flattened world-space point cloud).
        #   See `mast3r_slam/planning_pointcloud_buffer.py` for the detailed schema and rationale.
        shm_buf = LatestSharedKeyframeMapBuffer(
            prefix="mast3r_planning_kfmap",
            info_path=str(shm_info_path),
        )
        # IMPORTANT:
        #   `shm_max_points` is kept only for backward compatibility with older CLI/scripts.
        #   The keyframe-centric schema is sized by:
        #     - shm_max_keyframes (K)
        #     - shm_points_per_kf (P)
        shm_buf.ensure(max_keyframes=int(shm_max_keyframes), points_per_kf=int(shm_points_per_kf))

    # Optional TSDF(+semantic) latest-only shared-memory publisher.
    #
    # We keep this separate from the keyframe-map shm schema so existing pointcloud consumers
    # are not broken. The TSDF volume is local+rolling and meant for near-field planning.
    tsdf_buf: LatestSharedTsdfVolumeBuffer | None = None
    tsdf_vol: RollingTsdfSemanticVolume | RollingTsdfSemanticVolumeTorch | None = None
    tsdf_pose_table: dict | None = None
    if bool(publish_tsdf):
        tsdf_pose_table = _load_external_pose_table(tsdf_pose_json)
        if tsdf_pose_table is not None:
            print(
                "[PlanningTSDF] using external camera poses:",
                f"path={tsdf_pose_json}",
                f"key={tsdf_pose_key}",
                f"stride={int(tsdf_pose_frame_stride)}",
                f"pattern={tsdf_pose_frame_pattern}",
            )
        if tsdf_shm_info_path is None or str(tsdf_shm_info_path).strip() == "":
            tsdf_shm_info_path = str(Path(out_dir) / "tsdf_shm_info.json")
        tsdf_buf = LatestSharedTsdfVolumeBuffer(
            prefix="mast3r_planning_tsdf",
            info_path=str(tsdf_shm_info_path),
        )
        # Provide the (H,W) of the semantic/depth observation so the TSDF shm schema can
        # optionally publish per-frame debug images (e.g., semantic label ids).
        #
        # IMPORTANT:
        #   - `states.uimg` is a CPU shared tensor shaped (H,W,3) AFTER `img_downsample`.
        #   - These dimensions are stable during the run, so we can allocate the shared-memory
        #     frame buffers once here.
        #
        # This does NOT allocate any additional heavy buffers unless the TSDF shm schema
        # chooses to publish them (see `LatestSharedTsdfVolumeBuffer` in planning_pointcloud_buffer.py).
        frame_hw = None
        try:
            uh, uw = int(states.uimg.shape[0]), int(states.uimg.shape[1])
            if uh > 0 and uw > 0:
                frame_hw = (uh, uw)
        except Exception:
            frame_hw = None

        tsdf_buf.ensure(radius_m=float(tsdf_radius_m), voxel_m=float(tsdf_voxel_m), frame_hw=frame_hw)
        backend = str(tsdf_backend).lower().strip()
        if backend == "torch":
            # Torch backend is optional; fall back to numpy if torch is missing or CUDA isn't available.
            try:
                dev = str(tsdf_torch_device)
                if str(dev).startswith("cuda") and (not torch.cuda.is_available()):
                    raise RuntimeError("torch.cuda.is_available() is False")
                tsdf_vol = RollingTsdfSemanticVolumeTorch(
                    radius_m=float(tsdf_radius_m),
                    voxel_m=float(tsdf_voxel_m),
                    trunc_m=None if tsdf_trunc_m is None else float(tsdf_trunc_m),
                    max_weight=float(tsdf_max_weight),
                    semantic_band_m=None if tsdf_semantic_band_m is None else float(tsdf_semantic_band_m),
                    invalid_label=-1,
                    device=str(dev),
                    tsdf_dtype=str(tsdf_torch_dtype),
                )
                print(f"[PlanningTSDF] backend=torch device={dev} dtype={tsdf_torch_dtype}")
            except Exception as e:
                print(f"[PlanningTSDF] torch backend unavailable ({e!r}); falling back to numpy.")
                tsdf_vol = RollingTsdfSemanticVolume(
                    radius_m=float(tsdf_radius_m),
                    voxel_m=float(tsdf_voxel_m),
                    trunc_m=None if tsdf_trunc_m is None else float(tsdf_trunc_m),
                    max_weight=float(tsdf_max_weight),
                    semantic_band_m=None if tsdf_semantic_band_m is None else float(tsdf_semantic_band_m),
                    invalid_label=-1,
                )
        else:
            tsdf_vol = RollingTsdfSemanticVolume(
                radius_m=float(tsdf_radius_m),
                voxel_m=float(tsdf_voxel_m),
                trunc_m=None if tsdf_trunc_m is None else float(tsdf_trunc_m),
                max_weight=float(tsdf_max_weight),
                semantic_band_m=None if tsdf_semantic_band_m is None else float(tsdf_semantic_band_m),
                invalid_label=-1,
            )

    # Cache per-keyframe extracted point blocks (in keyframe camera coordinates).
    # We key by keyframe index because indices are stable while keyframes are appended.
    blocks: Dict[int, _KfBlock] = {}
    pose_filter: _PoseKalmanFilter | None = None
    if float(pose_kalman_pos) > 0.0 or float(pose_kalman_rot) > 0.0:
        pose_filter = _PoseKalmanFilter(
            pos_strength=float(pose_kalman_pos),
            rot_strength=float(pose_kalman_rot),
        )

    period = 1.0 / max(1e-3, float(fps))
    next_t = time.time()
    frame_counter = 0
    last_tsdf_err_print_t = 0.0
    last_tsdf_pose_err_print_t = 0.0
    last_tsdf_frame_id = -1
    last_tsdf_timestamp_s = 0.0
    last_tsdf_curr_T_WC: np.ndarray | None = None
    do_overlay = bool(show_window) or bool(save_images)

    # Main loop: run until SLAM terminates or user closes the window.
    try:
        while not bool(_should_stop["stop"]):
            try:
                # Exit when SLAM terminates.
                try:
                    if states.get_mode() == Mode.TERMINATED:
                        break
                except Exception:
                    pass

                # Get current frame snapshot.
                curr = states.get_frame()
                h, w = (int(x) for x in curr.img_shape.flatten().tolist())
                if h <= 0 or w <= 0:
                    time.sleep(0.01)
                    continue
                fid = int(getattr(curr, "frame_id", frame_counter))

                # Current image (CPU float32 in [0,1]) -> uint8 RGB.
                #
                # We may need the RGB image for two independent purposes:
                #   1) 2D overlay visualization in this process (`do_overlay`)
                #   2) TSDF viewer debugging in another process (shared-memory publish of `frame_rgb_u8`)
                #
                # Therefore we convert when EITHER is enabled.
                rgb_u8 = None
                if bool(do_overlay) or (tsdf_buf is not None):
                    try:
                        rgb = np.asarray(curr.uimg.detach().cpu().numpy(), dtype=np.float32).reshape(int(h), int(w), 3)
                        if rgb.max() <= 1.0 + 1e-6:
                            rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
                        else:
                            rgb_u8 = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
                    except Exception:
                        rgb_u8 = None

                # Intrinsics.
                if bool(config.get("use_calib", False)):
                    try:
                        Kt = keyframes.get_intrinsics()
                        if isinstance(Kt, torch.Tensor):
                            K = Kt.detach().cpu().numpy().astype(np.float32)
                        else:
                            K = np.asarray(Kt, dtype=np.float32)
                    except Exception:
                        K = _approx_intrinsics(h, w)
                else:
                    K = _approx_intrinsics(h, w)

                # Current pose matrix (camera->world). We optionally smooth it for publishing/debug.
                #
                # IMPORTANT:
                #   This does NOT change SLAM. It only affects:
                #     - the reprojection visualization in this helper process
                #     - the pose published to shared memory for planning consumption
                #     - the optional TSDF/ESDF output generated by this helper process
                M_WCf = _sim3_matrix(curr.T_WC)  # current camera -> world
                external_pose_used = False
                if tsdf_pose_table is not None:
                    try:
                        M_ext = _lookup_external_pose_matrix(
                            tsdf_pose_table,
                            frame_id=int(fid),
                            stride=int(tsdf_pose_frame_stride),
                            pattern=str(tsdf_pose_frame_pattern),
                            pose_key=str(tsdf_pose_key),
                        )
                        if M_ext is not None:
                            M_WCf = M_ext
                            external_pose_used = True
                    except Exception as e:
                        now_pose = time.time()
                        if now_pose - last_tsdf_pose_err_print_t > 2.0:
                            print(f"[PlanningTSDF] external pose lookup failed (fid={fid}): {e!r}")
                            last_tsdf_pose_err_print_t = now_pose
                if pose_filter is not None and not external_pose_used:
                    # NOTE:
                    #   This is publishing/debug smoothing only. It must never feed back into SLAM.
                    M_WCf = pose_filter.step(M_WCf)
                M_CWf = np.linalg.inv(M_WCf)  # world -> current camera

                # -----------------------------------------------------------------
                # Rolling TSDF(+semantic) fusion and publish (optional)
                # -----------------------------------------------------------------
                #
                # We integrate *per-frame* depth and semantics into a local rolling TSDF volume.
                # Depth source (per current frame):
                #   - frame.X_canon[...,2]  (camera Z, meters), computed by MASt3R for this frame.
                # Semantic source (per current frame):
                #   - frame.semantic_label  (H,W,3) RGB-packed hard labels on CPU.
                #
                # IMPORTANT:
                #   - This runs in this auxiliary process only.
                #   - It never affects SLAM tracking/optimization.
                #   - All errors are swallowed so SLAM cannot be crashed by this debug feature.
                if tsdf_buf is not None and tsdf_vol is not None:
                    try:
                        depth_hw = None
                        valid_hw = None
                        if getattr(curr, "X_canon", None) is not None:
                            Xc_any = curr.X_canon
                            if isinstance(tsdf_vol, RollingTsdfSemanticVolumeTorch):
                                # Keep depth on torch device for torch backend.
                                Xc_t = Xc_any if isinstance(Xc_any, torch.Tensor) else torch.as_tensor(Xc_any)
                                Xc_t = Xc_t.reshape(int(h), int(w), 3).to(dtype=torch.float32, device=tsdf_vol.device)
                                depth_hw = Xc_t[..., 2]
                                valid_hw = torch.isfinite(depth_hw) & (depth_hw > 0.0)
                            else:
                                Xc = Xc_any
                                if isinstance(Xc, torch.Tensor):
                                    Xc = Xc.detach().to("cpu").to(torch.float32).numpy()
                                Xc = np.asarray(Xc, dtype=np.float32).reshape(int(h), int(w), 3)
                                depth_hw = Xc[..., 2].astype(np.float32, copy=False)
                                valid_hw = np.isfinite(depth_hw) & (depth_hw > 0.0)

                        if depth_hw is not None:
                            label_hw = None
                            frame_sem_id_debug: np.ndarray | None = None

                            # Optional: external per-frame semantic label stream.
                            #
                            # If the user provides `tsdf_frame_sem_dir`, we load semantic label-id maps from disk
                            # and use them as an independent semantic observation for TSDF fusion.
                            #
                            # This is especially useful for offline segmenters (e.g., InternImage) that produce
                            # better masks than the lightweight on-device segmenter.
                            if tsdf_frame_sem_dir is not None and str(tsdf_frame_sem_dir).strip() != "":
                                try:
                                    lab_np = _load_external_semantic_label_hw(
                                        sem_dir=str(tsdf_frame_sem_dir),
                                        pattern=str(tsdf_frame_sem_pattern),
                                        frame_id=fid,
                                        out_h=int(h),
                                        out_w=int(w),
                                    )
                                    frame_sem_id_debug = lab_np.astype(np.int32, copy=False)
                                    if bool(tsdf_use_semantic):
                                        if isinstance(tsdf_vol, RollingTsdfSemanticVolumeTorch):
                                            label_hw = torch.as_tensor(
                                                lab_np,
                                                device=tsdf_vol.device,
                                                dtype=torch.int64,
                                            )
                                        else:
                                            label_hw = lab_np
                                except Exception as e:
                                    # Robustness requirement: this is a debug/output feature only.
                                    # If the external semantic cannot be loaded, we fall back to in-process semantics.
                                    print(f"[PlanningTSDF] external semantic load failed (fid={fid}): {e!r}")
                            # IMPORTANT:
                            #   Always prefer RAW per-frame semantic for TSDF publish/debug.
                            #   The tracker may overwrite `curr.semantic_label` with stabilized labels (warp+fuse).
                            sem_src = getattr(curr, "semantic_label_raw", None)
                            if sem_src is None:
                                sem_src = getattr(curr, "semantic_label", None)

                            if label_hw is None and bool(tsdf_use_semantic) and (sem_src is not None):
                                sem = sem_src
                                if isinstance(tsdf_vol, RollingTsdfSemanticVolumeTorch):
                                    sem_t = sem if isinstance(sem, torch.Tensor) else torch.as_tensor(sem)
                                    sem_t = sem_t.reshape(int(h), int(w), 3).to(device=tsdf_vol.device)
                                    # Decode float01 RGB packed ids on-device:
                                    # code = (R<<16)|(G<<8)|B after scaling to uint8.
                                    maxv = float(sem_t.max().item()) if sem_t.numel() > 0 else 0.0
                                    if maxv <= 1.0 + 1e-6:
                                        sem_u8 = torch.clamp(sem_t * 255.0, 0.0, 255.0).to(torch.uint8)
                                    else:
                                        sem_u8 = torch.clamp(sem_t, 0.0, 255.0).to(torch.uint8)
                                    r = sem_u8[..., 0].to(torch.int32)
                                    g = sem_u8[..., 1].to(torch.int32)
                                    b = sem_u8[..., 2].to(torch.int32)
                                    label_hw = (r << 16) | (g << 8) | b
                                else:
                                    if isinstance(sem, torch.Tensor):
                                        sem = sem.detach().cpu().numpy()
                                    sem = np.asarray(sem).reshape(int(h), int(w), 3)
                                    # Decode the RGB-packed label into an integer id per pixel.
                                    label_hw = _rgb_to_label_id(sem).astype(np.int64, copy=False)

                            if isinstance(tsdf_vol, RollingTsdfSemanticVolumeTorch):
                                # Torch backend: keep math on-device, then snapshot to CPU for shm publish.
                                K_t = torch.as_tensor(K, device=tsdf_vol.device, dtype=torch.float32).reshape(3, 3)
                                T_t = torch.as_tensor(M_WCf, device=tsdf_vol.device, dtype=torch.float32).reshape(4, 4)
                                tsdf_vol.integrate(
                                    depth_hw=depth_hw,
                                    valid_hw=valid_hw,
                                    label_hw=label_hw,
                                    rgb_hw=rgb_u8,
                                    T_WC_f32=T_t,
                                    K_f32=K_t,
                                    obs_weight=1.0,
                                )
                                snap = tsdf_vol.snapshot_cpu()
                                frame_sem_id_cpu = frame_sem_id_debug
                                if frame_sem_id_cpu is None and label_hw is not None:
                                    if isinstance(label_hw, torch.Tensor):
                                        frame_sem_id_cpu = label_hw.detach().to("cpu").to(torch.int32).numpy()
                                    else:
                                        frame_sem_id_cpu = np.asarray(label_hw, dtype=np.int32)
                                tsdf_buf.write(
                                    frame_id=fid,
                                    timestamp_s=float(time.time()),
                                    origin_w_f32=snap.origin_w,
                                    voxel_m=float(snap.voxel_m),
                                    curr_T_WC_f32=np.asarray(M_WCf, dtype=np.float32).reshape(4, 4),
                                    tsdf_f32=snap.tsdf,
                                    weight_f32=snap.weight,
                                    sem_label_i32=snap.sem_label,
                                    sem_weight_f32=snap.sem_weight,
                                    rgb_color_f32=snap.rgb_color,
                                    rgb_weight_f32=snap.rgb_weight,
                                    frame_sem_id_i32=frame_sem_id_cpu,
                                    frame_rgb_u8=rgb_u8,
                                )
                                last_tsdf_frame_id = int(fid)
                                last_tsdf_timestamp_s = float(time.time())
                                last_tsdf_curr_T_WC = np.asarray(M_WCf, dtype=np.float32).reshape(4, 4).copy()
                            else:
                                tsdf_vol.integrate(
                                    depth_hw=depth_hw,
                                    valid_hw=valid_hw,
                                    label_hw=label_hw,
                                    rgb_hw=rgb_u8,
                                    T_WC_f32=np.asarray(M_WCf, dtype=np.float32).reshape(4, 4),
                                    K_f32=np.asarray(K, dtype=np.float32).reshape(3, 3),
                                    obs_weight=1.0,
                                )

                                snap = tsdf_vol.snapshot()
                                tsdf_buf.write(
                                    frame_id=fid,
                                    timestamp_s=float(time.time()),
                                    origin_w_f32=snap.origin_w,
                                    voxel_m=float(snap.voxel_m),
                                    curr_T_WC_f32=np.asarray(M_WCf, dtype=np.float32).reshape(4, 4),
                                    tsdf_f32=snap.tsdf,
                                    weight_f32=snap.weight,
                                    sem_label_i32=snap.sem_label,
                                    sem_weight_f32=snap.sem_weight,
                                    rgb_color_f32=snap.rgb_color,
                                    rgb_weight_f32=snap.rgb_weight,
                                    # Publish the per-frame semantic label ids for debug visualization.
                                    # This allows a TSDF viewer process to show the segmentation image
                                    # corresponding to the integrated frame, without re-running the
                                    # segmentation model and without disk I/O.
                                    frame_sem_id_i32=frame_sem_id_debug
                                    if frame_sem_id_debug is not None
                                    else (None if label_hw is None else label_hw.astype(np.int32, copy=False)),
                                    # Publish the per-frame RGB image so a viewer can display a 50/50
                                    # RGB+semantic overlay without needing any additional IPC channel.
                                    frame_rgb_u8=rgb_u8,
                                )
                                last_tsdf_frame_id = int(fid)
                                last_tsdf_timestamp_s = float(time.time())
                                last_tsdf_curr_T_WC = np.asarray(M_WCf, dtype=np.float32).reshape(4, 4).copy()
                    except Exception as e:
                        now_t = float(time.time())
                        if now_t - last_tsdf_err_print_t >= 2.0:
                            last_tsdf_err_print_t = now_t
                            print(f"[PlanningTSDF] publish failed: {e!r}")
                            print(traceback.format_exc())

                # If we are not publishing the keyframe map and not rendering overlays, we are done.
                # This is the "TSDF-only" mode and should stay as lightweight as possible.
                if (shm_buf is None) and (not do_overlay):
                    frame_counter += 1
                    now = time.time()
                    if now < next_t:
                        time.sleep(max(0.0, next_t - now))
                    next_t = max(next_t + period, time.time())
                    continue

                # Update/rebuild keyframe blocks.
                #
                # We maintain a cache of per-keyframe points in KEYFRAME camera coordinates.
                # This makes it cheap to reproject them into the current camera for visualization,
                # and it matches the keyframe-centric shared-memory publisher (poses can update later).
                try:
                    n_kf = int(len(keyframes))
                except Exception:
                    n_kf = 0

                start_idx = max(0, int(n_kf) - int(max_keyframes))

                # When shared-memory publishing is enabled, we also track which keyframes became "dirty"
                # (their fused pointmap / semantics changed). We update those keyframes' point blocks in shm.
                dirty_idx: list[int] = []
                if shm_buf is not None:
                    try:
                        di = keyframes.get_dirty_idx()
                        if isinstance(di, torch.Tensor):
                            dirty_idx = [int(x) for x in di.detach().cpu().tolist()]
                        else:
                            dirty_idx = [int(x) for x in list(di)]
                    except Exception:
                        dirty_idx = []

                # Rebuild blocks for dirty keyframes first (so publish + visualization see the latest).
                max_points_cap = int(shm_points_per_kf) if shm_buf is not None else None
                for kf_idx in sorted(set(dirty_idx)):
                    if kf_idx < 0 or kf_idx >= int(n_kf):
                        continue
                    # If we publish shm, we only care about keyframes within shm capacity.
                    if shm_buf is not None and kf_idx >= int(shm_max_keyframes):
                        continue
                    try:
                        kf = keyframes[kf_idx]
                    except Exception:
                        continue
                    blk = _build_keyframe_block(
                        keyframe=kf,
                        stride=int(stride),
                        conf_threshold=float(conf_threshold),
                        max_points=max_points_cap,
                    )
                    if blk is not None:
                        blocks[kf_idx] = blk
                    else:
                        # No valid points after filtering; drop the cache entry.
                        blocks.pop(kf_idx, None)

                # IMPORTANT (performance / correctness):
                #   The point-cloud reprojection overlay is DEBUG VISUALIZATION ONLY.
                #   When `do_overlay=False` (publish-only / headless), we must NOT:
                #     - convert RGB images
                #     - concatenate points across keyframes
                #     - run the z-buffer projection
                #
                #   Otherwise we waste a lot of CPU, and (worse) we can accidentally hit
                #   exceptions (e.g., rgb_u8=None) and fall into the outer retry loop without
                #   throttling, which destroys SLAM FPS.
                overlay = None
                depth_hw_m = None
                depth_valid_hw = None
                if do_overlay:
                    # Also rebuild blocks in the *visualization window* if we detect N_updates changed,
                    # or if the block was never built (e.g., GUI started late).
                    for kf_idx in range(start_idx, int(n_kf)):
                        cached = blocks.get(kf_idx, None)
                        try:
                            kf = keyframes[kf_idx]
                        except Exception:
                            continue
                        n_updates = int(getattr(kf, "N_updates", 0))
                        if cached is None or int(cached.n_updates) != n_updates:
                            blk = _build_keyframe_block(
                                keyframe=kf,
                                stride=int(stride),
                                conf_threshold=float(conf_threshold),
                                max_points=max_points_cap,
                            )
                            if blk is not None:
                                blocks[kf_idx] = blk

                    # Build a concatenated set of points in the current camera frame (for overlay).
                    X_all = []
                    col_all = []  # visualization-only colors
                    for kf_idx, blk in list(blocks.items()):
                        # Drop blocks that are far outside our window.
                        if kf_idx < start_idx:
                            # Keep cached blocks for shm publishing even if they are outside the visualization window.
                            if shm_buf is None or kf_idx >= int(shm_max_keyframes):
                                del blocks[kf_idx]
                            continue
                        try:
                            kf = keyframes[kf_idx]
                        except Exception:
                            continue
                        M_WCk = _sim3_matrix(kf.T_WC)  # keyframe camera -> world
                        # Transform keyframe camera points to current camera:
                        #   X_Cf = (T_CWf * T_WCk) * X_k
                        M = M_CWf @ M_WCk  # 4x4
                        R = M[:3, :3]
                        t = M[:3, 3]
                        Xk = blk.Xk
                        Xcf = (Xk @ R.T) + t[None, :]
                        X_all.append(Xcf.astype(np.float32, copy=False))
                        col_all.append(blk.rgb_sem.astype(np.uint8, copy=False))

                    if len(X_all) == 0:
                        # Fall back to raw RGB background (if available).
                        overlay = rgb_u8
                        depth_hw_m = np.full((int(h), int(w)), np.nan, dtype=np.float32)
                        depth_valid_hw = np.zeros((int(h), int(w)), dtype=bool)
                    else:
                        X_cat = np.concatenate(X_all, axis=0)
                        col_cat = np.concatenate(col_all, axis=0)
                        overlay, depth_hw_m, depth_valid_hw = _project_points_to_image(
                            X_cam=X_cat,
                            rgb_u8=rgb_u8,
                            K=K,
                            colors_u8=col_cat,
                        )

                # Publish the latest keyframe-centric map for downstream planning (optional).
                #
                # IMPORTANT:
                #   - Publishing must never crash SLAM. Errors are swallowed.
                #   - We update POINT blocks only for dirty keyframes (large arrays).
                #   - We update POSE blocks every tick (small arrays), so consumers can always
                #     render using the latest loop-optimized keyframe poses.
                if shm_buf is not None:
                    try:
                        # 1) Update point blocks for dirty keyframes (within shm capacity).
                        for kf_idx in sorted(set(dirty_idx)):
                            if kf_idx < 0 or kf_idx >= int(n_kf):
                                continue
                            if kf_idx >= int(shm_max_keyframes):
                                continue
                            blk = blocks.get(kf_idx, None)
                            if blk is None:
                                # No valid points: publish empty.
                                shm_buf.update_keyframe_points(
                                    kf_slot=int(kf_idx),
                                    points_k_f32=np.zeros((0, 3), dtype=np.float32),
                                    label_id_i32=np.zeros((0,), dtype=np.int32),
                                    rgb_u8=None,
                                    n_points=0,
                                )
                                continue
                            shm_buf.update_keyframe_points(
                                kf_slot=int(kf_idx),
                                points_k_f32=blk.Xk,
                                label_id_i32=blk.label_id,
                                rgb_u8=blk.rgb_u8,
                                n_points=int(blk.Xk.shape[0]),
                            )

                        # 2) Publish current pose + keyframe poses.
                        #
                        # PERFORMANCE CRITICAL:
                        #   Do NOT loop over keyframes and call `.item()` / `.cpu()` for each one.
                        #   That pattern forces *many* small CUDA synchronizations and can destroy
                        #   the SLAM FPS (even when TSDF is disabled).
                        #
                        # Instead, we batch-convert:
                        #   - Sim3 -> 4x4 matrix for all keyframes in one CUDA op
                        #   - copy the whole (K,4,4) block back to CPU once
                        #   - copy dataset_idx back to CPU once
                        kf_count_pub = int(min(int(n_kf), int(shm_max_keyframes)))
                        if kf_count_pub > 0:
                            # Batch Sim3 -> matrix
                            try:
                                # `keyframes.T_WC` is a shared CUDA tensor of shape (K, 1, Sim3.embedded_dim).
                                # `lietorch.Sim3(...).matrix()` returns (K, 1, 4, 4).
                                kf_T_t = lietorch.Sim3(keyframes.T_WC[:kf_count_pub]).matrix()
                                if isinstance(kf_T_t, torch.Tensor):
                                    kf_T = (
                                        kf_T_t[:, 0]
                                        .detach()
                                        .to("cpu")
                                        .to(torch.float32)
                                        .numpy()
                                        .astype(np.float32, copy=False)
                                    )
                                else:
                                    kf_T = np.asarray(kf_T_t, dtype=np.float32).reshape(kf_count_pub, 1, 4, 4)[:, 0]
                            except Exception:
                                # Best-effort fallback: publish identity transforms.
                                kf_T = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (kf_count_pub, 1, 1))

                            # Batch dataset_idx -> CPU
                            try:
                                fid_t = keyframes.dataset_idx[:kf_count_pub]
                                if isinstance(fid_t, torch.Tensor):
                                    kf_fid = fid_t.detach().to("cpu").to(torch.int32).numpy().astype(np.int32, copy=False)
                                else:
                                    kf_fid = np.asarray(fid_t, dtype=np.int32).reshape(kf_count_pub)
                            except Exception:
                                kf_fid = np.full((kf_count_pub,), -1, dtype=np.int32)
                        else:
                            kf_T = np.zeros((0, 4, 4), dtype=np.float32)
                            kf_fid = np.zeros((0,), dtype=np.int32)

                        shm_buf.write(
                            frame_id=int(getattr(curr, "frame_id", frame_counter)),
                            timestamp_s=float(time.time()),
                            curr_T_WC_f32=np.asarray(M_WCf, dtype=np.float32).reshape(4, 4),
                            kf_T_WC_f32=kf_T,
                            kf_frame_id_i32=kf_fid,
                            kf_count=int(kf_count_pub),
                        )
                    except Exception:
                        pass

                # -----------------------------------------------------------------
                # Overlay rendering (debug-only). Never run in publish-only mode.
                # -----------------------------------------------------------------
                if do_overlay and overlay is not None and depth_hw_m is not None and depth_valid_hw is not None:
                    # Depth statistics overlay (debug-only).
                    try:
                        grid = 3
                        patch = 3
                        centers = _compute_grid_centers(int(h), int(w), grid)

                        # Draw patch boxes in red so you can visually correlate the sampled region.
                        p = int(max(1, patch))
                        if p % 2 == 0:
                            p += 1
                        r = p // 2
                        for (cy, cx) in centers:
                            x0, y0 = int(cx - r), int(cy - r)
                            x1, y1 = int(cx + r), int(cy + r)
                            cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 1)

                        means = [
                            _patch_mean_depth(depth_hw_m, depth_valid_hw, cy=int(cy), cx=int(cx), patch=patch)
                            for (cy, cx) in centers
                        ]

                        # Print means as a compact list (small font).
                        y_txt = 38
                        for i, v in enumerate(means[:16]):
                            txt = f"p{i:02d}: {v:5.2f}m" if np.isfinite(v) else f"p{i:02d}:  nan"
                            cv2.putText(
                                overlay,
                                txt,
                                (10, y_txt),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                overlay,
                                txt,
                                (10, y_txt),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (0, 0, 0),
                                1,
                                cv2.LINE_AA,
                            )
                            y_txt += 14
                    except Exception:
                        # Depth overlay must never crash the process.
                        pass

                    # Add simple pose text (planning debugging).
                    try:
                        # Display the same pose used for reprojection (filtered or raw).
                        t_wc = M_WCf[:3, 3]
                        cv2.putText(
                            overlay,
                            f"t=({t_wc[0]:.2f},{t_wc[1]:.2f},{t_wc[2]:.2f})",
                            (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            overlay,
                            f"t=({t_wc[0]:.2f},{t_wc[1]:.2f},{t_wc[2]:.2f})",
                            (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    except Exception:
                        pass

                    # Display and/or save.
                    if show_window:
                        cv2.imshow(window_name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                        k = cv2.waitKey(1) & 0xFF
                        if k in (27, ord("q")):
                            break

                    if save_images:
                        out_path = os.path.join(out_dir, f"frame_{frame_counter:06d}.png")
                        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                frame_counter += 1

                # Throttle to target FPS.
                now = time.time()
                if now < next_t:
                    time.sleep(max(0.0, next_t - now))
                next_t = max(next_t + period, time.time())
            except Exception:
                # Never crash SLAM due to auxiliary visualization/publishing. Keep running.
                #
                # HOWEVER:
                #   If we are shutting down (SIGTERM/SIGINT), do not swallow the exception and loop
                #   forever; break so `finally` can clean up shared memory blocks.
                if bool(_should_stop["stop"]):
                    break
                time.sleep(0.01)
                continue
    finally:
        if tsdf_vol is not None and last_tsdf_frame_id >= 0:
            _auto_save_final_esdf_snapshot(
                out_dir=str(out_dir),
                tsdf_vol=tsdf_vol,
                frame_id=int(last_tsdf_frame_id),
                timestamp_s=float(last_tsdf_timestamp_s),
                curr_T_WC_f32=last_tsdf_curr_T_WC,
                use_semantic=bool(tsdf_use_semantic),
                w_min=1.0,
                sem_w_min=1.0,
                dilate_iters=1,
            )
        if show_window:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass
        if shm_buf is not None:
            # Best-effort cleanup. Unlinking is important so new runs don't collide with old names.
            try:
                shm_buf.close(unlink=True)
            except Exception:
                pass
        if tsdf_buf is not None:
            try:
                tsdf_buf.close(unlink=True)
            except Exception:
                pass
