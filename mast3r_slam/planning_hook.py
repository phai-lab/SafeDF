"""
In-process "hook" for downstream planning to access pose + semantic point cloud.

Why this exists
---------------
The user requested a *single obvious place* ("hook") inside `main_semantic.py` where:
  - the current pose is already available, and
  - a semantic point cloud can be extracted from the current keyframe set.

This module provides a minimal in-process API:
  - `update_latest_planning_snapshot(...)` computes a snapshot and stores it in a module-global.
  - Any other code running in the SAME Python process can import this module and read
    `LATEST_PLANNING_SNAPSHOT` without dealing with shared memory / JSON / IPC.

Important limitations
---------------------
If your planner runs in a DIFFERENT OS process, it cannot directly access Python variables.
In that case you must use an IPC mechanism (e.g., the shared-memory publisher implemented
in `mast3r_slam/planning_pointcloud_buffer.py`).

Performance notes
-----------------
Building a point cloud from many keyframes can be expensive. This hook is intentionally:
  - opt-in (only runs when called), and
  - configurable via stride/max_keyframes to trade accuracy for speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import lietorch


def _sim3_matrix(T: lietorch.Sim3) -> np.ndarray:
    """
    Convert a lietorch Sim3 into a 4x4 float32 numpy matrix.

    The returned matrix maps points from camera frame to world frame:
      X_world = T_WC * X_cam
    """

    m = T.matrix()
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 3:
        m = m[0]
    return m


def _rgb_to_label_id(rgb_hw3: np.ndarray) -> np.ndarray:
    """
    Convert an RGB mask to a label-id map using a palette-agnostic 24-bit packing:
      id = (R << 16) | (G << 8) | B

    Notes:
      - If the RGB mask is bit-packed (as used in this repo for hard labels), this exactly recovers ids.
      - If the RGB mask is palette-based, this yields a deterministic id per color.
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


@dataclass
class PlanningSnapshot:
    """
    A single planning snapshot derived from SLAM state (in-process).

    Shapes:
      - T_WC      : (4,4) float32, camera->world pose matrix (current frame)
      - points_w  : (N,3) float32, semantic point cloud in world coordinates
      - colors_u8 : (N,3) uint8, per-point semantic colors aligned with points_w
    """

    frame_id: int
    T_WC: np.ndarray
    points_w: np.ndarray
    colors_u8: np.ndarray


# Module-global latest snapshot. This is intentionally a simple "hook variable":
# any code running in the SAME Python process can read it.
LATEST_PLANNING_SNAPSHOT: Optional[PlanningSnapshot] = None


def update_latest_planning_snapshot(
    *,
    frame,
    keyframes,
    max_keyframes: int = 30,
    stride: int = 4,
    conf_threshold: float = 1.5,
) -> PlanningSnapshot:
    """
    Compute and store the latest planning snapshot.

    Inputs:
      frame: current Frame (must have `frame_id` and `T_WC`)
      keyframes: SharedKeyframes (or a list-like of keyframe Frames)
      max_keyframes: include only the most recent K keyframes
      stride: downsample stride on the (H,W) pointmap grid (1 = no downsample)
      conf_threshold: minimum average confidence to keep a point (<=0 disables)

    Output:
      The computed PlanningSnapshot (also stored in `LATEST_PLANNING_SNAPSHOT`).
    """

    # Current pose.
    T_WC = _sim3_matrix(frame.T_WC)

    # Iterate recent keyframes and accumulate world points + semantic colors.
    try:
        n_kf = len(keyframes)
    except Exception:
        n_kf = 0
    start_idx = max(0, int(n_kf) - int(max_keyframes))

    pts_all: list[np.ndarray] = []
    col_all: list[np.ndarray] = []
    s = int(max(1, stride))

    for kf_idx in range(start_idx, int(n_kf)):
        try:
            kf = keyframes[kf_idx]
        except Exception:
            continue

        # Required geometry.
        if getattr(kf, "X_canon", None) is None:
            continue
        try:
            h, w = (int(x) for x in kf.img_shape.flatten().tolist())
        except Exception:
            continue
        if h <= 0 or w <= 0:
            continue

        X = kf.X_canon
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float32).reshape(h, w, 3)

        # Confidence gating (optional).
        try:
            C = kf.get_average_conf()
            if isinstance(C, torch.Tensor):
                C = C.detach().cpu().numpy()
            C = np.asarray(C, dtype=np.float32).reshape(h, w)
        except Exception:
            C = None

        Xs = X[::s, ::s].reshape(-1, 3)
        if C is not None:
            Cs = C[::s, ::s].reshape(-1)
            valid = np.isfinite(Xs).all(axis=1) & (Xs[:, 2] > 0.0) & np.isfinite(Cs)
            if conf_threshold > 0:
                valid = valid & (Cs >= float(conf_threshold))
        else:
            valid = np.isfinite(Xs).all(axis=1) & (Xs[:, 2] > 0.0)

        if not np.any(valid):
            continue
        Xs = Xs[valid]

        # Semantic colors (optional). If missing, use a constant cyan.
        rgb_sem_u8 = None
        if hasattr(kf, "semantic_label") and kf.semantic_label is not None:
            sem = kf.semantic_label
            if isinstance(sem, torch.Tensor):
                sem = sem.detach().cpu().numpy()
            sem = np.asarray(sem).reshape(h, w, 3)
            sem_ds = sem[::s, ::s]
            try:
                lab = _rgb_to_label_id(sem_ds)
                rgb_sem_u8 = _hash_colorize_label_id(lab).reshape(-1, 3)[valid]
            except Exception:
                rgb_sem_u8 = None
        if rgb_sem_u8 is None:
            rgb_sem_u8 = np.full((Xs.shape[0], 3), (0, 255, 255), dtype=np.uint8)

        # World transform.
        M_WCk = _sim3_matrix(kf.T_WC)
        Rwk = M_WCk[:3, :3]
        twk = M_WCk[:3, 3]
        Xw = (Xs @ Rwk.T) + twk[None, :]

        pts_all.append(Xw.astype(np.float32, copy=False))
        col_all.append(rgb_sem_u8.astype(np.uint8, copy=False))

    if len(pts_all) == 0:
        points_w = np.zeros((0, 3), dtype=np.float32)
        colors_u8 = np.zeros((0, 3), dtype=np.uint8)
    else:
        points_w = np.concatenate(pts_all, axis=0)
        colors_u8 = np.concatenate(col_all, axis=0)

    snap = PlanningSnapshot(
        frame_id=int(getattr(frame, "frame_id", -1)),
        T_WC=T_WC,
        points_w=points_w,
        colors_u8=colors_u8,
    )

    global LATEST_PLANNING_SNAPSHOT
    LATEST_PLANNING_SNAPSHOT = snap
    return snap

