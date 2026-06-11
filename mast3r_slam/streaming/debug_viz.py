"""
Streaming debug visualization (separate process).

This process displays (debug-focused and lightweight):
  - A 3-column view (requested for stability debugging):
      (left)  RGB + RAW semantic overlay (per-frame EfficientViT labels)
      (mid)   RGB + STABLE semantic overlay (post warp+fuse labels)
      (right) RGB + depth overlay (RAW or STABLE depth selectable)
    Depth is shown as a colormap blended over RGB for context.

  - A compact text readout of depth statistics sampled from a fixed grid (default 3x3 points),
    drawn on top of the depth panel. This is intentionally more stable than a single global min.

Optional (debug-only):
  - Multiple depth filtering modes can be selected (visualization only):
      - none  : no filtering
      - pixel : per-pixel random-walk Kalman filter on the depth map
      - pose  : pose-aware pixel Kalman. We still filter per-pixel, but we RESET the filter
               when the camera pose changes beyond a small threshold. This avoids the classic
               pitfall of pixel-temporal filtering during camera motion, where (u,v) no longer
               refers to the same physical point.
    IMPORTANT: Filtering here MUST NOT affect SLAM optimization or gating. We only change what
    is displayed in the GUI.

Why a separate process:
  - OpenCV GUI calls (imshow/waitKey) can block unpredictably.
  - We do not want GUI latency to slow down SLAM tracking.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Planning semantic point cloud consumer (shared-memory based).
#
# IMPORTANT:
#   The publisher now uses a keyframe-centric shared-memory schema (official-visualization-like):
#     - points are stored in keyframe camera coordinates
#     - keyframe poses can update at runtime (loop closure / backend)
#   The debug GUI reprojects points using the latest poses, without requiring the publisher to
#   recompute world-space points every GUI frame.
from mast3r_slam.planning_pointcloud_client import PlanningKeyframeMapClient
try:
    # SciPy is used for a fast 3D Euclidean distance transform (ESDF).
    # If unavailable, ESDF visualization MUST fail fast when explicitly requested.
    import scipy.ndimage as _ndi
except Exception:
    _ndi = None


class _PerPixelHardLabelEMA:
    """
    Extremely lightweight per-pixel temporal filter for hard labels (debug-only).

    Why this exists
    ---------------
    The user requested "semantic smoothing" in the debug visualization so the displayed
    segmentation does not flicker as much. A true "softmax probability EMA" would require
    per-pixel logits/probabilities (C channels) from the segmentation network, which we do
    NOT have in the SLAM main loop (and we explicitly want to avoid heavy data plumbing).

    Instead, we implement a minimal approximation that behaves like a 1D probability tracker
    per pixel while storing only TWO values per pixel:
      - current_label: int64 (the current best label)
      - weight: float32 (how confident we are in current_label vs alternatives)

    Update rule (vectorized, O(H*W))
    -------------------------------
    For each pixel p with observation y_t(p):
      w <- mu * w
      if y_t == label: w <- w + u
      else:           w <- w - u
      if w < 0:
        label <- y_t
        w <- -w

    Interpretation:
      - w behaves like a "margin" for the current label. Occasional noisy flips do not
        immediately change the displayed label when w is large.
      - mu < 1 introduces a forgetting factor, making the filter adapt over time.

    IMPORTANT
    ---------
    This filter is used ONLY for debug visualization. It must never affect SLAM tracking,
    matching, keyframe decisions, or any optimization.
    """

    def __init__(self, *, momentum: float, u: float) -> None:
        self.momentum = float(np.clip(momentum, 0.0, 1.0))
        self.u = float(max(u, 1e-6))
        self.label_hw: np.ndarray | None = None  # (H,W) int64
        self.weight_hw: np.ndarray | None = None  # (H,W) float32

    def reset(self) -> None:
        self.label_hw = None
        self.weight_hw = None

    def ensure_shape(self, h: int, w: int) -> None:
        if self.label_hw is None or self.weight_hw is None or self.label_hw.shape != (h, w):
            self.label_hw = np.zeros((h, w), dtype=np.int64)
            self.weight_hw = np.ones((h, w), dtype=np.float32)

    def step(self, obs_label_hw: np.ndarray) -> np.ndarray:
        """
        Update the filter with a new hard-label observation.

        Input:
          obs_label_hw: (H,W) int64 (or castable to int64)

        Output:
          filt_label_hw: (H,W) int64 filtered label map
        """

        y = np.asarray(obs_label_hw, dtype=np.int64)
        h, w = int(y.shape[0]), int(y.shape[1])
        self.ensure_shape(h, w)
        assert self.label_hw is not None and self.weight_hw is not None

        # First frame: initialize labels directly.
        if not np.any(self.weight_hw):  # safety; shouldn't happen
            self.label_hw[...] = y
            self.weight_hw.fill(1.0)
            return self.label_hw

        # Predict: decay the existing margin.
        if self.momentum < 1.0:
            self.weight_hw *= self.momentum

        same = (y == self.label_hw)
        # Update: +u for agreement, -u for disagreement.
        self.weight_hw[same] += self.u
        self.weight_hw[~same] -= self.u

        # Flip if disagreement overwhelms the stored margin.
        flip = self.weight_hw < 0.0
        if np.any(flip):
            self.label_hw[flip] = y[flip]
            self.weight_hw[flip] = -self.weight_hw[flip]

        return self.label_hw


def _alpha_blend_rgb(base_rgb_u8: np.ndarray, overlay_rgb_u8: np.ndarray, alpha: float) -> np.ndarray:
    """
    Alpha blend two RGB uint8 images.

    Inputs:
      base_rgb_u8:    (H,W,3) uint8 in RGB order
      overlay_rgb_u8: (H,W,3) uint8 in RGB order
      alpha:          overlay weight in [0,1]

    Output:
      blended_rgb_u8: (H,W,3) uint8 in RGB order
    """

    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 0.0:
        return base_rgb_u8
    if a >= 1.0:
        return overlay_rgb_u8
    out = (1.0 - a) * base_rgb_u8.astype(np.float32) + a * overlay_rgb_u8.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _hash_colorize(label_hw: np.ndarray) -> np.ndarray:
    """
    Deterministic pseudo-color map for integer labels (debug only).

    Input:
      label_hw: (H,W) int64/int32
    Output:
      rgb_u8: (H,W,3) uint8
    """

    # IMPORTANT:
    #   In some visualization paths (e.g., pointcloud reprojection), we use -1 as a sentinel
    #   for "no valid label / no point projected to this pixel". If we directly cast -1 to
    #   uint32, it becomes 0xFFFFFFFF and produces a very strong (often reddish) color.
    #
    #   That looks like the background "turns red", which is confusing and not intended.
    #   Therefore, we explicitly mask negative labels and render them as black.
    lab = np.asarray(label_hw)
    known = lab >= 0
    v = np.where(known, lab, 0).astype(np.uint32, copy=False)
    r = (v * 37 + 17) & 255
    g = (v * 17 + 59) & 255
    b = (v * 97 + 101) & 255
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    rgb[~known] = 0
    return rgb


def _approx_intrinsics(h: int, w: int, fov_deg: float) -> np.ndarray:
    """
    Build a synthetic pinhole intrinsics matrix K for visualization projection.

    IMPORTANT:
      - This is used ONLY inside the debug visualization process.
      - It must not affect SLAM, tracking, matching, or any optimization.
      - For the point-cloud reprojection panels, we do not have true camera intrinsics,
        so we assume a reasonable FOV to visualize the surroundings.

    Inputs:
      h,w: output image size
      fov_deg: assumed horizontal field-of-view (degrees)

    Output:
      K: (3,3) float32
    """

    fov = float(fov_deg) * (np.pi / 180.0)
    fx = 0.5 * float(w) / max(1e-6, np.tan(0.5 * fov))
    fy = fx
    cx = 0.5 * (float(w) - 1.0)
    cy = 0.5 * (float(h) - 1.0)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _draw_pose_overlay_rgb(
    img_rgb: np.ndarray,
    *,
    frame_id: int,
    T_WC: np.ndarray,
    last_t_wc: np.ndarray | None,
    font_scale: float,
) -> np.ndarray:
    """
    Draw pose translation and translation delta on an RGB uint8 image.

    Displayed values:
      - t_wc : camera translation in world coordinates (meters)
      - |dt| : ||t_wc(current) - t_wc(previous)|| (meters)

    Why this exists:
      - Users want to *see* pose jitter directly in the window, not only in stdout.
      - This is a debug-only overlay and must not affect SLAM.
    """

    out = np.array(img_rgb, copy=True)
    M = np.asarray(T_WC, dtype=np.float32).reshape(4, 4)
    t = M[:3, 3].astype(np.float32)
    if last_t_wc is None:
        dt = float("nan")
    else:
        dt = float(np.linalg.norm(t - np.asarray(last_t_wc, dtype=np.float32).reshape(3)))

    lines = [
        f"frame_id: {int(frame_id)}",
        f"t_wc: x={t[0]:+.3f}  y={t[1]:+.3f}  z={t[2]:+.3f}  (m)",
        f"|dt|: {dt:.4f} m",
    ]
    x0, y0 = 10, 20
    dy = int(max(12, round(28 * float(font_scale))))
    for i, text in enumerate(lines):
        y = y0 + i * dy
        # Draw twice (black then white) for readability on arbitrary backgrounds.
        cv2.putText(out, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _project_triplet_from_pointcloud_rgb(
    *,
    points_w: np.ndarray,
    rgb_pts_u8: np.ndarray | None,
    label_id: np.ndarray,
    T_WC: np.ndarray,
    out_h: int,
    out_w: int,
    fov_deg: float,
    max_points: int,
    depth_max_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render (RGB | Depth | Semantics) by reprojecting a *global* point cloud using the current pose.

    IMPORTANT (matches user requirement):
      - We do NOT display any directly transmitted camera RGB frame.
      - The RGB panel is synthesized by projecting per-point RGB (if available).
      - Depth/Semantics use the SAME nearest-point z-buffer (no sorting; O(N) np.minimum.at).

    Outputs are RGB uint8 images (H,W,3).
    """

    H, W = int(out_h), int(out_w)
    pts = np.asarray(points_w, dtype=np.float32).reshape(-1, 3)
    lab = np.asarray(label_id, dtype=np.int32).reshape(-1)
    rgb = None if rgb_pts_u8 is None else np.asarray(rgb_pts_u8, dtype=np.uint8).reshape(-1, 3)

    blank = np.zeros((H, W, 3), dtype=np.uint8)
    if pts.shape[0] == 0:
        return blank, blank, blank

    # Optional subsampling for speed (keep arrays aligned).
    #
    # IMPORTANT (performance / determinism):
    #   This function is called every GUI frame. Doing `rng.choice(..., replace=False)` is
    #   surprisingly expensive for large N because it needs to generate a permutation-like
    #   structure. That can easily dominate the CPU time.
    #
    #   We therefore use a deterministic *stride* subsampling that is O(m) and allocation-light:
    #     sel = [0, stride, 2*stride, ...]  (clipped to m samples)
    #
    #   For visualization, this is good enough and keeps latency low.
    m = int(max_points)
    if m > 0 and pts.shape[0] > m:
        n = int(pts.shape[0])
        stride = max(1, n // m)
        sel = np.arange(0, n, stride, dtype=np.int64)[:m]
        pts = pts[sel]
        lab = lab[sel]
        if rgb is not None:
            rgb = rgb[sel]

    # World -> camera.
    #
    # IMPORTANT (performance):
    #   Do NOT call `np.linalg.inv()` on a 4x4 matrix per frame if we can avoid it.
    #   For rigid transforms:
    #     Xw = R_wc * Xc + t_wc
    #   the inverse is:
    #     Xc = R_wc^T * (Xw - t_wc)
    #
    #   We implement this directly for speed and numerical stability.
    M_WC = np.asarray(T_WC, dtype=np.float32).reshape(4, 4)
    R_wc = M_WC[:3, :3]
    t_wc = M_WC[:3, 3]
    # Using row-vector convention: Xc_row = (Xw_row - t_wc_row) @ R_wc
    Xc = (pts - t_wc[None, :]) @ R_wc
    z = Xc[:, 2]
    valid = np.isfinite(Xc).all(axis=1) & np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return blank, blank, blank
    Xc = Xc[valid]
    z = z[valid]
    lab = lab[valid]
    if rgb is not None:
        rgb = rgb[valid]

    # Project with assumed intrinsics.
    K = _approx_intrinsics(H, W, float(fov_deg))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = (fx * (Xc[:, 0] / z) + cx).astype(np.int32)
    v = (fy * (Xc[:, 1] / z) + cy).astype(np.int32)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & np.isfinite(z)
    if not np.any(inside):
        return blank, blank, blank
    u = u[inside]
    v = v[inside]
    z = z[inside]
    lab = lab[inside]
    if rgb is not None:
        rgb = rgb[inside]

    # O(N) z-buffer without sorting:
    #   key = (z_int << 32) | idx
    pix = (v.astype(np.int64) * int(W) + u.astype(np.int64)).astype(np.int64)
    idx = np.arange(z.shape[0], dtype=np.uint64)
    z_int = np.clip(z * 10000.0, 0.0, float(2**31 - 1)).astype(np.uint64)
    key = (z_int << 32) | idx

    sentinel = np.uint64(0xFFFFFFFFFFFFFFFF)
    best_key = np.full((H * W,), sentinel, dtype=np.uint64)
    np.minimum.at(best_key, pix, key)

    chosen = best_key != sentinel
    chosen_flat = np.where(chosen)[0]
    if chosen_flat.size == 0:
        return blank, blank, blank
    best_idx = (best_key[chosen_flat] & np.uint64(0xFFFFFFFF)).astype(np.int64)

    # Depth (z).
    depth_flat = np.full((H * W,), np.nan, dtype=np.float32)
    depth_flat[chosen_flat] = z[best_idx].astype(np.float32)
    depth_hw = depth_flat.reshape(H, W)
    valid_hw = chosen.reshape(H, W)
    depth_rgb = _depth_to_colormap_rgb(depth_hw, valid_hw, max_depth_m=float(depth_max_m))

    # RGB-like panel (from per-point RGB).
    rgb_flat = np.zeros((H * W, 3), dtype=np.uint8)
    if rgb is not None:
        rgb_flat[chosen_flat] = rgb[best_idx]
    rgb_img = rgb_flat.reshape(H, W, 3)

    # Semantics panel (from nearest label_id).
    sem_label_flat = np.full((H * W,), np.int32(-1), dtype=np.int32)
    sem_label_flat[chosen_flat] = lab[best_idx].astype(np.int32)
    sem_rgb = _hash_colorize(sem_label_flat.reshape(H, W).astype(np.int64))

    return rgb_img, depth_rgb, sem_rgb


def _gather_points_in_current_camera_from_kf_map(
    *,
    curr_T_WC: np.ndarray,
    kf_T_WC: np.ndarray,
    kf_points_k: np.ndarray,
    kf_n_points: np.ndarray,
    kf_label_id: np.ndarray,
    kf_rgb_u8: np.ndarray | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Gather a capped set of points from a keyframe-centric map and transform them into the CURRENT camera frame.

    This is the core "official-style" trick:
      - points are stored in each keyframe's camera frame
      - we compose transforms on the consumer side:
          X_Cf = (T_CWf * T_WCk) * X_k

    Inputs
    ------
    curr_T_WC:
      (4,4) float32 camera->world of the current frame.
    kf_T_WC:
      (K,4,4) float32 camera->world for each keyframe (can update due to loop closure).
    kf_points_k:
      (K,P,3) float32 keyframe-local points (camera frame).
    kf_n_points:
      (K,) int32 valid point counts per keyframe.
    kf_label_id:
      (K,P) int32 label ids per point.
    kf_rgb_u8:
      (K,P,3) uint8 per-point appearance color (optional).
    max_points:
      Total cap across all keyframes for visualization speed.

    Outputs
    -------
    Xc:
      (N,3) float32 points in the current camera frame.
    lab:
      (N,) int32 label ids aligned with Xc.
    rgb:
      (N,3) uint8 RGB aligned with Xc (or None).

    Performance notes
    -----------------
    - We avoid any per-frame sorting.
    - We sample per keyframe using a deterministic stride (allocation-light).
    - We iterate keyframes from newest to oldest and stop once the budget is filled.
    """

    M_WCf = np.asarray(curr_T_WC, dtype=np.float32).reshape(4, 4)
    M_CWf = np.linalg.inv(M_WCf).astype(np.float32, copy=False)

    kf_T = np.asarray(kf_T_WC, dtype=np.float32).reshape(-1, 4, 4)
    K = int(kf_T.shape[0])
    if K <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32), None

    P = np.asarray(kf_points_k, dtype=np.float32).reshape(K, -1, 3)
    npts = np.asarray(kf_n_points, dtype=np.int32).reshape(K)
    lab_k = np.asarray(kf_label_id, dtype=np.int32).reshape(K, -1)
    rgb_k = None if kf_rgb_u8 is None else np.asarray(kf_rgb_u8, dtype=np.uint8).reshape(K, -1, 3)

    budget = int(max(0, max_points))
    if budget <= 0:
        budget = 10_000

    Xc_all: list[np.ndarray] = []
    lab_all: list[np.ndarray] = []
    rgb_all: list[np.ndarray] = []
    have_rgb = rgb_k is not None

    remaining = budget
    # Iterate newest->oldest to emphasize the local neighborhood for planning/debug.
    for kfi in range(K - 1, -1, -1):
        if remaining <= 0:
            break
        n = int(npts[kfi])
        if n <= 0:
            continue

        # Compute a per-keyframe quota to spread points across remaining keyframes.
        rem_kf = int(kfi + 1)
        quota = max(1, remaining // max(1, rem_kf))
        quota = min(quota, n, remaining)

        step = max(1, n // max(1, quota))
        sel = np.arange(0, n, step, dtype=np.int64)[:quota]

        Xk = P[kfi, sel, :]
        lab = lab_k[kfi, sel]
        if have_rgb:
            rgb = rgb_k[kfi, sel, :]

        # Keyframe camera -> current camera:
        #   X_Cf = (T_CWf * T_WCk) * X_k
        M = (M_CWf @ kf_T[kfi]).astype(np.float32, copy=False)
        A = M[:3, :3]
        t = M[:3, 3]
        Xc = (Xk @ A.T) + t[None, :]

        Xc_all.append(Xc.astype(np.float32, copy=False))
        lab_all.append(lab.astype(np.int32, copy=False))
        if have_rgb:
            rgb_all.append(rgb.astype(np.uint8, copy=False))

        remaining -= int(Xc.shape[0])

    if len(Xc_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32), None

    Xc_cat = np.concatenate(Xc_all, axis=0)
    lab_cat = np.concatenate(lab_all, axis=0)
    rgb_cat = None if not have_rgb else np.concatenate(rgb_all, axis=0)
    return Xc_cat, lab_cat, rgb_cat


def _project_triplet_from_kf_map(
    *,
    curr_T_WC: np.ndarray,
    kf_T_WC: np.ndarray,
    kf_points_k: np.ndarray,
    kf_n_points: np.ndarray,
    kf_label_id: np.ndarray,
    kf_rgb_u8: np.ndarray | None,
    out_h: int,
    out_w: int,
    fov_deg: float,
    max_points: int,
    depth_max_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render (RGB | Depth | Semantics) by reprojecting a KEYFRAME-CENTRIC map using the current pose.

    This is the keyframe-centric equivalent of `_project_triplet_from_pointcloud_rgb`.
    """

    H, W = int(out_h), int(out_w)
    blank = np.zeros((H, W, 3), dtype=np.uint8)

    Xc, lab, rgb = _gather_points_in_current_camera_from_kf_map(
        curr_T_WC=curr_T_WC,
        kf_T_WC=kf_T_WC,
        kf_points_k=kf_points_k,
        kf_n_points=kf_n_points,
        kf_label_id=kf_label_id,
        kf_rgb_u8=kf_rgb_u8,
        max_points=int(max_points),
    )
    if Xc.shape[0] == 0:
        return blank, blank, blank

    z = Xc[:, 2]
    valid = np.isfinite(Xc).all(axis=1) & np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return blank, blank, blank
    Xc = Xc[valid]
    z = z[valid]
    lab = lab[valid]
    if rgb is not None:
        rgb = rgb[valid]

    K = _approx_intrinsics(H, W, float(fov_deg))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = (fx * (Xc[:, 0] / z) + cx).astype(np.int32)
    v = (fy * (Xc[:, 1] / z) + cy).astype(np.int32)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & np.isfinite(z)
    if not np.any(inside):
        return blank, blank, blank
    u = u[inside]
    v = v[inside]
    z = z[inside]
    lab = lab[inside]
    if rgb is not None:
        rgb = rgb[inside]

    pix = (v.astype(np.int64) * int(W) + u.astype(np.int64)).astype(np.int64)
    idx = np.arange(z.shape[0], dtype=np.uint64)
    z_int = np.clip(z * 10000.0, 0.0, float(2**31 - 1)).astype(np.uint64)
    key = (z_int << 32) | idx
    sentinel = np.uint64(0xFFFFFFFFFFFFFFFF)
    best_key = np.full((H * W,), sentinel, dtype=np.uint64)
    np.minimum.at(best_key, pix, key)

    chosen = best_key != sentinel
    chosen_flat = np.where(chosen)[0]
    if chosen_flat.size == 0:
        return blank, blank, blank
    best_idx = (best_key[chosen_flat] & np.uint64(0xFFFFFFFF)).astype(np.int64)

    depth_flat = np.full((H * W,), np.nan, dtype=np.float32)
    depth_flat[chosen_flat] = z[best_idx].astype(np.float32)
    depth_hw = depth_flat.reshape(H, W)
    valid_hw = chosen.reshape(H, W)
    depth_rgb = _depth_to_colormap_rgb(depth_hw, valid_hw, max_depth_m=float(depth_max_m))

    rgb_flat = np.zeros((H * W, 3), dtype=np.uint8)
    if rgb is not None:
        rgb_flat[chosen_flat] = rgb[best_idx]
    rgb_img = rgb_flat.reshape(H, W, 3)

    sem_label_flat = np.full((H * W,), np.int32(-1), dtype=np.int32)
    sem_label_flat[chosen_flat] = lab[best_idx].astype(np.int32)
    sem_rgb = _hash_colorize(sem_label_flat.reshape(H, W).astype(np.int64))

    return rgb_img, depth_rgb, sem_rgb


def _project_pano_from_kf_map(
    *,
    curr_T_WC: np.ndarray,
    kf_T_WC: np.ndarray,
    kf_points_k: np.ndarray,
    kf_n_points: np.ndarray,
    kf_label_id: np.ndarray,
    kf_rgb_u8: np.ndarray | None,
    radius_m: float,
    out_h: int,
    out_w: int,
    v_fov_deg: float,
    max_points: int,
    mode: str,
    depth_max_m: float,
) -> np.ndarray:
    """
    Render a 360 panorama from a keyframe-centric map (camera-centric spherical projection).
    """

    H, W = int(out_h), int(out_w)
    pano_blank = np.zeros((H, W, 3), dtype=np.uint8)

    Xc, lab, rgb = _gather_points_in_current_camera_from_kf_map(
        curr_T_WC=curr_T_WC,
        kf_T_WC=kf_T_WC,
        kf_points_k=kf_points_k,
        kf_n_points=kf_n_points,
        kf_label_id=kf_label_id,
        kf_rgb_u8=kf_rgb_u8,
        max_points=int(max_points),
    )
    if Xc.shape[0] == 0:
        return pano_blank

    radius = float(radius_m)
    if not np.isfinite(radius) or radius <= 0:
        radius = float("inf")
    r = np.linalg.norm(Xc, axis=1)
    keep = np.isfinite(Xc).all(axis=1) & np.isfinite(r) & (r > 1e-6) & (r <= radius)
    if not np.any(keep):
        return pano_blank
    Xc = Xc[keep]
    r = r[keep]
    lab = lab[keep]
    if rgb is not None:
        rgb = rgb[keep]

    x = Xc[:, 0]
    y = Xc[:, 1]
    z = Xc[:, 2]
    az = np.arctan2(x, z)
    horiz = np.sqrt(np.maximum(1e-12, x * x + z * z))
    el = np.arctan2(-y, horiz)

    vfov = float(v_fov_deg) * (np.pi / 180.0)
    vfov = float(np.clip(vfov, 1e-3, np.pi - 1e-3))
    el_min = -0.5 * vfov
    el_max = +0.5 * vfov
    inside = (el >= el_min) & (el <= el_max) & np.isfinite(az) & np.isfinite(el)
    if not np.any(inside):
        return pano_blank
    az = az[inside]
    el = el[inside]
    r = r[inside]
    lab = lab[inside]
    if rgb is not None:
        rgb = rgb[inside]

    u = ((az + np.pi) / (2.0 * np.pi) * float(W - 1)).astype(np.int32)
    v = ((el_max - el) / vfov * float(H - 1)).astype(np.int32)
    inside_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H) & np.isfinite(r)
    if not np.any(inside_uv):
        return pano_blank
    u = u[inside_uv]
    v = v[inside_uv]
    r = r[inside_uv]
    lab = lab[inside_uv]
    if rgb is not None:
        rgb = rgb[inside_uv]

    pix = (v.astype(np.int64) * int(W) + u.astype(np.int64)).astype(np.int64)
    idx = np.arange(r.shape[0], dtype=np.uint64)
    r_int = np.clip(r * 10000.0, 0.0, float(2**31 - 1)).astype(np.uint64)
    key = (r_int << 32) | idx
    sentinel = np.uint64(0xFFFFFFFFFFFFFFFF)
    best_key = np.full((H * W,), sentinel, dtype=np.uint64)
    np.minimum.at(best_key, pix, key)
    chosen = best_key != sentinel
    chosen_flat = np.where(chosen)[0]
    if chosen_flat.size == 0:
        return pano_blank
    best_idx = (best_key[chosen_flat] & np.uint64(0xFFFFFFFF)).astype(np.int64)

    mode = str(mode).lower().strip()
    if mode == "depth":
        depth_flat = np.full((H * W,), np.nan, dtype=np.float32)
        depth_flat[chosen_flat] = r[best_idx].astype(np.float32)
        depth_hw = depth_flat.reshape(H, W)
        valid_hw = chosen.reshape(H, W)
        pano_rgb = _depth_to_colormap_rgb(depth_hw, valid_hw, max_depth_m=float(depth_max_m))
    elif mode == "rgb":
        rgb_flat = np.zeros((H * W, 3), dtype=np.uint8)
        if rgb is not None:
            rgb_flat[chosen_flat] = rgb[best_idx]
        pano_rgb = rgb_flat.reshape(H, W, 3)
    else:
        sem_flat = np.full((H * W,), np.int32(-1), dtype=np.int32)
        sem_flat[chosen_flat] = lab[best_idx].astype(np.int32)
        pano_rgb = _hash_colorize(sem_flat.reshape(H, W).astype(np.int64))

    try:
        txt = f"pano radius={radius:.2f}m  pts={int(r.shape[0])}"
        cv2.putText(pano_rgb, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pano_rgb, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    except Exception:
        pass

    return pano_rgb


def _compute_esdf_from_kf_map(
    *,
    curr_T_WC: np.ndarray,
    kf_T_WC: np.ndarray,
    kf_points_k: np.ndarray,
    kf_n_points: np.ndarray,
    kf_label_id: np.ndarray,
    radius_m: float,
    voxel_m: float,
    use_semantic: bool,
    obstacle_labels: set[int] | None,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Build a local 3D occupancy grid around the current camera and compute an ESDF slice.

    Returns (occ_rgb, esdf_rgb, min_dist_m) or None if computation failed/empty.

    - curr_T_WC: (4,4) current camera->world (Sim3 allowed).
    - kf_T_WC:   (K,4,4) keyframe camera->world (loop-optimized).
    - kf_points_k: (K,P,3) points in keyframe camera coords.
    - kf_n_points: (K,) int32, valid point count per keyframe.
    - kf_label_id: (K,P) int32 semantic labels.
    - radius_m: cube half-size in meters.
    - voxel_m: voxel size in meters.
    - use_semantic: if True, only obstacle_labels are considered occupied; otherwise all points occupy.
    - obstacle_labels: set of labels that are obstacles; if None -> all labels are obstacles.
    """
    if _ndi is None:
        # NOTE:
        #   This function can be called from multiple visualization code paths.
        #   We keep a defensive guard here, but the *intended* behavior is:
        #     - if ESDF is explicitly enabled, we raise early in `run_stream_debug_viz()`
        #       (fail fast, do not silently skip)
        #     - if ESDF is not enabled, this function is never called
        return None
    try:
        T_CW = np.linalg.inv(curr_T_WC).astype(np.float32, copy=False)  # world -> current
    except Exception:
        return None

    K = int(kf_T_WC.shape[0])
    P = int(kf_points_k.shape[1])
    if K == 0 or P == 0:
        return None

    r = float(radius_m)
    v = float(voxel_m)
    if r <= 0 or v <= 0:
        return None

    # Grid spans [-r, r] in all three axes (current camera frame).
    n = int(np.ceil((2.0 * r) / v)) + 1  # inclusive of both ends
    occ = np.zeros((n, n, n), dtype=bool)

    # Precompute T_CK for all keyframes: T_CK = T_CW * T_WK
    try:
        T_CK = T_CW[None, :, :] @ kf_T_WC  # (K,4,4)
    except Exception:
        return None

    pts_all = []
    labs_all = []
    for k in range(K):
        nk = int(kf_n_points[k])
        if nk <= 0:
            continue
        try:
            Xk = kf_points_k[k, :nk, :].astype(np.float32, copy=False)
        except Exception:
            continue
        try:
            M = T_CK[k]
            R = M[:3, :3]
            t = M[:3, 3]
            Xc = (Xk @ R.T) + t[None, :]
        except Exception:
            continue
        labs = None
        if use_semantic and kf_label_id is not None:
            try:
                labs = kf_label_id[k, :nk].astype(np.int32, copy=False)
            except Exception:
                labs = None
        pts_all.append(Xc)
        if labs is not None:
            labs_all.append(labs)
        else:
            labs_all.append(None)

    if not pts_all:
        return None

    # Concatenate and filter within the cube.
    Xc = np.concatenate(pts_all, axis=0)
    if any(l is not None for l in labs_all):
        lab_concat = []
        for labs, pts in zip(labs_all, pts_all):
            if labs is None:
                lab_concat.append(np.full((pts.shape[0],), -1, dtype=np.int32))
            else:
                lab_concat.append(labs)
        label = np.concatenate(lab_concat, axis=0)
    else:
        label = None

    finite = np.isfinite(Xc).all(axis=1)
    Xc = Xc[finite]
    if label is not None:
        label = label[finite]

    if Xc.shape[0] == 0:
        return None

    mask_cube = (
        (Xc[:, 0] >= -r)
        & (Xc[:, 0] <= r)
        & (Xc[:, 1] >= -r)
        & (Xc[:, 1] <= r)
        & (Xc[:, 2] >= -r)
        & (Xc[:, 2] <= r)
    )
    Xc = Xc[mask_cube]
    if label is not None:
        label = label[mask_cube]

    if Xc.shape[0] == 0:
        return None

    # Semantic filtering: keep only obstacle labels if requested.
    if use_semantic and obstacle_labels is not None and label is not None:
        mask_obs = np.isin(label, list(obstacle_labels))
        Xc = Xc[mask_obs]
    if Xc.shape[0] == 0:
        return None

    idx = np.floor((Xc + r) / v).astype(np.int32)
    inside = (
        (idx[:, 0] >= 0)
        & (idx[:, 0] < n)
        & (idx[:, 1] >= 0)
        & (idx[:, 1] < n)
        & (idx[:, 2] >= 0)
        & (idx[:, 2] < n)
    )
    idx = idx[inside]
    if idx.shape[0] == 0:
        return None

    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    try:
        dist_out = _ndi.distance_transform_edt(~occ) * v  # outside distance (meters)
    except Exception:
        return None

    min_dist = float(np.nanmin(dist_out)) if dist_out.size > 0 else float("nan")

    # Build middle slice images for visualization (x-z plane at mid y).
    mid = n // 2
    occ_slice = occ[:, mid, :]
    dist_slice = dist_out[:, mid, :]

    occ_img = (occ_slice.astype(np.uint8) * 255)
    occ_rgb = cv2.cvtColor(occ_img, cv2.COLOR_GRAY2RGB)

    # Clip distances to [0, radius] for visualization.
    dist_clipped = np.clip(dist_slice, 0.0, r)
    norm = (dist_clipped / max(r, 1e-6)).astype(np.float32)
    dist_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    esdf_rgb = cv2.applyColorMap(dist_u8, cv2.COLORMAP_JET)

    return occ_rgb, esdf_rgb, min_dist




def _project_pano_from_pointcloud_rgb(
    *,
    points_w: np.ndarray,
    rgb_pts_u8: np.ndarray | None,
    label_id: np.ndarray,
    T_WC: np.ndarray,
    radius_m: float,
    out_h: int,
    out_w: int,
    v_fov_deg: float,
    max_points: int,
    mode: str,
    depth_max_m: float,
) -> np.ndarray:
    """
    Render a 360-degree panorama around the current camera pose from a global point cloud.

    Mapping:
      - azimuth   = atan2(x, z)          in [-pi, +pi]  -> horizontal pixel
      - elevation = atan2(-y, sqrt(x^2+z^2)) in [-vfov/2, +vfov/2] -> vertical pixel

    IMPORTANT:
      - We apply a radius filter in camera coordinates: ||X_cam|| <= radius_m.
      - We use O(N) nearest-point z-buffer (range) without sorting.

    Output:
      pano_rgb_u8: (H,W,3) uint8 in RGB order.
    """

    H, W = int(out_h), int(out_w)
    pts = np.asarray(points_w, dtype=np.float32).reshape(-1, 3)
    lab = np.asarray(label_id, dtype=np.int32).reshape(-1)
    rgb = None if rgb_pts_u8 is None else np.asarray(rgb_pts_u8, dtype=np.uint8).reshape(-1, 3)

    pano_blank = np.zeros((H, W, 3), dtype=np.uint8)
    if pts.shape[0] == 0:
        return pano_blank

    # Optional subsampling for speed (keep arrays aligned).
    #
    # See `_project_triplet_from_pointcloud_rgb` for rationale. We use stride sampling to keep
    # the panorama rendering under a tight latency budget.
    m = int(max_points)
    if m > 0 and pts.shape[0] > m:
        n = int(pts.shape[0])
        stride = max(1, n // m)
        sel = np.arange(0, n, stride, dtype=np.int64)[:m]
        pts = pts[sel]
        lab = lab[sel]
        if rgb is not None:
            rgb = rgb[sel]

    # World -> camera (rigid inverse, no 4x4 inversion).
    #
    # See `_project_triplet_from_pointcloud_rgb` for derivation.
    M_WC = np.asarray(T_WC, dtype=np.float32).reshape(4, 4)
    R_wc = M_WC[:3, :3]
    t_wc = M_WC[:3, 3]
    Xc = (pts - t_wc[None, :]) @ R_wc

    # Radius filter in camera coordinates.
    radius = float(radius_m)
    if not np.isfinite(radius) or radius <= 0:
        radius = float("inf")
    r = np.linalg.norm(Xc, axis=1)
    keep = np.isfinite(Xc).all(axis=1) & np.isfinite(r) & (r > 1e-6) & (r <= radius)
    if not np.any(keep):
        return pano_blank
    Xc = Xc[keep]
    r = r[keep]
    lab = lab[keep]
    if rgb is not None:
        rgb = rgb[keep]

    # Camera convention: +x right, +y down, +z forward.
    x = Xc[:, 0]
    y = Xc[:, 1]
    z = Xc[:, 2]
    az = np.arctan2(x, z)
    horiz = np.sqrt(np.maximum(1e-12, x * x + z * z))
    el = np.arctan2(-y, horiz)

    vfov = float(v_fov_deg) * (np.pi / 180.0)
    vfov = float(np.clip(vfov, 1e-3, np.pi - 1e-3))
    el_min = -0.5 * vfov
    el_max = +0.5 * vfov
    inside = (el >= el_min) & (el <= el_max) & np.isfinite(az) & np.isfinite(el)
    if not np.any(inside):
        return pano_blank
    az = az[inside]
    el = el[inside]
    r = r[inside]
    lab = lab[inside]
    if rgb is not None:
        rgb = rgb[inside]

    u = ((az + np.pi) / (2.0 * np.pi) * float(W - 1)).astype(np.int32)
    v = ((el_max - el) / vfov * float(H - 1)).astype(np.int32)
    inside_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H) & np.isfinite(r)
    if not np.any(inside_uv):
        return pano_blank
    u = u[inside_uv]
    v = v[inside_uv]
    r = r[inside_uv]
    lab = lab[inside_uv]
    if rgb is not None:
        rgb = rgb[inside_uv]

    # O(N) nearest range without sorting.
    pix = (v.astype(np.int64) * int(W) + u.astype(np.int64)).astype(np.int64)
    idx = np.arange(r.shape[0], dtype=np.uint64)
    r_int = np.clip(r * 10000.0, 0.0, float(2**31 - 1)).astype(np.uint64)
    key = (r_int << 32) | idx
    sentinel = np.uint64(0xFFFFFFFFFFFFFFFF)
    best_key = np.full((H * W,), sentinel, dtype=np.uint64)
    np.minimum.at(best_key, pix, key)
    chosen = best_key != sentinel
    chosen_flat = np.where(chosen)[0]
    if chosen_flat.size == 0:
        return pano_blank
    best_idx = (best_key[chosen_flat] & np.uint64(0xFFFFFFFF)).astype(np.int64)

    mode = str(mode).lower().strip()
    if mode == "depth":
        depth_flat = np.full((H * W,), np.nan, dtype=np.float32)
        depth_flat[chosen_flat] = r[best_idx].astype(np.float32)
        depth_hw = depth_flat.reshape(H, W)
        valid_hw = chosen.reshape(H, W)
        pano_rgb = _depth_to_colormap_rgb(depth_hw, valid_hw, max_depth_m=float(depth_max_m))
    elif mode == "rgb":
        rgb_flat = np.zeros((H * W, 3), dtype=np.uint8)
        if rgb is not None:
            rgb_flat[chosen_flat] = rgb[best_idx]
        pano_rgb = rgb_flat.reshape(H, W, 3)
    else:
        sem_flat = np.full((H * W,), np.int32(-1), dtype=np.int32)
        sem_flat[chosen_flat] = lab[best_idx].astype(np.int32)
        pano_rgb = _hash_colorize(sem_flat.reshape(H, W).astype(np.int64))

    # Overlay radius info (debug).
    try:
        txt = f"pano radius={radius:.2f}m  pts={int(r.shape[0])}"
        cv2.putText(pano_rgb, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pano_rgb, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    except Exception:
        pass

    return pano_rgb


def _shift2d_pad_edge(arr_hw: np.ndarray, *, dy: int, dx: int) -> np.ndarray:
    """
    Shift a 2D array by (dy, dx) with edge-padding.

    This helper is used by the depth-guided semantic refinement (debug-only).

    Definition
    ----------
    out[y, x] = arr[clamp(y + dy), clamp(x + dx)]

    Inputs:
      arr_hw: (H,W) array (any dtype)
      dy,dx: integer pixel shifts

    Output:
      out_hw: (H,W) array with the same dtype as arr_hw

    Notes:
      - We use symmetric padding by abs(dy), abs(dx) so the slicing indices are always valid
        and non-negative (avoids subtle Python negative-index behavior).
      - `mode="edge"` performs replication padding, which is acceptable for visualization.
    """

    arr = np.asarray(arr_hw)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    dy = int(dy)
    dx = int(dx)
    if dy == 0 and dx == 0:
        return arr

    pad_y = abs(dy)
    pad_x = abs(dx)
    pad = ((pad_y, pad_y), (pad_x, pad_x))
    arr_pad = np.pad(arr, pad, mode="edge")
    y0 = pad_y + dy
    x0 = pad_x + dx
    return arr_pad[y0 : y0 + h, x0 : x0 + w]


def _depth_guided_label_refine_v0(
    label_hw: np.ndarray,
    depth_hw: np.ndarray,
    valid_hw: np.ndarray,
    *,
    sigma_m: float,
    iters: int,
) -> np.ndarray:
    """
    Very small depth-guided label refinement (debug-only, CRF-like but ultra-lightweight).

    What this does (high-level)
    ---------------------------
    The user requested a second-stage semantic optimization that uses the (filtered) depth map
    to stabilize the displayed semantic labels. Intuition:
      - Pixels with similar depth are more likely to belong to the same surface/object.
      - Depth discontinuities are likely semantic boundaries, so we should NOT propagate labels
        across large depth jumps.

    We implement a minimal local "weighted mode" update:
      - For each pixel, look at a 3x3 neighborhood (9 candidates).
      - Compute a depth similarity weight for each neighbor:
          w = exp( -0.5 * ((z_n - z_c) / sigma)^2 )
        where sigma controls edge preservation (smaller sigma => stronger boundary protection).
      - Choose the label that wins the largest SUM of weights among the 9 candidates.

    This approximates a single mean-field-like smoothing step, but without any global graph
    optimization. It is intentionally simple and fast, and runs only in the debug process.

    Inputs:
      label_hw: (H,W) int64
        Hard semantic labels to refine (RAW or STABLE).
      depth_hw: (H,W) float32
        Depth map in meters. IMPORTANT: we assume this is the SAME depth that is displayed in
        the depth panel (i.e., already includes any debug filtering if enabled).
      valid_hw: (H,W) bool
        Valid depth mask. We only refine pixels with valid depth; invalid pixels keep original label.
      sigma_m: float
        Depth similarity bandwidth in meters. Typical values: 0.05 ~ 0.20 depending on noise.
      iters: int
        Number of refinement iterations (>=1). Keep small (1-2) for real-time debug.

    Output:
      refined_label_hw: (H,W) int64

    IMPORTANT
    ---------
    This function is used ONLY for visualization. It must not affect SLAM tracking, gating,
    optimization, keyframe selection, or any stored semantic cache.
    """

    lab0 = np.asarray(label_hw, dtype=np.int64)
    z0 = np.asarray(depth_hw, dtype=np.float32)
    v0 = np.asarray(valid_hw, dtype=bool)
    h, w = int(lab0.shape[0]), int(lab0.shape[1])
    if z0.shape != (h, w) or v0.shape != (h, w):
        return lab0

    sigma = float(max(sigma_m, 1e-6))
    inv_sigma = 1.0 / sigma
    n_iters = int(max(1, iters))

    # Precompute depth neighbors and validity neighbors once; depth is fixed during refinement.
    shifts: List[Tuple[int, int]] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    z_neigh = [_shift2d_pad_edge(z0, dy=dy, dx=dx) for (dy, dx) in shifts]
    v_neigh = [_shift2d_pad_edge(v0, dy=dy, dx=dx) for (dy, dx) in shifts]

    lab = lab0.copy()
    for _ in range(n_iters):
        # Build candidate labels by shifting the CURRENT labels (labels evolve per iteration).
        lab_neigh = [_shift2d_pad_edge(lab, dy=dy, dx=dx) for (dy, dx) in shifts]

        # Stack to (K,H,W) where K=9.
        cand_labels = np.stack(lab_neigh, axis=0)

        # Compute depth-based weights for each candidate in a fully vectorized way.
        # Weight is zero if either the center or the neighbor depth is invalid.
        z_center = z0  # (H,W)
        v_center = v0  # (H,W)
        weights: List[np.ndarray] = []
        for zn, vn in zip(z_neigh, v_neigh):
            m = v_center & vn
            dz = np.where(m, (zn - z_center), 0.0).astype(np.float32, copy=False)
            w_hw = np.exp(-0.5 * (dz * inv_sigma) ** 2).astype(np.float32, copy=False)
            w_hw[~m] = 0.0
            weights.append(w_hw)
        cand_w = np.stack(weights, axis=0)  # (K,H,W) float32

        # Score each candidate label by summing weights of *all* candidates that share the same label.
        #
        # Since K is tiny (9), we can do a short Python loop over candidates while still being
        # fully vectorized over pixels (H*W). This avoids any per-pixel Python loops.
        scores = np.zeros_like(cand_w, dtype=np.float32)  # (K,H,W)
        for i in range(cand_labels.shape[0]):
            eq = cand_labels == cand_labels[i]  # (K,H,W) bool
            scores[i] = np.sum(cand_w * eq, axis=0)

        best = np.argmax(scores, axis=0).astype(np.int64)  # (H,W)
        refined = np.take_along_axis(cand_labels, best[None, ...], axis=0)[0]

        # For pixels without valid depth, keep the original labels unchanged.
        refined[~v_center] = lab[~v_center]
        lab = refined

    return lab


def _depth_to_colormap_rgb(
    depth_hw: np.ndarray,
    valid_hw: np.ndarray,
    *,
    max_depth_m: float,
) -> np.ndarray:
    """
    Create an RGB colormap visualization of depth.

    Inputs:
      depth_hw: (H,W) float32 depth in meters
      valid_hw: (H,W) bool, True where depth is valid
      max_depth_m: depth range upper bound used for normalization

    Output:
      rgb_u8: (H,W,3) uint8 RGB pseudo-color

    Notes:
      - Invalid pixels are rendered as black.
      - We use a fixed range [0, max_depth_m] for speed and repeatability.
    """

    z = np.asarray(depth_hw, dtype=np.float32)
    valid = np.asarray(valid_hw, dtype=bool)
    h, w = int(z.shape[0]), int(z.shape[1])

    z_clamped = np.clip(z, 0.0, float(max_depth_m))
    z_u8 = np.zeros((h, w), dtype=np.uint8)
    if np.any(valid):
        z_u8[valid] = (255.0 * (z_clamped[valid] / float(max_depth_m))).astype(np.uint8)

    # OpenCV produces BGR colormap; convert to RGB for internal consistency.
    bgr = cv2.applyColorMap(z_u8, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb


def _compute_grid_centers(h: int, w: int, grid: int) -> List[Tuple[int, int]]:
    """
    Compute (y,x) centers for an evenly spaced grid (e.g., 3x3).

    We avoid borders by placing points at i/(grid+1) fractions:
      i = 1..grid.
    """

    g = int(max(1, grid))
    ys = [int(round((i + 1) * (h - 1) / (g + 1))) for i in range(g)]
    xs = [int(round((i + 1) * (w - 1) / (g + 1))) for i in range(g)]
    return [(y, x) for y in ys for x in xs]


def _patch_mean_depth(
    depth_hw: np.ndarray,
    valid_hw: np.ndarray,
    *,
    cy: int,
    cx: int,
    patch: int,
) -> float:
    """
    Compute mean depth in a small square patch centered at (cy,cx).

    Inputs:
      depth_hw: (H,W) float32 depth in meters
      valid_hw: (H,W) bool validity mask
      cy,cx: center pixel coordinates
      patch: patch size (odd preferred, e.g. 3 => 3x3). Even values are rounded up.

    Output:
      mean_depth_m: float, NaN if patch has no valid pixels.
    """

    z = np.asarray(depth_hw, dtype=np.float32)
    valid = np.asarray(valid_hw, dtype=bool)
    h, w = int(z.shape[0]), int(z.shape[1])

    p = int(max(1, patch))
    if p % 2 == 0:
        p += 1
    r = p // 2

    y0 = max(0, int(cy) - r)
    y1 = min(h, int(cy) + r + 1)
    x0 = max(0, int(cx) - r)
    x1 = min(w, int(cx) + r + 1)

    sub_valid = valid[y0:y1, x0:x1]
    if not np.any(sub_valid):
        return float("nan")
    sub_z = z[y0:y1, x0:x1]
    return float(np.mean(sub_z[sub_valid]))


def _topk_semantic_nearest(
    *,
    label_hw: np.ndarray,
    depth_hw: np.ndarray,
    valid_hw: np.ndarray,
    k: int,
) -> List[Tuple[int, float, int, int]]:
    """
    Find the K semantic classes with the smallest (nearest) depth in the current frame.

    Inputs:
      label_hw: (H,W) int64
        Hard labels. IMPORTANT: labels can be large (e.g., 24-bit RGB codes) so we must NOT
        allocate arrays of size `max(label)+1`.
      depth_hw: (H,W) float32
        Depth in meters.
      valid_hw: (H,W) bool
        Valid depth mask.
      k: int
        Number of classes to return.

    Output:
      A list of tuples: [(label_id, min_depth_m, x, y), ...], sorted by min_depth ascending.

    Implementation notes:
      - This is debug-only, but we still keep it reasonably efficient:
          1) compress labels via `np.unique(..., return_inverse=True)` (small, <= #classes)
          2) compute per-class min via `np.minimum.at` (O(N_valid))
          3) for the selected top-K classes, locate an argmin pixel (K is small, e.g., 9)
      - If there are fewer than K present classes with valid depth, we return fewer entries.
    """

    k = int(max(0, k))
    if k <= 0:
        return []

    lab = np.asarray(label_hw, dtype=np.int64)
    z = np.asarray(depth_hw, dtype=np.float32)
    valid = np.asarray(valid_hw, dtype=bool)
    h, w = int(z.shape[0]), int(z.shape[1])

    if lab.shape != (h, w):
        return []

    if not np.any(valid):
        return []

    z_flat = z.reshape(-1)
    lab_flat = lab.reshape(-1)
    valid_flat = valid.reshape(-1)

    lab_v = lab_flat[valid_flat]
    z_v = z_flat[valid_flat]

    # Compress labels -> contiguous indices.
    uniq, inv = np.unique(lab_v, return_inverse=True)
    if uniq.size == 0:
        return []

    mins = np.full((uniq.size,), np.float32(np.inf), dtype=np.float32)
    np.minimum.at(mins, inv, z_v.astype(np.float32, copy=False))

    finite = np.isfinite(mins)
    if not np.any(finite):
        return []

    # Pick top-k smallest mins using argpartition (avoid full sort when possible).
    cand_idx = np.where(finite)[0]
    cand_mins = mins[cand_idx]
    kk = min(k, cand_idx.size)
    if kk <= 0:
        return []

    if cand_idx.size > kk:
        sel_local = np.argpartition(cand_mins, kk - 1)[:kk]
    else:
        sel_local = np.arange(cand_idx.size)
    sel_idx = cand_idx[sel_local]

    # Sort the selected K by depth ascending (K is tiny).
    order = np.argsort(mins[sel_idx])
    sel_idx = sel_idx[order]

    out: List[Tuple[int, float, int, int]] = []
    for ci in sel_idx.tolist():
        lab_id = int(uniq[ci])
        # Locate an argmin pixel for this label (only for the few selected labels).
        mask = valid_flat & (lab_flat == lab_id)
        if not np.any(mask):
            continue
        tmp = np.where(mask, z_flat, np.float32(np.inf))
        flat_idx = int(np.argmin(tmp))
        y = int(flat_idx // w)
        x = int(flat_idx % w)
        d = float(tmp[flat_idx])
        out.append((lab_id, d, x, y))

    return out


def _pose_delta_from_tqs(prev_tqs: np.ndarray, curr_tqs: np.ndarray) -> Tuple[float, float]:
    """
    Compute a small pose delta metric from (t,q,s) vectors.

    Inputs:
      prev_tqs: (>=7,) float, [tx,ty,tz,qx,qy,qz,qw,(scale)]
      curr_tqs: (>=7,) float

    Output:
      (delta_trans_m, delta_rot_rad)

    Notes:
      - We intentionally avoid importing lietorch here (debug process should remain lightweight).
      - Rotation delta uses quaternion angle:
          angle = 2 * acos(|dot(q1,q2)|) in [0, pi]
    """

    p = np.asarray(prev_tqs, dtype=np.float64).reshape(-1)
    c = np.asarray(curr_tqs, dtype=np.float64).reshape(-1)
    if p.shape[0] < 7 or c.shape[0] < 7:
        return 0.0, 0.0

    tp, qp = p[:3], p[3:7]
    tc, qc = c[:3], c[3:7]
    delta_t = float(np.linalg.norm(tc - tp))

    qp_n = qp / max(1e-12, float(np.linalg.norm(qp)))
    qc_n = qc / max(1e-12, float(np.linalg.norm(qc)))
    dot = float(np.clip(np.abs(np.dot(qp_n, qc_n)), 0.0, 1.0))
    delta_r = float(2.0 * np.arccos(dot))
    return delta_t, delta_r


class _Kalman1D:
    """
    Minimal 1D Kalman filter (random-walk model) for smoothing scalar readouts.

    This is used ONLY for debug visualization to reduce flicker in per-class min depth.
    It MUST NOT influence SLAM optimization or any gating logic.

    Model:
      x_t = x_{t-1} + w,   w ~ N(0, Q)
      z_t = x_t     + v,   v ~ N(0, R)

    State:
      x: filtered value (float)
      p: state variance (float)
    """

    def __init__(self, *, q: float, r: float, p0: float = 1.0) -> None:
        self.q = float(max(q, 1e-12))
        self.r = float(max(r, 1e-12))
        self.p0 = float(max(p0, 1e-12))
        self.x: float | None = None
        self.p: float = self.p0

    def reset(self) -> None:
        self.x = None
        self.p = self.p0

    def predict(self) -> None:
        # Random-walk: x stays the same; uncertainty grows by Q.
        self.p = self.p + self.q

    def update(self, z: float) -> float:
        z = float(z)
        if self.x is None or not np.isfinite(self.x):
            # First observation initializes the state.
            self.x = z
            self.p = self.p0
            return self.x

        # Kalman gain
        k = self.p / (self.p + self.r)
        # State update
        self.x = self.x + k * (z - self.x)
        # Covariance update
        self.p = (1.0 - k) * self.p
        return float(self.x)


class _KalmanDepthMap:
    """
    Vectorized per-pixel 1D Kalman filter for a depth map.

    Why per-pixel (instead of filtering only the scalar min depth)?
      - The global min depth statistic is extremely sensitive to missing data:
          a) the argmin pixel can jump when some pixels become invalid (holes / occlusion)
          b) a single noisy pixel can dominate the min
      - Filtering the entire depth map reduces flicker and also allows "remembering" the last
        seen depth at pixels that temporarily become invalid.

    This is still lightweight because:
      - the match grid is small (e.g., 224x224)
      - we run it only at the debug visualization FPS (e.g., 10 Hz)

    State:
      x_hw: (H,W) float32 filtered depth
      p_hw: (H,W) float32 variance

    Update rule (random-walk, per pixel):
      p <- p + Q
      K <- p / (p + R)
      x <- x + K*(z - x)   (only where measurement is valid)
      p <- (1-K)*p         (only where measurement is valid)

    IMPORTANT:
      - This is visualization-only; it must not be used for optimization.
    """

    def __init__(self, *, q: float, r: float, p0: float = 1.0) -> None:
        self.q = float(max(q, 1e-12))
        self.r = float(max(r, 1e-12))
        self.p0 = float(max(p0, 1e-12))
        self.x_hw: np.ndarray | None = None
        self.p_hw: np.ndarray | None = None

    def reset(self) -> None:
        self.x_hw = None
        self.p_hw = None

    def ensure_shape(self, h: int, w: int) -> None:
        if self.x_hw is None or self.p_hw is None or self.x_hw.shape != (h, w):
            self.x_hw = np.full((h, w), np.nan, dtype=np.float32)
            self.p_hw = np.full((h, w), self.p0, dtype=np.float32)

    def step(self, z_hw: np.ndarray, valid_hw: np.ndarray) -> np.ndarray:
        """
        Update the filter with a new depth measurement map.

        Inputs:
          z_hw: (H,W) float32 depth measurement
          valid_hw: (H,W) bool measurement validity mask

        Output:
          x_hw: (H,W) float32 filtered depth estimate
        """

        h, w = int(z_hw.shape[0]), int(z_hw.shape[1])
        self.ensure_shape(h, w)
        assert self.x_hw is not None and self.p_hw is not None

        # Predict: random walk -> x unchanged, p grows.
        self.p_hw = self.p_hw + self.q

        # Update only where we have a valid measurement this frame.
        if np.any(valid_hw):
            p = self.p_hw
            k = p / (p + self.r)

            x = self.x_hw
            z = z_hw.astype(np.float32, copy=False)

            # Initialize unseen pixels on first valid observation.
            init_mask = valid_hw & (~np.isfinite(x))
            if np.any(init_mask):
                x[init_mask] = z[init_mask]
                p[init_mask] = self.p0

            upd_mask = valid_hw & np.isfinite(x)
            if np.any(upd_mask):
                x[upd_mask] = x[upd_mask] + k[upd_mask] * (z[upd_mask] - x[upd_mask])
                p[upd_mask] = (1.0 - k[upd_mask]) * p[upd_mask]

            self.x_hw = x
            self.p_hw = p

        return self.x_hw


def run_stream_debug_viz(
    msg_queue,
    *,
    window_name: str = "MASt3R-SLAM Stream Debug",
    overlay_alpha: float = 0.6,
    scale: int = 2,
    box_radius: int = 3,
    depth_source: str = "stable",
    # ---------------------------------------------------------------------
    # Depth filtering mode (debug-only):
    #   - "none"  : no filtering
    #   - "pixel" : per-pixel Kalman filter on depth map
    #   - "pose"  : pose-aware pixel Kalman (reset on motion)
    #
    # Backward compatibility:
    #   Older call sites used `enable_kalman=True/False`. If filter_mode is "none" and
    #   enable_kalman=True, we treat it as filter_mode="pixel".
    # ---------------------------------------------------------------------
    filter_mode: str = "none",
    enable_kalman: bool = False,
    kalman_q: float = 0.02,
    kalman_r: float = 0.10,
    pose_reset_trans_m: float = 0.02,
    pose_reset_rot_deg: float = 2.0,
    # ---------------------------------------------------------------------
    # Grid sampling controls (debug-only).
    # ---------------------------------------------------------------------
    sample_grid: int = 3,
    sample_patch: int = 3,
    info_width: int = 220,
    depth_vis_max_m: float = 10.0,
    # ---------------------------------------------------------------------
    # Semantic smoothing (debug-only).
    #
    # The GUI can optionally smooth the displayed hard-label segmentation maps
    # to reduce flicker. This is purely a visualization filter.
    # ---------------------------------------------------------------------
    semantic_filter_mode: str = "none",  # none | ema
    semantic_filter_target: str = "raw",  # raw | stable | both
    semantic_filter_momentum: float = 0.98,
    semantic_filter_u: float = 1.0,
    # ---------------------------------------------------------------------
    # Stable semantic panel input selection (debug-only).
    #
    # The original pipeline provides BOTH:
    #   - RAW semantics   : per-frame segmentation output (unstable / noisy)
    #   - STABLE semantics: warp+fuse output (geometry-stabilized)
    #
    # For some debugging tasks, the user wants the middle panel to show:
    #   "RAW + post-processing", while the left panel shows "RAW (no post-processing)".
    #
    # This switch selects the SOURCE of the middle ("stable") semantic panel:
    #   - "stable": use the actual stable labels sent by SLAM (default behavior historically)
    #   - "raw"   : use RAW labels as the input for the middle panel (for A/B visualization)
    #
    # IMPORTANT:
    #   This affects ONLY what is displayed in the debug GUI. It does not change SLAM state.
    # ---------------------------------------------------------------------
    stable_semantic_source: str = "raw",  # stable | raw
    # ---------------------------------------------------------------------
    # Depth-guided semantic refinement (debug-only).
    #
    # This is a second-stage refinement that uses the (already filtered) depth map shown in
    # the GUI to further stabilize the displayed segmentation.
    #
    # IMPORTANT:
    #   - Default OFF, because it is a visualization heuristic.
    #   - It must never change any SLAM internals (matching / gating / optimization).
    # ---------------------------------------------------------------------
    # Default OFF. This is a visualization heuristic and can be relatively strong; we keep it
    # opt-in so the baseline GUI remains comparable to the underlying SLAM outputs.
    semantic_depth_refine: bool = False,
    semantic_depth_refine_target: str = "stable",  # raw | stable | both
    semantic_depth_refine_iters: int = 1,
    semantic_depth_sigma_m: float = 0.50,
    # ---------------------------------------------------------------------
    # Semantic nearest-by-depth markers (debug-only).
    #
    # We optionally mark and print the K closest semantic classes by minimum depth.
    # ---------------------------------------------------------------------
    semantic_topk: int = 9,
    # ---------------------------------------------------------------------
    # Planning point cloud reprojection panels (debug-only).
    #
    # This integrates the "reader-side" pointcloud visualization into the debug window.
    #
    # Layout:
    #   - legacy             : original 3-column semantic/depth debug panels
    #   - planning_pointcloud: show (RGB | Depth | Sem) from global pointcloud reprojection
    #                          and optionally a 360 panorama (2nd row).
    #
    # Requirements:
    #   - SLAM (or its helper process) must publish a latest-only pointcloud snapshot to shm:
    #       points_w (N,3), label_id (N,), rgb_u8 (N,3 optional), T_WC (4,4)
    #   - We never sort points; z-buffer is O(N) via np.minimum.at.
    # ---------------------------------------------------------------------
    viz_layout: str = "legacy",  # legacy | planning_pointcloud
    planning_pointcloud_outdir: str = "logs/planning_pointcloud",
    planning_pointcloud_info_filename: str = "shm_info.json",
    planning_pointcloud_enable_pano: bool = True,
    planning_pointcloud_radius_m: float = 2.0,
    planning_pointcloud_pano_h: int = 96,
    planning_pointcloud_pano_vfov_deg: float = 60.0,
    planning_pointcloud_pano_mode: str = "sem",  # rgb | sem | depth
    planning_pointcloud_fov_deg: float = 60.0,
    planning_pointcloud_max_points: int = 200_000,
    # Optional: compute a local 3D occupancy/ESDF around the current camera
    # and visualize slices (debug-only; does NOT affect SLAM).
    planning_pointcloud_esdf_enable: bool = False,
    planning_pointcloud_esdf_radius_m: float = 2.0,
    planning_pointcloud_esdf_voxel_m: float = 0.1,
    planning_pointcloud_esdf_use_semantic: bool = False,
    planning_pointcloud_esdf_obstacle_labels: str = "",
) -> None:
    """
    Main visualization loop.

    Message format (dict) - preferred:
      {
        "frame_id": int,
        "rgb_u8": np.ndarray (H,W,3) uint8 RGB,
        "rgb_sem_u8": np.ndarray (H,W,3) uint8 RGB | None, # optional background aligned to semantic
        "label_raw_hw": np.ndarray (H,W) int64,            # raw semantic labels
        "label_stable_hw": np.ndarray (H,W) int64 | None,  # stable semantic labels (optional)
        "sem_raw_rgb_u8": np.ndarray (H,W,3) uint8 | None,    # raw semantic RGB visualization (optional)
        "sem_stable_rgb_u8": np.ndarray (H,W,3) uint8 | None, # stable semantic RGB visualization (optional)
        "depth_raw_hw": np.ndarray (H,W) float32 | None,   # raw depth (optional)
        "depth_stable_hw": np.ndarray (H,W) float32 | None,# stable depth (optional)
        "pose_tqs": np.ndarray (8,) float32 (optional; [t(3), q(4), s(1)])

        # Backward compatibility:
        #   Older senders may include:
        #     "label_hw": np.ndarray (H,W) int64
        #     "depth_hw": np.ndarray (H,W) float32
        #     "points": List[Tuple[int,int,int,float,int]]
      }

    Notes:
      - The sender should implement latest-only behavior (queue maxsize=1 and drop-old),
        so this process always renders the newest state without applying backpressure.
    """

    # ---------------------------------------------------------------------
    # Dependency enforcement (fail fast).
    #
    # The user explicitly requested that if ESDF visualization is enabled, we MUST raise an
    # error when the required dependency is missing (instead of silently skipping ESDF).
    #
    # SciPy provides a fast and reliable 3D Euclidean distance transform implementation.
    # Without it, we cannot compute ESDF in this debug process.
    # ---------------------------------------------------------------------
    if bool(planning_pointcloud_esdf_enable) and _ndi is None:
        raise ImportError(
            "ESDF visualization was requested (planning_pointcloud_esdf_enable=True) "
            "but SciPy is not available. Install it with: `pip install scipy`."
        )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # ---------------------------------------------------------------------
    # Default window size (simple + predictable).
    #
    # The user requested a slightly larger default size, without any auto-resize logic.
    # This is a pure GUI convenience and does not affect any SLAM computation.
    # ---------------------------------------------------------------------
    try:
        cv2.resizeWindow(window_name, 1600, 900)
    except Exception:
        pass

    last: Dict[str, Any] | None = None
    # Visualization-only depth smoother (per-pixel).
    depth_kalman = _KalmanDepthMap(q=float(kalman_q), r=float(kalman_r), p0=1.0)

    # Pose-aware mode: keep last pose to detect motion and reset depth_kalman.
    last_pose_tqs: np.ndarray | None = None
    pose_reset_rot_rad = float(pose_reset_rot_deg) * (np.pi / 180.0)

    # Normalize filter mode.
    mode = str(filter_mode).lower().strip()
    if mode not in ("none", "pixel", "pose"):
        mode = "none"
    if mode == "none" and bool(enable_kalman):
        mode = "pixel"

    # Normalize depth source selection for the depth panel.
    # This only changes what we DISPLAY, never what SLAM computes.
    depth_src = str(depth_source).lower().strip()
    if depth_src not in ("stable", "raw"):
        depth_src = "stable"

    # Normalize semantic filter settings.
    sem_mode = str(semantic_filter_mode).lower().strip()
    if sem_mode not in ("none", "ema"):
        sem_mode = "none"
    sem_target = str(semantic_filter_target).lower().strip()
    if sem_target not in ("raw", "stable", "both"):
        sem_target = "raw"

    # -------------------------------------------------------------------------
    # IMPORTANT (per user request): show semantic smoothing as an A/B comparison.
    #
    # We display:
    #   - LEFT panel : RAW labels WITHOUT semantic smoothing
    #   - MIDDLE panel: RAW labels WITH semantic smoothing (if enabled)
    #
    # This makes the visual difference directly attributable to the smoothing filter.
    # The CLI still controls whether smoothing runs (sem_mode) and its strength parameters.
    #
    # NOTE:
    #   We keep parsing `semantic_filter_target` for backward compatibility, but in this
    #   compare mode the filter is effectively applied to the middle panel only.
    # -------------------------------------------------------------------------

    # Normalize which semantic labels feed the middle ("stable") panel.
    stable_src = str(stable_semantic_source).lower().strip()
    if stable_src not in ("stable", "raw"):
        stable_src = "raw"

    # Normalize depth-guided semantic refinement settings.
    #
    # Default is OFF (semantic_depth_refine=False). When enabled, we run a tiny local refinement
    # step after semantic temporal smoothing and after depth filtering, using the SAME depth map
    # that is displayed in the GUI.
    sem_depth_refine = bool(semantic_depth_refine)
    sem_depth_target = str(semantic_depth_refine_target).lower().strip()
    if sem_depth_target not in ("raw", "stable", "both"):
        sem_depth_target = "stable"
    sem_depth_iters = int(max(1, int(semantic_depth_refine_iters)))
    sem_depth_sigma = float(max(float(semantic_depth_sigma_m), 1e-6))

    raw_sem_filter = _PerPixelHardLabelEMA(
        momentum=float(semantic_filter_momentum), u=float(semantic_filter_u)
    )
    stable_sem_filter = _PerPixelHardLabelEMA(
        momentum=float(semantic_filter_momentum), u=float(semantic_filter_u)
    )

    # -------------------------------------------------------------------------
    # Planning point cloud mode setup (optional).
    # -------------------------------------------------------------------------
    layout = str(viz_layout).lower().strip()
    if layout not in ("legacy", "planning_pointcloud"):
        layout = "legacy"

    # We attach lazily to avoid a hard dependency on the publisher being ready at GUI startup.
    pc_client: PlanningKeyframeMapClient | None = None
    pc_last_t_wc: np.ndarray | None = None
    pc_outdir = str(planning_pointcloud_outdir)
    pc_info_name = str(planning_pointcloud_info_filename)
    esdf_obs_set: set[int] | None = None
    if bool(planning_pointcloud_esdf_use_semantic):
        try:
            toks = [t for t in str(planning_pointcloud_esdf_obstacle_labels).replace(",", " ").split() if len(t) > 0]
            if toks:
                esdf_obs_set = set(int(x) for x in toks)
        except Exception:
            esdf_obs_set = None

    while True:
        try:
            last = msg_queue.get_nowait()
        except Exception:
            pass

        if last is None:
            time.sleep(0.005)
            continue

        # ---------------------------------------------------------------------
        # Planning semantic point cloud reprojection view (optional).
        #
        # This path replaces the legacy 3-column semantic/depth GUI with a point-cloud-driven view:
        #   - (top row)  RGB | Depth | Semantics
        #   - (bottom)   360 panorama within a radius (optional)
        #
        # IMPORTANT:
        #   - We do NOT display any directly transmitted RGB camera frame.
        #   - Everything shown here is derived from:
        #       (global point cloud) + (current pose T_WC) -> reprojection.
        # ---------------------------------------------------------------------
        if layout == "planning_pointcloud":
            # Use the incoming debug RGB only to determine output resolution (match-grid size).
            try:
                rgb_msg_u8 = np.asarray(last.get("rgb_u8", None), dtype=np.uint8)
                h_pc, w_pc = int(rgb_msg_u8.shape[0]), int(rgb_msg_u8.shape[1])
            except Exception:
                time.sleep(0.005)
                continue

            # Attach to the published pointcloud snapshot (shared memory) if needed.
            if pc_client is None:
                try:
                    pc_client = PlanningKeyframeMapClient(out_dir=pc_outdir, info_filename=pc_info_name)
                    pc_client.attach()
                except Exception:
                    pc_client = None

            if pc_client is not None:
                try:
                    # IMPORTANT:
                    #   The keyframe-centric shm schema can be very large (K * P points).
                    #   We therefore copy only the small pose/id arrays for consistency, and keep point
                    #   tensors as views (copy_points=False) to avoid allocating hundreds of MB per frame.
                    snap = pc_client.read(copy=True, copy_points=False)
                    T_WC = np.asarray(snap.curr_T_WC, dtype=np.float32)

                    rgb_rgb, depth_rgb, sem_rgb = _project_triplet_from_kf_map(
                        curr_T_WC=T_WC,
                        kf_T_WC=np.asarray(snap.kf_T_WC, dtype=np.float32),
                        kf_points_k=np.asarray(snap.kf_points_k, dtype=np.float32),
                        kf_n_points=np.asarray(snap.kf_n_points, dtype=np.int32),
                        kf_label_id=np.asarray(snap.kf_label_id, dtype=np.int32),
                        kf_rgb_u8=np.asarray(snap.kf_rgb_u8, dtype=np.uint8) if getattr(snap, "kf_rgb_u8", None) is not None else None,
                        out_h=h_pc,
                        out_w=w_pc,
                        fov_deg=float(planning_pointcloud_fov_deg),
                        max_points=int(planning_pointcloud_max_points),
                        depth_max_m=float(depth_vis_max_m),
                    )
                    top_rgb = np.concatenate([rgb_rgb, depth_rgb, sem_rgb], axis=1)

                    esdf_row = None
                    esdf_min = None
                    if bool(planning_pointcloud_esdf_enable):
                        esdf_res = _compute_esdf_from_kf_map(
                            curr_T_WC=T_WC,
                            kf_T_WC=np.asarray(snap.kf_T_WC, dtype=np.float32),
                            kf_points_k=np.asarray(snap.kf_points_k, dtype=np.float32),
                            kf_n_points=np.asarray(snap.kf_n_points, dtype=np.int32),
                            kf_label_id=np.asarray(snap.kf_label_id, dtype=np.int32),
                            radius_m=float(planning_pointcloud_esdf_radius_m),
                            voxel_m=float(planning_pointcloud_esdf_voxel_m),
                            use_semantic=bool(planning_pointcloud_esdf_use_semantic),
                            obstacle_labels=esdf_obs_set,
                        )
                        if esdf_res is not None:
                            occ_rgb, esdf_rgb, esdf_min = esdf_res
                            # Resize panels to a common width (half of top row) for display.
                            target_w = max(1, top_rgb.shape[1] // 2)
                            occ_rgb = cv2.resize(occ_rgb, (target_w, occ_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                            esdf_rgb = cv2.resize(esdf_rgb, (target_w, esdf_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                            esdf_row = np.concatenate([occ_rgb, esdf_rgb], axis=1)
                            try:
                                txt = f"ESDF r={planning_pointcloud_esdf_radius_m:.1f}m v={planning_pointcloud_esdf_voxel_m:.2f}m min={esdf_min:.2f}m"
                                cv2.putText(esdf_row, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                                cv2.putText(esdf_row, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                            except Exception:
                                pass

                    if bool(planning_pointcloud_enable_pano):
                        pano_rgb = _project_pano_from_kf_map(
                            curr_T_WC=T_WC,
                            kf_T_WC=np.asarray(snap.kf_T_WC, dtype=np.float32),
                            kf_points_k=np.asarray(snap.kf_points_k, dtype=np.float32),
                            kf_n_points=np.asarray(snap.kf_n_points, dtype=np.int32),
                            kf_label_id=np.asarray(snap.kf_label_id, dtype=np.int32),
                            kf_rgb_u8=np.asarray(snap.kf_rgb_u8, dtype=np.uint8) if getattr(snap, "kf_rgb_u8", None) is not None else None,
                            radius_m=float(planning_pointcloud_radius_m),
                            out_h=int(planning_pointcloud_pano_h),
                            out_w=int(top_rgb.shape[1]),
                            v_fov_deg=float(planning_pointcloud_pano_vfov_deg),
                            max_points=int(planning_pointcloud_max_points),
                            mode=str(planning_pointcloud_pano_mode),
                            depth_max_m=float(depth_vis_max_m),
                        )
                        composite_rgb = np.concatenate([top_rgb, pano_rgb], axis=0)
                    else:
                        composite_rgb = top_rgb

                    if esdf_row is not None:
                        # Stack ESDF/occupancy row below the existing composite.
                        # Resize ESDF row width to match composite width for clean stacking.
                        esdf_row_resized = cv2.resize(
                            esdf_row,
                            (composite_rgb.shape[1], esdf_row.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        composite_rgb = np.concatenate([composite_rgb, esdf_row_resized], axis=0)

                    # Pose overlay (uses the pose from the pointcloud snapshot).
                    composite_rgb = _draw_pose_overlay_rgb(
                        composite_rgb,
                        frame_id=int(getattr(snap, "frame_id", last.get("frame_id", -1))),
                        T_WC=T_WC,
                        last_t_wc=pc_last_t_wc,
                        font_scale=0.45,
                    )
                    pc_last_t_wc = T_WC[:3, 3].copy()

                    vis = cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)
                    if int(scale) != 1:
                        vis = cv2.resize(
                            vis,
                            (vis.shape[1] * int(scale), vis.shape[0] * int(scale)),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    cv2.imshow(window_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # q or ESC
                        break
                    continue
                except Exception:
                    # If reading/rendering fails, fall through to a small waiting screen.
                    pass

            # Waiting screen if shm is not ready.
            composite_rgb = np.zeros((max(80, h_pc), max(400, 3 * w_pc), 3), dtype=np.uint8)
            cv2.putText(
                composite_rgb,
                "Waiting for planning pointcloud shared-memory...",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                composite_rgb,
                f"outdir={pc_outdir}  info={pc_info_name}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            vis = cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)
            if int(scale) != 1:
                vis = cv2.resize(
                    vis,
                    (vis.shape[1] * int(scale), vis.shape[0] * int(scale)),
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            continue

        rgb = last["rgb_u8"]
        rgb_sem_u8 = last.get("rgb_sem_u8", None)
        # Preferred fields (newer sender):
        label_raw_hw = last.get("label_raw_hw", None)
        label_stable_hw = last.get("label_stable_hw", None)
        sem_raw_rgb_u8 = last.get("sem_raw_rgb_u8", None)
        sem_stable_rgb_u8 = last.get("sem_stable_rgb_u8", None)
        depth_raw_hw = last.get("depth_raw_hw", None)
        depth_stable_hw = last.get("depth_stable_hw", None)
        # Backward compatible fields (older sender):
        label_hw_compat = last.get("label_hw", None)
        depth_hw_compat = last.get("depth_hw", None)
        fid = int(last.get("frame_id", -1))
        pose_tqs = last.get("pose_tqs", None)

        # ---------------------------------------------------------------------
        # Inputs and basic overlays.
        #
        # Shapes (expected):
        #   rgb_u8    : (H,W,3) uint8 RGB
        #   label_hw  : (H,W)   int64
        #   depth_hw  : (H,W)   float32
        #   pose_tqs  : (8,)    float32 (optional)
        # ---------------------------------------------------------------------
        rgb_u8 = np.asarray(rgb, dtype=np.uint8)
        # Optional semantic-aligned RGB background (may come from the original, un-cropped input).
        rgb_sem_base = None
        if rgb_sem_u8 is not None:
            try:
                t = np.asarray(rgb_sem_u8, dtype=np.uint8)
                if t.ndim == 3 and t.shape == rgb_u8.shape:
                    rgb_sem_base = t
            except Exception:
                rgb_sem_base = None
        if rgb_sem_base is None:
            rgb_sem_base = rgb_u8

        # If the sender did not provide the new fields, fall back to the legacy ones.
        if label_raw_hw is None and label_hw_compat is not None:
            label_raw_hw = label_hw_compat
        if label_stable_hw is None and label_hw_compat is not None:
            label_stable_hw = label_hw_compat

        label_raw_hw = np.asarray(label_raw_hw, dtype=np.int64) if label_raw_hw is not None else None
        label_stable_hw_in = (
            np.asarray(label_stable_hw, dtype=np.int64) if label_stable_hw is not None else None
        )

        h, w = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])

        # ---------------------------------------------------------------------
        # Depth filtering (visualization only).
        #
        # We compute the depth map FIRST because later semantic refinement may depend on the
        # (already filtered) depth `z_use` shown in the depth panel.
        #
        # IMPORTANT:
        #   This entire block is debug-only. It must never change SLAM state.
        # ---------------------------------------------------------------------
        # Select which depth to show on the depth panel:
        #   - "stable": prefer stable depth when provided; otherwise fall back to raw depth.
        #   - "raw":    prefer raw depth when provided; otherwise fall back to stable depth.
        depth_for_panel = None
        if depth_src == "stable":
            depth_for_panel = depth_stable_hw if depth_stable_hw is not None else depth_raw_hw
        else:
            depth_for_panel = depth_raw_hw if depth_raw_hw is not None else depth_stable_hw
        if depth_for_panel is None and depth_hw_compat is not None:
            depth_for_panel = depth_hw_compat

        z_use = None
        valid_use = None
        if depth_for_panel is not None:
            z = np.asarray(depth_for_panel, dtype=np.float32)
            valid = np.isfinite(z) & (z > 0.0)

            if mode == "pose" and pose_tqs is not None:
                try:
                    curr_pose = np.asarray(pose_tqs, dtype=np.float32).reshape(-1)
                    if last_pose_tqs is not None:
                        dt, dr = _pose_delta_from_tqs(last_pose_tqs, curr_pose)
                        if (dt > float(pose_reset_trans_m)) or (dr > float(pose_reset_rot_rad)):
                            depth_kalman.reset()
                    last_pose_tqs = curr_pose
                except Exception:
                    pass

            if mode in ("pixel", "pose"):
                x = depth_kalman.step(z, valid_hw=valid)
                z_use = x
                valid_use = np.isfinite(x) & (x > 0.0)
            else:
                z_use = z
                valid_use = valid

        if z_use is not None and valid_use is not None:
            depth_rgb = _depth_to_colormap_rgb(
                z_use, valid_use, max_depth_m=float(depth_vis_max_m)
            )
            depth_overlay = _alpha_blend_rgb(rgb_u8, depth_rgb, float(overlay_alpha))
        else:
            depth_overlay = rgb_u8.copy()

        # ---------------------------------------------------------------------
        # Prepare semantic label maps for visualization (RAW and STABLE).
        #
        # Order of operations (if enabled):
        #   1) Temporal smoothing (EMA-like) on hard labels (optional)
        #   2) Depth-guided refinement using the SAME depth we display (optional)
        #
        # Both steps are visualization-only; they must never change SLAM internals.
        # ---------------------------------------------------------------------
        # RAW labels for viz (left panel): ALWAYS unfiltered (A/B comparison).
        label_raw_for_viz = label_raw_hw

        # STABLE labels for viz (middle panel).
        #
        # IMPORTANT (per user request for A/B comparison):
        #   The middle panel should be allowed to use RAW labels as its SOURCE, then apply
        #   the depth-guided post-processing. This makes the difference between left vs middle
        #   purely attributable to the post-processing (not to warp+fuse).
        label_stable_src_hw = label_stable_hw_in if stable_src == "stable" else label_raw_hw
        label_stable_for_viz = label_stable_src_hw
        if (
            label_stable_src_hw is not None
            and sem_mode == "ema"
        ):
            # Semantic smoothing is applied to the middle panel (A/B comparison).
            label_stable_for_viz = stable_sem_filter.step(label_stable_src_hw)

        # Optional: depth-guided refinement to reduce label flicker while respecting depth edges.
        #
        # We intentionally use `z_use` (the filtered depth shown in the GUI) rather than the raw
        # depth input. This matches the user's request: "use the filtered depth to do a second
        # semantic optimization."
        if sem_depth_refine and z_use is not None and valid_use is not None:
            try:
                if label_raw_for_viz is not None and sem_depth_target in ("raw", "both"):
                    label_raw_for_viz = _depth_guided_label_refine_v0(
                        label_raw_for_viz,
                        z_use,
                        valid_use,
                        sigma_m=sem_depth_sigma,
                        iters=sem_depth_iters,
                    )
                if label_stable_for_viz is not None and sem_depth_target in ("stable", "both"):
                    label_stable_for_viz = _depth_guided_label_refine_v0(
                        label_stable_for_viz,
                        z_use,
                        valid_use,
                        sigma_m=sem_depth_sigma,
                        iters=sem_depth_iters,
                    )
            except Exception:
                # Debug process must be robust; if refinement fails for any reason, just skip it.
                pass

        # ---------------------------------------------------------------------
        # Build semantic overlays.
        #
        # IMPORTANT (per user request):
        #   Unify the semantic colormap in the debug view and always use the RAW palette.
        #
        # In this debug UI, the "RAW palette" is implemented by `_hash_colorize(label_hw)`.
        # Therefore, we intentionally visualize BOTH RAW and STABLE labels through the same
        # hash-colorizer so the two panels are directly comparable (same label -> same color).
        #
        # Note:
        #   The sender may provide `sem_stable_rgb_u8` (an RGB mask). We intentionally ignore
        #   that here to avoid any palette mismatch between RAW and STABLE panels.
        # ---------------------------------------------------------------------
        if label_raw_for_viz is not None:
            raw_sem_rgb = _hash_colorize(label_raw_for_viz)
            raw_sem_overlay = _alpha_blend_rgb(rgb_sem_base, raw_sem_rgb, float(overlay_alpha))
        else:
            raw_sem_overlay = rgb_sem_base.copy()

        if label_stable_for_viz is not None:
            stable_sem_rgb = _hash_colorize(label_stable_for_viz)
            stable_sem_overlay = _alpha_blend_rgb(rgb_sem_base, stable_sem_rgb, float(overlay_alpha))
        else:
            stable_sem_overlay = rgb_sem_base.copy()

        # ---------------------------------------------------------------------
        # Sample a fixed grid (default 3x3) and compute patch mean depth.
        # ---------------------------------------------------------------------
        centers = _compute_grid_centers(h, w, int(sample_grid))
        means: List[float] = []
        for (cy, cx) in centers:
            if z_use is None or valid_use is None:
                means.append(float("nan"))
            else:
                means.append(
                    _patch_mean_depth(z_use, valid_use, cy=int(cy), cx=int(cx), patch=int(sample_patch))
                )

        # Draw patch boxes on all three panels (so you can correlate semantics and depth).
        p = int(max(1, int(sample_patch)))
        if p % 2 == 0:
            p += 1
        r = p // 2
        for (cy, cx) in centers:
            x0, y0 = int(cx - r), int(cy - r)
            x1, y1 = int(cx + r), int(cy + r)
            cv2.rectangle(raw_sem_overlay, (x0, y0), (x1, y1), (255, 0, 0), 1)
            cv2.rectangle(stable_sem_overlay, (x0, y0), (x1, y1), (255, 0, 0), 1)
            cv2.rectangle(depth_overlay, (x0, y0), (x1, y1), (255, 0, 0), 1)

        # ---------------------------------------------------------------------
        # Semantic class nearest markers (yellow squares).
        #
        # IMPORTANT (per user request):
        #   We want an A/B comparison where the LEFT panel and the MIDDLE panel can have
        #   DIFFERENT yellow markers:
        #     - LEFT  markers are computed from the LEFT panel semantics (unfiltered RAW).
        #     - MID   markers are computed from the MID panel semantics (filtered RAW).
        #   The printed list is based on the FILTERED (MID) markers.
        #
        # Design:
        #   - We compute both marker sets using the CURRENT depth map shown in the depth panel (`z_use`).
        #   - We draw each marker set only on its corresponding semantic panel.
        #   - We also draw the FILTERED marker set on the DEPTH panel, since the depth panel is used
        #     for the readout and should be consistent with the printed list.
        # ---------------------------------------------------------------------
        sem_nearest_raw: List[Tuple[int, float, int, int]] = []
        sem_nearest_filt: List[Tuple[int, float, int, int]] = []
        if z_use is not None and valid_use is not None:
            if label_raw_for_viz is not None:
                sem_nearest_raw = _topk_semantic_nearest(
                    label_hw=label_raw_for_viz,
                    depth_hw=z_use,
                    valid_hw=valid_use,
                    k=int(semantic_topk),
                )
            if label_stable_for_viz is not None:
                sem_nearest_filt = _topk_semantic_nearest(
                    label_hw=label_stable_for_viz,
                    depth_hw=z_use,
                    valid_hw=valid_use,
                    k=int(semantic_topk),
                )

        # Helper: draw a set of yellow markers onto one or more panels.
        #
        # Yellow in RGB is (255,255,0).
        # The yellow marker size must match the red marker size (half-size `r`).
        def _draw_yellow_markers(
            markers: List[Tuple[int, float, int, int]], panels: Tuple[np.ndarray, ...]
        ) -> None:
            for lab_id, _d, x, y in markers:
                x = int(x)
                y = int(y)
                r0 = int(max(1, min(int(box_radius), int(r))))
                for panel in panels:
                    cv2.rectangle(panel, (x - r0, y - r0), (x + r0, y + r0), (255, 255, 0), 1)

                # Add a small numeric label next to the yellow marker (class id).
                txt = str(int(lab_id))
                tx = int(min(max(0, x + r0 + 2), w - 1))
                ty = int(min(max(10, y - r0 - 2), h - 1))
                for panel in panels:
                    cv2.putText(
                        panel,
                        txt,
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.30,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        panel,
                        txt,
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.30,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

        # LEFT: markers computed from unfiltered RAW semantics.
        _draw_yellow_markers(sem_nearest_raw, (raw_sem_overlay,))
        # MID + DEPTH: markers computed from filtered semantics (used for the printed list).
        _draw_yellow_markers(sem_nearest_filt, (stable_sem_overlay, depth_overlay))

        # ---------------------------------------------------------------------
        # Text overlay: put grid depth samples on the DEPTH panel (keep 3-column layout).
        # ---------------------------------------------------------------------
        header = f"fid={fid} mode={mode} depth={depth_src}"
        cv2.putText(
            depth_overlay,
            header,
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            depth_overlay,
            header,
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y = 34
        for i, v in enumerate(means[:16]):
            txt = f"p{i:02d}: {v:5.2f}m" if np.isfinite(v) else f"p{i:02d}:  nan"
            cv2.putText(
                depth_overlay,
                txt,
                (5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                depth_overlay,
                txt,
                (5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            y += 12

        # Print top-K semantic classes by nearest depth under the grid values.
        # Print the FILTERED markers list (middle-panel semantics).
        sem_nearest = sem_nearest_filt
        if sem_nearest:
            y += 4
            title = f"semantic_top{min(int(semantic_topk), len(sem_nearest))}:"
            for line, color in [(title, (255, 255, 255)), (title, (0, 0, 0))]:
                cv2.putText(
                    depth_overlay,
                    line,
                    (5, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            y += 12
            for i, (lab_id, d, x, ypix) in enumerate(sem_nearest[: int(semantic_topk)]):
                line = f"{i:02d} id={int(lab_id)} z={float(d):.2f} ({int(x)},{int(ypix)})"
                for txt, color in [(line, (255, 255, 255)), (line, (0, 0, 0))]:
                    cv2.putText(
                        depth_overlay,
                        txt,
                        (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                y += 10

        # Compose: [raw semantic | stable semantic | depth]
        composite_rgb = np.concatenate([raw_sem_overlay, stable_sem_overlay, depth_overlay], axis=1)

        # Convert to BGR for OpenCV window.
        vis = cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)
        if int(scale) != 1:
            vis = cv2.resize(
                vis,
                (vis.shape[1] * int(scale), vis.shape[0] * int(scale)),
                interpolation=cv2.INTER_NEAREST,
            )

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break

    cv2.destroyWindow(window_name)
