"""
Depth stabilization utilities for MASt3R-SLAM (planning-oriented, real-time first).

This module intentionally focuses on *very lightweight* operations that can run online:

  - Depth source selection (raw per-frame depth vs. keyframe-warp depth).
  - A simple, fast hole-filling method that does NOT re-introduce per-frame depth jitter.

Important design goals:
  - No per-pixel Python loops; everything is vectorized in torch.
  - No sorting/unique; O(N) scatter + a tiny fixed number of 2D convolutions.
  - The output is meant for downstream tasks (e.g., planning) that prefer "stable" depth
    over perfect instantaneous accuracy.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def warp_keyframe_depth_to_frame_v0(
    *,
    Xk_canon: torch.Tensor,
    T_CkCf,
    idx_k2f: torch.Tensor,
    q: torch.Tensor,
    valid: torch.Tensor,
    tau: float,
    size_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warp keyframe depth into the current frame using existing k->f pixel matches.

    Inputs:
      Xk_canon: (H*W, 3) float  - keyframe canonical pointmap in keyframe camera coordinates.
      T_CkCf:   Sim3/SE3-like   - relative pose mapping current frame -> keyframe (Cf -> Ck).
                                 In this codebase it is `T_CkCf` returned by tracking.
      idx_k2f:  (H*W) int64     - for each keyframe pixel k, the matched frame pixel index f.
      q:        (H*W) float     - match confidence per keyframe pixel.
      valid:    (H*W) bool      - match validity per keyframe pixel.
      tau:      float           - confidence threshold for writing warped depth.
      size_hw:  (H, W)          - match grid resolution.

    Outputs:
      depth_hw: (H, W) float32  - warped depth in the *current frame camera* (z after transform).
      mask_hw:  (H, W) bool     - True where depth_hw is valid (written by some match).

    Notes:
      - This is NOT a true z-buffer rendering; it is a fast match-based warp.
      - Duplicates in idx_k2f (multiple k mapping to the same f) will be "last write wins".
        We accept this to keep the method O(N) and real-time friendly.
    """

    h, w = (int(size_hw[0]), int(size_hw[1]))
    n = h * w

    Xk = Xk_canon.reshape(-1, 3)
    if Xk.shape[0] != n:
        raise ValueError(f"Expected Xk_canon length H*W={n}, got {Xk.shape[0]}")

    idx_k2f = idx_k2f.reshape(-1).to(torch.int64)
    q = q.reshape(-1).to(torch.float32)
    valid = valid.reshape(-1).to(torch.bool)
    if idx_k2f.numel() != n or q.numel() != n or valid.numel() != n:
        raise ValueError("idx_k2f/q/valid must be length H*W")

    # Relative pose in this repo maps Cf -> Ck (current frame to keyframe).
    # For warping keyframe points into the current frame, we need the inverse: Ck -> Cf.
    T_CfCk = T_CkCf.inv()

    # Transform keyframe points into the current frame camera coordinates.
    Xf_from_k = T_CfCk.act(Xk)
    z = Xf_from_k[:, 2].to(torch.float32)

    # Scatter warped depth into frame pixels for confident, valid matches.
    mask_k = valid & (q > float(tau))
    depth_f = torch.zeros((n,), device=Xk.device, dtype=torch.float32)
    mask_f = torch.zeros((n,), device=Xk.device, dtype=torch.bool)
    if mask_k.any():
        f_idx = idx_k2f[mask_k]
        depth_f[f_idx] = z[mask_k]
        mask_f[f_idx] = True

    return depth_f.view(h, w), mask_f.view(h, w)


def fill_holes_by_neighbor_average_v0(
    *,
    depth_hw: torch.Tensor,
    mask_hw: torch.Tensor,
    iters: int = 2,
    kernel_size: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extremely lightweight hole filling by iterative neighbor averaging (no raw depth).

    Why this method:
      - Planning typically prefers stable depth without high-frequency jitter.
      - Filling holes using per-frame raw depth re-introduces jitter, so we instead propagate
        the *warped/map* depth locally.

    Algorithm (per iteration):
      - Let valid be mask_hw in {0,1}.
      - Compute neighbor_count = conv2d(valid, ones)
      - Compute neighbor_sum   = conv2d(depth*valid, ones)
      - For holes with neighbor_count>0:
          depth = neighbor_sum / neighbor_count
          valid = 1

    Inputs:
      depth_hw: (H, W) float32
      mask_hw:  (H, W) bool
      iters:    int    - number of propagation iterations (small, e.g., 1-4).
      kernel_size: int - neighborhood size (default 3).

    Outputs:
      depth_filled: (H, W) float32
      mask_filled:  (H, W) bool

    Notes / pitfalls:
      - This is not edge-aware; it may blur across depth discontinuities. For the "very fast"
        goal, this is acceptable as a first step.
      - Use a tiny number of iterations to keep runtime minimal.
    """

    if depth_hw.ndim != 2 or mask_hw.ndim != 2:
        raise ValueError(f"Expected (H,W) depth/mask, got {tuple(depth_hw.shape)} {tuple(mask_hw.shape)}")
    if depth_hw.shape != mask_hw.shape:
        raise ValueError("depth_hw and mask_hw must have the same shape")

    iters = int(iters)
    if iters <= 0:
        return depth_hw, mask_hw
    kernel_size = int(kernel_size)
    if kernel_size % 2 != 1 or kernel_size <= 1:
        raise ValueError("kernel_size must be an odd integer >= 3")

    device = depth_hw.device
    depth = depth_hw.to(torch.float32)
    valid = mask_hw.to(torch.float32)

    # Build a fixed all-ones kernel for neighbor aggregation.
    k = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=torch.float32)
    pad = kernel_size // 2

    # Work in NCHW to use conv2d.
    depth_nchw = depth[None, None, ...]
    valid_nchw = valid[None, None, ...]

    for _ in range(iters):
        neighbor_count = F.conv2d(valid_nchw, k, padding=pad)
        neighbor_sum = F.conv2d(depth_nchw * valid_nchw, k, padding=pad)

        # Identify holes that can be filled in this iteration.
        #
        # Performance note:
        #   We intentionally avoid Python-side early exits (`.item()`, `.any()`) to prevent
        #   GPU synchronization. We run a small, fixed number of iterations instead.
        holes = (valid_nchw < 0.5) & (neighbor_count > 0.5)
        filled_values = neighbor_sum / torch.clamp_min(neighbor_count, 1.0)
        depth_nchw = torch.where(holes, filled_values, depth_nchw)
        valid_nchw = torch.where(holes, torch.ones_like(valid_nchw), valid_nchw)

    return depth_nchw[0, 0], (valid_nchw[0, 0] > 0.5)
