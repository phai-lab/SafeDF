"""
Semantic stabilization utilities for MASt3R-SLAM.

This module implements the *minimal-change* and *real-time-first* semantic pipeline:

(A) Semantic warp + fuse (V0, hard labels only, O(N), no per-frame sorting):
    - Use SLAM's already-computed k->f pixel matches to warp the *stable* keyframe semantics
      onto the current frame.
    - Then "fuse" by a simple overwrite rule:
        if a current pixel receives a confident warp -> take warped label
        else -> keep the per-frame segmentation output (hole filling).

(B) Optional semantic reweighting for geometry (V1.5, hard labels, symmetric +/- beta):
    - If labels agree for a match, multiply the match weight by (1 + beta).
    - If labels disagree, multiply the match weight by (1 - beta).
    - IMPORTANT: The semantic factor must NOT change match validity/gating (q thresholds)
      nor robust gating (Huber). It is designed to be applied ONLY on the final residual
      weight (i.e., after any gating and robust kernel computation).

All heavy operations are vectorized in torch; there is no Python for-loop and no argsort/unique.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to target device if needed (no-op if already on that device)."""

    if x.device == device:
        return x
    return x.to(device)


def _as_uint8_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB tensor to uint8 [0,255] without changing layout.

    Accepted input:
      - float in [0,1] or [0,255]
      - uint8 already

    Output:
      - uint8 with the same shape as input.

    Pitfall:
      - If the input is float, values may not be exact multiples of 1/255 (e.g., after resize).
        We use rounding to recover the nearest byte value.
    """

    if rgb.dtype == torch.uint8:
        return rgb

    rgb_f = rgb
    # Heuristic: treat values in [0,1] as normalized, otherwise assume already in [0,255].
    if rgb_f.is_floating_point():
        maxv = float(rgb_f.max().item()) if rgb_f.numel() > 0 else 0.0
        if maxv <= 1.0 + 1e-6:
            rgb_f = rgb_f * 255.0
        rgb_u8 = torch.clamp(rgb_f.round(), 0, 255).to(torch.uint8)
        return rgb_u8

    # Integer types (e.g., int16/int32): clamp then cast.
    return torch.clamp(rgb_f, 0, 255).to(torch.uint8)


def rgb_to_label_code(rgb_hwc: torch.Tensor) -> torch.Tensor:
    """
    Pack an RGB image (H, W, 3) into a single-channel "label code" (H, W) int64.

    Label code definition:
      code = (R << 16) | (G << 8) | B

    This is a *palette-agnostic* representation:
      - If your segmentation is stored as an RGB mask, this code uniquely represents the color.
      - If your segmentation is stored as class IDs, you can also use those IDs directly instead.

    Inputs:
      rgb_hwc: (H, W, 3) tensor, float or uint8.

    Outputs:
      label_code: (H, W) int64.
    """

    if rgb_hwc.ndim != 3 or rgb_hwc.shape[-1] != 3:
        raise ValueError(f"Expected RGB HWC (H,W,3), got shape={tuple(rgb_hwc.shape)}")

    rgb_u8 = _as_uint8_rgb(rgb_hwc)
    r = rgb_u8[..., 0].to(torch.int64)
    g = rgb_u8[..., 1].to(torch.int64)
    b = rgb_u8[..., 2].to(torch.int64)
    return (r << 16) | (g << 8) | b


def label_code_to_rgb(label_code_hw: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Unpack a label code (H, W) back to an RGB image (H, W, 3) in float [0,1].

    This is the inverse of `rgb_to_label_code` *for RGB-derived codes*.
    If you choose to encode class IDs into the code space, this will produce a deterministic
    but not necessarily "nice" visualization color (low IDs can look dark).
    """

    if label_code_hw.ndim != 2:
        raise ValueError(f"Expected label_code (H,W), got shape={tuple(label_code_hw.shape)}")

    code = label_code_hw.to(torch.int64)
    r = ((code >> 16) & 255).to(torch.uint8)
    g = ((code >> 8) & 255).to(torch.uint8)
    b = (code & 255).to(torch.uint8)
    rgb = torch.stack([r, g, b], dim=-1).to(dtype=dtype) / 255.0
    return rgb


def resize_label_nearest(label: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Nearest-neighbor resize for either:
      - hard label (H,W) -> (H2,W2)
      - RGB mask  (H,W,3) -> (H2,W2,3)

    Notes:
      - Nearest is mandatory for labels to avoid mixing classes/colors.
      - This runs on whatever device the input lives on (CPU/GPU).
    """

    th, tw = size_hw
    if label.ndim == 2:
        x = label[None, None].float()
        x = F.interpolate(x, size=(th, tw), mode="nearest")
        return x[0, 0].to(label.dtype)

    if label.ndim == 3 and label.shape[-1] == 3:
        # HWC -> NCHW
        x = label.permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=(th, tw), mode="nearest")
        return x[0].permute(1, 2, 0).to(label.dtype)

    raise ValueError(f"Unsupported label shape for resize: {tuple(label.shape)}")


def ensure_hard_label_hw(
    semantic: Optional[torch.Tensor],
    *,
    size_hw: Tuple[int, int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Convert a semantic tensor into a hard label tensor (H,W) int64 on the target device.

    Accepted semantic formats:
      1) Hard label: (H,W) integer type (class IDs).
      2) RGB mask:   (H,W,3) float/uint8 (palette colors).

    Returned format:
      - (H,W) int64 "hard label".
        *If input was RGB*, we return a palette-agnostic "label code" packed from RGB bytes.
        *If input was integer IDs*, we return those IDs as int64.

    Why we use label_code for RGB:
      - It avoids needing to know the dataset palette / number of classes.
      - Equality checks are exact and fast.
    """

    if semantic is None:
        return None

    semantic = _to_device(semantic, device)

    if semantic.ndim == 2:
        # Hard class IDs (H,W)
        if tuple(semantic.shape) != tuple(size_hw):
            semantic = resize_label_nearest(semantic, size_hw)
        return semantic.to(torch.int64)

    if semantic.ndim == 3 and semantic.shape[-1] == 3:
        # RGB palette mask (H,W,3)
        if tuple(semantic.shape[:2]) != tuple(size_hw):
            semantic = resize_label_nearest(semantic, size_hw)
        return rgb_to_label_code(semantic)

    raise ValueError(f"Unsupported semantic tensor shape: {tuple(semantic.shape)}")


def semantic_warp_and_fuse_v0(
    *,
    label_vit_f: torch.Tensor,
    label_stable_k: torch.Tensor,
    idx_k2f: torch.Tensor,
    q: torch.Tensor,
    valid: torch.Tensor,
    tau_warp: float,
) -> torch.Tensor:
    """
    V0 semantic warp + fuse (hard label only, O(N), no sort/unique, no Python loops).

    Inputs:
      label_vit_f:   (H,W) int64 - current-frame per-image segmentation (raw, may flicker).
      label_stable_k:(H,W) int64 - keyframe stable segmentation cache.
      idx_k2f:       (H*W) int64 - for each keyframe pixel k, the matched current-frame pixel f.
      q:             (H*W) float - match confidence per keyframe pixel k.
      valid:         (H*W) bool  - whether the k->f match is valid.
      tau_warp:      float      - confidence threshold for applying the warp overwrite.

    Output:
      label_stable_f:(H,W) int64 - stabilized current-frame segmentation.

    Algorithm (exactly as requested):
      - Start from per-frame segmentation as default (hole filling).
      - Overwrite only for matches that pass mask = valid & (q > tau_warp):
            f_idx = idx_k2f[mask]
            k_idx = arange(H*W)[mask]
            label_stable_f_flat[f_idx] = label_stable_k_flat[k_idx]

    Important pitfall (accepted in V0):
      - idx_k2f can contain duplicates (multiple k map to the same f).
        This overwrite is "last write wins" in the given index order.
        We explicitly avoid sorting to keep O(N) and satisfy the constraint.
    """

    if label_vit_f.ndim != 2 or label_stable_k.ndim != 2:
        raise ValueError(
            f"Expected (H,W) labels, got vit={tuple(label_vit_f.shape)} k={tuple(label_stable_k.shape)}"
        )
    if label_vit_f.shape != label_stable_k.shape:
        raise ValueError(
            f"Label shapes must match, got vit={tuple(label_vit_f.shape)} k={tuple(label_stable_k.shape)}"
        )

    h, w = label_vit_f.shape
    n = h * w

    idx_k2f = idx_k2f.reshape(-1).to(torch.int64)
    q = q.reshape(-1).to(torch.float32)
    valid = valid.reshape(-1).to(torch.bool)

    if idx_k2f.numel() != n or q.numel() != n or valid.numel() != n:
        raise ValueError(
            "idx_k2f/q/valid must be length H*W; "
            f"got H*W={n}, idx={idx_k2f.numel()}, q={q.numel()}, valid={valid.numel()}"
        )

    # Start from the per-frame segmentation as the default (hole filling).
    label_stable_f = label_vit_f.clone()

    # Apply a simple confidence+validity gate for warp overwrite.
    mask = valid & (q > float(tau_warp))
    if mask.any():
        # Vectorized overwrite: no Python loop, no sorting.
        f_idx = idx_k2f[mask]
        # k_idx are the linear indices of keyframe pixels that passed the mask.
        k_idx = torch.arange(n, device=idx_k2f.device, dtype=torch.int64)[mask]
        label_stable_f.reshape(-1)[f_idx] = label_stable_k.reshape(-1)[k_idx]

    return label_stable_f


def semantic_pointmap_update_v0(
    *,
    kf_sem_label: torch.Tensor,
    kf_sem_weight: torch.Tensor,
    raw_label_f: torch.Tensor,
    idx_k2f: torch.Tensor,
    q: torch.Tensor,
    valid: torch.Tensor,
    use_q: bool = True,
    momentum: float = 1.0,
) -> None:
    """
    V3 semantic pointmap update (keyframe fusion) with a lightweight streaming rule.

    This function updates the *keyframe-side* semantic cache in-place, using already-computed
    SLAM pixel matches and per-match confidence.

    Inputs (shapes are for the MASt3R match grid, i.e., the same H×W used for pointmap fusion):
      kf_sem_label:  (H,W) int64   - keyframe semantic label cache (one label per pixel).
      kf_sem_weight: (H,W) float32 - keyframe semantic weight/vote cache (one scalar per pixel).
      raw_label_f:   (H,W) int64   - current-frame raw segmentation hard labels (EfficientViT output).
      idx_k2f:       (H*W) int64   - for each keyframe pixel k, the matched current-frame pixel f.
      q:             (H*W) float   - match confidence per keyframe pixel k.
      valid:         (H*W) bool    - whether the k->f match is valid.
      use_q:         bool          - if True, update magnitude u := q; else u := 1.0.
      momentum:      float         - per-update decay factor mu in (0, 1]. When mu < 1, we apply
                                     a "touch-only" decay:
                                       kf_sem_weight[k_idx] *= mu
                                     only for keyframe pixels that are touched by valid matches
                                     in the current frame. This adds inertia without a full H×W
                                     decay pass.

    Update rule (exactly as requested, per keyframe pixel n=k):
      Let y = raw_label_f[m] where m = idx_k2f[k], and u = q[k] (or 1.0 if use_q=False).
        if y == kf_sem_label[k]:
            kf_sem_weight[k] += u
        else:
            kf_sem_weight[k] -= u
            if kf_sem_weight[k] < 0:
                kf_sem_label[k]  = y
                kf_sem_weight[k] = -kf_sem_weight[k]

    Why this is fast / minimal:
      - O(H*W) vectorized update, no sorting and no Python loops.
      - No 80-dim logits/probabilities are stored; keyframe cache stays label+scalar weight only.

    Important pitfalls / assumptions:
      - We assume one match per keyframe pixel (which holds for the existing idx_k2f structure),
        so there are no atomic write conflicts on the keyframe side.
      - `idx_k2f` may contain invalid indices for invalid matches; we guard with `valid`.
      - This function does not change any SLAM match gating/valid selection. It only updates
        the semantic cache for later use (e.g., keyframe->frame warp).

    Output:
      None. (Updates kf_sem_label / kf_sem_weight in-place.)
    """

    if kf_sem_label.ndim != 2 or kf_sem_weight.ndim != 2 or raw_label_f.ndim != 2:
        raise ValueError(
            "Expected (H,W) tensors for kf_sem_label/kf_sem_weight/raw_label_f, "
            f"got label={tuple(kf_sem_label.shape)} weight={tuple(kf_sem_weight.shape)} raw={tuple(raw_label_f.shape)}"
        )
    if kf_sem_label.shape != kf_sem_weight.shape or kf_sem_label.shape != raw_label_f.shape:
        raise ValueError(
            "Semantic pointmap tensors must share the same (H,W) shape, "
            f"got label={tuple(kf_sem_label.shape)} weight={tuple(kf_sem_weight.shape)} raw={tuple(raw_label_f.shape)}"
        )

    h, w = kf_sem_label.shape
    n = h * w

    idx_k2f = idx_k2f.reshape(-1).to(torch.int64)
    q = q.reshape(-1).to(torch.float32)
    valid = valid.reshape(-1).to(torch.bool)
    if idx_k2f.numel() != n or q.numel() != n or valid.numel() != n:
        raise ValueError(
            "idx_k2f/q/valid must be length H*W; "
            f"got H*W={n}, idx={idx_k2f.numel()}, q={q.numel()}, valid={valid.numel()}"
        )

    # Flatten views for vectorized per-pixel update on the keyframe side.
    k_label = kf_sem_label.reshape(-1)
    k_weight = kf_sem_weight.reshape(-1)
    f_label = raw_label_f.reshape(-1)

    mask = valid
    if not mask.any():
        return

    k_idx = torch.arange(n, device=idx_k2f.device, dtype=torch.int64)[mask]
    f_idx = idx_k2f[mask]

    mu = float(momentum)
    if mu <= 0.0 or mu > 1.0:
        raise ValueError(f"Expected semantic_pointmap momentum mu in (0, 1], got mu={mu}")

    # Optional "touch-only" momentum/decay:
    #   We decay only the pixels that receive an observation in this frame, so the update remains
    #   O(#valid) rather than O(H*W). This helps prevent frequent flip-flops when observations are noisy,
    #   while keeping the implementation real-time friendly.
    if mu < 1.0:
        k_weight[k_idx] = k_weight[k_idx] * mu

    # Gather current-frame observation y for each valid keyframe pixel k.
    y = f_label[f_idx].to(torch.int64)

    # Update magnitude u: either match confidence q or constant 1.0.
    if use_q:
        u = q[mask]
    else:
        u = torch.ones_like(q[mask])

    # Current cache state for the affected keyframe pixels.
    cur_label = k_label[k_idx]
    cur_weight = k_weight[k_idx]

    same = y == cur_label
    # +u when same, -u when different
    new_weight = cur_weight + torch.where(same, u, -u)

    # If the running weight becomes negative, we flip the winner label to y and mirror the weight.
    flip = new_weight < 0.0
    new_label = torch.where(flip, y, cur_label)
    new_weight = torch.where(flip, -new_weight, new_weight)

    k_label[k_idx] = new_label
    k_weight[k_idx] = new_weight


def semantic_bonus_sqrt_factor_v0(
    *,
    label_k: torch.Tensor,
    label_f: torch.Tensor,
    idx_k2f: torch.Tensor,
    q: torch.Tensor,
    valid: torch.Tensor,
    beta: float,
    tau_sem: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute a per-match *sqrt* semantic reweighting factor for geometry weighting (V1.5 symmetric +/- beta).

    This returns sqrt_factor such that if your original residual weight is w (in a least squares sense),
    you can apply:
        same = I[label_f[idx_k2f[k]] == label_k[k]]
        g    = (1 + beta) if same else (1 - beta)
        w'   = w * g
    by multiplying the *square-root information* (sqrt(w)) by:
        sqrt_factor = sqrt(g)

    Inputs:
      label_k:  (H,W) int64 - keyframe labels (stable or raw, but should be consistent across frames).
      label_f:  (H,W) int64 - frame labels (stable or raw, controlled by ablation switch).
      idx_k2f:  (H*W) int64 - k->f pixel mapping (linear indices).
      q:        (H*W) float - match confidence (used only for optional tau_sem gate).
      valid:    (H*W) bool  - match validity (used as a safety gate for semantic reweighting).
      beta:     float       - semantic strength in [0, 1). Values outside are clamped for safety.
      tau_sem:  float|None  - (optional) only apply semantic factor when q > tau_sem
                              (does NOT affect match validity).

    Output:
      sqrt_factor: (H*W, 1) float32 on the same device as idx_k2f.

    Design constraints satisfied:
      - "No gating change": this factor is intended to be multiplied *after* any match validity logic.
        (In our integration we apply it after robust weighting as well, so it cannot influence Huber gating.)
      - O(N), no sorting, no Python loops.
    """

    if label_k.ndim != 2 or label_f.ndim != 2:
        raise ValueError(
            f"Expected (H,W) labels, got k={tuple(label_k.shape)} f={tuple(label_f.shape)}"
        )
    if label_k.shape != label_f.shape:
        raise ValueError(f"Label shapes must match, got k={tuple(label_k.shape)} f={tuple(label_f.shape)}")

    h, w = label_k.shape
    n = h * w

    idx_k2f = idx_k2f.reshape(-1).to(torch.int64)
    q = q.reshape(-1).to(torch.float32)
    valid = valid.reshape(-1).to(torch.bool)

    if idx_k2f.numel() != n or q.numel() != n or valid.numel() != n:
        raise ValueError(
            "idx_k2f/q/valid must be length H*W; "
            f"got H*W={n}, idx={idx_k2f.numel()}, q={q.numel()}, valid={valid.numel()}"
        )

    beta = float(beta)
    if beta < 0.0:
        # Clamp for robustness: negative beta would swap the intended meaning of "same/different".
        beta = 0.0
    if beta >= 1.0:
        # Clamp for robustness: mismatched semantics use (1 - beta), which must stay >= 0
        # to keep sqrt(g) real-valued.
        beta = 1.0 - 1e-6

    # Default: neutral factor everywhere (g=1 -> sqrt_factor=1).
    sqrt_factor = torch.ones((n, 1), device=idx_k2f.device, dtype=torch.float32)

    # Apply semantic factor only where matches are valid (and optionally q is high enough).
    #
    # IMPORTANT:
    #   This mask is NOT the match gating used by tracking/back-end optimization.
    #   It only decides where we apply the semantic factor. When the mask is false,
    #   the factor stays 1 (i.e., semantics has no effect for that match).
    mask = valid
    if tau_sem is not None:
        mask = mask & (q > float(tau_sem))

    if mask.any():
        k_idx = torch.arange(n, device=idx_k2f.device, dtype=torch.int64)[mask]
        f_idx = idx_k2f[mask]

        # same[k] compares keyframe pixel k against its matched frame pixel f.
        same = label_f.reshape(-1)[f_idx] == label_k.reshape(-1)[k_idx]
        # V1.5 symmetric reweighting:
        #   g = (1 + beta)  if same
        #       (1 - beta)  otherwise
        #
        # We implement this without branching via:
        #   g = (1 - beta) + 2 * beta * same
        # where `same` is 0/1.
        g = (1.0 - beta) + (2.0 * beta) * same.to(torch.float32)
        sqrt_factor[k_idx, 0] = torch.sqrt(g)

    return sqrt_factor
