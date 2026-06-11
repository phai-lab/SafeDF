"""
Rolling local TSDF (+ optional semantic) fusion for downstream planning.

Why this module exists
----------------------
In MASt3R-SLAM we already maintain a *keyframe-centric* geometric map (per-keyframe pointmaps)
and (optionally) a semantic cache. For CBF / planning, however, a **volumetric** map is often
more convenient than a raw point cloud:
  - TSDF provides a smooth implicit surface representation.
  - From TSDF we can later derive occupancy and ESDF if needed.
  - A voxel grid is also a natural place to fuse noisy per-frame hard-label semantics.

Design goals (real-time oriented)
--------------------------------
1) Local rolling volume:
   We keep a fixed-size volume (e.g., radius 2m) centered around the *current* camera position.
   This avoids unbounded memory and keeps per-update compute bounded.

2) Vectorized CPU update:
   Each TSDF integration step is O(#voxels) and fully vectorized in NumPy.
   With small volumes (e.g., 41^3 ~ 69k voxels for 2m@0.1m) this is feasible at a few Hz.

3) Sim3-aware poses:
   MASt3R-SLAM uses lietorch.Sim3 for poses (camera->world) where the 3x3 block can be `s*R`.
   We therefore always use the full 3x3 `A` block and its inverse for world<->camera transforms.

4) Optional lightweight semantic fusion:
   We do NOT store a full C-way probability per voxel (too heavy).
   Instead we store only:
     - `sem_label`  : int32 label id
     - `sem_weight` : float32 vote/weight
   Update rule is the same "streaming vote with sign flip" used elsewhere in this repo:
     if y == sem_label: sem_weight += u
     else:              sem_weight -= u; if sem_weight < 0: sem_label=y; sem_weight=-sem_weight

Inputs / outputs
----------------
Inputs per integration step:
  - depth_hw   : (H,W) float32, depth in meters (camera Z)
  - valid_hw   : (H,W) bool, validity mask (optional)
  - label_hw   : (H,W) int32/int64, semantic hard labels (optional)
  - T_WC_f32   : (4,4) float32, Sim3 camera->world matrix from SLAM
  - K_f32      : (3,3) float32 intrinsics matching (H,W)

Outputs stored in the volume:
  - tsdf       : (Nx,Ny,Nz) float32 in [-1, 1]
  - weight     : (Nx,Ny,Nz) float32 (accumulated integration weight)
  - sem_label  : (Nx,Ny,Nz) int32
  - sem_weight : (Nx,Ny,Nz) float32

Important note about "every frame"
----------------------------------
This module can be fed at *any* rate (e.g., every video frame, or at the semantic FPS).
In this repo, it is typically run in a **separate process** that periodically reads the latest
SharedStates frame (pose + pointmap Z + semantic label) and integrates it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _sim3_inv_A_t(M_WC: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the Sim3 3x3 block and translation and return (A_inv, t_wc).

    The pose is assumed to be a camera->world Sim3 matrix:
      X_w = A * X_c + t
    where A = s*R (uniform scale * rotation).

    For row-vector batches we use:
      X_c = (X_w - t) @ A_inv.T

    Returns:
      A_inv: (3,3) float32
      t_wc : (3,)  float32
    """

    M = np.asarray(M_WC, dtype=np.float32).reshape(4, 4)
    A = M[:3, :3].astype(np.float32, copy=False)
    t = M[:3, 3].astype(np.float32, copy=False)
    # A is small (3x3); inversion cost is negligible compared to voxel projection.
    A_inv = np.linalg.inv(A).astype(np.float32)
    return A_inv, t


@dataclass
class TsdfVolumeSnapshot:
    """
    A small, explicit container for publishing/visualization.

    All arrays are on CPU (NumPy).
    """

    origin_w: np.ndarray  # (3,) float32, world coordinate of voxel (0,0,0) center
    voxel_m: float
    tsdf: np.ndarray  # (Nx,Ny,Nz) float32
    weight: np.ndarray  # (Nx,Ny,Nz) float32
    sem_label: np.ndarray  # (Nx,Ny,Nz) int32
    sem_weight: np.ndarray  # (Nx,Ny,Nz) float32
    rgb_color: Optional[np.ndarray] = None  # (Nx,Ny,Nz,3) float32 in [0,1]
    rgb_weight: Optional[np.ndarray] = None  # (Nx,Ny,Nz) float32


class RollingTsdfSemanticVolume:
    """
    Rolling local TSDF volume with optional semantic fusion.

    Coordinate convention
    ---------------------
    The volume is stored in a **world-aligned** axis system:
      axis-0 => world X
      axis-1 => world Y
      axis-2 => world Z

    Each voxel stores the TSDF value at the voxel center.

    Rolling / shifting
    ------------------
    We keep the volume centered around the current camera translation `t_wc`.
    When the camera moves more than ~0.5 voxel, we shift (integer-roll) the arrays and clear
    the newly exposed border region.

    This is a pragmatic real-time choice:
      - It avoids resampling / trilinear warps.
      - It keeps compute bounded.
      - For planning in the near field, this is usually sufficient.
    """

    def __init__(
        self,
        *,
        radius_m: float,
        voxel_m: float,
        trunc_m: Optional[float] = None,
        max_weight: float = 100.0,
        semantic_band_m: Optional[float] = None,
        invalid_label: int = -1,
    ) -> None:
        self.radius_m = float(radius_m)
        self.voxel_m = float(voxel_m)
        self.trunc_m = float(trunc_m) if trunc_m is not None else float(3.0 * self.voxel_m)
        self.max_weight = float(max_weight)
        self.semantic_band_m = float(semantic_band_m) if semantic_band_m is not None else float(self.voxel_m)
        self.invalid_label = int(invalid_label)

        if self.radius_m <= 0 or self.voxel_m <= 0:
            raise ValueError("radius_m and voxel_m must be positive.")
        if self.trunc_m <= 0:
            raise ValueError("trunc_m must be positive.")

        # Use an odd number of voxels so there is a well-defined center voxel.
        self._n_half = int(np.ceil(self.radius_m / self.voxel_m))
        n = 2 * self._n_half + 1
        self.dims = (n, n, n)

        # World origin of voxel (0,0,0) *center*.
        self.origin_w = np.zeros((3,), dtype=np.float32)
        self._center_w: Optional[np.ndarray] = None

        # Allocate voxel grids.
        self.tsdf = np.ones(self.dims, dtype=np.float32)
        self.weight = np.zeros(self.dims, dtype=np.float32)
        self.sem_label = np.full(self.dims, int(self.invalid_label), dtype=np.int32)
        self.sem_weight = np.zeros(self.dims, dtype=np.float32)
        self.rgb_color = np.zeros(self.dims + (3,), dtype=np.float32)
        self.rgb_weight = np.zeros(self.dims, dtype=np.float32)

        # Precompute voxel center offsets relative to `origin_w`.
        #
        # Shape: (V,3) float32 where columns are (x,y,z) in meters.
        nx, ny, nz = self.dims
        xs = (np.arange(nx, dtype=np.float32) * self.voxel_m).reshape(nx, 1, 1)
        ys = (np.arange(ny, dtype=np.float32) * self.voxel_m).reshape(1, ny, 1)
        zs = (np.arange(nz, dtype=np.float32) * self.voxel_m).reshape(1, 1, nz)
        # Broadcast to (nx,ny,nz,3) then flatten to (V,3).
        self._offsets_flat = np.stack(
            [
                np.broadcast_to(xs, (nx, ny, nz)),
                np.broadcast_to(ys, (nx, ny, nz)),
                np.broadcast_to(zs, (nx, ny, nz)),
            ],
            axis=-1,
        ).reshape(-1, 3)

    def reset(self, *, center_w: np.ndarray) -> None:
        """
        Clear the volume and re-center it at `center_w`.
        """

        c = np.asarray(center_w, dtype=np.float32).reshape(3)
        self._center_w = c.copy()
        self.origin_w = (c - float(self._n_half) * float(self.voxel_m)).astype(np.float32)
        self.tsdf.fill(1.0)
        self.weight.fill(0.0)
        self.sem_label.fill(np.int32(self.invalid_label))
        self.sem_weight.fill(0.0)
        self.rgb_color.fill(0.0)
        self.rgb_weight.fill(0.0)

    def _clear_border(self, *, shift_xyz: Tuple[int, int, int]) -> None:
        """
        Clear the newly exposed border region after an integer roll.

        Args:
          shift_xyz:
            The *origin shift* in voxel units (dx,dy,dz) where:
              origin_w += [dx,dy,dz] * voxel_m
            Internally we roll arrays by (-dx,-dy,-dz), so we clear the wrapped region.
        """

        dx, dy, dz = (int(shift_xyz[0]), int(shift_xyz[1]), int(shift_xyz[2]))
        nx, ny, nz = self.dims

        def _clear_x(ix0: int, ix1: int) -> None:
            self.tsdf[ix0:ix1, :, :].fill(1.0)
            self.weight[ix0:ix1, :, :].fill(0.0)
            self.sem_label[ix0:ix1, :, :].fill(np.int32(self.invalid_label))
            self.sem_weight[ix0:ix1, :, :].fill(0.0)
            self.rgb_color[ix0:ix1, :, :, :].fill(0.0)
            self.rgb_weight[ix0:ix1, :, :].fill(0.0)

        def _clear_y(iy0: int, iy1: int) -> None:
            self.tsdf[:, iy0:iy1, :].fill(1.0)
            self.weight[:, iy0:iy1, :].fill(0.0)
            self.sem_label[:, iy0:iy1, :].fill(np.int32(self.invalid_label))
            self.sem_weight[:, iy0:iy1, :].fill(0.0)
            self.rgb_color[:, iy0:iy1, :, :].fill(0.0)
            self.rgb_weight[:, iy0:iy1, :].fill(0.0)

        def _clear_z(iz0: int, iz1: int) -> None:
            self.tsdf[:, :, iz0:iz1].fill(1.0)
            self.weight[:, :, iz0:iz1].fill(0.0)
            self.sem_label[:, :, iz0:iz1].fill(np.int32(self.invalid_label))
            self.sem_weight[:, :, iz0:iz1].fill(0.0)
            self.rgb_color[:, :, iz0:iz1, :].fill(0.0)
            self.rgb_weight[:, :, iz0:iz1].fill(0.0)

        # If origin moved +dx, we rolled -dx, so the last dx slabs are new.
        if dx > 0:
            _clear_x(nx - dx, nx)
        elif dx < 0:
            _clear_x(0, -dx)

        if dy > 0:
            _clear_y(ny - dy, ny)
        elif dy < 0:
            _clear_y(0, -dy)

        if dz > 0:
            _clear_z(nz - dz, nz)
        elif dz < 0:
            _clear_z(0, -dz)

    def shift_to_center(self, *, center_w: np.ndarray) -> None:
        """
        Shift (roll) the volume so it stays centered around `center_w`.

        This uses an integer voxel shift (no resampling). If the required shift is huge
        (greater than the volume itself), we reset instead.
        """

        c = np.asarray(center_w, dtype=np.float32).reshape(3)
        if self._center_w is None:
            self.reset(center_w=c)
            return

        delta = (c - self._center_w).astype(np.float32)
        shift_vox = np.round(delta / float(self.voxel_m)).astype(np.int32)
        dx, dy, dz = int(shift_vox[0]), int(shift_vox[1]), int(shift_vox[2])
        if dx == 0 and dy == 0 and dz == 0:
            return

        nx, ny, nz = self.dims
        if abs(dx) >= nx or abs(dy) >= ny or abs(dz) >= nz:
            # The camera jumped too far; a roll would just wipe everything anyway.
            self.reset(center_w=c)
            return

        # Update center and origin in world coordinates.
        self._center_w = (self._center_w + shift_vox.astype(np.float32) * float(self.voxel_m)).astype(np.float32)
        self.origin_w = (self.origin_w + shift_vox.astype(np.float32) * float(self.voxel_m)).astype(np.float32)

        # Roll arrays so that values remain aligned with world coordinates after the origin shift.
        roll = (-dx, -dy, -dz)
        self.tsdf = np.roll(self.tsdf, shift=roll, axis=(0, 1, 2))
        self.weight = np.roll(self.weight, shift=roll, axis=(0, 1, 2))
        self.sem_label = np.roll(self.sem_label, shift=roll, axis=(0, 1, 2))
        self.sem_weight = np.roll(self.sem_weight, shift=roll, axis=(0, 1, 2))
        self.rgb_color = np.roll(self.rgb_color, shift=roll, axis=(0, 1, 2))
        self.rgb_weight = np.roll(self.rgb_weight, shift=roll, axis=(0, 1, 2))
        self._clear_border(shift_xyz=(dx, dy, dz))

    def integrate(
        self,
        *,
        depth_hw: np.ndarray,
        T_WC_f32: np.ndarray,
        K_f32: np.ndarray,
        valid_hw: Optional[np.ndarray] = None,
        label_hw: Optional[np.ndarray] = None,
        rgb_hw: Optional[np.ndarray] = None,
        obs_weight: float = 1.0,
    ) -> None:
        """
        Integrate a depth (+ optional semantic) observation into the TSDF volume.

        Args:
          depth_hw:
            (H,W) float32 depth in meters (camera Z).
          T_WC_f32:
            (4,4) float32 Sim3 camera->world.
          K_f32:
            (3,3) float32 intrinsics for the given (H,W).
          valid_hw:
            (H,W) bool mask for depth validity. If None, we infer from depth.
          label_hw:
            (H,W) int labels aligned with depth. If None, semantic fusion is skipped.
          obs_weight:
            Scalar integration weight for this observation (kept simple for real-time).
        """

        depth = np.asarray(depth_hw, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"depth_hw must be HxW, got {depth.shape}")
        H, W = int(depth.shape[0]), int(depth.shape[1])

        if valid_hw is None:
            valid = np.isfinite(depth) & (depth > 0.0)
        else:
            valid = np.asarray(valid_hw, dtype=bool)
            if valid.shape != depth.shape:
                raise ValueError(f"valid_hw shape {valid.shape} != depth shape {depth.shape}")
            valid = valid & np.isfinite(depth) & (depth > 0.0)

        # Center/shift the volume to the current camera translation (world).
        M = np.asarray(T_WC_f32, dtype=np.float32).reshape(4, 4)
        t_wc = M[:3, 3].astype(np.float32, copy=False)
        self.shift_to_center(center_w=t_wc)

        # Compute world->camera transform for Sim3.
        A_inv, t_wc = _sim3_inv_A_t(M)

        # Build voxel centers in world coordinates (flattened).
        Xw = (self._offsets_flat + self.origin_w.reshape(1, 3)).astype(np.float32, copy=False)
        # Transform to camera coordinates.
        Xc = (Xw - t_wc.reshape(1, 3)) @ A_inv.T
        z = Xc[:, 2].astype(np.float32, copy=False)
        in_front = np.isfinite(z) & (z > 1e-6) & np.isfinite(Xc).all(axis=1)
        if not np.any(in_front):
            return

        Xc = Xc[in_front]
        z = z[in_front]
        vox_lin = np.nonzero(in_front)[0].astype(np.int64)

        K = np.asarray(K_f32, dtype=np.float32).reshape(3, 3)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        u = (fx * (Xc[:, 0] / z) + cx).astype(np.int32)
        v = (fy * (Xc[:, 1] / z) + cy).astype(np.int32)
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(inside):
            return
        u = u[inside]
        v = v[inside]
        z = z[inside]
        vox_lin = vox_lin[inside]

        d = depth[v, u]
        m = valid[v, u]
        if not np.any(m):
            return
        d = d[m]
        z = z[m]
        vox_lin = vox_lin[m]
        u = u[m]
        v = v[m]

        # Signed distance along the camera ray.
        sdf = d - z

        # Standard TSDF truncation band update:
        #   update only voxels where sdf is within [-trunc, +trunc]
        # This updates a thin band in front of and behind the observed surface.
        trunc = float(self.trunc_m)
        band = (sdf >= -trunc) & (sdf <= trunc)
        if not np.any(band):
            return
        sdf = sdf[band]
        vox_lin = vox_lin[band]
        u = u[band]
        v = v[band]

        tsdf_obs = (sdf / trunc).astype(np.float32, copy=False)

        # Map flattened indices into 3D indices (x,y,z) for in-place updates.
        nx, ny, nz = self.dims
        # Flatten order corresponds to (x,y,z) with z changing fastest in our offsets construction.
        # We used reshape(-1,3) from (nx,ny,nz,3), so the flatten index matches:
        #   lin = ((x * ny) + y) * nz + z
        z_idx = (vox_lin % nz).astype(np.int32)
        y_idx = ((vox_lin // nz) % ny).astype(np.int32)
        x_idx = (vox_lin // (ny * nz)).astype(np.int32)

        # TSDF fusion (weighted average).
        w_obs = float(max(0.0, obs_weight))
        if w_obs <= 0.0:
            return

        w_old = self.weight[x_idx, y_idx, z_idx]
        tsdf_old = self.tsdf[x_idx, y_idx, z_idx]
        w_new = np.minimum(w_old + w_obs, float(self.max_weight)).astype(np.float32)
        tsdf_new = ((w_old * tsdf_old) + (w_obs * tsdf_obs)) / np.maximum(w_new, 1e-6)
        self.weight[x_idx, y_idx, z_idx] = w_new
        self.tsdf[x_idx, y_idx, z_idx] = np.clip(tsdf_new, -1.0, 1.0).astype(np.float32)

        # Surface-near selection for semantic/RGB fusion.
        surf_band = float(max(float(self.semantic_band_m), float(self.voxel_m)))
        if surf_band <= 0.0:
            return
        close = np.abs(sdf) <= surf_band
        if not np.any(close):
            return
        u_c = u[close]
        v_c = v[close]
        x_c = x_idx[close]
        y_c = y_idx[close]
        z_c = z_idx[close]

        if label_hw is not None:
            lab = np.asarray(label_hw)
            if lab.shape != (H, W):
                raise ValueError(f"label_hw shape {lab.shape} != depth shape {(H, W)}")

            y_obs = lab[v_c, u_c].astype(np.int32, copy=False)
            x_s, y_s, z_s = x_c, y_c, z_c
            if int(self.invalid_label) >= 0:
                valid_lab = y_obs != np.int32(self.invalid_label)
                if np.any(valid_lab):
                    y_obs = y_obs[valid_lab]
                    x_s = x_s[valid_lab]
                    y_s = y_s[valid_lab]
                    z_s = z_s[valid_lab]
                else:
                    y_obs = None
            if y_obs is not None:
                cur = self.sem_label[x_s, y_s, z_s]
                wv = self.sem_weight[x_s, y_s, z_s]
                same = cur == y_obs
                wv_new = wv + (w_obs * same.astype(np.float32)) - (w_obs * (~same).astype(np.float32))

                flip = wv_new < 0.0
                if np.any(flip):
                    cur = cur.copy()
                    wv_new = wv_new.copy()
                    cur[flip] = y_obs[flip]
                    wv_new[flip] = -wv_new[flip]

                self.sem_label[x_s, y_s, z_s] = cur
                self.sem_weight[x_s, y_s, z_s] = wv_new.astype(np.float32)

        if rgb_hw is not None:
            rgb = np.asarray(rgb_hw)
            if rgb.shape != (H, W, 3):
                raise ValueError(f"rgb_hw shape {rgb.shape} != expected {(H, W, 3)}")
            rgb_obs = rgb[v_c, u_c].astype(np.float32, copy=False)
            if rgb_obs.size > 0:
                if float(np.nanmax(rgb_obs)) > 1.0 + 1e-6:
                    rgb_obs = np.clip(rgb_obs / 255.0, 0.0, 1.0)
                else:
                    rgb_obs = np.clip(rgb_obs, 0.0, 1.0)
                w_old_rgb = self.rgb_weight[x_c, y_c, z_c]
                rgb_old = self.rgb_color[x_c, y_c, z_c, :]
                w_new_rgb = np.minimum(w_old_rgb + w_obs, float(self.max_weight)).astype(np.float32)
                rgb_new = ((w_old_rgb[:, None] * rgb_old) + (w_obs * rgb_obs)) / np.maximum(w_new_rgb[:, None], 1e-6)
                self.rgb_weight[x_c, y_c, z_c] = w_new_rgb
                self.rgb_color[x_c, y_c, z_c, :] = np.clip(rgb_new, 0.0, 1.0).astype(np.float32)

    def snapshot(self) -> TsdfVolumeSnapshot:
        """
        Return a lightweight snapshot (views) of the current volume state.

        Consumers that need strict immutability should copy the arrays themselves.
        """

        return TsdfVolumeSnapshot(
            origin_w=self.origin_w.astype(np.float32, copy=True),
            voxel_m=float(self.voxel_m),
            tsdf=self.tsdf,
            weight=self.weight,
            sem_label=self.sem_label,
            sem_weight=self.sem_weight,
            rgb_color=self.rgb_color,
            rgb_weight=self.rgb_weight,
        )


class RollingTsdfSemanticVolumeTorch:
    """
    Torch (optionally CUDA) implementation of the rolling TSDF(+semantic) volume.

    This mirrors `RollingTsdfSemanticVolume` but stores voxel grids as torch tensors.
    It is intended as an OPTIONAL acceleration path for the planning auxiliary process.
    """

    def __init__(
        self,
        *,
        radius_m: float,
        voxel_m: float,
        trunc_m: Optional[float] = None,
        max_weight: float = 100.0,
        semantic_band_m: Optional[float] = None,
        invalid_label: int = -1,
        device: str = "cuda",
        tsdf_dtype: str = "float32",
    ) -> None:
        import torch

        self.radius_m = float(radius_m)
        self.voxel_m = float(voxel_m)
        self.trunc_m = float(trunc_m) if trunc_m is not None else float(3.0 * self.voxel_m)
        self.max_weight = float(max_weight)
        self.semantic_band_m = float(semantic_band_m) if semantic_band_m is not None else float(self.voxel_m)
        self.invalid_label = int(invalid_label)

        if self.radius_m <= 0 or self.voxel_m <= 0:
            raise ValueError("radius_m and voxel_m must be positive.")
        if self.trunc_m <= 0:
            raise ValueError("trunc_m must be positive.")

        self.device = torch.device(str(device))
        td = str(tsdf_dtype).lower().strip()
        if td == "float16":
            self.tsdf_dtype = torch.float16
        elif td == "float32":
            self.tsdf_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported tsdf_dtype: {tsdf_dtype!r} (expected float32/float16)")

        # Use an odd number of voxels so there is a well-defined center voxel.
        self._n_half = int(np.ceil(self.radius_m / self.voxel_m))
        n = 2 * self._n_half + 1
        self.dims = (int(n), int(n), int(n))
        V = int(n * n * n)

        # World origin of voxel (0,0,0) *center*.
        self.origin_w = torch.zeros((3,), device=self.device, dtype=torch.float32)
        self._center_w: Optional[torch.Tensor] = None

        # Allocate voxel grids.
        self.tsdf = torch.ones(self.dims, device=self.device, dtype=self.tsdf_dtype)
        # Keep weights in float32 for numerical stability even if tsdf is float16.
        self.weight = torch.zeros(self.dims, device=self.device, dtype=torch.float32)
        self.sem_label = torch.full(self.dims, int(self.invalid_label), device=self.device, dtype=torch.int32)
        self.sem_weight = torch.zeros(self.dims, device=self.device, dtype=torch.float32)
        self.rgb_color = torch.zeros(self.dims + (3,), device=self.device, dtype=torch.float32)
        self.rgb_weight = torch.zeros(self.dims, device=self.device, dtype=torch.float32)

        # Precompute voxel center offsets relative to origin_w. Shape: (V,3).
        nx, ny, nz = self.dims
        xs = torch.arange(nx, device=self.device, dtype=torch.float32).view(nx, 1, 1) * float(self.voxel_m)
        ys = torch.arange(ny, device=self.device, dtype=torch.float32).view(1, ny, 1) * float(self.voxel_m)
        zs = torch.arange(nz, device=self.device, dtype=torch.float32).view(1, 1, nz) * float(self.voxel_m)
        self._offsets_flat = torch.stack(
            [
                xs.expand(nx, ny, nz),
                ys.expand(nx, ny, nz),
                zs.expand(nx, ny, nz),
            ],
            dim=-1,
        ).reshape(V, 3)

    def reset(self, *, center_w: "np.ndarray | object") -> None:
        import torch

        c = torch.as_tensor(center_w, dtype=torch.float32, device=self.device).reshape(3)
        self._center_w = c.clone()
        self.origin_w = (c - float(self._n_half) * float(self.voxel_m)).to(torch.float32)
        self.tsdf.fill_(1.0)
        self.weight.zero_()
        self.sem_label.fill_(int(self.invalid_label))
        self.sem_weight.zero_()
        self.rgb_color.zero_()
        self.rgb_weight.zero_()

    def _clear_border(self, *, shift_xyz: Tuple[int, int, int]) -> None:
        dx, dy, dz = (int(shift_xyz[0]), int(shift_xyz[1]), int(shift_xyz[2]))
        nx, ny, nz = self.dims

        def _clear_x(ix0: int, ix1: int) -> None:
            self.tsdf[ix0:ix1, :, :].fill_(1.0)
            self.weight[ix0:ix1, :, :].zero_()
            self.sem_label[ix0:ix1, :, :].fill_(int(self.invalid_label))
            self.sem_weight[ix0:ix1, :, :].zero_()
            self.rgb_color[ix0:ix1, :, :, :].zero_()
            self.rgb_weight[ix0:ix1, :, :].zero_()

        def _clear_y(iy0: int, iy1: int) -> None:
            self.tsdf[:, iy0:iy1, :].fill_(1.0)
            self.weight[:, iy0:iy1, :].zero_()
            self.sem_label[:, iy0:iy1, :].fill_(int(self.invalid_label))
            self.sem_weight[:, iy0:iy1, :].zero_()
            self.rgb_color[:, iy0:iy1, :, :].zero_()
            self.rgb_weight[:, iy0:iy1, :].zero_()

        def _clear_z(iz0: int, iz1: int) -> None:
            self.tsdf[:, :, iz0:iz1].fill_(1.0)
            self.weight[:, :, iz0:iz1].zero_()
            self.sem_label[:, :, iz0:iz1].fill_(int(self.invalid_label))
            self.sem_weight[:, :, iz0:iz1].zero_()
            self.rgb_color[:, :, iz0:iz1, :].zero_()
            self.rgb_weight[:, :, iz0:iz1].zero_()

        if dx > 0:
            _clear_x(nx - dx, nx)
        elif dx < 0:
            _clear_x(0, -dx)

        if dy > 0:
            _clear_y(ny - dy, ny)
        elif dy < 0:
            _clear_y(0, -dy)

        if dz > 0:
            _clear_z(nz - dz, nz)
        elif dz < 0:
            _clear_z(0, -dz)

    def shift_to_center(self, *, center_w: "np.ndarray | object") -> None:
        import torch

        c = torch.as_tensor(center_w, dtype=torch.float32, device=self.device).reshape(3)
        if self._center_w is None:
            self.reset(center_w=c)
            return

        delta = c - self._center_w
        shift_vox = torch.round(delta / float(self.voxel_m)).to(torch.int32)
        dx, dy, dz = (int(shift_vox[0].item()), int(shift_vox[1].item()), int(shift_vox[2].item()))
        if dx == 0 and dy == 0 and dz == 0:
            return

        nx, ny, nz = self.dims
        if abs(dx) >= nx or abs(dy) >= ny or abs(dz) >= nz:
            self.reset(center_w=c)
            return

        shift_m = shift_vox.to(torch.float32) * float(self.voxel_m)
        self._center_w = (self._center_w + shift_m).to(torch.float32)
        self.origin_w = (self.origin_w + shift_m).to(torch.float32)

        roll = (-dx, -dy, -dz)
        self.tsdf = torch.roll(self.tsdf, shifts=roll, dims=(0, 1, 2))
        self.weight = torch.roll(self.weight, shifts=roll, dims=(0, 1, 2))
        self.sem_label = torch.roll(self.sem_label, shifts=roll, dims=(0, 1, 2))
        self.sem_weight = torch.roll(self.sem_weight, shifts=roll, dims=(0, 1, 2))
        self.rgb_color = torch.roll(self.rgb_color, shifts=roll, dims=(0, 1, 2))
        self.rgb_weight = torch.roll(self.rgb_weight, shifts=roll, dims=(0, 1, 2))
        self._clear_border(shift_xyz=(dx, dy, dz))

    def integrate(
        self,
        *,
        depth_hw: "np.ndarray | object",
        T_WC_f32: "np.ndarray | object",
        K_f32: "np.ndarray | object",
        valid_hw: Optional["np.ndarray | object"] = None,
        label_hw: Optional["np.ndarray | object"] = None,
        rgb_hw: Optional["np.ndarray | object"] = None,
        obs_weight: float = 1.0,
    ) -> None:
        import torch

        # Convert inputs.
        depth = torch.as_tensor(depth_hw, device=self.device, dtype=torch.float32)
        if depth.ndim != 2:
            raise ValueError(f"depth_hw must be HxW, got {tuple(depth.shape)}")
        H, W = int(depth.shape[0]), int(depth.shape[1])

        if valid_hw is None:
            valid = torch.isfinite(depth) & (depth > 0.0)
        else:
            valid = torch.as_tensor(valid_hw, device=self.device, dtype=torch.bool)
            if tuple(valid.shape) != (H, W):
                raise ValueError(f"valid_hw shape {tuple(valid.shape)} != depth shape {(H, W)}")
            valid = valid & torch.isfinite(depth) & (depth > 0.0)

        M = torch.as_tensor(T_WC_f32, device=self.device, dtype=torch.float32).reshape(4, 4)
        t_wc = M[:3, 3]
        self.shift_to_center(center_w=t_wc)

        A = M[:3, :3]
        A_inv = torch.linalg.inv(A)

        Xw = self._offsets_flat + self.origin_w.view(1, 3)
        Xc = (Xw - t_wc.view(1, 3)) @ A_inv.T
        z = Xc[:, 2]
        in_front = torch.isfinite(z) & (z > 1e-6) & torch.isfinite(Xc).all(dim=1)
        if not bool(in_front.any()):
            return

        vox_lin = torch.nonzero(in_front, as_tuple=False).view(-1).to(torch.int64)
        Xc = Xc[in_front]
        z = z[in_front]

        K = torch.as_tensor(K_f32, device=self.device, dtype=torch.float32).reshape(3, 3)
        fx, fy = float(K[0, 0].item()), float(K[1, 1].item())
        cx, cy = float(K[0, 2].item()), float(K[1, 2].item())
        u = (fx * (Xc[:, 0] / z) + cx).to(torch.int32)
        v = (fy * (Xc[:, 1] / z) + cy).to(torch.int32)
        inside = (u >= 0) & (u < int(W)) & (v >= 0) & (v < int(H))
        if not bool(inside.any()):
            return

        u = u[inside]
        v = v[inside]
        z = z[inside]
        vox_lin = vox_lin[inside]

        d = depth[v.to(torch.int64), u.to(torch.int64)]
        m = valid[v.to(torch.int64), u.to(torch.int64)]
        if not bool(m.any()):
            return
        d = d[m]
        z = z[m]
        vox_lin = vox_lin[m]
        u = u[m]
        v = v[m]

        sdf = d - z
        trunc = float(self.trunc_m)
        band = (sdf >= -trunc) & (sdf <= trunc)
        if not bool(band.any()):
            return

        sdf = sdf[band]
        vox_lin = vox_lin[band]
        u = u[band]
        v = v[band]

        tsdf_obs = (sdf / trunc).to(torch.float32)
        w_obs = float(max(0.0, obs_weight))
        if w_obs <= 0.0:
            return

        tsdf_flat = self.tsdf.reshape(-1)
        weight_flat = self.weight.reshape(-1)

        w_old = weight_flat[vox_lin]
        tsdf_old = tsdf_flat[vox_lin].to(torch.float32)
        w_new = torch.minimum(w_old + w_obs, torch.tensor(float(self.max_weight), device=self.device, dtype=torch.float32))
        tsdf_new = ((w_old * tsdf_old) + (w_obs * tsdf_obs)) / torch.clamp(w_new, min=1e-6)
        weight_flat[vox_lin] = w_new
        tsdf_flat[vox_lin] = torch.clamp(tsdf_new, -1.0, 1.0).to(self.tsdf_dtype)

        # Surface-near fusion for semantic/RGB.
        surf_band = float(max(float(self.semantic_band_m), float(self.voxel_m)))
        if surf_band <= 0.0:
            return
        close = torch.abs(sdf) <= surf_band
        if not bool(close.any()):
            return

        vox_c = vox_lin[close]
        u_c = u[close].to(torch.int64)
        v_c = v[close].to(torch.int64)

        if label_hw is not None:
            lab = torch.as_tensor(label_hw, device=self.device)
            if lab.ndim != 2 or tuple(lab.shape) != (H, W):
                raise ValueError(f"label_hw shape {tuple(lab.shape)} != depth shape {(H, W)}")
            y_obs = lab[v_c, u_c].to(torch.int32)
            vox_s = vox_c
            if int(self.invalid_label) >= 0:
                valid_lab = y_obs != int(self.invalid_label)
                if bool(valid_lab.any()):
                    y_obs = y_obs[valid_lab]
                    vox_s = vox_s[valid_lab]
                else:
                    y_obs = None
            if y_obs is not None:
                sem_label_flat = self.sem_label.reshape(-1)
                sem_weight_flat = self.sem_weight.reshape(-1)
                cur = sem_label_flat[vox_s]
                wv = sem_weight_flat[vox_s]
                same = cur == y_obs
                wv_new = wv + (w_obs * same.to(torch.float32)) - (w_obs * (~same).to(torch.float32))

                flip = wv_new < 0.0
                if bool(flip.any()):
                    cur = cur.clone()
                    wv_new = wv_new.clone()
                    cur[flip] = y_obs[flip]
                    wv_new[flip] = -wv_new[flip]

                sem_label_flat[vox_s] = cur
                sem_weight_flat[vox_s] = wv_new.to(torch.float32)

        if rgb_hw is not None:
            rgb = torch.as_tensor(rgb_hw, device=self.device, dtype=torch.float32)
            if rgb.ndim != 3 or tuple(rgb.shape) != (H, W, 3):
                raise ValueError(f"rgb_hw shape {tuple(rgb.shape)} != expected {(H, W, 3)}")
            rgb_obs = rgb[v_c, u_c]
            maxv = float(rgb_obs.max().item()) if rgb_obs.numel() > 0 else 0.0
            if maxv > 1.0 + 1e-6:
                rgb_obs = torch.clamp(rgb_obs / 255.0, 0.0, 1.0)
            else:
                rgb_obs = torch.clamp(rgb_obs, 0.0, 1.0)
            rgb_color_flat = self.rgb_color.reshape(-1, 3)
            rgb_weight_flat = self.rgb_weight.reshape(-1)
            w_old_rgb = rgb_weight_flat[vox_c]
            rgb_old = rgb_color_flat[vox_c]
            w_new_rgb = torch.minimum(
                w_old_rgb + w_obs,
                torch.tensor(float(self.max_weight), device=self.device, dtype=torch.float32),
            )
            rgb_new = ((w_old_rgb[:, None] * rgb_old) + (w_obs * rgb_obs)) / torch.clamp(w_new_rgb[:, None], min=1e-6)
            rgb_weight_flat[vox_c] = w_new_rgb
            rgb_color_flat[vox_c] = torch.clamp(rgb_new, 0.0, 1.0)

    def snapshot_cpu(self) -> TsdfVolumeSnapshot:
        """
        Return a CPU numpy snapshot for publishing/visualization.
        """
        import torch

        origin_w = self.origin_w.detach().cpu().numpy().astype(np.float32, copy=True)
        tsdf = self.tsdf.detach().to(torch.float32).cpu().numpy()
        weight = self.weight.detach().cpu().numpy()
        sem_label = self.sem_label.detach().cpu().numpy()
        sem_weight = self.sem_weight.detach().cpu().numpy()
        rgb_color = self.rgb_color.detach().cpu().numpy()
        rgb_weight = self.rgb_weight.detach().cpu().numpy()
        return TsdfVolumeSnapshot(
            origin_w=origin_w,
            voxel_m=float(self.voxel_m),
            tsdf=tsdf,
            weight=weight,
            sem_label=sem_label,
            sem_weight=sem_weight,
            rgb_color=rgb_color,
            rgb_weight=rgb_weight,
        )
