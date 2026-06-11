"""
User-friendly consumer wrapper for the planning semantic point cloud shared memory.

Motivation
----------
The producer side (SLAM) publishes a latest-only point cloud + pose snapshot via shared memory.
Internally, the publisher writes a small `shm_info.json` file that contains the shared memory names.

Some users do not want to manually deal with that JSON. This wrapper hides the detail:
  - You provide only the `out_dir` (the same directory you already pass to SLAM).
  - The client attaches lazily and returns numpy arrays you can feed into planning/control.

Important
---------
This is read-only and does not affect SLAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mast3r_slam.planning_pointcloud_buffer import SharedPointCloudReader
from mast3r_slam.planning_pointcloud_buffer import SharedKeyframeMapReader
from mast3r_slam.planning_pointcloud_buffer import SharedTsdfVolumeReader


@dataclass
class PlanningPointCloudSnapshot:
    """
    A convenience snapshot container for downstream planning.

    Shapes:
      - T_WC    : (4,4) float32, camera->world pose matrix
      - points_w: (N,3) float32, world-space point cloud
      - label_id: (N,) int32, semantic class id aligned with points_w
      - rgb_u8  : (N,3) uint8, per-point RGB color aligned with points_w (optional; may be None)
      - traj_w  : (K,3) float32, keyframe trajectory translations in world coordinates (optional; may be None)
    """

    frame_id: int
    timestamp_s: float
    T_WC: object
    points_w: object
    label_id: object
    rgb_u8: object
    traj_w: object


class PlanningPointCloudClient:
    """
    Minimal consumer API for the latest planning semantic point cloud.

    Usage (downstream process):
      client = PlanningPointCloudClient(out_dir="logs/planning_pointcloud_limo2")
      snap = client.read()  # returns numpy arrays
    """

    def __init__(self, *, out_dir: str, info_filename: str = "shm_info.json") -> None:
        self.out_dir = Path(out_dir)
        self.info_path = self.out_dir / str(info_filename)
        self._reader: Optional[SharedPointCloudReader] = None

    def attach(self) -> None:
        """Attach to shared memory using `<out_dir>/shm_info.json`."""
        # IMPORTANT (robustness / race condition):
        #   In many real deployments, the consumer process can start before the producer has
        #   created `<out_dir>/shm_info.json` and/or before the shared memory segments exist.
        #
        #   If we set `self._reader` before `reader.attach()` succeeds, we can end up with a
        #   "half-initialized" reader instance:
        #     - `self._reader` is not None, so `read()` will not call `attach()` again
        #     - but the underlying SharedPointCloudReader has not attached any shm blocks yet
        #       and will throw: "Reader is not attached. Call attach() first."
        #
        #   To avoid that failure mode, we only assign to `self._reader` AFTER a successful attach.
        reader = SharedPointCloudReader(info_path=str(self.info_path))
        reader.attach()
        self._reader = reader

    def read(self, *, copy: bool = True) -> PlanningPointCloudSnapshot:
        """
        Read the latest snapshot.

        Args:
          copy: If True (recommended), returns copies of arrays (safe against concurrent writes).
        """

        if self._reader is None:
            self.attach()
        assert self._reader is not None
        # NOTE:
        #   Even with the atomic attach above, we keep this try/except as a safety net in case
        #   the shared memory was unlinked/restarted while this client stays alive. In that case,
        #   we reset and force a re-attach on the next call.
        try:
            snap = self._reader.read(copy=bool(copy))
        except RuntimeError as e:
            if "not attached" in str(e).lower():
                self.close()
                self.attach()
                assert self._reader is not None
                snap = self._reader.read(copy=bool(copy))
            else:
                raise
        return PlanningPointCloudSnapshot(
            frame_id=int(snap.frame_id),
            timestamp_s=float(snap.timestamp_s),
            T_WC=snap.T_WC,
            points_w=snap.points_w,
            label_id=snap.label_id,
            rgb_u8=getattr(snap, "rgb_u8", None),
            traj_w=getattr(snap, "traj_w", None),
        )

    def close(self) -> None:
        """Detach from shared memory segments."""

        if self._reader is not None:
            self._reader.close()
        self._reader = None


@dataclass
class PlanningKeyframeMapSnapshot:
    """
    A convenience snapshot container for downstream planning/debug that mirrors the
    official visualization's data model (keyframe-local points + dynamic poses).

    Shapes (K = kf_count, P = points_per_kf)
    ---------------------------------------
    - curr_T_WC      : (4,4) float32, current camera->world pose
    - kf_T_WC        : (K,4,4) float32, keyframe camera->world poses
    - kf_frame_id    : (K,) int32, keyframe frame ids
    - kf_n_points    : (K,) int32, valid point count per keyframe (<= P)
    - kf_points_k    : (K,P,3) float32, keyframe-local points in keyframe camera frame
    - kf_label_id    : (K,P) int32, per-point semantic ids
    - kf_rgb_u8      : (K,P,3) uint8, per-point appearance RGB (optional for planning)

    IMPORTANT
    ---------
    - This snapshot can reference very large shared-memory arrays. By default, the client
      returns views (copy=False) so that reading does not allocate hundreds of MB per call.
    - Consumers that need strict snapshot consistency can request copying (copy=True),
      optionally including `copy_points=True` (may be expensive).
    """

    frame_id: int
    timestamp_s: float
    curr_T_WC: object
    kf_T_WC: object
    kf_frame_id: object
    kf_n_points: object
    kf_points_k: object
    kf_label_id: object
    kf_rgb_u8: object


class PlanningKeyframeMapClient:
    """
    Consumer API for the keyframe-centric shared-memory map.

    This is intended to mimic how the official visualization works:
      - Points are stored in keyframe camera coordinates.
      - Keyframe poses can change at any time due to loop closure / backend optimization.
      - Consumers render/reproject using the latest poses, without needing the publisher to
        recompute world-space points every frame.
    """

    def __init__(self, *, out_dir: str, info_filename: str = "shm_info.json") -> None:
        self.out_dir = Path(out_dir)
        self.info_path = self.out_dir / str(info_filename)
        self._reader: Optional[SharedKeyframeMapReader] = None

    def attach(self) -> None:
        """
        Attach to shared memory using `<out_dir>/shm_info.json`.

        IMPORTANT (race condition):
          The consumer process can start before the producer creates the info file or shm blocks.
          We only store the reader after a successful attach to avoid "half-attached" instances.
        """

        reader = SharedKeyframeMapReader(info_path=str(self.info_path))
        reader.attach()
        self._reader = reader

    def read(
        self,
        *,
        copy: bool = False,
        copy_points: bool = False,
    ) -> PlanningKeyframeMapSnapshot:
        """
        Read the latest keyframe-map snapshot.

        Args:
          copy:
            If True, copies small pose/id arrays to avoid concurrent-write tearing.
          copy_points:
            If True (only meaningful when copy=True), also copies the full per-keyframe point tensors.
            WARNING: this can be very expensive (hundreds of MB).
        """

        if self._reader is None:
            self.attach()
        assert self._reader is not None

        try:
            snap = self._reader.read(copy=bool(copy), copy_points=bool(copy_points))
        except RuntimeError as e:
            # Safety net: re-attach if the producer restarted/unlinked shm while we stayed alive.
            if "not attached" in str(e).lower():
                self.close()
                self.attach()
                assert self._reader is not None
                snap = self._reader.read(copy=bool(copy), copy_points=bool(copy_points))
            else:
                raise

        return PlanningKeyframeMapSnapshot(
            frame_id=int(snap.frame_id),
            timestamp_s=float(snap.timestamp_s),
            curr_T_WC=snap.curr_T_WC,
            kf_T_WC=snap.kf_T_WC,
            kf_frame_id=snap.kf_frame_id,
            kf_n_points=snap.kf_n_points,
            kf_points_k=snap.kf_points_k,
            kf_label_id=snap.kf_label_id,
            kf_rgb_u8=snap.kf_rgb_u8,
        )

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
        self._reader = None


@dataclass
class PlanningTsdfVolumeSnapshot:
    """
    Convenience snapshot container for a rolling TSDF(+semantic) volume.

    Shapes (Nx,Ny,Nz are determined by (radius_m, voxel_m))
    ------------------------------------------------------
    - origin_w   : (3,) float32, world position of voxel (0,0,0) center
    - voxel_m    : float, voxel size in meters
    - dims       : (3,) int32, [nx, ny, nz]
    - curr_T_WC  : (4,4) float32, current camera->world pose
    - tsdf       : (nx,ny,nz) float32 in [-1,1]
    - weight     : (nx,ny,nz) float32
    - sem_label  : (nx,ny,nz) int32
    - sem_weight : (nx,ny,nz) float32
    - frame_sem_id: (H,W) int32 (optional) per-frame semantic label ids used for the latest integration step
    - frame_rgb_u8: (H,W,3) uint8 (optional) per-frame RGB image aligned with the latest integration step

    Why `frame_sem_id` exists
    -------------------------
    Downstream visualization/debugging often needs to show the *exact* semantic observation
    that was fused into the TSDF on this step.

    We intentionally publish `frame_sem_id` as a 2D int-label image (not RGB) because:
      - it is compact (H*W int32),
      - it avoids ambiguity about palettes / dataset color schemes,
      - viewers can colorize it with the correct dataset palette (e.g., ADE20K).

    IMPORTANT:
      - `frame_sem_id` is NOT required for fusion.
      - It is published only when the producer allocates the TSDF shared-memory schema v2.
        Consumers must treat it as optional.
    """

    frame_id: int
    timestamp_s: float
    origin_w: object
    voxel_m: float
    dims: object
    curr_T_WC: object
    tsdf: object
    weight: object
    sem_label: object
    sem_weight: object
    frame_sem_id: object
    frame_rgb_u8: object


class PlanningTsdfVolumeClient:
    """
    Consumer API for the rolling TSDF volume published by SLAM.

    The producer writes a separate info file (default: `tsdf_shm_info.json`) under the same out_dir.
    """

    def __init__(self, *, out_dir: str, info_filename: str = "tsdf_shm_info.json") -> None:
        self.out_dir = Path(out_dir)
        self.info_path = self.out_dir / str(info_filename)
        self._reader: Optional[SharedTsdfVolumeReader] = None

    def attach(self) -> None:
        reader = SharedTsdfVolumeReader(info_path=str(self.info_path))
        reader.attach()
        self._reader = reader

    def read(self, *, copy: bool = True) -> PlanningTsdfVolumeSnapshot:
        if self._reader is None:
            self.attach()
        assert self._reader is not None
        try:
            snap = self._reader.read(copy=bool(copy))
        except RuntimeError as e:
            if "not attached" in str(e).lower():
                self.close()
                self.attach()
                assert self._reader is not None
                snap = self._reader.read(copy=bool(copy))
            else:
                raise

        return PlanningTsdfVolumeSnapshot(
            frame_id=int(snap.frame_id),
            timestamp_s=float(snap.timestamp_s),
            origin_w=snap.origin_w,
            voxel_m=float(snap.voxel_m),
            dims=snap.dims,
            curr_T_WC=snap.curr_T_WC,
            tsdf=snap.tsdf,
            weight=snap.weight,
            sem_label=snap.sem_label,
            sem_weight=snap.sem_weight,
            frame_sem_id=getattr(snap, "frame_sem_id", None),
            frame_rgb_u8=getattr(snap, "frame_rgb_u8", None),
        )

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
        self._reader = None
