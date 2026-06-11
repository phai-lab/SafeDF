"""
Latest-only shared-memory publisher/reader for a planning-oriented semantic point cloud.

Why this exists
---------------
The user wants:
  - A semantic point cloud derived from SLAM keyframes (geometry + semantic colors/labels).
  - The current camera pose (T_WC) at the same time.
  - NO GUI requirement (visualization can be disabled).
  - A way for a downstream planning process to consume this output without blocking SLAM.

If the producer (SLAM) and consumer (planning) are in different OS processes, Python variables
are not shared. Therefore, we need a small IPC mechanism.

This module implements a minimal, "latest-only" shared-memory contract:
  - The producer overwrites a fixed-size buffer each update (no queues, no backpressure).
  - The consumer can read the newest snapshot at any time.
  - The memory layout is described by a small JSON info file (names, shapes, dtypes).

Important design choices
------------------------
1) Latest-only:
   We intentionally DO NOT accumulate frames. Planning/control usually wants the newest map/pose.
2) Fixed maximum points:
   Shared memory needs a fixed size. The producer writes up to `max_points`, plus a `count`.
3) Weak atomicity:
   We write large arrays first, then write a small meta block last. Consumers can treat the
   meta as a "frame boundary" marker and can retry if they detect a torn read.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class _ShmBlock:
    """A single shared memory block + a typed numpy view."""

    shm: shared_memory.SharedMemory
    arr: np.ndarray


class LatestSharedPointCloudBuffer:
    """
    Producer-side latest-only point cloud buffer implemented with shared memory.

    What is published
    -----------------
    - meta_f64[2]        : [frame_id_as_float, timestamp_s]
    - count_i32[1]       : number of valid points written this frame (<= max_points)
    - traj_count_i32[1]  : number of valid keyframe trajectory points written this frame (<= max_keyframes)
    - T_WC_f32[4,4]      : current camera pose as a 4x4 matrix (camera -> world)
    - points_w_f32[N,3]  : world-space point cloud (N=max_points)
    - label_id_i32[N]    : per-point semantic class id (int32)
    - rgb_u8[N,3]        : per-point RGB color (uint8, 0..255) for visualization/debug only
    - traj_w_f32[M,3]    : keyframe trajectory translations in world coordinates (M=max_keyframes)

    Notes:
      - For planning/control, semantic should be a class ID, not an RGB color.
      - `rgb_u8` exists so a consumer can render an "RGB-like" view by reprojecting the point cloud
        into the current camera pose. This avoids passing full images across processes.
      - If you want a semantic palette for visualization, compute it from label_id on the consumer side.
      - `traj_w_f32` exists to align the downstream trajectory visualization with the official SLAM
        visualization: when loop closure / backend optimization updates keyframe poses, the published
        keyframe translations also change. A consumer should redraw the trajectory from this array
        (instead of appending per-frame poses forever).
    """

    def __init__(self, *, prefix: str, info_path: str) -> None:
        self.prefix = str(prefix)
        self.info_path = Path(info_path)

        self.max_points: Optional[int] = None
        self.max_keyframes: Optional[int] = None
        self._meta: Optional[_ShmBlock] = None
        self._count: Optional[_ShmBlock] = None
        self._traj_count: Optional[_ShmBlock] = None
        self._T_WC: Optional[_ShmBlock] = None
        self._points: Optional[_ShmBlock] = None
        self._label_id: Optional[_ShmBlock] = None
        self._rgb_u8: Optional[_ShmBlock] = None
        self._traj_w: Optional[_ShmBlock] = None

        self._names: Dict[str, str] = {}

    def _create_block(self, *, key: str, shape: Tuple[int, ...], dtype: np.dtype) -> _ShmBlock:
        """
        Allocate a shared memory block and return a numpy view.

        IMPORTANT:
          We include PID + current time in the name to reduce the chance of collisions with
          stale segments from previous runs.
        """

        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * int(dtype.itemsize)
        name = f"{self.prefix}_{key}_{os.getpid()}_{int(time.time()*1e6)}"
        shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
        arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
        self._names[key] = name
        return _ShmBlock(shm=shm, arr=arr)

    def ensure(self, *, max_points: int, max_keyframes: int = 2000) -> None:
        """
        Ensure the shared memory blocks exist for a given `max_points` and `max_keyframes`.

        If max_points/max_keyframes changes, we destroy and recreate blocks.
        """

        n = int(max(1, max_points))
        m = int(max(1, max_keyframes))
        if self.max_points == n and self.max_keyframes == m and self._meta is not None:
            return

        self.close(unlink=True)

        self.max_points = n
        self.max_keyframes = m
        self._names = {}

        self._meta = self._create_block(key="meta_f64", shape=(2,), dtype=np.float64)
        self._count = self._create_block(key="count_i32", shape=(1,), dtype=np.int32)
        self._traj_count = self._create_block(key="traj_count_i32", shape=(1,), dtype=np.int32)
        self._T_WC = self._create_block(key="T_WC_f32", shape=(4, 4), dtype=np.float32)
        self._points = self._create_block(key="points_w_f32", shape=(n, 3), dtype=np.float32)
        self._label_id = self._create_block(key="label_id_i32", shape=(n,), dtype=np.int32)
        self._rgb_u8 = self._create_block(key="rgb_u8", shape=(n, 3), dtype=np.uint8)
        self._traj_w = self._create_block(key="traj_w_f32", shape=(m, 3), dtype=np.float32)

        # Write an info json to disk so other processes can attach.
        self.info_path.parent.mkdir(parents=True, exist_ok=True)
        info = dict(
            max_points=int(n),
            max_keyframes=int(m),
            names=self._names,
            dtypes=dict(
                meta_f64="float64[2]",
                count_i32="int32[1]",
                traj_count_i32="int32[1]",
                T_WC_f32="float32[4,4]",
                points_w_f32="float32[max_points,3]",
                label_id_i32="int32[max_points]",
                rgb_u8="uint8[max_points,3]",
                traj_w_f32="float32[max_keyframes,3]",
            ),
        )
        self.info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    def write(
        self,
        *,
        frame_id: int,
        timestamp_s: float,
        T_WC_f32: np.ndarray,
        points_w_f32: np.ndarray,
        label_id_i32: np.ndarray,
        rgb_u8: Optional[np.ndarray] = None,
        traj_w_f32: Optional[np.ndarray] = None,
    ) -> None:
        """
        Publish the latest point cloud snapshot into shared memory.

        Inputs:
          frame_id: integer frame id
          timestamp_s: float seconds
          T_WC_f32: (4,4) float32 pose matrix (camera->world)
          points_w_f32: (N,3) float32 points in world coordinates
          label_id_i32: (N,) int32 semantic class ids aligned with points
        """

        if self.max_points is None or self._meta is None:
            raise RuntimeError("PointCloudBuffer is not initialized; call ensure(max_points=...) first.")

        assert self._count is not None
        assert self._traj_count is not None
        assert self._T_WC is not None
        assert self._points is not None
        assert self._label_id is not None
        assert self._rgb_u8 is not None
        assert self._traj_w is not None

        nmax = int(self.max_points)
        kmax = int(self.max_keyframes or 1)
        pts = np.asarray(points_w_f32, dtype=np.float32).reshape(-1, 3)
        lab = np.asarray(label_id_i32, dtype=np.int32).reshape(-1)
        rgb = None if rgb_u8 is None else np.asarray(rgb_u8).reshape(-1, 3)
        n = int(min(nmax, pts.shape[0], lab.shape[0]))
        if rgb is not None:
            n = int(min(n, rgb.shape[0]))

        # Keyframe trajectory (optional).
        # Shape: (K,3) float32 translations in world coordinates.
        traj = None if traj_w_f32 is None else np.asarray(traj_w_f32, dtype=np.float32).reshape(-1, 3)
        k = 0 if traj is None else int(min(kmax, traj.shape[0]))

        # Write bulk arrays first (latest-only overwrite).
        if n > 0:
            self._points.arr[:n, :] = pts[:n, :]
            self._label_id.arr[:n] = lab[:n]
            if rgb is not None:
                self._rgb_u8.arr[:n, :] = np.asarray(rgb[:n, :], dtype=np.uint8)
            else:
                self._rgb_u8.arr[:n, :].fill(0)
        if n < nmax:
            # Optional: clear the tail to make debugging easier for consumers.
            self._points.arr[n:, :].fill(np.nan)
            self._label_id.arr[n:].fill(np.int32(-1))
            self._rgb_u8.arr[n:, :].fill(0)

        if k > 0:
            self._traj_w.arr[:k, :] = traj[:k, :]
        if k < kmax:
            self._traj_w.arr[k:, :].fill(np.nan)
        self._traj_count.arr[0] = np.int32(k)

        self._count.arr[0] = np.int32(n)
        self._T_WC.arr[...] = np.asarray(T_WC_f32, dtype=np.float32).reshape(4, 4)

        # Write meta LAST as a soft "frame boundary" marker.
        self._meta.arr[0] = float(int(frame_id))
        self._meta.arr[1] = float(timestamp_s)

    def close(self, *, unlink: bool) -> None:
        """
        Close (and optionally unlink) all shared memory segments.

        - close() detaches from the segment in this process.
        - unlink() removes the shared memory name (new attaches fail, existing attaches keep working
          until they also close).
        """

        blocks = [self._meta, self._count, self._traj_count, self._T_WC, self._points, self._label_id, self._rgb_u8, self._traj_w]
        for b in blocks:
            if b is None:
                continue
            try:
                b.shm.close()
            except Exception:
                pass
            if unlink:
                try:
                    b.shm.unlink()
                except Exception:
                    pass

        self._meta = None
        self._count = None
        self._traj_count = None
        self._T_WC = None
        self._points = None
        self._label_id = None
        self._rgb_u8 = None
        self._traj_w = None
        self.max_points = None
        self.max_keyframes = None
        self._names = {}


class SharedPointCloudReader:
    """
    Consumer-side helper to attach to the point cloud shared memory and read snapshots.

    Usage:
      reader = SharedPointCloudReader(info_path=".../shm_info.json")
      reader.attach()
      snap = reader.read(copy=True)
      # snap.points_w, snap.label_id, snap.T_WC are numpy arrays on CPU.
    """

    def __init__(self, *, info_path: str) -> None:
        self.info_path = Path(info_path)
        self._info: Optional[dict] = None
        self._blocks: Dict[str, _ShmBlock] = {}

    def attach(self) -> None:
        info = json.loads(self.info_path.read_text(encoding="utf-8"))
        self._info = info
        n = int(info["max_points"])
        m = int(info.get("max_keyframes", 0) or 0)
        names = dict(info["names"])

        def _attach_block(key: str, shape: Tuple[int, ...], dtype: np.dtype) -> _ShmBlock:
            shm = shared_memory.SharedMemory(name=names[key], create=False)
            arr = np.ndarray(shape=shape, dtype=np.dtype(dtype), buffer=shm.buf)
            return _ShmBlock(shm=shm, arr=arr)

        self._blocks["meta_f64"] = _attach_block("meta_f64", (2,), np.float64)
        self._blocks["count_i32"] = _attach_block("count_i32", (1,), np.int32)
        # Backward-compatibility: older publishers might not publish a trajectory.
        if "traj_count_i32" in names and m > 0:
            self._blocks["traj_count_i32"] = _attach_block("traj_count_i32", (1,), np.int32)
        self._blocks["T_WC_f32"] = _attach_block("T_WC_f32", (4, 4), np.float32)
        self._blocks["points_w_f32"] = _attach_block("points_w_f32", (n, 3), np.float32)
        self._blocks["label_id_i32"] = _attach_block("label_id_i32", (n,), np.int32)
        # Backward-compatibility: older publishers might not publish rgb_u8.
        if "rgb_u8" in names:
            self._blocks["rgb_u8"] = _attach_block("rgb_u8", (n, 3), np.uint8)
        if "traj_w_f32" in names and m > 0:
            self._blocks["traj_w_f32"] = _attach_block("traj_w_f32", (m, 3), np.float32)

    @dataclass
    class Snapshot:
        frame_id: int
        timestamp_s: float
        T_WC: np.ndarray  # (4,4) float32
        points_w: np.ndarray  # (N,3) float32
        label_id: np.ndarray  # (N,) int32
        rgb_u8: Optional[np.ndarray]  # (N,3) uint8 or None
        traj_w: Optional[np.ndarray]  # (K,3) float32 or None (keyframe trajectory translations)

    def read(self, *, copy: bool = True, max_retries: int = 3) -> "SharedPointCloudReader.Snapshot":
        """
        Read a consistent-ish snapshot.

        We read meta before and after the bulk copy. If meta changed, we retry a few times.
        This reduces the likelihood of returning a torn read.
        """

        if not self._blocks:
            raise RuntimeError("Reader is not attached. Call attach() first.")

        meta = self._blocks["meta_f64"].arr
        count_arr = self._blocks["count_i32"].arr
        T = self._blocks["T_WC_f32"].arr
        P = self._blocks["points_w_f32"].arr
        L = self._blocks["label_id_i32"].arr
        rgb_block = self._blocks.get("rgb_u8", None)
        rgb_arr = None if rgb_block is None else rgb_block.arr
        traj_count_block = self._blocks.get("traj_count_i32", None)
        traj_w_block = self._blocks.get("traj_w_f32", None)
        traj_count_arr = None if traj_count_block is None else traj_count_block.arr
        traj_w_arr = None if traj_w_block is None else traj_w_block.arr

        for _ in range(int(max(1, max_retries))):
            fid0 = int(meta[0])
            ts0 = float(meta[1])
            n = int(count_arr[0])
            n = max(0, min(n, P.shape[0]))
            k = 0
            if traj_count_arr is not None and traj_w_arr is not None:
                k = int(traj_count_arr[0])
                k = max(0, min(k, traj_w_arr.shape[0]))

            if copy:
                T_out = np.array(T, copy=True)
                P_out = np.array(P[:n, :], copy=True)
                L_out = np.array(L[:n], copy=True)
                rgb_out = None if rgb_arr is None else np.array(rgb_arr[:n, :], copy=True)
                traj_out = None if traj_w_arr is None else np.array(traj_w_arr[:k, :], copy=True)
            else:
                # WARNING:
                #   Returning views means the producer can overwrite data while the consumer reads it.
                #   Use copy=True unless you know what you're doing.
                T_out = T
                P_out = P[:n, :]
                L_out = L[:n]
                rgb_out = None if rgb_arr is None else rgb_arr[:n, :]
                traj_out = None if traj_w_arr is None else traj_w_arr[:k, :]

            fid1 = int(meta[0])
            ts1 = float(meta[1])
            if fid0 == fid1 and ts0 == ts1:
                return SharedPointCloudReader.Snapshot(
                    frame_id=fid1,
                    timestamp_s=ts1,
                    T_WC=T_out,
                    points_w=P_out,
                    label_id=L_out,
                    rgb_u8=rgb_out,
                    traj_w=traj_out,
                )

        # Fall back: return the last read even if meta changed.
        n_last = int(count_arr[0])
        n_last = max(0, min(n_last, P.shape[0]))
        k_last = 0
        if traj_count_arr is not None and traj_w_arr is not None:
            k_last = int(traj_count_arr[0])
            k_last = max(0, min(k_last, traj_w_arr.shape[0]))
        return SharedPointCloudReader.Snapshot(
            frame_id=int(meta[0]),
            timestamp_s=float(meta[1]),
            T_WC=np.array(T, copy=True) if copy else T,
            points_w=np.array(P[:n_last, :], copy=True) if copy else P[:n_last, :],
            label_id=np.array(L[:n_last], copy=True) if copy else L[:n_last],
            rgb_u8=None
            if rgb_arr is None
            else (np.array(rgb_arr[:n_last, :], copy=True) if copy else rgb_arr[:n_last, :]),
            traj_w=None
            if traj_w_arr is None
            else (np.array(traj_w_arr[:k_last, :], copy=True) if copy else traj_w_arr[:k_last, :]),
        )

    def close(self) -> None:
        for b in self._blocks.values():
            try:
                b.shm.close()
            except Exception:
                pass
        self._blocks = {}
        self._info = None


# =============================================================================
# V2: Keyframe-centric shared memory (official-visualization-like)
# =============================================================================
#
# Motivation (user request)
# ------------------------
# The original `LatestSharedPointCloudBuffer` publishes a *flattened* world-space point cloud.
# That representation is easy to consume, but it has a fundamental drawback for SLAM systems:
#
#   - Loop closure / backend optimization can update keyframe poses at any time.
#   - If points were already transformed into world coordinates, then "the map" becomes stale
#     unless the publisher recomputes and republishes *all* world-space points every update.
#
# The official visualization avoids this by storing each keyframe's canonical pointmap in that
# keyframe's camera frame, and *only* applying the latest keyframe pose during rendering.
#
# This section implements the same idea for planning/debug IPC:
#   - Publish per-keyframe points in keyframe camera coordinates.
#   - Publish the latest keyframe poses (T_WC) every tick (small).
#   - Consumers render/reproject by combining (T_CW_current * T_WC_keyframe) at read time.
#
# IMPORTANT constraints
# ---------------------
# - Shared memory blocks must have a fixed maximum size.
# - We keep "latest-only" semantics: we overwrite in place, no queues.
# - Copying the full per-keyframe point tensor every read is expensive (hundreds of MB),
#   so the reader defaults to returning *views* (copy=False). This is acceptable for debug
#   visualization; consumers that require strict consistency can copy explicitly.


class LatestSharedKeyframeMapBuffer:
    """
    Producer-side latest-only keyframe map buffer implemented with shared memory.

    What is published
    -----------------
    - meta_f64[2]              : [frame_id_as_float, timestamp_s] written LAST as a soft boundary marker
    - kf_count_i32[1]          : number of valid keyframes published (<= max_keyframes)
    - curr_T_WC_f32[4,4]       : current camera pose (camera -> world) as a 4x4 float32 matrix
    - kf_T_WC_f32[K,4,4]       : keyframe poses (camera -> world) as float32 matrices
    - kf_frame_id_i32[K]       : keyframe frame ids (stable identifiers for debugging)
    - kf_n_points_i32[K]       : valid point count per keyframe (<= points_per_kf)
    - kf_points_k_f32[K,P,3]   : keyframe-local 3D points (camera frame), float32
    - kf_label_id_i32[K,P]     : per-point semantic integer id (int32, NOT RGB)
    - kf_rgb_u8[K,P,3]         : per-point appearance RGB (uint8), for "RGB-like" reprojection

    Notes:
      - This buffer stores points in keyframe camera coordinates. Consumers must apply the latest
        keyframe pose to get world points, or compose with current pose to get current-camera points.
      - `kf_rgb_u8` is optional for planning, but extremely useful for debugging (reproject to "RGB").
      - Semantic is stored as integer IDs (e.g., EfficientViT hard labels packed or direct ids).
    """

    def __init__(self, *, prefix: str, info_path: str) -> None:
        self.prefix = str(prefix)
        self.info_path = Path(info_path)

        self.max_keyframes: Optional[int] = None
        self.points_per_kf: Optional[int] = None

        self._meta: Optional[_ShmBlock] = None
        self._kf_count: Optional[_ShmBlock] = None
        self._curr_T_WC: Optional[_ShmBlock] = None
        self._kf_T_WC: Optional[_ShmBlock] = None
        self._kf_frame_id: Optional[_ShmBlock] = None
        self._kf_n_points: Optional[_ShmBlock] = None
        self._kf_points_k: Optional[_ShmBlock] = None
        self._kf_label_id: Optional[_ShmBlock] = None
        self._kf_rgb_u8: Optional[_ShmBlock] = None

        self._names: Dict[str, str] = {}

    def _create_block(self, *, key: str, shape: Tuple[int, ...], dtype: np.dtype) -> _ShmBlock:
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * int(dtype.itemsize)
        name = f"{self.prefix}_{key}_{os.getpid()}_{int(time.time()*1e6)}"
        shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
        arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
        self._names[key] = name
        return _ShmBlock(shm=shm, arr=arr)

    def ensure(self, *, max_keyframes: int, points_per_kf: int) -> None:
        """
        Ensure the shared memory blocks exist for a given `max_keyframes` and `points_per_kf`.

        If either changes, we destroy and recreate blocks.
        """

        kmax = int(max(1, max_keyframes))
        pmax = int(max(1, points_per_kf))
        if self.max_keyframes == kmax and self.points_per_kf == pmax and self._meta is not None:
            return

        self.close(unlink=True)
        self.max_keyframes = kmax
        self.points_per_kf = pmax
        self._names = {}

        self._meta = self._create_block(key="meta_f64", shape=(2,), dtype=np.float64)
        self._kf_count = self._create_block(key="kf_count_i32", shape=(1,), dtype=np.int32)
        self._curr_T_WC = self._create_block(key="curr_T_WC_f32", shape=(4, 4), dtype=np.float32)
        self._kf_T_WC = self._create_block(key="kf_T_WC_f32", shape=(kmax, 4, 4), dtype=np.float32)
        self._kf_frame_id = self._create_block(key="kf_frame_id_i32", shape=(kmax,), dtype=np.int32)
        self._kf_n_points = self._create_block(key="kf_n_points_i32", shape=(kmax,), dtype=np.int32)
        self._kf_points_k = self._create_block(key="kf_points_k_f32", shape=(kmax, pmax, 3), dtype=np.float32)
        self._kf_label_id = self._create_block(key="kf_label_id_i32", shape=(kmax, pmax), dtype=np.int32)
        self._kf_rgb_u8 = self._create_block(key="kf_rgb_u8", shape=(kmax, pmax, 3), dtype=np.uint8)

        # Initialize with safe defaults to make consumer-side debugging easier.
        self._kf_count.arr[0] = np.int32(0)
        self._curr_T_WC.arr[...] = np.eye(4, dtype=np.float32)
        self._kf_T_WC.arr[...] = np.eye(4, dtype=np.float32)[None, :, :]
        self._kf_frame_id.arr.fill(np.int32(-1))
        self._kf_n_points.arr.fill(np.int32(0))
        self._kf_points_k.arr.fill(np.nan)
        self._kf_label_id.arr.fill(np.int32(-1))
        self._kf_rgb_u8.arr.fill(np.uint8(0))
        self._meta.arr[0] = 0.0
        self._meta.arr[1] = 0.0

        # Write schema description for consumers.
        self.info_path.parent.mkdir(parents=True, exist_ok=True)
        info = dict(
            schema="keyframe_map_v1",
            max_keyframes=int(kmax),
            points_per_kf=int(pmax),
            names=self._names,
            dtypes=dict(
                meta_f64="float64[2]",
                kf_count_i32="int32[1]",
                curr_T_WC_f32="float32[4,4]",
                kf_T_WC_f32="float32[max_keyframes,4,4]",
                kf_frame_id_i32="int32[max_keyframes]",
                kf_n_points_i32="int32[max_keyframes]",
                kf_points_k_f32="float32[max_keyframes,points_per_kf,3]",
                kf_label_id_i32="int32[max_keyframes,points_per_kf]",
                kf_rgb_u8="uint8[max_keyframes,points_per_kf,3]",
            ),
        )
        self.info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    def update_keyframe_points(
        self,
        *,
        kf_slot: int,
        points_k_f32: np.ndarray,
        label_id_i32: np.ndarray,
        rgb_u8: Optional[np.ndarray],
        n_points: int,
    ) -> None:
        """
        Update the stored points for a single keyframe slot.

        IMPORTANT:
          - This writes ONLY one keyframe's point block.
          - It does NOT write meta. Meta should be written once per publisher tick.

        Inputs:
          kf_slot: integer in [0, max_keyframes)
          points_k_f32: (N,3) float32 points in keyframe camera coordinates
          label_id_i32: (N,) int32 semantic ids aligned with points_k_f32
          rgb_u8: (N,3) uint8 per-point appearance RGB aligned with points_k_f32 (optional)
          n_points: number of valid points to publish (<= points_per_kf)
        """

        if self.max_keyframes is None or self.points_per_kf is None or self._kf_points_k is None:
            raise RuntimeError("KeyframeMapBuffer is not initialized; call ensure(max_keyframes=..., points_per_kf=...) first.")

        assert self._kf_n_points is not None
        assert self._kf_label_id is not None
        assert self._kf_rgb_u8 is not None

        k = int(kf_slot)
        if k < 0 or k >= int(self.max_keyframes):
            return

        pmax = int(self.points_per_kf)
        pts = np.asarray(points_k_f32, dtype=np.float32).reshape(-1, 3)
        lab = np.asarray(label_id_i32, dtype=np.int32).reshape(-1)
        n = int(max(0, min(int(n_points), pmax, int(pts.shape[0]), int(lab.shape[0]))))

        if n > 0:
            self._kf_points_k.arr[k, :n, :] = pts[:n, :]
            self._kf_label_id.arr[k, :n] = lab[:n]
            if rgb_u8 is not None:
                rgb = np.asarray(rgb_u8, dtype=np.uint8).reshape(-1, 3)
                if rgb.shape[0] >= n:
                    self._kf_rgb_u8.arr[k, :n, :] = rgb[:n, :]
                else:
                    self._kf_rgb_u8.arr[k, :n, :].fill(np.uint8(0))
            else:
                self._kf_rgb_u8.arr[k, :n, :].fill(np.uint8(0))

        # Clear tail for debug clarity.
        if n < pmax:
            self._kf_points_k.arr[k, n:, :].fill(np.nan)
            self._kf_label_id.arr[k, n:].fill(np.int32(-1))
            self._kf_rgb_u8.arr[k, n:, :].fill(np.uint8(0))

        self._kf_n_points.arr[k] = np.int32(n)

    def write(
        self,
        *,
        frame_id: int,
        timestamp_s: float,
        curr_T_WC_f32: np.ndarray,
        kf_T_WC_f32: np.ndarray,
        kf_frame_id_i32: np.ndarray,
        kf_count: int,
    ) -> None:
        """
        Publish the latest *poses* into shared memory.

        IMPORTANT:
          This method updates:
            - current pose
            - keyframe poses
            - keyframe ids
            - keyframe count
            - meta (last)

          It does NOT update keyframe points. Use `update_keyframe_points()` for that.
        """

        if self.max_keyframes is None or self._meta is None:
            raise RuntimeError("KeyframeMapBuffer is not initialized; call ensure(...) first.")
        assert self._kf_count is not None
        assert self._curr_T_WC is not None
        assert self._kf_T_WC is not None
        assert self._kf_frame_id is not None

        kmax = int(self.max_keyframes)
        k = int(max(0, min(int(kf_count), kmax)))

        self._curr_T_WC.arr[...] = np.asarray(curr_T_WC_f32, dtype=np.float32).reshape(4, 4)

        T = np.asarray(kf_T_WC_f32, dtype=np.float32).reshape(-1, 4, 4)
        fid = np.asarray(kf_frame_id_i32, dtype=np.int32).reshape(-1)
        kk = int(min(k, int(T.shape[0]), int(fid.shape[0])))

        if kk > 0:
            self._kf_T_WC.arr[:kk, :, :] = T[:kk, :, :]
            self._kf_frame_id.arr[:kk] = fid[:kk]

        if kk < kmax:
            # Fill the tail with identity and invalid ids.
            self._kf_T_WC.arr[kk:, :, :] = np.eye(4, dtype=np.float32)[None, :, :]
            self._kf_frame_id.arr[kk:].fill(np.int32(-1))

        self._kf_count.arr[0] = np.int32(kk)

        # Meta last: soft frame-boundary marker.
        self._meta.arr[0] = float(int(frame_id))
        self._meta.arr[1] = float(timestamp_s)

    def close(self, *, unlink: bool) -> None:
        blocks = [
            self._meta,
            self._kf_count,
            self._curr_T_WC,
            self._kf_T_WC,
            self._kf_frame_id,
            self._kf_n_points,
            self._kf_points_k,
            self._kf_label_id,
            self._kf_rgb_u8,
        ]
        for b in blocks:
            if b is None:
                continue
            try:
                b.shm.close()
            except Exception:
                pass
            if unlink:
                try:
                    b.shm.unlink()
                except Exception:
                    pass
        self._meta = None
        self._kf_count = None
        self._curr_T_WC = None
        self._kf_T_WC = None
        self._kf_frame_id = None
        self._kf_n_points = None
        self._kf_points_k = None
        self._kf_label_id = None
        self._kf_rgb_u8 = None
        self.max_keyframes = None
        self.points_per_kf = None
        self._names = {}


class SharedKeyframeMapReader:
    """
    Consumer-side helper to attach to the keyframe map shared memory and read snapshots.

    IMPORTANT:
      This schema can be large (hundreds of MB). Copying the full point tensor every read can
      be extremely expensive. Therefore:
        - `read(copy=False)` returns views (fast, but can observe torn updates).
        - `read(copy=True, copy_points=False)` copies only small pose/id arrays.
        - `read(copy=True, copy_points=True)` copies everything (may be slow / memory heavy).
    """

    def __init__(self, *, info_path: str) -> None:
        self.info_path = Path(info_path)
        self._info: Optional[dict] = None
        self._blocks: Dict[str, _ShmBlock] = {}

    def attach(self) -> None:
        info = json.loads(self.info_path.read_text(encoding="utf-8"))
        if str(info.get("schema", "")).strip() != "keyframe_map_v1":
            raise RuntimeError(f"Unsupported shm schema: {info.get('schema')!r}")
        self._info = info
        kmax = int(info["max_keyframes"])
        pmax = int(info["points_per_kf"])
        names = dict(info["names"])

        def _attach_block(key: str, shape: Tuple[int, ...], dtype: np.dtype) -> _ShmBlock:
            shm = shared_memory.SharedMemory(name=names[key], create=False)
            arr = np.ndarray(shape=shape, dtype=np.dtype(dtype), buffer=shm.buf)
            return _ShmBlock(shm=shm, arr=arr)

        self._blocks["meta_f64"] = _attach_block("meta_f64", (2,), np.float64)
        self._blocks["kf_count_i32"] = _attach_block("kf_count_i32", (1,), np.int32)
        self._blocks["curr_T_WC_f32"] = _attach_block("curr_T_WC_f32", (4, 4), np.float32)
        self._blocks["kf_T_WC_f32"] = _attach_block("kf_T_WC_f32", (kmax, 4, 4), np.float32)
        self._blocks["kf_frame_id_i32"] = _attach_block("kf_frame_id_i32", (kmax,), np.int32)
        self._blocks["kf_n_points_i32"] = _attach_block("kf_n_points_i32", (kmax,), np.int32)
        self._blocks["kf_points_k_f32"] = _attach_block("kf_points_k_f32", (kmax, pmax, 3), np.float32)
        self._blocks["kf_label_id_i32"] = _attach_block("kf_label_id_i32", (kmax, pmax), np.int32)
        self._blocks["kf_rgb_u8"] = _attach_block("kf_rgb_u8", (kmax, pmax, 3), np.uint8)

    @dataclass
    class Snapshot:
        frame_id: int
        timestamp_s: float
        kf_count: int
        curr_T_WC: np.ndarray
        kf_T_WC: np.ndarray
        kf_frame_id: np.ndarray
        kf_n_points: np.ndarray
        kf_points_k: np.ndarray
        kf_label_id: np.ndarray
        kf_rgb_u8: np.ndarray

    def read(
        self,
        *,
        copy: bool = False,
        copy_points: bool = False,
        max_retries: int = 3,
    ) -> "SharedKeyframeMapReader.Snapshot":
        if not self._blocks:
            raise RuntimeError("Reader is not attached. Call attach() first.")

        meta = self._blocks["meta_f64"].arr
        kf_count_arr = self._blocks["kf_count_i32"].arr
        curr_T = self._blocks["curr_T_WC_f32"].arr
        kf_T = self._blocks["kf_T_WC_f32"].arr
        kf_fid = self._blocks["kf_frame_id_i32"].arr
        kf_np = self._blocks["kf_n_points_i32"].arr
        kf_pts = self._blocks["kf_points_k_f32"].arr
        kf_lab = self._blocks["kf_label_id_i32"].arr
        kf_rgb = self._blocks["kf_rgb_u8"].arr

        for _ in range(int(max(1, max_retries))):
            fid0 = int(meta[0])
            ts0 = float(meta[1])
            k = int(kf_count_arr[0])
            k = max(0, min(k, kf_T.shape[0]))

            if copy:
                curr_T_out = np.array(curr_T, copy=True)
                kf_T_out = np.array(kf_T[:k, :, :], copy=True)
                kf_fid_out = np.array(kf_fid[:k], copy=True)
                kf_np_out = np.array(kf_np[:k], copy=True)
                if copy_points:
                    kf_pts_out = np.array(kf_pts[:k, :, :], copy=True)
                    kf_lab_out = np.array(kf_lab[:k, :], copy=True)
                    kf_rgb_out = np.array(kf_rgb[:k, :, :], copy=True)
                else:
                    kf_pts_out = kf_pts[:k, :, :]
                    kf_lab_out = kf_lab[:k, :]
                    kf_rgb_out = kf_rgb[:k, :, :]
            else:
                curr_T_out = curr_T
                kf_T_out = kf_T[:k, :, :]
                kf_fid_out = kf_fid[:k]
                kf_np_out = kf_np[:k]
                kf_pts_out = kf_pts[:k, :, :]
                kf_lab_out = kf_lab[:k, :]
                kf_rgb_out = kf_rgb[:k, :, :]

            fid1 = int(meta[0])
            ts1 = float(meta[1])
            if fid0 == fid1 and ts0 == ts1:
                return SharedKeyframeMapReader.Snapshot(
                    frame_id=fid1,
                    timestamp_s=ts1,
                    kf_count=k,
                    curr_T_WC=curr_T_out,
                    kf_T_WC=kf_T_out,
                    kf_frame_id=kf_fid_out,
                    kf_n_points=kf_np_out,
                    kf_points_k=kf_pts_out,
                    kf_label_id=kf_lab_out,
                    kf_rgb_u8=kf_rgb_out,
                )

        # Fall back: return last read even if meta changed.
        k_last = int(kf_count_arr[0])
        k_last = max(0, min(k_last, kf_T.shape[0]))
        return SharedKeyframeMapReader.Snapshot(
            frame_id=int(meta[0]),
            timestamp_s=float(meta[1]),
            kf_count=k_last,
            curr_T_WC=np.array(curr_T, copy=True) if copy else curr_T,
            kf_T_WC=np.array(kf_T[:k_last, :, :], copy=True) if copy else kf_T[:k_last, :, :],
            kf_frame_id=np.array(kf_fid[:k_last], copy=True) if copy else kf_fid[:k_last],
            kf_n_points=np.array(kf_np[:k_last], copy=True) if copy else kf_np[:k_last],
            kf_points_k=(np.array(kf_pts[:k_last, :, :], copy=True) if (copy and copy_points) else kf_pts[:k_last, :, :]),
            kf_label_id=(np.array(kf_lab[:k_last, :], copy=True) if (copy and copy_points) else kf_lab[:k_last, :]),
            kf_rgb_u8=(np.array(kf_rgb[:k_last, :, :], copy=True) if (copy and copy_points) else kf_rgb[:k_last, :, :]),
        )

    def close(self) -> None:
        for b in self._blocks.values():
            try:
                b.shm.close()
            except Exception:
                pass
        self._blocks = {}
        self._info = None


# =============================================================================
# TSDF volume (latest-only) shared memory
# =============================================================================
#
# Motivation
# ----------
# Downstream planning (e.g., CBF) often benefits from a volumetric representation.
# We publish a **local rolling TSDF** volume (see `mast3r_slam/planning_tsdf.py`) via shared memory:
#   - fixed-size 3D arrays (tsdf/weight/semantics)
#   - a small metadata block describing origin+voxel size and the current camera pose
#
# Design principles (consistent with other buffers)
# ------------------------------------------------
# - latest-only overwrite (no queues)
# - fixed maximum size determined by (radius_m, voxel_m)
# - write bulk arrays first, write meta last to reduce torn reads


class LatestSharedTsdfVolumeBuffer:
    """
    Producer-side latest-only TSDF(+semantic) volume buffer implemented with shared memory.

    What is published
    -----------------
    - meta_f64[2]         : [frame_id_as_float, timestamp_s] written LAST as a soft boundary marker
    - grid_f32[4]         : [origin_x, origin_y, origin_z, voxel_m]
    - dims_i32[3]         : [nx, ny, nz] (voxel grid dimensions)
    - curr_T_WC_f32[4,4]  : current camera->world pose (Sim3 matrix stored as float32 4x4)
    - tsdf_f32[nx,ny,nz]  : TSDF values in [-1,1]
    - weight_f32[nx,ny,nz]: TSDF integration weights
    - sem_label_i32[nx,ny,nz]  : per-voxel semantic hard label id (int32)
    - sem_weight_f32[nx,ny,nz] : per-voxel semantic vote weight (float32)
    - frame_sem_id_i32[H,W]    : (OPTIONAL, v2) per-frame semantic class IDs used for fusion/visualization
    - frame_rgb_u8[H,W,3]      : (OPTIONAL, v3) per-frame RGB image aligned with the fused observation

    Notes
    -----
    - This buffer publishes a *local* volume. The world pose can jump (e.g., relocalization),
      so consumers should not assume global consistency across long durations.
    - Consumers typically extract a surface mesh (marching cubes) from `tsdf_f32` for visualization
      or convert to occupancy/ESDF for planning.
    """

    def __init__(self, *, prefix: str, info_path: str) -> None:
        self.prefix = str(prefix)
        self.info_path = Path(info_path)

        self.radius_m: Optional[float] = None
        self.voxel_m: Optional[float] = None
        self.dims: Optional[Tuple[int, int, int]] = None

        self._meta: Optional[_ShmBlock] = None
        self._grid: Optional[_ShmBlock] = None
        self._dims: Optional[_ShmBlock] = None
        self._T_WC: Optional[_ShmBlock] = None
        self._tsdf: Optional[_ShmBlock] = None
        self._weight: Optional[_ShmBlock] = None
        self._sem_label: Optional[_ShmBlock] = None
        self._sem_weight: Optional[_ShmBlock] = None
        self._rgb_color: Optional[_ShmBlock] = None
        self._rgb_weight: Optional[_ShmBlock] = None
        self._frame_sem_id: Optional[_ShmBlock] = None
        self._frame_rgb_u8: Optional[_ShmBlock] = None

        # The per-frame semantic id map (H,W) is optional because:
        #   - It is not required for TSDF/semantic fusion itself.
        #   - It is only needed for debugging/visualization (e.g., showing the segmentation
        #     image next to the TSDF surface in a viewer process).
        #
        # When enabled, it is allocated with a fixed (H,W) determined by the SLAM image size
        # (after any img_downsample). Those dimensions are stable for the whole run.
        self.frame_hw: Optional[Tuple[int, int]] = None

        self._names: Dict[str, str] = {}

    def _create_block(self, *, key: str, shape: Tuple[int, ...], dtype: np.dtype) -> _ShmBlock:
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * int(dtype.itemsize)
        name = f"{self.prefix}_{key}_{os.getpid()}_{int(time.time()*1e6)}"
        shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
        arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
        self._names[key] = name
        return _ShmBlock(shm=shm, arr=arr)

    @staticmethod
    def _dims_from_radius_voxel(*, radius_m: float, voxel_m: float) -> Tuple[int, int, int]:
        """
        Match the `RollingTsdfSemanticVolume` sizing rule:
          n_half = ceil(radius/voxel)
          dims  = 2*n_half + 1  (odd, centered)
        """

        r = float(radius_m)
        v = float(voxel_m)
        n_half = int(np.ceil(max(1e-6, r) / max(1e-6, v)))
        n = 2 * n_half + 1
        return (int(n), int(n), int(n))

    def ensure(self, *, radius_m: float, voxel_m: float, frame_hw: Optional[Tuple[int, int]] = None) -> None:
        """
        Ensure the shared memory blocks exist for a given (radius_m, voxel_m).

        If parameters change, we destroy and recreate blocks.
        """

        r = float(radius_m)
        v = float(voxel_m)
        dims = self._dims_from_radius_voxel(radius_m=r, voxel_m=v)
        fhw = None if frame_hw is None else (int(frame_hw[0]), int(frame_hw[1]))
        if (
            self.radius_m == r
            and self.voxel_m == v
            and self.dims == dims
            and self.frame_hw == fhw
            and self._meta is not None
        ):
            return

        self.close(unlink=True)
        self.radius_m = r
        self.voxel_m = v
        self.dims = dims
        self.frame_hw = fhw
        self._names = {}

        nx, ny, nz = dims
        self._meta = self._create_block(key="meta_f64", shape=(2,), dtype=np.float64)
        self._grid = self._create_block(key="grid_f32", shape=(4,), dtype=np.float32)
        self._dims = self._create_block(key="dims_i32", shape=(3,), dtype=np.int32)
        self._T_WC = self._create_block(key="curr_T_WC_f32", shape=(4, 4), dtype=np.float32)
        self._tsdf = self._create_block(key="tsdf_f32", shape=(nx, ny, nz), dtype=np.float32)
        self._weight = self._create_block(key="weight_f32", shape=(nx, ny, nz), dtype=np.float32)
        self._sem_label = self._create_block(key="sem_label_i32", shape=(nx, ny, nz), dtype=np.int32)
        self._sem_weight = self._create_block(key="sem_weight_f32", shape=(nx, ny, nz), dtype=np.float32)
        self._rgb_color = self._create_block(key="rgb_color_f32", shape=(nx, ny, nz, 3), dtype=np.float32)
        self._rgb_weight = self._create_block(key="rgb_weight_f32", shape=(nx, ny, nz), dtype=np.float32)
        if fhw is not None:
            H, W = int(fhw[0]), int(fhw[1])
            self._frame_sem_id = self._create_block(key="frame_sem_id_i32", shape=(H, W), dtype=np.int32)
            # Per-frame RGB image aligned with the semantic/depth observation.
            #
            # This is used ONLY for visualization/debugging (e.g., 50/50 RGB+semantic overlay).
            # It does NOT affect TSDF fusion.
            self._frame_rgb_u8 = self._create_block(key="frame_rgb_u8", shape=(H, W, 3), dtype=np.uint8)
        else:
            self._frame_sem_id = None
            self._frame_rgb_u8 = None

        # Initialize with safe defaults for consumers.
        assert self._grid is not None
        assert self._dims is not None
        assert self._T_WC is not None
        assert self._tsdf is not None
        assert self._weight is not None
        assert self._sem_label is not None
        assert self._sem_weight is not None
        assert self._rgb_color is not None
        assert self._rgb_weight is not None
        assert self._meta is not None

        self._grid.arr[:] = np.array([0.0, 0.0, 0.0, float(v)], dtype=np.float32)
        self._dims.arr[:] = np.array([nx, ny, nz], dtype=np.int32)
        self._T_WC.arr[...] = np.eye(4, dtype=np.float32)
        self._tsdf.arr.fill(np.float32(1.0))
        self._weight.arr.fill(np.float32(0.0))
        self._sem_label.arr.fill(np.int32(-1))
        self._sem_weight.arr.fill(np.float32(0.0))
        self._rgb_color.arr.fill(np.float32(0.0))
        self._rgb_weight.arr.fill(np.float32(0.0))
        self._meta.arr[0] = 0.0
        self._meta.arr[1] = 0.0
        if self._frame_sem_id is not None:
            # Fill with -1 meaning "unknown/not provided". This is useful if the viewer
            # starts before semantic input becomes available.
            self._frame_sem_id.arr.fill(np.int32(-1))
        if self._frame_rgb_u8 is not None:
            # Initialize to black to avoid uninitialized memory being interpreted as an image.
            self._frame_rgb_u8.arr.fill(np.uint8(0))

        # Write schema description for consumers.
        self.info_path.parent.mkdir(parents=True, exist_ok=True)
        schema = "tsdf_volume_v3" if fhw is not None else "tsdf_volume_v1"
        info = dict(
            schema=schema,
            radius_m=float(r),
            voxel_m=float(v),
            dims=[int(nx), int(ny), int(nz)],
            frame_hw=[int(fhw[0]), int(fhw[1])] if fhw is not None else None,
            names=self._names,
            dtypes=dict(
                meta_f64="float64[2]",
                grid_f32="float32[4]",
                dims_i32="int32[3]",
                curr_T_WC_f32="float32[4,4]",
                tsdf_f32="float32[nx,ny,nz]",
                weight_f32="float32[nx,ny,nz]",
                sem_label_i32="int32[nx,ny,nz]",
                sem_weight_f32="float32[nx,ny,nz]",
                rgb_color_f32="float32[nx,ny,nz,3]",
                rgb_weight_f32="float32[nx,ny,nz]",
                frame_sem_id_i32="int32[H,W]" if fhw is not None else None,
                frame_rgb_u8="uint8[H,W,3]" if fhw is not None else None,
            ),
        )
        self.info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    def write(
        self,
        *,
        frame_id: int,
        timestamp_s: float,
        origin_w_f32: np.ndarray,
        voxel_m: float,
        curr_T_WC_f32: np.ndarray,
        tsdf_f32: np.ndarray,
        weight_f32: np.ndarray,
        sem_label_i32: np.ndarray,
        sem_weight_f32: np.ndarray,
        rgb_color_f32: Optional[np.ndarray] = None,
        rgb_weight_f32: Optional[np.ndarray] = None,
        frame_sem_id_i32: Optional[np.ndarray] = None,
        frame_rgb_u8: Optional[np.ndarray] = None,
    ) -> None:
        """
        Publish the latest TSDF volume snapshot into shared memory.

        IMPORTANT:
          This overwrites the entire volume arrays each call (latest-only).
          Keep the volume reasonably small and update at a moderate rate.
        """

        if self._meta is None or self._grid is None or self._dims is None:
            raise RuntimeError("TsdfVolumeBuffer is not initialized; call ensure(radius_m=..., voxel_m=...) first.")
        assert self._T_WC is not None
        assert self._tsdf is not None
        assert self._weight is not None
        assert self._sem_label is not None
        assert self._sem_weight is not None
        assert self._rgb_color is not None
        assert self._rgb_weight is not None

        # Bulk arrays first.
        self._grid.arr[:] = np.array(
            [
                float(np.asarray(origin_w_f32, dtype=np.float32).reshape(3)[0]),
                float(np.asarray(origin_w_f32, dtype=np.float32).reshape(3)[1]),
                float(np.asarray(origin_w_f32, dtype=np.float32).reshape(3)[2]),
                float(voxel_m),
            ],
            dtype=np.float32,
        )
        self._T_WC.arr[...] = np.asarray(curr_T_WC_f32, dtype=np.float32).reshape(4, 4)

        self._tsdf.arr[...] = np.asarray(tsdf_f32, dtype=np.float32).reshape(self._tsdf.arr.shape)
        self._weight.arr[...] = np.asarray(weight_f32, dtype=np.float32).reshape(self._weight.arr.shape)
        self._sem_label.arr[...] = np.asarray(sem_label_i32, dtype=np.int32).reshape(self._sem_label.arr.shape)
        self._sem_weight.arr[...] = np.asarray(sem_weight_f32, dtype=np.float32).reshape(self._sem_weight.arr.shape)
        if rgb_color_f32 is not None:
            self._rgb_color.arr[...] = np.asarray(rgb_color_f32, dtype=np.float32).reshape(self._rgb_color.arr.shape)
        else:
            self._rgb_color.arr.fill(np.float32(0.0))
        if rgb_weight_f32 is not None:
            self._rgb_weight.arr[...] = np.asarray(rgb_weight_f32, dtype=np.float32).reshape(self._rgb_weight.arr.shape)
        else:
            self._rgb_weight.arr.fill(np.float32(0.0))
        if self._frame_sem_id is not None and frame_sem_id_i32 is not None:
            # Publish the per-frame semantic id map for debugging/visualization.
            #
            # IMPORTANT:
            #   - This does NOT affect TSDF fusion. It is only an "observation snapshot" of the
            #     segmentation used by the fusion step.
            #   - The viewer can map these ids to a dataset palette (e.g., ADE20K class_colors).
            self._frame_sem_id.arr[...] = np.asarray(frame_sem_id_i32, dtype=np.int32).reshape(
                self._frame_sem_id.arr.shape
            )
        if self._frame_rgb_u8 is not None and frame_rgb_u8 is not None:
            # Publish the per-frame RGB image aligned with the observation.
            #
            # IMPORTANT:
            #   - Visualization only. This does NOT affect fusion.
            #   - Stored as uint8 to reduce shared-memory footprint and avoid float conversions
            #     in downstream viewers.
            self._frame_rgb_u8.arr[...] = np.asarray(frame_rgb_u8, dtype=np.uint8).reshape(
                self._frame_rgb_u8.arr.shape
            )

        # Meta LAST as soft boundary marker.
        self._meta.arr[0] = float(int(frame_id))
        self._meta.arr[1] = float(timestamp_s)

    def close(self, *, unlink: bool) -> None:
        blocks = [
            self._meta,
            self._grid,
            self._dims,
            self._T_WC,
            self._tsdf,
            self._weight,
            self._sem_label,
            self._sem_weight,
            self._rgb_color,
            self._rgb_weight,
            self._frame_sem_id,
            self._frame_rgb_u8,
        ]
        for b in blocks:
            if b is None:
                continue
            try:
                b.shm.close()
            except Exception:
                pass
            if unlink:
                try:
                    b.shm.unlink()
                except Exception:
                    pass

        self.radius_m = None
        self.voxel_m = None
        self.dims = None
        self.frame_hw = None
        self._meta = None
        self._grid = None
        self._dims = None
        self._T_WC = None
        self._tsdf = None
        self._weight = None
        self._sem_label = None
        self._sem_weight = None
        self._rgb_color = None
        self._rgb_weight = None
        self._frame_sem_id = None
        self._frame_rgb_u8 = None
        self._names = {}


class SharedTsdfVolumeReader:
    """
    Consumer-side helper to attach to the TSDF volume shared memory and read snapshots.

    The reader supports `copy=False` to return views without allocating the full volume.
    """

    def __init__(self, *, info_path: str) -> None:
        self.info_path = Path(info_path)
        self._info: Optional[dict] = None
        self._blocks: Dict[str, _ShmBlock] = {}

    def attach(self) -> None:
        info = json.loads(self.info_path.read_text(encoding="utf-8"))
        self._info = info
        names = dict(info["names"])
        dims = tuple(int(x) for x in info["dims"])
        nx, ny, nz = dims
        frame_hw = info.get("frame_hw", None)
        if frame_hw is not None:
            H = int(frame_hw[0])
            W = int(frame_hw[1])
        else:
            H, W = 0, 0

        def _attach_block(key: str, shape: Tuple[int, ...], dtype: np.dtype) -> _ShmBlock:
            shm = shared_memory.SharedMemory(name=names[key], create=False)
            arr = np.ndarray(shape=shape, dtype=np.dtype(dtype), buffer=shm.buf)
            return _ShmBlock(shm=shm, arr=arr)

        self._blocks["meta_f64"] = _attach_block("meta_f64", (2,), np.float64)
        self._blocks["grid_f32"] = _attach_block("grid_f32", (4,), np.float32)
        self._blocks["dims_i32"] = _attach_block("dims_i32", (3,), np.int32)
        self._blocks["curr_T_WC_f32"] = _attach_block("curr_T_WC_f32", (4, 4), np.float32)
        self._blocks["tsdf_f32"] = _attach_block("tsdf_f32", (nx, ny, nz), np.float32)
        self._blocks["weight_f32"] = _attach_block("weight_f32", (nx, ny, nz), np.float32)
        self._blocks["sem_label_i32"] = _attach_block("sem_label_i32", (nx, ny, nz), np.int32)
        self._blocks["sem_weight_f32"] = _attach_block("sem_weight_f32", (nx, ny, nz), np.float32)
        if "rgb_color_f32" in names:
            self._blocks["rgb_color_f32"] = _attach_block("rgb_color_f32", (nx, ny, nz, 3), np.float32)
        if "rgb_weight_f32" in names:
            self._blocks["rgb_weight_f32"] = _attach_block("rgb_weight_f32", (nx, ny, nz), np.float32)
        # Optional per-frame semantic ids for visualization (schema v2).
        if "frame_sem_id_i32" in names and frame_hw is not None:
            self._blocks["frame_sem_id_i32"] = _attach_block("frame_sem_id_i32", (H, W), np.int32)
        # Optional per-frame RGB image for visualization (schema v3).
        if "frame_rgb_u8" in names and frame_hw is not None:
            self._blocks["frame_rgb_u8"] = _attach_block("frame_rgb_u8", (H, W, 3), np.uint8)

    @dataclass
    class Snapshot:
        frame_id: int
        timestamp_s: float
        origin_w: np.ndarray
        voxel_m: float
        dims: np.ndarray
        curr_T_WC: np.ndarray
        tsdf: np.ndarray
        weight: np.ndarray
        sem_label: np.ndarray
        sem_weight: np.ndarray
        rgb_color: Optional[np.ndarray]
        rgb_weight: Optional[np.ndarray]
        frame_sem_id: Optional[np.ndarray]
        frame_rgb_u8: Optional[np.ndarray]

    def read(self, *, copy: bool = False, max_retries: int = 3) -> "SharedTsdfVolumeReader.Snapshot":
        if not self._blocks:
            raise RuntimeError("Reader is not attached. Call attach() first.")

        meta = self._blocks["meta_f64"].arr
        grid = self._blocks["grid_f32"].arr
        dims = self._blocks["dims_i32"].arr
        T = self._blocks["curr_T_WC_f32"].arr
        tsdf = self._blocks["tsdf_f32"].arr
        w = self._blocks["weight_f32"].arr
        lab = self._blocks["sem_label_i32"].arr
        sw = self._blocks["sem_weight_f32"].arr
        rgb = self._blocks.get("rgb_color_f32", None)
        rgb_w = self._blocks.get("rgb_weight_f32", None)
        frame_sem = self._blocks.get("frame_sem_id_i32", None)
        frame_rgb = self._blocks.get("frame_rgb_u8", None)

        for _ in range(int(max(1, max_retries))):
            fid0 = int(meta[0])
            ts0 = float(meta[1])

            if copy:
                grid_out = np.array(grid, copy=True)
                dims_out = np.array(dims, copy=True)
                T_out = np.array(T, copy=True)
                tsdf_out = np.array(tsdf, copy=True)
                w_out = np.array(w, copy=True)
                lab_out = np.array(lab, copy=True)
                sw_out = np.array(sw, copy=True)
                rgb_out = None if rgb is None else np.array(rgb.arr, copy=True)
                rgb_w_out = None if rgb_w is None else np.array(rgb_w.arr, copy=True)
                frame_sem_out = None
                if frame_sem is not None:
                    frame_sem_out = np.array(frame_sem.arr, copy=True)
                frame_rgb_out = None
                if frame_rgb is not None:
                    frame_rgb_out = np.array(frame_rgb.arr, copy=True)
            else:
                grid_out = grid
                dims_out = dims
                T_out = T
                tsdf_out = tsdf
                w_out = w
                lab_out = lab
                sw_out = sw
                rgb_out = None if rgb is None else rgb.arr
                rgb_w_out = None if rgb_w is None else rgb_w.arr
                frame_sem_out = None if frame_sem is None else frame_sem.arr
                frame_rgb_out = None if frame_rgb is None else frame_rgb.arr

            fid1 = int(meta[0])
            ts1 = float(meta[1])
            if fid0 == fid1 and ts0 == ts1:
                return SharedTsdfVolumeReader.Snapshot(
                    frame_id=fid1,
                    timestamp_s=ts1,
                    origin_w=np.asarray(grid_out[:3], dtype=np.float32),
                    voxel_m=float(grid_out[3]),
                    dims=np.asarray(dims_out, dtype=np.int32),
                    curr_T_WC=T_out,
                    tsdf=tsdf_out,
                    weight=w_out,
                    sem_label=lab_out,
                    sem_weight=sw_out,
                    rgb_color=rgb_out,
                    rgb_weight=rgb_w_out,
                    frame_sem_id=frame_sem_out,
                    frame_rgb_u8=frame_rgb_out,
                )

        # Fall back: return last read even if meta changed.
        grid_out = np.array(grid, copy=True) if copy else grid
        return SharedTsdfVolumeReader.Snapshot(
            frame_id=int(meta[0]),
            timestamp_s=float(meta[1]),
            origin_w=np.asarray(grid_out[:3], dtype=np.float32),
            voxel_m=float(grid_out[3]),
            dims=np.array(dims, copy=True) if copy else dims,
            curr_T_WC=np.array(T, copy=True) if copy else T,
            tsdf=np.array(tsdf, copy=True) if copy else tsdf,
            weight=np.array(w, copy=True) if copy else w,
            sem_label=np.array(lab, copy=True) if copy else lab,
            sem_weight=np.array(sw, copy=True) if copy else sw,
            rgb_color=(np.array(rgb.arr, copy=True) if (copy and rgb is not None) else (None if rgb is None else rgb.arr)),
            rgb_weight=(np.array(rgb_w.arr, copy=True) if (copy and rgb_w is not None) else (None if rgb_w is None else rgb_w.arr)),
            frame_sem_id=(np.array(frame_sem.arr, copy=True) if (copy and frame_sem is not None) else (None if frame_sem is None else frame_sem.arr)),
            frame_rgb_u8=(np.array(frame_rgb.arr, copy=True) if (copy and frame_rgb is not None) else (None if frame_rgb is None else frame_rgb.arr)),
        )

    def close(self) -> None:
        for b in self._blocks.values():
            try:
                b.shm.close()
            except Exception:
                pass
        self._blocks = {}
        self._info = None
