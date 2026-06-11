"""
Shared-memory publisher for streaming debug outputs (latest-only).

Goal
----
Sometimes you want the main SLAM process to keep producing "debug signals" (semantic/depth)
but you do NOT want an OpenCV GUI window to pop up (headless deployment, remote run, etc.).

This module provides a minimal mechanism to expose the latest debug payload to *other*
processes via `multiprocessing.shared_memory`, without adding any networking stack.

Design constraints
------------------
  - Latest-only (no backpressure, no latency buildup).
  - Minimal overhead (simple memcpy into shared memory).
  - No dependency on SLAM internals; the caller controls what it writes.

What is published
-----------------
We publish (optionally) the following arrays on a fixed (H,W) grid:
  - rgb_u8           : uint8  (H,W,3)  RGB
  - label_raw_hw     : int64  (H,W)    raw semantic label id/code
  - label_stable_hw  : int64  (H,W)    stable semantic label id/code (may be absent)
  - depth_raw_hw     : float32(H,W)    raw depth (may be absent)
  - depth_stable_hw  : float32(H,W)    stable depth (may be absent)

Additionally we publish a small metadata block:
  - frame_id (int64)
  - timestamp_s (float64)

Consumer note
-------------
A consumer process can attach to the shared memory blocks by name, using the `info.json`
that the publisher writes to disk (default: logs/stream_debug/shm_info.json).
This is intentionally "not an API": it's a tiny, inspectable contract for debugging.
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


class LatestSharedDebugBuffer:
    """
    Latest-only shared debug buffer implemented with shared memory.

    Typical usage in the SLAM main process:
      buf = LatestSharedDebugBuffer(prefix="mast3r_dbg", info_path="logs/stream_debug/shm_info.json")
      buf.ensure(shape_hw=(H,W))
      buf.write(frame_id=..., rgb_u8=..., label_raw=..., label_stable=..., depth_raw=..., depth_stable=...)

    A consumer process can:
      - read the json file for shm names + shapes
      - attach using `shared_memory.SharedMemory(name=...)`
      - create numpy views and read the latest data at any time
    """

    def __init__(self, *, prefix: str, info_path: str) -> None:
        self.prefix = str(prefix)
        self.info_path = Path(info_path)

        self.shape_hw: Optional[Tuple[int, int]] = None

        self._meta: Optional[_ShmBlock] = None  # float64[2] -> [frame_id_as_float, timestamp]
        self._rgb: Optional[_ShmBlock] = None
        self._label_raw: Optional[_ShmBlock] = None
        self._label_stable: Optional[_ShmBlock] = None
        self._depth_raw: Optional[_ShmBlock] = None
        self._depth_stable: Optional[_ShmBlock] = None

        # Keep names in a dict for easy export.
        self._names: Dict[str, str] = {}

    def _create_block(self, *, key: str, shape: Tuple[int, ...], dtype: np.dtype) -> _ShmBlock:
        """
        Allocate a shared memory block and return a numpy view.

        IMPORTANT:
          We include PID + current time in the name to avoid collisions with stale segments
          from previous runs (common during debugging).
        """

        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * int(dtype.itemsize)
        name = f"{self.prefix}_{key}_{os.getpid()}_{int(time.time()*1e6)}"
        shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
        arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
        self._names[key] = name
        return _ShmBlock(shm=shm, arr=arr)

    def ensure(self, *, shape_hw: Tuple[int, int]) -> None:
        """
        Ensure the shared memory blocks exist for a given (H,W).

        If shape changes, we destroy and recreate blocks.
        """

        h, w = int(shape_hw[0]), int(shape_hw[1])
        if self.shape_hw == (h, w) and self._meta is not None:
            return

        # Recreate everything on shape change.
        self.close(unlink=True)

        self.shape_hw = (h, w)
        self._names = {}

        # meta: float64[2] -> [frame_id, timestamp_s]
        self._meta = self._create_block(key="meta", shape=(2,), dtype=np.float64)
        self._rgb = self._create_block(key="rgb_u8", shape=(h, w, 3), dtype=np.uint8)
        self._label_raw = self._create_block(key="label_raw_i64", shape=(h, w), dtype=np.int64)
        self._label_stable = self._create_block(key="label_stable_i64", shape=(h, w), dtype=np.int64)
        self._depth_raw = self._create_block(key="depth_raw_f32", shape=(h, w), dtype=np.float32)
        self._depth_stable = self._create_block(key="depth_stable_f32", shape=(h, w), dtype=np.float32)

        # Write an info json to disk so other processes can attach.
        self.info_path.parent.mkdir(parents=True, exist_ok=True)
        info = dict(
            shape_hw=[h, w],
            names=self._names,
            dtypes=dict(
                meta="float64[2]",
                rgb_u8="uint8[H,W,3]",
                label_raw_i64="int64[H,W]",
                label_stable_i64="int64[H,W]",
                depth_raw_f32="float32[H,W]",
                depth_stable_f32="float32[H,W]",
            ),
        )
        self.info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    def write(
        self,
        *,
        frame_id: int,
        timestamp_s: float,
        rgb_u8: np.ndarray,
        label_raw_hw: np.ndarray,
        label_stable_hw: Optional[np.ndarray],
        depth_raw_hw: Optional[np.ndarray],
        depth_stable_hw: Optional[np.ndarray],
    ) -> None:
        """
        Publish the latest debug payload into shared memory.

        Inputs are expected to already be CPU numpy arrays with the exact shapes:
          - rgb_u8          : (H,W,3) uint8 RGB
          - label_raw_hw    : (H,W)   int64
          - label_stable_hw : (H,W)   int64 or None (if None -> copy raw)
          - depth_raw_hw    : (H,W)   float32 or None (if None -> fill NaN)
          - depth_stable_hw : (H,W)   float32 or None (if None -> fill NaN)
        """

        if self.shape_hw is None or self._meta is None:
            raise RuntimeError("SharedDebugBuffer is not initialized; call ensure(shape_hw=...) first.")

        h, w = self.shape_hw
        if rgb_u8.shape[:2] != (h, w):
            raise ValueError(f"rgb_u8 shape mismatch: expected {(h,w,3)}, got {rgb_u8.shape}")

        # Copy arrays (latest-only).
        assert self._rgb is not None
        assert self._label_raw is not None
        assert self._label_stable is not None
        assert self._depth_raw is not None
        assert self._depth_stable is not None

        self._rgb.arr[...] = rgb_u8
        self._label_raw.arr[...] = label_raw_hw
        if label_stable_hw is None:
            self._label_stable.arr[...] = label_raw_hw
        else:
            self._label_stable.arr[...] = label_stable_hw

        if depth_raw_hw is None:
            self._depth_raw.arr.fill(np.nan)
        else:
            self._depth_raw.arr[...] = depth_raw_hw

        if depth_stable_hw is None:
            self._depth_stable.arr.fill(np.nan)
        else:
            self._depth_stable.arr[...] = depth_stable_hw

        # Meta is written LAST so a consumer can treat it as a "frame boundary" marker:
        # if (frame_id,timestamp) changes, the bulk arrays above are already updated.
        #
        # NOTE:
        #   This is still not a strict atomic transaction, but it reduces the likelihood of a
        #   consumer seeing a new frame_id with old array contents.
        self._meta.arr[0] = float(int(frame_id))
        self._meta.arr[1] = float(timestamp_s)

    def close(self, *, unlink: bool) -> None:
        """
        Close (and optionally unlink) all shared memory segments.

        IMPORTANT:
          - `close()` detaches from the segment in this process.
          - `unlink()` removes the shared memory name. Existing attached processes keep working
            until they also close, but new attaches will fail.
        """

        blocks = [
            self._meta,
            self._rgb,
            self._label_raw,
            self._label_stable,
            self._depth_raw,
            self._depth_stable,
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
        self._rgb = None
        self._label_raw = None
        self._label_stable = None
        self._depth_raw = None
        self._depth_stable = None
        self.shape_hw = None
