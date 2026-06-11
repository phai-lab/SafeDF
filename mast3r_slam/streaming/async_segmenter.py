"""
Asynchronous segmentation for streaming inputs (latest-only).

Purpose:
  - Segmentation inference can fluctuate in latency and shares GPU with SLAM.
  - To keep SLAM real-time, we run segmentation in a background thread and keep only the
    latest result. The SLAM loop consumes the newest available label map without blocking.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SegmentationResult:
    """Hard-label segmentation result."""

    label_hw: np.ndarray  # (H, W) int64
    timestamp_s: float


class LatestOnlyAsyncSegmenter:
    """
    Latest-only asynchronous segmenter.

    Producer (main thread):
      - `submit(rgb_float01, ts)` is non-blocking (drops pending input if needed).

    Consumer (SLAM main thread):
      - `get_latest()` returns the latest available segmentation output (or None).
    """

    def __init__(self, *, target_fps: int = 10, dataset: str = "ade20k") -> None:
        self.period = 1.0 / max(1, int(target_fps))
        # Which EfficientViT segmentation head to use ("ade20k" vs "cityscapes").
        # This is passed through to `segment_image_efficientvit_labels(...)`.
        self.dataset = str(dataset)
        self._in_q: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=1)
        self._stop_flag = threading.Event()
        self._th = threading.Thread(target=self._worker, daemon=True)

        self._lock = threading.Lock()
        self._latest: Optional[SegmentationResult] = None

    def start(self) -> None:
        self._th.start()

    def stop(self) -> None:
        self._stop_flag.set()
        self._th.join(timeout=2.0)

    def submit(self, img_rgb_float01: np.ndarray, timestamp_s: float) -> None:
        """
        Submit an image for segmentation (non-blocking).

        Input:
          img_rgb_float01: (H,W,3) float32 in [0,1]
        """
        if self._in_q.full():
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._in_q.put_nowait((img_rgb_float01, float(timestamp_s)))
        except queue.Full:
            pass

    def get_latest(self) -> Optional[SegmentationResult]:
        """Get the newest segmentation output (non-blocking)."""
        with self._lock:
            return self._latest

    def _worker(self) -> None:
        # Import here to avoid importing torch/EfficientViT unless the feature is used.
        from mast3r_slam.efficientvit_segmenter import segment_image_efficientvit_labels

        next_t = time.perf_counter()
        while not self._stop_flag.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += self.period

            try:
                img, ts = self._in_q.get_nowait()
            except queue.Empty:
                continue

            try:
                # IMPORTANT:
                #   We do not store logits/probabilities; we only return hard labels for
                #   real-time semantic stabilization / debug.
                label = segment_image_efficientvit_labels(img, dataset=self.dataset)
                with self._lock:
                    self._latest = SegmentationResult(label_hw=label, timestamp_s=float(ts))
            except Exception as e:
                # Don't crash the worker; segmentation may fail temporarily (e.g., OOM).
                print("[AsyncSeg] error:", repr(e))
                time.sleep(0.05)
