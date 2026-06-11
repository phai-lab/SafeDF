"""
AirSim RGB streaming source (latest-only, non-blocking).

This implementation is adapted from `test_streaming.py` in the repo.

Key requirements:
  - The SLAM main loop must never block on AirSim network latency.
  - We keep only the latest frame (queue maxsize=1) to avoid latency buildup.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RGBFrame:
    """A single RGB frame from AirSim."""

    rgb_u8: np.ndarray  # (H, W, 3) uint8 RGB
    timestamp_s: float  # wall-clock timestamp (seconds)


class AirSimRGBAsync:
    """
    Background thread grabs RGB frames from AirSim.

    Main-thread API:
      - `try_get_latest()` returns immediately (or None).
      - The main thread can run SLAM at its own pace without waiting for the simulator.
    """

    def __init__(
        self,
        *,
        ip: str = "127.0.0.1",
        camera: str = "0",
        vehicle_name: str = "",
        target_fps: int = 30,
        image_data_order: str = "rgb",
    ) -> None:
        self.ip = ip
        self.camera = camera
        self.vehicle_name = vehicle_name
        self.period = 1.0 / max(1, int(target_fps))
        self.image_data_order = str(image_data_order).lower()
        if self.image_data_order not in ("bgr", "rgb"):
            raise ValueError(
                f"image_data_order must be 'bgr' or 'rgb', got {image_data_order!r}"
            )

        self._q: queue.Queue[RGBFrame] = queue.Queue(maxsize=1)
        self._stop_flag = threading.Event()
        self._th = threading.Thread(target=self._worker, daemon=True)

        self.frames_grabbed = 0
        self._last: Optional[RGBFrame] = None

    def start(self) -> None:
        self._th.start()

    def stop(self) -> None:
        self._stop_flag.set()
        self._th.join(timeout=2.0)

    def try_get_latest(self) -> Optional[RGBFrame]:
        """Non-blocking get of the newest frame."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def last_frame(self) -> Optional[RGBFrame]:
        """Return the last grabbed frame (may be None)."""
        return self._last

    def _worker(self) -> None:
        # Import airsim lazily so non-streaming users do not require it.
        import airsim  # type: ignore

        # AirSim connections can fail transiently (e.g., simulator not started yet).
        # We keep retrying so the main program does not have to be restarted.
        client = None
        while not self._stop_flag.is_set():
            try:
                client = airsim.VehicleClient(self.ip)
                client.confirmConnection()
                break
            except Exception as e:
                # Do not crash the worker; keep retrying until the user stops the program.
                print("[AirSimGrab] connect error:", repr(e))
                time.sleep(0.5)
                continue

        if client is None or self._stop_flag.is_set():
            return

        req = [airsim.ImageRequest(self.camera, airsim.ImageType.Scene, False, False)]

        # Use a "next_t" scheduler to avoid drift from sleep inaccuracies.
        next_t = time.perf_counter()
        while not self._stop_flag.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += self.period

            try:
                r = client.simGetImages(req, vehicle_name=self.vehicle_name)[0]
                if r.width == 0 or r.height == 0 or len(r.image_data_uint8) == 0:
                    continue

                buf = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
                # IMPORTANT (color order):
                #   Different AirSim setups / API versions may expose `image_data_uint8` in either
                #   BGR or RGB order. The rest of MASt3R-SLAM assumes RGB numpy images.
                #
                #   We therefore make the color order explicit and configurable:
                #     - image_data_order='bgr' -> convert BGR -> RGB
                #     - image_data_order='rgb' -> use as-is
                #
                # Shape:
                #   (H, W, 3) uint8
                img = buf.reshape(r.height, r.width, 3)
                if self.image_data_order == "bgr":
                    rgb = img[..., ::-1].copy()
                else:
                    # IMPORTANT:
                    #   `np.frombuffer(...).reshape(...)` creates a view into AirSim's bytes buffer.
                    #   That buffer becomes invalid as soon as the next RPC returns (or `r` is freed),
                    #   so we MUST copy here to avoid subtle corruption (e.g., "RGB order looks wrong"
                    #   or random flicker) when the main thread consumes the frame later.
                    rgb = img.copy()
                ts = time.time()
                frm = RGBFrame(rgb_u8=rgb, timestamp_s=ts)

                self.frames_grabbed += 1
                self._last = frm

                # Latest-only: drop old frame if any.
                if self._q.full():
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                self._q.put_nowait(frm)

            except Exception as e:
                # Do not kill the worker on transient network/sim errors.
                print("[AirSimGrab] error:", repr(e))
                time.sleep(0.2)
