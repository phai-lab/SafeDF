"""
ZMQ RGB streaming source for online SLAM (latest-only, non-blocking).

Why this module exists
----------------------
Some deployments (e.g., LIMO robot with `ros_zmq_bridge`) provide RGB frames over a ZMQ
socket (typically `tcp://<ip>:<port>`). For SLAM we need:
  - non-blocking acquisition (the SLAM loop must never wait for network jitter)
  - latest-only semantics (do not build up latency by buffering old frames)
  - a small, stable API matching other streaming sources (e.g., AirSimRGBAsync)

This file provides:
  - `ZmqRGBAsync`: background-thread grabber that decodes JPEG bytes to RGB uint8
  - `ZmqJoystickTeleop` (optional): joystick -> ZMQ PUB command sender (JSON {"v","w"})

IMPORTANT
---------
We import heavy/optional dependencies lazily inside `start()`:
  - `pyzmq` is required only if you actually use this streaming source.
  - `opencv-python` (`cv2`) is required because we decode JPEG bytes.
  - `pygame` is required only if you enable joystick teleop.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ZmqRGBFrame:
    """
    A single RGB frame decoded from a ZMQ message.

    Attributes:
      rgb_u8:
        (H, W, 3) uint8 in RGB channel order.
      timestamp_s:
        Wall-clock timestamp (seconds) when the frame was received/decoded.
    """

    rgb_u8: np.ndarray
    timestamp_s: float


class ZmqRGBAsync:
    """
    Latest-only ZMQ RGB grabber.

    Message format assumption
    -------------------------
    This class expects each ZMQ message payload to be JPEG bytes (single-part message).
    If your publisher prepends a topic prefix into the message bytes, you can pass a `topic`
    prefix and we will strip it before decoding.

    Real-time behavior
    ------------------
    - We use `ZMQ_CONFLATE=1` to keep only the latest message in the socket.
    - We also use a `queue.Queue(maxsize=1)` so the main thread only ever consumes
      the latest decoded frame (no latency buildup).
    """

    def __init__(
        self,
        *,
        ip: str,
        vid_port: int,
        topic: bytes = b"",
        rcv_hwm: int = 1,
        poll_timeout_ms: int = 50,
    ) -> None:
        self.ip = str(ip)
        self.vid_port = int(vid_port)
        self.topic = bytes(topic)
        self.rcv_hwm = int(rcv_hwm)
        self.poll_timeout_ms = int(poll_timeout_ms)

        self._q: queue.Queue[ZmqRGBFrame] = queue.Queue(maxsize=1)
        self._stop_flag = threading.Event()
        self._th: Optional[threading.Thread] = None

        self._ctx = None
        self._sock = None

        self.frames_grabbed = 0
        self._last: Optional[ZmqRGBFrame] = None

    def start(self) -> None:
        """
        Start the background worker thread.

        Fail-fast policy:
          If required packages (`pyzmq`, `opencv-python`) are not available, raise an ImportError
          immediately. Streaming is an explicit opt-in feature, so silent degradation is not desired.
        """

        try:
            import zmq  # type: ignore
        except Exception as e:
            raise ImportError("ZMQ streaming requires `pyzmq`. Please install `pyzmq`.") from e

        try:
            import cv2  # type: ignore
        except Exception as e:
            raise ImportError(
                "ZMQ streaming requires OpenCV for JPEG decoding. Please install `opencv-python`."
            ) from e

        # Store modules on self so the worker can access them without re-importing.
        self._zmq = zmq
        self._cv2 = cv2

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(f"tcp://{self.ip}:{self.vid_port}")
        self._sock.setsockopt(zmq.SUBSCRIBE, self.topic)
        self._sock.setsockopt(zmq.RCVHWM, int(self.rcv_hwm))
        # Latest-only behavior on the socket itself (drops older queued messages).
        self._sock.setsockopt(zmq.CONFLATE, 1)

        self._stop_flag.clear()
        self._th = threading.Thread(target=self._worker, daemon=True)
        self._th.start()

    def stop(self) -> None:
        """Stop the worker thread and close the ZMQ socket."""

        self._stop_flag.set()
        try:
            if self._th is not None:
                self._th.join(timeout=2.0)
        except Exception:
            pass

        # Close socket last. Use linger=0 so shutdown is fast and does not block.
        try:
            if self._sock is not None:
                self._sock.close(linger=0)
        except Exception:
            pass

        self._sock = None
        self._ctx = None

    def try_get_latest(self) -> Optional[ZmqRGBFrame]:
        """
        Non-blocking get of the newest decoded frame.

        Returns:
          - `ZmqRGBFrame` if a new frame is available since the last call.
          - `None` if no new frame is currently available.
        """

        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def last_frame(self) -> Optional[ZmqRGBFrame]:
        """Return the last decoded frame (may be None)."""

        return self._last

    def _worker(self) -> None:
        """
        Background loop: receive JPEG bytes, decode, and publish into a latest-only queue.

        Implementation details:
          - We poll with a small timeout to react quickly to shutdown.
          - We tolerate decode errors and continue (robust for streaming).
        """

        sock = self._sock
        if sock is None:
            return

        zmq = self._zmq
        cv2 = self._cv2

        while not self._stop_flag.is_set():
            try:
                if not sock.poll(timeout=int(self.poll_timeout_ms)):
                    continue
                msg = sock.recv()

                # Optional topic stripping:
                # If the publisher embeds the topic as a prefix in the message bytes,
                # remove it before JPEG decode.
                payload = msg
                if self.topic and payload.startswith(self.topic):
                    payload = payload[len(self.topic) :]
                    # If a delimiter (space) is used after topic, strip it too.
                    if payload and payload[0] == 0x20:
                        payload = payload.lstrip(b" ")

                bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                frm = ZmqRGBFrame(rgb_u8=rgb, timestamp_s=time.time())
                self.frames_grabbed += 1
                self._last = frm

                # Latest-only: drop previous decoded frame if any.
                if self._q.full():
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                self._q.put_nowait(frm)
            except Exception:
                # Streaming robustness:
                # Do NOT crash the worker on transient errors. The main SLAM loop must keep running.
                continue


class ZmqJoystickTeleop:
    """
    Joystick -> ZMQ cmd publisher (JSON {"v": <m/s>, "w": <rad/s>}).

    This is optional and only used when explicitly enabled by the caller.
    It is designed to run in a background thread and never block SLAM.
    """

    def __init__(
        self,
        *,
        ip: str,
        cmd_port: int,
        max_linear: float,
        max_angular: float,
        deadzone: float = 0.05,
        lin_axis: int = 1,
        ang_axis: int = 2,
        rate_hz: float = 20.0,
    ) -> None:
        self.ip = str(ip)
        self.cmd_port = int(cmd_port)
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.deadzone = float(deadzone)
        self.lin_axis = int(lin_axis)
        self.ang_axis = int(ang_axis)
        self.rate_hz = float(rate_hz)

        self._ctx = None
        self._sock = None
        self._th: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @staticmethod
    def _dz(x: float, dz: float) -> float:
        """Deadzone helper: suppress small joystick noise."""

        return 0.0 if abs(float(x)) < float(dz) else float(x)

    def start(self) -> None:
        """
        Start joystick teleop.

        Fail-fast policy:
          - Requires `pygame` (joystick) and `pyzmq` (publisher).
          - If either is missing, raise immediately.
        """

        try:
            import zmq  # type: ignore
        except Exception as e:
            raise ImportError("Joystick teleop requires `pyzmq`. Please install `pyzmq`.") from e

        try:
            import pygame  # type: ignore
        except Exception as e:
            raise ImportError("Joystick teleop requires `pygame`. Please install `pygame`.") from e

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick found.")
        stick = pygame.joystick.Joystick(0)
        stick.init()

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.connect(f"tcp://{self.ip}:{self.cmd_port}")

        self._stop.clear()
        self._th = threading.Thread(target=self._worker, args=(pygame, stick), daemon=True)
        self._th.start()

    def stop(self) -> None:
        """Stop teleop worker and close socket."""

        self._stop.set()
        try:
            if self._th is not None:
                self._th.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._sock is not None:
                self._sock.close(linger=0)
        except Exception:
            pass

        self._sock = None
        self._ctx = None

    def _worker(self, pygame, stick) -> None:
        import json as _json

        period = 1.0 / max(1e-3, float(self.rate_hz))
        next_t = time.time()
        while not self._stop.is_set():
            try:
                pygame.event.pump()
                lin_raw = float(stick.get_axis(self.lin_axis))
                ang_raw = float(stick.get_axis(self.ang_axis))
                v = -self._dz(lin_raw, self.deadzone) * float(self.max_linear)
                w = -self._dz(ang_raw, self.deadzone) * float(self.max_angular)
                if self._sock is not None:
                    self._sock.send_string(_json.dumps({"v": float(v), "w": float(w)}))
            except Exception:
                pass

            now = time.time()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
            next_t = max(next_t + period, time.time())

