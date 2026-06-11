import argparse
import datetime
import json
import pathlib
import os
import sys
import time
import subprocess
import atexit
import cv2
import numpy as np
import lietorch
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame, create_frame_semantic
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualizatio_semantics import WindowMsg, run_visualization
from mast3r_slam.visualization_utils import depth2rgb
from mast3r_slam.semantic_stabilizer import ensure_hard_label_hw, label_code_to_rgb
from mast3r_slam.lietorch_utils import as_SE3
import torch.multiprocessing as mp


# -----------------------------------------------------------------------------
# Lightweight timing / profiling
#
# Why:
#   Speedy MASt3R accelerates only some parts of the pipeline (e.g., attention /
#   RoPE / AMP). End-to-end SLAM FPS may still be dominated by other components
#   (semantic segmentation, matching backend, GN, I/O, etc.).
#
# Design goals:
#   - When disabled, have negligible overhead (single `if` checks; no monkeypatching).
#   - When enabled, print averages every N frames and (optionally) time key MASt3R
#     and matching functions via monkeypatching. Profiling mode WILL affect FPS
#     because it introduces CUDA synchronization to measure GPU time.
# -----------------------------------------------------------------------------
# IMPORTANT:
#   This profiling mode *forces CUDA synchronization* via `torch.cuda.Event` timing.
#   That can severely reduce FPS and distort real-time performance.
#
#   Keep it OFF by default. Enable explicitly via:
#     MAST3R_SLAM_ENABLE_TIMING=1 python main_semantic.py ...
#
#   We intentionally avoid adding more CLI flags here to keep the interface stable.
ENABLE_TIMING = bool(int(os.environ.get("MAST3R_SLAM_ENABLE_TIMING", "0")))
TIMING_PRINT_EVERY = 30

_timing = None
if ENABLE_TIMING:
    import os
    from collections import defaultdict

    class _Timing:
        def __init__(self, print_every: int = 30):
            self.print_every = int(print_every)
            self._sum_ms = defaultdict(float)
            self._count = defaultdict(int)
            self._t0 = time.perf_counter()
            self._frames = 0

        def add_ms(self, key: str, ms: float):
            self._sum_ms[key] += float(ms)
            self._count[key] += 1

        def tick_frame(self):
            self._frames += 1

        def maybe_print(self, frame_idx: int):
            if self.print_every <= 0:
                return
            if frame_idx <= 0 or frame_idx % self.print_every != 0:
                return
            dt = time.perf_counter() - self._t0
            fps = (self._frames / dt) if dt > 0 else 0.0

            def avg(key: str) -> float:
                c = self._count.get(key, 0)
                return (self._sum_ms.get(key, 0.0) / c) if c else 0.0

            pid = os.getpid()
            proc_name = mp.current_process().name if hasattr(mp, "current_process") else "unknown"
            print(
                "[TIMING]"
                f" pid={pid}"
                f" proc={proc_name}"
                f" frames={self._frames}"
                f" fps={fps:.2f}"
                f" dataset_ms={avg('dataset_ms'):.2f}"
                f" frame_ms={avg('frame_ms'):.2f}"
                f" track_ms={avg('track_ms'):.2f}"
                f" mono_ms={avg('mono_ms'):.2f}"
                f" mast3r_asym_ms={avg('mast3r_asym_ms'):.2f}"
                f" mast3r_sym_ms={avg('mast3r_sym_ms'):.2f}"
                f" match_ms={avg('match_ms'):.2f}"
                f" backend_add_ms={avg('backend_add_ms'):.2f}"
                f" backend_solve_ms={avg('backend_solve_ms'):.2f}"
            )

            # Reset window
            self._sum_ms.clear()
            self._count.clear()
            self._t0 = time.perf_counter()
            self._frames = 0

    _timing = _Timing(print_every=TIMING_PRINT_EVERY)

    def _wrap_cuda_ms(key: str, fn):
        def wrapped(*args, **kwargs):
            if not torch.cuda.is_available():
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                _timing.add_ms(key, (time.perf_counter() - t0) * 1000.0)
                return out

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*args, **kwargs)
            end.record()
            end.synchronize()
            _timing.add_ms(key, start.elapsed_time(end))
            return out

        return wrapped

    # Monkeypatch a few hotspots to isolate where time goes.
    # This affects BOTH the main and backend processes (spawn imports this module),
    # but only when ENABLE_TIMING=True.
    import mast3r_slam.mast3r_utils as _mu
    import mast3r_slam.matching as _mm

    _mu.mast3r_asymmetric_inference = _wrap_cuda_ms(
        "mast3r_asym_ms", _mu.mast3r_asymmetric_inference
    )
    _mu.mast3r_decode_symmetric_batch = _wrap_cuda_ms(
        "mast3r_sym_ms", _mu.mast3r_decode_symmetric_batch
    )
    _mu.mast3r_inference_mono = _wrap_cuda_ms("mono_ms", _mu.mast3r_inference_mono)
    _mm.match = _wrap_cuda_ms("match_ms", _mm.match)

    # Ensure this module's imported symbol points at the wrapped function too.
    mast3r_inference_mono = _mu.mast3r_inference_mono


def maybe_save_depth_image(frame, depth_dir, scale=1.0):
    return
    if depth_dir is None or frame.X_canon is None:
        return
    h, w = frame.img_shape.flatten().long().cpu().tolist()
    depth = frame.X_canon.view(h, w, 3)[..., 2].detach().cpu().numpy()
    # depth = depth * float(scale)
    # print(depth.mean())
    depth_vis = depth2rgb(depth)
    out_path = depth_dir / f"{int(frame.frame_id):06d}.png"
    cv2.imwrite(str(out_path), (depth_vis * 255).astype("uint8"))


def semantic_depth_hook(label_hw: torch.Tensor, depth_hw: torch.Tensor, frame_id: int) -> None:
    """
    User hook for downstream planning (disabled by default).

    Parameters:
      label_hw: (H, W) int64
        Hard semantic label IDs on the MASt3R match grid. The label space is whatever your
        segmentation network outputs (e.g., 0..C-1). If your semantics came in as an RGB mask,
        this may be a 24-bit packed "label code".

      depth_hw: (H, W) float32
        Depth map on the same grid as `label_hw`. Depending on `--depth_source` this can be:
          - raw_z     : per-frame pointmap Z (fast, can jitter)
          - kf_warp_z : keyframe-warp Z + fast hole fill (more stable)

      frame_id: int
        Current frame index.

    Notes:
      - This function is intentionally a no-op. To integrate your planner, either:
          (a) replace its body, or
          (b) uncomment the single call site in the main loop.
      - Keeping the hook out of the SLAM modules avoids unintended feedback loops.
    """

    # No-op by design.
    return


def _pose_matrix_from_json_entry(entry: dict, pose_key: str) -> np.ndarray:
    if pose_key not in entry:
        raise KeyError(f"pose key {pose_key!r} not found")
    T = np.asarray(entry[pose_key], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"pose matrix has shape {T.shape}, expected (4, 4)")
    if abs(float(T[3, 3])) > 1e-12 and abs(float(T[3, 3]) - 1.0) > 1e-9:
        T = T / float(T[3, 3])
    return T


def _rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    # Project slightly non-orthogonal pose-json rotations back to SO(3).
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt

    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    if q[3] < 0.0:
        q = -q
    return q


def _load_external_tracking_pose_override(args):
    pose_json = str(getattr(args, "external_tracking_pose_json", "") or "")
    if not pose_json:
        return None

    pose_path = pathlib.Path(pose_json)
    table = json.loads(pose_path.read_text())
    pose_key = str(args.external_tracking_pose_key)
    frame_pattern = str(args.external_tracking_pose_frame_pattern)
    frame_stride = int(args.external_tracking_pose_frame_stride)
    coord_space = str(args.external_tracking_pose_space)

    first_key = frame_pattern.format(frame_id=0)
    if first_key not in table:
        candidates = [k for k, v in table.items() if isinstance(v, dict) and pose_key in v]
        if not candidates:
            raise ValueError(f"No pose entries with key {pose_key!r} in {pose_path}")
        first_key = sorted(candidates)[0]

    T0 = _pose_matrix_from_json_entry(table[first_key], pose_key)
    print(
        "[ExternalTrackingPose] overriding frame.T_WC from"
        f" {pose_path} key={pose_key} stride={frame_stride}"
        f" space={coord_space} anchor={first_key}"
    )
    return {
        "table": table,
        "pose_key": pose_key,
        "frame_pattern": frame_pattern,
        "frame_stride": frame_stride,
        "coord_space": coord_space,
        "T0_inv": np.linalg.inv(T0),
    }


def _external_tracking_sim3_for_frame(pose_override, frame_id: int, device, dtype):
    src_frame_id = int(frame_id) * int(pose_override["frame_stride"])
    pose_key = pose_override["pose_key"]
    frame_key = pose_override["frame_pattern"].format(frame_id=src_frame_id)
    table = pose_override["table"]
    if frame_key not in table:
        raise KeyError(f"external tracking pose missing frame key {frame_key!r}")

    T = _pose_matrix_from_json_entry(table[frame_key], pose_key)
    if pose_override["coord_space"] == "first-frame":
        T = pose_override["T0_inv"] @ T
    elif pose_override["coord_space"] != "aligned-world":
        raise ValueError(f"Unsupported external tracking pose space {pose_override['coord_space']!r}")

    qx, qy, qz, qw = _rotation_matrix_to_quat_xyzw(T[:3, :3])
    tx, ty, tz = T[:3, 3].astype(np.float64).tolist()
    data = torch.tensor(
        [[tx, ty, tz, qx, qy, qz, qw, 1.0]],
        device=device,
        dtype=dtype,
    )
    return lietorch.Sim3(data)


def min_depth_per_class(
    label_hw: torch.Tensor, depth_hw: torch.Tensor, valid_hw: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-class minimum depth (and counts) using a fast scatter-reduction.

    Inputs:
      label_hw: (H, W) int64
      depth_hw: (H, W) float32
      valid_hw: (H, W) bool
      num_classes: int

    Outputs:
      min_depth: (num_classes,) float32  (inf if a class has no valid pixels)
      count:     (num_classes,) int64

    Performance:
      - O(H*W) without sorting/unique.
      - Requires label IDs to be in [0, num_classes).
    """

    labels = label_hw.reshape(-1).to(torch.int64)
    depth = depth_hw.reshape(-1).to(torch.float32)
    valid = valid_hw.reshape(-1).to(torch.bool) & torch.isfinite(depth) & (depth > 0.0)

    if labels.numel() == 0:
        return (
            torch.full((num_classes,), float("inf"), device=label_hw.device, dtype=torch.float32),
            torch.zeros((num_classes,), device=label_hw.device, dtype=torch.int64),
        )

    # Filter to labels in range.
    in_range = (labels >= 0) & (labels < int(num_classes))
    valid = valid & in_range

    if not valid.any():
        return (
            torch.full((num_classes,), float("inf"), device=label_hw.device, dtype=torch.float32),
            torch.zeros((num_classes,), device=label_hw.device, dtype=torch.int64),
        )

    labels_v = labels[valid]
    depth_v = depth[valid]

    # Count per class.
    count = torch.bincount(labels_v, minlength=int(num_classes)).to(torch.int64)

    # Min depth per class via scatter_reduce.
    min_depth = torch.full(
        (int(num_classes),), float("inf"), device=label_hw.device, dtype=torch.float32
    )
    # `scatter_reduce_` is available in PyTorch 2.0+.
    min_depth.scatter_reduce_(
        0, labels_v, depth_v, reduce="amin", include_self=True
    )
    return min_depth, count


def min_depth_and_argmin_per_class(
    label_hw: torch.Tensor,
    depth_hw: torch.Tensor,
    valid_hw: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-class minimum depth AND the pixel location that attains that minimum.

    Inputs:
      label_hw: (H, W) int64
      depth_hw: (H, W) float32
      valid_hw: (H, W) bool
      num_classes: int

    Outputs:
      min_depth:  (C,) float32  (inf if class absent)
      argmin_idx: (C,) int64    (linear index in [0, H*W), or -1 if class absent)
      count:      (C,) int64    (#valid pixels per class)

    Why this is designed this way:
      - For debug/planning, we often want to *visualize* which pixel produced the minimum depth.
      - We avoid Python loops over classes by using scatter reductions.

    Implementation details:
      - First compute `min_depth[c]` via scatter_reduce(amin).
      - Then find pixels that match this min (depth == min_depth[label]) and scatter_reduce(amin)
        over their linear indices to pick a deterministic argmin per class.
    """

    h, w = (int(label_hw.shape[0]), int(label_hw.shape[1]))
    n = h * w
    labels = label_hw.reshape(-1).to(torch.int64)
    depth = depth_hw.reshape(-1).to(torch.float32)
    valid = valid_hw.reshape(-1).to(torch.bool) & torch.isfinite(depth) & (depth > 0.0)

    # Filter to labels in range.
    in_range = (labels >= 0) & (labels < int(num_classes))
    valid = valid & in_range

    min_depth = torch.full((int(num_classes),), float("inf"), device=label_hw.device, dtype=torch.float32)
    count = torch.zeros((int(num_classes),), device=label_hw.device, dtype=torch.int64)
    argmin_idx = torch.full((int(num_classes),), -1, device=label_hw.device, dtype=torch.int64)

    if not valid.any():
        return min_depth, argmin_idx, count

    labels_v = labels[valid]
    depth_v = depth[valid]
    idx_v = torch.arange(n, device=label_hw.device, dtype=torch.int64)[valid]

    count = torch.bincount(labels_v, minlength=int(num_classes)).to(torch.int64)
    min_depth.scatter_reduce_(0, labels_v, depth_v, reduce="amin", include_self=True)

    # Identify which valid pixels attain the per-class minimum.
    min_for_pixel = min_depth[labels_v]
    is_min = depth_v == min_for_pixel
    if is_min.any():
        labels_min = labels_v[is_min]
        idx_min = idx_v[is_min]
        # Scatter-reduce the smallest linear index among min-attaining pixels.
        argmin_tmp = torch.full((int(num_classes),), n + 1, device=label_hw.device, dtype=torch.int64)
        argmin_tmp.scatter_reduce_(0, labels_min, idx_min, reduce="amin", include_self=True)
        argmin_idx = torch.where(count > 0, argmin_tmp, torch.full_like(argmin_tmp, -1))
        # For safety: any still-large entries (should not happen) are treated as "absent".
        argmin_idx = torch.where(argmin_idx <= n, argmin_idx, torch.full_like(argmin_idx, -1))

    return min_depth, argmin_idx, count


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            if _timing is None:
                factor_graph.add_factors(
                    kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
                )
            else:
                t0 = time.perf_counter()
                factor_graph.add_factors(
                    kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
                )
                torch.cuda.synchronize()
                _timing.add_ms("backend_add_ms", (time.perf_counter() - t0) * 1000.0)

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if _timing is None:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        else:
            t0 = time.perf_counter()
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
            torch.cuda.synchronize()
            _timing.add_ms("backend_solve_ms", (time.perf_counter() - t0) * 1000.0)

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)

        if _timing is not None:
            _timing.tick_frame()
            _timing.maybe_print(int(idx))


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")

    # -------------------------------------------------------------------------
    # Input source selection (dataset vs. streaming)
    #
    # Motivation:
    #   For online deployment (e.g., AirSim), we want to run SLAM on a live RGB stream
    #   without blocking the main loop on streaming latency.
    #
    # Design:
    #   - "dataset": existing behavior (read images from dataset path).
    #   - "airsim":  stream RGB frames asynchronously from AirSim (latest-only).
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--input_source",
        choices=["dataset", "airsim", "limo_zmq"],
        default="dataset",
        help="Input source: dataset, airsim streaming, or limo_zmq streaming (default: dataset).",
    )
    parser.add_argument("--airsim_ip", default="127.0.0.1", help="AirSim IP (default: 127.0.0.1).")
    parser.add_argument("--airsim_camera", default="0", help="AirSim camera name/index (default: 0).")
    parser.add_argument("--airsim_vehicle_name", default="", help="AirSim vehicle name (default: empty).")
    parser.add_argument("--airsim_target_fps", type=int, default=30, help="AirSim grab target FPS (default: 30).")
    parser.add_argument(
        "--airsim_image_order",
        choices=["bgr", "rgb"],
        default="bgr",
        help=(
            "Color order of AirSim `image_data_uint8` (default: rgb). "
            "The rest of the code assumes RGB; if your AirSim build returns BGR, set this to 'bgr'."
        ),
    )

    # -------------------------------------------------------------------------
    # LIMO ZMQ streaming (ros_zmq_bridge)
    #
    # Motivation:
    #   Some deployments stream camera images over ZMQ (e.g., a ROS bridge on the robot).
    #   This mirrors the AirSim "latest-only" design: SLAM must not block on network latency.
    #
    # Behavior:
    #   - Video: SUB receives JPEG bytes and decodes them to RGB uint8.
    #   - Cmd:   (optional) joystick teleop publishes JSON {"v":..., "w":...} over PUB.
    #
    # IMPORTANT:
    #   - ZMQ and OpenCV are imported lazily only when `--input_source limo_zmq` is used.
    #   - Joystick teleop is also optional and only starts if `--enable_joystick` is set.
    # -------------------------------------------------------------------------
    parser.add_argument("--bridge_ip", type=str, default="127.0.0.1", help="IP of ros_zmq_bridge host (default: 127.0.0.1).")
    parser.add_argument("--bridge_vid_port", type=int, default=5555, help="ZMQ video port (SUB) (default: 5555).")
    parser.add_argument("--bridge_cmd_port", type=int, default=5556, help="ZMQ cmd port (PUB) (default: 5556).")
    parser.add_argument(
        "--enable_joystick",
        action="store_true",
        help="Enable joystick teleop -> ZMQ cmd publisher (only meaningful for --input_source limo_zmq).",
    )
    parser.add_argument("--max_linear", type=float, default=0.5, help="Max linear velocity for joystick teleop (m/s).")
    parser.add_argument("--max_angular", type=float, default=1.0, help="Max angular velocity for joystick teleop (rad/s).")
    parser.add_argument("--deadzone", type=float, default=0.05, help="Joystick deadzone (default: 0.05).")
    parser.add_argument("--joystick_lin_axis", type=int, default=1, help="Joystick axis index for linear velocity (default: 1).")
    parser.add_argument("--joystick_ang_axis", type=int, default=2, help="Joystick axis index for angular velocity (default: 2).")
    parser.add_argument("--joystick_rate_hz", type=float, default=20.0, help="Joystick command publish rate (Hz).")
    parser.add_argument(
        "--stream_img_size",
        type=int,
        default=224,
        help="Streaming resize/crop size fed into MASt3R resize_img (default: 224).",
    )
    parser.add_argument(
        "--stream_sleep_ms",
        type=int,
        default=1,
        help="When no new streaming frame is available, sleep this many ms (default: 1).",
    )
    parser.add_argument(
        "--stream_max_frames",
        type=int,
        default=0,
        help="Max frames to process in streaming mode; 0 means run until terminated (default: 0).",
    )

    # -------------------------------------------------------------------------
    # Streaming segmentation controls (avoid slowing SLAM)
    #
    # Motivation:
    #   EfficientViT segmentation (even if fast) can still contend with SLAM on the GPU.
    #   To keep SLAM real-time, we can run segmentation asynchronously at a capped FPS and
    #   use the latest available labels without blocking tracking.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--stream_async_semantic",
        dest="stream_async_semantic",
        action="store_true",
        default=True,
        help="Run streaming segmentation asynchronously (default: enabled).",
    )
    parser.add_argument(
        "--stream_sync_semantic",
        dest="stream_async_semantic",
        action="store_false",
        help="Run streaming segmentation synchronously (debug; may slow SLAM).",
    )
    parser.add_argument(
        "--stream_semantic_fps",
        type=int,
        default=10,
        help="Segmentation target FPS in streaming async mode (default: 10).",
    )

    # -------------------------------------------------------------------------
    # EfficientViT head selection (ADE20K vs Cityscapes)
    #
    # Motivation:
    #   EfficientViT provides dataset-specific segmentation heads (different class counts).
    #   For ablation / deployment, users may want to switch between ADE20K and Cityscapes
    #   without editing code.
    #
    # Weight file convention used in this repo:
    #   - ADE20K     : `efficientvit/l2.pt`
    #   - Cityscapes : `efficientvit/l2_cityscapes.pt`
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--efficientvit_dataset",
        choices=["ade20k", "cityscapes"],
        default="ade20k",
        help="EfficientViT segmentation head/dataset (default: ade20k).",
    )

    # -------------------------------------------------------------------------
    # Streaming debug visualization (optional)
    #
    # This spawns a separate process with an OpenCV window so GUI latency cannot
    # block the SLAM main loop. It is purely for debugging.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--enable_stream_debug_viz",
        action="store_true",
        default=False,
        help="Show a debug window with semantic+depth overlays and sampled depth stats (default: disabled).",
    )
    parser.add_argument(
        "--stream_viz_headless",
        action="store_true",
        default=False,
        help=(
            "Headless debug mode: do not open an OpenCV window, but keep publishing the latest "
            "debug semantic/depth payload for other processes to attach to (default: disabled)."
        ),
    )
    parser.add_argument("--stream_viz_fps", type=int, default=10, help="Debug viz update FPS (default: 10).")
    parser.add_argument(
        "--stream_viz_topk",
        type=int,
        default=9,
        help="Show top-K semantic classes by nearest depth in the debug window (default: 9).",
    )
    parser.add_argument("--stream_viz_alpha", type=float, default=0.6, help="Segmentation overlay alpha (default: 0.6).")
    parser.add_argument("--stream_viz_scale", type=int, default=2, help="Debug viz scale factor (default: 2).")
    parser.add_argument(
        "--stream_viz_semantic_source",
        choices=["stable", "raw"],
        default="stable",
        help=(
            "Semantic source for the debug overlay (default: stable). "
            "'stable' uses the post-warp stabilized semantic (if available), "
            "'raw' uses the per-frame EfficientViT output."
        ),
    )
    parser.add_argument(
        "--stream_viz_depth_source",
        choices=["stable", "raw"],
        default="stable",
        help=(
            "Depth source shown in the debug window (default: stable). "
            "'stable' prefers keyframe-warp depth when available; "
            "'raw' shows current-frame pointmap Z."
        ),
    )
    # -------------------------------------------------------------------------
    # Stream debug viz: planning point cloud reprojection layout (optional).
    #
    # Motivation:
    #   We previously provided a separate "reader" script to visualize the planning point cloud.
    #   Users requested moving that visualization into the existing stream debug window so
    #   everything is visible in a single OpenCV window.
    #
    # IMPORTANT:
    #   - This is visualization-only.
    #   - It relies on the planning pointcloud publisher shared memory
    #     (typically enabled via --enable_planning_pointcloud_publish).
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--stream_viz_layout",
        choices=["legacy", "planning_pointcloud"],
        default="planning_pointcloud",
        help=(
            "Stream debug window layout (default: planning_pointcloud). "
            "'legacy' shows the original semantic/depth 3-column view; "
            "'planning_pointcloud' shows pointcloud reprojection triplet (+ optional panorama)."
        ),
    )
    parser.add_argument(
        "--stream_viz_pc_outdir",
        type=str,
        default="",
        help=(
            "Outdir used to locate planning pointcloud shm_info.json for debug viz. "
            "If empty, uses --planning_pointcloud_outdir (default: empty)."
        ),
    )
    parser.add_argument(
        "--stream_viz_pc_info_filename",
        type=str,
        default="shm_info.json",
        help="Planning pointcloud shared-memory info filename (default: shm_info.json).",
    )
    parser.add_argument(
        "--stream_viz_pc_radius_m",
        type=float,
        default=2.0,
        help="Panorama radius (meters) around current pose for pointcloud debug viz (default: 2.0).",
    )
    parser.add_argument(
        "--stream_viz_pc_pano",
        dest="stream_viz_pc_pano",
        action="store_true",
        default=True,
        help="Enable the panorama (second row) in pointcloud debug viz (default: enabled).",
    )
    parser.add_argument(
        "--stream_viz_pc_no_pano",
        dest="stream_viz_pc_pano",
        action="store_false",
        help="Disable the panorama (second row) in pointcloud debug viz.",
    )
    parser.add_argument(
        "--stream_viz_pc_pano_h",
        type=int,
        default=96,
        help="Panorama height in pointcloud debug viz (default: 96).",
    )
    parser.add_argument(
        "--stream_viz_pc_pano_vfov_deg",
        type=float,
        default=60.0,
        help="Panorama vertical FOV in degrees (default: 60).",
    )
    parser.add_argument(
        "--stream_viz_pc_pano_mode",
        choices=["rgb", "sem", "depth"],
        default="sem",
        help="Panorama coloring mode in pointcloud debug viz (default: sem).",
    )
    parser.add_argument(
        "--stream_viz_pc_fov_deg",
        type=float,
        default=60.0,
        help="Assumed pinhole FOV in degrees for triplet reprojection (default: 60).",
    )
    parser.add_argument(
        "--stream_viz_pc_max_points",
        type=int,
        default=10_000,
        help=(
            "Consumer-side max points for pointcloud reprojection (default: 10000). "
            "Lower values reduce CPU load/latency in debug visualization."
        ),
    )
    parser.add_argument(
        "--stream_viz_pc_esdf",
        action="store_true",
        default=False,
        help="Enable local 3D ESDF/occupancy visualization from the planning pointcloud (default: disabled).",
    )
    parser.add_argument(
        "--stream_viz_pc_esdf_radius",
        type=float,
        default=2.0,
        help="Local ESDF cube half-size in meters (default: 2.0).",
    )
    parser.add_argument(
        "--stream_viz_pc_esdf_voxel",
        type=float,
        default=0.1,
        help="ESDF voxel size in meters (default: 0.1).",
    )
    parser.add_argument(
        "--stream_viz_pc_esdf_use_semantic",
        action="store_true",
        default=False,
        help="If set, only obstacle labels contribute to occupancy when computing ESDF.",
    )
    parser.add_argument(
        "--stream_viz_pc_esdf_obstacle_labels",
        type=str,
        default="",
        help="Space/comma-separated obstacle label IDs for ESDF. Empty means all labels are obstacles.",
    )
    # -------------------------------------------------------------------------
    # Optional: Kalman smoothing for debug min-depth readouts (visualization only)
    #
    # Motivation:
    #   Even when depth is produced by kf-warp + hole-fill, per-class minimum depth can
    #   still fluctuate frame-to-frame due to:
    #     - argmin pixel hopping within a region
    #     - noisy/partial depth coverage
    #     - segmentation boundary jitter
    #
    # This filter MUST NOT affect SLAM. It only smooths the values shown in the debug window.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--stream_viz_kalman",
        action="store_true",
        default=False,
        help="Enable a simple 1D Kalman filter on displayed min-depth values (default: disabled).",
    )
    parser.add_argument(
        "--stream_viz_kalman_q",
        type=float,
        default=1e-4,
        help=(
            "Kalman process noise Q for debug depth smoothing (default: 1e-4). "
            "Smaller Q => stronger smoothing (slower to react)."
        ),
    )
    parser.add_argument(
        "--stream_viz_kalman_r",
        type=float,
        default=0.20,
        help=(
            "Kalman measurement noise R for debug depth smoothing (default: 0.20). "
            "Larger R => stronger smoothing (trust measurements less)."
        ),
    )
    # -------------------------------------------------------------------------
    # Debug visualization depth filter mode (visualization only).
    #
    # Motivation:
    #   A per-pixel temporal filter can "break" when the camera moves because the same pixel
    #   no longer corresponds to the same scene point. To debug stability in both hover and motion,
    #   we provide three modes in the debug window:
    #     - none  : show depth as-is
    #     - pixel : pixel-wise Kalman (strong smoothing, may smear under motion)
    #     - pose  : pose-aware pixel Kalman (reset smoothing state when pose changes too much)
    #
    # IMPORTANT:
    #   This affects ONLY the debug window. It must NOT affect SLAM tracking/optimization.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--stream_viz_filter_mode",
        choices=["none", "pixel", "pose"],
        default="none",
        help=(
            "Depth filtering mode in the debug window (default: none). "
            "Use 'pixel' for per-pixel Kalman smoothing, or 'pose' for pose-aware smoothing."
        ),
    )
    parser.add_argument(
        "--stream_viz_sample_grid",
        type=int,
        default=3,
        help="Grid size for depth sampling in debug window (default: 3 => 3x3 samples).",
    )
    parser.add_argument(
        "--stream_viz_sample_patch",
        type=int,
        default=3,
        help="Patch size for each sampled depth statistic (default: 3 => 3x3 mean).",
    )
    parser.add_argument(
        "--stream_viz_info_width",
        type=int,
        default=220,
        help="Width (pixels) of the debug info panel (default: 220).",
    )
    parser.add_argument(
        "--stream_viz_depth_vis_max",
        type=float,
        default=10.0,
        help="Max depth (meters) for depth colormap visualization (default: 10.0).",
    )
    parser.add_argument(
        "--stream_viz_pose_reset_trans",
        type=float,
        default=0.02,
        help="Pose-aware filter reset translation threshold in meters (default: 0.02).",
    )
    parser.add_argument(
        "--stream_viz_pose_reset_rot_deg",
        type=float,
        default=2.0,
        help="Pose-aware filter reset rotation threshold in degrees (default: 2.0).",
    )
    # -------------------------------------------------------------------------
    # Debug visualization: semantic smoothing (visualization only)
    #
    # Motivation:
    #   Even with stabilized semantics, the displayed segmentation can flicker due to:
    #     - per-frame network noise (raw)
    #     - incomplete warp coverage (stable)
    #     - noisy boundaries / small regions
    #
    # We provide an optional per-pixel hard-label EMA-like filter in the debug process.
    # It stores only (label, weight) per pixel (no logits/probabilities) and is fast.
    #
    # IMPORTANT:
    #   This affects ONLY the debug window. It must NOT affect SLAM.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--stream_viz_semantic_filter",
        choices=["none", "ema"],
        default="ema",
        help="Semantic smoothing mode in debug window (default: ema).",
    )
    parser.add_argument(
        "--stream_viz_semantic_filter_target",
        choices=["raw", "stable", "both"],
        default="raw",
        help="Which semantic panel(s) to smooth in debug window (default: raw).",
    )
    parser.add_argument(
        "--stream_viz_semantic_momentum",
        type=float,
        default=0.98,
        help="Semantic EMA momentum mu in debug window (default: 0.98).",
    )
    parser.add_argument(
        "--stream_viz_semantic_u",
        type=float,
        default=1.0,
        help="Semantic EMA update magnitude u in debug window (default: 1.0).",
    )

    # -------------------------------------------------------------------------
    # Debug visualization: depth-guided semantic refinement (visualization only)
    #
    # Motivation:
    #   Even after temporal smoothing (EMA) and even after SLAM-based warp stabilization,
    #   the displayed segmentation can still show spatial flicker along surfaces.
    #
    # This optional refinement uses the *filtered depth map shown in the GUI* to perform a
    # tiny CRF-like post-processing step that:
    #   - encourages label agreement between pixels with similar depth
    #   - avoids propagating labels across large depth discontinuities (depth edges)
    #
    # IMPORTANT:
    #   - Default OFF: this is a visualization heuristic, not part of the SLAM pipeline.
    #   - It must NOT affect any SLAM gating/optimization decisions.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--stream_viz_semantic_depth_refine",
        dest="stream_viz_semantic_depth_refine",
        action="store_true",
        default=False,
        help="Enable depth-guided semantic refinement in debug window (default: disabled).",
    )
    parser.add_argument(
        "--stream_viz_no_semantic_depth_refine",
        dest="stream_viz_semantic_depth_refine",
        action="store_false",
        help="Disable depth-guided semantic refinement in debug window.",
    )
    parser.add_argument(
        "--stream_viz_semantic_depth_refine_target",
        choices=["raw", "stable", "both"],
        default="stable",
        help="Which semantic panel(s) to refine using depth (default: stable).",
    )
    parser.add_argument(
        "--stream_viz_semantic_depth_refine_iters",
        type=int,
        default=1,
        help="Number of depth-guided refinement iterations (default: 1).",
    )
    parser.add_argument(
        "--stream_viz_semantic_depth_sigma",
        type=float,
        default=0.50,
        help="Depth similarity bandwidth sigma in meters for refinement (default: 0.50).",
    )
    parser.add_argument(
        "--stream_viz_stable_semantic_source",
        choices=["stable", "raw"],
        default="raw",
        help="Source of the middle semantic panel in debug window (default: raw).",
    )

    # -------------------------------------------------------------------------
    # Semantic input / output controls
    #
    # Motivation:
    #   - Even if (A) semantic warp and (B) semantic bonus are disabled, running this
    #     entrypoint (`main_semantic.py`) could still spend time producing per-frame
    #     segmentation (EfficientViT) via the dataloader.
    #   - This switch allows a true runtime baseline (close to `main.py`) while keeping
    #     the same executable/script.
    #
    # Constraints:
    #   - Disabling semantic input implies there is no segmentation to stabilize, so (A)
    #     has no effect and (B) will also be inactive in practice.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--enable_semantic_input",
        dest="enable_semantic_input",
        action="store_true",
        default=True,
        help="Enable reading/running per-frame segmentation from the dataset (default: enabled).",
    )
    parser.add_argument(
        "--disable_semantic_input",
        dest="enable_semantic_input",
        action="store_false",
        help="Disable per-frame segmentation input (runtime baseline closer to main.py).",
    )

    # -------------------------------------------------------------------------
    # Planning-oriented semantic point cloud visualization (CPU, OpenCV)
    #
    # Motivation:
    #   Downstream planning/control often wants a semantic "map view" that is derived from SLAM
    #   keyframes. This utility runs in a separate process and:
    #     - gathers keyframe pointmaps (downsampled)
    #     - colors them by semantic (legacy RGB semantic mask, no voting by default)
    #     - projects them into the current camera image using the current pose
    #     - optionally shows a window and saves overlay frames to disk
    #
    # IMPORTANT:
    #   This is visualization-only and must not block SLAM. Keep it optional.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--enable_planning_pointcloud_viz",
        action="store_true",
        default=False,
        help="Enable a separate CPU process that saves semantic pointcloud projection overlays (default: disabled).",
    )
    parser.add_argument(
        "--enable_planning_pointcloud_publish",
        action="store_true",
        default=False,
        help=(
            "Enable a separate CPU process that publishes the latest semantic point cloud + pose for downstream planning "
            "(no window, no image saving; default: disabled)."
        ),
    )
    parser.add_argument(
        "--planning_pointcloud_outdir",
        type=str,
        default="logs/planning_pointcloud",
        help="Output directory for saved overlay frames (default: logs/planning_pointcloud).",
    )
    parser.add_argument(
        "--planning_pointcloud_fps",
        type=float,
        default=5.0,
        help="Target FPS for the planning pointcloud process (default: 5).",
    )
    parser.add_argument(
        "--planning_pointcloud_max_keyframes",
        type=int,
        default=30,
        help="Max number of most-recent keyframes to include (default: 30).",
    )
    parser.add_argument(
        "--planning_pointcloud_stride",
        type=int,
        default=4,
        help="Downsample stride on the keyframe pointmap grid (default: 4).",
    )
    parser.add_argument(
        "--planning_pointcloud_conf_threshold",
        type=float,
        default=1.5,
        help="Minimum average confidence to keep a point (default: 1.5).",
    )
    parser.add_argument(
        "--planning_pointcloud_headless",
        action="store_true",
        default=False,
        help="Do not open an OpenCV window; still save frames to disk (default: disabled).",
    )
    parser.add_argument(
        "--planning_pointcloud_no_save_images",
        action="store_true",
        default=False,
        help="Do not save overlay PNG frames to disk (default: disabled).",
    )
    parser.add_argument(
        "--planning_pointcloud_publish_shm",
        action="store_true",
        default=False,
        help="Publish the latest semantic point cloud + pose via shared memory for downstream planning (default: disabled).",
    )
    parser.add_argument(
        "--planning_pointcloud_shm_info",
        type=str,
        default="",
        help="Path to write shared-memory info JSON. If empty, defaults to <planning_pointcloud_outdir>/shm_info.json.",
    )
    parser.add_argument(
        "--planning_pointcloud_shm_max_points",
        type=int,
        default=200_000,
        help="Maximum number of points published to shared memory (default: 200000).",
    )
    parser.add_argument(
        "--planning_pointcloud_shm_max_keyframes",
        type=int,
        default=1024,
        help=(
            "Shared-memory capacity (max keyframes). Used by the keyframe-centric publisher schema "
            "(default: 1024)."
        ),
    )
    parser.add_argument(
        "--planning_pointcloud_shm_points_per_kf",
        type=int,
        default=8192,
        help=(
            "Shared-memory capacity (points per keyframe). Used by the keyframe-centric publisher schema "
            "(default: 8192)."
        ),
    )
    # -------------------------------------------------------------------------
    # Optional: Kalman smoothing on the PUBLISHED pose (planning/debug only).
    #
    # Motivation:
    #   Downstream planning/control can be sensitive to pose jitter. We provide an optional
    #   smoother that runs inside the planning pointcloud publisher process and only changes
    #   the `T_WC` that is published/visualized there.
    #
    # Design:
    #   - 0.0 disables filtering.
    #   - Larger values => stronger smoothing (trust measurements less).
    #
    # IMPORTANT:
    #   This must NEVER affect SLAM tracking/optimization. It is output-only.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--planning_pointcloud_pose_kalman_pos",
        type=float,
        default=0.0,
        help="Kalman smoothing strength for published pose translation (0 disables; larger => smoother).",
    )
    parser.add_argument(
        "--planning_pointcloud_pose_kalman_rot",
        type=float,
        default=0.0,
        help="Kalman smoothing strength for published pose rotation (0 disables; larger => smoother).",
    )

    # -------------------------------------------------------------------------
    # Planning TSDF(+semantic) publish (rolling local volume)
    #
    # Motivation:
    #   A point cloud is often inconvenient for CBF/control. A rolling TSDF volume provides:
    #     - a smooth implicit surface (TSDF zero level-set)
    #     - a natural carrier for semantic fusion (voxel-level voting)
    #
    # Design:
    #   - Runs in the SAME auxiliary process as `--enable_planning_pointcloud_publish` (CPU-side).
    #   - Uses per-frame depth from `frame.X_canon[...,2]` (current frame pointmap Z).
    #   - Optionally fuses per-frame semantic labels into voxels (hard label + weight).
    #
    # IMPORTANT:
    #   This is output-only. It MUST NOT affect SLAM.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--enable_planning_tsdf_publish",
        action="store_true",
        default=False,
        help="Publish a rolling TSDF(+semantic) volume via shared memory (default: disabled).",
    )
    parser.add_argument(
        "--planning_tsdf_shm_info",
        type=str,
        default="",
        help="Path to write TSDF shared-memory info JSON. If empty, defaults to <planning_pointcloud_outdir>/tsdf_shm_info.json.",
    )
    parser.add_argument(
        "--planning_tsdf_radius_m",
        type=float,
        default=2.0,
        help="Rolling TSDF radius in meters (default: 2.0).",
    )
    parser.add_argument(
        "--planning_tsdf_voxel_m",
        type=float,
        default=0.1,
        help="Rolling TSDF voxel size in meters (default: 0.1).",
    )
    parser.add_argument(
        "--planning_tsdf_trunc_m",
        type=float,
        default=0.0,
        help="TSDF truncation distance in meters (<=0 uses 3*voxel; default: 0).",
    )
    parser.add_argument(
        "--planning_tsdf_max_weight",
        type=float,
        default=100.0,
        help="Maximum TSDF integration weight per voxel (default: 100).",
    )
    parser.add_argument(
        "--planning_tsdf_use_semantic",
        dest="planning_tsdf_use_semantic",
        action="store_true",
        default=True,
        help="Fuse semantic hard labels into TSDF voxels (default: enabled).",
    )
    parser.add_argument(
        "--planning_tsdf_no_semantic",
        dest="planning_tsdf_use_semantic",
        action="store_false",
        help="Disable semantic fusion in TSDF voxels (geometry only).",
    )
    parser.add_argument(
        "--planning_tsdf_semantic_band_m",
        type=float,
        default=0.0,
        help="Semantic update band around surface in meters (<=0 uses voxel size; default: 0).",
    )
    parser.add_argument(
        "--planning_tsdf_frame_sem_dir",
        type=str,
        default="",
        help=(
            "Optional directory containing per-frame semantic label files to be fused into TSDF. "
            "When set, TSDF semantic fusion will prefer these labels over EfficientViT outputs. "
            "Supported formats: .npy (recommended; stores label-id map), and single-channel .png. "
            "The file name is formed by --planning_tsdf_frame_sem_pattern with {frame_id}."
        ),
    )
    parser.add_argument(
        "--planning_tsdf_frame_sem_pattern",
        type=str,
        default="{frame_id:06d}.npy",
        help=(
            "Filename pattern for per-frame semantic label files inside --planning_tsdf_frame_sem_dir. "
            "Python format string with {frame_id}. Example: '{frame_id:06d}.npy' for 000123.npy."
        ),
    )
    parser.add_argument(
        "--planning_tsdf_pose_json",
        type=str,
        default="",
        help=(
            "Optional pose_intrinsic_imu.json used only by the planning TSDF/ESDF publisher. "
            "When set, TSDF integration uses these camera-to-world poses instead of MASt3R poses; "
            "SLAM tracking is unchanged."
        ),
    )
    parser.add_argument(
        "--planning_tsdf_pose_key",
        type=str,
        default="aligned_pose",
        help="Pose matrix key inside --planning_tsdf_pose_json entries (default: aligned_pose).",
    )
    parser.add_argument(
        "--planning_tsdf_pose_frame_stride",
        type=int,
        default=1,
        help=(
            "Map processed frame_id to pose-json source frame_id by multiplication. "
            "For config/base_subsample10.yaml on ScanNet++, use 10."
        ),
    )
    parser.add_argument(
        "--planning_tsdf_pose_frame_pattern",
        type=str,
        default="frame_{frame_id:06d}",
        help="Pose-json frame key pattern after stride mapping (default: frame_{frame_id:06d}).",
    )
    parser.add_argument(
        "--external_tracking_pose_json",
        type=str,
        default="",
        help=(
            "Optional pose_intrinsic_imu.json used to override SLAM frame.T_WC after MASt3R "
            "depth/pointmap inference. This changes tracking/map poses, but does not replace "
            "MASt3R/DAPS depth."
        ),
    )
    parser.add_argument(
        "--external_tracking_pose_key",
        type=str,
        default="aligned_pose",
        help="Pose matrix key inside --external_tracking_pose_json entries (default: aligned_pose).",
    )
    parser.add_argument(
        "--external_tracking_pose_frame_stride",
        type=int,
        default=1,
        help=(
            "Map processed frame_id to pose-json source frame_id by multiplication. "
            "For config/base_subsample10.yaml on ScanNet++, use 10."
        ),
    )
    parser.add_argument(
        "--external_tracking_pose_frame_pattern",
        type=str,
        default="frame_{frame_id:06d}",
        help="Pose-json frame key pattern after stride mapping (default: frame_{frame_id:06d}).",
    )
    parser.add_argument(
        "--external_tracking_pose_space",
        choices=["first-frame", "aligned-world"],
        default="first-frame",
        help=(
            "Coordinate space for overridden frame.T_WC. Use first-frame to preserve the original "
            "ScanNet++ ESDF/CBF protocol; aligned-world is mainly for debugging."
        ),
    )
    parser.add_argument(
        "--planning_tsdf_backend",
        choices=["numpy", "torch"],
        default="numpy",
        help="TSDF fusion backend in planning process (default: numpy).",
    )
    parser.add_argument(
        "--planning_tsdf_torch_device",
        type=str,
        default="cuda",
        help="Torch device for TSDF fusion when --planning_tsdf_backend=torch (default: cuda).",
    )
    parser.add_argument(
        "--planning_tsdf_torch_dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Torch dtype for TSDF/weights when --planning_tsdf_backend=torch (default: float32).",
    )

    # -------------------------------------------------------------------------
    # Optional: spawn the pointcloud trajectory monitor (debug/validation only).
    #
    # Motivation:
    #   Some users report the trajectory visualization in
    #   `scripts/monitor_pointcloud_semantic_control.py --show_traj` looks incorrect.
    #   To reproduce/validate this quickly, we can spawn that script automatically in the
    #   same run, right after the planning pointcloud publisher starts.
    #
    # IMPORTANT:
    #   - This is NOT used by SLAM and does NOT affect any SLAM estimate.
    #   - The monitor runs in a separate OS process, so it cannot block the SLAM loop.
    #   - We run it with `--no_control` to avoid sending any commands to a robot/bridge.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--enable_monitor_pointcloud_traj",
        action="store_true",
        default=False,
        help="Spawn scripts/monitor_pointcloud_semantic_control.py with --show_traj (default: disabled).",
    )
    parser.add_argument(
        "--monitor_pointcloud_traj_hz",
        type=float,
        default=5.0,
        help="Polling rate for the trajectory monitor (default: 5Hz).",
    )
    parser.add_argument(
        "--monitor_pointcloud_traj_size",
        type=int,
        default=600,
        help="Trajectory window size in pixels (default: 600).",
    )
    parser.add_argument(
        "--monitor_pointcloud_fov_deg",
        type=float,
        default=60.0,
        help="FOV passed to the monitor script (default: 60).",
    )

    # -------------------------------------------------------------------------
    # Optional: save stabilized semantic masks per frame for later visualization.
    #
    # Design:
    #   - We save the *post-stabilization* semantic (i.e., after tracking computed matches
    #     and (A) performed warp+fuse) as a PNG image per frame.
    #   - Video encoding is intentionally not done here (user can compress later).
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--save_stable_semantic",
        action="store_true",
        default=False,
        help="Save per-frame stabilized semantic mask images to disk (default: disabled).",
    )
    parser.add_argument(
        "--stable_semantic_dir",
        default="",
        help=(
            "Output directory for stabilized semantic frames. "
            "If empty, defaults to logs/stable_semantic/<save-as>/."
        ),
    )
    parser.add_argument(
        "--save_raw_semantic",
        action="store_true",
        default=False,
        help="Save per-frame raw/pre-fusion semantic mask images to disk (default: disabled).",
    )
    parser.add_argument(
        "--raw_semantic_dir",
        default="",
        help=(
            "Output directory for raw/pre-fusion semantic frames. "
            "If empty, defaults to logs/raw_semantic/<save-as>/."
        ),
    )
    parser.add_argument(
        "--save_semantic_rgb",
        action="store_true",
        default=False,
        help="Save SLAM-aligned RGB frames for semantic overlay videos (default: disabled).",
    )
    parser.add_argument(
        "--semantic_rgb_dir",
        default="",
        help=(
            "Output directory for SLAM-aligned RGB frames. "
            "If empty, defaults to logs/semantic_rgb/<save-as>/."
        ),
    )

    # -------------------------------------------------------------------------
    # Semantic stabilization / geometry bonus ablations
    #
    # Design constraints (as requested):
    #   - No matching logic changes.
    #   - No geometry<->semantic iterative loop (single pass per frame).
    #   - Semantic warp is O(N) with vectorized overwrite (no per-frame sorting).
    #   - Semantic in geometry uses a symmetric (1 +/- beta) factor (V1.5) and MUST NOT change gating.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--enable_semantic_warp",
        dest="enable_semantic_warp",
        action="store_true",
        default=True,
        help="Enable (A) semantic warp+fuse using k->f matches (default: enabled).",
    )
    parser.add_argument(
        "--disable_semantic_warp",
        dest="enable_semantic_warp",
        action="store_false",
        help="Disable (A) semantic warp+fuse (ablation).",
    )
    parser.add_argument(
        "--use_semantic_in_geo",
        dest="use_semantic_in_geo",
        action="store_true",
        default=False,
        help="Enable (B) semantic bonus in tracking geometry optimization (default: disabled).",
    )
    parser.add_argument(
        "--disable_semantic_in_geo",
        dest="use_semantic_in_geo",
        action="store_false",
        help="Disable (B) semantic bonus in geometry optimization (ablation).",
    )
    parser.add_argument(
        "--use_stable_semantic_in_geo",
        dest="use_stable_semantic_in_geo",
        action="store_true",
        default=False,
        help=(
            "When (B) is enabled, use stabilized per-frame semantic label for reweighting "
            "(default: False, i.e., use raw per-frame semantic as an independent observation)."
        ),
    )
    parser.add_argument(
        "--use_raw_semantic_in_geo",
        dest="use_stable_semantic_in_geo",
        action="store_false",
        help=(
            "When (B) is enabled, use raw per-frame semantic label for reweighting "
            "(this is the default and recommended to avoid 'self-fulfilling' consistency "
            "when A-warp overwrites the frame label)."
        ),
    )
    parser.add_argument(
        "--semantic_beta",
        type=float,
        default=0.2,
        help=(
            "Semantic strength beta for (B) (V1.5 symmetric): "
            "w' = w * ((1+beta) if same else (1-beta)) (default: 0.2)."
        ),
    )
    parser.add_argument(
        "--semantic_tau_warp",
        type=float,
        default=None,
        help=(
            "Confidence threshold tau_warp for (A) overwrite. "
            "If not set, defaults to tracking.Q_conf from the YAML config."
        ),
    )
    parser.add_argument(
        "--semantic_tau",
        type=float,
        default=None,
        help=(
            "Optional confidence threshold tau_sem for applying the (B) bonus only when q > tau_sem. "
            "Default: None (no extra gate)."
        ),
    )

    # -------------------------------------------------------------------------
    # V3: Semantic PointMap (keyframe semantic fusion cache)
    #
    # Goal:
    #   Make semantics "more stable over time" by fusing per-frame hard-label observations
    #   back into the active keyframe, similar in spirit to how the geometric pointmap is fused.
    #
    # Design constraints:
    #   - Hard labels only (no logits/probabilities stored in the keyframe).
    #   - O(N) vectorized update (no sorting, no Python for-loops over pixels).
    #   - When disabled, the existing V1 semantic pipeline (warp+fuse, saving, etc.) is unchanged.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--enable_semantic_pointmap",
        dest="enable_semantic_pointmap",
        action="store_true",
        default=True,
        help="Enable V3 semantic pointmap fusion cache on keyframes (default: enabled).",
    )
    parser.add_argument(
        "--disable_semantic_pointmap",
        dest="enable_semantic_pointmap",
        action="store_false",
        help="Disable V3 semantic pointmap fusion cache on keyframes (ablation).",
    )
    parser.add_argument(
        "--semantic_pointmap_init_weight",
        type=float,
        default=1.0,
        help="V3 initial per-pixel semantic weight for a new keyframe (default: 1.0).",
    )
    parser.add_argument(
        "--semantic_pointmap_momentum",
        type=float,
        default=1.0,
        help=(
            "V3 per-update momentum/decay mu for keyframe semantic weights (default: 1.0, i.e., no decay). "
            "When mu<1, we apply w[k_idx] *= mu only for pixels touched by valid matches in the current frame."
        ),
    )
    parser.add_argument(
        "--semantic_pointmap_use_q",
        dest="semantic_pointmap_use_q",
        action="store_true",
        default=True,
        help="V3 use match confidence q as the update magnitude u (default: enabled).",
    )
    parser.add_argument(
        "--semantic_pointmap_no_q",
        dest="semantic_pointmap_use_q",
        action="store_false",
        help="V3 do NOT use q (use constant u=1.0) for semantic pointmap updates (ablation).",
    )

    # -------------------------------------------------------------------------
    # Debug: semantic geometry factor statistics
    #
    # Motivation:
    #   When users report "no performance difference", the first question is whether the
    #   semantic factor is actually non-trivial (i.e., not all-ones and not near-constant).
    #
    # What we print (when enabled):
    #   - mean/std/min/max of the per-match sqrt factor used in tracking
    #   - applied_ratio: fraction of matches where the semantic factor is applied
    #
    # Interpretation:
    #   - std == 0 and mean == 1   -> semantic factor not applied (effectively disabled)
    #   - std ~= 0 and mean != 1   -> near-constant scaling (often no change in argmin)
    #   - std > 0                 -> non-trivial reweighting is happening
    #
    # Design constraint:
    #   This is pure diagnostics and MUST NOT change match gating/valid logic.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--debug_semantic_geo_stats",
        action="store_true",
        default=False,
        help="Print semantic geometry factor statistics (default: disabled).",
    )
    parser.add_argument(
        "--debug_semantic_geo_every",
        type=int,
        default=30,
        help="When debug is enabled, print stats every N frames (default: 30).",
    )
    parser.add_argument(
        "--debug_semantic_geo_first",
        type=int,
        default=5,
        help="When debug is enabled, also print stats for the first N frames (default: 5).",
    )

    # -------------------------------------------------------------------------
    # Per-frame depth + semantic statistics for downstream planning (very lightweight)
    #
    # Goal:
    #   Provide a runnable path to compute, for each semantic category, the minimum depth
    #   observed in the current frame. This is often useful for quick obstacle proximity
    #   signals in planning.
    #
    # Notes:
    #   - Depth source is configurable:
    #       * raw_z     : per-frame pointmap Z (fastest, can jitter)
    #       * kf_warp_z : keyframe-warp Z + fast hole-fill (more stable)
    #   - We print stats (optionally) and also provide a one-line "hook" you can uncomment
    #     to pass (label, depth) tensors into your own planner.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--depth_source",
        choices=["raw_z", "kf_warp_z"],
        default="raw_z",
        help="Depth source for per-frame outputs: raw_z or kf_warp_z (default: raw_z).",
    )
    parser.add_argument(
        "--depth_tau",
        type=float,
        default=None,
        help=(
            "Confidence threshold for kf_warp_z depth writes. "
            "If not set, defaults to tracking.Q_conf from the YAML config."
        ),
    )
    parser.add_argument(
        "--depth_fill_iters",
        type=int,
        default=2,
        help="Number of neighbor-propagation iterations for kf_warp_z hole filling (default: 2).",
    )
    parser.add_argument(
        "--depth_fill_kernel",
        type=int,
        default=3,
        help="Kernel size for kf_warp_z hole filling (odd integer, default: 3).",
    )
    parser.add_argument(
        "--print_min_depth_per_class",
        action="store_true",
        default=False,
        help="Print per-frame min depth per semantic class (default: disabled).",
    )
    parser.add_argument(
        "--print_min_depth_every",
        type=int,
        default=30,
        help="When printing is enabled, print every N frames (default: 30).",
    )
    parser.add_argument(
        "--semantic_num_classes",
        type=int,
        default=0,
        help=(
            "Number of semantic classes for fast per-class reduction. "
            "If 0, we auto-set to max(label)+1 (may be large if labels are RGB codes)."
        ),
    )

    args = parser.parse_args()

    load_config(args.config)
    print(args.dataset)
    print(config)

    # -------------------------------------------------------------------------
    # Inject semantic ablation switches into the global config dict.
    #
    # Note:
    #   - The tracker reads these values from `config.get('semantic', {})`.
    #   - We set them here (after YAML load) so that CLI flags override YAML defaults.
    # -------------------------------------------------------------------------
    config.setdefault("semantic", {})
    config["semantic"]["enable_semantic_warp"] = bool(args.enable_semantic_warp)
    config["semantic"]["use_semantic_in_geo"] = bool(args.use_semantic_in_geo)
    config["semantic"]["use_stable_semantic_in_geo"] = bool(args.use_stable_semantic_in_geo)
    config["semantic"]["semantic_beta"] = float(args.semantic_beta)
    config["semantic"]["semantic_tau_warp"] = (
        float(args.semantic_tau_warp)
        if args.semantic_tau_warp is not None
        else float(config["tracking"]["Q_conf"])
    )
    config["semantic"]["semantic_tau"] = (
        float(args.semantic_tau) if args.semantic_tau is not None else None
    )
    config["semantic"]["enable_semantic_pointmap"] = bool(args.enable_semantic_pointmap)
    config["semantic"]["semantic_pointmap_init_weight"] = float(args.semantic_pointmap_init_weight)
    config["semantic"]["semantic_pointmap_momentum"] = float(args.semantic_pointmap_momentum)
    config["semantic"]["semantic_pointmap_use_q"] = bool(args.semantic_pointmap_use_q)
    config["semantic"]["debug_semantic_geo_stats"] = bool(args.debug_semantic_geo_stats)
    config["semantic"]["debug_semantic_geo_every"] = int(args.debug_semantic_geo_every)
    config["semantic"]["debug_semantic_geo_first"] = int(args.debug_semantic_geo_first)
    # EfficientViT segmentation taxonomy selection (ADE20K vs Cityscapes).
    # This is consumed by:
    #   - async segmenter worker (`LatestOnlyAsyncSegmenter`)
    #   - sync segmentation calls (when async is disabled)
    #   - dataset-side online segmentation fallback (if used)
    config["semantic"]["efficientvit_dataset"] = str(args.efficientvit_dataset)

    # Depth readout config (for downstream planning/statistics only; does not affect SLAM).
    config.setdefault("depth", {})
    config["depth"]["depth_source"] = str(args.depth_source)
    config["depth"]["depth_tau"] = (
        float(args.depth_tau) if args.depth_tau is not None else float(config["tracking"]["Q_conf"])
    )
    config["depth"]["depth_fill_iters"] = int(args.depth_fill_iters)
    config["depth"]["depth_fill_kernel"] = int(args.depth_fill_kernel)

    # Prepare stable semantic output directory (if enabled).
    stable_semantic_dir = None
    if args.save_stable_semantic:
        stable_semantic_dir = (
            pathlib.Path(args.stable_semantic_dir)
            if args.stable_semantic_dir
            else pathlib.Path("logs") / "stable_semantic" / str(args.save_as)
        )
        stable_semantic_dir.mkdir(parents=True, exist_ok=True)
    raw_semantic_dir = None
    if args.save_raw_semantic:
        raw_semantic_dir = (
            pathlib.Path(args.raw_semantic_dir)
            if args.raw_semantic_dir
            else pathlib.Path("logs") / "raw_semantic" / str(args.save_as)
        )
        raw_semantic_dir.mkdir(parents=True, exist_ok=True)
    semantic_rgb_dir = None
    if args.save_semantic_rgb:
        semantic_rgb_dir = (
            pathlib.Path(args.semantic_rgb_dir)
            if args.semantic_rgb_dir
            else pathlib.Path("logs") / "semantic_rgb" / str(args.save_as)
        )
        semantic_rgb_dir.mkdir(parents=True, exist_ok=True)

    viz_cfg = config.get("visualization", {})
    save_depth_images = viz_cfg.get("save_depth_images", False)
    depth_save_dir = pathlib.Path(viz_cfg.get("depth_save_dir", "logs/depth"))
    if save_depth_images:
        depth_save_dir.mkdir(parents=True, exist_ok=True)
    else:
        depth_save_dir = None

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    # -------------------------------------------------------------------------
    # Input source initialization
    #
    # Dataset mode:
    #   - Existing behavior: load dataset from disk, optionally load semantic from disk or
    #     run online segmentation inside the dataloader.
    #
    # Streaming (AirSim) mode:
    #   - Create an asynchronous AirSim frame grabber (latest-only).
    #   - Optionally run EfficientViT segmentation asynchronously to avoid slowing SLAM.
    #   - We set the "network input size" to `--stream_img_size` (default 224), which matches
    #     the rest of the codebase assumptions (`resize_img` supports 224/512).
    # -------------------------------------------------------------------------
    dataset = None
    stream_grabber = None
    stream_segmenter = None
    stream_teleop = None
    stream_last_label = None  # cached last available label map (numpy HxW int64)
    dataset_async_semantic = False  # dataset video replay can optionally reuse async segmentation path

    stream_viz_queue = None
    stream_viz_proc = None
    stream_intrinsics = None
    stream_debug_shm = None

    if args.input_source == "dataset":
        dataset = load_dataset(args.dataset)
        dataset.img_size = int(args.stream_img_size)
        dataset.subsample(config["dataset"]["subsample"])
        h, w = dataset.get_img_shape()[0]
        img_size = int(dataset.img_size)

        if args.calib:
            with open(args.calib, "r") as f:
                intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
            config["use_calib"] = True
            dataset.use_calibration = True
            dataset.camera_intrinsics = Intrinsics.from_calib(
                dataset.img_size,
                intrinsics["width"],
                intrinsics["height"],
                intrinsics["calibration"],
            )

        # ---------------------------------------------------------------------
        # IMPORTANT (MP4 replay + real-time safety):
        #
        # `MonocularDataset.__getitem__()` has an "online segmentation" fallback. For MP4 input
        # (and other sources that have no semantic masks on disk), this would run EfficientViT
        # synchronously inside the dataloader for *every* frame.
        #
        # That behavior is often undesirable for two reasons:
        #   1) It blocks the SLAM loop (latency spikes / FPS drop).
        #   2) It can contend with MASt3R (and optional visualization) on the same GPU, which can
        #      destabilize tracking and trigger "Failed to relocalize" cascades.
        #
        # Therefore, for MP4 replay we *optionally* reuse the same async segmenter used in
        # streaming mode (latest-only, capped FPS). This keeps behavior close to online.
        #
        # Note:
        #   - This is only activated for MP4Dataset to minimize impact on existing dataset flows
        #     where semantic masks may already exist on disk.
        # ---------------------------------------------------------------------
        dataset_is_mp4 = dataset.__class__.__name__ == "MP4Dataset"
        dataset_async_semantic = bool(dataset_is_mp4 and args.enable_semantic_input and args.stream_async_semantic)
        if dataset_async_semantic:
            from mast3r_slam.streaming.async_segmenter import LatestOnlyAsyncSegmenter

            stream_segmenter = LatestOnlyAsyncSegmenter(
                target_fps=int(args.stream_semantic_fps),
                dataset=str(args.efficientvit_dataset),
            )
            stream_segmenter.start()

        # Optional debug visualization also works in dataset mode (e.g., MP4 replay).
        #
        # Headless mode:
        #   If `--stream_viz_headless` is set, we do NOT spawn any OpenCV window process.
        #   Instead, we publish the latest debug payload to shared memory so other processes
        #   can attach without affecting SLAM timing.
        if args.enable_stream_debug_viz and (not bool(args.stream_viz_headless)):
            from mast3r_slam.streaming.debug_viz import run_stream_debug_viz

            # Backward compatibility:
            #   - Historically, `--stream_viz_kalman` toggled a single smoothing implementation.
            #   - We now support `--stream_viz_filter_mode {none,pixel,pose}`.
            # If the old flag is enabled but the new mode is left at default, treat it as "pixel".
            viz_filter_mode = str(args.stream_viz_filter_mode)
            if bool(args.stream_viz_kalman) and viz_filter_mode == "none":
                viz_filter_mode = "pixel"

            stream_viz_queue = mp.Queue(maxsize=1)
            stream_viz_proc = mp.Process(
                target=run_stream_debug_viz,
                args=(stream_viz_queue,),
                kwargs=dict(
                    overlay_alpha=float(args.stream_viz_alpha),
                    scale=int(args.stream_viz_scale),
                    filter_mode=str(viz_filter_mode),
                    depth_source=str(args.stream_viz_depth_source),
                    enable_kalman=bool(args.stream_viz_kalman),
                    kalman_q=float(args.stream_viz_kalman_q),
                    kalman_r=float(args.stream_viz_kalman_r),
                    sample_grid=int(args.stream_viz_sample_grid),
                    sample_patch=int(args.stream_viz_sample_patch),
                    info_width=int(args.stream_viz_info_width),
                    depth_vis_max_m=float(args.stream_viz_depth_vis_max),
                    pose_reset_trans_m=float(args.stream_viz_pose_reset_trans),
                    pose_reset_rot_deg=float(args.stream_viz_pose_reset_rot_deg),
                    semantic_filter_mode=str(args.stream_viz_semantic_filter),
                    semantic_filter_target=str(args.stream_viz_semantic_filter_target),
                    semantic_filter_momentum=float(args.stream_viz_semantic_momentum),
                    semantic_filter_u=float(args.stream_viz_semantic_u),
                    semantic_depth_refine=bool(args.stream_viz_semantic_depth_refine),
                    semantic_depth_refine_target=str(args.stream_viz_semantic_depth_refine_target),
                    semantic_depth_refine_iters=int(args.stream_viz_semantic_depth_refine_iters),
                    semantic_depth_sigma_m=float(args.stream_viz_semantic_depth_sigma),
                    stable_semantic_source=str(args.stream_viz_stable_semantic_source),
                    semantic_topk=int(args.stream_viz_topk),
                    viz_layout=str(args.stream_viz_layout),
                    planning_pointcloud_outdir=(
                        str(args.stream_viz_pc_outdir)
                        if str(args.stream_viz_pc_outdir)
                        else str(args.planning_pointcloud_outdir)
                    ),
                    planning_pointcloud_info_filename=str(args.stream_viz_pc_info_filename),
                    planning_pointcloud_enable_pano=bool(args.stream_viz_pc_pano),
                    planning_pointcloud_radius_m=float(args.stream_viz_pc_radius_m),
                    planning_pointcloud_pano_h=int(args.stream_viz_pc_pano_h),
                    planning_pointcloud_pano_vfov_deg=float(args.stream_viz_pc_pano_vfov_deg),
                    planning_pointcloud_pano_mode=str(args.stream_viz_pc_pano_mode),
                    planning_pointcloud_fov_deg=float(args.stream_viz_pc_fov_deg),
                    planning_pointcloud_max_points=int(args.stream_viz_pc_max_points),
                    planning_pointcloud_esdf_enable=bool(args.stream_viz_pc_esdf),
                    planning_pointcloud_esdf_radius_m=float(args.stream_viz_pc_esdf_radius),
                    planning_pointcloud_esdf_voxel_m=float(args.stream_viz_pc_esdf_voxel),
                    planning_pointcloud_esdf_use_semantic=bool(args.stream_viz_pc_esdf_use_semantic),
                    planning_pointcloud_esdf_obstacle_labels=str(args.stream_viz_pc_esdf_obstacle_labels),
                ),
                daemon=True,
            )
            stream_viz_proc.start()
        elif args.enable_stream_debug_viz and bool(args.stream_viz_headless):
            from mast3r_slam.streaming.shared_debug_buffer import LatestSharedDebugBuffer

            # Shared memory info file is intentionally a stable, known path.
            # A consumer can read this json to discover shared memory names + shape.
            stream_debug_shm = LatestSharedDebugBuffer(
                prefix="mast3r_stream_dbg",
                info_path="logs/stream_debug/shm_info.json",
            )
            print("[StreamDebug] headless enabled (no window). Shared-memory info: logs/stream_debug/shm_info.json")
    else:
        # Streaming mode (no dataset on disk): AirSim or LIMO ZMQ.
        img_size = int(args.stream_img_size)
        h, w = img_size, img_size

        from mast3r_slam.streaming.async_segmenter import LatestOnlyAsyncSegmenter
        from mast3r_slam.streaming.debug_viz import run_stream_debug_viz

        if str(args.input_source) == "airsim":
            from mast3r_slam.streaming.airsim_source import AirSimRGBAsync

            stream_grabber = AirSimRGBAsync(
                ip=args.airsim_ip,
                camera=args.airsim_camera,
                vehicle_name=args.airsim_vehicle_name,
                target_fps=args.airsim_target_fps,
                image_data_order=str(args.airsim_image_order),
            )
            stream_grabber.start()
        elif str(args.input_source) == "limo_zmq":
            # NOTE:
            #   This is intentionally lazy-imported so users without ZMQ/OpenCV can still run
            #   dataset/AirSim modes without installing extra packages.
            from mast3r_slam.streaming.zmq_source import ZmqJoystickTeleop, ZmqRGBAsync

            stream_grabber = ZmqRGBAsync(ip=str(args.bridge_ip), vid_port=int(args.bridge_vid_port))
            stream_grabber.start()

            # Optional joystick teleop runs in a background thread and publishes commands to the bridge.
            if bool(args.enable_joystick):
                stream_teleop = ZmqJoystickTeleop(
                    ip=str(args.bridge_ip),
                    cmd_port=int(args.bridge_cmd_port),
                    max_linear=float(args.max_linear),
                    max_angular=float(args.max_angular),
                    deadzone=float(args.deadzone),
                    lin_axis=int(args.joystick_lin_axis),
                    ang_axis=int(args.joystick_ang_axis),
                    rate_hz=float(args.joystick_rate_hz),
                )
                stream_teleop.start()
        else:
            raise ValueError(f"Unsupported input_source for streaming: {args.input_source!r}")

        # Optional async segmentation: runs at capped FPS and never blocks tracking.
        if args.enable_semantic_input and args.stream_async_semantic:
            stream_segmenter = LatestOnlyAsyncSegmenter(
                target_fps=int(args.stream_semantic_fps),
                dataset=str(args.efficientvit_dataset),
            )
            stream_segmenter.start()

        # Optional debug visualization in a separate process.
        if args.enable_stream_debug_viz and (not bool(args.stream_viz_headless)):
            # Backward compatibility: map `--stream_viz_kalman` to filter_mode="pixel" if needed.
            viz_filter_mode = str(args.stream_viz_filter_mode)
            if bool(args.stream_viz_kalman) and viz_filter_mode == "none":
                viz_filter_mode = "pixel"

            stream_viz_queue = mp.Queue(maxsize=1)
            stream_viz_proc = mp.Process(
                target=run_stream_debug_viz,
                args=(stream_viz_queue,),
                kwargs=dict(
                    overlay_alpha=float(args.stream_viz_alpha),
                    scale=int(args.stream_viz_scale),
                    filter_mode=str(viz_filter_mode),
                    depth_source=str(args.stream_viz_depth_source),
                    enable_kalman=bool(args.stream_viz_kalman),
                    kalman_q=float(args.stream_viz_kalman_q),
                    kalman_r=float(args.stream_viz_kalman_r),
                    sample_grid=int(args.stream_viz_sample_grid),
                    sample_patch=int(args.stream_viz_sample_patch),
                    info_width=int(args.stream_viz_info_width),
                    depth_vis_max_m=float(args.stream_viz_depth_vis_max),
                    pose_reset_trans_m=float(args.stream_viz_pose_reset_trans),
                    pose_reset_rot_deg=float(args.stream_viz_pose_reset_rot_deg),
                    semantic_filter_mode=str(args.stream_viz_semantic_filter),
                    semantic_filter_target=str(args.stream_viz_semantic_filter_target),
                    semantic_filter_momentum=float(args.stream_viz_semantic_momentum),
                    semantic_filter_u=float(args.stream_viz_semantic_u),
                    semantic_depth_refine=bool(args.stream_viz_semantic_depth_refine),
                    semantic_depth_refine_target=str(args.stream_viz_semantic_depth_refine_target),
                    semantic_depth_refine_iters=int(args.stream_viz_semantic_depth_refine_iters),
                    semantic_depth_sigma_m=float(args.stream_viz_semantic_depth_sigma),
                    stable_semantic_source=str(args.stream_viz_stable_semantic_source),
                    semantic_topk=int(args.stream_viz_topk),
                    viz_layout=str(args.stream_viz_layout),
                    planning_pointcloud_outdir=(
                        str(args.stream_viz_pc_outdir)
                        if str(args.stream_viz_pc_outdir)
                        else str(args.planning_pointcloud_outdir)
                    ),
                    planning_pointcloud_info_filename=str(args.stream_viz_pc_info_filename),
                    planning_pointcloud_enable_pano=bool(args.stream_viz_pc_pano),
                    planning_pointcloud_radius_m=float(args.stream_viz_pc_radius_m),
                    planning_pointcloud_pano_h=int(args.stream_viz_pc_pano_h),
                    planning_pointcloud_pano_vfov_deg=float(args.stream_viz_pc_pano_vfov_deg),
                    planning_pointcloud_pano_mode=str(args.stream_viz_pc_pano_mode),
                    planning_pointcloud_fov_deg=float(args.stream_viz_pc_fov_deg),
                    planning_pointcloud_max_points=int(args.stream_viz_pc_max_points),
                ),
                daemon=True,
            )
            stream_viz_proc.start()
        elif args.enable_stream_debug_viz and bool(args.stream_viz_headless):
            from mast3r_slam.streaming.shared_debug_buffer import LatestSharedDebugBuffer

            stream_debug_shm = LatestSharedDebugBuffer(
                prefix="mast3r_stream_dbg",
                info_path="logs/stream_debug/shm_info.json",
            )
            print("[StreamDebug] headless enabled (no window). Shared-memory info: logs/stream_debug/shm_info.json")

        # Optional calibration for streaming mode.
        if args.calib:
            with open(args.calib, "r") as f:
                intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
            config["use_calib"] = True
            stream_intrinsics = Intrinsics.from_calib(
                img_size,
                intrinsics["width"],
                intrinsics["height"],
                intrinsics["calibration"],
            )

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    planning_viz_proc = None
    monitor_proc: subprocess.Popen | None = None
    enable_pc_viz = bool(args.enable_planning_pointcloud_viz)
    enable_pc_publish = bool(args.enable_planning_pointcloud_publish)
    enable_tsdf_publish = bool(args.enable_planning_tsdf_publish)

    if enable_pc_viz or enable_pc_publish or enable_tsdf_publish:
        # Spawn a separate, CPU-only process for planning integration.
        #
        # Two modes are supported:
        #   (1) --enable_planning_pointcloud_viz:
        #         Visualization/recording mode (optional window + optional PNG saving).
        #   (2) --enable_planning_pointcloud_publish:
        #         Publish-only mode (no window, no PNG saving). This is the recommended mode when
        #         a downstream planner wants to consume point cloud + pose but you don't want any GUI/IO.
        #
        # IMPORTANT:
        #   This must not block SLAM. We pass shared states/keyframes directly (shared memory).
        from mast3r_slam.planning_pointcloud_viz import run_planning_pointcloud_viz

        # We only show/save the 2D overlay when the pointcloud viz mode is enabled.
        show_window = bool(enable_pc_viz) and (not bool(args.planning_pointcloud_headless))
        save_images = bool(enable_pc_viz) and (not bool(args.planning_pointcloud_no_save_images))
        publish_shm = bool(args.planning_pointcloud_publish_shm)

        # Publish-only mode forces: publish_shm=True, show_window=False, save_images=False.
        if enable_pc_publish:
            show_window = False
            save_images = False
            publish_shm = True

        tsdf_trunc_m = None if float(args.planning_tsdf_trunc_m) <= 0.0 else float(args.planning_tsdf_trunc_m)
        tsdf_sem_band_m = (
            None if float(args.planning_tsdf_semantic_band_m) <= 0.0 else float(args.planning_tsdf_semantic_band_m)
        )

        planning_viz_proc = mp.Process(
            target=run_planning_pointcloud_viz,
            args=(config, states, keyframes),
            kwargs=dict(
                out_dir=str(args.planning_pointcloud_outdir),
                fps=float(args.planning_pointcloud_fps),
                max_keyframes=int(args.planning_pointcloud_max_keyframes),
                stride=int(args.planning_pointcloud_stride),
                conf_threshold=float(args.planning_pointcloud_conf_threshold),
                show_window=bool(show_window),
                save_images=bool(save_images),
                publish_shm=bool(publish_shm),
                shm_info_path=str(args.planning_pointcloud_shm_info),
                shm_max_points=int(args.planning_pointcloud_shm_max_points),
                shm_max_keyframes=int(args.planning_pointcloud_shm_max_keyframes),
                shm_points_per_kf=int(args.planning_pointcloud_shm_points_per_kf),
                pose_kalman_pos=float(args.planning_pointcloud_pose_kalman_pos),
                pose_kalman_rot=float(args.planning_pointcloud_pose_kalman_rot),
                publish_tsdf=bool(enable_tsdf_publish),
                tsdf_shm_info_path=str(args.planning_tsdf_shm_info),
                tsdf_radius_m=float(args.planning_tsdf_radius_m),
                tsdf_voxel_m=float(args.planning_tsdf_voxel_m),
                tsdf_trunc_m=tsdf_trunc_m,
                tsdf_max_weight=float(args.planning_tsdf_max_weight),
                tsdf_use_semantic=bool(args.planning_tsdf_use_semantic),
                tsdf_semantic_band_m=tsdf_sem_band_m,
                tsdf_frame_sem_dir=str(args.planning_tsdf_frame_sem_dir),
                tsdf_frame_sem_pattern=str(args.planning_tsdf_frame_sem_pattern),
                tsdf_pose_json=str(args.planning_tsdf_pose_json),
                tsdf_pose_key=str(args.planning_tsdf_pose_key),
                tsdf_pose_frame_stride=int(args.planning_tsdf_pose_frame_stride),
                tsdf_pose_frame_pattern=str(args.planning_tsdf_pose_frame_pattern),
                tsdf_backend=str(args.planning_tsdf_backend),
                tsdf_torch_device=str(args.planning_tsdf_torch_device),
                tsdf_torch_dtype=str(args.planning_tsdf_torch_dtype),
            ),
            daemon=True,
        )
        planning_viz_proc.start()

        # Optionally spawn the trajectory monitor script (debug only).
        #
        # IMPORTANT:
        #   - Requires that the publisher is enabled (we enforce `--enable_planning_pointcloud_publish`).
        #   - We always pass `--no_control` so no bridge/robot commands are sent.
        if bool(args.enable_monitor_pointcloud_traj):
            try:
                if not enable_pc_publish:
                    print("[PCMonitor] enable_monitor_pointcloud_traj ignored: requires --enable_planning_pointcloud_publish")
                else:
                    monitor_cmd = [
                        sys.executable,
                        os.path.join("scripts", "monitor_pointcloud_semantic_control.py"),
                        "--outdir",
                        str(args.planning_pointcloud_outdir),
                        "--hz",
                        str(float(args.monitor_pointcloud_traj_hz)),
                        "--fov_deg",
                        str(float(args.monitor_pointcloud_fov_deg)),
                        "--show_traj",
                        "--traj_size",
                        str(int(args.monitor_pointcloud_traj_size)),
                        "--no_control",
                    ]
                    monitor_proc = subprocess.Popen(monitor_cmd)

                    # Ensure we terminate the monitor process when SLAM exits.
                    def _kill_monitor() -> None:
                        try:
                            if monitor_proc is not None and monitor_proc.poll() is None:
                                monitor_proc.terminate()
                        except Exception:
                            pass

                    atexit.register(_kill_monitor)
                    print(f"[PCMonitor] spawned: {' '.join(monitor_cmd)}")
            except Exception as e:
                print(f"[PCMonitor] failed to spawn monitor script: {e!r}")

    model = load_mast3r(device=device)
    model.share_memory()

    has_calib = dataset.has_calib() if dataset is not None else (stream_intrinsics is not None)
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K_frame = (
            dataset.camera_intrinsics.K_frame
            if dataset is not None
            else stream_intrinsics.K_frame
        )
        K = torch.from_numpy(K_frame).to(device, dtype=torch.float32)
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset is not None and dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        dense_traj_file = save_dir / f"{seq_name}_dense.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if dense_traj_file.exists():
            dense_traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)
    external_tracking_pose = _load_external_tracking_pose_override(args)
    last_msg = WindowMsg()

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    i = 0
    fps_timer = time.time()
    first_frame_wall_start_s = None

    frames = []
    dense_pose_records = []
    last_stream_viz_send_t = 0.0

    def record_dense_pose(timestamp_s: float, frame) -> None:
        try:
            T_WC = as_SE3(frame.T_WC)
            x, y, z, qx, qy, qz, qw = T_WC.data.detach().cpu().numpy().reshape(-1)
            dense_pose_records.append(
                (
                    float(timestamp_s),
                    float(x),
                    float(y),
                    float(z),
                    float(qx),
                    float(qy),
                    float(qz),
                    float(qw),
                )
            )
        except Exception:
            pass

    def save_dense_traj(logdir: pathlib.Path, logfile: str) -> None:
        if not dense_pose_records:
            return
        logdir.mkdir(exist_ok=True, parents=True)
        out_path = logdir / logfile
        with open(out_path, "w") as f:
            for rec in dense_pose_records:
                f.write(" ".join(str(v) for v in rec) + "\n")

    def apply_external_tracking_pose(frame) -> None:
        if external_tracking_pose is None:
            return
        frame.T_WC = _external_tracking_sim3_for_frame(
            external_tracking_pose,
            int(frame.frame_id),
            device=device,
            dtype=frame.T_WC.data.dtype,
        )

    def _save_rgb_tensor_png(rgb, out_path: pathlib.Path) -> None:
        if rgb is None:
            return
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            return
        rgb_u8 = (rgb * 255.0).clip(0, 255).astype("uint8")
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)

    def maybe_save_semantic_rgb(frame, out_dir: pathlib.Path | None):
        """Save the SLAM-aligned RGB frame used for semantic overlays."""

        if out_dir is None:
            return
        out_path = out_dir / f"{int(frame.frame_id):06d}.png"
        _save_rgb_tensor_png(frame.uimg, out_path)

    def maybe_save_stable_semantic(frame, out_dir: pathlib.Path | None):
        """
        Save the current frame's post-fusion semantic_label as a PNG.

        The saved image preserves the repo's existing packed-RGB semantic format.
        Use scripts/blend_rgb_sem_to_video.py with --sem-mode label-code to render
        class-color overlays from these frames.
        """

        if out_dir is None or frame.semantic_label is None:
            return
        out_path = out_dir / f"{int(frame.frame_id):06d}.png"
        _save_rgb_tensor_png(frame.semantic_label, out_path)

    def maybe_save_raw_semantic(frame, out_dir: pathlib.Path | None):
        """Save the raw/pre-fusion semantic observation for the current frame."""

        if out_dir is None:
            return
        sem = getattr(frame, "semantic_label_raw", None)
        if sem is None:
            raw_hw = getattr(frame, "_semantic_label_hw_raw", None)
            if raw_hw is not None:
                sem = label_code_to_rgb(raw_hw.detach().cpu().to(torch.int64))
        if sem is None:
            return
        out_path = out_dir / f"{int(frame.frame_id):06d}.png"
        _save_rgb_tensor_png(sem, out_path)

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        # Termination conditions:
        #   - Dataset mode: stop when all frames are processed.
        #   - Streaming mode: optionally stop after N frames (otherwise run until terminated).
        if dataset is not None:
            if i == len(dataset):
                states.set_mode(Mode.TERMINATED)
                break
        else:
            if int(args.stream_max_frames) > 0 and i >= int(args.stream_max_frames):
                states.set_mode(Mode.TERMINATED)
                break

        # ---------------------------------------------------------------------
        # Fetch the next RGB frame (+ optional semantic) from the selected input source.
        #
        # IMPORTANT (streaming mode):
        #   - The main loop must never block on streaming latency.
        #   - If no new frame is available, we sleep briefly and continue.
        # ---------------------------------------------------------------------
        semantic_label = None
        timestamp = None
        img = None
        if dataset is not None:
            # Dataset mode (existing behavior).
            if dataset_async_semantic:
                # MP4 replay path: do NOT run segmentation inside the dataloader.
                #
                # Instead, we fetch only RGB frames from the dataset and run EfficientViT in an
                # optional async thread (latest-only). This mirrors the online streaming design:
                # SLAM never blocks on segmentation, and segmentation runs at a capped FPS.
                dataset.use_semantic = False
                if _timing is None:
                    timestamp, img = dataset[i]
                else:
                    t0 = time.perf_counter()
                    timestamp, img = dataset[i]
                    _timing.add_ms("dataset_ms", (time.perf_counter() - t0) * 1000.0)

                if args.enable_semantic_input:
                    if stream_segmenter is not None:
                        stream_segmenter.submit(img, float(timestamp))
                        seg = stream_segmenter.get_latest()
                        if seg is not None:
                            semantic_label = seg.label_hw
                    else:
                        # Synchronous fallback (debug): can slow down SLAM; use only when needed.
                        from mast3r_slam.efficientvit_segmenter import segment_image_efficientvit_labels

                        semantic_label = segment_image_efficientvit_labels(
                            img, dataset=str(args.efficientvit_dataset)
                        )

                    # If no new label is available yet, reuse the last label to avoid flicker.
                    if semantic_label is None and stream_last_label is not None:
                        semantic_label = stream_last_label
                    if semantic_label is not None:
                        stream_last_label = semantic_label

                    # If this is the very first frame and the async worker hasn't produced a label
                    # yet, we optionally do a one-time synchronous init so downstream code can see
                    # semantics from frame 0.
                    if semantic_label is None and i == 0:
                        from mast3r_slam.efficientvit_segmenter import segment_image_efficientvit_labels

                        semantic_label = segment_image_efficientvit_labels(
                            img, dataset=str(args.efficientvit_dataset)
                        )
                        stream_last_label = semantic_label
            else:
                dataset.use_semantic = bool(args.enable_semantic_input)
                if args.enable_semantic_input:
                    if _timing is None:
                        timestamp, img, semantic_label = dataset[i]
                    else:
                        t0 = time.perf_counter()
                        timestamp, img, semantic_label = dataset[i]
                        _timing.add_ms("dataset_ms", (time.perf_counter() - t0) * 1000.0)
                else:
                    if _timing is None:
                        timestamp, img = dataset[i]
                    else:
                        t0 = time.perf_counter()
                        timestamp, img = dataset[i]
                        _timing.add_ms("dataset_ms", (time.perf_counter() - t0) * 1000.0)
        else:
            # AirSim streaming mode.
            frm = stream_grabber.try_get_latest() if stream_grabber is not None else None
            if frm is None:
                time.sleep(max(0.0, float(args.stream_sleep_ms)) / 1000.0)
                continue

            timestamp = float(frm.timestamp_s)
            # MASt3R expects float01 RGB input (HxWx3).
            img = frm.rgb_u8.astype("float32") / 255.0

            if args.enable_semantic_input:
                # Asynchronous segmentation (preferred): submit the latest frame and fetch the latest label.
                if stream_segmenter is not None:
                    stream_segmenter.submit(img, timestamp)
                    seg = stream_segmenter.get_latest()
                    if seg is not None:
                        semantic_label = seg.label_hw
                else:
                    # Synchronous segmentation (debug): can slow SLAM; use only for validation.
                    from mast3r_slam.efficientvit_segmenter import segment_image_efficientvit_labels
                    semantic_label = segment_image_efficientvit_labels(
                        img, dataset=str(args.efficientvit_dataset)
                    )

                # If no new label is available yet, reuse the last label to avoid flicker.
                if semantic_label is None and stream_last_label is not None:
                    semantic_label = stream_last_label
                if semantic_label is not None:
                    stream_last_label = semantic_label

        if save_frames:
            frames.append(img)

        if first_frame_wall_start_s is None:
            first_frame_wall_start_s = time.time()

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )

        # Create the frame with or without semantic attached.
        if args.enable_semantic_input and (semantic_label is not None):
            if _timing is None:
                frame = create_frame_semantic(
                    i,
                    img,
                    T_WC,
                    semantic_label,
                    img_size=img_size,
                    device=device,
                )
            else:
                t0 = time.perf_counter()
                frame = create_frame_semantic(
                    i,
                    img,
                    T_WC,
                    semantic_label,
                    img_size=img_size,
                    device=device,
                )
                _timing.add_ms("frame_ms", (time.perf_counter() - t0) * 1000.0)
        else:
            if _timing is None:
                frame = create_frame(
                    i,
                    img,
                    T_WC,
                    img_size=img_size,
                    device=device,
                )
            else:
                t0 = time.perf_counter()
                frame = create_frame(
                    i,
                    img,
                    T_WC,
                    img_size=img_size,
                    device=device,
                )
                _timing.add_ms("frame_ms", (time.perf_counter() - t0) * 1000.0)

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            apply_external_tracking_pose(frame)
            # Use the current frame's Sim3 scale to scale depth before saving.
            scale = float(frame.T_WC.data[..., -1])
            maybe_save_depth_image(frame, depth_save_dir, scale=scale)
            maybe_save_semantic_rgb(frame, semantic_rgb_dir)
            maybe_save_raw_semantic(frame, raw_semantic_dir)
            maybe_save_stable_semantic(frame, stable_semantic_dir)

            # -----------------------------------------------------------------
            # V3: initialize keyframe semantic pointmap cache (optional)
            #
            # When enabled, we create:
            #   - frame.sem_label  : (H,W) int64   raw per-frame semantic labels
            #   - frame.sem_weight : (H,W) float32 initialized to a constant
            #
            # IMPORTANT:
            #   - This initialization is ONLY for keyframes.
            #   - When V3 is disabled, we do nothing and the V1 pipeline remains unchanged.
            # -----------------------------------------------------------------
            if args.enable_semantic_pointmap and args.enable_semantic_input and (frame.semantic_label is not None):
                h_s, w_s = (int(x) for x in frame.img_shape.flatten().tolist())
                label_hw = ensure_hard_label_hw(
                    frame.semantic_label, size_hw=(h_s, w_s), device=device
                )
                frame.sem_label = label_hw
                frame.sem_weight = torch.full(
                    (h_s, w_s),
                    float(args.semantic_pointmap_init_weight),
                    device=device,
                    dtype=torch.float32,
                )

            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            record_dense_pose(timestamp, frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            if _timing is None:
                add_new_kf, match_info, try_reloc = tracker.track(frame)
            else:
                t0 = time.perf_counter()
                add_new_kf, match_info, try_reloc = tracker.track(frame)
                _timing.add_ms("track_ms", (time.perf_counter() - t0) * 1000.0)
            apply_external_tracking_pose(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)
            scale = float(frame.T_WC.data[..., -1])
            maybe_save_depth_image(frame, depth_save_dir, scale=scale)
            maybe_save_semantic_rgb(frame, semantic_rgb_dir)
            maybe_save_raw_semantic(frame, raw_semantic_dir)
            # Save stabilized semantic after tracking (i.e., after (A) warp+fuse).
            # We only attempt this in TRACKING mode since INIT/RELOC may not produce stabilized output.
            maybe_save_stable_semantic(frame, stable_semantic_dir)

            # -----------------------------------------------------------------
            # PLANNING HOOK (in-process, simplest possible "entry point")
            #
            # If you are OK with running planning in the SAME Python process (even if it blocks),
            # you can directly access:
            #   - current pose : `frame.T_WC` (lietorch.Sim3)
            #   - point cloud  : per-keyframe `keyframes[k].X_canon` and `keyframes[k].T_WC`
            #   - semantics    : per-keyframe `keyframes[k].semantic_label` (RGB mask)
            #
            # For convenience, you can also compute a ready-to-use snapshot (world points + colors)
            # and store it in a module-global variable:
            #   `mast3r_slam.planning_hook.LATEST_PLANNING_SNAPSHOT`
            #
            # Uncomment the two lines below to enable it.
            #
            # IMPORTANT:
            #   This is IN-PROCESS ONLY. If your planner runs in another OS process, you must use IPC
            #   (e.g., `--planning_pointcloud_publish_shm`).
            # -----------------------------------------------------------------
            # from mast3r_slam.planning_hook import update_latest_planning_snapshot
            # update_latest_planning_snapshot(frame=frame, keyframes=keyframes, max_keyframes=30, stride=4, conf_threshold=1.5)

            # -----------------------------------------------------------------
            # Downstream planning/statistics: access per-frame semantic + depth
            #
            # The tracker may attach the following tensors to `frame`:
            #   - frame._semantic_label_hw_raw : (H,W) int64 (raw semantic on match grid)
            #   - frame._depth_hw              : (H,W) float32 (depth on match grid)
            #   - frame._depth_hw_valid        : (H,W) bool
            #
            # IMPORTANT:
            #   This logic is intentionally kept in the main process so it cannot affect SLAM.
            #   To integrate your planner, uncomment the single-line HOOK call below.
            # -----------------------------------------------------------------
            label_hw = getattr(frame, "_semantic_label_hw_raw", None)
            depth_hw = getattr(frame, "_depth_hw", None)
            depth_valid_hw = getattr(frame, "_depth_hw_valid", None)

            if label_hw is not None and depth_hw is not None:
                # HOOK (single line): Uncomment the next line to feed your planner.
                # semantic_depth_hook(label_hw, depth_hw, frame_id=int(frame.frame_id))

                # Optional streaming debug visualization (separate process, non-blocking).
                #
                # IMPORTANT (as requested):
                #   All debug-only operations (margin crop, global min selection, Kalman smoothing)
                #   are implemented in `mast3r_slam/streaming/debug_viz.py`.
                #
                # This main process only sends the raw inputs required for visualization:
                #   - RGB for background
                #   - semantic label map (for overlay)
                #   - depth map (for global min depth selection / smoothing)
                #
                # We rate-limit updates to avoid excessive GPU->CPU synchronization and IPC traffic.
                if args.enable_stream_debug_viz and (stream_viz_queue is not None):
                    now_t = time.time()
                    viz_period = 1.0 / max(1, int(args.stream_viz_fps))
                    if (now_t - last_stream_viz_send_t) >= viz_period:
                        last_stream_viz_send_t = now_t

                        # Convert match-grid RGB for background visualization.
                        rgb_u8 = (
                            (frame.uimg.detach().cpu().numpy() * 255.0)
                            .clip(0, 255)
                            .astype("uint8")
                        )

                        # Visualization-only alignment helper:
                        #
                        # Problem:
                        #   The SLAM image `frame.uimg` is produced by `resize_img()` which does a
                        #   resize+center-crop pipeline (e.g., for 224 it center-crops to square).
                        #   Some semantic sources (e.g., MP4 async EfficientViT labels, external masks)
                        #   may have been resized to the match grid via a simple (H,W)->(224,224)
                        #   nearest resize WITHOUT applying the same crop transform.
                        #
                        # Result:
                        #   Even if RGB and semantic have the same (H,W), overlay can look "misaligned"
                        #   because they correspond to different image coordinate systems.
                        #
                        # Visualization-only fix:
                        #   Build an additional RGB background `rgb_sem_u8` by squashing the ORIGINAL
                        #   input frame `img` to the match grid (H,W). The debug process will use this
                        #   background ONLY for the semantic panels, keeping SLAM logic unchanged.
                        rgb_sem_u8 = None
                        try:
                            if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
                                # `img` is float01 RGB in the original input resolution.
                                img_u8 = (img * 255.0).clip(0, 255).astype("uint8")
                                rgb_sem_u8 = cv2.resize(
                                    img_u8,
                                    (int(w_s), int(h_s)),
                                    interpolation=cv2.INTER_AREA,
                                )
                        except Exception:
                            rgb_sem_u8 = None

                        # Select which semantic to visualize in the debug window:
                        # We send BOTH raw and stable semantics so the debug UI can display them
                        # side-by-side without changing any SLAM logic.
                        #
                        # Raw semantic:
                        #   `frame._semantic_label_hw_raw` is captured before semantic warp+fuse.
                        # Stable semantic:
                        #   `frame.semantic_label` is the shared-memory RGB representation that the tracker
                        #   overwrites after semantic warp+fuse (A). We decode it back to integer label IDs.
                        #
                        # IMPORTANT:
                        #   This is visualization-only. Nothing here feeds back into SLAM.
                        h_s, w_s = (int(x) for x in label_hw.shape)

                        label_raw_cpu = (
                            label_hw.detach().to("cpu").to(torch.int64).numpy()
                        )

                        label_stable_cpu = None
                        if frame.semantic_label is not None:
                            try:
                                label_stable_t = ensure_hard_label_hw(
                                    frame.semantic_label,
                                    size_hw=(h_s, w_s),
                                    device=torch.device("cpu"),
                                )
                                label_stable_cpu = (
                                    label_stable_t.detach().to(torch.int64).numpy()
                                )
                            except Exception:
                                label_stable_cpu = None

                        # Semantic RGB visualizations (to match `--save_stable_semantic` output style):
                        #
                        # - `sem_stable_rgb_u8` comes directly from `frame.semantic_label` after tracking,
                        #    exactly like `maybe_save_stable_semantic()` writes it to disk.
                        # - `sem_raw_rgb_u8` comes from the preserved pre-tracking tensor reference
                        #    `frame._semantic_label_rgb_raw` when available.
                        #
                        # This avoids any "re-colorization" differences (e.g., hash palettes) that can
                        # make stable-vs-raw look misleading in the GUI.
                        sem_stable_rgb_u8 = None
                        if frame.semantic_label is not None:
                            try:
                                sem = frame.semantic_label.detach().cpu()
                                sem_u8 = (sem.numpy() * 255.0).clip(0, 255).astype("uint8")
                                if sem_u8.ndim == 3 and sem_u8.shape[-1] == 3:
                                    sem_stable_rgb_u8 = sem_u8
                            except Exception:
                                sem_stable_rgb_u8 = None

                        # Depth is produced on the match grid (H,W). We send it to the debug process
                        # and let that process decide how to filter / pick the min value.
                        #
                        # Note:
                        #   This conversion synchronizes GPU->CPU, but it is rate-limited by
                        #   `--stream_viz_fps` and is debug-only.
                        depth_stable_cpu = depth_hw.detach().to("cpu").to(torch.float32).numpy()

                        # Also derive the raw depth map (current-frame pointmap Z) for debug display.
                        # This is fast and does not require any extra matching or warping.
                        depth_raw_cpu = None
                        try:
                            if frame.X_canon is not None:
                                depth_raw_t = (
                                    frame.X_canon.reshape(-1, 3)[:, 2]
                                    .view(h_s, w_s)
                                    .to(torch.float32)
                                )
                                depth_raw_cpu = depth_raw_t.detach().to("cpu").numpy()
                        except Exception:
                            depth_raw_cpu = None

                        # Optional (debug-only): send the current camera pose as a compact vector.
                        #
                        # Why we send it:
                        #   The debug visualizer can optionally run a "pose-aware" depth smoother,
                        #   where a pixel-wise temporal filter is RESET if the camera moves too much.
                        #   This avoids the classic pitfall of per-pixel smoothing during motion.
                        #
                        # Format:
                        #   pose_tqs: (8,) float32 ~ [tx,ty,tz,qx,qy,qz,qw,scale]
                        #
                        # IMPORTANT:
                        #   This is used ONLY in the debug process. It MUST NOT feed back into SLAM.
                        pose_tqs = None
                        try:
                            pose_tqs = (
                                frame.T_WC.data.detach()
                                .to("cpu")
                                .to(torch.float32)
                                .numpy()
                                .reshape(-1)
                            )
                        except Exception:
                            pose_tqs = None

                        msg = dict(
                            frame_id=int(frame.frame_id),
                            rgb_u8=rgb_u8,
                            rgb_sem_u8=rgb_sem_u8,
                            # New message format (preferred):
                            label_raw_hw=label_raw_cpu,
                            label_stable_hw=label_stable_cpu,
                            sem_stable_rgb_u8=sem_stable_rgb_u8,
                            depth_raw_hw=depth_raw_cpu,
                            depth_stable_hw=depth_stable_cpu,
                            # Backward compatibility (older debug_viz versions):
                            label_hw=label_stable_cpu if label_stable_cpu is not None else label_raw_cpu,
                            depth_hw=depth_stable_cpu,
                            pose_tqs=pose_tqs,
                        )
                        # Latest-only send: drop old message if the queue is full.
                        try:
                            if stream_viz_queue.full():
                                _ = stream_viz_queue.get_nowait()
                            stream_viz_queue.put_nowait(msg)
                        except Exception:
                            pass
                # Headless debug publishing (no OpenCV window).
                #
                # When enabled, we publish the *same* payload that would have been visualized
                # into shared memory, so other processes can attach and "grab" the latest
                # semantic/depth without adding any blocking GUI calls.
                if args.enable_stream_debug_viz and bool(args.stream_viz_headless) and (stream_debug_shm is not None):
                    now_t = time.time()
                    viz_period = 1.0 / max(1, int(args.stream_viz_fps))
                    if (now_t - last_stream_viz_send_t) >= viz_period:
                        last_stream_viz_send_t = now_t

                        # Convert match-grid RGB for background.
                        rgb_u8 = (
                            (frame.uimg.detach().cpu().numpy() * 255.0)
                            .clip(0, 255)
                            .astype("uint8")
                        )

                        # Raw labels are always available here (captured before warp+fuse).
                        h_s, w_s = (int(x) for x in label_hw.shape)
                        label_raw_cpu = label_hw.detach().to("cpu").to(torch.int64).numpy()

                        # Decode stable label if available; otherwise leave None.
                        label_stable_cpu = None
                        if frame.semantic_label is not None:
                            try:
                                label_stable_t = ensure_hard_label_hw(
                                    frame.semantic_label,
                                    size_hw=(h_s, w_s),
                                    device=torch.device("cpu"),
                                )
                                label_stable_cpu = label_stable_t.detach().to(torch.int64).numpy()
                            except Exception:
                                label_stable_cpu = None

                        # Stable depth is whatever the tracker computed for downstream (depends on --depth_source).
                        depth_stable_cpu = depth_hw.detach().to("cpu").to(torch.float32).numpy()

                        # Raw depth: current-frame pointmap Z.
                        depth_raw_cpu = None
                        try:
                            if frame.X_canon is not None:
                                depth_raw_t = (
                                    frame.X_canon.reshape(-1, 3)[:, 2]
                                    .view(h_s, w_s)
                                    .to(torch.float32)
                                )
                                depth_raw_cpu = depth_raw_t.detach().to("cpu").numpy()
                        except Exception:
                            depth_raw_cpu = None

                        # Initialize shared memory blocks on first use (or on shape change).
                        try:
                            stream_debug_shm.ensure(shape_hw=(h_s, w_s))
                            stream_debug_shm.write(
                                frame_id=int(frame.frame_id),
                                timestamp_s=float(now_t),
                                rgb_u8=rgb_u8,
                                label_raw_hw=label_raw_cpu,
                                label_stable_hw=label_stable_cpu,
                                depth_raw_hw=depth_raw_cpu,
                                depth_stable_hw=depth_stable_cpu,
                            )
                        except Exception:
                            # Never let debug publishing affect SLAM.
                            pass

                if args.print_min_depth_per_class:
                    every = int(args.print_min_depth_every)
                    fid = int(frame.frame_id)
                    if fid < 5 or (every > 0 and fid % every == 0):
                        num_classes = int(args.semantic_num_classes)
                        if num_classes <= 0:
                            # Auto mode assumes label IDs are small (0..C-1). If labels are large
                            # 24-bit RGB codes, set --semantic_num_classes explicitly.
                            max_label = int(label_hw.max().item()) if label_hw.numel() > 0 else 0
                            num_classes = max_label + 1

                        if depth_valid_hw is None:
                            depth_valid_hw = torch.isfinite(depth_hw) & (depth_hw > 0.0)

                        min_d, cnt = min_depth_per_class(
                            label_hw=label_hw,
                            depth_hw=depth_hw,
                            valid_hw=depth_valid_hw,
                            num_classes=num_classes,
                        )
                        present = torch.nonzero(cnt > 0, as_tuple=False).reshape(-1)
                        items = []
                        for c in present.tolist():
                            items.append(
                                f"{c}:{float(min_d[c].item()):.3f}({int(cnt[c].item())})"
                            )
                        print(
                            f"[MinDepthPerClass] frame={fid} depth_source={args.depth_source} "
                            + " ".join(items)
                        )

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            apply_external_tracking_pose(frame)
            states.set_frame(frame)
            scale = float(frame.T_WC.data[..., -1])
            maybe_save_depth_image(frame, depth_save_dir, scale=scale)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            # -----------------------------------------------------------------
            # V3: initialize new keyframe semantic pointmap cache (optional)
            #
            # We explicitly initialize from the RAW per-frame semantic observation.
            # During tracking, (A) semantic warp may overwrite `frame.semantic_label` for
            # visualization/output, so we prefer the preserved raw hard label if available.
            # -----------------------------------------------------------------
            if args.enable_semantic_pointmap and args.enable_semantic_input:
                h_s, w_s = (int(x) for x in frame.img_shape.flatten().tolist())
                raw_hw = getattr(frame, "_semantic_label_hw_raw", None)
                if raw_hw is None and (frame.semantic_label is not None):
                    # Fallback: if raw was not preserved for some reason, derive from the current
                    # semantic_label tensor. Note that this may be stabilized (post-warp) depending
                    # on runtime settings.
                    raw_hw = ensure_hard_label_hw(
                        frame.semantic_label, size_hw=(h_s, w_s), device=device
                    )
                if raw_hw is not None:
                    frame.sem_label = raw_hw.to(torch.int64)
                    frame.sem_weight = torch.full(
                        (h_s, w_s),
                        float(args.semantic_pointmap_init_weight),
                        device=device,
                        dtype=torch.float32,
                    )

            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        record_dense_pose(timestamp, frame)
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        if _timing is not None:
            _timing.tick_frame()
            _timing.maybe_print(i)
        i += 1

    # -------------------------------------------------------------------------
    # Cleanup streaming workers (if any)
    #
    # These are optional components; we shut them down explicitly to avoid leaving
    # background threads/processes running after SLAM exits.
    # -------------------------------------------------------------------------
    if stream_segmenter is not None:
        try:
            stream_segmenter.stop()
        except Exception:
            pass
    if stream_grabber is not None:
        try:
            stream_grabber.stop()
        except Exception:
            pass
    if stream_teleop is not None:
        try:
            stream_teleop.stop()
        except Exception:
            pass
    if stream_viz_proc is not None:
        try:
            stream_viz_proc.terminate()
        except Exception:
            pass
        try:
            stream_viz_proc.join(timeout=1.0)
        except Exception:
            pass
    if planning_viz_proc is not None:
        # Best-effort shutdown for the planning pointcloud process.
        try:
            planning_viz_proc.terminate()
        except Exception:
            pass
        try:
            # The planning publisher traps SIGTERM to clean up shared memory, so give it a bit
            # more time to exit gracefully. If it still does not exit, fall back to SIGKILL.
            planning_viz_proc.join(timeout=5.0)
        except Exception:
            pass
        try:
            if planning_viz_proc.is_alive():
                planning_viz_proc.kill()
                planning_viz_proc.join(timeout=1.0)
        except Exception:
            pass
    if stream_debug_shm is not None:
        # Explicitly close/unlink shared memory segments to avoid leaking them across runs.
        # This is best-effort and must not raise during shutdown.
        try:
            stream_debug_shm.close(unlink=True)
        except Exception:
            pass

    if enable_tsdf_publish:
        try:
            from mast3r_slam.global_esdf_snapshot import save_global_esdf_snapshot

            last_frame_id = -1
            try:
                if len(keyframes) > 0:
                    last_frame_id = int(keyframes.last_keyframe().frame_id)
            except Exception:
                last_frame_id = -1

            save_global_esdf_snapshot(
                out_dir=str(args.planning_pointcloud_outdir),
                keyframes=keyframes,
                voxel_m=float(args.planning_tsdf_voxel_m),
                stride=int(args.planning_pointcloud_stride),
                conf_threshold=float(args.planning_pointcloud_conf_threshold),
                frame_id=int(last_frame_id),
                timestamp_s=float(time.time()),
                scene_id="",
                use_semantic=bool(args.planning_tsdf_use_semantic),
                padding_vox=2,
                pose_json=str(args.planning_tsdf_pose_json),
                pose_key=str(args.planning_tsdf_pose_key),
                pose_frame_stride=int(args.planning_tsdf_pose_frame_stride),
                pose_frame_pattern=str(args.planning_tsdf_pose_frame_pattern),
            )
        except Exception as e:
            print(f"[PlanningTSDF] failed to save global ESDF snapshot: {e!r}")

    try:
        timing_out_dir = pathlib.Path(args.planning_pointcloud_outdir)
        timing_out_dir.mkdir(exist_ok=True, parents=True)
        if first_frame_wall_start_s is not None:
            loop_elapsed_s = float(time.time() - first_frame_wall_start_s)
            loop_fps = float(i) / loop_elapsed_s if loop_elapsed_s > 1e-9 else 0.0
        else:
            loop_elapsed_s = 0.0
            loop_fps = 0.0
        timing_payload = {
            "num_frames_processed": int(i),
            "first_frame_to_last_frame_elapsed_s": loop_elapsed_s,
            "first_frame_to_last_frame_fps": loop_fps,
            "timing_start_note": "starts immediately before processing the first frame after initialization",
            "timing_end_note": "ends after the last frame exits the main SLAM loop, before final result saving/postprocessing",
        }
        timing_json = timing_out_dir / "reconstruction_timing.json"
        timing_txt = timing_out_dir / "reconstruction_timing.txt"
        timing_json.write_text(json.dumps(timing_payload, indent=2) + "\n")
        timing_txt.write_text(
            "\n".join(
                [
                    f"num_frames_processed: {int(i)}",
                    f"first_frame_to_last_frame_elapsed_s: {loop_elapsed_s:.6f}",
                    f"first_frame_to_last_frame_fps: {loop_fps:.6f}",
                ]
            )
            + "\n"
        )
        print(f"[Timing] saved reconstruction timing: {timing_json}")
    except Exception as e:
        print(f"[Timing] failed to save reconstruction timing: {e!r}")

    if dataset is not None and dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        save_dense_traj(save_dir, f"{seq_name}_dense.txt")
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    backend.join()
    if not args.no_viz:
        viz.join()
