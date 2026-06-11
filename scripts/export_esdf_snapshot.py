#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mast3r_slam.esdf_snapshot import make_esdf_snapshot, parse_obstacle_labels, save_esdf_snapshot


def main() -> int:
    p = argparse.ArgumentParser(description="Export an offline ESDF snapshot from the TSDF shared-memory publisher.")
    p.add_argument("--outdir", type=str, required=True, help="Directory containing tsdf_shm_info.json.")
    p.add_argument("--snapshot", type=Path, required=True, help="Output .npz snapshot path.")
    p.add_argument("--scene-id", type=str, default="", help="Optional scene id to store in the snapshot.")
    p.add_argument("--w-min", type=float, default=1.0, help="Minimum TSDF weight for valid geometry.")
    p.add_argument("--use-semantic", action="store_true", help="Use semantic labels to filter occupancy.")
    p.add_argument("--obstacle-labels", type=str, default="", help="Comma-separated semantic labels considered obstacles.")
    p.add_argument("--sem-w-min", type=float, default=1.0, help="Minimum semantic weight for valid semantics.")
    p.add_argument("--dilate", type=int, default=1, help="Binary dilation iterations on occupancy.")
    p.add_argument("--min-frame-id", type=int, default=1, help="Wait until the publisher reaches at least this frame id.")
    p.add_argument("--timeout-s", type=float, default=10.0, help="Wait timeout while polling shared memory.")
    args = p.parse_args()

    deadline = time.time() + max(0.0, float(args.timeout_s))
    last_frame_id = None
    last_weight_sum = None
    while True:
        snapshot = make_esdf_snapshot(
            outdir=args.outdir,
            scene_id=args.scene_id,
            w_min=float(args.w_min),
            use_semantic=bool(args.use_semantic),
            obstacle_labels=parse_obstacle_labels(args.obstacle_labels),
            sem_w_min=float(args.sem_w_min),
            dilate_iters=int(args.dilate),
        )
        last_frame_id = int(snapshot.frame_id)
        last_weight_sum = float(np.sum(snapshot.weight))
        if last_frame_id >= int(args.min_frame_id):
            break
        if time.time() > deadline:
            raise RuntimeError(
                f"Timed out waiting for frame_id >= {args.min_frame_id}; "
                f"last_frame_id={last_frame_id} weight_sum={last_weight_sum:.3f}"
            )
        time.sleep(0.2)

    out_path = save_esdf_snapshot(snapshot, args.snapshot)
    print(
        f"saved={out_path} frame_id={snapshot.frame_id} voxel_m={snapshot.voxel_m:.4f} "
        f"dims={tuple(int(x) for x in snapshot.dims)} weight_sum={last_weight_sum:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
