from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from mast3r_slam.planning_pointcloud_client import PlanningTsdfVolumeClient

try:
    from scipy import ndimage as ndi
except Exception as e:  # pragma: no cover
    raise ImportError("scipy is required for ESDF snapshot export.") from e


@dataclass(frozen=True)
class EsdfSnapshot:
    schema: str
    frame_id: int
    timestamp_s: float
    scene_id: str
    radius_m: float
    voxel_m: float
    dims: np.ndarray
    origin_w: np.ndarray
    curr_T_WC: np.ndarray
    esdf: np.ndarray
    occ: np.ndarray
    tsdf: np.ndarray
    weight: np.ndarray
    sem_label: np.ndarray
    sem_weight: np.ndarray
    use_semantic: bool
    obstacle_labels: np.ndarray
    w_min: float
    sem_w_min: float
    dilate_iters: int


def parse_obstacle_labels(arg: str | Iterable[int] | None) -> Optional[list[int]]:
    if arg is None:
        return None
    if isinstance(arg, str):
        labels = [int(x) for x in str(arg).split(",") if str(x).strip()]
        return labels if labels else None
    labels = [int(x) for x in arg]
    return labels if labels else None


def compute_esdf_from_tsdf(
    *,
    tsdf: np.ndarray,
    weight: np.ndarray,
    sem_label: np.ndarray,
    sem_weight: np.ndarray,
    voxel_m: float,
    w_min: float,
    use_semantic: bool,
    obstacle_labels: Optional[Iterable[int]],
    sem_w_min: float,
    dilate_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid_geom = weight > float(w_min)
    occ_geom = (tsdf < 0) & valid_geom

    if use_semantic:
        valid_sem = (sem_weight > float(sem_w_min)) & valid_geom
        if obstacle_labels:
            obst_mask = np.isin(sem_label, np.asarray(list(obstacle_labels), dtype=np.int32))
        else:
            obst_mask = np.ones_like(sem_label, dtype=bool)
        occ = occ_geom & valid_sem & obst_mask
    else:
        occ = occ_geom

    if int(dilate_iters) > 0:
        occ = ndi.binary_dilation(occ, iterations=int(dilate_iters))

    d_free = ndi.distance_transform_edt(~occ).astype(np.float32) * float(voxel_m)
    d_occ = ndi.distance_transform_edt(occ).astype(np.float32) * float(voxel_m)
    esdf = d_free
    esdf[occ] = -d_occ[occ]
    return esdf.astype(np.float32, copy=False), occ.astype(np.uint8, copy=False)


def make_esdf_snapshot(
    *,
    outdir: str | Path,
    scene_id: str = "",
    w_min: float = 1.0,
    use_semantic: bool = False,
    obstacle_labels: Optional[Iterable[int]] = None,
    sem_w_min: float = 1.0,
    dilate_iters: int = 1,
) -> EsdfSnapshot:
    client = PlanningTsdfVolumeClient(out_dir=str(outdir))
    snap = client.read(copy=True)
    client.close()

    info_path = Path(outdir) / "tsdf_shm_info.json"
    radius_m = 0.0
    if info_path.exists():
        import json

        info = json.loads(info_path.read_text(encoding="utf-8"))
        radius_m = float(info.get("radius_m", 0.0))

    esdf, occ = compute_esdf_from_tsdf(
        tsdf=np.asarray(snap.tsdf, dtype=np.float32),
        weight=np.asarray(snap.weight, dtype=np.float32),
        sem_label=np.asarray(snap.sem_label, dtype=np.int32),
        sem_weight=np.asarray(snap.sem_weight, dtype=np.float32),
        voxel_m=float(snap.voxel_m),
        w_min=float(w_min),
        use_semantic=bool(use_semantic),
        obstacle_labels=obstacle_labels,
        sem_w_min=float(sem_w_min),
        dilate_iters=int(dilate_iters),
    )

    labels = parse_obstacle_labels(obstacle_labels)
    return EsdfSnapshot(
        schema="mast3r_esdf_snapshot_v1",
        frame_id=int(snap.frame_id),
        timestamp_s=float(snap.timestamp_s),
        scene_id=str(scene_id),
        radius_m=float(radius_m),
        voxel_m=float(snap.voxel_m),
        dims=np.asarray(snap.dims, dtype=np.int32),
        origin_w=np.asarray(snap.origin_w, dtype=np.float32),
        curr_T_WC=np.asarray(snap.curr_T_WC, dtype=np.float32),
        esdf=np.asarray(esdf, dtype=np.float32),
        occ=np.asarray(occ, dtype=np.uint8),
        tsdf=np.asarray(snap.tsdf, dtype=np.float32),
        weight=np.asarray(snap.weight, dtype=np.float32),
        sem_label=np.asarray(snap.sem_label, dtype=np.int32),
        sem_weight=np.asarray(snap.sem_weight, dtype=np.float32),
        use_semantic=bool(use_semantic),
        obstacle_labels=np.asarray(labels if labels is not None else [], dtype=np.int32),
        w_min=float(w_min),
        sem_w_min=float(sem_w_min),
        dilate_iters=int(dilate_iters),
    )


def save_esdf_snapshot(snapshot: EsdfSnapshot, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        schema=np.asarray(snapshot.schema),
        frame_id=np.asarray(snapshot.frame_id, dtype=np.int64),
        timestamp_s=np.asarray(snapshot.timestamp_s, dtype=np.float64),
        scene_id=np.asarray(snapshot.scene_id),
        radius_m=np.asarray(snapshot.radius_m, dtype=np.float32),
        voxel_m=np.asarray(snapshot.voxel_m, dtype=np.float32),
        dims=np.asarray(snapshot.dims, dtype=np.int32),
        origin_w=np.asarray(snapshot.origin_w, dtype=np.float32),
        curr_T_WC=np.asarray(snapshot.curr_T_WC, dtype=np.float32),
        esdf=np.asarray(snapshot.esdf, dtype=np.float32),
        occ=np.asarray(snapshot.occ, dtype=np.uint8),
        tsdf=np.asarray(snapshot.tsdf, dtype=np.float32),
        weight=np.asarray(snapshot.weight, dtype=np.float32),
        sem_label=np.asarray(snapshot.sem_label, dtype=np.int32),
        sem_weight=np.asarray(snapshot.sem_weight, dtype=np.float32),
        use_semantic=np.asarray(int(snapshot.use_semantic), dtype=np.int32),
        obstacle_labels=np.asarray(snapshot.obstacle_labels, dtype=np.int32),
        w_min=np.asarray(snapshot.w_min, dtype=np.float32),
        sem_w_min=np.asarray(snapshot.sem_w_min, dtype=np.float32),
        dilate_iters=np.asarray(snapshot.dilate_iters, dtype=np.int32),
    )
    return out_path


def load_esdf_snapshot(path: str | Path) -> EsdfSnapshot:
    arr = np.load(Path(path), allow_pickle=False)
    schema = str(arr["schema"].item())
    if schema != "mast3r_esdf_snapshot_v1":
        raise ValueError(f"Unsupported ESDF snapshot schema: {schema}")
    return EsdfSnapshot(
        schema=schema,
        frame_id=int(arr["frame_id"].item()),
        timestamp_s=float(arr["timestamp_s"].item()),
        scene_id=str(arr["scene_id"].item()),
        radius_m=float(arr["radius_m"].item()),
        voxel_m=float(arr["voxel_m"].item()),
        dims=np.asarray(arr["dims"], dtype=np.int32),
        origin_w=np.asarray(arr["origin_w"], dtype=np.float32),
        curr_T_WC=np.asarray(arr["curr_T_WC"], dtype=np.float32),
        esdf=np.asarray(arr["esdf"], dtype=np.float32),
        occ=np.asarray(arr["occ"], dtype=np.uint8),
        tsdf=np.asarray(arr["tsdf"], dtype=np.float32),
        weight=np.asarray(arr["weight"], dtype=np.float32),
        sem_label=np.asarray(arr["sem_label"], dtype=np.int32),
        sem_weight=np.asarray(arr["sem_weight"], dtype=np.float32),
        use_semantic=bool(int(arr["use_semantic"].item())),
        obstacle_labels=np.asarray(arr["obstacle_labels"], dtype=np.int32),
        w_min=float(arr["w_min"].item()),
        sem_w_min=float(arr["sem_w_min"].item()),
        dilate_iters=int(arr["dilate_iters"].item()),
    )
