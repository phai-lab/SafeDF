from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from mast3r_slam.esdf_snapshot import EsdfSnapshot, save_esdf_snapshot

try:
    from scipy import ndimage as ndi
except Exception as e:  # pragma: no cover
    raise ImportError("scipy is required for global ESDF snapshot export.") from e


def _sim3_matrix(T) -> np.ndarray:
    m = T.matrix()
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 3:
        m = m[0]
    return m


def _load_pose_table(pose_json: str | Path | None) -> dict | None:
    if pose_json is None or str(pose_json).strip() == "":
        return None
    data = json.loads(Path(pose_json).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected pose JSON object, got {type(data).__name__}: {pose_json}")
    return data


def _lookup_pose_matrix(
    pose_table: dict | None,
    *,
    frame_id: int,
    stride: int,
    pattern: str,
    pose_key: str,
) -> np.ndarray | None:
    if pose_table is None:
        return None
    source_frame_id = int(frame_id) * max(1, int(stride))
    key = str(pattern).format(frame_id=source_frame_id)
    meta = pose_table.get(key)
    if meta is None:
        meta = pose_table.get(str(source_frame_id))
    if meta is None:
        return None
    if str(pose_key) not in meta:
        return None
    return np.asarray(meta[str(pose_key)], dtype=np.float32).reshape(4, 4)


def _rgb_to_label_id(rgb_hw3: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb_hw3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3, got {arr.shape}")
    if arr.dtype != np.uint8:
        maxv = float(arr.max()) if arr.size else 0.0
        if maxv <= 1.0 + 1e-6:
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    r = arr[..., 0].astype(np.uint32)
    g = arr[..., 1].astype(np.uint32)
    b = arr[..., 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


def _extract_keyframe_points_world(
    *,
    keyframe,
    stride: int,
    conf_threshold: float,
    pose_table: dict | None = None,
    pose_frame_stride: int = 1,
    pose_frame_pattern: str = "frame_{frame_id:06d}",
    pose_key: str = "aligned_pose",
) -> tuple[np.ndarray, np.ndarray]:
    h, w = (int(x) for x in keyframe.img_shape.flatten().tolist())
    if h <= 0 or w <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    if keyframe.X_canon is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    s = int(max(1, stride))
    X = keyframe.X_canon
    if isinstance(X, torch.Tensor):
        Xs = X.reshape(h, w, 3)[::s, ::s].reshape(-1, 3).detach().to("cpu").to(torch.float32).numpy()
    else:
        Xs = np.asarray(X, dtype=np.float32).reshape(h, w, 3)[::s, ::s].reshape(-1, 3)

    C = keyframe.get_average_conf()
    if isinstance(C, torch.Tensor):
        Cs = C.reshape(h, w)[::s, ::s].reshape(-1).detach().to("cpu").to(torch.float32).numpy()
    else:
        Cs = np.asarray(C, dtype=np.float32).reshape(h, w)[::s, ::s].reshape(-1)

    valid = np.isfinite(Xs).all(axis=1) & np.isfinite(Cs) & (Xs[:, 2] > 0.0)
    if float(conf_threshold) > 0.0:
        valid &= Cs >= float(conf_threshold)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    Xs = Xs[valid].astype(np.float32, copy=False)

    labels = np.full((Xs.shape[0],), np.int32(-1), dtype=np.int32)
    sem = getattr(keyframe, "semantic_label", None)
    if sem is not None:
        if isinstance(sem, torch.Tensor):
            sem = sem.detach().cpu().numpy()
        sem = np.asarray(sem).reshape(h, w, 3)
        try:
            lab = _rgb_to_label_id(sem[::s, ::s]).reshape(-1)
            labels = lab[valid].astype(np.int32, copy=False)
        except Exception:
            pass

    frame_id = int(getattr(keyframe, "frame_id", 0))
    M_WC = _lookup_pose_matrix(
        pose_table,
        frame_id=frame_id,
        stride=int(pose_frame_stride),
        pattern=str(pose_frame_pattern),
        pose_key=str(pose_key),
    )
    if M_WC is None:
        M_WC = _sim3_matrix(keyframe.T_WC)
    A = M_WC[:3, :3]
    t = M_WC[:3, 3]
    Xw = (Xs @ A.T) + t[None, :]
    return Xw.astype(np.float32, copy=False), labels


def _compute_global_bounds(
    *,
    keyframes,
    stride: int,
    conf_threshold: float,
    pose_table: dict | None = None,
    pose_frame_stride: int = 1,
    pose_frame_pattern: str = "frame_{frame_id:06d}",
    pose_key: str = "aligned_pose",
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    mins = None
    maxs = None
    used = 0
    n_kf = int(len(keyframes))
    for kf_idx in range(n_kf):
        try:
            kf = keyframes[kf_idx]
        except Exception:
            continue
        Xw, _ = _extract_keyframe_points_world(
            keyframe=kf,
            stride=stride,
            conf_threshold=conf_threshold,
            pose_table=pose_table,
            pose_frame_stride=pose_frame_stride,
            pose_frame_pattern=pose_frame_pattern,
            pose_key=pose_key,
        )
        if Xw.size == 0:
            continue
        xyz_min = np.min(Xw, axis=0)
        xyz_max = np.max(Xw, axis=0)
        mins = xyz_min if mins is None else np.minimum(mins, xyz_min)
        maxs = xyz_max if maxs is None else np.maximum(maxs, xyz_max)
        used += 1
    return mins, maxs, used


def _voxel_linear_idx(ijk: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = dims
    return (ijk[:, 0] * (ny * nz) + ijk[:, 1] * nz + ijk[:, 2]).astype(np.int64, copy=False)


def save_global_esdf_snapshot(
    *,
    out_dir: str | Path,
    keyframes,
    voxel_m: float,
    stride: int,
    conf_threshold: float,
    frame_id: int,
    timestamp_s: float,
    scene_id: str = "",
    use_semantic: bool = True,
    padding_vox: int = 2,
    pose_json: str | Path | None = None,
    pose_key: str = "aligned_pose",
    pose_frame_stride: int = 1,
    pose_frame_pattern: str = "frame_{frame_id:06d}",
) -> Path | None:
    voxel_m = float(voxel_m)
    stride = int(max(1, stride))
    pose_table = _load_pose_table(pose_json)
    if pose_table is not None:
        print(
            "[PlanningTSDF] global ESDF using external camera poses:",
            f"path={pose_json}",
            f"key={pose_key}",
            f"stride={int(pose_frame_stride)}",
            f"pattern={pose_frame_pattern}",
        )
    mins, maxs, used = _compute_global_bounds(
        keyframes=keyframes,
        stride=stride,
        conf_threshold=conf_threshold,
        pose_table=pose_table,
        pose_frame_stride=pose_frame_stride,
        pose_frame_pattern=pose_frame_pattern,
        pose_key=pose_key,
    )
    if mins is None or maxs is None or used <= 0:
        print("[PlanningTSDF] global ESDF skipped: no valid global keyframe points")
        return None

    pad = float(max(0, int(padding_vox))) * voxel_m
    origin_w = np.floor((mins - pad) / voxel_m).astype(np.float32) * voxel_m
    upper_w = np.ceil((maxs + pad) / voxel_m).astype(np.float32) * voxel_m
    dims_xyz = np.maximum(1, np.round((upper_w - origin_w) / voxel_m).astype(np.int32) + 1)
    dims = (int(dims_xyz[0]), int(dims_xyz[1]), int(dims_xyz[2]))

    occ = np.zeros(dims, dtype=np.uint8)
    weight = np.zeros(dims, dtype=np.float32)
    sem_label = np.full(dims, -1, dtype=np.int32)
    sem_weight = np.zeros(dims, dtype=np.float32)

    n_kf = int(len(keyframes))
    for kf_idx in range(n_kf):
        try:
            kf = keyframes[kf_idx]
        except Exception:
            continue
        Xw, labels = _extract_keyframe_points_world(
            keyframe=kf,
            stride=stride,
            conf_threshold=conf_threshold,
            pose_table=pose_table,
            pose_frame_stride=pose_frame_stride,
            pose_frame_pattern=pose_frame_pattern,
            pose_key=pose_key,
        )
        if Xw.size == 0:
            continue
        ijk = np.rint((Xw - origin_w.reshape(1, 3)) / voxel_m).astype(np.int32)
        keep = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < dims[0]) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < dims[1]) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < dims[2])
        )
        if not np.any(keep):
            continue
        ijk = ijk[keep]
        labels = labels[keep]
        lin = _voxel_linear_idx(ijk, dims)

        lin_unique, counts = np.unique(lin, return_counts=True)
        occ.reshape(-1)[lin_unique] = 1
        weight.reshape(-1)[lin_unique] += counts.astype(np.float32, copy=False)

        if use_semantic:
            valid_lab = labels >= 0
            if np.any(valid_lab):
                lin_lab = lin[valid_lab]
                lab_vals = labels[valid_lab]
                order = np.argsort(lin_lab, kind="mergesort")
                lin_lab = lin_lab[order]
                lab_vals = lab_vals[order]
                first = np.r_[0, np.flatnonzero(lin_lab[1:] != lin_lab[:-1]) + 1]
                sem_label.reshape(-1)[lin_lab[first]] = lab_vals[first]
                sem_weight.reshape(-1)[lin_lab[first]] += 1.0

    occ_bool = occ.astype(bool, copy=False)
    d_free = ndi.distance_transform_edt(~occ_bool).astype(np.float32) * voxel_m
    d_occ = ndi.distance_transform_edt(occ_bool).astype(np.float32) * voxel_m
    esdf = d_free
    esdf[occ_bool] = -d_occ[occ_bool]
    tsdf = np.clip(esdf / max(voxel_m, 1e-6), -1.0, 1.0).astype(np.float32, copy=False)

    curr_T_WC = np.eye(4, dtype=np.float32)
    try:
        last_kf = keyframes[max(0, n_kf - 1)]
        curr_T_WC_ext = _lookup_pose_matrix(
            pose_table,
            frame_id=int(getattr(last_kf, "frame_id", 0)),
            stride=int(pose_frame_stride),
            pattern=str(pose_frame_pattern),
            pose_key=str(pose_key),
        )
        if curr_T_WC_ext is not None:
            curr_T_WC = curr_T_WC_ext.astype(np.float32, copy=False)
        else:
            curr_T_WC = _sim3_matrix(last_kf.T_WC).astype(np.float32, copy=False)
    except Exception:
        pass

    radius_m = 0.5 * float(np.max((np.asarray(dims, dtype=np.float32) - 1.0) * voxel_m))
    snapshot = EsdfSnapshot(
        schema="mast3r_esdf_snapshot_v1",
        frame_id=int(frame_id),
        timestamp_s=float(timestamp_s),
        scene_id=str(scene_id),
        radius_m=radius_m,
        voxel_m=voxel_m,
        dims=np.asarray(dims, dtype=np.int32),
        origin_w=np.asarray(origin_w, dtype=np.float32),
        curr_T_WC=np.asarray(curr_T_WC, dtype=np.float32).reshape(4, 4),
        esdf=np.asarray(esdf, dtype=np.float32),
        occ=np.asarray(occ, dtype=np.uint8),
        tsdf=np.asarray(tsdf, dtype=np.float32),
        weight=np.asarray(weight, dtype=np.float32),
        sem_label=np.asarray(sem_label, dtype=np.int32),
        sem_weight=np.asarray(sem_weight, dtype=np.float32),
        use_semantic=bool(use_semantic),
        obstacle_labels=np.asarray([], dtype=np.int32),
        w_min=0.0,
        sem_w_min=0.0,
        dilate_iters=0,
    )
    out_path = Path(out_dir) / "global_esdf_snapshot.npz"
    save_esdf_snapshot(snapshot, out_path)
    print(
        f"[PlanningTSDF] saved global ESDF snapshot: {out_path} "
        f"dims={dims} voxel_m={voxel_m:.4f} keyframes={used}"
    )
    return out_path
