#!/usr/bin/env python3
"""Recompute the paper simulation tables from raw rollout JSON files.

This is the real metric recomputation path. It does not read the final paper
CSV tables. Given the saved controller rollouts and ScanNet++ meshes, it
recomputes per-trajectory progress, matched-progress mesh clearance, and
matched-progress collision rates, then aggregates Table I and Table III.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

SCENES = ["281bc17764", "689fec23d7", "7cd2ac43b4", "8a20d62ac0", "b26e64c4b0", "bc03d88fc3"]
METHOD_ORDER = ["SaferSplat", "Ours (ESDF)", "Ours (Semantic ESDF)"]
RISK_ORDER = ["high", "mid", "low"]
RISK_LABEL = {"high": "High", "mid": "Mid", "low": "Low"}

EXPECTED_MAIN = {
    "SaferSplat": ("0.149", "0.360", "0.101", "0.216"),
    "Ours (ESDF)": ("0.002", "0.697", "0.106", "0.218"),
    "Ours (Semantic ESDF)": ("0.002", "0.603", "0.108", "0.200"),
}
EXPECTED_RISK = {
    ("high", "Ours (ESDF)"): ("0.669", "0.128", "0.146"),
    ("mid", "Ours (ESDF)"): ("0.777", "0.092", "0.248"),
    ("low", "Ours (ESDF)"): ("0.647", "0.098", "0.261"),
    ("high", "Ours (Semantic ESDF)"): ("0.534", "0.134", "0.116"),
    ("mid", "Ours (Semantic ESDF)"): ("0.631", "0.093", "0.224"),
    ("low", "Ours (Semantic ESDF)"): ("0.647", "0.098", "0.261"),
}


@dataclass
class Rollout:
    scene: str
    traj_id: int
    method: str
    risk: str | None
    start: np.ndarray
    goal: np.ndarray
    traj_xyz: np.ndarray
    total_time: np.ndarray

    @property
    def progress_series(self) -> np.ndarray:
        denom = float(np.linalg.norm(self.goal - self.start))
        if denom < 1e-12 or self.traj_xyz.size == 0:
            return np.full((len(self.traj_xyz),), np.nan, dtype=np.float64)
        return np.linalg.norm(self.traj_xyz - self.start[None, :], axis=1) / denom

    @property
    def progress(self) -> float:
        series = self.progress_series
        return float(np.nanmax(series)) if series.size else math.nan

    @property
    def mean_time(self) -> float:
        return float(np.mean(self.total_time)) if self.total_time.size else math.nan


def fmt(x: float) -> str:
    return f"{x:.3f}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def rollout_path(method: str, scene: str, gs_dir: Path, esdf_dir: Path) -> Path:
    if method == "SaferSplat":
        return gs_dir / f"scannetpp_{scene}_cbf_goal_meshscale_eqtime_matchbalanced_50x3_clear20.json"
    if method == "Ours (ESDF)":
        return esdf_dir / f"scannetpp_{scene}_esdf_goal_meshscale_matchbalanced_50x3_clear20_baseline.json"
    if method == "Ours (Semantic ESDF)":
        return esdf_dir / f"scannetpp_{scene}_esdf_goal_meshscale_matchbalanced_50x3_clear20_targetrisk.json"
    raise KeyError(method)


def load_rollouts(scene: str, method: str, gs_dir: Path, esdf_dir: Path) -> Dict[int, Rollout]:
    path = rollout_path(method, scene, gs_dir, esdf_dir)
    obj = read_json(path)
    out: Dict[int, Rollout] = {}
    for fallback_id, row in enumerate(obj.get("total_data", [])):
        traj = np.asarray(row.get("traj", []), dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] < 3:
            traj_xyz = np.zeros((0, 3), dtype=np.float64)
        else:
            traj_xyz = traj[:, :3]
        traj_id = int(row.get("traj_id", fallback_id))
        out[traj_id] = Rollout(
            scene=scene,
            traj_id=traj_id,
            method=method,
            risk=row.get("risk_class"),
            start=np.asarray(row.get("start", traj_xyz[0] if len(traj_xyz) else [np.nan, np.nan, np.nan]), dtype=np.float64),
            goal=np.asarray(row.get("goal", traj_xyz[-1] if len(traj_xyz) else [np.nan, np.nan, np.nan]), dtype=np.float64),
            traj_xyz=traj_xyz,
            total_time=np.asarray(row.get("total_time", []), dtype=np.float64),
        )
    return out


def build_scene(mesh_path: Path):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh: {mesh_path}")
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    return scene


def compute_distance(scene, points_xyz: np.ndarray) -> np.ndarray:
    import open3d as o3d

    if points_xyz.size == 0:
        return np.zeros((0,), dtype=np.float64)
    pts = o3d.core.Tensor(points_xyz.astype(np.float32))
    return scene.compute_distance(pts).numpy().astype(np.float64)


def prefix_until_progress(rollout: Rollout, matched_progress: float) -> np.ndarray:
    progress = rollout.progress_series
    if progress.size == 0 or not np.isfinite(matched_progress):
        return rollout.traj_xyz[:0]
    idx = np.flatnonzero(progress >= matched_progress - 1e-12)
    end = int(idx[0]) if idx.size else len(progress) - 1
    return rollout.traj_xyz[: end + 1]


def min_clearance_at_progress(scene, rollout: Rollout, matched_progress: float, radius: float) -> float:
    prefix = prefix_until_progress(rollout, matched_progress)
    if prefix.size == 0:
        return math.nan
    clearance = compute_distance(scene, prefix) - float(radius)
    return float(np.min(clearance)) if clearance.size else math.nan


def mean(xs: Iterable[float]) -> float:
    arr = np.asarray([x for x in xs if np.isfinite(x)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else math.nan


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def validate(table_i: Sequence[Sequence[str]], table_iii: Sequence[Sequence[str]]) -> None:
    errors: List[str] = []
    for row in table_i:
        actual = tuple(row[1:5])
        expected = EXPECTED_MAIN[row[0]]
        if actual != expected:
            errors.append(f"Table I mismatch for {row[0]}: {actual} != {expected}")
    for row in table_iii:
        risk = row[0].lower()
        esdf = tuple(row[2:5])
        sem = tuple(row[5:8])
        if esdf != EXPECTED_RISK[(risk, "Ours (ESDF)")]:
            errors.append(f"Table III mismatch for {risk}/ESDF: {esdf} != {EXPECTED_RISK[(risk, 'Ours (ESDF)')]}")
        if sem != EXPECTED_RISK[(risk, "Ours (Semantic ESDF)")]:
            errors.append(f"Table III mismatch for {risk}/Semantic: {sem} != {EXPECTED_RISK[(risk, 'Ours (Semantic ESDF)')]}")
    if errors:
        raise SystemExit("\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gs-rollout-dir", type=Path, required=True)
    parser.add_argument("--esdf-rollout-dir", type=Path, required=True)
    parser.add_argument("--scannetpp-data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("repro/recomputed_tables"))
    parser.add_argument("--radius", type=float, default=0.03)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    all_records: List[dict] = []
    risk_records: List[dict] = []

    for scene_id in SCENES:
        mesh_path = args.scannetpp_data_root / scene_id / "scans" / "mesh_aligned_0.05.ply"
        scene = build_scene(mesh_path)
        scene_rollouts = {
            method: load_rollouts(scene_id, method, args.gs_rollout_dir, args.esdf_rollout_dir)
            for method in METHOD_ORDER
        }
        common_ids = sorted(set.intersection(*(set(v.keys()) for v in scene_rollouts.values())))
        for traj_id in common_ids:
            trio = {method: scene_rollouts[method][traj_id] for method in METHOD_ORDER}
            matched_progress = min(r.progress for r in trio.values())
            risk = trio["Ours (Semantic ESDF)"].risk or trio["Ours (ESDF)"].risk or "unknown"
            for method, rollout in trio.items():
                mp_clearance = min_clearance_at_progress(scene, rollout, matched_progress, args.radius)
                record = {
                    "scene": scene_id,
                    "traj_id": traj_id,
                    "risk": risk,
                    "method": method,
                    "progress": rollout.progress,
                    "mean_time_s": rollout.mean_time,
                    "matched_progress": matched_progress,
                    "mp_clearance_m": mp_clearance,
                    "mp_collision": float(mp_clearance < 0.0) if np.isfinite(mp_clearance) else math.nan,
                }
                all_records.append(record)
                if method in {"Ours (ESDF)", "Ours (Semantic ESDF)"}:
                    risk_records.append(record.copy())

    main_rows: List[Tuple[str, str, str, str, str]] = []
    main_csv_rows: List[dict] = []
    for method in METHOD_ORDER:
        rows = [r for r in all_records if r["method"] == method]
        time_s = mean(r["mean_time_s"] for r in rows)
        progress = mean(r["progress"] for r in rows)
        mp_clear = mean(r["mp_clearance_m"] for r in rows)
        mp_coll = mean(r["mp_collision"] for r in rows)
        main_rows.append((method, fmt(time_s), fmt(progress), fmt(mp_clear), fmt(mp_coll)))
        main_csv_rows.append(
            {
                "method": method,
                "n_total": len(rows),
                "computation_time_mean_s": time_s,
                "progress_to_goal_mean": progress,
                "matched_progress_min_clearance_mean_m": mp_clear,
                "matched_progress_collision_rate": mp_coll,
            }
        )

    risk_rows: List[Tuple[str, str, str, str, str, str, str, str]] = []
    risk_csv_rows: List[dict] = []
    for risk in RISK_ORDER:
        esdf_rows = [r for r in risk_records if r["risk"] == risk and r["method"] == "Ours (ESDF)"]
        sem_rows = [r for r in risk_records if r["risk"] == risk and r["method"] == "Ours (Semantic ESDF)"]
        n = len(esdf_rows)
        e_prog, e_clear, e_coll = mean(r["progress"] for r in esdf_rows), mean(r["mp_clearance_m"] for r in esdf_rows), mean(r["mp_collision"] for r in esdf_rows)
        s_prog, s_clear, s_coll = mean(r["progress"] for r in sem_rows), mean(r["mp_clearance_m"] for r in sem_rows), mean(r["mp_collision"] for r in sem_rows)
        risk_rows.append((RISK_LABEL[risk], str(n), fmt(e_prog), fmt(e_clear), fmt(e_coll), fmt(s_prog), fmt(s_clear), fmt(s_coll)))
        for method, rows, prog, clear, coll in [
            ("Ours (ESDF)", esdf_rows, e_prog, e_clear, e_coll),
            ("Ours (Semantic ESDF)", sem_rows, s_prog, s_clear, s_coll),
        ]:
            risk_csv_rows.append(
                {
                    "method": method,
                    "risk": risk,
                    "n": len(rows),
                    "progress_to_goal_mean": prog,
                    "matched_progress_min_clearance_mean_m": clear,
                    "matched_progress_collision_rate": coll,
                }
            )

    print("Table I: recomputed from raw rollout JSON + meshes")
    print(markdown_table(["Method", "Time (s) ↓", "Progress ↑", "MP Clearance (m) ↑", "MP Collision ↓"], main_rows))
    print()
    print("Table III: recomputed risk-group ablation")
    print(
        markdown_table(
            ["Risk", "N", "ESDF Progress ↑", "ESDF MP Clear. ↑", "ESDF MP Coll. ↓", "Sem Progress ↑", "Sem MP Clear. ↑", "Sem MP Coll. ↓"],
            risk_rows,
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "recomputed_table_i.csv", ["method", "n_total", "computation_time_mean_s", "progress_to_goal_mean", "matched_progress_min_clearance_mean_m", "matched_progress_collision_rate"], main_csv_rows)
    write_csv(args.out_dir / "recomputed_table_iii.csv", ["method", "risk", "n", "progress_to_goal_mean", "matched_progress_min_clearance_mean_m", "matched_progress_collision_rate"], risk_csv_rows)
    write_csv(args.out_dir / "recomputed_matched_progress_rows.csv", ["scene", "traj_id", "risk", "method", "progress", "mean_time_s", "matched_progress", "mp_clearance_m", "mp_collision"], all_records)

    if not args.no_validate:
        validate(main_rows, risk_rows)
        print("\nValidation: recomputed rounded values match the paper tables.")
    print(f"Saved recomputed CSVs: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
