#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DATA_ROOT_DEFAULT = Path("data/scannetpp")
MAPPING_CSV_DEFAULT = Path("resources/mesh_to_ade20k_all.csv")
OUTDIR_DEFAULT = Path("outputs/semantic_risk_groups_balanced")

DEFAULT_EXCLUDED_ADE20K = {
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "sidewalk",
    "earth",
    "mountain",
    "water",
    "house",
    "sea",
    "field",
    "rock",
    "column",
    "path",
    "stairs",
    "stairway",
    "river",
    "bridge",
    "fence",
    "railing",
    "base",
    "pole",
    "land",
    "streetlight",
    "tower",
    "awning",
    "windowpane",
    "door",
    "screen door",
    "runway",
    "traffic light",
}


def _parse_csv_set(text: str) -> set[str]:
    return {x.strip().lower() for x in str(text).split(",") if x.strip()}


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    vals = sorted(float(x) for x in values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _load_mapping(mapping_csv: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    with mapping_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = str(row["raw_label"]).strip().lower()
            mapping[raw] = {
                "ade20k_class": str(row["ade20k_class"]).strip(),
                "ade20k_id": str(row["ade20k_id"]).strip(),
                "mapping_status": str(row["mapping_status"]).strip(),
            }
    return mapping


def _safe_obb_stats(group: dict) -> tuple[float, float, float] | None:
    obb = group.get("obb", {})
    if not isinstance(obb, dict):
        return None
    axes = obb.get("axesLengths", [])
    if not isinstance(axes, list) or len(axes) != 3:
        return None
    try:
        ax = [float(x) for x in axes]
    except Exception:
        return None
    if not all(math.isfinite(x) and x > 1e-6 for x in ax):
        return None
    diag = float(math.sqrt(ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2))
    volume = float(ax[0] * ax[1] * ax[2])
    xy_radius = float(0.5 * max(ax[0], ax[1]))
    return diag, volume, xy_radius


def _load_scene_groups(scene_dir: Path) -> list[dict]:
    anno_path = scene_dir / "scans" / "segments_anno.json"
    if not anno_path.exists():
        return []
    data = json.loads(anno_path.read_text())
    groups = data.get("segGroups", [])
    return groups if isinstance(groups, list) else []


def _assign_scene_specific_tiers(ade_rows: list[dict]) -> list[dict]:
    rows = sorted(
        ade_rows,
        key=lambda r: (-float(r["median_diag_m"]), -int(r["object_count"]), str(r["ade20k_class"])),
    )
    tier_orders = [
        ("high", "mid", "low"),
        ("mid", "low", "high"),
        ("low", "high", "mid"),
    ]
    out: list[dict] = []
    for chunk_idx, start in enumerate(range(0, len(rows), 3)):
        chunk = rows[start : start + 3]
        chunk = sorted(
            chunk,
            key=lambda r: (-int(r["object_count"]), -float(r["median_diag_m"]), str(r["ade20k_class"])),
        )
        order = tier_orders[chunk_idx % len(tier_orders)]
        for i, row in enumerate(chunk):
            cur = dict(row)
            cur["risk_tier"] = order[i]
            cur["size_chunk"] = int(chunk_idx)
            out.append(cur)
    return out


def _build_scene_config(scene_id: str, scene_dir: Path, mapping: dict[str, dict[str, str]], excluded_ade: set[str]) -> dict:
    groups = _load_scene_groups(scene_dir)
    ade_stats: dict[str, dict[str, object]] = {}
    raw_labels_by_ade: dict[str, set[str]] = defaultdict(set)

    for group in groups:
        if not isinstance(group, dict):
            continue
        raw_label = str(group.get("label", "")).strip().lower()
        if not raw_label:
            continue
        meta = mapping.get(raw_label)
        if meta is None:
            continue
        ade_name = meta["ade20k_class"].strip()
        status = meta["mapping_status"]
        if status in {"ignored", "unmapped"} or not ade_name or ade_name.lower() in excluded_ade:
            continue
        obb_stats = _safe_obb_stats(group)
        if obb_stats is None:
            continue
        diag, volume, xy_radius = obb_stats
        ade_row = ade_stats.setdefault(
            ade_name,
            {
                "ade20k_class": ade_name,
                "ade20k_id": int(meta["ade20k_id"]) if meta["ade20k_id"] else -1,
                "object_count": 0,
                "diag_values": [],
                "volume_values": [],
                "xy_radius_values": [],
            },
        )
        ade_row["object_count"] = int(ade_row["object_count"]) + 1
        ade_row["diag_values"].append(diag)
        ade_row["volume_values"].append(volume)
        ade_row["xy_radius_values"].append(xy_radius)
        raw_labels_by_ade[ade_name].add(raw_label)

    ade_rows: list[dict] = []
    for ade_name, row in ade_stats.items():
        ade_rows.append(
            {
                "ade20k_class": ade_name,
                "ade20k_id": int(row["ade20k_id"]),
                "object_count": int(row["object_count"]),
                "median_diag_m": _median(row["diag_values"]),
                "median_volume_m3": _median(row["volume_values"]),
                "median_xy_radius_m": _median(row["xy_radius_values"]),
                "raw_mesh_labels": sorted(raw_labels_by_ade[ade_name]),
            }
        )

    balanced_rows = _assign_scene_specific_tiers(ade_rows)

    config = {
        "strategy": "scene_specific_size_balanced_cyclic_by_ade_median_diag",
        "scene_id": scene_id,
        "notes": (
            f"Non-structural ADE classes present in {scene_id} were sorted by scene-local median "
            "OBB diagonal and assigned cyclically to high/mid/low to reduce size bias while keeping "
            "ADE-tier semantics aligned with ESDF inflation."
        ),
        "ade20k_by_risk": {
            tier: [int(row["ade20k_id"]) for row in balanced_rows if row["risk_tier"] == tier]
            for tier in ("high", "mid", "low")
        },
        "ade20k_names_by_risk": {
            tier: [str(row["ade20k_class"]) for row in balanced_rows if row["risk_tier"] == tier]
            for tier in ("high", "mid", "low")
        },
        "mesh_labels_by_risk": {
            tier: sorted(
                {
                    raw
                    for row in balanced_rows
                    if row["risk_tier"] == tier
                    for raw in row["raw_mesh_labels"]
                }
            )
            for tier in ("high", "mid", "low")
        },
    }
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Export scene-specific semantic risk config jsons.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--mapping-csv", type=Path, default=MAPPING_CSV_DEFAULT)
    parser.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT)
    parser.add_argument("--scene-id", action="append", dest="scene_ids", help="Scene id to process. Repeatable.")
    parser.add_argument(
        "--exclude-ade20k",
        type=str,
        default=",".join(sorted(DEFAULT_EXCLUDED_ADE20K)),
        help="Comma-separated ADE20K classes to exclude from object-centric risk grouping.",
    )
    args = parser.parse_args()

    mapping = _load_mapping(args.mapping_csv)
    excluded_ade = _parse_csv_set(args.exclude_ade20k)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.scene_ids:
        scene_ids = list(dict.fromkeys(args.scene_ids))
    else:
        scene_ids = sorted(p.name for p in args.data_root.iterdir() if p.is_dir() and len(p.name) == 10)

    manifest: list[dict[str, object]] = []
    for scene_id in scene_ids:
        scene_dir = args.data_root / scene_id
        if not scene_dir.exists():
            print(f"[skip] missing scene dir: {scene_dir}")
            continue
        config = _build_scene_config(scene_id, scene_dir, mapping, excluded_ade)
        out_json = args.outdir / f"{scene_id}_scene_specific_risk.json"
        out_json.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "scene_id": scene_id,
                "num_high_ade": len(config["ade20k_by_risk"]["high"]),
                "num_mid_ade": len(config["ade20k_by_risk"]["mid"]),
                "num_low_ade": len(config["ade20k_by_risk"]["low"]),
                "num_high_raw": len(config["mesh_labels_by_risk"]["high"]),
                "num_mid_raw": len(config["mesh_labels_by_risk"]["mid"]),
                "num_low_raw": len(config["mesh_labels_by_risk"]["low"]),
                "out_json": str(out_json),
            }
        )
        print(f"[ok] {scene_id}: {out_json}")

    manifest_csv = args.outdir / "scene_specific_risk_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_id",
                "num_high_ade",
                "num_mid_ade",
                "num_low_ade",
                "num_high_raw",
                "num_mid_raw",
                "num_low_raw",
                "out_json",
            ],
        )
        writer.writeheader()
        for row in manifest:
            writer.writerow(row)
    print(f"[ok] manifest: {manifest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
