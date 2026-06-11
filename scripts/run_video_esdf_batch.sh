#!/usr/bin/env bash
set -euo pipefail

VIDEO_DIR="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_ROOT="${2:-${REPO_ROOT}/logs/video_esdf_batch}"
CONFIG_PATH="${CONFIG_PATH:-config/base.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STREAM_IMG_SIZE="${STREAM_IMG_SIZE:-224}"
TSDF_RADIUS_M="${TSDF_RADIUS_M:-2.5}"
TSDF_VOXEL_M="${TSDF_VOXEL_M:-0.05}"
TSDF_SEMANTIC_BAND_M="${TSDF_SEMANTIC_BAND_M:-0.1}"
SAVE_AS="${SAVE_AS:-default}"

if [[ -z "${VIDEO_DIR}" ]]; then
  echo "Usage: $0 <video_dir> [out_root]" >&2
  exit 1
fi

if [[ ! -d "${VIDEO_DIR}" ]]; then
  echo "Video directory does not exist: ${VIDEO_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

shopt -s nullglob
videos=("${VIDEO_DIR}"/*.mp4 "${VIDEO_DIR}"/*.MP4 "${VIDEO_DIR}"/*.mov "${VIDEO_DIR}"/*.MOV)
shopt -u nullglob

if [[ ${#videos[@]} -eq 0 ]]; then
  echo "No video files found in: ${VIDEO_DIR}" >&2
  exit 1
fi

for video in "${videos[@]}"; do
  name="$(basename "${video}")"
  stem="${name%.*}"
  outdir="${OUT_ROOT}/${stem}"

  echo "============================================================"
  echo "VIDEO : ${video}"
  echo "OUT   : ${outdir}"
  echo "============================================================"

  mkdir -p "${outdir}"
  save_args=()
  if [[ "${SAVE_AS}" != "default" ]]; then
    save_args=(--save-as "${SAVE_AS}/${stem}")
  fi

  "${PYTHON_BIN}" "${REPO_ROOT}/main_semantic.py" \
    --dataset "${video}" \
    --config "${REPO_ROOT}/${CONFIG_PATH}" \
    "${save_args[@]}" \
    --no-viz \
    --stream_img_size "${STREAM_IMG_SIZE}" \
    --efficientvit_dataset ade20k \
    --enable_semantic_input \
    --use_semantic_in_geo \
    --use_stable_semantic_in_geo \
    --semantic_beta 0.2 \
    --planning_pointcloud_outdir "${outdir}" \
    --enable_planning_tsdf_publish \
    --planning_tsdf_radius_m "${TSDF_RADIUS_M}" \
    --planning_tsdf_voxel_m "${TSDF_VOXEL_M}" \
    --planning_tsdf_use_semantic \
    --planning_tsdf_semantic_band_m "${TSDF_SEMANTIC_BAND_M}"
done

echo "Done. Outputs under: ${OUT_ROOT}"
