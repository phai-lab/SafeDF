#!/usr/bin/env python3
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Iterable, Iterator

import cv2
import numpy as np


def _load_class_palette(name: str) -> np.ndarray | None:
    name = str(name).lower().strip()
    if name in {"", "hash", "none"}:
        return None

    efficientvit_root = pathlib.Path(__file__).resolve().parents[1] / "efficientvit"
    if str(efficientvit_root) not in sys.path:
        sys.path.append(str(efficientvit_root))

    from applications.efficientvit_seg.eval_efficientvit_seg_model import (  # type: ignore
        ADE20KDataset,
        CityscapesDataset,
    )

    if name in {"ade", "ade20k"}:
        return np.asarray(ADE20KDataset.class_colors, dtype=np.uint8)
    if name in {"cityscape", "cityscapes", "cs"}:
        return np.asarray(CityscapesDataset.class_colors, dtype=np.uint8)
    raise ValueError(f"Unsupported palette: {name!r}")


def _hash_label_colors(label: np.ndarray) -> np.ndarray:
    x = label.astype(np.uint64, copy=False)
    r = (x * np.uint64(37) + np.uint64(17)) & np.uint64(255)
    g = (x * np.uint64(67) + np.uint64(29)) & np.uint64(255)
    b = (x * np.uint64(97) + np.uint64(43)) & np.uint64(255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _colorize_label_code_frame(sem_bgr: np.ndarray, palette_rgb: np.ndarray | None) -> np.ndarray:
    sem_rgb = cv2.cvtColor(sem_bgr, cv2.COLOR_BGR2RGB)
    label = (
        sem_rgb[..., 0].astype(np.int64) << 16
        | sem_rgb[..., 1].astype(np.int64) << 8
        | sem_rgb[..., 2].astype(np.int64)
    )

    color_rgb = _hash_label_colors(label)
    if palette_rgb is not None:
        valid = (label >= 0) & (label < int(len(palette_rgb)))
        if np.any(valid):
            color_rgb[valid] = palette_rgb[label[valid]]
    return cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)


def _sorted_rgb_paths(rgb_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = [p for p in rgb_dir.glob("*.png") if p.is_file()]

    def key(p: pathlib.Path):
        try:
            return float(p.stem)
        except Exception:
            return p.name

    return sorted(paths, key=key)


def _sorted_sem_paths(sem_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = [p for p in sem_dir.glob("*.png") if p.is_file()]

    def key(p: pathlib.Path):
        m = re.search(r"(\d+)$", p.stem)
        return int(m.group(1)) if m else p.name

    return sorted(paths, key=key)


def _estimate_fps_from_rgb_filenames(rgb_paths: list[pathlib.Path]) -> float | None:
    if len(rgb_paths) < 2:
        return None
    ts: list[float] = []
    for p in rgb_paths:
        try:
            ts.append(float(p.stem))
        except Exception:
            return None
    diffs = np.diff(np.asarray(ts, dtype=np.float64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    dt = float(np.median(diffs))
    if dt <= 0:
        return None
    fps = 1.0 / dt
    if not np.isfinite(fps):
        return None
    return float(np.clip(fps, 1.0, 120.0))


def _iter_blended_frames(
    rgb_paths: list[pathlib.Path],
    sem_paths: list[pathlib.Path],
    *,
    rgb_offset: int,
    sem_offset: int,
    alpha: float,
    target: str,
    limit: int | None,
    sem_mode: str,
    palette_rgb: np.ndarray | None,
) -> tuple[tuple[int, int], Iterator[np.ndarray]]:
    if rgb_offset < 0 or sem_offset < 0:
        raise ValueError("Offsets must be >= 0.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    if target not in {"rgb", "sem"}:
        raise ValueError("target must be one of: rgb, sem")

    rgb_paths = rgb_paths[rgb_offset:]
    sem_paths = sem_paths[sem_offset:]
    n = min(len(rgb_paths), len(sem_paths))
    if limit is not None:
        n = min(n, limit)
    rgb_paths = rgb_paths[:n]
    sem_paths = sem_paths[:n]

    if n == 0:
        raise RuntimeError("No paired frames found after applying offsets/limit.")

    first_rgb = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    first_sem = cv2.imread(str(sem_paths[0]), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise FileNotFoundError(f"Failed to read RGB frame: {rgb_paths[0]}")
    if first_sem is None:
        raise FileNotFoundError(f"Failed to read semantic frame: {sem_paths[0]}")

    if target == "rgb":
        out_h, out_w = first_rgb.shape[:2]
    else:
        out_h, out_w = first_sem.shape[:2]

    def gen() -> Iterator[np.ndarray]:
        for i, (rp, sp) in enumerate(zip(rgb_paths, sem_paths, strict=True), start=1):
            rgb = cv2.imread(str(rp), cv2.IMREAD_COLOR)
            sem = cv2.imread(str(sp), cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(f"Failed to read RGB frame: {rp}")
            if sem is None:
                raise FileNotFoundError(f"Failed to read semantic frame: {sp}")
            if sem_mode == "label-code":
                sem = _colorize_label_code_frame(sem, palette_rgb)

            if target == "rgb":
                if sem.shape[:2] != rgb.shape[:2]:
                    sem = cv2.resize(
                        sem,
                        (rgb.shape[1], rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                blended = cv2.addWeighted(rgb, 1.0 - alpha, sem, alpha, 0.0)
            else:
                if rgb.shape[:2] != sem.shape[:2]:
                    rgb = cv2.resize(
                        rgb,
                        (sem.shape[1], sem.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                blended = cv2.addWeighted(rgb, 1.0 - alpha, sem, alpha, 0.0)

            if blended.shape[0] != out_h or blended.shape[1] != out_w:
                blended = cv2.resize(blended, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if i % 50 == 0:
                print(f"[blend] {i}/{n}", flush=True)
            yield blended

    return (out_w, out_h), gen()


def _write_video_ffmpeg(
    frames: Iterable[np.ndarray],
    *,
    size: tuple[int, int],
    fps: float,
    output_path: pathlib.Path,
    crf: int,
    preset: str,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(crf),
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            if frame.shape[0] != h or frame.shape[1] != w or frame.shape[2] != 3:
                raise ValueError(
                    f"Unexpected frame shape {frame.shape}, expected ({h}, {w}, 3)."
                )
            proc.stdin.write(frame.tobytes())
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret}")


def _write_video_opencv(
    frames: Iterable[np.ndarray],
    *,
    size: tuple[int, int],
    fps: float,
    output_path: pathlib.Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")
    try:
        for frame in frames:
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Blend RGB frames with saved semantic frames at 50/50 opacity and encode a video."
        )
    )
    parser.add_argument(
        "--rgb-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing RGB frames (*.png).",
    )
    parser.add_argument(
        "--sem-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing semantic frames (*.png), e.g. 000001.png ...",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("blended_overlay.mp4"),
        help="Output video path.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Opacity of semantic overlay in [0,1] (0.5 = 50/50).",
    )
    parser.add_argument(
        "--fps",
        type=str,
        default="auto",
        help='FPS for the output video (float) or "auto" (estimate from RGB filenames).',
    )
    parser.add_argument(
        "--target",
        choices=["rgb", "sem"],
        default="sem",
        help="Output resolution target: match RGB size or semantic size.",
    )
    parser.add_argument(
        "--sem-mode",
        choices=["rgb", "label-code"],
        default="rgb",
        help="How to interpret semantic PNGs: RGB mask, or packed label-code RGB.",
    )
    parser.add_argument(
        "--palette",
        choices=["ade20k", "cityscapes", "hash"],
        default="ade20k",
        help="Class color palette used when --sem-mode label-code.",
    )
    parser.add_argument(
        "--rgb-offset",
        type=int,
        default=0,
        help="Skip first N RGB frames before pairing.",
    )
    parser.add_argument(
        "--sem-offset",
        type=int,
        default=0,
        help="Skip first N semantic frames before pairing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only encode first N paired frames (for quick tests).",
    )
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="Force OpenCV VideoWriter instead of ffmpeg/libx264.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="ffmpeg x264 CRF (lower = higher quality, bigger file).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="veryfast",
        help="ffmpeg x264 preset (e.g., ultrafast, veryfast, medium, slow).",
    )
    args = parser.parse_args()

    if not args.rgb_dir.is_dir():
        raise FileNotFoundError(f"rgb dir not found: {args.rgb_dir}")
    if not args.sem_dir.is_dir():
        raise FileNotFoundError(f"sem dir not found: {args.sem_dir}")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")

    rgb_paths = _sorted_rgb_paths(args.rgb_dir)
    sem_paths = _sorted_sem_paths(args.sem_dir)
    if not rgb_paths:
        raise RuntimeError(f"No *.png found in {args.rgb_dir}")
    if not sem_paths:
        raise RuntimeError(f"No *.png found in {args.sem_dir}")

    if args.fps == "auto":
        fps = _estimate_fps_from_rgb_filenames(rgb_paths) or 30.0
    else:
        fps = float(args.fps)
        if fps <= 0:
            raise ValueError("fps must be > 0.")

    palette_rgb = _load_class_palette(args.palette) if args.sem_mode == "label-code" else None

    print(
        f"[info] rgb={len(rgb_paths)} sem={len(sem_paths)} fps={fps:.3f} alpha={args.alpha:.2f} "
        f"target={args.target} sem_mode={args.sem_mode} palette={args.palette}",
        flush=True,
    )

    size, frames_iter = _iter_blended_frames(
        rgb_paths,
        sem_paths,
        rgb_offset=args.rgb_offset,
        sem_offset=args.sem_offset,
        alpha=args.alpha,
        target=args.target,
        limit=args.limit,
        sem_mode=args.sem_mode,
        palette_rgb=palette_rgb,
    )
    print(f"[info] output_size={size[0]}x{size[1]} output={args.output}", flush=True)

    if not args.no_ffmpeg:
        try:
            _write_video_ffmpeg(
                frames_iter,
                size=size,
                fps=fps,
                output_path=args.output,
                crf=args.crf,
                preset=args.preset,
            )
            print(f"[done] Saved: {args.output}")
            return
        except Exception as e:
            print(f"[warn] ffmpeg encode failed ({e}); falling back to OpenCV mp4v.", file=sys.stderr)

    size, frames_iter = _iter_blended_frames(
        rgb_paths,
        sem_paths,
        rgb_offset=args.rgb_offset,
        sem_offset=args.sem_offset,
        alpha=args.alpha,
        target=args.target,
        limit=args.limit,
        sem_mode=args.sem_mode,
        palette_rgb=palette_rgb,
    )
    _write_video_opencv(frames_iter, size=size, fps=fps, output_path=args.output)
    print(f"[done] Saved: {args.output}")


if __name__ == "__main__":
    main()
