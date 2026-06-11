#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path


# Edit these defaults to control which 3 videos get stacked when you run with no args.
DEFAULT_INPUTS = [
    Path("logs/desk_rgb_sem_blend_640x480.mp4"),
    Path("logs/desk_rgb_sem_blend_pc_640x480.mp4"),
    Path("logs/desk_rgb_sem_blend_pc_momen_640x480.mp4"),
]

# Short titles shown on the top-left of each panel (left -> right).
DEFAULT_LABELS = ["Original", "SimBlend", "SemCache"]

DEFAULT_OUTPUT = Path("logs/desk_3up.mp4")
DEFAULT_TILE_W = 640
DEFAULT_TILE_H = 480
DEFAULT_FPS = 30
DEFAULT_BITRATE = "12M"


def _detect_fontfile() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def _drawtext_filter(label: str, fontfile: str | None) -> str:
    # Note: drawtext needs escaping for ':' and '\''. Keep labels simple (letters/numbers/_/-) to avoid surprises.
    safe_label = label.replace(":", "\\:").replace("'", "\\'")
    parts = [
        "drawtext=",
        f"text='{safe_label}':",
        "x=12:y=12:",
        "fontsize=28:",
        "fontcolor=white:",
        "box=1:boxcolor=black@0.55:boxborderw=10",
    ]
    if fontfile:
        parts.insert(1, f"fontfile={fontfile}:")
    return "".join(parts)


def build_filter(
    *,
    tile_w: int,
    tile_h: int,
    fps: int,
    labels: list[str],
    fontfile: str | None,
) -> str:
    if len(labels) != 3:
        raise ValueError("labels must have exactly 3 items")

    per_stream = []
    for idx, label in enumerate(labels):
        per_stream.append(
            f"[{idx}:v]"
            f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease,"
            f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,"
            f"fps={fps},"
            f"{_drawtext_filter(label, fontfile)}"
            f"[v{idx}]"
        )

    return ";".join(per_stream) + ";[v0][v1][v2]hstack=inputs=3:shortest=1[v]"


def run(cmd: list[str]) -> None:
    print("+ " + shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def try_run(cmd: list[str]) -> bool:
    print("+ " + shlex.join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _ffmpeg_capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)


def _ffmpeg_has_encoder(ffmpeg: str, encoder: str) -> bool:
    try:
        out = _ffmpeg_capture([ffmpeg, "-hide_banner", "-encoders"])
    except Exception:
        return False
    return f" {encoder} " in out


def _ffmpeg_encoder_help(ffmpeg: str, encoder: str) -> str:
    try:
        return _ffmpeg_capture([ffmpeg, "-hide_banner", "-h", f"encoder={encoder}"])
    except Exception:
        return ""


def _encoder_supports_option(encoder_help: str, option: str) -> bool:
    # Small heuristic; robust enough for deciding whether to pass -preset/-crf.
    return f"{option} " in encoder_help or f"{option}:" in encoder_help


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Horizontally stack 3 videos into one MP4 (with labels).",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="3 input videos (if omitted, uses DEFAULT_INPUTS in the script)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"output mp4 (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--labels",
        nargs=3,
        default=DEFAULT_LABELS,
        help="3 labels for left/mid/right panels",
    )
    parser.add_argument("--tile-w", type=int, default=DEFAULT_TILE_W)
    parser.add_argument("--tile-h", type=int, default=DEFAULT_TILE_H)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--codec",
        default="auto",
        help="output video codec (default: auto -> libx264 if available else mpeg4)",
    )
    parser.add_argument("--preset", default="veryfast", help="x264 preset (ignored if unsupported)")
    parser.add_argument("--crf", type=int, default=18, help="x264 CRF (ignored if unsupported)")
    parser.add_argument(
        "--bitrate",
        default=DEFAULT_BITRATE,
        help=f"fallback bitrate if CRF/preset unsupported (default: {DEFAULT_BITRATE})",
    )
    parser.add_argument("--verbose", action="store_true", help="show ffmpeg output")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found in PATH")

    inputs = [Path(p) for p in args.inputs] if args.inputs else list(DEFAULT_INPUTS)
    if len(inputs) != 3:
        raise SystemExit(f"expected exactly 3 inputs, got {len(inputs)}")
    for p in inputs:
        if not p.exists():
            raise SystemExit(f"input not found: {p}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fontfile = _detect_fontfile()
    filter_complex = build_filter(
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        fps=args.fps,
        labels=list(args.labels),
        fontfile=fontfile,
    )

    codec = args.codec
    if codec == "auto":
        codec = "libx264" if _ffmpeg_has_encoder(ffmpeg, "libx264") else "mpeg4"
    if codec == "libx264" and not _ffmpeg_has_encoder(ffmpeg, "libx264"):
        raise SystemExit("ffmpeg does not have libx264; try --codec mpeg4")

    encoder_help = _ffmpeg_encoder_help(ffmpeg, codec) if codec == "libx264" else ""
    supports_preset = _encoder_supports_option(encoder_help, "preset")
    supports_crf = _encoder_supports_option(encoder_help, "crf")

    base_cmd = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "info" if args.verbose else "warning",
        "-i",
        str(inputs[0]),
        "-i",
        str(inputs[1]),
        "-i",
        str(inputs[2]),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-an",
    ]

    tail_cmd = [
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]

    def make_cmd(selected_codec: str, encode_opts: list[str]) -> list[str]:
        return base_cmd + ["-c:v", selected_codec] + encode_opts + tail_cmd

    tried = []
    if codec == "libx264":
        x264_opts: list[str] = []
        if supports_preset:
            x264_opts += ["-preset", str(args.preset)]
        if supports_crf:
            x264_opts += ["-crf", str(args.crf)]
        if not x264_opts:
            x264_opts = ["-b:v", str(args.bitrate)]

        cmd1 = make_cmd("libx264", x264_opts)
        tried.append(cmd1)
        ok = try_run(cmd1)
        if not ok:
            cmd2 = make_cmd("libx264", ["-b:v", str(args.bitrate)])
            tried.append(cmd2)
            ok = try_run(cmd2)
        if not ok and args.codec == "auto":
            cmd3 = make_cmd("mpeg4", ["-q:v", "3"])
            tried.append(cmd3)
            ok = try_run(cmd3)
        if not ok:
            raise SystemExit(
                "ffmpeg failed to encode output; try rerun with `--verbose` or set `--codec mpeg4`."
            )
    elif codec == "mpeg4":
        cmd = make_cmd("mpeg4", ["-q:v", "3"])
        tried.append(cmd)
        run(cmd)
    else:
        cmd = make_cmd(codec, ["-b:v", str(args.bitrate)])
        tried.append(cmd)
        run(cmd)

    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
