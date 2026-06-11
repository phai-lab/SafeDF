import math
import pathlib
import sys

import cv2
import numpy as np
import torch

# --- Online segmentation (lazy EfficientViT) ---
#
# IMPORTANT DESIGN NOTE:
#   We support multiple EfficientViT segmentation heads (ADE20K vs Cityscapes).
#   To keep real-time behavior and avoid re-loading weights on every call, we cache
#   the initialized model per dataset name (and per device).
#
#   Key constraints from the project:
#     - segmentation is used as an auxiliary signal (debug / semantics) and must remain fast
#     - callers may want to switch datasets via CLI without editing code
_ESEG_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
_ESEG_CACHE: dict[str, dict[str, object]] = {}


def _resolve_efficientvit_weight_path(efficientvit_root: pathlib.Path, dataset: str) -> pathlib.Path:
    """
    Resolve the expected weight path for EfficientViT-L2 segmentation.

    We intentionally support multiple common filename conventions to reduce friction:
      - ADE20K:
          * `efficientvit/l2.pt` (this repo's original convention)
          * `efficientvit/efficientvit_seg_l2_ade20k.pt` (upstream-ish naming)
          * `efficientvit/assets/checkpoints/efficientvit_seg/efficientvit_seg_l2_ade20k.pt`
      - Cityscapes:
          * `efficientvit/l2_cityscapes.pt` (our earlier suggested name)
          * `efficientvit/efficientvit_seg_l2_cityscapes.pt` (already present in this repo)
          * `efficientvit/assets/checkpoints/efficientvit_seg/efficientvit_seg_l2_cityscapes.pt`

    Rationale:
      Users often download checkpoints with upstream filenames. If we hard-code a single name,
      the CLI switch becomes brittle and produces confusing "file not found" errors.
    """

    ds = str(dataset).lower()
    candidates: list[pathlib.Path]
    if ds in ("ade20k", "ade"):
        candidates = [
            efficientvit_root / "l2.pt",
            efficientvit_root / "efficientvit_seg_l2_ade20k.pt",
            efficientvit_root / "assets" / "checkpoints" / "efficientvit_seg" / "efficientvit_seg_l2_ade20k.pt",
        ]
    elif ds in ("cityscapes", "cityscape", "cs"):
        candidates = [
            efficientvit_root / "l2_cityscapes.pt",
            efficientvit_root / "efficientvit_seg_l2_cityscapes.pt",
            efficientvit_root / "assets" / "checkpoints" / "efficientvit_seg" / "efficientvit_seg_l2_cityscapes.pt",
        ]
    else:
        raise ValueError(f"Unsupported EfficientViT dataset: {dataset!r} (expected ade20k or cityscapes)")

    for p in candidates:
        if p.exists():
            return p

    # None found: raise with a helpful error listing all candidates.
    tried = "\n".join([f"  - {str(p)}" for p in candidates])
    raise FileNotFoundError(
        f"EfficientViT weight not found for dataset={ds!r}. Tried:\n{tried}"
    )


def _init_online_segmenter(*, dataset: str = "ade20k") -> dict[str, object]:
    """
    Lazily load EfficientViT segmentation model and helpers for a given dataset.

    Inputs:
      dataset: str
        - "ade20k" (default) or "cityscapes"

    Returns:
      A dict cache entry containing:
        - "model"       : torch.nn.Module
        - "transform"   : torchvision transform
        - "class_colors": tuple of RGB tuples
        - "resize"      : resize function for logits
        - "get_canvas"  : helper to draw a colored canvas

    Weight files:
      - ADE20K uses:     `efficientvit/l2.pt`
      - Cityscapes uses: `efficientvit/l2_cityscapes.pt`
    """

    ds = str(dataset).lower()
    if ds in _ESEG_CACHE:
        return _ESEG_CACHE[ds]

    efficientvit_root = pathlib.Path(__file__).resolve().parent.parent / "efficientvit"
    if str(efficientvit_root) not in sys.path:
        sys.path.append(str(efficientvit_root))

    try:
        from torchvision import transforms  # type: ignore
        from efficientvit.seg_model_zoo import create_efficientvit_seg_model  # type: ignore
        from efficientvit.models.utils import resize as evit_resize  # type: ignore
        from applications.efficientvit_seg.eval_efficientvit_seg_model import (  # type: ignore
            ADE20KDataset,
            CityscapesDataset,
            ToTensor,
            get_canvas,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to import EfficientViT modules: {e}")

    # Select model name + class color palette based on the dataset.
    #
    # NOTE:
    #   Upstream EfficientViT uses different heads (num classes) for ADE20K vs Cityscapes.
    #   Using the wrong head/weights pairing will silently produce nonsense labels, so we
    #   make the choice explicit here.
    if ds in ("ade20k", "ade"):
        model_name = "efficientvit-seg-l2-ade20k"
        class_colors = ADE20KDataset.class_colors
    elif ds in ("cityscapes", "cityscape", "cs"):
        model_name = "efficientvit-seg-l2-cityscapes"
        class_colors = CityscapesDataset.class_colors
    else:
        raise ValueError(f"Unsupported EfficientViT dataset: {dataset!r} (expected ade20k or cityscapes)")

    # Resolve weight file path using a set of common filename conventions.
    # If no file is found, `_resolve_efficientvit_weight_path` raises a FileNotFoundError
    # that lists all candidate paths it tried (helpful for debugging).
    weight_path = _resolve_efficientvit_weight_path(efficientvit_root, ds)

    model = create_efficientvit_seg_model(
        model_name, weight_url=str(weight_path)
    ).to(_ESEG_DEVICE)
    model.eval()

    transform = transforms.Compose(
        [ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    entry: dict[str, object] = {
        "model": model,
        "transform": transform,
        "class_colors": class_colors,
        "resize": evit_resize,
        "get_canvas": get_canvas,
    }
    _ESEG_CACHE[ds] = entry
    return entry


def segment_image_efficientvit(img_rgb_float01: np.ndarray, *, dataset: str = "ade20k") -> np.ndarray:
    """Run EfficientViT segmentation and return a colored mask (H, W, 3) in [0,1].

    - img_rgb_float01: HxWx3 float array in [0,1]
    """
    entry = _init_online_segmenter(dataset=dataset)
    model = entry["model"]  # type: ignore[assignment]
    transform = entry["transform"]  # type: ignore[assignment]
    class_colors = entry["class_colors"]  # type: ignore[assignment]
    evit_resize = entry["resize"]  # type: ignore[assignment]
    get_canvas = entry["get_canvas"]  # type: ignore[assignment]

    img_u8 = (img_rgb_float01 * 255.0).clip(0, 255).astype(np.uint8)

    h, w = img_u8.shape[:2]
    if h < w:
        th = 512
        tw = int(math.ceil(w / h * th / 32.0) * 32)
    else:
        tw = 512
        th = int(math.ceil(h / w * tw / 32.0) * 32)

    if (th, tw) != (h, w):
        img_in = cv2.resize(img_u8, (tw, th), interpolation=cv2.INTER_CUBIC)
    else:
        img_in = img_u8

    dummy_label = np.zeros(img_in.shape[:2], dtype=np.int64)
    data = transform({"data": img_in, "label": dummy_label})["data"]
    with torch.inference_mode():
        out = model(torch.unsqueeze(data, 0).to(_ESEG_DEVICE))
        if out.shape[-2:] != img_u8.shape[:2]:
            out = evit_resize(out, size=img_u8.shape[:2])
        pred = torch.argmax(out, dim=1).cpu().numpy()[0]

    canvas = get_canvas(img_u8, pred, class_colors, opacity=1.0)
    return canvas.astype(np.float32) / 255.0


def segment_image_efficientvit_labels(img_rgb_float01: np.ndarray, *, dataset: str = "ade20k") -> np.ndarray:
    """
    Run EfficientViT segmentation and return hard labels (H, W) int64.

    This is intended for downstream logic (e.g., planning statistics) that needs class IDs rather
    than visualization colors.

    Input:
      img_rgb_float01: (H,W,3) float32 in [0,1]

    Output:
      pred: (H,W) int64, where values are class IDs (e.g., ADE20K indices).

    Performance notes:
      - This shares the same lazy model initialization as `segment_image_efficientvit`.
      - It performs the same "friendly resize" to multiples of 32, then resizes logits back to
        the input resolution.
    """

    entry = _init_online_segmenter(dataset=dataset)
    model = entry["model"]  # type: ignore[assignment]
    transform = entry["transform"]  # type: ignore[assignment]
    evit_resize = entry["resize"]  # type: ignore[assignment]

    img_u8 = (img_rgb_float01 * 255.0).clip(0, 255).astype(np.uint8)

    h, w = img_u8.shape[:2]
    if h < w:
        th = 512
        tw = int(math.ceil(w / h * th / 32.0) * 32)
    else:
        tw = 512
        th = int(math.ceil(h / w * tw / 32.0) * 32)

    if (th, tw) != (h, w):
        img_in = cv2.resize(img_u8, (tw, th), interpolation=cv2.INTER_CUBIC)
    else:
        img_in = img_u8

    dummy_label = np.zeros(img_in.shape[:2], dtype=np.int64)
    data = transform({"data": img_in, "label": dummy_label})["data"]
    with torch.inference_mode():
        out = model(torch.unsqueeze(data, 0).to(_ESEG_DEVICE))
        if out.shape[-2:] != img_u8.shape[:2]:
            out = evit_resize(out, size=img_u8.shape[:2])
        pred = torch.argmax(out, dim=1).to(torch.int64).cpu().numpy()[0]

    return pred
