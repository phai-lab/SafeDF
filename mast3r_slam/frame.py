import dataclasses
from enum import Enum
from typing import Optional
import lietorch
import torch
import torch.nn.functional as F
from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.config import config
import numpy as np

class Mode(Enum):
    INIT = 0
    TRACKING = 1
    RELOC = 2
    TERMINATED = 3


@dataclasses.dataclass
class Frame:
    frame_id: int
    img: torch.Tensor
    img_shape: torch.Tensor
    img_true_shape: torch.Tensor
    uimg: torch.Tensor
    T_WC: lietorch.Sim3 = lietorch.Sim3.Identity(1)
    X_canon: Optional[torch.Tensor] = None
    C: Optional[torch.Tensor] = None
    feat: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None
    N: int = 0
    N_updates: int = 0
    K: Optional[torch.Tensor] = None
    semantic_label: Optional[torch.Tensor] = None
    # Raw (pre-stabilization) semantic label for the current frame.
    #
    # Motivation:
    #   The tracker may overwrite `semantic_label` with stabilized semantics (warp+fuse).
    #   For planning TSDF publishing/debugging we sometimes want the RAW segmentation regardless
    #   of whether stabilization is enabled.
    #
    # Format:
    #   Same as `semantic_label`: (H,W,3) float on CPU in [0,1], using palette-agnostic 24-bit packing.
    semantic_label_raw: Optional[torch.Tensor] = None
    # ---------------------------------------------------------------------
    # V3: Semantic PointMap cache (optional, keyframes only)
    #
    # These fields are intentionally kept minimal:
    #   - sem_label  : (H, W) int64   hard label per pixel
    #   - sem_weight : (H, W) float32 vote/weight per pixel
    #
    # Important:
    #   - They are only meaningful when `config["semantic"]["enable_semantic_pointmap"] == True`.
    #   - When the feature is disabled, these remain None and the V1 pipeline is unchanged.
    # ---------------------------------------------------------------------
    sem_label: Optional[torch.Tensor] = None
    sem_weight: Optional[torch.Tensor] = None

    def get_score(self, C):
        filtering_score = config["tracking"]["filtering_score"]
        if filtering_score == "median":
            score = torch.median(C)  # Is this slower than mean? Is it worth it?
        elif filtering_score == "mean":
            score = torch.mean(C)
        return score

    def update_pointmap(self, X: torch.Tensor, C: torch.Tensor):
        filtering_mode = config["tracking"]["filtering_mode"]

        if self.N == 0:
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
            self.N_updates = 1
            if filtering_mode == "best_score":
                self.score = self.get_score(C)
            return

        if filtering_mode == "first":
            if self.N_updates == 1:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
        elif filtering_mode == "recent":
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
        elif filtering_mode == "best_score":
            new_score = self.get_score(C)
            if new_score > self.score:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
                self.score = new_score
        elif filtering_mode == "indep_conf":
            new_mask = C > self.C
            self.X_canon[new_mask.repeat(1, 3)] = X[new_mask.repeat(1, 3)]
            self.C[new_mask] = C[new_mask]
            self.N = 1
        elif filtering_mode == "weighted_pointmap":
            self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
            self.C = self.C + C
            self.N += 1
        elif filtering_mode == "weighted_spherical":

            def cartesian_to_spherical(P):
                r = torch.linalg.norm(P, dim=-1, keepdim=True)
                x, y, z = torch.tensor_split(P, 3, dim=-1)
                phi = torch.atan2(y, x)
                theta = torch.acos(z / r)
                spherical = torch.cat((r, phi, theta), dim=-1)
                return spherical

            def spherical_to_cartesian(spherical):
                r, phi, theta = torch.tensor_split(spherical, 3, dim=-1)
                x = r * torch.sin(theta) * torch.cos(phi)
                y = r * torch.sin(theta) * torch.sin(phi)
                z = r * torch.cos(theta)
                P = torch.cat((x, y, z), dim=-1)
                return P

            spherical1 = cartesian_to_spherical(self.X_canon)
            spherical2 = cartesian_to_spherical(X)
            spherical = ((self.C * spherical1) + (C * spherical2)) / (self.C + C)

            self.X_canon = spherical_to_cartesian(spherical)
            self.C = self.C + C
            self.N += 1

        self.N_updates += 1
        return

    def get_average_conf(self):
        return self.C / self.N if self.C is not None else None


def create_frame(i, img, T_WC, img_size=512, device="cuda:0"):
    img = resize_img(img, img_size)
    rgb = img["img"].to(device=device)
    img_shape = torch.tensor(img["true_shape"], device=device)
    img_true_shape = img_shape.clone()
    uimg = torch.from_numpy(img["unnormalized_img"]) / 255.0
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        uimg = uimg[::downsample, ::downsample]
        img_shape = img_shape // downsample
    frame = Frame(i, rgb, img_shape, img_true_shape, uimg, T_WC)
    return frame

def create_frame_semantic(i, img, T_WC, semantic_label, img_size=512, device="cuda:0"):
    # NOTE:
    #   `resize_img` performs a center-crop after resizing. If `semantic_label` is produced on the
    #   *pre-crop* RGB image (common), we must apply the SAME resize+crop to semantics, otherwise
    #   RGB vs semantic overlays (and TSDF fusion) will be spatially misaligned.
    img_in = img
    in_h, in_w = (int(img_in.shape[0]), int(img_in.shape[1]))

    img = resize_img(img_in, img_size)
    rgb = img["img"].to(device=device)
    img_shape = torch.tensor(img["true_shape"], device=device)
    img_true_shape = img_shape.clone()
    uimg = torch.from_numpy(img["unnormalized_img"]) / 255.0
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        uimg = uimg[::downsample, ::downsample]
        img_shape = img_shape // downsample
    # -------------------------------------------------------------------------
    # Semantic label handling (real-time + minimal-change):
    #
    # Internally, the tracker operates on hard labels (H,W) int for stabilization
    # and optional geometry bonus.
    #
    # However, this codebase currently shares `frame.semantic_label` across processes
    # (for visualization) using a CPU float tensor shaped (H,W,3).
    #
    # To keep changes minimal and avoid breaking the shared-memory visualization path:
    #   - If the input semantic is RGB (H,W,3), we keep it as RGB (resized by nearest).
    #   - If the input semantic is hard label IDs (H,W), we *encode* it into an RGB
    #     representation by bit-packing into 24-bit color (invertible):
    #         code = label_id
    #         R = (code >> 16) & 255, G = (code >> 8) & 255, B = code & 255
    #     This keeps the storage type compatible while allowing exact recovery of
    #     the integer label code later (no palette assumption required).
    #
    # IMPORTANT:
    #   - Resizing is always nearest-neighbor to avoid mixing labels.
    # -------------------------------------------------------------------------
    if isinstance(semantic_label, np.ndarray):
        semantic_label = torch.from_numpy(semantic_label)

    if semantic_label is not None:
        target_hw = tuple(int(x) for x in img_shape.flatten())
        crop_h, crop_w = (int(img["unnormalized_img"].shape[0]), int(img["unnormalized_img"].shape[1]))

        # Compute the intermediate "resized-before-crop" size used by `resize_img`.
        # This matches `mast3r_slam/mast3r_utils.py::resize_img`.
        if int(img_size) == 224:
            long_edge_size = int(round(float(img_size) * max(float(in_w) / max(1, in_h), float(in_h) / max(1, in_w))))
        else:
            long_edge_size = int(img_size)
        S = max(in_w, in_h)
        resized_w = int(round(float(in_w) * float(long_edge_size) / max(1.0, float(S))))
        resized_h = int(round(float(in_h) * float(long_edge_size) / max(1.0, float(S))))

        # Center-crop offsets in the resized image.
        x0 = int((resized_w - int(crop_w)) // 2)
        y0 = int((resized_h - int(crop_h)) // 2)

        def _maybe_resize_to_input_hw(x: torch.Tensor) -> torch.Tensor:
            """Best-effort: make semantic match the pre-resize RGB resolution before applying crop."""
            if x.ndim == 2:
                if tuple(x.shape) == (in_h, in_w):
                    return x
                xi = x[None, None].float()
                xi = F.interpolate(xi, size=(in_h, in_w), mode="nearest")
                return xi[0, 0].to(torch.int64)
            if x.ndim == 3:
                if tuple(x.shape[:2]) == (in_h, in_w):
                    return x
                xc = x.permute(2, 0, 1).unsqueeze(0).float()
                xc = F.interpolate(xc, size=(in_h, in_w), mode="nearest")
                return xc.squeeze(0).permute(1, 2, 0)
            return x

        def _resize_crop_downsample_label_hw(label_hw: torch.Tensor) -> torch.Tensor:
            # Resize to pre-crop size.
            x = label_hw[None, None].float()
            x = F.interpolate(x, size=(resized_h, resized_w), mode="nearest")
            y = x[0, 0].to(torch.int64)
            # Crop to match `unnormalized_img`.
            y = y[y0 : y0 + crop_h, x0 : x0 + crop_w]
            # Downsample to match `uimg`/`img_shape`.
            if int(downsample) > 1:
                y = y[:: int(downsample), :: int(downsample)]
            return y

        def _resize_crop_downsample_rgb_hw3(rgb_hw3: torch.Tensor) -> torch.Tensor:
            # Resize to pre-crop size.
            x = rgb_hw3.permute(2, 0, 1).unsqueeze(0).float()  # (1,3,H,W)
            x = F.interpolate(x, size=(resized_h, resized_w), mode="nearest")
            y = x.squeeze(0).permute(1, 2, 0)  # (H,W,3)
            # Crop to match `unnormalized_img`.
            y = y[y0 : y0 + crop_h, x0 : x0 + crop_w, :]
            # Downsample to match `uimg`/`img_shape`.
            if int(downsample) > 1:
                y = y[:: int(downsample), :: int(downsample), :]
            return y

        # Case 1: hard labels (H,W) -> encode to RGB (H,W,3) float in [0,1]
        if semantic_label.ndim == 2:
            label_hw = _maybe_resize_to_input_hw(semantic_label.to(torch.int64))
            if tuple(label_hw.shape) == (in_h, in_w):
                label_hw = _resize_crop_downsample_label_hw(label_hw)
            else:
                # Fallback: preserve previous behavior if we cannot match input resolution.
                if tuple(label_hw.shape) != target_hw:
                    x = label_hw[None, None].float()
                    x = F.interpolate(x, size=target_hw, mode="nearest")
                    label_hw = x[0, 0].to(torch.int64)

            # Bit-pack into 24-bit RGB (invertible, palette-agnostic).
            r = ((label_hw >> 16) & 255).to(torch.float32) / 255.0
            g = ((label_hw >> 8) & 255).to(torch.float32) / 255.0
            b = (label_hw & 255).to(torch.float32) / 255.0
            semantic_label = torch.stack([r, g, b], dim=-1)

        # Case 2: RGB mask (H,W,3) or CHW (3,H,W) -> standardize to HWC and resize
        elif semantic_label.ndim == 3:
            if semantic_label.shape[0] == 3 and semantic_label.shape[-1] != 3:
                # CHW -> HWC
                semantic_label = semantic_label.permute(1, 2, 0)

            if semantic_label.shape[-1] != 3:
                raise ValueError(f"Unsupported semantic_label shape: {semantic_label.shape}")

            semantic_label = _maybe_resize_to_input_hw(semantic_label)
            if tuple(semantic_label.shape[:2]) == (in_h, in_w):
                semantic_label = _resize_crop_downsample_rgb_hw3(semantic_label)
            elif tuple(semantic_label.shape[:2]) != target_hw:
                x = semantic_label.permute(2, 0, 1).unsqueeze(0).float()  # (1,3,H,W)
                x = F.interpolate(x, size=target_hw, mode="nearest")
                semantic_label = x.squeeze(0).permute(1, 2, 0)  # (H,W,3)
            else:
                semantic_label = semantic_label.float()

            # Normalize to [0,1] if input looks like uint8 or [0,255] float.
            maxv = float(semantic_label.max().item()) if semantic_label.numel() > 0 else 0.0
            if maxv > 1.0 + 1e-6:
                semantic_label = semantic_label / 255.0

        else:
            raise ValueError(f"Unsupported semantic_label ndim: {semantic_label.ndim}")

    # Keep a copy of RAW semantic so downstream processes can access it even if the tracker
    # overwrites `frame.semantic_label` with stabilized labels.
    semantic_label_raw = None if semantic_label is None else semantic_label.detach().clone()
    frame = Frame(
        i,
        rgb,
        img_shape,
        img_true_shape,
        uimg,
        T_WC,
        semantic_label=semantic_label,
        semantic_label_raw=semantic_label_raw,
    )
    return frame

class SharedStates:
    def __init__(self, manager, h, w, dtype=torch.float32, device="cuda"):
        self.h, self.w = h, w
        self.dtype = dtype
        self.device = device

        self.lock = manager.RLock()
        self.paused = manager.Value("i", 0)
        self.mode = manager.Value("i", Mode.INIT)
        self.reloc_sem = manager.Value("i", 0)
        self.global_optimizer_tasks = manager.list()
        self.edges_ii = manager.list()
        self.edges_jj = manager.list()

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)
        downsample = max(1, int(config["dataset"]["img_downsample"]))
        h_down = max(1, h // downsample)
        w_down = max(1, w // downsample)

        # fmt:off
        # shared state for the current frame (used for reloc/visualization)
        self.dataset_idx = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(3, h, w, device=device, dtype=dtype).share_memory_()
        # self.uimg = torch.zeros(h, w, 3, device="cpu", dtype=dtype).share_memory_() #4192
        self.uimg = torch.zeros(h_down, w_down, 3, device="cpu", dtype=dtype).share_memory_()

        self.img_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = lietorch.Sim3.Identity(1, device=device, dtype=dtype).data.share_memory_()
        # self.X = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_() #4192
        self.X = torch.zeros(h_down * w_down, 3, device=device, dtype=dtype).share_memory_()
        # self.C = torch.zeros(h * w, 1, device=device, dtype=dtype).share_memory_() #4192
        self.C = torch.zeros(h_down * w_down, 1, device=device, dtype=dtype).share_memory_()
        self.feat = torch.zeros(1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        
        # self.semantic_label = torch.zeros(h, w, 3, device="cpu", dtype=dtype).share_memory_() #4192
        self.semantic_label = torch.zeros(h_down, w_down, 3, device="cpu", dtype=dtype).share_memory_()

        # RAW semantic (pre-stabilization) for downstream consumers (e.g., TSDF publisher/viewer).
        self.semantic_label_raw = torch.zeros(h_down, w_down, 3, device="cpu", dtype=dtype).share_memory_()
        self.semantic_label_raw_valid = torch.zeros(1, device="cpu", dtype=torch.int32).share_memory_()
        # fmt: on

    def set_frame(self, frame):
        with self.lock:
            self.dataset_idx[:] = frame.frame_id
            self.img[:] = frame.img
            self.uimg[:] = frame.uimg
            self.img_shape[:] = frame.img_shape
            self.img_true_shape[:] = frame.img_true_shape
            self.T_WC[:] = frame.T_WC.data
            self.X[:] = frame.X_canon
            self.C[:] = frame.C
            self.feat[:] = frame.feat
            self.pos[:] = frame.pos
            if frame.semantic_label is not None:
                self.semantic_label[:] = frame.semantic_label
            # Maintain a raw semantic copy (prefer explicit raw field, fall back to current semantic).
            raw = getattr(frame, "semantic_label_raw", None)
            if raw is None:
                raw = frame.semantic_label
            if raw is not None:
                self.semantic_label_raw[:] = raw
                self.semantic_label_raw_valid[0] = 1
            else:
                self.semantic_label_raw.fill_(0)
                self.semantic_label_raw_valid[0] = 0

    def get_frame(self):
        with self.lock:
            frame = Frame(
                int(self.dataset_idx[0]),
                self.img,
                self.img_shape,
                self.img_true_shape,
                self.uimg,
                lietorch.Sim3(self.T_WC),
                semantic_label=self.semantic_label,
            )
            frame.X_canon = self.X
            frame.C = self.C
            frame.feat = self.feat
            frame.pos = self.pos
            if int(self.semantic_label_raw_valid[0].item()) != 0:
                frame.semantic_label_raw = self.semantic_label_raw
            else:
                frame.semantic_label_raw = None
            
            return frame

    def queue_global_optimization(self, idx):
        with self.lock:
            self.global_optimizer_tasks.append(idx)

    def queue_reloc(self):
        with self.lock:
            self.reloc_sem.value += 1

    def dequeue_reloc(self):
        with self.lock:
            if self.reloc_sem.value == 0:
                return
            self.reloc_sem.value -= 1

    def get_mode(self):
        with self.lock:
            return self.mode.value

    def set_mode(self, mode):
        with self.lock:
            self.mode.value = mode

    def pause(self):
        with self.lock:
            self.paused.value = 1

    def unpause(self):
        with self.lock:
            self.paused.value = 0

    def is_paused(self):
        with self.lock:
            return self.paused.value == 1


class SharedKeyframes:
    def __init__(self, manager, h, w, buffer=512, dtype=torch.float32, device="cuda"):
        self.lock = manager.RLock()
        self.n_size = manager.Value("i", 0)

        self.h, self.w = h, w
        self.buffer = buffer
        self.dtype = dtype
        self.device = device

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)
        downsample = max(1, int(config["dataset"]["img_downsample"]))
        h_down = max(1, h // downsample)
        w_down = max(1, w // downsample)

        # fmt:off
        self.dataset_idx = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(buffer, 3, h, w, device=device, dtype=dtype).share_memory_()
        # self.uimg = torch.zeros(buffer, h, w, 3, device="cpu", dtype=dtype).share_memory_() #4192
        self.uimg = torch.zeros(buffer, h_down, w_down, 3, device="cpu", dtype=dtype).share_memory_()
        # raise NotImplementedError(self.uimg.shape,h,w)
        self.img_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, device=device, dtype=dtype).share_memory_()
        # self.X = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype).share_memory_() #4192
        self.X = torch.zeros(buffer, h_down * w_down, 3, device=device, dtype=dtype).share_memory_()
        # self.C = torch.zeros(buffer, h * w, 1, device=device, dtype=dtype).share_memory_() #4192
        self.C = torch.zeros(buffer, h_down * w_down, 1, device=device, dtype=dtype).share_memory_()
        self.N = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.N_updates = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.feat = torch.zeros(buffer, 1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(buffer, 1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        self.is_dirty = torch.zeros(buffer, 1, device=device, dtype=torch.bool).share_memory_()
        self.K = torch.zeros(3, 3, device=device, dtype=dtype).share_memory_()
        # self.semantic_label = torch.zeros(buffer, h, w, 3, device="cpu", dtype=dtype).share_memory_() #4192
        self.semantic_label = torch.zeros(buffer, h_down, w_down, 3, device="cpu", dtype=dtype).share_memory_()

        # -----------------------------------------------------------------
        # V3: Semantic PointMap storage (optional)
        #
        # Motivation:
        #   We want a keyframe-side semantic cache that becomes more stable over time, similar to how
        #   the geometric pointmap is fused. To keep memory minimal, we store only:
        #     - one hard label per pixel
        #     - one scalar weight per pixel
        #
        # Behavior when disabled:
        #   - We do NOT allocate these tensors, and all V1 behavior remains unchanged.
        #
        # Note:
        #   We allocate these on `device` (typically CUDA) because updates happen inside the tracker,
        #   and we want to avoid per-frame CPU<->GPU transfers.
        # -----------------------------------------------------------------
        sem_cfg = config.get("semantic", {})
        self.enable_semantic_pointmap = bool(sem_cfg.get("enable_semantic_pointmap", False))
        if self.enable_semantic_pointmap:
            self.sem_label = torch.zeros(
                buffer, h_down, w_down, device=device, dtype=torch.long
            ).share_memory_()
            self.sem_weight = torch.zeros(
                buffer, h_down, w_down, device=device, dtype=torch.float32
            ).share_memory_()
        # fmt: on

    def __getitem__(self, idx) -> Frame:
        with self.lock:
            # put all of the data into a frame
            kf = Frame(
                int(self.dataset_idx[idx]),
                self.img[idx],
                self.img_shape[idx],
                self.img_true_shape[idx],
                self.uimg[idx],
                lietorch.Sim3(self.T_WC[idx]),
                semantic_label=self.semantic_label[idx].clone() if self.semantic_label[idx].sum() > 0 else None
            )
            kf.X_canon = self.X[idx]
            kf.C = self.C[idx]
            kf.feat = self.feat[idx]
            kf.pos = self.pos[idx]
            kf.N = int(self.N[idx])
            kf.N_updates = int(self.N_updates[idx])
            if config["use_calib"]:
                kf.K = self.K
            if self.enable_semantic_pointmap:
                # Attach semantic pointmap tensors to the returned Frame. These tensors live on
                # the shared storage device (typically CUDA) and are updated by the tracker.
                kf.sem_label = self.sem_label[idx]
                kf.sem_weight = self.sem_weight[idx]
            return kf

    def __setitem__(self, idx, value: Frame) -> None:
        with self.lock:
            self.n_size.value = max(idx + 1, self.n_size.value)

            # set the attributes
            self.dataset_idx[idx] = value.frame_id
            self.img[idx] = value.img
            # raise NotImplementedError(value.uimg.shape, self.uimg[idx].shape, self.img[idx].shape, value.img.shape)
            self.uimg[idx] = value.uimg
            self.img_shape[idx] = value.img_shape
            self.img_true_shape[idx] = value.img_true_shape
            self.T_WC[idx] = value.T_WC.data
            self.X[idx] = value.X_canon
            self.C[idx] = value.C
            self.feat[idx] = value.feat
            self.pos[idx] = value.pos
            self.N[idx] = value.N
            self.N_updates[idx] = value.N_updates
            self.is_dirty[idx] = True

            if value.semantic_label is not None:
                tensor = value.semantic_label
                tensor = tensor.to(self.semantic_label.device, dtype=self.semantic_label.dtype)
                if tensor.shape != self.semantic_label[idx].shape:
                    import torch.nn.functional as F
                    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                    tensor = F.interpolate(tensor, size=self.semantic_label[idx].shape[:2], mode='nearest')
                    tensor = tensor.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
                self.semantic_label[idx] = tensor

            if self.enable_semantic_pointmap:
                # Persist V3 keyframe semantic cache if provided.
                #
                # IMPORTANT:
                #   We only write when the feature is enabled. This guarantees that disabling the
                #   feature keeps the V1 pipeline identical (no extra state, no updates).
                if value.sem_label is not None:
                    self.sem_label[idx] = value.sem_label.to(self.sem_label.device, dtype=self.sem_label.dtype)
                if value.sem_weight is not None:
                    self.sem_weight[idx] = value.sem_weight.to(self.sem_weight.device, dtype=self.sem_weight.dtype)
            return idx

    def __len__(self):
        with self.lock:
            return self.n_size.value

    def append(self, value: Frame):
        with self.lock:
            self[self.n_size.value] = value

    def pop_last(self):
        with self.lock:
            self.n_size.value -= 1

    def last_keyframe(self) -> Optional[Frame]:
        with self.lock:
            if self.n_size.value == 0:
                return None
            return self[self.n_size.value - 1]

    def update_T_WCs(self, T_WCs, idx) -> None:
        with self.lock:
            self.T_WC[idx] = T_WCs.data

    def get_dirty_idx(self):
        with self.lock:
            idx = torch.where(self.is_dirty)[0]
            self.is_dirty[:] = False
            return idx

    def set_intrinsics(self, K):
        assert config["use_calib"]
        with self.lock:
            self.K[:] = K

    def get_intrinsics(self):
        assert config["use_calib"]
        with self.lock:
            return self.K
