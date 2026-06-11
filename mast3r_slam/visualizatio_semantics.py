import dataclasses
import weakref
from pathlib import Path

import imgui
import lietorch
import torch
import moderngl
import moderngl_window as mglw
import numpy as np
from in3d.camera import Camera, ProjectionMatrix, lookat
from in3d.pose_utils import translation_matrix
from in3d.color import hex2rgba
from in3d.geometry import Axis
from in3d.viewport_window import ViewportWindow
from in3d.window import WindowEvents
from in3d.image import Image
from moderngl_window import resources
from moderngl_window.timers.clock import Timer

from mast3r_slam.frame import Mode
from mast3r_slam.geometry import get_pixel_coords
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.visualization_utils import (
    Frustums,
    Lines,
    depth2rgb,
    image_with_text,
)
from mast3r_slam.config import load_config, config, set_global_config


@dataclasses.dataclass
class WindowMsg:
    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5
    # Semantic visualization controls (visualization-only).
    show_semantic: bool = True
    semantic_alpha: float = 0.7
    semantic_mode: str = "semantic"  # "overlay" or "semantic" (semantic-only)
    # ---------------------------------------------------------------------
    # Semantic point cloud coloring source (visualization-only).
    #
    # The SLAM pipeline may provide semantics in two different formats:
    #
    #   (A) `frame.semantic_label` / `keyframe.semantic_label`
    #       A 3-channel RGB mask (H,W,3) intended primarily for visualization and
    #       cross-process sharing. Depending on the upstream code path it may reflect
    #       raw or stabilized semantics and it may be palette-based or bit-packed.
    #
    #   (B) `keyframe.sem_label` (V3 semantic pointmap cache)
    #       A hard-label ID map (H,W) that is updated over time via match-based voting
    #       and aligns with the geometric pointmap grid (same pixel indexing).
    #
    # The user requested two checkboxes to select which semantic source is used to
    # color the point cloud.
    #
    # Behavior:
    #   - If both are enabled and `keyframe.sem_label` exists, we prefer the V3 cache
    #     because it is typically more stable.
    #   - If neither is enabled, the point cloud is colored by the regular RGB image.
    #
    # IMPORTANT:
    #   This is visualization-only; it must NOT affect tracking, matching, or optimization.
    # ---------------------------------------------------------------------
    use_semantic_rgb_mask: bool = True
    use_semantic_pointmap_cache: bool = False


class Window(WindowEvents):
    title = "MASt3R-SLAM Semantic"
    window_size = (1960, 1080)

    def __init__(self, states, keyframes, main2viz, viz2main, **kwargs):
        super().__init__(**kwargs)
        self.ctx.gc_mode = "auto"
        # bit hacky, but detect whether user is using 4k monitor
        self.scale = 1.0
        if self.wnd.buffer_size[0] > 2560:
            self.set_font_scale(2.0)
            self.scale = 2
        self.clear = hex2rgba("#1E2326", alpha=1)
        resources.register_dir((Path(__file__).parent.parent / "resources").resolve())

        self.line_prog = self.load_program("programs/lines.glsl")
        self.surfelmap_prog = self.load_program("programs/surfelmap.glsl")
        self.trianglemap_prog = self.load_program("programs/trianglemap.glsl")
        self.pointmap_prog = self.surfelmap_prog

        width, height = self.wnd.size
        self.camera = Camera(
            ProjectionMatrix(width, height, 60, width // 2, height // 2, 0.05, 100),
            lookat(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        )
        self.axis = Axis(self.line_prog, 0.1, 3 * self.scale)
        self.frustums = Frustums(self.line_prog)
        self.lines = Lines(self.line_prog)

        self.viewport = ViewportWindow("Scene", self.camera)
        self.state = WindowMsg()
        self.keyframes = keyframes
        self.states = states

        self.show_all = True
        self.show_keyframe_edges = True
        self.culling = True
        self.follow_cam = True

        self.depth_bias = 0.001
        self.frustum_scale = 0.05

        self.dP_dz = None

        self.line_thickness = 3
        self.show_keyframe = True
        self.show_curr_pointmap = True
        self.show_axis = True

        self.textures = dict()
        # Semantic textures used for point cloud coloring (visualization-only).
        self.semantic_textures = dict()
        # Cache semantic RGB images per keyframe to avoid repeatedly converting large tensors.
        # This is especially important when the semantic source is `keyframe.sem_label` which
        # typically lives on GPU memory.
        self._semantic_rgb_cache = {}
        self.mtime = self.pointmap_prog.extra["meta"].resolved_path.stat().st_mtime
        self.curr_img, self.kf_img = Image(), Image()
        # 2D preview images for the current frame and the latest keyframe (visualization-only).
        self.curr_semantic_img, self.kf_semantic_img = Image(), Image()
        self.curr_img_np, self.kf_img_np = None, None
        self.curr_semantic_np, self.kf_semantic_np = None, None

        self.main2viz = main2viz
        self.viz2main = viz2main

    def semantic_mask_to_rgb(self, semantic_mask: torch.Tensor, label_to_color: dict = None) -> np.ndarray:
        """
        Return the semantic mask as an RGB image (visualization-only).

        - If the mask is single-channel (H,W), we expand it to (H,W,3) by repeating the channel.
        - If the mask is already RGB (H,W,3), we return it directly.
        """
        if semantic_mask is None:
            return np.zeros((1, 1, 3), dtype=np.float32)
        arr = semantic_mask.cpu().numpy() if isinstance(semantic_mask, torch.Tensor) else semantic_mask
        if arr.ndim == 2:
            # Single-channel -> expand to 3 channels for convenience.
            arr = np.stack([arr]*3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pass  # Already RGB.
        else:
            raise ValueError(f"Unsupported semantic_mask shape: {arr.shape}")
        # If input is uint8 in [0,255], convert to float32 in [0,1] for GPU textures.
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        return arr

    def semantic_rgb_to_label_id(self, semantic_rgb: np.ndarray) -> np.ndarray:
        """
        Decode an RGB semantic mask into an integer label ID map (H,W).

        Why:
          The visualization historically stores semantics as an RGB mask (`frame.semantic_label`)
          for cross-process sharing. In this repo, hard labels can be encoded into RGB via a
          palette-agnostic 24-bit packing:

            code = label_id
            R = (code >> 16) & 255
            G = (code >> 8)  & 255
            B = (code >> 0)  & 255

          This function reverses that packing so we can apply a consistent palette (same as
          `debug_viz.py`) for both sources.

        Input:
          semantic_rgb: (H,W,3) float32 in [0,1] or uint8 in [0,255]
        Output:
          label_hw: (H,W) uint32 label ids

        Notes:
          - If the upstream RGB mask is a true dataset palette (not bit-packed), this decode
            will produce "arbitrary but deterministic" IDs. That is still acceptable for
            visualization consistency.
        """

        arr = np.asarray(semantic_rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"semantic_rgb must be HxWx3, got {arr.shape}")
        if arr.dtype == np.uint8:
            rgb_u8 = arr
        else:
            # Accept float in [0,1] or [0,255].
            maxv = float(np.max(arr)) if arr.size else 0.0
            if maxv <= 1.0 + 1e-6:
                rgb_u8 = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                rgb_u8 = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        r = rgb_u8[..., 0].astype(np.uint32)
        g = rgb_u8[..., 1].astype(np.uint32)
        b = rgb_u8[..., 2].astype(np.uint32)
        return (r << 16) | (g << 8) | b

    def label_id_to_rgb(self, label_hw: np.ndarray) -> np.ndarray:
        """
        Convert integer label IDs (H,W) into a deterministic RGB visualization (float32 in [0,1]).

        Why:
          - `keyframe.sem_label` is a hard-label ID map (V3 semantic pointmap cache).
          - The point cloud shader expects a 3-channel texture (`img`) to color each point.
          - We do NOT want to depend on a fixed dataset palette here, so we use a simple
            hash-based coloring that works for any label ID range.

        Input:
          label_hw: (H,W) int/uint
        Output:
          rgb_hw3: (H,W,3) float32 in [0,1]
        """

        v = np.asarray(label_hw, dtype=np.uint32)
        r = (v * 37 + 17) & 255
        g = (v * 17 + 59) & 255
        b = (v * 97 + 101) & 255
        rgb_u8 = np.stack([r, g, b], axis=-1).astype(np.uint8)
        return rgb_u8.astype(np.float32) / 255.0

    def _get_semantic_rgb_for_keyframe(self, keyframe) -> np.ndarray | None:
        """
        Select the semantic RGB image used to color a KEYFRAME point cloud.

        Priority:
          1) V3 semantic pointmap cache (`keyframe.sem_label`) if enabled and available.
          2) Legacy semantic RGB mask (`keyframe.semantic_label`) if enabled and available.
          3) None (no semantic coloring).
        """

        # Prefer V3 cache when enabled.
        if (
            self.state.use_semantic_pointmap_cache
            and hasattr(keyframe, "sem_label")
            and keyframe.sem_label is not None
        ):
            try:
                lab = keyframe.sem_label.detach().cpu().numpy()
                return self.label_id_to_rgb(lab)
            except Exception:
                return None

        # Fall back to legacy RGB mask.
        if self.state.use_semantic_rgb_mask and hasattr(keyframe, "semantic_label") and keyframe.semantic_label is not None:
            try:
                # Convert the legacy RGB mask into label IDs (when possible) and then apply the
                # SAME deterministic palette as used for the V3 cache. This ensures both sources
                # look consistent and avoids relying on any external dataset palette.
                rgb = self.semantic_mask_to_rgb(
                    keyframe.semantic_label, getattr(keyframe, "label_to_color", {})
                )
                lab = self.semantic_rgb_to_label_id(rgb)
                return self.label_id_to_rgb(lab)
            except Exception:
                return None

        return None

    def _get_semantic_rgb_for_frame(self, frame) -> np.ndarray | None:
        """
        Select the semantic RGB image used for the CURRENT frame overlays.

        Note:
          - The V3 semantic pointmap cache is keyframe-side, so the current frame cannot use it.
          - If the user disables the legacy RGB mask checkbox, we simply do not show semantic
            overlays for the current frame.
        """

        if self.state.use_semantic_rgb_mask and hasattr(frame, "semantic_label") and frame.semantic_label is not None:
            try:
                rgb = self.semantic_mask_to_rgb(frame.semantic_label, getattr(frame, "label_to_color", {}))
                lab = self.semantic_rgb_to_label_id(rgb)
                return self.label_id_to_rgb(lab)
            except Exception:
                return None
        return None

    def blend_semantic_rgb(self, rgb_img: np.ndarray, semantic_rgb: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """
        Blend an RGB image with a semantic RGB overlay (visualization-only).
        """
        if semantic_rgb is None or semantic_rgb.sum() == 0:
            return rgb_img
        
        return rgb_img * (1 - alpha) + semantic_rgb * alpha

    def render(self, t: float, frametime: float):
        self.viewport.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        if self.culling:
            self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.clear(*self.clear)

        self.ctx.point_size = 2
        if self.show_axis:
            self.axis.render(self.camera)

        curr_frame = self.states.get_frame()
        h, w = curr_frame.img_shape.flatten()
        self.frustums.make_frustum(h, w)

        # Update current-frame RGB + semantic preview images (visualization-only).
        self.curr_img_np = curr_frame.uimg.numpy()
        self.curr_img.write(self.curr_img_np)
        
        # Current-frame semantic overlay (visualization-only).
        #
        # IMPORTANT:
        #   - The current frame does not have a V3 semantic pointmap cache; only keyframes do.
        #   - We therefore visualize only the legacy RGB mask when enabled.
        semantic_rgb = self._get_semantic_rgb_for_frame(curr_frame)
        if semantic_rgb is not None:
            if self.state.semantic_mode == "overlay":
                self.curr_semantic_np = self.blend_semantic_rgb(
                    self.curr_img_np, semantic_rgb, self.state.semantic_alpha
                )
            else:
                # Treat any non-"overlay" mode as "semantic-only" for robustness.
                self.curr_semantic_np = semantic_rgb
        else:
            self.curr_semantic_np = self.curr_img_np
        
        self.curr_semantic_img.write(self.curr_semantic_np)

        cam_T_WC = as_SE3(curr_frame.T_WC).cpu()
        if self.follow_cam:
            T_WC = cam_T_WC.matrix().numpy().astype(
                dtype=np.float32
            ) @ translation_matrix(np.array([0, 0, -2], dtype=np.float32))
            self.camera.follow_cam(np.linalg.inv(T_WC))
        else:
            self.camera.unfollow_cam()
        self.frustums.add(
            cam_T_WC,
            scale=self.frustum_scale,
            color=[0, 1, 0, 1],
            thickness=self.line_thickness * self.scale,
        )

        with self.keyframes.lock:
            N_keyframes = len(self.keyframes)
            dirty_idx = self.keyframes.get_dirty_idx()

        for kf_idx in dirty_idx:
            keyframe = self.keyframes[kf_idx]
            h, w = keyframe.img_shape.flatten()
            X = self.frame_X(keyframe)
            C = keyframe.get_average_conf().cpu().numpy().astype(np.float32)

            if keyframe.frame_id not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures[keyframe.frame_id] = ptex, ctex, itex
                
                # Create a semantic texture for this keyframe (visualization-only).
                semantic_tex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.semantic_textures[keyframe.frame_id] = semantic_tex
                
                ptex, ctex, itex = self.textures[keyframe.frame_id]
                itex.write(keyframe.uimg.numpy().astype(np.float32).tobytes())

            ptex, ctex, itex = self.textures[keyframe.frame_id]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())
            
            # Update keyframe semantic texture only when the keyframe is marked dirty.
            # This matches the overall visualization design: update resources only when data changes.
            semantic_rgb = self._get_semantic_rgb_for_keyframe(keyframe)
            if semantic_rgb is not None:
                # Cache the raw semantic RGB for later reuse (avoid repeated GPU->CPU conversion).
                self._semantic_rgb_cache[int(keyframe.frame_id)] = semantic_rgb
                semantic_tex = self.semantic_textures[keyframe.frame_id]
                # IMPORTANT (per user request):
                #   When coloring the POINT CLOUD by semantics, do NOT blend semantic colors with RGB.
                #   The semantic texture must contain pure semantic colors so the point colors are
                #   unambiguous and stable.
                semantic_tex.write(semantic_rgb.astype(np.float32).tobytes())

        for kf_idx in range(N_keyframes):
            keyframe = self.keyframes[kf_idx]
            h, w = keyframe.img_shape.flatten()
            if kf_idx == N_keyframes - 1:
                self.kf_img_np = keyframe.uimg.numpy()
                self.kf_img.write(self.kf_img_np)
                
                # Update the latest-keyframe semantic preview image (visualization-only).
                # Keyframe semantic preview image follows the SAME semantic source selection
                # as the point cloud coloring.
                semantic_rgb = self._semantic_rgb_cache.get(int(keyframe.frame_id), None)
                if semantic_rgb is None:
                    semantic_rgb = self._get_semantic_rgb_for_keyframe(keyframe)
                    if semantic_rgb is not None:
                        self._semantic_rgb_cache[int(keyframe.frame_id)] = semantic_rgb
                if semantic_rgb is not None:
                    if self.state.semantic_mode == "overlay":
                        self.kf_semantic_np = self.blend_semantic_rgb(
                            self.kf_img_np, semantic_rgb, self.state.semantic_alpha
                        )
                    else:
                        self.kf_semantic_np = semantic_rgb
                else:
                    self.kf_semantic_np = self.kf_img_np
                
                self.kf_semantic_img.write(self.kf_semantic_np)

            color = [1, 0, 0, 1]
            if self.show_keyframe:
                self.frustums.add(
                    as_SE3(keyframe.T_WC.cpu()),
                    scale=self.frustum_scale,
                    color=color,
                    thickness=self.line_thickness * self.scale,
                )

            ptex, ctex, itex = self.textures[keyframe.frame_id]
            # Always refresh the RGB texture used for non-semantic coloring.
            itex.write(keyframe.uimg.numpy().astype(np.float32).tobytes())
            if self.state.show_semantic and keyframe.frame_id in self.semantic_textures:
                semantic_tex = self.semantic_textures[keyframe.frame_id]
                # Use cached semantic RGB to avoid repeatedly converting tensors.
                semantic_rgb = self._semantic_rgb_cache.get(int(keyframe.frame_id), None)
                if semantic_rgb is None:
                    semantic_rgb = self._get_semantic_rgb_for_keyframe(keyframe)
                    if semantic_rgb is not None:
                        self._semantic_rgb_cache[int(keyframe.frame_id)] = semantic_rgb
                if semantic_rgb is not None:
                    # IMPORTANT (per user request):
                    #   Do NOT blend semantic colors with RGB for point cloud coloring.
                    semantic_tex.write(semantic_rgb.astype(np.float32).tobytes())

            if self.show_all:
                # Choose which texture is used to color the point cloud during rendering.
                if self.state.show_semantic and keyframe.frame_id in self.semantic_textures:
                    semantic_tex = self.semantic_textures[keyframe.frame_id]
                    self.render_pointmap(keyframe.T_WC.cpu(), w, h, ptex, ctex, semantic_tex)
                else:
                    self.render_pointmap(keyframe.T_WC.cpu(), w, h, ptex, ctex, itex)

        if self.show_keyframe_edges:
            with self.states.lock:
                ii = torch.tensor(self.states.edges_ii, dtype=torch.long)
                jj = torch.tensor(self.states.edges_jj, dtype=torch.long)
                if ii.numel() > 0 and jj.numel() > 0:
                    T_WCi = lietorch.Sim3(self.keyframes.T_WC[ii, 0])
                    T_WCj = lietorch.Sim3(self.keyframes.T_WC[jj, 0])
            if ii.numel() > 0 and jj.numel() > 0:
                t_WCi = T_WCi.matrix()[:, :3, 3].cpu().numpy()
                t_WCj = T_WCj.matrix()[:, :3, 3].cpu().numpy()
                self.lines.add(
                    t_WCi,
                    t_WCj,
                    thickness=self.line_thickness * self.scale,
                    color=[0, 1, 0, 1],
                )
        if self.show_curr_pointmap and self.states.get_mode() != Mode.INIT:
            if config["use_calib"]:
                curr_frame.K = self.keyframes.get_intrinsics()
            h, w = curr_frame.img_shape.flatten()
            X = self.frame_X(curr_frame)
            C = curr_frame.C.cpu().numpy().astype(np.float32)
            if "curr" not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures["curr"] = ptex, ctex, itex
                
                # Create the current-frame semantic texture (visualization-only).
                semantic_tex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.semantic_textures["curr"] = semantic_tex
            
            ptex, ctex, itex = self.textures["curr"]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())
            
            # Choose which texture is used to color the current-frame point cloud.
            if self.state.show_semantic and "curr" in self.semantic_textures:
                semantic_tex = self.semantic_textures["curr"]
                # IMPORTANT (per user request):
                #   Do NOT blend the semantic colors with the RGB image when coloring the point cloud.
                #   The point cloud should be colored by the semantic texture directly.
                if semantic_rgb is not None:
                    semantic_tex.write(semantic_rgb.astype(np.float32).tobytes())
                else:
                    semantic_tex.write(self.curr_img_np.astype(np.float32).tobytes())
                self.render_pointmap(
                    curr_frame.T_WC.cpu(),
                    w, h, ptex, ctex, semantic_tex,
                    use_img=True,
                    depth_bias=self.depth_bias,
                )
            else:
                itex.write(depth2rgb(X[..., -1], colormap="turbo"))
                self.render_pointmap(
                    curr_frame.T_WC.cpu(),
                    w, h, ptex, ctex, itex,
                    use_img=True,
                    depth_bias=self.depth_bias,
                )

        self.lines.render(self.camera)
        self.frustums.render(self.camera)
        self.render_ui()

    def render_ui(self):
        self.wnd.use()
        imgui.new_frame()

        io = imgui.get_io()
        # get window size and full screen
        window_size = io.display_size
        imgui.set_next_window_size(window_size[0], window_size[1])
        imgui.set_next_window_position(0, 0)
        self.viewport.render()

        imgui.set_next_window_size(
            window_size[0] / 4, 15 * window_size[1] / 16, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_position(
            32 * self.scale, 32 * self.scale, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_focus()
        imgui.begin("Semantic GUI", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        new_state = dataclasses.replace(self.state)
        _, new_state.is_paused = imgui.checkbox("pause", self.state.is_paused)

        imgui.spacing()
        _, new_state.C_conf_threshold = imgui.slider_float(
            "C_conf_threshold", self.state.C_conf_threshold, 0, 10
        )

        imgui.spacing()
        
        # Semantic Visualization Control
        imgui.text("Semantic Visualization Control")
        _, new_state.show_semantic = imgui.checkbox("Show Semantic", self.state.show_semantic)
        if new_state.show_semantic:
            # The user requested two checkboxes to select the semantic point cloud coloring source.
            # These flags affect ONLY visualization.
            changed_rgb, new_state.use_semantic_rgb_mask = imgui.checkbox(
                "Use Semantic RGB Mask", self.state.use_semantic_rgb_mask
            )
            changed_cache, new_state.use_semantic_pointmap_cache = imgui.checkbox(
                "Use Semantic PointMap Cache (V3)", self.state.use_semantic_pointmap_cache
            )
            # Enforce mutual exclusivity (requested by the user):
            #   - Only ONE source can be enabled at a time.
            #   - If the user turns one ON, we automatically turn the other OFF.
            #
            # Rationale:
            #   Rendering cannot simultaneously use two different semantic sources for the same
            #   point cloud. Keeping them mutually exclusive avoids ambiguous behavior.
            if changed_rgb and new_state.use_semantic_rgb_mask:
                new_state.use_semantic_pointmap_cache = False
            if changed_cache and new_state.use_semantic_pointmap_cache:
                new_state.use_semantic_rgb_mask = False
            # Safety: if both are somehow true, prefer the V3 cache (more stable).
            if new_state.use_semantic_rgb_mask and new_state.use_semantic_pointmap_cache:
                new_state.use_semantic_rgb_mask = False

            imgui.same_line()
            _, new_state.semantic_alpha = imgui.slider_float(
                "Semantic Alpha", self.state.semantic_alpha, 0.0, 1.0
            )
            imgui.spacing()
            # Use radio buttons for the overlay/semantic mode selection.
            imgui.text("Semantic Mode:")
            modes = ["overlay", "semantic"]
            for mode in modes:
                if imgui.radio_button(mode, self.state.semantic_mode == mode):
                    new_state.semantic_mode = mode
        imgui.spacing()

        imgui.text("Basic Control")

        _, self.show_all = imgui.checkbox("show all", self.show_all)
        imgui.same_line()
        _, self.follow_cam = imgui.checkbox("follow cam", self.follow_cam)

        imgui.spacing()
        shader_options = [
            "surfelmap.glsl",
            "trianglemap.glsl",
        ]
        current_shader = shader_options.index(
            self.pointmap_prog.extra["meta"].resolved_path.name
        )

        for i, shader in enumerate(shader_options):
            if imgui.radio_button(shader, current_shader == i):
                current_shader = i

        selected_shader = shader_options[current_shader]
        if selected_shader != self.pointmap_prog.extra["meta"].resolved_path.name:
            self.pointmap_prog = self.load_program(f"programs/{selected_shader}")

        imgui.spacing()

        _, self.show_keyframe_edges = imgui.checkbox(
            "show_keyframe_edges", self.show_keyframe_edges
        )
        imgui.spacing()

        _, self.pointmap_prog["show_normal"].value = imgui.checkbox(
            "show_normal", self.pointmap_prog["show_normal"].value
        )
        imgui.same_line()
        _, self.culling = imgui.checkbox("culling", self.culling)
        if "radius" in self.pointmap_prog:
            _, self.pointmap_prog["radius"].value = imgui.drag_float(
                "radius",
                self.pointmap_prog["radius"].value,
                0.0001,
                min_value=0.0,
                max_value=0.1,
            )
        if "slant_threshold" in self.pointmap_prog:
            _, self.pointmap_prog["slant_threshold"].value = imgui.drag_float(
                "slant_threshold",
                self.pointmap_prog["slant_threshold"].value,
                0.1,
                min_value=0.0,
                max_value=1.0,
            )
        _, self.show_keyframe = imgui.checkbox("show_keyframe", self.show_keyframe)
        _, self.show_curr_pointmap = imgui.checkbox(
            "show_curr_pointmap", self.show_curr_pointmap
        )
        _, self.show_axis = imgui.checkbox("show_axis", self.show_axis)
        _, self.line_thickness = imgui.drag_float(
            "line_thickness", self.line_thickness, 0.1, 10, 0.5
        )

        _, self.frustum_scale = imgui.drag_float(
            "frustum_scale", self.frustum_scale, 0.001, 0, 0.1
        )

        imgui.spacing()
        imgui.text("Image Display")
        gui_size = imgui.get_content_region_available()
        scale = gui_size[0] / self.curr_img.texture.size[0]
        scale = min(self.scale, scale)
        size = (
            self.curr_img.texture.size[0] * scale,
            self.curr_img.texture.size[1] * scale,
        )
        
        # Show original RGB images
        image_with_text(self.kf_img, size, "Keyframe RGB", same_line=False)
        image_with_text(self.curr_img, size, "Current RGB", same_line=False)
        
        # Show semantic images
        if self.state.show_semantic:
            image_with_text(self.kf_semantic_img, size, "Keyframe Semantic", same_line=False)
            image_with_text(self.curr_semantic_img, size, "Current Semantic", same_line=False)

        imgui.end()

        if new_state != self.state:
            # If semantic source toggles change, clear the semantic cache so the next frame
            # immediately reflects the new selection.
            if (
                new_state.use_semantic_rgb_mask != self.state.use_semantic_rgb_mask
                or new_state.use_semantic_pointmap_cache != self.state.use_semantic_pointmap_cache
            ):
                self._semantic_rgb_cache.clear()
            self.state = new_state
            self.send_msg()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def send_msg(self):
        self.viz2main.put(self.state)

    def render_pointmap(self, T_WC, w, h, ptex, ctex, itex, use_img=True, depth_bias=0):
        w, h = int(w), int(h)
        ptex.use(0)
        ctex.use(1)
        itex.use(2)
        model = T_WC.matrix().numpy().astype(np.float32).T

        vao = self.ctx.vertex_array(self.pointmap_prog, [], skip_errors=True)
        vao.program["m_camera"].write(self.camera.gl_matrix())
        vao.program["m_model"].write(model)
        vao.program["m_proj"].write(self.camera.proj_mat.gl_matrix())

        vao.program["pointmap"].value = 0
        vao.program["confs"].value = 1
        vao.program["img"].value = 2
        vao.program["width"].value = w
        vao.program["height"].value = h
        vao.program["conf_threshold"] = self.state.C_conf_threshold
        vao.program["use_img"] = use_img
        if "depth_bias" in self.pointmap_prog:
            vao.program["depth_bias"] = depth_bias
        vao.render(mode=moderngl.POINTS, vertices=w * h)
        vao.release()

    def frame_X(self, frame):
        if config["use_calib"]:
            Xs = frame.X_canon[None]
            if self.dP_dz is None:
                device = Xs.device
                dtype = Xs.dtype
                img_size = frame.img_shape.flatten()[:2]
                K = frame.K
                p = get_pixel_coords(
                    Xs.shape[0], img_size, device=device, dtype=dtype
                ).view(*Xs.shape[:-1], 2)
                tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
                tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
                self.dP_dz = torch.empty(
                    p.shape[:-1] + (3, 1), device=device, dtype=dtype
                )
                self.dP_dz[..., 0, 0] = tmp1
                self.dP_dz[..., 1, 0] = tmp2
                self.dP_dz[..., 2, 0] = 1.0
                self.dP_dz = self.dP_dz[..., 0].cpu().numpy().astype(np.float32)
            return (Xs[..., 2:3].cpu().numpy().astype(np.float32) * self.dP_dz)[0]

        return frame.X_canon.cpu().numpy().astype(np.float32)


def run_visualization(cfg, states, keyframes, main2viz, viz2main) -> None:
    set_global_config(cfg)

    config_cls = Window
    backend = "glfw"
    window_cls = mglw.get_local_window_cls(backend)

    window = window_cls(
        title=config_cls.title,
        size=config_cls.window_size,
        fullscreen=False,
        resizable=True,
        visible=True,
        gl_version=(3, 3),
        aspect_ratio=None,
        vsync=True,
        samples=4,
        cursor=True,
        backend=backend,
    )
    window.print_context_info()
    mglw.activate_context(window=window)
    window.ctx.gc_mode = "auto"
    timer = Timer()
    window_config = config_cls(
        states=states,
        keyframes=keyframes,
        main2viz=main2viz,
        viz2main=viz2main,
        ctx=window.ctx,
        wnd=window,
        timer=timer,
    )
    # Avoid the event assigning in the property setter for now
    # We want the even assigning to happen in WindowConfig.__init__
    # so users are free to assign them in their own __init__.
    window._config = weakref.ref(window_config)

    # Swap buffers once before staring the main loop.
    # This can trigged additional resize events reporting
    # a more accurate buffer size
    window.swap_buffers()
    window.set_default_viewport()

    timer.start()

    while not window.is_closing:
        current_time, delta = timer.next_frame()

        if window_config.clear_color is not None:
            window.clear(*window_config.clear_color)

        # Always bind the window framebuffer before calling render
        window.use()

        window.render(current_time, delta)
        if not window.is_closing:
            window.swap_buffers()

    state = window_config.state
    window.destroy()
    state.is_terminated = True
    viz2main.put(state)
