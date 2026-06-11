import torch
from mast3r_slam.frame import Frame
from mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_slam.nonlinear_optimizer import check_convergence, huber
from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric
from mast3r_slam.semantic_stabilizer import (
    ensure_hard_label_hw,
    label_code_to_rgb,
    semantic_bonus_sqrt_factor_v0,
    semantic_pointmap_update_v0,
    semantic_warp_and_fuse_v0,
)
from mast3r_slam.depth_stabilizer import (
    fill_holes_by_neighbor_average_v0,
    warp_keyframe_depth_to_frame_v0,
)


class FrameTracker:
    def __init__(self, model, frames, device):
        self.cfg = config["tracking"]
        self.model = model
        self.keyframes = frames
        self.device = device

        self.reset_idx_f2k()

        # --- Semantic stabilization cache (last keyframe only) ---
        # We cache the *stable hard label* of the last keyframe on GPU to avoid
        # re-uploading and re-decoding the keyframe semantic mask every frame.
        #
        # Tracking always uses `last_keyframe()` (no multi-keyframe fusion here), so
        # caching only the last keyframe is sufficient and keeps changes minimal.
        self._stable_label_k_frame_id = None
        self._stable_label_k = None

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self):
        self.idx_f2k = None

    def track(self, frame: Frame):
        keyframe = self.keyframes.last_keyframe()

        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
        )
        # Save idx for next
        self.idx_f2k = idx_f2k.clone()

        # Get rid of batch dim
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        # ---------------------------------------------------------------------
        # (A) Semantic warp + fuse (V0, hard labels, O(N), no per-frame sorting)
        #
        # Inserted strictly AFTER matching (idx_f2k/Qk/valid_match_k) and BEFORE pose
        # optimization, as requested.
        #
        # Inputs (conceptually):
        #   - label_vit_f:    current-frame segmentation hard label (H,W) int
        #   - label_stable_k: last-keyframe cached stable hard label (H,W) int
        #   - idx_k2f:        k->f pixel mapping (H*W)
        #   - q/valid:        match confidence + validity for each keyframe pixel k
        #
        # Implementation notes:
        #   - The system stores semantic masks for visualization as RGB (H,W,3) float on CPU.
        #   - For stabilization we convert to a hard-label (H,W) int64:
        #       * RGB -> palette-agnostic "label code" (R<<16|G<<8|B)
        #       * int labels -> use directly
        #   - We keep `frame.semantic_label` on CPU because SharedStates.set_frame expects CPU.
        # ---------------------------------------------------------------------
        sem_cfg = config.get("semantic", {})
        enable_semantic_warp = bool(sem_cfg.get("enable_semantic_warp", True))
        use_semantic_in_geo = bool(sem_cfg.get("use_semantic_in_geo", False))
        enable_semantic_pointmap = bool(sem_cfg.get("enable_semantic_pointmap", False))
        semantic_pointmap_use_q = bool(sem_cfg.get("semantic_pointmap_use_q", True))
        semantic_pointmap_momentum = float(sem_cfg.get("semantic_pointmap_momentum", 1.0))
        # Default to RAW semantics for geometry reweighting to keep it an "independent observation".
        # Using stabilized semantics here can become "self-fulfilling" because A-warp overwrites
        # frame labels using keyframe labels along matches, making label agreement nearly constant.
        use_stable_semantic_in_geo = bool(sem_cfg.get("use_stable_semantic_in_geo", False))
        semantic_beta = float(sem_cfg.get("semantic_beta", 0.2))
        tau_warp = float(sem_cfg.get("semantic_tau_warp", self.cfg.get("Q_conf", 0.0)))
        tau_sem = sem_cfg.get("semantic_tau", None)
        tau_sem = None if tau_sem is None else float(tau_sem)
        debug_sem_geo_stats = bool(sem_cfg.get("debug_semantic_geo_stats", False))
        debug_sem_geo_every = int(sem_cfg.get("debug_semantic_geo_every", 30))
        debug_sem_geo_first = int(sem_cfg.get("debug_semantic_geo_first", 5))

        # ---------------------------------------------------------------------
        # Depth source selection for downstream tasks (e.g., planning)
        #
        # This is intentionally *read-only* with respect to SLAM:
        #   - It does NOT affect matching, valid/gating, or pose optimization.
        #   - It only computes an auxiliary per-frame depth map for downstream use.
        #
        # Supported modes (V0):
        #   - "raw_z":     use current-frame pointmap Z directly (fastest, may jitter).
        #   - "kf_warp_z": warp keyframe fused pointmap into current frame using matches+pose,
        #                 then fill holes by a few iterations of neighbor propagation.
        #
        # Output tensors stored on `frame` (if computed):
        #   - frame._depth_hw       : (H,W) float32
        #   - frame._depth_hw_valid : (H,W) bool
        # ---------------------------------------------------------------------
        depth_cfg = config.get("depth", {})
        depth_source = str(depth_cfg.get("depth_source", "raw_z"))
        depth_tau = float(depth_cfg.get("depth_tau", self.cfg.get("Q_conf", 0.0)))
        depth_fill_iters = int(depth_cfg.get("depth_fill_iters", 2))
        depth_fill_kernel = int(depth_cfg.get("depth_fill_kernel", 3))

        # Fast exit: if semantic is not provided and we have no cached stable keyframe label,
        # skip all semantic work to avoid any unnecessary GPU<->CPU sync.
        label_vit_f = None
        label_stable_k = None
        label_stable_f = None
        if (enable_semantic_warp or use_semantic_in_geo or enable_semantic_pointmap) and (
            frame.semantic_label is not None
            or keyframe.semantic_label is not None
            or self._stable_label_k is not None
            or getattr(keyframe, "sem_label", None) is not None
        ):
            # Semantic target resolution must match MASt3R match grid (H,W) so that H*W == len(idx_k2f).
            h_s, w_s = (int(x) for x in frame.img_shape.flatten().tolist())

            # Convert current-frame semantic to hard label on GPU (if provided).
            if frame.semantic_label is not None:
                label_vit_f = ensure_hard_label_hw(
                    frame.semantic_label, size_hw=(h_s, w_s), device=idx_f2k.device
                )
                # Preserve the *raw* per-frame hard labels for later use (V3 initialization).
                #
                # Why:
                #   (A) semantic warp may overwrite `frame.semantic_label` (for visualization/output),
                #   but V3 explicitly needs the raw per-frame observation as the keyframe init label.
                #
                # Shape:
                #   (H, W) int64 on the match grid.
                frame._semantic_label_hw_raw = label_vit_f.detach()

            # Convert last-keyframe semantic to hard label on GPU (prefer cached stable label).
            if enable_semantic_warp or use_semantic_in_geo or enable_semantic_pointmap:
                if (
                    self._stable_label_k is not None
                    and self._stable_label_k_frame_id == int(keyframe.frame_id)
                    and tuple(self._stable_label_k.shape) == (h_s, w_s)
                ):
                    label_stable_k = self._stable_label_k
                elif enable_semantic_pointmap and (getattr(keyframe, "sem_label", None) is not None):
                    # V3: when semantic pointmap is enabled, prefer the keyframe semantic cache
                    # as the keyframe-side "stable" semantic source.
                    #
                    # This cache is updated over time using matches+confidence, so it becomes
                    # progressively more stable than any single-frame segmentation.
                    if tuple(keyframe.sem_label.shape) == (h_s, w_s):
                        label_stable_k = keyframe.sem_label
                        self._stable_label_k_frame_id = int(keyframe.frame_id)
                        self._stable_label_k = label_stable_k
                elif keyframe.semantic_label is not None:
                    label_stable_k = ensure_hard_label_hw(
                        keyframe.semantic_label,
                        size_hw=(h_s, w_s),
                        device=idx_f2k.device,
                    )
                    self._stable_label_k_frame_id = int(keyframe.frame_id)
                    self._stable_label_k = label_stable_k

            # Run semantic warp+fuse if enabled and inputs are available.
            if enable_semantic_warp and (label_vit_f is not None) and (label_stable_k is not None):
                label_stable_f = semantic_warp_and_fuse_v0(
                    label_vit_f=label_vit_f,
                    label_stable_k=label_stable_k,
                    idx_k2f=idx_f2k,
                    q=Qk,
                    valid=valid_match_k,
                    tau_warp=tau_warp,
                )

                # Keep shared-memory visualization unchanged: store RGB (H,W,3) float on CPU.
                frame.semantic_label = label_code_to_rgb(label_stable_f).cpu()

        # Update keyframe pointmap after registration (need pose)
        frame.update_pointmap(Xff, Cff)

        use_calib = config["use_calib"]
        img_size = frame.img.shape[-2:]
        if use_calib:
            K = keyframe.K
        else:
            K = None

        # Get poses and point correspondneces and confidences
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        # Get valid
        # Use canonical confidence average
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_kf = valid_match_k & valid_Q

        match_frac = valid_opt.sum() / valid_opt.numel()
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
            return False, [], True

        try:
            # Track
            # -----------------------------------------------------------------
            # (B) Optional semantic reweighting for geometry (V1.5 symmetric +/- beta)
            #
            # If enabled, we compute a per-match sqrt semantic factor (hard labels):
            #   same = I[label_f[idx_k2f[k]] == label_k[k]]
            #   g    = (1 + beta) if same else (1 - beta)
            #   sqrt_factor = sqrt(g)
            #
            # Critical requirement:
            #   - This bonus MUST NOT affect match gating (valid_opt) or Huber gating.
            #   - We therefore apply it inside `solve()` AFTER the robust kernel weight.
            # -----------------------------------------------------------------
            bonus_sqrt_factor = None
            if use_semantic_in_geo and (label_stable_k is not None):
                # Decide which frame semantic to use (ablation):
                #   - Recommended/default: use raw per-frame semantic as an *independent observation*
                #     (avoids "self-fulfilling" same-label tests when A-warp overwrites the frame label).
                #   - If stable semantic is explicitly requested but warp is disabled/unavailable,
                #     fall back to raw vit.
                label_used_f = None
                if use_stable_semantic_in_geo and (label_stable_f is not None):
                    label_used_f = label_stable_f
                elif label_vit_f is not None:
                    label_used_f = label_vit_f

                if label_used_f is not None:
                    bonus_sqrt_factor = semantic_bonus_sqrt_factor_v0(
                        label_k=label_stable_k,
                        label_f=label_used_f,
                        idx_k2f=idx_f2k,
                        q=Qk,
                        valid=valid_match_k,
                        beta=semantic_beta,
                        tau_sem=tau_sem,
                    )
                    if debug_sem_geo_stats:
                        # Print lightweight diagnostics to confirm whether the semantic factor is:
                        #   - not applied at all (mean==1, std==0)
                        #   - near-constant scaling (std~=0, mean!=1)
                        #   - non-trivial reweighting (std>0)
                        #
                        # IMPORTANT:
                        #   This is purely for debugging. It does NOT change any gating/valid logic.
                        fid = int(frame.frame_id)
                        if fid < debug_sem_geo_first or (
                            debug_sem_geo_every > 0 and fid % debug_sem_geo_every == 0
                        ):
                            valid_flat = valid_match_k.reshape(-1).to(torch.bool)
                            q_flat = Qk.reshape(-1).to(torch.float32)
                            mask_flat = valid_flat
                            if tau_sem is not None:
                                mask_flat = mask_flat & (q_flat > float(tau_sem))

                            applied_ratio = (
                                float(mask_flat.float().mean().item())
                                if mask_flat.numel() > 0
                                else 0.0
                            )

                            sf = bonus_sqrt_factor.reshape(-1).to(torch.float32)
                            # These .item() calls synchronize GPU->CPU, but only when debug is enabled
                            # and only on a sparse schedule (first few frames / every N frames).
                            sf_mean = float(sf.mean().item()) if sf.numel() > 0 else 1.0
                            sf_std = float(sf.std(unbiased=False).item()) if sf.numel() > 0 else 0.0
                            sf_min = float(sf.min().item()) if sf.numel() > 0 else 1.0
                            sf_max = float(sf.max().item()) if sf.numel() > 0 else 1.0
                            print(
                                "[SemanticGeoDebug] "
                                f"frame={fid} beta={semantic_beta:.3f} "
                                f"tau_sem={tau_sem if tau_sem is not None else 'None'} "
                                f"applied_ratio={applied_ratio:.3f} "
                                f"sqrt_factor(mean/std/min/max)={sf_mean:.6f}/{sf_std:.6f}/{sf_min:.6f}/{sf_max:.6f}"
                            )

            if not use_calib:
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt, bonus_sqrt_factor
                )
            else:
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    meas_k,
                    valid_meas_k,
                    K,
                    img_size,
                    bonus_sqrt_factor,
                )
        except Exception as e:
            print(f"Cholesky failed {frame.frame_id}")
            return False, [], True

        frame.T_WC = T_WCf

        # -----------------------------------------------------------------
        # Depth output for downstream tasks (optional, does not affect SLAM)
        #
        # We compute/store a depth map on the `frame` object to be consumed by
        # downstream logic (e.g., planning) in the main process.
        #
        # Shape convention:
        #   - depth maps are on the match grid (same H×W as pointmap/matches).
        #   - dtype is float32.
        # -----------------------------------------------------------------
        frame._depth_hw = None
        frame._depth_hw_valid = None
        if depth_source == "raw_z":
            # Raw depth from the current frame pointmap: z channel of (H*W,3).
            # This is extremely fast but may jitter because it is per-frame.
            if frame.X_canon is not None:
                h_s, w_s = (int(x) for x in frame.img_shape.flatten().tolist())
                depth_hw = frame.X_canon.reshape(-1, 3)[:, 2].view(h_s, w_s).to(torch.float32)
                frame._depth_hw = depth_hw
                frame._depth_hw_valid = torch.isfinite(depth_hw) & (depth_hw > 0.0)
        elif depth_source == "kf_warp_z":
            # Keyframe-warp depth:
            #   - Warp keyframe fused pointmap into current frame camera using T_CkCf^{-1}
            #   - Scatter by idx_k2f under valid & (q>tau)
            #   - Fill holes via a few iterations of neighbor propagation (no raw depth)
            if keyframe.X_canon is not None:
                h_s, w_s = (int(x) for x in frame.img_shape.flatten().tolist())
                depth_hw, mask_hw = warp_keyframe_depth_to_frame_v0(
                    Xk_canon=keyframe.X_canon,
                    T_CkCf=T_CkCf,
                    idx_k2f=idx_f2k,
                    q=Qk,
                    valid=valid_match_k,
                    tau=depth_tau,
                    size_hw=(h_s, w_s),
                )
                depth_hw, mask_hw = fill_holes_by_neighbor_average_v0(
                    depth_hw=depth_hw,
                    mask_hw=mask_hw,
                    iters=depth_fill_iters,
                    kernel_size=depth_fill_kernel,
                )
                frame._depth_hw = depth_hw
                frame._depth_hw_valid = mask_hw & torch.isfinite(depth_hw) & (depth_hw > 0.0)
        else:
            # Unknown mode: do not crash SLAM; leave depth unset.
            frame._depth_hw = None
            frame._depth_hw_valid = None

        # Use pose to transform points to update keyframe
        Xkk = T_CkCf.act(Xkf)
        keyframe.update_pointmap(Xkk, Ckf)

        # -----------------------------------------------------------------
        # V3: Semantic PointMap update (keyframe fusion)
        #
        # Inserted at the same location as geometric pointmap fusion (right after
        # `keyframe.update_pointmap(...)`), using the same already-computed k->f matches.
        #
        # IMPORTANT:
        #   - This only updates keyframe semantic cache (kf.sem_label / kf.sem_weight).
        #   - It does NOT modify matching, valid/gating, or the geometry optimizer.
        #   - When the V3 flag is disabled, this block is a no-op and V1 behavior is unchanged.
        # -----------------------------------------------------------------
        if (
            enable_semantic_pointmap
            and (label_vit_f is not None)
            and (getattr(keyframe, "sem_label", None) is not None)
            and (getattr(keyframe, "sem_weight", None) is not None)
        ):
            semantic_pointmap_update_v0(
                kf_sem_label=keyframe.sem_label,
                kf_sem_weight=keyframe.sem_weight,
                raw_label_f=label_vit_f,
                idx_k2f=idx_f2k,
                q=Qk,
                valid=valid_match_k,
                use_q=semantic_pointmap_use_q,
                momentum=semantic_pointmap_momentum,
            )
            # Keep the GPU cache in sync: subsequent frames will warp from the updated keyframe
            # semantic pointmap without any extra conversions.
            self._stable_label_k_frame_id = int(keyframe.frame_id)
            self._stable_label_k = keyframe.sem_label.detach()
        # write back the fitered pointmap
        self.keyframes[len(self.keyframes) - 1] = keyframe

        # Keyframe selection
        n_valid = valid_kf.sum()
        match_frac_k = n_valid / valid_kf.numel()
        unique_frac_f = (
            torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]

        # If this frame becomes a new keyframe, update the cached stable keyframe label.
        # This makes the next frame warp from a stable anchor without re-uploading/re-decoding.
        if new_kf:
            if enable_semantic_pointmap and (label_vit_f is not None):
                # V3: the keyframe semantic cache is initialized from the RAW per-frame labels.
                # We mirror that here in the GPU cache so the next frame can warp immediately.
                self._stable_label_k_frame_id = int(frame.frame_id)
                self._stable_label_k = label_vit_f.detach()
            elif enable_semantic_warp and (label_stable_f is not None):
                self._stable_label_k_frame_id = int(frame.frame_id)
                self._stable_label_k = label_stable_f.detach()
            elif label_vit_f is not None:
                # Warp disabled/unavailable: treat raw label as the "stable" cache for consistency.
                self._stable_label_k_frame_id = int(frame.frame_id)
                self._stable_label_k = label_vit_f.detach()
            else:
                self._stable_label_k_frame_id = None
                self._stable_label_k = None

        # Rest idx if new keyframe
        if new_kf:
            self.reset_idx_f2k()

        return (
            new_kf,
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )

    def get_points_poses(self, frame, keyframe, idx_f2k, img_size, use_calib, K=None):
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC

        # Average confidence
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        meas_k = None
        valid_meas_k = None

        if use_calib:
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # Setup pixel coordinates
            uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
            uv_k = uv_k.view(-1, 2)
            meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
            # Avoid any bad calcs in log
            valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
            meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xk, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(self, sqrt_info, r, J, bonus_sqrt_factor=None):
        """
        Solve one Gauss-Newton step with robust (Huber) weighting.

        Parameters:
          sqrt_info: (..., D) float
            Base square-root information (includes geometric validity + match confidence).
            IMPORTANT: this must NOT include the semantic bonus factor, otherwise the robust
            kernel (Huber) would see a scaled residual and could change its inlier/outlier behavior.

          bonus_sqrt_factor: (..., 1) float | None
            Optional multiplicative factor applied *after* robust weighting.
            In V1.5 we use a symmetric factor (1 +/- beta) based on hard-label agreement.
            Applying it AFTER Huber guarantees we do not affect match gating (valid masks)
            or robust gating (Huber inlier/outlier behavior), because Huber is computed from
            `whitened_r = sqrt_info * r` before this factor is applied.
        """

        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        if bonus_sqrt_factor is not None:
            # Apply semantic factor only here (after Huber), so it cannot change robust gating.
            robust_sqrt_info = robust_sqrt_info * bonus_sqrt_factor
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid, bonus_sqrt_factor=None):
        last_error = 0
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        # Precalculate distance and ray for obs k
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # r = z-h(x)
            r = rd_k - rd_f_Ck
            # Jacobian
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(
                sqrt_info, r, J, bonus_sqrt_factor=bonus_sqrt_factor
            )
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf

    def opt_pose_calib_sim3(
        self,
        Xf,
        Xk,
        T_WCf,
        T_WCk,
        Qk,
        valid,
        meas_k,
        valid_meas_k,
        K,
        img_size,
        bonus_sqrt_factor=None,
    ):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            valid2 = valid_proj & valid_meas_k
            sqrt_info2 = valid2 * sqrt_info

            # r = z-h(x)
            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(
                sqrt_info2, r, J, bonus_sqrt_factor=bonus_sqrt_factor
            )
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf
