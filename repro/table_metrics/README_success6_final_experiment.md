# Success6 Final Experiment Commands

This README records the final six-scene protocol used for the reported
`3DGS`, `Ours ESDF`, and `Ours ESDF semantic-aware` comparison.

## Scenes

```bash
SCENES=(281bc17764 689fec23d7 7cd2ac43b4 8a20d62ac0 b26e64c4b0 bc03d88fc3)
```

## Common Paths

```bash
MAST3R_ROOT=/home/nuo/MASt3R-SLAM
SAFERSPLAT_ROOT=/home/nuo/carcrash/safer-splat
CBF_ROOT=/home/nuo/carcrash/safer-splat/cbfcontrol/safer-splat
DATA_ROOT=/home/nuo/carcrash/safer-splat/hdd0/datasets/Journey9ni_raw_data_part/scannetpp/data
MAST3R_LOG_ROOT=/home/nuo/MASt3R-SLAM/logs
RISK_OUT=/home/nuo/MASt3R-SLAM/outputs/semantic_risk_groups_balanced
PAIR_ROOT=/home/nuo/carcrash/safer-splat/trajs_local/success6_object_ring_eval
GS_OUT=/home/nuo/carcrash/safer-splat/trajs_local/success6_3dgs_goal_eqtime_matchbalanced_50x3_clear20
```

## 1. Reconstruct MASt3R-SLAM ESDF

Run in the `mast3r-slam` environment.

```bash
source /home/nuo/anaconda3/etc/profile.d/conda.sh
conda activate mast3r-slam
cd /home/nuo/MASt3R-SLAM

for SCENE in 281bc17764 689fec23d7 7cd2ac43b4 8a20d62ac0 b26e64c4b0 bc03d88fc3; do
  python main_semantic.py \
    --dataset "/home/nuo/carcrash/safer-splat/hdd0/datasets/Journey9ni_raw_data_part/scannetpp/data/${SCENE}/iphone/rgb" \
    --config config/base_subsample10.yaml \
    --no-viz \
    --efficientvit_dataset ade20k \
    --enable_semantic_input \
    --use_semantic_in_geo \
    --use_stable_semantic_in_geo \
    --semantic_beta 0.2 \
    --planning_pointcloud_outdir "/home/nuo/MASt3R-SLAM/logs/planning_pointcloud_scannetpp_${SCENE}_iphone" \
    --enable_planning_tsdf_publish \
    --planning_tsdf_radius_m 2.5 \
    --planning_tsdf_voxel_m 0.05 \
    --planning_tsdf_use_semantic \
    --planning_tsdf_semantic_band_m 0.1
done
```

Main outputs:

```text
/home/nuo/MASt3R-SLAM/logs/planning_pointcloud_scannetpp_<SCENE>_iphone/global_esdf_snapshot.npz
/home/nuo/MASt3R-SLAM/logs/planning_pointcloud_scannetpp_<SCENE>_iphone/reconstruction_timing.json
```

## 2. Train 3DGS Equal-Time Checkpoints

Run in the `safersplat` environment. This uses each scene's
`reconstruction_timing.json` to choose the equal-time 3DGS checkpoint.
Skip this step if the equal-time checkpoints already exist.

```bash
source /home/nuo/anaconda3/etc/profile.d/conda.sh
conda activate safersplat
bash /home/nuo/carcrash/safer-splat/scripts/train_scannetpp_iphone_3dgs_success6.sh
```

Final equal-time checkpoints used in the paper table:

```text
281bc17764  run=2026-04-01_233539  step=7085
689fec23d7  run=2026-04-01_231403  step=8357
7cd2ac43b4  run=2026-04-01_230456  step=7000
8a20d62ac0  run=2026-04-01_232128  step=5520
b26e64c4b0  run=2026-04-01_232837  step=6779
bc03d88fc3  run=2026-04-01_225814  step=9527
```

## 3. Export Scene-Specific Risk Configs

Run in the `mast3r-slam` environment.

```bash
source /home/nuo/anaconda3/etc/profile.d/conda.sh
conda activate mast3r-slam
cd /home/nuo/MASt3R-SLAM

python scripts/export_scene_specific_semantic_risk_configs.py \
  --scene-id 281bc17764 \
  --scene-id 689fec23d7 \
  --scene-id 7cd2ac43b4 \
  --scene-id 8a20d62ac0 \
  --scene-id b26e64c4b0 \
  --scene-id bc03d88fc3
```

Main outputs:

```text
/home/nuo/MASt3R-SLAM/outputs/semantic_risk_groups_balanced/<SCENE>_scene_specific_risk.json
```

## 4. Generate Matched Object-Ring Trajectories

Run in the `safersplat` environment.

Final trajectory protocol:

```text
matched sampling: rank_matched_cycle
target count: 50 per risk group
clearance threshold: 0.20 m
robot radius for generation: 0.03 m
reference mode: straight line with required straight-line collision
```

```bash
source /home/nuo/anaconda3/etc/profile.d/conda.sh
conda activate safersplat
cd /home/nuo/carcrash/safer-splat/cbfcontrol/safer-splat

for SCENE in 281bc17764 689fec23d7 7cd2ac43b4 8a20d62ac0 b26e64c4b0 bc03d88fc3; do
  python run_scannetpp_mesh_object_ring.py \
    --scene-id "${SCENE}" \
    --n-trajs 150 \
    --seed 0 \
    --sample-mode column3d \
    --radius 0.03 \
    --margin 0.02 \
    --object-standoff 0.20 \
    --ring-extra-min 0.05 \
    --ring-extra-max 0.35 \
    --angle-jitter-deg 15 \
    --risk-balanced \
    --n-per-risk 50 \
    --risk-config-json "/home/nuo/MASt3R-SLAM/outputs/semantic_risk_groups_balanced/${SCENE}_scene_specific_risk.json" \
    --risk-sampling-mode rank_matched_cycle \
    --traj-mode straight \
    --require-straight-collision \
    --min-start-goal-clearance 0.20 \
    --out "/home/nuo/carcrash/safer-splat/trajs_local/success6_object_ring_eval/scannetpp_${SCENE}_mesh_object_ring_matchbalanced_50x3_clear20.json"
done
```

The requested count is 150 per scene. The accepted count can be smaller
because the generator enforces clearance and straight-line collision constraints.

## 5. Run Ours ESDF Baseline

Run in the `safersplat` environment.

Controller settings:

```text
method=esdf
nominal_mode=goal
alpha=5.0
beta=1.0
dt=0.05
n_steps=500
radius=0.03
```

```bash
source /home/nuo/anaconda3/etc/profile.d/conda.sh
conda activate safersplat
cd /home/nuo/carcrash/safer-splat/cbfcontrol/safer-splat

mkdir -p /home/nuo/carcrash/safer-splat/trajs_local/success6_object_ring_eval/summaries

for SCENE in 281bc17764 689fec23d7 7cd2ac43b4 8a20d62ac0 b26e64c4b0 bc03d88fc3; do
  PAIRS="/home/nuo/carcrash/safer-splat/trajs_local/success6_object_ring_eval/scannetpp_${SCENE}_mesh_object_ring_matchbalanced_50x3_clear20.json"
  MESH="/home/nuo/carcrash/safer-splat/hdd0/datasets/Journey9ni_raw_data_part/scannetpp/data/${SCENE}/scans/mesh_aligned_0.05.ply"
  POSE_JSON="/home/nuo/carcrash/safer-splat/hdd0/datasets/Journey9ni_raw_data_part/scannetpp/data/${SCENE}/iphone/pose_intrinsic_imu.json"
  ESDF_SNAPSHOT="/home/nuo/MASt3R-SLAM/logs/planning_pointcloud_scannetpp_${SCENE}_iphone/global_esdf_snapshot.npz"
  OUT_JSON="/home/nuo/carcrash/safer-splat/trajs_local/success6_object_ring_eval/scannetpp_${SCENE}_esdf_goal_meshscale_matchbalanced_50x3_clear20_baseline.json"
  SUMMARY_JSON="/home/nuo/carcrash/safer-splat/trajs_local/success6_object_ring_eval/summaries/scannetpp_${SCENE}_esdf_goal_meshscale_matchbalanced_50x3_clear20_baseline_summary.json"
  NUM=$(python - <<PY
import json
from pathlib import Path
print(len(json.loads(Path("${PAIRS}").read_text())["total_data"]))
PY
)

  python run_scannetpp_esdf_cbf_ref.py \
    --scene-id "${SCENE}" \
    --esdf-snapshot "${ESDF_SNAPSHOT}" \
    --pairs "${PAIRS}" \
    --num "${NUM}" \
    --method esdf \
    --nominal-mode goal \
    --alpha 5.0 \
    --beta 1.0 \
    --dt 0.05 \
    --n-steps 500 \
    --radius 0.03 \
    --device cuda \
    --pose-json "${POSE_JSON}" \
    --out "${OUT_JSON}"

  python summarize_cbf_metrics.py \
    --traj-json "${OUT_JSON}" \
    --mesh "${MESH}" \
    --pairs "${PAIRS}" \
    --radius 0.03 \
    --save-json "${SUMMARY_JSON}"
done
```

## 6. Run Ours Semantic ESDF

Run in the `safersplat` environment.

Final semantic-aware setting:

```text
target-risk high dilation = 0.15 m
target-risk mid  dilation = 0.05 m
target-risk low  dilation = 0.00 m
```

```bash
source /home/nuo/anaconda3/etc/profile.d/conda.sh
conda activate safersplat
cd /home/nuo/carcrash/safer-splat/cbfcontrol/safer-splat

mkdir -p /home/nuo/carcrash/safer-splat/trajs_local/success6_object_ring_eval/summaries

for SCENE in 281bc17764 689fec23d7 7cd2ac43b4 8a20d62ac0 b26e64c4b0 bc03d88fc3; do
  PAIRS="/home/nuo/carcrash/safer-splat/trajs_local/success6_object_ring_eval/scannetpp_${SCENE}_mesh_object_ring_matchbalanced_50x3_clear20.json"
  MESH="/home/nuo/carcrash/safer-splat/hdd0/datasets/Journey9ni_raw_data_part/scannetpp/data/${SCENE}/scans/mesh_aligned_0.05.ply"
  POSE_JSON="/home/nuo/carcrash/safer-splat/hdd0/datasets/Journey9ni_raw_data_part/scannetpp/data/${SCENE}/iphone/pose_intrinsic_imu.json"
  ESDF_SNAPSHOT="/home/nuo/MASt3R-SLAM/logs/planning_pointcloud_scannetpp_${SCENE}_iphone/global_esdf_snapshot.npz"
  OUT_JSON="/home/nuo/carcrash/safer-splat/trajs_local/success6_object_ring_eval/scannetpp_${SCENE}_esdf_goal_meshscale_matchbalanced_50x3_clear20_targetrisk.json"
  SUMMARY_JSON="/home/nuo/carcrash/safer-splat/trajs_local/success6_object_ring_eval/summaries/scannetpp_${SCENE}_esdf_goal_meshscale_matchbalanced_50x3_clear20_targetrisk_summary.json"
  NUM=$(python - <<PY
import json
from pathlib import Path
print(len(json.loads(Path("${PAIRS}").read_text())["total_data"]))
PY
)

  python run_scannetpp_esdf_cbf_ref.py \
    --scene-id "${SCENE}" \
    --esdf-snapshot "${ESDF_SNAPSHOT}" \
    --pairs "${PAIRS}" \
    --num "${NUM}" \
    --method esdf \
    --nominal-mode goal \
    --alpha 5.0 \
    --beta 1.0 \
    --dt 0.05 \
    --n-steps 500 \
    --radius 0.03 \
    --device cuda \
    --pose-json "${POSE_JSON}" \
    --target-risk-high-dilation 0.15 \
    --target-risk-mid-dilation 0.05 \
    --target-risk-low-dilation 0.00 \
    --out "${OUT_JSON}"

  python summarize_cbf_metrics.py \
    --traj-json "${OUT_JSON}" \
    --mesh "${MESH}" \
    --pairs "${PAIRS}" \
    --radius 0.03 \
    --save-json "${SUMMARY_JSON}"
done
```

## 7. Run 3DGS / SaferSplat CBF

Run in the `safersplat` environment.

This script uses the same matched trajectories and the equal-time
checkpoints listed above.

```bash
source /home/nuo/anaconda3/etc/profile.d/conda.sh
conda activate safersplat
bash /home/nuo/carcrash/safer-splat/scripts/run_scannetpp_success6_3dgs_matchbalanced_clear20.sh
```

Background launcher:

```bash
source /home/nuo/anaconda3/etc/profile.d/conda.sh
conda activate safersplat
bash /home/nuo/carcrash/safer-splat/scripts/launch_success6_3dgs_matchbalanced_clear20_bg.sh
```

Main outputs:

```text
/home/nuo/carcrash/safer-splat/trajs_local/success6_3dgs_goal_eqtime_matchbalanced_50x3_clear20/scannetpp_<SCENE>_cbf_goal_meshscale_eqtime_matchbalanced_50x3_clear20.json
/home/nuo/carcrash/safer-splat/trajs_local/success6_3dgs_goal_eqtime_matchbalanced_50x3_clear20/summaries/scannetpp_<SCENE>_cbf_goal_meshscale_eqtime_matchbalanced_50x3_clear20_summary.json
```

## 8. Aggregate Final Tables

Run in the `safersplat` environment after all six 3DGS summaries exist.

```bash
source /home/nuo/anaconda3/etc/profile.d/conda.sh
conda activate safersplat
bash /home/nuo/carcrash/safer-splat/scripts/aggregate_success6_3methods.sh
```

Final outputs:

```text
/home/nuo/MASt3R-SLAM/outputs/semantic_risk_groups_balanced/success6_3methods_main_table.csv
/home/nuo/MASt3R-SLAM/outputs/semantic_risk_groups_balanced/success6_3methods_scene_rows.csv
```

The ESDF-only semantic trend tables are:

```text
/home/nuo/MASt3R-SLAM/outputs/semantic_risk_groups_balanced/success6_matchbalanced_main_table_v2.csv
/home/nuo/MASt3R-SLAM/outputs/semantic_risk_groups_balanced/success6_matchbalanced_subtable_v2.csv
/home/nuo/MASt3R-SLAM/outputs/semantic_risk_groups_balanced/success6_matchbalanced_subtable_delta_v2.csv
```

## 9. Final Visualization Command

Example for `bc03d88fc3`. This shows mesh, ESDF, 3DGS point cloud,
reference trajectories, 3DGS trajectories, and semantic ESDF trajectories.

```bash
source /home/nuo/anaconda3/etc/profile.d/conda.sh
conda activate safersplat
cd /home/nuo/carcrash/safer-splat/cbfcontrol/safer-splat

SCENE=bc03d88fc3
RUN_DIR=2026-04-01_225814
EQ_STEP=9527
PORT=18093

python visualize_scannetpp_rgbd_viser.py \
  --data-root /home/nuo/carcrash/safer-splat/hdd0/datasets/Journey9ni_raw_data_part/scannetpp/data \
  --scene-id ${SCENE} \
  --layers mesh,esdf,gsplat,traj \
  --pointcloud-space world \
  --mesh-style semantic \
  --mesh-opacity 0.35 \
  --esdf-snapshot /home/nuo/MASt3R-SLAM/logs/planning_pointcloud_scannetpp_${SCENE}_iphone/global_esdf_snapshot.npz \
  --esdf-style semantic \
  --esdf-align first-frame \
  --gsplat-config /home/nuo/carcrash/safer-splat/outputs/${SCENE}/splatfacto/${RUN_DIR}/config.yml \
  --gsplat-ckpt-step ${EQ_STEP} \
  --gsplat-space aligned_world \
  --gsplat-render points \
  --gsplat-max-splats 200000 \
  --gsplat-opacity-thresh 0.05 \
  --gsplat-point-size 0.003 \
  --traj-json-gs /home/nuo/carcrash/safer-splat/trajs_local/success6_3dgs_goal_eqtime_matchbalanced_50x3_clear20/scannetpp_${SCENE}_cbf_goal_meshscale_eqtime_matchbalanced_50x3_clear20.json \
  --traj-json-esdf /home/nuo/carcrash/safer-splat/trajs_local/success6_object_ring_eval/scannetpp_${SCENE}_esdf_goal_meshscale_matchbalanced_50x3_clear20_targetrisk.json \
  --traj-json-ref /home/nuo/carcrash/safer-splat/trajs_local/success6_object_ring_eval/scannetpp_${SCENE}_mesh_object_ring_matchbalanced_50x3_clear20.json \
  --traj-max 999 \
  --traj-select first \
  --traj-every 2 \
  --traj-colormap tab20 \
  --traj-gs-style dashed \
  --traj-esdf-style solid \
  --traj-dash-len 0.08 \
  --traj-gap-len 0.055 \
  --traj-gs-line-width 7 \
  --traj-esdf-line-width 7 \
  --traj-ref-line-width 3 \
  --traj-ref-color-mode fixed \
  --traj-ref-color 230 230 230 \
  --traj-note none \
  --host 127.0.0.1 \
  --port ${PORT}
```

Other equal-time 3DGS run IDs:

```text
281bc17764  RUN_DIR=2026-04-01_233539  EQ_STEP=7085   PORT=18094
689fec23d7  RUN_DIR=2026-04-01_231403  EQ_STEP=8357   PORT=18095
7cd2ac43b4  RUN_DIR=2026-04-01_230456  EQ_STEP=7000   PORT=18096
8a20d62ac0  RUN_DIR=2026-04-01_232128  EQ_STEP=5520   PORT=18097
b26e64c4b0  RUN_DIR=2026-04-01_232837  EQ_STEP=6779   PORT=18098
bc03d88fc3  RUN_DIR=2026-04-01_225814  EQ_STEP=9527   PORT=18093
```

## Metric Definition

`Minimum Clearance to Mesh` is computed from the ground-truth mesh for all
methods:

```text
minimum clearance = min point-to-mesh distance along rollout - robot radius
```

Collision rate is the fraction of trajectories with negative minimum
mesh clearance. Backend-specific CBF safety values are not used for the
main cross-method safety comparison.

