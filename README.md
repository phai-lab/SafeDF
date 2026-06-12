<div align="center">

# Embedding Semantic Risk into Distance Fields and CBFs for Online Monocular Safe Control

<p>
  <a href="https://safedf.github.io/index.html"><img src="https://img.shields.io/badge/Project-Page-6f42c1" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2606.01605"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper"></a>
  <a href="#bibtex"><img src="https://img.shields.io/badge/Cite-BibTeX-blue" alt="Cite"></a>
</p>

<p>
  <b>Dawei Zhang</b><sup>1*</sup>,
  <b>Nuo Chen</b><sup>3*</sup>,
  <b>Shuo Liu</b><sup>2</sup>,
  <b>Roberto Tron</b><sup>1</sup>,
  <b>Zhiwen Fan</b><sup>3</sup>
  <br>
  <sup>1</sup>Boston University &nbsp;&nbsp;
  <sup>2</sup>Boston University Mechanical Engineering &nbsp;&nbsp;
  <sup>3</sup>Texas A&amp;M University ECE
  <br>
  <sup>*</sup>Equal contribution.
</p>

<img src="assets/overview.png" alt="SafeDF system overview" width="96%">

</div>

## Overview
SafeDF is an online monocular perception-to-control framework that embeds semantic risk into the distance field used by Control Barrier Function (CBF)-based safe navigation and teleoperation. From RGB video, a foundation-model-based SLAM front end reconstructs dense 3-D geometry, while semantic observations are fused into the reconstructed scene. The resulting geometric-semantic representation is converted into a semantic-aware ESDF, where semantic labels identify safety-relevant regions and impose class-dependent inflation before field computation.

## Checkpoints and Data
SafeDF expects model weights in the following locations:

```text
checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
efficientvit/l2.pt
```

Download the MASt3R weights from the MASt3R release and the EfficientViT
ADE20K segmentation weight from HuggingFace:

```bash
mkdir -p checkpoints efficientvit

wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
  -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl \
  -P checkpoints/
wget https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l2_ade20k.pt \
  -O efficientvit/l2.pt
```

The simulation benchmark uses the six ScanNet++ scenes listed below. Download
ScanNet++ from the [official dataset page](https://scannetpp.mlsg.cit.tum.de/scannetpp/)
and arrange each scene as `${DATA_ROOT}/${SCENE}/iphone/rgb` with the aligned
mesh at `${DATA_ROOT}/${SCENE}/scans/mesh_aligned_0.05.ply`.

## Installation
Create a clean environment named `safedf`:

```bash
git clone https://github.com/phai-lab/SafeDF.git
cd SafeDF

conda create -n safedf python=3.11 -y
conda activate safedf

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install -e efficientvit
pip install -r requirements_semantic_esdf.txt
pip install --no-build-isolation -e .
```

## Semantic ESDF Reconstruction
To quickly try SafeDF on a monocular RGB video, run the semantic ESDF
reconstruction with the non-subsampled configuration:

```bash
python main_semantic.py \
  --dataset /path/to/video.mp4 \
  --config config/base.yaml \
  --save-as example_mp4 \
  --no-viz \
  --stream_img_size 224 \
  --efficientvit_dataset ade20k \
  --enable_semantic_input \
  --use_semantic_in_geo \
  --use_stable_semantic_in_geo \
  --semantic_beta 0.2 \
  --planning_pointcloud_outdir logs/example_mp4_esdf \
  --enable_planning_tsdf_publish \
  --planning_tsdf_radius_m 2.5 \
  --planning_tsdf_voxel_m 0.05 \
  --planning_tsdf_use_semantic \
  --planning_tsdf_semantic_band_m 0.1 \
  --planning_pointcloud_no_save_images
```

This writes the semantic ESDF snapshot to:

```text
logs/example_mp4_esdf/global_esdf_snapshot.npz
```

For a higher-fidelity reconstruction, use a larger MASt3R input resolution and
a smaller TSDF voxel size:

```bash
python main_semantic.py \
  --dataset /path/to/video.mp4 \
  --config config/base_subsample10.yaml \
  --save-as example \
  --no-viz \
  --stream_img_size 512 \
  --efficientvit_dataset ade20k \
  --enable_semantic_input \
  --use_semantic_in_geo \
  --use_stable_semantic_in_geo \
  --semantic_beta 0.2 \
  --planning_pointcloud_outdir logs/example_semantic_esdf \
  --enable_planning_tsdf_publish \
  --planning_tsdf_radius_m 2.5 \
  --planning_tsdf_voxel_m 0.025 \
  --planning_tsdf_use_semantic \
  --planning_tsdf_semantic_band_m 0.1 \
  --planning_pointcloud_no_save_images
```

This writes:

```text
logs/example_semantic_esdf/global_esdf_snapshot.npz
```

## Visualization
Interactive Viser visualization:

```bash
python scripts/view_esdf_snapshot_viser.py \
  --snapshot logs/example_semantic_esdf/global_esdf_snapshot.npz \
  --mesh-style semantic \
  --host 127.0.0.1 \
  --port 8080
```

Render an RGB/semantic-ESDF comparison video with the same camera trajectory:

```bash
python scripts/render_rgb_esdf_side_by_side.py \
  --video /path/to/video.mp4 \
  --snapshot logs/example_semantic_esdf/global_esdf_snapshot.npz \
  --traj logs/example_dense.txt \
  --out logs/example_semantic_esdf/rgb_semantic_esdf.mp4 \
  --size 720 \
  --semantic-mesh \
  --background 0.88,0.89,0.90 \
  --lighting-profile soft \
  --sun-intensity 22000 \
  --ibl-intensity 6000 \
  --roughness 1.0 \
  --match-video-length
```

Batch processing for a folder of videos:

```bash
bash scripts/run_video_esdf_batch.sh /path/to/videos logs/video_esdf_batch
```

For quality-oriented offline rendering, override the resolution and voxel size:

```bash
STREAM_IMG_SIZE=512 TSDF_VOXEL_M=0.025 \
  bash scripts/run_video_esdf_batch.sh /path/to/videos logs/video_esdf_batch_hifi
```

## Simulation Benchmark
The simulation benchmark uses six ScanNet++ scenes and reports the released
SafeDF table artifacts. This repository provides the semantic ESDF
reconstruction path, scene-specific semantic risk export, and table verification
scripts.

Set the common paths first:

```bash
MAST3R_ROOT=/path/to/SafeDF
DATA_ROOT=/path/to/scannetpp/data
RISK_OUT=${MAST3R_ROOT}/outputs/semantic_risk_groups_balanced
SCENES=(281bc17764 689fec23d7 7cd2ac43b4 8a20d62ac0 b26e64c4b0 bc03d88fc3)
```

Reconstruct the ESDF snapshots for the six ScanNet++ scenes:

```bash
conda activate safedf
cd ${MAST3R_ROOT}

for SCENE in "${SCENES[@]}"; do
  python main_semantic.py \
    --dataset "${DATA_ROOT}/${SCENE}/iphone/rgb" \
    --config config/base_subsample10.yaml \
    --no-viz \
    --efficientvit_dataset ade20k \
    --enable_semantic_input \
    --use_semantic_in_geo \
    --use_stable_semantic_in_geo \
    --semantic_beta 0.2 \
    --planning_pointcloud_outdir "${MAST3R_ROOT}/logs/planning_pointcloud_scannetpp_${SCENE}_iphone" \
    --enable_planning_tsdf_publish \
    --planning_tsdf_radius_m 2.5 \
    --planning_tsdf_voxel_m 0.05 \
    --planning_tsdf_use_semantic \
    --planning_tsdf_semantic_band_m 0.1
done
```

Export the scene-specific semantic risk groups. The mapping CSV in
`resources/mesh_to_ade20k_all.csv` maps ScanNet++ mesh labels to the ADE20K
labels predicted by EfficientViT; the export script then balances scene objects
into low-, mid-, and high-risk groups for the benchmark.

```bash
conda activate safedf
cd ${MAST3R_ROOT}

python scripts/export_scene_specific_semantic_risk_configs.py \
  --data-root "${DATA_ROOT}" \
  --mapping-csv "${MAST3R_ROOT}/resources/mesh_to_ade20k_all.csv" \
  --outdir "${RISK_OUT}" \
  --scene-id 281bc17764 \
  --scene-id 689fec23d7 \
  --scene-id 7cd2ac43b4 \
  --scene-id 8a20d62ac0 \
  --scene-id b26e64c4b0 \
  --scene-id bc03d88fc3
```

## Hardware Streaming
For LIMO-style deployment, SafeDF can read RGB frames from a ZMQ bridge and optionally publish joystick velocity commands:

```bash
python main_semantic.py \
  --input_source limo_zmq \
  --bridge_ip 127.0.0.1 \
  --bridge_vid_port 5555 \
  --bridge_cmd_port 5556 \
  --enable_joystick \
  --config config/base.yaml \
  --stream_img_size 224 \
  --efficientvit_dataset ade20k \
  --enable_semantic_input \
  --use_semantic_in_geo \
  --use_stable_semantic_in_geo \
  --semantic_beta 0.2 \
  --planning_pointcloud_outdir logs/limo_semantic_esdf \
  --enable_planning_tsdf_publish \
  --planning_tsdf_radius_m 2.5 \
  --planning_tsdf_voxel_m 0.05 \
  --planning_tsdf_use_semantic \
  --planning_tsdf_semantic_band_m 0.1
```

Use `--stream_img_size 512` and `--planning_tsdf_voxel_m 0.025` only when
prioritizing reconstruction quality over online speed. The robot-side ROS/ZMQ
bridge and controller parameters are deployment-specific; this repository
exposes the reconstruction and semantic ESDF side used by the hardware
experiments.

## BibTeX
```bibtex
@misc{zhang2026embeddingsemanticriskdistance,
      title={Embedding Semantic Risk into Distance Fields and CBFs for Online Monocular Safe Control}, 
      author={Dawei Zhang and Nuo Chen and Shuo Liu and Roberto Tron and Zhiwen Fan},
      year={2026},
      eprint={2606.01605},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.01605}, 
}
```

## Acknowledgements
This code builds on
[MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM) for online monocular
dense SLAM and [EfficientViT](https://github.com/mit-han-lab/efficientvit) for
semantic segmentation. We thank the authors of these projects for releasing
their code and models.
