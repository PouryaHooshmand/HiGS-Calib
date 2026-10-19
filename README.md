# HiGS-Calib: A Hierarchical 3D Gaussian Splatting based Targetless Local-Consistent LiDAR-Camera Calibration Method

This repository contains the implementation of **HiGS-Calib**, a targetless, multi-modal LiDAR-camera extrinsic calibration framework that couples hierarchical 3D Gaussian Splatting (3DGS) with deep dense optical flow (RoMa) and photometric edge alignment to achieve accurate and robust sensor calibration in natural driving environments without calibration targets.

---

## Table of Contents

- [Key Improvements & Features](#key-improvements--features)
- [System Requirements](#system-requirements)
- [Installation & Environment Setup](#installation--environment-setup)
  - [Recommended: Pixi (Fast & Reproducible)](#recommended-pixi-fast--reproducible)
  - [Alternative: Conda & Pip](#alternative-conda--pip)
- [Dataset Preparation](#dataset-preparation)
- [Weights (RoMa & DINOv2)](#weights-roma--dinov2)
- [Quick Start](#quick-start)
- [Extrinsic Initialization & Evaluation Modes](#extrinsic-initialization--evaluation-modes)
- [Visualization & Monitoring](#visualization--monitoring)
  - [Real-Time LiDAR-Camera Projection Visualization](#real-time-lidar-camera-projection-visualization)
  - [TensorBoard Metric Tracking](#tensorboard-metric-tracking)
- [Outputs & Experiment Tracking](#outputs--experiment-tracking)
- [Command-Line Arguments Reference](#command-line-arguments-reference)
- [Citation & Acknowledgments](#citation--acknowledgments)

---

## Key Improvements & Features

Compared to the initial release, this repository includes several major enhancements:

- **Unified Pixi Environment**: Fast, hermetic, and reproducible environment management using [Pixi](https://pixi.sh). Automatically provisions CUDA 12.8, PyTorch 2.7, Qt6-enabled OpenCV, TensorBoard, and accelerates compilation of 3DGS CUDA rasterizer kernels (`diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`, `pytorch3d`).
- **Flexible Flow Weight Resolution**: RoMa and DINOv2 models can be loaded from a custom directory (`--flow_path`), the local `./flows/` directory, or downloaded automatically via PyTorch Hub.
- **Refactored Hierarchical Gaussian Model**: Re-architected `CalibrateGaussianModel` and `CalibrateGaussianListModel` with `GaussianModel` as their base class, eliminating code duplication and streamlining optimization across multi-scale pyramid levels.
- **Controlled Perturbation & Benchmark Suite**: Comprehensive options to initialize and evaluate calibration under synthetic disturbances:
  - Custom calibration file loading (`calib_cam_to_pose.txt` or `calib_cam_to_velo.txt`).
  - Ground-truth initialization (`--init_from_gt`).
  - Deterministic Euler angle offsets (`--rot_error_deg`) and translation offsets (`--trans_error_m`).
  - Uniform random perturbations (`--rand_rot_deg`, `--rand_trans_m`) with reproducible seeds (`--calib_seed`).
- **In-Loop Projection Visualization & Animations**: Real-time projection of temporal LiDAR sweeps onto camera images with export of iteration snapshots, final calibrated projection images, and animated convergence reels (`.webp`).
- **TensorBoard Logging**: Live monitoring of training EMA loss, Euler rotation error norm, and translation error norm.
- **Automated Experiment Tracking & Results Export**:
  - Run isolation under `output/runs/YYYY-MM-DD_HH-MM-SS_seq<N>_cam<ID>[_<exp_name>]/`.
  - Automatic generation of `calibration_metrics.json` (per-axis rotation/translation errors, calibrated $T_{cl}$ matrix).
  - Production-ready export of updated KITTI-360 `calib_cam_to_pose_cam<ID>.txt`.
  - Latest-run symlinks (`output/runs/latest` and `output/tensorboard/<run_name>`).

---

## System Requirements

- **OS**: Linux (tested on Ubuntu 22.04 / 24.04 x86_64)
- **GPU**: NVIDIA GPU with CUDA support (CUDA 12.0+; compute capability $\ge$ 7.0 recommended, 16GB+ VRAM)
- **Compiler**: GCC / G++ 11+ or 12+ with Ninja build system

---

## Installation & Environment Setup

### Recommended: Pixi (Fast & Reproducible)

We recommend using [Pixi](https://pixi.sh/) for one-step installation. Pixi manages system dependencies (GCC, CUDA toolkit, PyTorch) and builds CUDA extensions without manual dependency resolution.

1. **Install Pixi** (if not already installed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```
2. **Install project dependencies and build submodules**:
   ```bash
   git clone https://github.com/.../HiGS-Calib.git
   cd HiGS-Calib
   pixi install
   ```
   Pixi automatically provisions all conda and PyPI dependencies specified in `pixi.toml` and compiles `diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`, and `pytorch3d`.

3. **Verify installation with test task**:
   ```bash
   pixi run test-calib
   ```

### Alternative: Conda & Pip

If you prefer standard Conda:
```bash
conda create -n higs python=3.12 -y
conda activate higs

# Install PyTorch & CUDA
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.4 -y
conda install -c conda-forge ninja gcc_linux-64 gxx_linux-64 "opencv>=4.10" tensorboard -y

# Install PyPI dependencies and CUDA extensions
pip install plyfile tqdm open3d==0.19.0 joblib romatch==0.1.2
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
pip install --no-build-isolation "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@3dgs_accel"
pip install --no-build-isolation "git+https://gitlab.inria.fr/bkerbl/simple-knn.git"
pip install --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim.git"
```

---

## Dataset Preparation

HiGS-Calib is evaluated on the [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) benchmark dataset.

1. **Download Reorganized Subset (Recommended)**:
   The prepared KITTI-360 subset formatted for HiGS-Calib can be downloaded from our [Google Drive link](https://drive.google.com/drive/folders/13Ftx61708xNP4JHke_9VBFOawWCZfcit?usp=sharing).

2. **Directory Layout**:
   Extract the dataset to a path (e.g. `data/KITTI360`):
   ```text
   data/KITTI360/
   ├── calibration/
   │   ├── perspective.txt            # Camera intrinsics and distortion parameters
   │   ├── calib_cam_to_pose.txt      # Camera -> vehicle (body/IMU) poses
   │   └── calib_cam_to_velo.txt      # Ground truth cam00 -> Velodyne extrinsic matrix
   ├── data_2d_raw/
   │   └── <seq>/image_{00,01}/data_rgb/ # Camera perspective frames (e.g. 0000000000.png)
   ├── data_3d_raw/
   │   └── <seq>/velodyne_points/data/   # Raw LiDAR Velodyne .bin point cloud scans
   └── data_poses/
       └── <seq>/poses.txt            # Vehicle trajectory poses
   ```

3. **Sequence Mapping**:
   The `--data_seq` argument uses a 0-based index `0…4`:
   - `0`: Drive `2013_05_28_drive_0009_sync` (Segment 1)
   - `1`: Drive `2013_05_28_drive_0010_sync` (Segment 1)
   - `2`: Drive `2013_05_28_drive_0010_sync` (Segment 2)
   - `3`: Drive `2013_05_28_drive_0009_sync` (Segment 2)
   - `4`: Drive `2013_05_28_drive_0009_sync` (Segment 3)

---

## Weights (RoMa & DINOv2)

HiGS-Calib uses RoMa and DINOv2 for dense optical flow extraction between adjacent viewpoint frames:
- `roma_outdoor.pth`
- `dinov2_vitl14_pretrain.pth`

### Resolution Order:
The codebase resolves model weights in the following order:
1. Custom path specified via `--flow_path <path>` (points either to a folder containing both `.pth` files or directly to a file).
2. Local folder `./flows/` in the repository root.
3. **Automatic Download**: If weights are not found locally, the code automatically downloads them from PyTorch Hub.

Pre-downloaded weights are also available on our [Google Drive link](https://drive.google.com/drive/folders/13Ftx61708xNP4JHke_9VBFOawWCZfcit?usp=sharing).

---

## Quick Start

### Basic Calibration Run

Run calibration using Pixi:
```bash
pixi run python calibrate.py \
    -s data/KITTI360 \
    --optimizer_type sparse_adam \
    --cam_id 00 \
    --data_seq 2 \
    --iterations 3000 \
    --visualize_projection
```

Or using an active conda environment:
```bash
python calibrate.py \
    -s data/KITTI360 \
    --optimizer_type sparse_adam \
    --cam_id 00 \
    --data_seq 2 \
    --iterations 3000 \
    --visualize_projection
```

### Preconfigured Pixi Tasks
```bash
# Run quick calibration test
pixi run test-calib

# Launch TensorBoard dashboard
pixi run tensorboard
```

---

## Extrinsic Initialization & Evaluation Modes

HiGS-Calib supports diverse initialization modes to evaluate calibration accuracy and robustness under synthetic disturbances.

### 1. Default Nominal Initialization
By default, the calibrator starts with the standard nominal LiDAR-camera orthogonal orientation:
```math
R_{\text{nominal}} = \begin{bmatrix} 0 & -1 & 0 \\ 0 & 0 & -1 \\ 1 & 0 & 0 \end{bmatrix}, \quad t = \mathbf{0}
```

### 2. Loading from a Custom Calibration File
Load initial extrinsics from an existing `calib_cam_to_pose.txt` (KITTI format) or a $3\times 4$ / $4\times 4$ `calib_cam_to_velo.txt`:
```bash
python calibrate.py -s data/KITTI360 --cam_id 00 --data_seq 2 \
    --init_calib_file data/KITTI360/calibration/calib_cam_to_pose.txt
```

### 3. Initializing from Ground Truth with Controlled Offsets
Initialize extrinsics directly from ground truth ($T_{cl}$), optionally adding deterministic or random perturbations:

- **Deterministic perturbation**:
  ```bash
  python calibrate.py -s data/KITTI360 --cam_id 00 --data_seq 2 \
      --init_from_gt \
      --rot_error_deg 2.0 -1.5 1.0 \
      --trans_error_m 0.1 -0.05 0.15
  ```

- **Uniform random perturbation** with reproducible seed:
  ```bash
  python calibrate.py -s data/KITTI360 --cam_id 00 --data_seq 2 \
      --init_from_gt \
      --rand_rot_deg 5.0 \
      --rand_trans_m 0.2 \
      --calib_seed 42
  ```

---

## Visualization & Monitoring

### Real-Time LiDAR-Camera Projection Visualization
Pass the `--visualize_projection` flag to inspect point cloud alignment in real time:
```bash
python calibrate.py -s data/KITTI360 --cam_id 00 --data_seq 2 --visualize_projection
```
- In headless/remote environments, the script automatically catches windowing exceptions and saves progress to disk.
- Generated artifacts in `<run_dir>/visualizations/`:
  - `latest.jpg`: Live rendered camera view with color-coded LiDAR depth overlay.
  - `snapshots/iter_<iteration>.jpg`: Snapshots captured every 100 iterations.
  - `final_calibrated_projection.jpg`: Final high-resolution projection after optimization.
  - `calibration_projection.webp`: Animated visualization showcasing convergence from initial misalignment to the final calibrated state.

### TensorBoard Metric Tracking
Loss and calibration errors are logged in real time. Launch TensorBoard via Pixi:
```bash
pixi run tensorboard
# or manually:
tensorboard --logdir=output/tensorboard
```
Available metrics:
- `Loss/train`: Exponential moving average loss.
- `rotation_err/train`: L2 norm of Euler rotation error (degrees) against ground truth.
- `translation_err/train`: L2 norm of translation error (meters) against ground truth.

---

## Outputs & Experiment Tracking

Every execution generates a self-contained, timestamped run directory under `output/runs/`:

```text
output/
├── runs/
│   ├── latest -> 2026-10-19_10-35-26_seq3_cam00/   # Symlink to latest run
│   └── 2026-10-19_10-35-26_seq3_cam00/
│       ├── calib_cam_to_pose_cam00.txt  # Calibrated extrinsics in KITTI-360 pose format
│       ├── calibration_metrics.json     # Quantitative evaluation metrics and error norms
│       ├── config.json                  # Parsed CLI configuration for reproducibility
│       ├── cfg_args                     # Namespace arguments string (3DGS compatible)
│       ├── cameras.json                 # Serialized camera intrinsics
│       ├── input.ply                    # Initial point cloud
│       ├── tensorboard/                 # Run-specific TensorBoard event logs
│       └── visualizations/              # Visual projections and animations
│           ├── latest.jpg
│           ├── final_calibrated_projection.jpg
│           ├── calibration_projection.webp
│           └── snapshots/
│               ├── iter_00100.jpg
│               └── ...
└── tensorboard/
    └── <run_name> -> ../runs/<run_name>/tensorboard # Symlink for multi-run comparison
```

### Quantitative Metrics Output (`calibration_metrics.json`)
```json
{
    "sequence": 3,
    "cam_id": "00",
    "iterations": 3000,
    "final_loss": 0.0704,
    "final_rotation_error_norm": 0.2453,
    "final_translation_error_norm": 0.1551,
    "rotation_error_per_axis": [0.2380, 0.0517, 0.0294],
    "translation_error_per_axis": [0.1109, 0.0688, 0.0838],
    "calibrated_rotation_cl": [
        [ 0.0426, -0.9990, -0.0125 ],
        [-0.0842,  0.0089, -0.9964 ],
        [ 0.9955,  0.0435, -0.0837 ]
    ],
    "calibrated_translation_cl": [ 0.1514, -0.1765, -0.9131 ],
    "calibrated_pose_file": "calib_cam_to_pose_cam00.txt"
}
```

---

## Command-Line Arguments Reference

### Data & Model Parameters
| Parameter | Default | Description |
|---|---|---|
| `-s, --source_path` | `""` | Path to dataset root directory (e.g., `data/KITTI360`). |
| `-m, --model_path` | `"output"` | Base directory for outputs and run records. |
| `--cam_id` | `None` | Camera identifier: `"00"` (left perspective) or `"01"` (right perspective). |
| `--data_seq` | `None` | Sequence index (`0` through `4`). |
| `--flow_path` | `""` | Directory containing flow model weights (`roma_outdoor.pth`, `dinov2_vitl14_pretrain.pth`). Falls back to `./flows/` then PyTorch Hub. |
| `--exp_name` | `None` | Optional experiment name / tag appended to run folder name. |

### Optimization & Pipeline Parameters
| Parameter | Default | Description |
|---|---|---|
| `--iterations` | `10000` | Number of calibration/optimization iterations (e.g. `3000` for quick runs). |
| `--optimizer_type` | `"default"` | Optimizer implementation: `"default"` or `"sparse_adam"` (requires `diff_gaussian_rasterization` with `3dgs_accel`). |
| `--visualize_projection` | `False` | Real-time LiDAR projection visualization onto camera image with disk snapshot and animated WebP export. |
| `--disable_viewer` | `False` | Disable the real-time SIBR GUI network viewer. |
| `--detect_anomaly` | `False` | Enable PyTorch autograd anomaly detection (sets `CUDA_LAUNCH_BLOCKING=1`). |
| `--quiet` | `False` | Suppress non-essential verbose console outputs. |

### Extrinsics Initialization & Disturbance Parameters
| Parameter | Default | Description |
|---|---|---|
| `--init_calib_file` | `None` | Path to custom calibration file (`calib_cam_to_pose.txt` or `calib_cam_to_velo.txt`). |
| `--init_from_gt` | `False` | Initialize extrinsics directly from ground truth $T_{cl}$. |
| `--rot_error_deg` | `None` | Deterministic Euler error $(R_x, R_y, R_z)$ in degrees added to initial/GT extrinsics. |
| `--trans_error_m` | `None` | Deterministic translation error $(t_x, t_y, t_z)$ in meters added to initial/GT extrinsics. |
| `--rand_rot_deg` | `None` | Uniform random Euler perturbation bound $[-\delta, +\delta]$ in degrees per axis. |
| `--rand_trans_m` | `None` | Uniform random translation perturbation bound $[-\delta, +\delta]$ in meters per axis. |
| `--calib_seed` | `None` | Integer random seed for deterministic sampling of random perturbations. |

---

## Citation & Acknowledgments

If you use this work, please cite the original paper:
```bibtex
@article{higscalib2024,
  title={HiGS-Calib: A Hierarchical 3D Gaussian Splatting based Targetless Local-Consistent LiDAR-Camera Calibration Method},
  author={...},
  journal={...},
  year={2024}
}
```

This project builds upon:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [RoMa: Robust Dense Feature Matching](https://github.com/Parskatt/RoMa)
- [KITTI-360 Dataset](https://www.cvlibs.net/datasets/kitti-360/)
