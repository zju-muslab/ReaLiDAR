# ReaLiDAR

Official implementation of **ReaLiDAR: Material-Aware LiDAR Simulation for Robust 3D Sensing in Complex Environments**.

Yuli Wu, Duheng Pan, Puhe Hao, Yu'an Huang, Feng Lin, **Li Lu**\*  
*CCF Transactions on Pervasive Computing and Interaction*, 2026

## 🔍 Overview

ReaLiDAR simulates material-dependent LiDAR returns through light transport and waveform reconstruction. It models reflection and transmission at surfaces such as glass, mirrors, and retroreflectors, including indirect returns in complex scenes.

This repository provides RS-M1+ and VLP-16 rendering, garage and mirror scene configurations, and evaluation scripts for Chamfer distance (CD) and intensity RMSE.

## 🚀 Quick Start

### 1. Environment setup

Requires **Python 3.10** and an **NVIDIA GPU** for rendering. Tested with Mitsuba 3.7.1 and PyTorch 2.4.1+cu121.

```bash
git clone https://github.com/zju-muslab/ReaLiDAR.git
cd ReaLiDAR
conda create -n realidar python=3.10
conda activate realidar
python -m pip install -r requirements.txt
```

### 2. Scene data

[Download the scene data package (87.5 MB)](https://github.com/zju-muslab/ReaLiDAR/releases/download/v1.0.0/paper-scenes-v1.tar.gz), or run:

```bash
curl -fL -o paper-scenes-v1.tar.gz \
    https://github.com/zju-muslab/ReaLiDAR/releases/download/v1.0.0/paper-scenes-v1.tar.gz
tar -xzf paper-scenes-v1.tar.gz
python scripts/verify_data.py
```

The package contains scene meshes, calibrated material response maps, and ground truth. Extract it at the repository root to populate `data/assets/` and `data/ground_truth/`. For an external data location, see [data instructions](data/README.md).

### 3. Reproduce the scenes

```bash
# RS-M1+: garage and mirror
python scripts/reproduce.py --case garage_rs_m1 --gpu 0 --output outputs/garage_rs_m1
python scripts/reproduce.py --case mirror_rs_m1 --gpu 0 --output outputs/mirror_rs_m1

# VLP-16: garage
python scripts/reproduce.py --case garage_vlp16 --gpu 0 --output outputs/garage_vlp16
```

Each command renders a point cloud and computes CD and intensity RMSE. Outputs are saved as `points.bin` and `metrics.json`. Defaults use SPP=1024, max depth=5, and seed=0. Run commands from the repository root and choose a fresh output directory.

Add `--visualize` to save an interactive GT/Sim comparison alongside the results. To compare all three completed runs in one offline HTML viewer:

```bash
python scripts/compare_point_clouds.py --case all --results-dir outputs \
    --output outputs/comparison.html
```

Open the HTML in a browser to rotate, zoom, and compare intensity or overlaid geometry. The viewer uses the same coordinate transforms and filters as evaluation.

## 🛠 Render Your Own Scene

Follow the [scene and material guide](docs/SCENE_GUIDE.md) to render an existing Mitsuba scene with LiDAR, set the sensor pose, and assign **glass**, **principled**, or **conductor** materials to objects.

## 📂 Repository Layout

```text
python/       LiDAR renderer, waveform processing, and evaluation
configs/      Reproduction settings for the three sensor/scene cases
scenes/       Garage and mirror scene XMLs
scripts/      Reproduction, rendering, GT/Sim visualization, and data verification
docs/         Scene rendering and material assignment
```

## 📖 Citation

If you use ReaLiDAR in your research, please cite:

```bibtex
@article{wu2026realidar,
  author  = {Wu, Yuli and Pan, Duheng and Hao, Puhe and Huang, Yu'an and Lin, Feng and Lu, Li},
  title   = {{ReaLiDAR}: Material-Aware {LiDAR} Simulation for Robust {3D} Sensing in Complex Environments},
  journal = {CCF Transactions on Pervasive Computing and Interaction},
  year    = {2026},
  note    = {To appear}
}
```

## License

Original ReaLiDAR source code is licensed under [CC BY-NC-SA 4.0](LICENSE), except where otherwise noted. Commercial use requires separate permission from the relevant rights holders.

Third-party components retain their original licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). The separately distributed scene/GT data requires its own distribution terms.

## Acknowledgements

Built on Mitsuba and Dr.Jit. See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution.
