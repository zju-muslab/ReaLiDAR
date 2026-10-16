# Scene data

[Download paper-scenes-v1.tar.gz (87.5 MB)](https://github.com/zju-muslab/ReaLiDAR/releases/download/v1.0.0/paper-scenes-v1.tar.gz).

From the repository root:

```bash
tar -xzf paper-scenes-v1.tar.gz
python scripts/verify_data.py
```

Contains scene meshes, calibrated glass response maps, and ground truth for the three reproduction cases. Checksums are recorded in [manifest.json](manifest.json) and [ARCHIVE.json](ARCHIVE.json).
For data stored elsewhere, pass `--data-dir /path/to/data` to the reproduction script; that directory must contain `assets/` and `ground_truth/`.
