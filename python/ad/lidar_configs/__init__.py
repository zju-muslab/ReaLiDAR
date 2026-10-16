"""Shared scan configuration for the two selected sensors.

Scene creation and experimental sensor discovery are intentionally outside this interface.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os

@dataclass(frozen=True)
class LiDARConfig:
    name: str
    H: int
    FREQ: float
    PPS: int
    VFOV_MIN: float
    VFOV_STEP: float
    MAX_DEPTH: int = 8
    BEAM_ANGLE: float = 0.15
    BEAM_WIDTH: float = 0.075
    SPP: int = 1024
    EMITTER_INTENSITY: float = 1000.0
    SENSOR_HEIGHT: float = 0.8
    MAX_RANGE: float = 100.0
    PULSE_WIDTH: float = 5.0
    DIST_RESOLUTION: float = 0.05
    beam_file: Optional[str] = None
    BEAM_DIVERGENCE: float = 0.0

    @property
    def W(self) -> int:
        """Number of horizontal channels = PPS / (FREQ * H)."""
        return int(self.PPS / (self.FREQ * self.H))

    @property
    def VFOV_MAX(self) -> Optional[float]:
        """Top elevation angle = VFOV_MIN + VFOV_STEP * (H - 1).

        Returns None for solid-state configs (beam_file is set).
        """
        if self.beam_file is not None:
            return None
        return self.VFOV_MIN + self.VFOV_STEP * (self.H - 1)

    @property
    def VFOV_ANGLES(self):
        """Elevation angle array (H,) in degrees, ascending.

        Returns None for solid-state configs (beam_file is set).
        """
        if self.beam_file is not None:
            return None
        import numpy as np
        return np.linspace(self.VFOV_MIN, self.VFOV_MAX, self.H)

    def build_beam_config(self) -> str:
        """Build this sensor's beam direction table, store it in the global
        registry, and return the registry key (== ``self.name``).

        Idempotent — if ``self.name`` is already registered the existing table
        is reused without recomputation.

        Routing logic
        -------------
        * ``beam_file=None``  → rotating scanner: directions come from a
          uniform VFOV linspace via :func:`ad.beam_config.make_rotating_lidar_beams`.
        * ``beam_file`` set   → solid-state / arbitrary: per-beam
          (horizontal, vertical) angles are loaded from the file and converted
          to Cartesian unit vectors.

        Returns
        -------
        str
            Registry key for use as ``beam_config=`` in
            ``build_lidar_scene_dict`` / ``LiDARPointCloud``.
        """
        from ad.beam_config import register_beam_config, list_beam_configs

        if self.name in list_beam_configs():
            return self.name   # already registered — nothing to do

        if self.beam_file is not None:
            dirs = self._load_dirs_from_file()
        else:
            dirs = self._make_rotating_dirs()

        register_beam_config(self.name, dirs)
        return self.name

    def _make_rotating_dirs(self):
        """Compute (W*H, 3) direction table for a rotating scanner from VFOV linspace."""
        import numpy as np
        from ad.beam_config import make_rotating_lidar_beams
        vfov_angles = np.linspace(self.VFOV_MIN, self.VFOV_MAX, self.H)
        return make_rotating_lidar_beams(self.W, self.H, vfov_angles)

    def _load_dirs_from_file(self):
        """Load (N, 3) beam direction array from ``self.beam_file``.

        File format (one beam per line, whitespace-separated):
            horizontal_deg  vertical_deg            (2 columns)
            ring_id  horizontal_deg  vertical_deg   (3 columns — ring_id ignored)

        All beams are flattened into a single list in file order.
        Direction convention: X forward, Y left, Z up.
        """
        import numpy as np
        from ad.beam_config import make_arbitrary_beam_config

        path = self.beam_file
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"LiDARConfig '{self.name}': beam_file not found: {path}"
            )

        rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    rows.append((float(parts[0]), float(parts[1])))
                elif len(parts) >= 3:
                    rows.append((float(parts[1]), float(parts[2])))  # skip ring_id

        if not rows:
            raise ValueError(f"LiDARConfig '{self.name}': beam_file is empty: {path}")

        angles = np.array(rows, dtype=np.float32)   # (N, 2): az, el
        az     = np.deg2rad(angles[:, 0])
        el     = np.deg2rad(angles[:, 1])
        cos_el = np.cos(el)
        dirs   = np.stack([
            cos_el * np.cos(az),   # X = forward
            cos_el * np.sin(az),   # Y = left
            np.sin(el),            # Z = up
        ], axis=1)                 # (N, 3)
        return make_arbitrary_beam_config(dirs)


def get_lidar_config(name):
    import importlib
    if name not in ("rs_m1plus", "vlp16"):
        raise ValueError(f"Unsupported release sensor: {name}")
    return importlib.import_module(f"ad.lidar_configs.{name}").CONFIG
