"""
LiDAR Sensor — beam-index time encoding

Generates primary rays for LiDAR simulation.  Each film pixel (w, h)
maps to a flat beam index i = h * W + w.  The beam direction is looked
up from a pre-registered table (no azimuth/elevation assumptions).

Time encoding
-------------
  ray.time = float32(i)

Float32 can represent all integers up to 2^24 = 16 777 216 exactly, so
any realistic LiDAR channel count is encoded without error.  The emitter
decodes with UInt32(round(ray.time)) — no subtraction, no epsilon hacks.

Beam config
-----------
Set up via ad.register_beam_config(name, beam_dirs) before loading the
scene.  Pass the same name to both sensor and emitter props.
"""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
import numpy as np


class LiDARSensor(mi.Sensor):
    """
    Custom LiDAR sensor.  Generates one ray per film pixel, direction
    and time set from a registered beam direction table.
    """

    def __init__(self, props):
        super().__init__(props)

        to_world_scalar = props.get('to_world', mi.ScalarTransform4f())
        to_world_mat_float32 = np.array(to_world_scalar.matrix, dtype=np.float32)
        self.to_world = mi.Transform4f(to_world_mat_float32)
        
        self.position = self.to_world.translation()

        # Film dimensions
        self.W = props.get('width', 100)
        self.H = props.get('height', 64)
        self.N = self.W * self.H

        # Load beam directions from registry
        beam_config_name = props.get('beam_config', '')
        from ad.beam_config import get_beam_config
        beam_dirs_np = get_beam_config(beam_config_name)   # (N, 3) float32

        if len(beam_dirs_np) != self.N:
            raise ValueError(
                f"beam_config '{beam_config_name}' has {len(beam_dirs_np)} beams "
                f"but sensor W×H = {self.W}×{self.H} = {self.N}"
            )

        # Store as DrJit Vector3f for vectorised gather
        self.local_beam_dirs = mi.Vector3f(
            mi.Float(beam_dirs_np[:, 0].copy()),
            mi.Float(beam_dirs_np[:, 1].copy()),
            mi.Float(beam_dirs_np[:, 2].copy()),
        )

        self.beam_divergence = float(props.get('beam_divergence', 0.0))

        self.m_needs_sample_2 = True
        self.m_needs_sample_3 = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_pixel(self, sample2):
        """
        Map normalised sampler output sample2 ∈ [0,1)² to flat beam index.

        sample2.x = (w + jitter) / W,  sample2.y = (h + jitter) / H
        floor() recovers exact pixel indices regardless of jitter.
        """
        w_idx = mi.UInt32(dr.clamp(dr.floor(sample2.x * self.W), 0, self.W - 1))
        h_idx = mi.UInt32(dr.clamp(dr.floor(sample2.y * self.H), 0, self.H - 1))
        return h_idx * self.W + w_idx   # flat index, row-major

    # ------------------------------------------------------------------
    # Mitsuba sensor interface
    # ------------------------------------------------------------------

    def sample_ray(self, time, sample1, sample2, sample3, active=True):
        idx = self._decode_pixel(sample2)
        local_dir = dr.gather(mi.Vector3f, self.local_beam_dirs, idx)
        world_dir = dr.normalize(self.to_world @ local_dir)

        ray = mi.Ray3f()
        ray.o = self.position
        ray.d = world_dir
        ray.time = mi.Float(idx)     # flat beam index as float32
        ray.wavelengths = []

        return ray, mi.Spectrum(1.0)

    def sample_ray_differential(self, time, sample1, sample2, sample3, active=True):
        idx = self._decode_pixel(sample2)
        local_dir = dr.gather(mi.Vector3f, self.local_beam_dirs, idx)

        world_dir = dr.normalize(self.to_world @ local_dir)

        ray = mi.RayDifferential3f()
        ray.o = self.position
        ray.d = world_dir
        ray.time = mi.Float(idx)
        ray.wavelengths = []
        ray.has_differentials = False

        return ray, mi.Spectrum(1.0)

    def needs_aperture_sample(self):
        return False

    def bbox(self):
        return mi.BoundingBox3f(self.position, self.position)

    def to_string(self):
        return (f"LiDARSensor[\n"
                f"  size = {self.W} x {self.H} = {self.N} beams,\n"
                f"  position = {self.position},\n"
                f"]")


mi.register_sensor("lidar_sensor", lambda props: LiDARSensor(props))
