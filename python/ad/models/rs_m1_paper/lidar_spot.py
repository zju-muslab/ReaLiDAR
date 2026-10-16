"""
LiDAR Spot Emitter — beam-index table lookup

Monochromatic point-source emitter for LiDAR simulation.
Each of the N = W×H channels fires in an arbitrary pre-registered direction.

Time encoding (matches LiDARSensor)
------------------------------------
  ray.time = float32(i)   where i = h * W + w  (flat beam index)

Decoding in get_state_at_time():
  i = UInt32(round(time))          — exact for i < 2^24 = 16 777 216
  beam_dir = gather(beam_dirs, i)  — vectorised table lookup

No azimuth/elevation arithmetic, no dt_h/dt_v, no epsilon hacks.
Works for rotating scanners, solid-state arrays, non-uniform VFOV,
and any other beam pattern — as long as a beam_config is registered.
"""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
import numpy as np


class LiDARSpotEmitter(mi.Emitter):
    """
    Point-source LiDAR emitter with per-channel beam directions.

    Physical properties:
    - Monochromatic emission, single position (origin)
    - Tight beam cone (DeltaPosition + spotlight attenuation)
    - 1/r² intensity falloff
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
                f"but emitter W×H = {self.W}×{self.H} = {self.N}"
            )

        # Store as DrJit Vector3f for vectorised gather
        self.local_beam_dirs = mi.Vector3f(
            mi.Float(beam_dirs_np[:, 0].copy()),
            mi.Float(beam_dirs_np[:, 1].copy()),
            mi.Float(beam_dirs_np[:, 2].copy()),
        )

        # Emission intensity
        intensity_val = props.get('intensity', 1000.0)
        if isinstance(intensity_val, (int, float)):
            self.intensity = mi.Color3f(intensity_val)
        else:
            self.intensity = intensity_val

        # Laser beam geometry (spotlight cone)
        beam_angle_deg = props.get('beam_angle', 0.1)   # cutoff angle
        beam_width_deg = props.get('beam_width', 0.05)  # full-intensity cone

        self.beam_angle = np.radians(beam_angle_deg)
        self.beam_width = np.radians(beam_width_deg)
        self.cos_cutoff = np.cos(self.beam_angle)
        self.cos_beam   = np.cos(self.beam_width)

        # DeltaPosition: point source at single world position
        self.m_flags = int(mi.EmitterFlags.DeltaPosition)

    # ------------------------------------------------------------------
    # Core: beam direction from ray time
    # ------------------------------------------------------------------

    def get_state_at_time(self, time):
        """
        Retrieve beam direction for the channel encoded in ray.time.

        time = float32(i) where i is the flat beam index.
        round() recovers i exactly (float32 is exact for integers < 2^24).
        """
        idx = mi.UInt32(dr.round(time))
        idx = dr.clamp(idx, 0, self.N - 1)
        # retrieve local beam dirs
        local_dir = dr.gather(mi.Vector3f, self.local_beam_dirs, idx)
        # transform to world space
        world_dir = dr.normalize(self.to_world @ local_dir)
        
        return world_dir

    # ------------------------------------------------------------------
    # Beam attenuation
    # ------------------------------------------------------------------

    def eval_beam_attenuation(self, emitter_dir, point_dir):
        """
        Spotlight cone attenuation based on angle from beam centre.

        - Inside beam_width:              attenuation = 1.0
        - Between beam_width and cutoff:  linear falloff
        - Outside cutoff:                 attenuation = 0.0
        """
        if not isinstance(emitter_dir, mi.Vector3f):
            emitter_dir = mi.Vector3f(emitter_dir)
        if not isinstance(point_dir, mi.Vector3f):
            point_dir = mi.Vector3f(point_dir)

        cos_angle = dr.dot(emitter_dir, point_dir)

        inside_beam   = cos_angle >= self.cos_beam
        inside_cutoff = cos_angle >= self.cos_cutoff

        ramp = (cos_angle - self.cos_cutoff) / (self.cos_beam - self.cos_cutoff)
        ramp = dr.clamp(ramp, 0.0, 1.0)

        return dr.select(
            inside_beam,
            mi.Float(1.0),
            dr.select(inside_cutoff, ramp, mi.Float(0.0))
        )

    # ------------------------------------------------------------------
    # Mitsuba emitter interface
    # ------------------------------------------------------------------

    def sample_direction(self, it, sample, active=True):
        """
        NEE: return direction from surface point to emitter + emitted radiance.
        """
        world_beam_dir = self.get_state_at_time(it.time)

        d_vec = it.p - self.position
        dist  = dr.norm(d_vec)
        d_dir = dr.select(dist > 0, d_vec / dist, mi.Vector3f(0, 0, 1))

        cos_angle  = dr.dot(world_beam_dir, d_dir)
        attenuation = dr.select(cos_angle >= self.cos_cutoff,
                                mi.Float(1.0), mi.Float(0.0))

        ds       = mi.DirectionSample3f()
        ds.p     = self.position
        ds.n     = mi.Normal3f(world_beam_dir)
        ds.uv    = mi.Point2f(0, 0)
        ds.d     = -d_dir
        ds.dist  = dist
        ds.delta = True
        ds.pdf   = mi.Float(1.0)     # always 1 for DeltaPosition emitter
        ds.time  = it.time

        inv_dist_sq = dr.select(dist > 0, 1.0 / (dist * dist), mi.Float(0.0))
        value = self.intensity * attenuation * inv_dist_sq

        return ds, mi.Spectrum(value)

    def eval(self, si, active=True):
        # Point sources cannot be directly hit (measure zero)
        return mi.Color3f(0.0)

    def pdf_direction(self, it, ds, active=True):
        world_beam_dir   = self.get_state_at_time(it.time)
        emitter_to_point = dr.normalize(it.p - self.position)
        attenuation      = self.eval_beam_attenuation(world_beam_dir, emitter_to_point)
        return dr.select(attenuation > 0, ds.pdf, mi.Float(0.0))

    def sample_ray(self, time, wavelength_sample, position_sample,
                   direction_sample, active=True):
        pass   # not used for LiDAR (sensor-driven simulation)

    def bbox(self):
        return mi.BoundingBox3f(self.position, self.position)

    def flags(self, active=True):
        return self.m_flags

    def to_string(self):
        return (f"LiDARSpotEmitter[\n"
                f"  beams = {self.N}  ({self.W} × {self.H}),\n"
                f"  intensity = {self.intensity},\n"
                f"  beam_angle = {np.degrees(self.beam_angle):.3f}°,\n"
                f"  beam_width = {np.degrees(self.beam_width):.3f}°,\n"
                f"]")


mi.register_emitter("lidar_spot", lambda props: LiDARSpotEmitter(props))
