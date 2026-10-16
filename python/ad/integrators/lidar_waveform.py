"""
LiDAR Waveform Integrator - Per-bounce full-waveform differentiable LiDAR renderer

This integrator implements LiDAR physics with per-bounce data:
1. At bounce 0: NEE samples the LiDAR emitter directly (direct backscatter)
2. At bounces >= 1: NEE uses a "virtual point emitter" at the first-hit surface p1.
   The virtual emitter's radiance = BSDF(p1, ω_lidar_in → ω_to_pk) * L_lidar_at_p1,
   where L_lidar_at_p1 is the LiDAR radiance arriving at p1 (from bounce-0 NEE).
3. Returns per-bounce arrays: [radiance_0, ..., radiance_n] and [distance_0, ..., distance_n]
4. spp samples per channel with max_depth bounces -> KDE reconstruction of waveform
   from [spp x max_depth] shaped data

Architecture:
  receiver -> p1 -> LiDAR                      (bounce 0: direct backscatter)
  receiver -> p1 -> p2 -> (virtual p1) -> ...  (bounce 1+: virtual point NEE)

Based on Path Replay Backpropagation (PRB) for differentiability.
"""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
import numpy as np
from typing import Tuple, Optional, List

# Handle both module and standalone imports
from ad.integrators.common import RBIntegrator, ADIntegrator, mis_weight, solid_angle_to_area_jacobian


class LiDARWaveformIntegrator(RBIntegrator):
    r"""
    LiDAR Waveform Integrator - Per-bounce architecture with virtual point emitter NEE.

    At bounce 0, performs direct NEE to the LiDAR emitter (delta-position source).
    At bounces >= 1, performs NEE to the first-hit surface point p1, which acts as
    a virtual point emitter with radiance = BSDF(p1, ω_in → ω_out) * L_lidar_at_p1.

    Returns per-bounce data as AOVs:
        [bounce_0_radiance, bounce_0_distance, bounce_1_radiance, bounce_1_distance, ...]

    For KDE reconstruction, collect [spp, max_depth] shaped radiance and distance
    arrays per channel, flatten, filter zero entries, then apply KDE.

    Parameters:
    -----------
    max_depth : int
        Maximum path depth (number of bounces). Default: 8

    rr_depth : int
        Depth at which to start Russian roulette termination. Default: 2

    num_waveform_bins : int
        Number of bins in output waveform per channel. Default: 256

    max_range : float
        Maximum distance to record in meters. Default: 100.0

    hide_emitters : bool
        Hide directly visible emitters. Default: True
    """

    def __init__(self, props=mi.Properties()):
        super().__init__(props)

        # Path tracing parameters
        self.max_depth = props.get('max_depth', 8)
        self.rr_depth = props.get('rr_depth', 2)

        # Waveform parameters
        self.num_bins = props.get('num_waveform_bins', 256)
        self.max_range = props.get('max_range', 100.0)

        # For LiDAR, we typically don't want direct emission (pulse going out)
        self.hide_emitters = props.get('hide_emitters', True)

        # Retro-reflection threshold: cos of the beam cutoff angle.
        # Specularly reflected rays whose direction deviates from the exact
        # retro direction by more than this angle are discarded.
        # Should match the emitter's beam_angle (degrees).
        retro_beam_angle = props.get('retro_beam_angle', 0.15)
        self.retro_cos_threshold = float(np.cos(np.deg2rad(retro_beam_angle)))

    def render(self, scene, sensor=0, seed=0, spp=0, develop=True, evaluate=True):
        """
        Override render() to save raw per-ray AOVs before film averaging.

        After calling mi.render(scene, spp=N), access:
            integrator.raw_aovs  - list of numpy arrays, per-bounce data
            integrator.raw_spp   - samples per pixel used
        """
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        with dr.suspend_grad():
            sampler, spp = self.prepare(
                sensor=sensor, seed=seed, spp=spp, aovs=self.aov_names()
            )

            ray, weight, pos = self.sample_rays(scene, sensor, sampler)

            L, valid, aovs, _ = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                δaovs=None,
                state_in=None,
                active=mi.Bool(True)
            )

            # Save raw per-ray AOVs before film averaging
            dr.eval(L, *aovs)
            self.raw_aovs = [np.array(aov) for aov in aovs]
            self.raw_spp = spp

            # Normal film splatting (produces averaged image)
            block = film.create_block()
            block.set_coalesce(block.coalesce() and spp >= 4)

            ADIntegrator._splat_to_block(
                block, film, pos,
                value=L * weight,
                weight=1.0,
                alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                aovs=aovs,
                wavelengths=ray.wavelengths
            )

            del sampler, ray, weight, pos, L, valid

            film.put_block(block)

            return film.develop()

    def sample_rays(self, scene, sensor, sampler, reparam=None):
        """
        Override sample_rays to use fixed jitter=0.5 (pixel center) instead of
        random jitter. This ensures all SPP samples for the same channel fire in
        exactly the same beam direction, eliminating float32 decode errors in
        lidar_sensor._decode_pixel().

        Root cause of the bug: float32(1/W) * W < 1.0 (e.g. for W=78750,
        float32(1/78750) * 78750 ≈ 0.99994), so pos * scale * W underflows
        slightly below pos and floor() returns pos-1. With jitter=0.5,
        (pos+0.5) * scale * W ≈ pos+0.5, safely away from integer boundaries.
        """
        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()

        if film.sample_border():
            film_size += 2 * border_size

        spp = sampler.sample_count()

        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

        # ---- Beam divergence: extract sample_in_pixel before idx is divided by spp ----
        has_divergence = hasattr(sensor, 'beam_divergence') and sensor.beam_divergence > 0
        if has_divergence:
            if not hasattr(self, '_cached_spp') or self._cached_spp != spp:
                self._div_ox, self._div_oy = \
                    self._make_beam_offsets(spp, sensor.beam_divergence)
                self._cached_spp = spp

            if 1 << dr.log2i(spp) == spp:
                sample_in_pixel = idx & mi.UInt32(spp - 1)
            else:
                sample_in_pixel = idx % mi.UInt32(spp)

        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        pos = mi.Vector2i()
        pos.y = idx // film_size[0]
        pos.x = dr.fma(mi.UInt32(mi.Int32(-film_size[0])), pos.y, idx)

        if film.sample_border():
            pos -= border_size

        pos += mi.Vector2i(film.crop_offset())

        # No-jitter: pixel center (0.5) instead of random sampler offset.
        # (pos + 0.5) is exact in float32 for any pos < 2^23, so the
        # round-trip through scale * W always lands at pos + 0.5 + tiny_err,
        # and floor() reliably returns pos for all channels.
        pos_f = mi.Vector2f(pos) + mi.Vector2f(0.5)

        scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale
        pos_adjusted = dr.fma(pos_f, scale, offset)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        with dr.resume_grad():
            ray, weight = sensor.sample_ray_differential(
                time=time,
                sample1=wavelength_sample,
                sample2=pos_adjusted,
                sample3=aperture_sample
            )

        # ---- Apply beam divergence: rotate ray.d by Fibonacci spiral offset ----
        if has_divergence:
            delta_x = dr.gather(mi.Float, self._div_ox, sample_in_pixel)
            delta_y = dr.gather(mi.Float, self._div_oy, sample_in_pixel)

            d = ray.d
            ref_up = mi.Vector3f(0, 0, 1)
            use_alt = dr.abs(dr.dot(d, ref_up)) > mi.Float(0.999)
            ref_up = dr.select(use_alt, mi.Vector3f(1, 0, 0), ref_up)
            right = dr.normalize(dr.cross(d, ref_up))
            up = dr.cross(right, d)

            ray.d = dr.normalize(d + delta_x * right + delta_y * up)
            # weight is NOT modified — Gaussian attenuation handled by emitter

        splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f

        return ray, weight, splatting_pos

    @staticmethod
    def _make_beam_offsets(spp, beam_divergence):
        """Generate Fibonacci spiral angular offsets for SPP sub-rays.

        Returns (offset_x, offset_y) as DrJit Float arrays of length spp.
        Each entry is an angular offset in radians: delta_az, delta_el.
        """
        cutoff = 2.0 * beam_divergence
        golden = (1 + np.sqrt(5)) / 2
        ox = np.zeros(spp, dtype=np.float32)
        oy = np.zeros(spp, dtype=np.float32)
        for i in range(spp):
            r = np.sqrt(i / spp) * cutoff
            theta = 2 * np.pi * i / golden
            ox[i] = r * np.cos(theta)
            oy[i] = r * np.sin(theta)
        return mi.Float(ox.tolist()), mi.Float(oy.tolist())

    def aov_names(self):
        """Return names of per-bounce AOVs: radiance, distance, mesh shape_id for each bounce.

        `bounce_<i>_shape_id` is the integer index (as float, 0 = miss / non-mesh shape)
        assigned by `ad.mesh_utils.tag_shape_ids(scene)` on the scene's Mesh shapes.
        Use `tag_shape_ids` after `mi.load_dict` to populate the attribute; otherwise
        these AOVs will all be 0.
        """
        names = []
        for i in range(self.max_depth):
            names.extend([f'bounce_{i}_radiance',
                          f'bounce_{i}_distance',
                          f'bounce_{i}_shape_id'])
        return names

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               **kwargs) -> Tuple[mi.Spectrum,
                                   mi.Bool,
                                   List[mi.Float],
                                   mi.Spectrum]:
        """
        Sample the integrator with per-bounce radiance/distance recording.

        Bounce 0: NEE to the LiDAR delta-position emitter (direct backscatter).
        Bounces >= 1: NEE to p1 (first-hit point) as a virtual point emitter.
            Virtual emitter value = BSDF(p1, ω_lidar_in → ω_to_pk) * L_lidar_at_p1

        Returns:
            tuple: (L, valid, aovs, state_out)
                - L: mi.Spectrum - accumulated radiance (sum of all bounces)
                - valid: mi.Bool - whether ray hit any surface
                - aovs: List[mi.Float] - per-bounce data:
                    [bounce_0_rad, bounce_0_dist, bounce_1_rad, bounce_1_dist, ...]
                    Each is mi.Float with one value per ray.
                    Zero means no contribution at that bounce.
                - state_out: mi.Spectrum - state for differential phase (= L)
        """
        primal = mode == dr.ADMode.Primal

        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)
        accumulated_distance = mi.Float(0.0)
        L = mi.Spectrum(0 if primal else state_in)
        δL = mi.Spectrum(δL if δL is not None else 0)

        β = mi.Spectrum(1)
        η = mi.Float(1)
        active = mi.Bool(active)

        # Per-bounce storage: radiance, total distance, mesh shape_id for each bounce
        bounce_rad   = [mi.Float(0.0) for _ in range(self.max_depth)]
        bounce_dist  = [mi.Float(0.0) for _ in range(self.max_depth)]
        bounce_shape = [mi.Float(0.0) for _ in range(self.max_depth)]

        pi = scene.ray_intersect_preliminary(ray,
                                             coherent=True,
                                             reorder=False,
                                             active=active)

        # Variables caching information from the previous bounce
        ray_prev        = mi.Ray3f(ray)
        pi_prev         = dr.zeros(mi.PreliminaryIntersection3f)
        si_prev         = dr.zeros(mi.SurfaceInteraction3f)
        bsdf_pdf_prev   = mi.Float(1.0)
        bsdf_delta_prev = mi.Bool(True)

        # BSDF context
        bsdf_ctx = mi.BSDFContext()

        if dr.hint(self.hide_emitters, mode='scalar'):
            skip_emitters = pi.is_valid() & (pi.shape.emitter() != None) & active
            si_skip = pi.compute_surface_interaction(ray, mi.RayFlags.Minimal, skip_emitters)
            ray_skip = si_skip.spawn_ray(ray.d)
            pi_after_skip = self.skip_area_emitters(scene, ray_skip, True, skip_emitters)
            pi[skip_emitters] = pi_after_skip

        # ---- Virtual point emitter state (set at bounce 0, used at bounces >= 1) ----
        # si_first:            surface interaction at p1 (first hit); si_first.wi holds
        #                      the direction from p1 toward the LiDAR in p1's local frame
        # L_lidar_at_p1:       LiDAR radiance arriving at p1 (queried unconditionally)
        # bsdf_weight_at_p1:   bsdf.sample() weight at p1 (= reflectance ρ for a mirror).
        #                      Used in virtual NEE instead of eval() for delta p1 surfaces,
        #                      since eval() returns 0 for delta BSDFs even on valid paths.
        # dist_to_first:       receiver -> p1 distance
        # emitter_dist_from_p1: p1 -> LiDAR distance (for round-trip distance completion)
        # first_hit_set:       mask of rays that have a valid p1 cached
        si_first = dr.zeros(mi.SurfaceInteraction3f)
        L_lidar_at_p1 = mi.Spectrum(0.0)
        bsdf_weight_at_p1 = mi.Spectrum(0.0)
        dist_to_first = mi.Float(0.0)
        emitter_dist_from_p1 = mi.Float(0.0)
        first_hit_set = mi.Bool(False)

        # ---- Glass (dielectric) pass-through state ----
        # When p1 is dielectric (glass), the virtual NEE connects to the last
        # dielectric surface (pk-1 = glass back) instead of p1 (glass front),
        # avoiding shadow rays that cross the glass mesh.
        # glass_throughput accumulates bsdf_weights through all dielectric
        # surfaces in the initial glass pass (forward path reciprocity).
        last_diel_p = mi.Point3f(0.0)
        last_diel_dist = mi.Float(0.0)   # accumulated distance to last_diel_p
        glass_throughput = mi.Spectrum(1.0)
        passed_glass = mi.Bool(False)     # True after first non-dielectric hit

        for _bounce_iter in range(self.max_depth):
            active_next = mi.Bool(active)

            # ---------------------- Ray Intersection ----------------------
            with dr.resume_grad(when=not primal):
                si = pi.compute_surface_interaction(ray, ray_flags=mi.RayFlags.All)

                if not primal and _bounce_iter >= 1:
                    si_prev_diff = pi_prev.compute_surface_interaction(
                        ray_prev, ray_flags=mi.RayFlags.Minimal
                    )
                    si_prev = dr.replace_grad(si_prev, si_prev_diff)
                    si_detached = dr.detach(si)
                    wi_global = dr.normalize(si_prev.p - si_detached.p)
                    si.wi = dr.replace_grad(si.wi, si_detached.to_local(wi_global))

            bsdf = si.bsdf(ray)
            # Pure dielectric: has Transmission but NOT DiffuseReflection.
            # This distinguishes roughdielectric (pure glass) from blendbsdf
            # (roughdielectric + diffuse), where flags() is the static union
            # of both children and always includes Transmission even when the
            # actual texel is fully diffuse (weight=1).
            _is_pure_dielectric = (
                mi.has_flag(bsdf.flags(), mi.BSDFFlags.Transmission)
                & ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.DiffuseReflection)
            )

            # Accumulate travel distance (receiver -> surface_1 -> ... -> surface_n)
            accumulated_distance += si.t

            # ---------------------- Mesh shape_id capture ----------------------
            # Read per-shape ID from the Mesh face attribute (O(1) per ray).
            # Requires ad.mesh_utils.tag_shape_ids(scene) to have been called.
            # For non-Mesh shapes or missed rays, returns 0.0.
            shape_id_here = si.shape.eval_attribute_1(
                "face_shape_id", si, si.is_valid())
            _store_shape = (depth == mi.UInt32(_bounce_iter)) & si.is_valid()
            bounce_shape[_bounce_iter] = dr.select(
                _store_shape, shape_id_here, bounce_shape[_bounce_iter])

            # ---------------------- Direct Emission ----------------------
            if self.hide_emitters:
                active_next &= ~((depth == 0) & ~si.is_valid())

            # MIS weight for emitter sample from previous bounce (BSDF-sampled path)
            ds = mi.DirectionSample3f(scene, si=si, ref=si_prev)
            mis = mis_weight(
                bsdf_pdf_prev,
                scene.pdf_emitter_direction(si_prev, ds, ~bsdf_delta_prev)
            )

            with dr.resume_grad(when=not primal):
                Le = β * mis * ds.emitter.eval(si, active_next)

            # ---------------------- NEE ----------------------
            # Use `depth < self.max_depth` (not `depth + 1`) to allow NEE at the last bounce.
            # This is critical for LiDAR: bounce-0 direct backscatter is the primary signal.
            # With max_depth=1, we still want bounce-0 NEE to fire.
            active_next &= (depth < self.max_depth) & si.is_valid()

            # Mask for bounce-0: sample LiDAR emitter directly (direct backscatter)
            is_bounce_0 = (depth == mi.UInt32(0))

            # ---- Bounce 0: standard NEE to LiDAR emitter ----
            active_em_0 = active_next & is_bounce_0 & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            ds_0, em_weight_0 = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em_0)
            active_em_0 &= (ds_0.pdf != 0.0)

            wo_0 = si.to_local(ds_0.d)
            bsdf_val_0, bsdf_pdf_0 = bsdf.eval_pdf(bsdf_ctx, si, wo_0, active_em_0)
            mis_em_0 = dr.select(ds_0.delta, mi.Float(1.0), mis_weight(ds_0.pdf, bsdf_pdf_0))

            Lr_dir_0 = β * mis_em_0 * bsdf_val_0 * em_weight_0

            # Record bounce-0 contribution
            nee_scalar_0 = (Lr_dir_0.x + Lr_dir_0.y + Lr_dir_0.z) / 3.0
            total_dist_0 = accumulated_distance + ds_0.dist
            store_0 = (depth == mi.UInt32(0)) & active_em_0
            bounce_rad[0] = dr.select(store_0, nee_scalar_0, bounce_rad[0])
            bounce_dist[0] = dr.select(store_0, total_dist_0, bounce_dist[0])

            # Cache first-hit state for virtual point emitter at later bounces.
            # Query emitter unconditionally at bounce 0 (not gated by smooth BSDF)
            # so that specular first-hit surfaces also get a valid L_lidar_at_p1.
            # si_first.wi will hold the direction from p1 toward the LiDAR in p1's
            # local frame (set by compute_surface_interaction from the initial ray).
            update_first = is_bounce_0 & si.is_valid() & ~first_hit_set
            ds_p1, em_weight_p1 = scene.sample_emitter_direction(
                si, mi.Point2f(0.5, 0.5), True, update_first)
            si_first = dr.select(update_first, si, si_first)
            L_lidar_at_p1 = dr.select(update_first, em_weight_p1, L_lidar_at_p1)
            dist_to_first = dr.select(update_first, accumulated_distance, dist_to_first)
            emitter_dist_from_p1 = dr.select(update_first, ds_p1.dist, emitter_dist_from_p1)
            first_hit_set |= update_first

            # ---- Bounces >= 1: virtual point emitter NEE ----
            # Skip dielectric surfaces (glass back face etc.) — no physical NEE here.
            active_em_k = active_next & ~is_bounce_0 & first_hit_set & ~_is_pure_dielectric

            p1_is_delta = mi.has_flag(si_first.bsdf().flags(), mi.BSDFFlags.Delta)
            p1_is_pure_dielectric = (
                mi.has_flag(si_first.bsdf().flags(), mi.BSDFFlags.Transmission)
                & ~mi.has_flag(si_first.bsdf().flags(), mi.BSDFFlags.DiffuseReflection)
            )

            # ---- Choose connection target: p1 or last_diel_p (pk-1) ----
            # When p1 is dielectric (glass), connect to pk-1 (glass back face)
            # instead of p1 (glass front).  This avoids shadow rays crossing
            # the glass mesh and uses the stored glass throughput (forward path
            # reciprocity) instead of the peaked BTDF eval at p1.
            # When p1 is opaque, connect to p1 as usual.
            nee_target = dr.select(p1_is_pure_dielectric, last_diel_p, si_first.p)

            vec_to_target = nee_target - si.p
            dist_pk_target = dr.norm(vec_to_target)
            dir_to_target = dr.select(dist_pk_target > 0.0,
                                       vec_to_target / dist_pk_target,
                                       mi.Vector3f(0, 0, 1))

            # Shadow ray: pk → target (p1 or pk-1).  Just ray_test (cheap).
            # When target=pk-1, the shadow ray doesn't cross glass.
            shadow_ray = si.spawn_ray_to(nee_target)
            occluded = scene.ray_test(shadow_ray, active_em_k)
            active_em_k &= ~occluded

            # ---- bsdf_p1_val: throughput from target back to LiDAR via p1 ----
            # Case 1: p1 is dielectric → use glass_throughput (reciprocity)
            # Case 2: p1 is delta → use stored bsdf_weight_at_p1
            # Case 3: p1 is smooth opaque → use bsdf.eval() at p1
            #
            # For case 3, dir_to_target == dir_to_p1, so eval is correct.
            si_p1_query = mi.SurfaceInteraction3f(si_first)
            si_p1_query.wi = si_first.to_local(-dir_to_target)
            wo_toward_lidar = si_first.wi
            bsdf_p1_val_smooth = si_first.bsdf().eval(
                bsdf_ctx, si_p1_query, wo_toward_lidar, active_em_k)

            bsdf_p1_val = dr.select(
                p1_is_pure_dielectric, glass_throughput,
                dr.select(p1_is_delta, bsdf_weight_at_p1, bsdf_p1_val_smooth)
            )

            # BSDF at pk for direction toward target:
            wo_pk_local = si.to_local(dir_to_target)
            bsdf_val_pk, bsdf_pdf_pk = bsdf.eval_pdf(bsdf_ctx, si, wo_pk_local, active_em_k)

            # 1/r² distance correction for delta p1 (mirror / thin_dielectric).
            #
            # L_lidar_at_p1 = I_e / emitter_dist_from_p1²  (LiDAR point-source falloff at p1).
            # For a delta surface, p1 is NOT a secondary area emitter — it merely redirects
            # the LiDAR beam.  The beam continues to diverge from the original source, so the
            # correct falloff at pk uses the total path from LiDAR to pk via p1:
            #   dist_pk_to_lidar_via_p1 = dist(pk→p1) + dist(p1→LiDAR)
            #                           = dist_pk_p1 + emitter_dist_from_p1
            # Correction: multiply L_lidar_at_p1 by (emitter_dist_from_p1 / dist_pk_to_lidar_via_p1)²
            # so that L_virtual = bsdf_p1_val * I_e / dist_pk_to_lidar_via_p1²  ✓
            #
            # For smooth p1: p1 is a genuine secondary area emitter whose exitant radiance
            # is distance-invariant (radiance conservation), so no correction is needed.
            # dist correction only fires for delta p1 (never dielectric),
            # so nee_target == si_first.p and dist_pk_target == dist(pk, p1).
            dist_pk_to_lidar_via_p1 = dist_pk_target + emitter_dist_from_p1
            lidar_dist_correction = dr.select(
                p1_is_delta & (dist_pk_to_lidar_via_p1 > mi.Float(0.0)),
                (emitter_dist_from_p1 / dist_pk_to_lidar_via_p1) ** 2,
                mi.Float(1.0)
            )

            # Virtual emitter has pdf = 1 (deterministic sample), mis = 1
            L_virtual = bsdf_p1_val * L_lidar_at_p1 * lidar_dist_correction
            Lr_dir_k = β * bsdf_val_pk * L_virtual

            # Record per-bounce contribution for bounces >= 1
            nee_scalar_k = (Lr_dir_k.x + Lr_dir_k.y + Lr_dir_k.z) / 3.0
            # Full round-trip path: LiDAR -> p1 -> p2 -> ... -> pk -> p1 -> LiDAR
            #   emitter_dist_from_p1 : LiDAR -> p1  (same as p1 -> LiDAR, symmetric)
            # Round-trip distance:
            #   When target = p1:   accum + dist_pk_p1 + emitter_dist
            #   When target = pk-1: accum + dist_pk_pk-1 + (last_diel_dist - dist_to_first) + emitter_dist
            #     where (last_diel_dist - dist_to_first) is the one-way distance through glass
            #     (from p1 to pk-1), which also appears in the return leg pk-1 → p1 → LiDAR.
            # Simplified: accum + dist_pk_target + last_diel_dist - dist_to_first + emitter_dist
            # For non-dielectric p1: last_diel_dist = dist_to_first → reduces to original formula.
            glass_internal_dist = dr.select(p1_is_pure_dielectric,
                                             last_diel_dist - dist_to_first,
                                             mi.Float(0.0))
            total_dist_k = accumulated_distance + dist_pk_target + glass_internal_dist + emitter_dist_from_p1

            # Store at depth+1 because the virtual emitter at p1 is itself an additional
            # scattering event.  Physical event count for a path hitting the virtual emitter
            # at depth k:  p1(scatter 1) + ... + pk(scatter k) + p1-virtual(scatter k+1)
            # = k+1 total scatters → store in slot k+1.
            # Range stops at max_depth-1 so that i+1 never exceeds max_depth-1.
            for i in range(1, self.max_depth - 1):
                store_k = (depth == mi.UInt32(i)) & active_em_k
                bounce_rad[i + 1] = dr.select(store_k, nee_scalar_k, bounce_rad[i + 1])
                bounce_dist[i + 1] = dr.select(store_k, total_dist_k, bounce_dist[i + 1])

            # Combined NEE contribution (for L accumulation used by differential phase)
            Lr_dir = dr.select(is_bounce_0, Lr_dir_0, Lr_dir_k)

            # ---------------------- BSDF Sampling ----------------------
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)

            # Store bsdf_weight at p1 for use in virtual NEE at later bounces.
            # Done after bsdf.sample() so bsdf_weight and bsdf_sample are available.
            #
            # The virtual NEE at pk needs: bsdf_p1_val = f(ω_pk→p1→LiDAR) * |cos θ|
            # which equals bsdf.eval() for smooth BSDFs.  For delta BSDFs, eval() = 0,
            # so we reconstruct the same quantity from the forward sample:
            #   f * |cos θ| = bsdf_sample.pdf * bsdf_weight   (since weight = f*cos/pdf)
            #
            # For dielectric transmission there is an additional η^4 correction because
            # the forward importance-transport weight is (n_i/n_t)^2 = 1/η^2, while the
            # reverse path (glass→air) needs (n_t_rev/n_i_rev)^2 = (n_t/n_i)^2 = η^2:
            #   stored:  bsdf_sample.pdf * bsdf_weight = T * (1/η^2) = T/η^2
            #   needed:  T * η^2
            #   correction: η^4
            #
            # Unified formula (correct for conductor, dielectric reflection, transmission):
            #   nee_bsdf_weight = bsdf_sample.pdf * bsdf_weight * eta4_corr
            #
            # See BSDF_RECIPROCITY_VIRTUAL_NEE.md for derivation and test verification.
            _eta_sq = bsdf_sample.eta * bsdf_sample.eta       # (n_t/n_i)^2
            _is_trans = dr.abs(bsdf_sample.eta - 1.0) > mi.Float(1e-4)
            _eta4_corr = dr.select(_is_trans, _eta_sq * _eta_sq, mi.Float(1.0))
            nee_bsdf_weight = bsdf_sample.pdf * bsdf_weight * _eta4_corr
            bsdf_weight_at_p1 = dr.select(update_first, nee_bsdf_weight, bsdf_weight_at_p1)

            # ---- Update glass pass-through state ----
            # Track dielectric surfaces after p1 until the first non-dielectric hit.
            # glass_throughput accumulates bsdf_weight through all glass surfaces.
            #
            # NOTE: glass_throughput = product of importance weights (1/η² × η² = 1.0
            # for a glass slab), NOT physical Fresnel transmittance T².  The forward
            # path's T² comes from MC sampling probability; the return path's T² is
            # NOT captured here (known approximation — physically should be T⁴ total,
            # integrator gives T²).  See optimization_logic.md for analysis.
            _on_glass = _is_pure_dielectric & first_hit_set & ~passed_glass & active_next
            last_diel_p = dr.select(_on_glass, si.p, last_diel_p)
            last_diel_dist = dr.select(_on_glass, accumulated_distance, last_diel_dist)
            glass_throughput = dr.select(_on_glass,
                                          glass_throughput * bsdf_weight,
                                          glass_throughput)
            # Mark when we leave the glass pass (first non-dielectric after glass)
            _just_left_glass = ~_is_pure_dielectric & first_hit_set & ~passed_glass & active_next
            passed_glass |= _just_left_glass

            # ---- Retro-reflection: delta BSDF first hit → specular ray back to emitter ----
            # Standard bounce-0 NEE is disabled for delta surfaces (has_flag(Smooth)=False),
            # and eval() returns 0 for delta BSDFs, so the retro-reflection path
            #   LiDAR → mirror(scatter #1) → LiDAR receiver
            # is never captured by normal NEE or BSDF-sampled direct emission.
            #
            # Fix: after BSDF sampling at depth=0 on a delta surface, check whether
            # the reflected ray direction aligns with the direction to the emitter
            # (within the beam cone).  If so, contribute bsdf_weight * em_weight_p1
            # at bounce-0 (1 scatter event).
            #
            # cos_retro = dot(reflected_world, emitter_dir_from_p1)
            #   ≈ 1   for perfect retro-reflection (mirror normal ∥ incident ray)
            #   ≈ cos(2θ) for a mirror tilted by θ from perfect retro alignment
            #
            # For a vertical mirror hit by a non-horizontal beam, cos_retro < 1,
            # and no channel qualifies unless the beam is nearly perpendicular to the
            # mirror face.  em_weight_p1 == 0 guards against out-of-beam mirrors.
            retro_active = (active_next
                            & is_bounce_0
                            & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Delta)
                            & update_first)
            # reflected world-space direction (from bsdf_sample, before β update)
            d_reflected = si.to_world(bsdf_sample.wo)
            cos_retro = dr.dot(d_reflected, ds_p1.d)
            retro_active &= (cos_retro > mi.Float(self.retro_cos_threshold))

            retro_scalar = dr.mean(bsdf_weight * em_weight_p1)  # β = 1 at depth=0
            retro_dist   = mi.Float(2.0) * emitter_dist_from_p1

            # Overwrite bounce-0 only for rays where retro is active.
            # (bounce_rad[0] from smooth NEE is 0 here since active_em_0 was False.)
            bounce_rad[0]  = dr.select(retro_active, retro_scalar, bounce_rad[0])
            bounce_dist[0] = dr.select(retro_active, retro_dist,   bounce_dist[0])

            # Accumulate into L (for differential phase compatibility)
            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)
            ray_next = si.spawn_ray(si.to_world(bsdf_sample.wo))
            β *= bsdf_weight
            η *= bsdf_sample.eta

            si_prev = dr.detach(si, True)
            bsdf_pdf_prev = bsdf_sample.pdf
            bsdf_delta_prev = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # ---------------------- Russian Roulette ----------------------
            β_max = dr.max(mi.unpolarized_spectrum(β))
            active_next &= (β_max != 0)

            rr_prob = dr.minimum(β_max * η**2, .95)
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            # ----------------- Find the next interaction -----------------
            pi_next = scene.ray_intersect_preliminary(ray_next,
                                                      coherent=False,
                                                      reorder=False,
                                                      active=active_next)

            # ------------------ Differential phase only ------------------
            if dr.hint(not primal, mode='scalar'):
                si_next = pi_next.compute_surface_interaction(ray_next,
                                                              ray_flags=mi.RayFlags.Minimal,
                                                              active=active_next)

                with dr.resume_grad():
                    J = solid_angle_to_area_jacobian(
                        si.p, si_next.p, si_next.n, active_next & si_next.is_valid()
                    )

                    wo_world_diff = dr.normalize(si_next.p - si.p)
                    wo_world = dr.replace_grad(
                        ray_next.d,
                        dr.select(si_next.is_valid(), wo_world_diff, ray_next.d)
                    )
                    wo = si.to_local(wo_world)

                    bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next)

                    Lr_ind = L * dr.replace_grad(
                        1,
                        (bsdf_val / dr.detach(bsdf_val)) *
                        (J / dr.detach(J))
                    )

                    Lo = Le + Lr_dir + Lr_ind

                    attached_contrib = dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo)
                    if dr.hint(attached_contrib, mode='scalar'):
                        raise Exception(
                            "The contribution computed by the differential "
                            "rendering phase is not attached to the AD graph! "
                            "Raising an exception since this is usually "
                            "indicative of a bug (for example, you may have "
                            "forgotten to call dr.enable_grad(..) on one of "
                            "the scene parameters, or you may be trying to "
                            "optimize a parameter that does not generate "
                            "derivatives in detached PRB.)")

                    if dr.hint(mode == dr.ADMode.Backward, mode='scalar'):
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

            # ----------- Reorder threads for the next iteration --------
            reorder_hint = dr.reinterpret_array(mi.UInt32, pi_next.shape)
            reorder_hint = (reorder_hint << 1) | dr.select(active_next, 1, 0)
            depth = dr.reorder_threads(reorder_hint, 16, depth)

            # ------------------ Update loop variables ------------------
            depth[si.is_valid()] += 1
            active = active_next

            pi_prev = pi
            pi = pi_next
            ray_prev = ray
            ray = ray_next

        # Broadcast any slots that were never written to (remain as scalar mi.Float(0.0))
        # to the full per-ray size.  accumulated_distance is always the right size after
        # at least one loop iteration, so multiplying by 0 gives a properly-sized zero.
        _broadcast_zero = accumulated_distance * mi.Float(0.0)
        for i in range(self.max_depth):
            bounce_rad[i]   = bounce_rad[i]   + _broadcast_zero
            bounce_dist[i]  = bounce_dist[i]  + _broadcast_zero
            bounce_shape[i] = bounce_shape[i] + _broadcast_zero

        # Build per-bounce AOVs: [rad_0, dist_0, shape_0, rad_1, dist_1, shape_1, ...]
        aovs = []
        for i in range(self.max_depth):
            aovs.append(bounce_rad[i])
            aovs.append(bounce_dist[i])
            aovs.append(bounce_shape[i])


        return (
            L if primal else δL,
            depth != 0,
            aovs,
            L
        )

    def to_string(self):
        return (f"LiDARWaveformIntegrator[\n"
                f"  max_depth = {self.max_depth},\n"
                f"  rr_depth = {self.rr_depth},\n"
                f"  num_bins = {self.num_bins},\n"
                f"  max_range = {self.max_range} m,\n"
                f"  hide_emitters = {self.hide_emitters},\n"
                f"]")


# Register the integrator
mi.register_integrator("lidar_waveform", lambda props: LiDARWaveformIntegrator(props))
