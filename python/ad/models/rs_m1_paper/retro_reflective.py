"""
RetroReflective BSDF — angle-independent backscatter for LiDAR simulation

Models materials like road-sign retroreflective tape, glass bead arrays, or
corner-cube reflectors that return light toward the source regardless of angle.

BSDF model
----------
    f(ωi, ωo) = albedo * [ (1-w)/π  +  w * (n+2)/(2π) * max(0, dot(ωo,ωi))^n ]

    Terms:
      (1-w)/π       Lambertian baseline — surface is always visible
      w * D_retro   retroreflective Phong lobe peaked at ωo = ωi

Physical interpretation
-----------------------
The retro lobe is peaked at wo = wi, i.e., the outgoing direction equals the
incoming direction.  In LiDAR geometry, the emitter and sensor are co-located,
so this means the maximum signal is returned regardless of surface tilt.

For standard specular (GGX/Phong) BSDFs, the peak is at ωo = reflect(ωi, n),
which coincides with ωi only at normal incidence (θ=0).  The retroreflective
lobe achieves this for ALL θi — which is why retroreflective targets (road
signs, survey panels) are bright in LiDAR at any scan angle.

Energy conservation
-------------------
    ∫_hemisphere f(ωi,ωo) cos(θo) dωo
    = albedo * [(1-w) + w * I_retro(θi)]
    ≤ albedo ≤ 1   for any θi ∈ [0°, 90°]

where I_retro(θi) ≤ 1 because the lobe partially bleeds below the surface at
oblique angles, reducing the above-surface integral.  Energy conservation
holds for all valid (ωi, ωo) pairs. ✓

Reciprocity: f(ωi,ωo) = f(ωo,ωi) because dot(ωo,ωi) is symmetric. ✓

Importance sampling
-------------------
Mixture sampler:
  - With prob w      : Phong importance sampling around ωi (retro lobe)
                       cos_alpha = u^(1/(n+1)), phi = 2π*v
  - With prob (1-w)  : cosine hemisphere (Lambertian)

PDF:
    p(ωo) = w * (n+1)/(2π) * max(0, dot(ωo,ωi))^n
           + (1-w) * cos(θo)/π

Parameters
----------
albedo       (float) : total reflectivity [0, 1].  Default 0.95
sharpness    (float) : Phong exponent n ≥ 0.       Default 50
                       Higher → narrower lobe → stronger angle-independent backscatter.
                       n ≈ 5    : broad retroreflective lobe (~30° half-angle)
                       n ≈ 50   : moderate, like glass bead tape
                       n ≈ 200  : sharp, like corner-cube array
retro_weight (float) : fraction of energy in retro lobe [0, 1].  Default 0.9
                       0 → pure Lambertian; 1 → pure retroreflective
"""

from __future__ import annotations

import drjit as dr
import mitsuba as mi


class RetroReflectiveBSDF(mi.BSDF):
    """
    Retroreflective BSDF: scatters incident light back toward the source
    regardless of incidence angle.  Ideal for modelling high-intensity
    LiDAR calibration targets and retroreflective road markings.
    """

    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.m_albedo = props.get('albedo', 0.95)
        self.m_sharpness = dr.maximum(1e-4, float(props.get('sharpness', 50.0)))
        self.m_retro_weight = float(props.get('retro_weight', 0.9))

        # Treat as a single glossy component (internally a diffuse+retro mixture)
        flags = (mi.BSDFFlags.GlossyReflection |
                 mi.BSDFFlags.FrontSide        |
                 mi.BSDFFlags.BackSide)
        self.m_components = [flags]
        self.m_flags = flags

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(self, ctx, si, wo, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        active = active & (cos_theta_i > 0.0) & (cos_theta_o > 0.0)

        n = mi.Float(self.m_sharpness)
        w = mi.Float(self.m_retro_weight)
        a = mi.Float(self.m_albedo)

        # Lambertian term  (always ≥ 0)
        f_diff = (1.0 - w) / dr.pi

        # Retroreflective lobe peaked at wo = wi
        cos_retro = dr.maximum(mi.Float(0.0), dr.dot(wo, si.wi))
        f_retro   = w * (n + 2.0) / (2.0 * dr.pi) * dr.power(cos_retro, n)

        f = a * (f_diff + f_retro)
        return dr.select(active, f, mi.Spectrum(0.0))

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def pdf(self, ctx, si, wo, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        active = active & (cos_theta_i > 0.0) & (cos_theta_o > 0.0)

        n = mi.Float(self.m_sharpness)
        w = mi.Float(self.m_retro_weight)

        # Phong PDF centred on wi: (n+1)/(2π) * max(0, dot(wo, wi))^n
        cos_retro = dr.maximum(mi.Float(0.0), dr.dot(wo, si.wi))
        pdf_retro = (n + 1.0) / (2.0 * dr.pi) * dr.power(cos_retro, n)

        # Cosine-hemisphere PDF
        pdf_diff = mi.warp.square_to_cosine_hemisphere_pdf(wo)

        p = w * pdf_retro + (1.0 - w) * pdf_diff
        return dr.select(active, p, mi.Float(0.0))

    def eval_pdf(self, ctx, si, wo, active):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, ctx, si, sample1, sample2, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        active = active & (cos_theta_i > 0.0)

        n = mi.Float(self.m_sharpness)
        w = mi.Float(self.m_retro_weight)

        # Choose: retro lobe or diffuse
        use_retro = sample1 < w

        # ---- Retro lobe: importance sample around si.wi ----
        # Phong: cos_alpha = xi^(1/(n+1)), phi = 2π * xj
        cos_alpha = dr.power(dr.maximum(mi.Float(1e-6), sample2.y), dr.rcp(n + 1.0))
        sin_alpha = dr.safe_sqrt(dr.maximum(mi.Float(0.0), 1.0 - cos_alpha * cos_alpha))
        phi_retro = 2.0 * dr.pi * sample2.x

        # Sample in the frame where si.wi is the z-axis
        wo_retro_local = mi.Vector3f(
            sin_alpha * dr.cos(phi_retro),
            sin_alpha * dr.sin(phi_retro),
            cos_alpha,
        )
        retro_frame = mi.Frame3f(si.wi)
        wo_retro    = retro_frame.to_world(wo_retro_local)

        # ---- Diffuse: cosine-hemisphere ----
        wo_diff = mi.warp.square_to_cosine_hemisphere(sample2)

        # ---- Combine ----
        wo     = dr.select(use_retro, wo_retro, wo_diff)
        active = active & (mi.Frame3f.cos_theta(wo) > 0.0)

        bs = mi.BSDFSample3f()
        bs.wo                = wo
        bs.eta               = mi.Float(1.0)
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type      = mi.UInt32(+mi.BSDFFlags.GlossyReflection)
        bs.pdf               = self.pdf(ctx, si, wo, active)

        # Throughput weight: f(wi, wo) * cos(θo) / pdf
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        f           = self.eval(ctx, si, wo, active)
        weight = dr.select(
            active & (bs.pdf > 0.0),
            f * cos_theta_o / bs.pdf,
            mi.Spectrum(0.0),
        )

        return bs, weight

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def to_string(self):
        return (f"RetroReflectiveBSDF[\n"
                f"  albedo       = {self.m_albedo},\n"
                f"  sharpness    = {self.m_sharpness},\n"
                f"  retro_weight = {self.m_retro_weight},\n"
                f"]")


mi.register_bsdf("retro_reflective_legacy", lambda props: RetroReflectiveBSDF(props))
