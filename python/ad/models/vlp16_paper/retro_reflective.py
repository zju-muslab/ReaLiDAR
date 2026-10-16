"""
RetroReflective BSDF — angle-independent backscatter for LiDAR simulation

Models materials like road-sign retroreflective tape, glass bead arrays, or
corner-cube reflectors that return light toward the source regardless of angle.

BSDF model
----------
eval() returns the rendering-equation THROUGHPUT T (= f·cosθo-equivalent), to match
standard Mitsuba plugins and the LiDAR integrator (which uses eval directly as the
NEE throughput, expecting it to already include foreshortening):

    T(ωi, ωo) = albedo * [ (1-w)/π · cosθo
                           + w * (n+2)/(2π) * max(0, dot(ωo,ωi))^n * E(β) ]

    Terms:
      (1-w)/π · cosθo   Lambertian baseline with explicit geometric foreshortening.
      w * D_retro       retroreflective Phong lobe peaked at ωo = ωi.
      E(β)              entrance angularity R_A(β)/R_A(0): retro brightness falls off
                        as the panel tilts away from the beam (β from cosβ=√(cosθi·cosθo)).
                        Disable with prop entrance_angularity=False.

    IMPORTANT — why the retro term has NO explicit cosθo:
      E(β) already contains the cosβ projected-area (foreshortening) loss. Datasheet
      R_A(β) = I/(E⊥·A) normalizes by the FIXED sample area A (not the projected
      area A·cosβ), so R_A(β)/R_A(0) = cosβ · η_opt(β)/η_opt(0). Multiplying an extra
      cosθo onto the retro term would double-count cosβ. The Lambert term, by
      contrast, is a true BRDF and needs its cosθo applied explicitly.

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


# Entrance-angularity E(β) = R_A(β)/R_A(0): fraction of head-on retro brightness
# retained at entrance angle β. COMPROMISE visible-light datasheet shape
# (3M DG3 R_A(30°)/R_A(0°)≈0.38; ORALITE VC104+ ≈0.21@40°, ≈0.04@60°; rest
# interpolated). TO BE REPLACED by a 905nm traffic-sign-measured R_A(β) curve.
_E_BETA_DEG = [0., 10., 20., 30., 40., 50., 60., 75., 90.]
_E_VAL      = [1.0, 0.85, 0.62, 0.38, 0.21, 0.10, 0.04, 0.01, 0.0]


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
        # entrance-angularity E(β) on the retro lobe (default on; can disable)
        self.m_use_E = bool(props.get('entrance_angularity', True))

        # Treat as a single glossy component (internally a diffuse+retro mixture)
        flags = (mi.BSDFFlags.GlossyReflection |
                 mi.BSDFFlags.FrontSide        |
                 mi.BSDFFlags.BackSide)
        self.m_components = [flags]
        self.m_flags = flags

    # ------------------------------------------------------------------
    # Entrance angularity E(β)
    # ------------------------------------------------------------------

    def _E_factor(self, wi, wo):
        """Entrance-angularity E(β)=R_A(β)/R_A(0) on the retro lobe.

        Symmetric in (wi, wo) so reciprocity is preserved: cosβ = sqrt(cosθi·cosθo).
        At backscatter (wo=wi) this collapses to E(incidence angle).  Piecewise-
        linear interp over the _E_BETA_DEG/_E_VAL datasheet table; clamps to the
        last value beyond the table.  Gated by self.m_use_E."""
        if not self.m_use_E:
            return mi.Float(1.0)
        cz = dr.sqrt(dr.maximum(mi.Frame3f.cos_theta(wi), 0.0)
                     * dr.maximum(mi.Frame3f.cos_theta(wo), 0.0))
        beta = dr.acos(dr.clip(cz, 0.0, 1.0)) * (180.0 / dr.pi)
        E = mi.Float(_E_VAL[0])
        for k in range(len(_E_BETA_DEG) - 1):
            x0, x1 = _E_BETA_DEG[k], _E_BETA_DEG[k + 1]
            y0, y1 = _E_VAL[k], _E_VAL[k + 1]
            val = y0 + (y1 - y0) * (beta - x0) / (x1 - x0)
            E = dr.select((beta >= x0) & (beta < x1), val, E)
        E = dr.select(beta >= _E_BETA_DEG[-1], mi.Float(_E_VAL[-1]), E)
        return E

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

        # eval returns the rendering-equation THROUGHPUT (f·cosθo-equivalent), to
        # match standard Mitsuba plugins and what the lidar integrator expects.
        #
        # Lambertian term carries the geometric foreshortening cosθo explicitly.
        diff_throughput = (1.0 - w) / dr.pi * cos_theta_o

        # Retroreflective lobe peaked at wo = wi, scaled by entrance angularity.
        # NO explicit cosθo here: E(β)=R_A(β)/R_A(0) already contains the cosβ
        # projected-area loss, because datasheet R_A normalizes by the fixed
        # sample area A (not the projected area). Adding cosθo would double-count.
        cos_retro = dr.maximum(mi.Float(0.0), dr.dot(wo, si.wi))
        retro_throughput = (w * (n + 2.0) / (2.0 * dr.pi) * dr.power(cos_retro, n)
                            * self._E_factor(si.wi, wo))

        f = a * (diff_throughput + retro_throughput)
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

        # Throughput weight = eval(wo) / pdf.  eval already carries the correct
        # per-lobe cosine (cosθo on the Lambert term; the retro term's foreshortening
        # is inside E(β)), so do NOT multiply an extra cosθo here.
        f = self.eval(ctx, si, wo, active)
        weight = dr.select(
            active & (bs.pdf > 0.0),
            f / bs.pdf,
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


mi.register_bsdf("retro_reflective_angular", lambda props: RetroReflectiveBSDF(props))
