# Assigning materials to LiDAR scene objects

The XML snippets below show how to assign materials to scene objects. Numerical values are examples.

## Define a material and reference it from an object

Define a BSDF with a unique ID inside the scene, then reference it from a shape:

```xml
<bsdf type="twosided" id="mat_wall">
    <bsdf type="diffuse">
        <rgb name="reflectance" value="0.5 0.5 0.5"/>
    </bsdf>
</bsdf>
<shape type="ply" id="wall">
    <string name="filename" value="meshes/wall.ply"/>
    <ref name="bsdf" id="mat_wall"/>
</shape>
```

Changing a shared material changes every object referencing it. To edit one object, duplicate its BSDF with a new unique ID and update only that object's reference. Keep asset paths portable relative to the scene layout.

## Choosing a model

| Surface | Plugin | Main parameters |
| --- | --- | --- |
| Matte surface | `diffuse` | `reflectance` |
| Paint or mixed diffuse/specular response | `principled` | `base_color`, `roughness`, `metallic` |
| Mirror | `conductor` | `material` or appropriate optical parameters |
| Smooth glass | `dielectric` | `int_ior`, `ext_ior` |
| Rough glass | `roughdielectric` | `alpha`, `int_ior`, `ext_ior` |
| Retroreflective surface | `retro_reflective_legacy` or `retro_reflective_angular` | `albedo`, `retro_weight`, `sharpness`; optional `entrance_angularity` for the angular model |

LiDAR response depends on reflectance/transmittance near the sensor wavelength. Visible photograph colors alone do not determine these parameters. Existing RGB-mode scenes often use equal RGB channels as effective grayscale response values; these are model parameters, not a complete spectral measurement.

## Diffuse and principled surfaces

Diffuse reflectance normally lies in [0,1]. Use `twosided` where a suitable ordinary surface should be visible from both sides; do not wrap every material indiscriminately, especially transmissive media.

```xml
<bsdf type="twosided" id="mat_painted_road">
    <bsdf type="principled">
        <rgb name="base_color" value="0.111 0.111 0.111"/>
        <float name="roughness" value="0.55"/>
        <float name="metallic" value="0.5"/>
    </bsdf>
</bsdf>
```

Roughness controls the specular distribution; metallic controls the metallic model mixture. Both normally lie in [0,1]. Roughness is not a monotonic brightness control: incidence, observation direction, path geometry, and calibration also affect returns. `principled.roughness` and `roughdielectric.alpha` are different plugin parameters and should not be equated numerically.

## Mirrors and normals

An ideal conductor mirror example is:

```xml
<bsdf type="conductor" id="mat_mirror">
    <string name="material" value="none"/>
</bsdf>
```

For a mesh representing planar walls, pillars, or mirrors, inappropriate smooth normals can distort reflection directions. Where flat face normals are appropriate, add this inside its shape:

```xml
<boolean name="face_normals" value="true"/>
```

Do not apply this indiscriminately to curved objects. Check geometry, orientation, normals, and sensor/emitter poses before attributing an error to the BSDF.

## Glass

Use `dielectric` for an ideally smooth glass interface:

```xml
<bsdf type="dielectric" id="mat_glass_smooth">
    <float name="int_ior" value="1.5"/>
    <float name="ext_ior" value="1.0"/>
</bsdf>
```

This model produces specular reflection and refraction without a surface roughness parameter. Do not add `alpha` or `distribution` to `dielectric`.

Use `roughdielectric` when the surface needs microfacet roughness:

```xml
<bsdf type="roughdielectric" id="mat_glass">
    <string name="distribution" value="ggx"/>
    <float name="alpha" value="0.3"/>
    <float name="int_ior" value="1.5"/>
    <float name="ext_ior" value="1.0"/>
</bsdf>
```

For `roughdielectric`, `alpha` controls microfacet roughness and `distribution="ggx"` selects the microfacet distribution. Both glass models use `int_ior`/`ext_ior` to determine refraction and Fresnel response. Reference the chosen BSDF ID from the shape using `<ref name="bsdf" id="mat_glass_smooth"/>` or `<ref name="bsdf" id="mat_glass"/>`.

Thickness, normals, and the inside/outside medium convention matter for both models. A thin-sheet approximation is not interchangeable with a thick dielectric. See the Mitsuba references for [dielectric](https://mitsuba.readthedocs.io/en/v3.7.1/src/generated/plugins_bsdfs.html#smooth-dielectric-material-dielectric) and [roughdielectric](https://mitsuba.readthedocs.io/en/v3.7.1/src/generated/plugins_bsdfs.html#rough-dielectric-material-roughdielectric).

## Retroreflective surfaces

Define a retroreflective material and reference it from the target shape:

```xml
<bsdf type="retro_reflective_legacy" id="mat_retro_sign">
    <float name="albedo" value="0.8"/>
    <float name="retro_weight" value="0.03"/>
    <float name="sharpness" value="50"/>
</bsdf>
```

`albedo` sets the overall response level, `retro_weight` sets the retro lobe mixture weight, and `sharpness` controls lobe width. `albedo` and `retro_weight` normally lie in [0, 1].

For the angular model, use `type="retro_reflective_angular"`. This model also accepts `<boolean name="entrance_angularity" value="true"/>` to enable incidence-angle modulation; it is enabled by default.
