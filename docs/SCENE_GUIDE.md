# Render an existing scene with LiDAR

## Run

Use a self-contained Mitsuba 3 XML scene with geometry in meters. Resolve XML includes and filename variables first; asset paths are relative to the input XML.

```bash
python scripts/render_scene.py --scene path/to/scene.xml --sensor vlp16 \
    --position 0 0 1 --rotation 0 0 0 --spp 1024 --output outputs/my_scene
```

The script preserves objects, materials, and transforms, and replaces the camera, lights, and integrator with a LiDAR setup. Both sensor and emitter receive the same pose. Position is in world meters; rotation is roll, pitch, yaw in degrees, applied as `Rz(yaw) Ry(pitch) Rx(roll)`. At zero rotation, local +X points forward, +Y left, and +Z up. Output points use the sensor coordinate system.

Choose `--sensor rs_m1plus` or `--sensor vlp16`. Defaults use binary beam response without Gaussian compensation for RS-M1+, and Gaussian response with matched compensation for VLP-16. To override, use `--beam-response binary` or `--beam-response gaussian --beam-divergence 0.0015` (half-angle in radians).

Use a fresh output directory. Outputs include `points.bin`, the generated `lidar_scene.xml`, rendering settings, and a mesh ID map. `first_ray_shape_ids.npy` records the first sampled ray's mesh ID at each bounce, rather than the strongest return's object ID. IDs refer to runtime meshes; Mitsuba may merge shapes during loading.

Optional flags: `--compute-tags`, `--robust-peaks`, and repeated `--raydrop RHO RANGE` pairs.

## Assign materials

Define each material inside `<scene>`, then reference its ID from an object. These example values are starting points for editing, rather than measured material properties.

```xml
<!-- Painted or mixed diffuse/specular surface -->
<bsdf type="principled" id="paint">
    <rgb name="base_color" value="0.3 0.3 0.3"/>
    <float name="roughness" value="0.6"/>
    <float name="metallic" value="0.0"/>
</bsdf>

<!-- Smooth glass with reflection and refraction -->
<bsdf type="dielectric" id="glass">
    <float name="int_ior" value="1.5"/>
    <float name="ext_ior" value="1.0"/>
</bsdf>

<!-- Glass with microfacet surface roughness -->
<bsdf type="roughdielectric" id="glass_rough">
    <string name="distribution" value="ggx"/>
    <float name="alpha" value="0.01"/>
    <float name="int_ior" value="1.5"/>
    <float name="ext_ior" value="1.0"/>
</bsdf>

<!-- Ideal reflecting conductor -->
<bsdf type="conductor" id="mirror">
    <string name="material" value="none"/>
</bsdf>

<shape type="ply" id="window">
    <string name="filename" value="meshes/window.ply"/>
    <ref name="bsdf" id="glass"/>
</shape>
```

The window above uses smooth glass. Change its reference to `glass_rough`, `paint`, or `mirror` to select another material. A shared material affects every object referencing it; give an object its own BSDF ID to edit it independently.

- **Principled:** `base_color` controls effective color/reflectance; `roughness` broadens the specular response; `metallic` mixes metallic behavior. These parameters normally lie in [0, 1]. Visible RGB colors alone do not establish LiDAR wavelength reflectance.
- **Smooth glass (`dielectric`):** models an ideally smooth interface with reflection and refraction. Set `int_ior`/`ext_ior` for the interior/exterior refractive indices. This plugin has no `alpha` or `distribution` parameter.
- **Rough glass (`roughdielectric`):** adds microfacet roughness. `alpha` controls roughness and `distribution="ggx"` selects the microfacet distribution; `int_ior`/`ext_ior` retain the same meaning. Use `dielectric` when an ideally smooth interface is intended.
- **Conductor:** `material="none"` gives an ideal mirror. Mesh normals determine reflection direction. For flat mesh panels, `face_normals=true` can be appropriate; preserve smooth normals on curved surfaces.

For either glass model, use consistent normals and appropriate mesh thickness. The released garage additionally uses calibrated spatial response maps; these maps are not required to assign either glass BSDF.

For retroreflective surfaces, see the [material definitions and parameters](MATERIALS.md#retroreflective-surfaces).
