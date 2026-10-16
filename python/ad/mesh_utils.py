"""Per-mesh object IDs for the shared integrator AOVs."""
import numpy as np

def tag_shape_ids(scene, name_attribute: str = 'face_shape_id'):
    """Add a per-face `face_shape_id` attribute to every Mesh in the scene.

    Each mesh gets a unique integer ID stored as a constant Float attribute
    across all its faces. The integrator can later read it via
    `si.shape.eval_attribute_1("face_shape_id", si, active)` — O(1) per ray.

    Returns
    -------
    dict
        {shape_id: shape_name}  — useful for downstream label/color mapping.
    """
    import mitsuba as mi
    id_to_name = {}
    next_id = 1   # 0 reserved for "no hit"
    for sh in scene.shapes():
        if isinstance(sh, mi.Mesh):
            n_faces = sh.face_count()
            sh.add_attribute(
                name_attribute, 1,
                np.full(n_faces, float(next_id), dtype=np.float32)
            )
            id_to_name[next_id] = sh.id() if sh.id() else f'mesh_{next_id}'
            next_id += 1
        else:
            print(f"  [tag_shape_ids] WARNING: shape '{sh.id()}' is non-Mesh "
                  f"({type(sh).__name__}), no shape_id attached")
    return id_to_name
