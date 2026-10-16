"""
beam_config.py — LiDAR beam direction registry

Decouples beam direction assignment from the LiDAR grid topology.
Supports any beam pattern: rotating scanner, solid-state, non-uniform
VFOV, per-column varying beam counts, etc.

Core idea
---------
A LiDAR with N total beams is fully described by:
  beam_dirs[i] = unit direction vector for beam i   (i ∈ [0, N-1])

The flat index i is carried through the render pipeline via ray.time:
  Sensor encodes:  ray.time = float32(pixel_flat)   exact for N < 2^24
  Emitter decodes: i = UInt32(round(ray.time)) → gather beam_dirs[i]

This eliminates the fragile dt_h/dt_v arithmetic and epsilon hacks,
and makes no assumption about azimuth/elevation distribution.

Usage
-----
import ad

# Option 1: register custom beam dirs (N×3 unit-vector array)
ad.register_beam_config('my_lidar', beam_dirs_np)

# Option 2: use the rotating-scanner helper
vfov = np.linspace(-24.9, 2.0, 64)   # arbitrary non-uniform angles
beam_dirs = ad.make_rotating_lidar_beams(W=100, H=64, vfov_angles_deg=vfov)
ad.register_beam_config('velodyne_hdl64', beam_dirs)

# Then reference by name in scene dicts / build_lidar_scene_dict(..., beam_config='velodyne_hdl64')
"""

import numpy as np

# ---------------------------------------------------------------------------
# Global registry: name → (N, 3) float32 unit-vector array
# ---------------------------------------------------------------------------
_registry: dict[str, np.ndarray] = {}


def register_beam_config(name: str, beam_dirs: np.ndarray) -> None:
    """
    Register a named beam direction array.

    Parameters
    ----------
    name       : unique string key referenced in sensor/emitter props
    beam_dirs  : (N, 3) float32 array of unit direction vectors.
                 Row i corresponds to flat pixel index i.
                 For a W×H grid: i = h * W + w  (row-major, h=row, w=col).
    """
    beam_dirs = np.asarray(beam_dirs, dtype=np.float32)
    if beam_dirs.ndim != 2 or beam_dirs.shape[1] != 3:
        raise ValueError(f"beam_dirs must be (N, 3), got {beam_dirs.shape}")
    _registry[name] = beam_dirs


def get_beam_config(name: str) -> np.ndarray:
    """
    Retrieve beam direction array by name.

    Raises KeyError with a helpful message if name is not registered.
    """
    if name not in _registry:
        keys = list(_registry.keys())
        raise KeyError(
            f"Beam config '{name}' not found. "
            f"Registered configs: {keys}. "
            f"Call ad.register_beam_config('{name}', beam_dirs) first."
        )
    return _registry[name]


def list_beam_configs() -> list[str]:
    """Return names of all registered beam configs."""
    return list(_registry.keys())


# ---------------------------------------------------------------------------
# Helpers for common beam patterns
# ---------------------------------------------------------------------------

def make_rotating_lidar_beams(
    W: int,
    H: int,
    vfov_angles_deg: np.ndarray,
) -> np.ndarray:
    """
    Build beam direction table for a rotating LiDAR scanner.

    Layout (row-major, matches sensor pixel_flat = h * W + w):
      beam i = h * W + w
      azimuth   phi[w]   = 2π * w / W            (uniform horizontal sweep)
      elevation theta[h] = vfov_angles_deg[h]     (arbitrary per-row)

    Non-uniform VFOV is fully supported: pass any (H,) array of angles,

    Different columns can have different effective beam counts by simply
    masking or duplicating rows in vfov_angles_deg before calling.

    Parameters
    ----------
    W              : number of horizontal steps
    H              : number of elevation channels
    vfov_angles_deg: (H,) array of elevation angles in degrees.

    Returns
    -------
    beam_dirs : (W*H, 3) float32 unit-vector array, row-major (h*W + w).
                Coordinate frame: X forward, Y left, Z up.
    """
    vfov_angles_deg = np.asarray(vfov_angles_deg, dtype=np.float32)
    if len(vfov_angles_deg) != H:
        raise ValueError(
            f"vfov_angles_deg must have length H={H}, got {len(vfov_angles_deg)}"
        )

    theta = np.deg2rad(vfov_angles_deg)[:, None]          # (H, 1)
    phi   = (2.0 * np.pi * np.arange(W, dtype=np.float32) / W)[None, :]  # (1, W)

    cos_t = np.cos(theta)   # (H, W) via broadcast
    sin_t = np.sin(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    # Cartesian: x = forward (cos θ cos φ), y = left (cos θ sin φ), z = up (sin θ)
    dx = (cos_t * cos_p).ravel()   # (H*W,)
    dy = (cos_t * sin_p).ravel()
    dz = np.broadcast_to(sin_t, (H, W)).ravel().copy()

    return np.stack([dx, dy, dz], axis=1).astype(np.float32)   # (H*W, 3)


def make_arbitrary_beam_config(
    directions: np.ndarray,
) -> np.ndarray:
    """
    Normalize an arbitrary (N, 3) direction array and return as beam_dirs.

    Use this for solid-state LiDARs or any custom beam pattern where you
    already have direction vectors but want to ensure unit length.

    Parameters
    ----------
    directions : (N, 3) array, any dtype

    Returns
    -------
    beam_dirs : (N, 3) float32 unit-vector array
    """
    d = np.asarray(directions, dtype=np.float32)
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError(f"directions must be (N, 3), got {d.shape}")
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("directions contains zero-length vectors")
    return (d / norms).astype(np.float32)
