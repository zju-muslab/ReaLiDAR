"""Shared LPC2 point cloud I/O, unchanged from both snapshots."""
import os
import numpy as np

def load_point_cloud_bin(path: str) -> np.ndarray:
    """
    Load point cloud from .bin file.

    Returns
    -------
    data : ndarray (N, 5) [x, y, z, intensity, echo_order]
    """
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'LPC2':
            raise ValueError(f"Not a LiDAR point cloud file (magic={magic!r})")
        N = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.frombuffer(f.read(N * 5 * 4), dtype=np.float32).reshape(N, 5).copy()
    return data


def save_point_cloud_bin(path: str, points: np.ndarray):
    """
    Save point cloud to a compact binary file.

    Format (all little-endian):
        magic  : 4 bytes  b'LPC2'
        N      : int32    total number of points
        data   : float32  (N, 5) [x, y, z, intensity, echo_index]

    Parameters
    ----------
    path : str
        Output .bin file path.
    points : ndarray (N, 5) [x, y, z, intensity, echo_index]
        Unified point cloud from extract_world_point_cloud() or
        extract_local_point_cloud(). echo_index = 0 for first return,
        1 for second return.
    """
    data = np.ascontiguousarray(points, dtype=np.float32)
    N = len(data)

    with open(path, 'wb') as f:
        f.write(b'LPC2')
        f.write(np.array([N], dtype=np.int32).tobytes())
        f.write(data.tobytes())

    size_kb = os.path.getsize(path) / 1024
    n1 = int(np.sum(data[:, 4] == 0))
    n2 = int(np.sum(data[:, 4] == 1))
    print(f"  Saved point cloud: {path} ({n1} 1st + {n2} 2nd = {N} pts, {size_kb:.1f} KB)")
