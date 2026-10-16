"""Recovered VLP-16 GT filtering and channel-aligned projection."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
X_MIN, Z_MAX = 1.5, 0.45
VLP_H, VLP_W = 16, 1875
VLP_VERTICAL_DEGREES = np.arange(-15.0, 16.0, 2.0)

def _read_pcd(path: str | Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    start = raw.index(b'DATA binary')
    start = raw.index(b'\n', start) + 1
    data = np.frombuffer(raw[start:], dtype=np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('i', '<f4')]))
    return np.column_stack([data['x'], data['y'], data['z'], data['i']])


def _rotate_gt_minus15(xyz: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(-15.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle),  np.cos(angle), 0.0],
                         [0.0,            0.0,           1.0]])
    return xyz @ rotation.T


def _load_real_pcd_xyzid(path: str | Path) -> np.ndarray:
    """Return filtered GT columns x/y/z/intensity/distance in comparison frame."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path).rename(columns={
            'Points_m_X': 'x', 'Points_m_Y': 'y', 'Points_m_Z': 'z'})
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        intensity = df['intensity'].to_numpy(dtype=np.float32)
    else:
        pcd = _read_pcd(path)
        xyz = _rotate_gt_minus15(pcd[:, :3])
        intensity = pcd[:, 3]
    distance = np.linalg.norm(xyz, axis=1)
    mask = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
    mask &= (distance > 0.0) & (xyz[:, 0] >= X_MIN) & (xyz[:, 2] <= Z_MAX)
    return np.column_stack([xyz[mask], intensity[mask], distance[mask]])


def _sim_pts_to_xyzid(points: np.ndarray) -> np.ndarray:
    """Apply the agreed VLP simulation-side filter, without flipping y."""
    xyz = points[:, :3]
    intensity = points[:, 3]
    distance = np.linalg.norm(xyz, axis=1)
    mask = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
    mask &= (distance > 3.0) & (xyz[:, 0] > X_MIN)
    return np.column_stack([xyz[mask], intensity[mask], distance[mask]])


def _polar_rows_cols(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map xyz to VLP image cells with non-overlapping angular bins."""
    step = 2.0 * np.pi / VLP_W
    beta = np.mod(np.pi - np.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * np.pi)
    cols = np.floor(beta / step).astype(np.int64)
    cols = np.clip(cols, 0, VLP_W - 1)
    elevation = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
    rows = np.abs(elevation[:, None] - np.deg2rad(VLP_VERTICAL_DEGREES)[None, :]).argmin(axis=1)
    return rows, cols


def points_to_pseudo_image(points_xyzid: np.ndarray) -> np.ndarray:
    """Nearest-return range image using channel-aligned VLP polar bins."""
    image = np.full((VLP_H, VLP_W, 5), np.nan, dtype=np.float64)
    if len(points_xyzid) == 0:
        return image
    rows, cols = _polar_rows_cols(points_xyzid[:, :3])
    # Stable distance sort + unique makes nearest-point selection explicit.
    distance = points_xyzid[:, 4]
    order = np.argsort(distance, kind='stable')
    flat = rows[order] * VLP_W + cols[order]
    _, first = np.unique(flat, return_index=True)
    chosen = order[first]
    image[rows[chosen], cols[chosen]] = points_xyzid[chosen, :5]
    return image
