"""Shared RS-M1+ GT filtering and angular projection, identical for both scenes."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def _load_real_pcd_xyzid(csv_path):
    df = pd.read_csv(csv_path).rename(columns={
        'Points_m_X': 'x', 'Points_m_Y': 'y', 'Points_m_Z': 'z'})
    df = df[(df['distance'] != 0) & (df['z'] <= 0.45)].dropna()
    df['distance'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    return df[['x', 'y', 'z', 'intensity', 'distance']].values


def _sim_pts_to_xyzid(pts):
    xyz = pts[:, :3]; intensity = pts[:, 3]
    distance = np.linalg.norm(xyz, axis=1)
    mask = distance > 3
    return np.column_stack([xyz, intensity, distance])[mask]


def _load_angle_mapping(angle_file):
    data = np.loadtxt(angle_file)
    rings = data[:, 0].astype(int)
    h_angles = data[:, 1]; v_angles = data[:, 2]
    ring_groups = {}
    for ring in np.unique(rings):
        mask = rings == ring
        group = np.column_stack((h_angles[mask], v_angles[mask]))
        order = np.lexsort((-group[:, 0], -group[:, 1]))
        ring_groups[ring] = group[order]
    angle_to_index = {}
    cols_per_ring = 125; num_rows = 126
    for ring, group in ring_groups.items():
        col_offset = ring * cols_per_ring
        for i, (ha, va) in enumerate(group):
            row = i // cols_per_ring
            col = col_offset + i % cols_per_ring
            angle_to_index[(ha, va)] = (row, col)
    total_cols = cols_per_ring * len(ring_groups)
    return angle_to_index, (num_rows, total_cols, 5)


def _points_to_pseudo_image(points_xyzid, angle_mapping, image_shape):
    pseudo_image = np.full(image_shape, np.nan)
    distance_cache = np.full(image_shape[:2], np.inf)
    xy_dist = np.sqrt(points_xyzid[:, 0]**2 + points_xyzid[:, 1]**2)
    h_angles = np.degrees(np.arctan2(points_xyzid[:, 1], points_xyzid[:, 0]))
    v_angles = np.degrees(np.arctan2(points_xyzid[:, 2], xy_dist))
    distances = points_xyzid[:, 4]
    angle_pairs = np.array(list(angle_mapping.keys()))
    kdtree = cKDTree(angle_pairs)
    _, indices = kdtree.query(np.column_stack([h_angles, v_angles]))
    for i in range(len(points_xyzid)):
        key = tuple(angle_pairs[indices[i]])
        if key in angle_mapping:
            row, col = angle_mapping[key]
            if distances[i] < distance_cache[row, col]:
                distance_cache[row, col] = distances[i]
                pseudo_image[row, col, :5] = points_xyzid[i, :5]
    return pseudo_image
