"""Common CD and valid-pixel RMSE definitions for both sensor protocols."""
import numpy as np
from scipy.spatial import cKDTree


def evaluate(gt, sim, gt_image, sim_image, point_count):
    if len(gt) == 0 or len(sim) == 0:
        raise ValueError('CD requires nonempty filtered GT and simulated point sets')
    g, s = gt_image[..., 3], sim_image[..., 3]
    valid = ~np.isnan(g) & ~np.isnan(s)
    dvalid = ~np.isnan(gt_image[..., 4]) & ~np.isnan(sim_image[..., 4])
    return dict(
        chamfer_distance=float(cKDTree(gt[:, :3]).query(sim[:, :3])[0].mean() +
                               cKDTree(sim[:, :3]).query(gt[:, :3])[0].mean()),
        intensity_rmse=float(np.sqrt(np.mean((g[valid] - s[valid]) ** 2))),
        distance_rmse=float(np.sqrt(np.mean((gt_image[..., 4][dvalid] - sim_image[..., 4][dvalid]) ** 2))),
        points=int(point_count), gt_points=int(len(gt)), filtered_sim_points=int(len(sim)),
        joint_pixels=int(valid.sum()), gt_valid_pixels=int(np.isfinite(g).sum()),
        sim_valid_pixels=int(np.isfinite(s).sum()))
