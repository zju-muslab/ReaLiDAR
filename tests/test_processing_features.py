"""Exercise named BSDFs, compensation, ray-drop, robust peaks, and tags."""
from pathlib import Path
import json
import os
import sys
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'python'))

import numpy as np
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import ad
ad.configure('rs_m1_paper')
from ad.point_cloud import LiDARPointCloud
from ad.waveform_processing import save_point_cloud_tags, load_point_cloud_tags


def main():
    (BASE / 'outputs').mkdir(exist_ok=True)
    checks = []
    ctx = mi.BSDFContext()
    si = mi.SurfaceInteraction3f()
    si.wi = mi.Vector3f(2 ** -0.5, 0, 2 ** -0.5)
    for name, expected in [('retro_reflective_legacy', 1 / np.pi),
                           ('retro_reflective_angular', 2 ** -0.5 / np.pi)]:
        bsdf = mi.load_dict(dict(type=name, albedo=1.0, retro_weight=0.0, sharpness=50.0))
        actual = np.asarray(bsdf.eval(ctx, si, si.wi, True)).ravel()
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
    checks.append('Both named retro models coexist and retain different eval conventions')
    ad.register_beam_config('processing_test', np.array([[1, 0, 0]], dtype=np.float32))
    spp = 64
    pc = LiDARPointCloud(W=1, H=1, beam_config='processing_test', max_range=30,
                        distance_resolution=0.1)
    rad0 = np.full(spp, 20 * 1000 / (100 * np.pi * 25), dtype=np.float32)
    rad1 = np.full(spp, 10 * 1000 / (100 * np.pi * 49), dtype=np.float32)
    dist0, dist1 = np.full(spp, 10, np.float32), np.full(spp, 14, np.float32)
    ids = np.full(spp, 3, np.float32)
    aovs = [rad0, dist0, ids, rad1, dist1, ids]
    base = dict(spp=spp, max_depth=2, emitter_intensity=1000, method='binning',
                aov_layout='radiance_distance_shape', beam_response='binary')
    wave, peaks, fractions = pc.reconstruct_all_channels(aovs, **base, compute_tags=True)
    points, _, tags, fracs = pc.extract_local_point_cloud(
        peaks, echo_mode='dual_strongest', all_fracs=fractions)
    assert len(points) == 2
    np.testing.assert_allclose(points[:, 3], [20, 10], rtol=1e-5)
    np.testing.assert_array_equal(tags, [0, 1])
    np.testing.assert_allclose(fracs, [1, 0])
    with tempfile.TemporaryDirectory(dir=BASE / 'outputs') as tmp:
        file = str(Path(tmp) / 'points.bin')
        save_point_cloud_tags(file, tags, fracs)
        restored = load_point_cloud_tags(file)
        np.testing.assert_array_equal(restored[0], tags)
        np.testing.assert_allclose(restored[1], fracs)
    checks.append('Semantic fractions, real/ghost tags, dual echoes, and tag sidecars')
    _, robust, robust_fracs = pc.reconstruct_all_channels(aovs, **base, robust_peaks=True, compute_tags=True)
    assert len(robust[0]) == 2 and len(robust_fracs[0]) == 2
    checks.append('Robust peak extraction combined with semantic fractions')
    _, dropped = pc.reconstruct_all_channels(aovs, **base, raydrop=[(0.8, 5)])
    assert dropped == [[]]
    checks.append('Ray-drop reference filtering rejects weak raw returns')
    _, strongest = pc.reconstruct_all_channels(aovs, **base, peak_mode='strongest')
    assert len(strongest[0]) == 1
    checks.append('Optional strongest-bin extraction')
    weights = np.exp(-8 * np.arange(spp, dtype=np.float64) / spp)
    gauss_aovs = [rad0 * weights, dist0, ids, rad1 * weights, dist1, ids]
    policy = dict(base, beam_response='gaussian', beam_divergence=0.0015)
    g_wave, _ = pc.reconstruct_all_channels(gauss_aovs, **policy)
    np.testing.assert_allclose(g_wave, wave, rtol=2e-6, atol=1e-5)
    checks.append('Matched Gaussian disk weights plus compensation recover binary reference intensity')
    two = dict(base, aov_layout='radiance_distance')
    old_wave, _ = pc.reconstruct_all_channels([rad0, dist0, rad1, dist1], **two)
    np.testing.assert_allclose(old_wave, wave)
    checks.append('Explicit legacy two-field AOV adapter matches three-field waveform')
    for overrides in [dict(gaussian_compensation=True), dict(beam_divergence=0.0015),
                      dict(beam_response='gaussian'), dict(aov_layout='unknown')]:
        try:
            pc.reconstruct_all_channels(aovs, **dict(base, **overrides))
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid settings accepted: {overrides}')
    checks.append('Contradictory compensation and malformed AOV settings rejected')
    result = dict(passed=True, checks=checks)
    (BASE / 'outputs/feature_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
