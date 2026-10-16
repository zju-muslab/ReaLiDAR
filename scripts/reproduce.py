"""Render and evaluate one portable paper configuration."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CASES = ['garage_rs_m1', 'mirror_rs_m1', 'garage_vlp16']


def evaluate_points(config, points, data_dir):
    from evaluation.metrics import evaluate
    if config['sensor'] == 'vlp16':
        from evaluation import vlp16 as evaluator
        gt = evaluator._load_real_pcd_xyzid(data_dir / config['gt'])
        sim = evaluator._sim_pts_to_xyzid(points)
        gt_image = evaluator.points_to_pseudo_image(gt)
        sim_image = evaluator.points_to_pseudo_image(sim)
    else:
        from evaluation import rs_m1 as evaluator
        gt = evaluator._load_real_pcd_xyzid(data_dir / config['gt'])
        sim = evaluator._sim_pts_to_xyzid(points)
        mapping, shape = evaluator._load_angle_mapping(
            ROOT / 'python/ad/lidar_configs/rs_m1plus_all_rings_angles.txt')
        gt_image = evaluator._points_to_pseudo_image(gt, mapping, shape)
        sim_image = evaluator._points_to_pseudo_image(sim, mapping, shape)
    return evaluate(gt, sim, gt_image, sim_image, len(points))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--spp', type=int)
    parser.add_argument('--max-depth', type=int)
    parser.add_argument('--seed', type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--render-only', action='store_true')
    modes.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--points', type=Path, help='LPC2 input for --evaluate-only')
    parser.add_argument('--raydrop', type=float, nargs=2, action='append', metavar=('RHO', 'RANGE'))
    parser.add_argument('--robust-peaks', action='store_true')
    parser.add_argument('--compute-tags', action='store_true')
    parser.add_argument('--visualize', action='store_true', help='Save an offline interactive GT/Sim comparison.html')
    parser.add_argument('--peak-mode', choices=['all', 'strongest'], default='all')
    args = parser.parse_args()
    if args.evaluate_only and args.points is None:
        parser.error('--evaluate-only requires --points')
    if args.points is not None and not args.evaluate_only:
        parser.error('--points is only used with --evaluate-only')
    config = json.loads((ROOT / 'configs' / f'{args.case}.json').read_text())
    for argument, key in [(args.spp, 'spp'), (args.max_depth, 'max_depth'), (args.seed, 'seed')]:
        if argument is not None:
            config[key] = argument
    if config['spp'] <= 0 or config['max_depth'] <= 0:
        parser.error('SPP and max depth must be positive')
    data_dir = args.data_dir.resolve()
    if (not args.render_only or args.visualize) and not (data_dir / config['gt']).is_file():
        raise FileNotFoundError(f'Missing ground truth: {data_dir / config["gt"]}')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {output}')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'python'))
    import numpy as np
    from ad.io import load_point_cloud_bin, save_point_cloud_bin
    result = dict(case=args.case, settings=config,
                  environment={name: importlib.metadata.version(name)
                               for name in ['mitsuba', 'drjit', 'numpy', 'scipy', 'pandas', 'torch']})
    if args.evaluate_only:
        points = load_point_cloud_bin(str(args.points))
        output.mkdir(parents=True)
    else:
        tree = ET.parse(ROOT / config['scene'])
        root = tree.getroot()
        for node in root.iter('string'):
            if node.get('name') != 'filename':
                continue
            declared = ((ROOT / config['scene']).parent / node.get('value')).resolve()
            relative = declared.relative_to(ROOT / 'data')
            path = data_dir / relative
            if not path.is_file():
                raise FileNotFoundError(f'Missing scene resource: {path}')
            node.set('value', str(path))
        root.find("integrator/integer[@name='max_depth']").set('value', str(config['max_depth']))
        for node in root.findall('.//sampler/integer'):
            if node.get('name') == 'sample_count':
                node.set('value', str(config['spp']))
        import mitsuba as mi
        mi.set_variant('cuda_ad_rgb')
        import ad
        ad.configure(config['profile'])
        from ad.lidar_configs import get_lidar_config
        from ad.mesh_utils import tag_shape_ids
        from ad.point_cloud import LiDARPointCloud
        from ad.waveform_processing import save_point_cloud_tags
        cfg = get_lidar_config(config['sensor'])
        beam = cfg.build_beam_config()
        output.mkdir(parents=True)
        tree.write(output / 'effective_scene.xml', encoding='utf-8', xml_declaration=True)
        start = time.perf_counter()
        scene = mi.load_file(str(output / 'effective_scene.xml'))
        mapping = tag_shape_ids(scene)
        (output / 'shape_id_map.json').write_text(json.dumps(mapping, indent=2) + '\n')
        emitter = scene.emitters()[0]
        actual_divergence = float(getattr(emitter, 'beam_divergence', 0.0))
        actual_response = 'gaussian' if actual_divergence > 0 else 'binary'
        if actual_response != config['beam_response'] or not np.isclose(actual_divergence, config['beam_divergence']):
            raise ValueError('Emitter angular response does not match reconstruction configuration')
        integrator = scene.integrator()
        mi.render(scene, spp=config['spp'], seed=config['seed'])
        cloud = LiDARPointCloud(W=cfg.W, H=cfg.H, beam_config=beam,
                               pulse_width=cfg.PULSE_WIDTH, max_range=cfg.MAX_RANGE,
                               distance_resolution=cfg.DIST_RESOLUTION)
        reconstructed = cloud.reconstruct_all_channels(
            integrator.raw_aovs, integrator.raw_spp, config['max_depth'],
            method=config['method'], emitter_intensity=config['emitter_intensity'],
            aov_layout=config['aov_layout'], beam_response=actual_response,
            beam_divergence=actual_divergence, gaussian_compensation=config['gaussian_compensation'],
            raydrop=args.raydrop, robust_peaks=args.robust_peaks,
            peak_mode=args.peak_mode, compute_tags=args.compute_tags)
        _, peaks = reconstructed[:2]
        if args.compute_tags:
            points, _, tags, fractions = cloud.extract_local_point_cloud(
                peaks, echo_mode=config['echo_mode'], all_fracs=reconstructed[2])
            save_point_cloud_tags(str(output / 'points.bin'), tags, fractions)
        else:
            points, _ = cloud.extract_local_point_cloud(peaks, echo_mode=config['echo_mode'])
        save_point_cloud_bin(str(output / 'points.bin'), points)
        ids = [integrator.raw_aovs[3*b+2] for b in range(config['max_depth'])]
        preview = np.stack([array.reshape(cfg.H * cfg.W, integrator.raw_spp)[:, 0] for array in ids]).astype(np.int32)
        np.save(output / 'first_ray_shape_ids.npy', preview)
        result.update(render_seconds=time.perf_counter() - start,
                      points=int(len(points)), reconstruction_policy=cloud.last_reconstruction_policy,
                      shape_aovs=dict(mesh_count=len(mapping), stride=3,
                                      max_id_by_bounce=[float(np.max(array)) for array in ids]),
                      optional_processing=dict(raydrop=args.raydrop, robust_peaks=args.robust_peaks,
                                               compute_tags=args.compute_tags, peak_mode=args.peak_mode))
        (output / 'render_metadata.json').write_text(json.dumps(result, indent=2) + '\n')
    if not args.render_only:
        result['metrics'] = evaluate_points(config, points, data_dir)
        (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result['metrics'], indent=2))
    if args.visualize:
        from compare_point_clouds import write_comparison
        write_comparison([args.case], [args.points if args.evaluate_only else output / 'points.bin'],
                         data_dir, output / 'comparison.html')
    sources = {}
    for name, module in list(sys.modules.items()):
        if name == 'ad' or name.startswith(('ad.', 'evaluation.')):
            path = getattr(module, '__file__', None)
            if path:
                path = Path(path).resolve()
                relative = path.relative_to(ROOT)
                sources[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'runtime_sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    print(f'Saved results to {output}')


if __name__ == '__main__':
    main()
