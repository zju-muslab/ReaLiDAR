"""Convert an existing Mitsuba XML scene to active LiDAR rendering."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--sensor', choices=['rs_m1plus', 'vlp16'], default='vlp16')
    parser.add_argument('--beam-response', choices=['binary', 'gaussian'])
    parser.add_argument('--beam-divergence', type=float, help='Gaussian 1/e-squared half-angle, radians')
    parser.add_argument('--retro-model', choices=['legacy', 'angular'], help='Resolve the old unqualified retro_reflective XML type')
    parser.add_argument('--position', type=float, nargs=3, default=[0, 0, 1], metavar=('X', 'Y', 'Z'))
    parser.add_argument('--rotation', type=float, nargs=3, default=[0, 0, 0], metavar=('ROLL', 'PITCH', 'YAW'))
    parser.add_argument('--spp', type=int, default=1024)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compute-tags', action='store_true')
    parser.add_argument('--robust-peaks', action='store_true')
    parser.add_argument('--raydrop', type=float, nargs=2, action='append', metavar=('RHO', 'RANGE'))
    args = parser.parse_args()
    if args.spp <= 0 or args.max_depth <= 0:
        parser.error('SPP and max depth must be positive')
    response = args.beam_response or ('gaussian' if args.sensor == 'vlp16' else 'binary')
    divergence = args.beam_divergence if args.beam_divergence is not None else (0.0015 if response == 'gaussian' else 0.0)
    if (response == 'gaussian' and divergence <= 0) or (response == 'binary' and divergence != 0):
        parser.error('Use positive divergence for Gaussian response and zero for binary response')
    source = args.scene.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {output}')
    tree = ET.parse(source)
    root = tree.getroot()
    if root.tag != 'scene' or root.findall('.//include'):
        raise ValueError('Use a self-contained Mitsuba <scene> XML; expand XML includes first')
    # Preserve all geometry/material transforms; replace camera/lighting setup.
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == 'emitter' or (parent is root and child.tag in ('sensor', 'integrator')):
                parent.remove(child)
    for node in root.iter('string'):
        if node.get('name') == 'filename':
            value = node.get('value')
            if '$' in value:
                raise ValueError('Resolve filename variables before converting the scene')
            path = (source.parent / value).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            node.set('value', str(path))
    for node in root.iter('bsdf'):
        if node.get('type') == 'retro_reflective':
            if args.retro_model is None:
                parser.error('Use --retro-model legacy or angular for an unqualified retro_reflective BSDF')
            node.set('type', 'retro_reflective_' + args.retro_model)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'python'))
    import numpy as np
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import ad
    profile = 'vlp16_paper' if response == 'gaussian' else 'rs_m1_paper'
    ad.configure(profile)
    from ad.lidar_configs import get_lidar_config
    from ad.mesh_utils import tag_shape_ids
    from ad.point_cloud import LiDARPointCloud
    from ad.io import save_point_cloud_bin
    from ad.waveform_processing import save_point_cloud_tags
    cfg = get_lidar_config(args.sensor)
    beam = cfg.build_beam_config()
    roll, pitch, yaw = args.rotation
    pose = (mi.ScalarTransform4f.translate(args.position)
            @ mi.ScalarTransform4f.rotate([0, 0, 1], yaw)
            @ mi.ScalarTransform4f.rotate([0, 1, 0], pitch)
            @ mi.ScalarTransform4f.rotate([1, 0, 0], roll))
    transform = ET.Element('transform', name='to_world')
    ET.SubElement(transform, 'matrix', value=' '.join(map(str, np.asarray(pose.matrix).ravel())))
    integrator = ET.SubElement(root, 'integrator', type='lidar_waveform')
    ET.SubElement(integrator, 'integer', name='max_depth', value=str(args.max_depth))
    ET.SubElement(integrator, 'integer', name='rr_depth', value='4')
    for kind in ('sensor', 'emitter'):
        node = ET.SubElement(root, kind, type='lidar_sensor' if kind == 'sensor' else 'lidar_spot')
        ET.SubElement(node, 'integer', name='width', value=str(cfg.W))
        ET.SubElement(node, 'integer', name='height', value=str(cfg.H))
        ET.SubElement(node, 'string', name='beam_config', value=beam)
        node.append(copy.deepcopy(transform))
        if kind == 'sensor':
            ET.SubElement(node, 'float', name='beam_divergence', value=str(divergence))
            sampler = ET.SubElement(node, 'sampler', type='independent')
            ET.SubElement(sampler, 'integer', name='sample_count', value=str(args.spp))
            film = ET.SubElement(node, 'film', type='hdrfilm')
            ET.SubElement(film, 'integer', name='width', value=str(cfg.W))
            ET.SubElement(film, 'integer', name='height', value=str(cfg.H))
            ET.SubElement(film, 'rfilter', type='box')
        else:
            ET.SubElement(node, 'float', name='intensity', value='1000')
            if response == 'gaussian':
                ET.SubElement(node, 'float', name='beam_divergence', value=str(divergence))
            else:
                ET.SubElement(node, 'float', name='beam_angle', value='0.15')
                ET.SubElement(node, 'float', name='beam_width', value='0.075')
    output.mkdir(parents=True)
    tree.write(output / 'lidar_scene.xml', encoding='utf-8', xml_declaration=True)
    scene = mi.load_file(str(output / 'lidar_scene.xml'))
    id_map = tag_shape_ids(scene)
    (output / 'shape_id_map.json').write_text(json.dumps(id_map, indent=2) + '\n')
    mi.render(scene, spp=args.spp, seed=args.seed)
    cloud = LiDARPointCloud(W=cfg.W, H=cfg.H, beam_config=beam, pulse_width=cfg.PULSE_WIDTH,
                           distance_resolution=cfg.DIST_RESOLUTION, max_range=cfg.MAX_RANGE)
    raw = scene.integrator()
    reconstructed = cloud.reconstruct_all_channels(
        raw.raw_aovs, raw.raw_spp, args.max_depth, method='binning', emitter_intensity=1000,
        aov_layout='radiance_distance_shape', beam_response=response,
        beam_divergence=divergence, gaussian_compensation=response == 'gaussian',
        compute_tags=args.compute_tags, robust_peaks=args.robust_peaks, raydrop=args.raydrop)
    _, peaks = reconstructed[:2]
    if args.compute_tags:
        points, _, tags, fractions = cloud.extract_local_point_cloud(
            peaks, echo_mode='strongest', all_fracs=reconstructed[2])
        save_point_cloud_tags(str(output / 'points.bin'), tags, fractions)
    else:
        points, _ = cloud.extract_local_point_cloud(peaks, echo_mode='strongest')
    save_point_cloud_bin(str(output / 'points.bin'), points)
    ids = np.stack([raw.raw_aovs[3*b+2].reshape(cfg.W * cfg.H, raw.raw_spp)[:, 0]
                    for b in range(args.max_depth)]).astype(np.int32)
    np.save(output / 'first_ray_shape_ids.npy', ids)
    metadata = dict(sensor=args.sensor, beam_response=response, beam_divergence=divergence,
                    spp=args.spp, max_depth=args.max_depth, seed=args.seed, position=args.position,
                    rotation_roll_pitch_yaw=args.rotation, points=int(len(points)),
                    mesh_count=len(id_map), reconstruction=cloud.last_reconstruction_policy)
    (output / 'render_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
