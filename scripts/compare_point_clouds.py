"""Write an offline interactive GT/Sim point-cloud comparison (no GPU required)."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = ['garage_rs_m1', 'mirror_rs_m1', 'garage_vlp16']
LABELS = {'garage_rs_m1': 'RS-M1+ / Garage', 'mirror_rs_m1': 'RS-M1+ / Mirror',
          'garage_vlp16': 'VLP-16 / Garage'}


def load_comparison(case, points_path, data_dir):
    """Use the metric evaluator's exact filters and GT coordinate convention."""
    from ad.io import load_point_cloud_bin
    config = json.loads((ROOT / 'configs' / f'{case}.json').read_text())
    evaluator = importlib.import_module('evaluation.vlp16' if config['sensor'] == 'vlp16'
                                       else 'evaluation.rs_m1')
    gt = evaluator._load_real_pcd_xyzid(data_dir / config['gt'])
    sim = evaluator._sim_pts_to_xyzid(load_point_cloud_bin(str(points_path)))
    return gt, sim


def write_comparison(cases, point_paths, data_dir, output, max_points=60000):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'scene'}] * 3],
                        subplot_titles=['Ground truth', 'Simulation', 'Overlay'],
                        horizontal_spacing=0.025)
    summaries, layouts = [], []
    for index, (case, path) in enumerate(zip(cases, point_paths)):
        gt, sim = load_comparison(case, path, data_dir)
        if not len(gt) or not len(sim):
            raise ValueError(f'{case}: comparison needs nonempty filtered GT and Sim')
        if not np.isfinite(gt[:, :4]).all() or not np.isfinite(sim[:, :4]).all():
            raise ValueError(f'{case}: comparison contains non-finite coordinates/intensities')
        rng = np.random.default_rng(0)
        displayed = [cloud if max_points == 0 or len(cloud) <= max_points else
                     cloud[np.sort(rng.choice(len(cloud), max_points, replace=False))]
                     for cloud in [gt, sim]]
        summary = dict(case=case, gt_points=len(gt), sim_points=len(sim),
                       displayed_gt_points=len(displayed[0]), displayed_sim_points=len(displayed[1]),
                       coordinate_frame='sensor frame; evaluator filters; no additional alignment',
                       gt_z_rotation_degrees=-15 if case == 'garage_vlp16' else 0)
        summaries.append(summary)
        for cloud, col, name, color in [(displayed[0], 1, 'GT', None),
                                         (displayed[1], 2, 'Sim', None),
                                         (displayed[0], 3, 'GT', '#00a6a6'),
                                         (displayed[1], 3, 'Sim', '#e35a88')]:
            marker = dict(size=1.7, opacity=0.85)
            if color is None:
                marker.update(color=cloud[:, 3], coloraxis='coloraxis')
            else:
                marker.update(color=color)
            fig.add_trace(go.Scatter3d(
                x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2], customdata=cloud[:, 3],
                mode='markers', name=name, visible=index == 0, marker=marker,
                showlegend=col == 3,
                hovertemplate='x=%{x:.2f} m<br>y=%{y:.2f} m<br>z=%{z:.2f} m'
                              '<br>Intensity=%{customdata:.2f}<extra>%{fullData.name}</extra>'),
                row=1, col=col)
        combined = np.vstack([gt[:, :3], sim[:, :3]])
        low, high = combined.min(axis=0), combined.max(axis=0)
        pad = np.maximum((high - low) * 0.03, 0.1)
        scene = dict(aspectmode='data', bgcolor='#f7f9fc',
                     camera=dict(eye=dict(x=1.6, y=1.8, z=1.2), up=dict(x=0, y=0, z=1)))
        for axis, label, lo, hi in zip('xyz', ['X (m)', 'Y (m)', 'Z (m)'], low-pad, high+pad):
            scene[axis+'axis'] = dict(title=label, range=[float(lo), float(hi)])
        title = (f'{LABELS[case]} — GT: {len(gt):,} points · Sim: {len(sim):,} points'
                 '<br><sup>Intensity: shared 0–255 scale · Overlay: GT teal / Sim pink</sup>')
        layouts.append(dict(title=dict(text=title, x=0.025, y=0.98, yanchor='top'),
                            scene=scene, scene2=scene, scene3=scene))
    buttons = [dict(label=LABELS[case], method='update',
                    args=[dict(visible=[i // 4 == selected for i in range(4 * len(cases))]),
                          layouts[selected]]) for selected, case in enumerate(cases)]
    fig.update_layout(**layouts[0], height=720, margin=dict(l=0, r=30, t=145, b=40),
                      paper_bgcolor='white', font=dict(family='Arial, sans-serif', color='#243247'),
                      coloraxis=dict(colorscale='Viridis', cmin=0, cmax=255,
                                     colorbar=dict(title='Intensity', thickness=12, len=0.65)),
                      legend=dict(orientation='h', x=0.75, y=0),
                      updatemenus=[dict(type='buttons', direction='right', buttons=buttons,
                                        x=0, y=1.15, xanchor='left', yanchor='top')])
    sync = r"""
const gd = document.getElementById('{plot_id}');
const caseNames = __CASE_NAMES__;
const initial = caseNames.indexOf(new URLSearchParams(location.search).get('case'));
if (initial > 0) {
  const button = gd.layout.updatemenus[0].buttons[initial];
  Plotly.update(gd, button.args[0], {...button.args[1], 'updatemenus[0].active': initial});
}
let syncing = false;
gd.on('plotly_relayout', function(event) {
  if (syncing) return;
  const key = Object.keys(event).find(k => /^scene[23]?\.camera$/.test(k));
  if (!key) return;
  syncing = true;
  const update = {};
  ['scene', 'scene2', 'scene3'].forEach(s => { update[s + '.camera'] = event[key]; });
  Plotly.relayout(gd, update).then(() => { syncing = false; });
});
""".replace('__CASE_NAMES__', json.dumps(cases))
    html = fig.to_html(full_html=True, include_plotlyjs=True, post_script=sync,
                       config=dict(responsive=True, displaylogo=False, scrollZoom=True),
                       div_id='realidar-comparison')
    note = ('<div style="font:14px Arial,sans-serif;color:#43536a;padding:12px 24px">'
            '<b>Controls:</b> drag to rotate; scroll to zoom; right-drag to pan. '
            'Cameras are synchronized across the three panels. '
            '<b>Protocol:</b> evaluator-filtered sensor coordinates; VLP-16 GT includes the fixed '
            '−15° Z rotation. No ICP or extra alignment. Display subsampling, if any, does not '
            'change metrics. Overlay colors show the source, not intensity.</div>')
    html = html.replace('<body>', '<body style="margin:0">'+note)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding='utf-8')
    output.with_suffix('.json').write_text(json.dumps(summaries, indent=2)+'\n')
    print(f'Saved offline 3D comparison: {output}')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES+['all'], required=True)
    parser.add_argument('--points', type=Path, help='Single-case LPC2 point cloud')
    parser.add_argument('--results-dir', type=Path, default=ROOT/'outputs',
                        help='Contains <case>/points.bin (used when --points is absent)')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, required=True, help='Output .html file')
    parser.add_argument('--max-points', type=int, default=60000,
                        help='Display limit per cloud; 0 keeps all points. Metrics are unaffected.')
    args = parser.parse_args()
    if args.max_points < 0:
        parser.error('--max-points must be nonnegative')
    if args.output.suffix.lower() != '.html':
        parser.error('--output must end in .html')
    if args.case == 'all' and args.points is not None:
        parser.error('--points requires a single case; use --results-dir for all cases')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT/'python'))
    cases = CASES if args.case == 'all' else [args.case]
    paths = [args.points or args.results_dir/case/'points.bin' for case in cases]
    print(json.dumps(write_comparison(cases, paths, args.data_dir, args.output, args.max_points), indent=2))


if __name__ == '__main__':
    main()
