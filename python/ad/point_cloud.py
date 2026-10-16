"""Shared waveform API with explicit AOV layout and beam response policy."""
import numpy as np
from .waveform_processing import LiDARPointCloud as _WaveformPointCloud
from .io import save_point_cloud_bin, load_point_cloud_bin


class LiDARPointCloud(_WaveformPointCloud):
    def reconstruct_all_channels(self, raw_aovs, spp, max_depth,
                                 sample_chunk=2_000_000, emitter_intensity=None,
                                 method='kde', *, aov_layout, beam_response,
                                 beam_divergence=0.0, gaussian_compensation=None,
                                 raydrop=None, peak_mode='all', compute_tags=False,
                                 robust_peaks=False):
        """Reconstruct with all optional processing features preserved.

        Layout is 'radiance_distance_shape' (3 fields per bounce) or legacy
        'radiance_distance' (2 fields). Binary beams never receive Gaussian
        compensation. Gaussian beams require positive divergence and the
        matched Fibonacci disk sampling used by the shared integrator.
        An explicit contradictory compensation setting is rejected.
        """
        layouts = {'radiance_distance': 2, 'radiance_distance_shape': 3}
        if aov_layout not in layouts:
            raise ValueError(f'Unknown AOV layout: {aov_layout}')
        stride = layouts[aov_layout]
        if len(raw_aovs) != stride * max_depth:
            raise ValueError(f'Expected {stride * max_depth} AOVs, got {len(raw_aovs)}')
        if beam_response not in ('binary', 'gaussian'):
            raise ValueError(f'Unknown beam response: {beam_response}')
        expected = beam_response == 'gaussian'
        if expected and beam_divergence <= 0:
            raise ValueError('Gaussian response requires positive beam divergence')
        if not expected and beam_divergence != 0:
            raise ValueError('Binary response must use zero divergence in this sampling model')
        if gaussian_compensation is not None and bool(gaussian_compensation) != expected:
            raise ValueError('Gaussian compensation must match the beam response and sampling model')
        if stride == 2:
            expanded = []
            for bounce in range(max_depth):
                rad, dist = raw_aovs[2 * bounce:2 * bounce + 2]
                expanded.extend([rad, dist, np.zeros_like(rad)])
            raw_aovs = expanded
        self.last_reconstruction_policy = dict(
            aov_layout=aov_layout, beam_response=beam_response,
            gaussian_compensation=expected, beam_divergence=beam_divergence,
            gaussian_mean_weight=(float(np.mean(np.exp(-8.0 * np.arange(spp, dtype=np.float64) / spp)))
                                  if expected else 1.0))
        return super().reconstruct_all_channels(
            raw_aovs, spp, max_depth, sample_chunk=sample_chunk,
            emitter_intensity=emitter_intensity, method=method, raydrop=raydrop,
            beam_divergence=beam_divergence if expected else 0.0,
            peak_mode=peak_mode, compute_tags=compute_tags, robust_peaks=robust_peaks)
