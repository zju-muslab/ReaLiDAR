"""
LiDAR Point Cloud Post-Processing

GPU-accelerated KDE waveform reconstruction and peak detection for all channels,
producing 4D point clouds (x, y, z, intensity) with echo mode selection.

Pipeline:
    raw_aovs (numpy) -> PyTorch GPU KDE -> peak detection (numpy) -> point cloud
"""

import os
import numpy as np
import time as pytime
from typing import List, Tuple, Optional, Dict

try:
    import torch
except ImportError:
    torch = None


class LiDARPointCloud:
    """
    Reconstruct waveforms and extract point clouds from raw LiDAR AOVs.

    Parameters
    ----------
    W, H : int
        Film dimensions (horizontal points, vertical channels).
    beam_config : str
        Name of a registered beam direction table (see ad.register_beam_config).
        The table must contain W*H unit direction vectors, one per pixel
        (row-major: index i = h * W + w).
    origin : array-like, optional
        Sensor origin in world coordinates. Default: [0, 0, 0].
    pulse_width : float
        Laser pulse width in nanoseconds. Converted to distance sigma via
        pulse_width * 0.3 / 2.355. Default: 10.0 ns.
    distance_resolution : float
        Bin width of the reconstructed waveform in meters. Default: 0.05 m.
    max_range : float
        Maximum range in meters (round-trip distance). Default: 100.0.
    """

    def __init__(self, W: int, H: int, beam_config: str, origin=None,
                 pulse_width: float = 10.0, distance_resolution: float = 0.05,
                 max_range: float = 100.0):
        self.W = W
        self.H = H
        self.beam_config = beam_config
        self.origin = np.array(origin if origin is not None else [0.0, 0.0, 0.0],
                               dtype=np.float64)

        self.max_range = max_range
        self.distance_resolution = distance_resolution
        pulse_width_distance = pulse_width * 0.3/2.355  # ns * m/ns = m -> transform to sigma
        self.bandwidth = pulse_width_distance
        self.num_bins = int(np.ceil(max_range / distance_resolution))

        self.n_pixels = W * H

        # KDE bin centers: uniform grid with distance_resolution step
        self.bin_centers = np.arange(self.num_bins, dtype=np.float32) * distance_resolution

        # Cache beam directions
        self._beam_dirs = None

    def compute_beam_directions(self) -> np.ndarray:
        """
        Return beam direction for every channel from the registered beam config.

        Returns
        -------
        directions : ndarray, shape (n_pixels, 3)
            Unit beam direction vectors, pixel-ordered (h * W + w).
        """
        if self._beam_dirs is not None:
            return self._beam_dirs

        from ad.beam_config import get_beam_config
        beam_dirs = get_beam_config(self.beam_config)   # (N, 3) float32
        self._beam_dirs = beam_dirs.astype(np.float64)
        return self._beam_dirs

    def reconstruct_all_channels(self, raw_aovs: List[np.ndarray], spp: int,
                                 max_depth: int, sample_chunk: int = 2_000_000,
                                 emitter_intensity: float = None,
                                 method: str = 'kde',
                                 raydrop: list = None,
                                 beam_divergence: float = 0.0,
                                 peak_mode: str = 'all',
                                 compute_tags: bool = False,
                                 robust_peaks: bool = False,
                                 ) -> Tuple[np.ndarray, list]:
        """
        Waveform reconstruction for all channels.

        Parameters
        ----------
        raw_aovs, spp, max_depth, sample_chunk, emitter_intensity :
            See _reconstruct_kde for details.
        method : str
            'kde'     — GPU-accelerated sparse Gaussian KDE (accurate, ~3s).
            'binning' — simple histogram binning, argmax peak (fast, ~0.2s).
        raydrop : list of (rho, range) tuples, or None
            If provided, compute a radiance threshold from the Lambertian
            formula L = I_e * rho / (pi * r^2) for each (rho, range) pair,
            then average them. Raw samples with radiance below this threshold
            are discarded before waveform reconstruction.
            Requires emitter_intensity to be set.
        beam_divergence : float
            Beam divergence half-angle in radians (0 = no divergence).
            When > 0, intensity calibration compensates for the Gaussian
            beam profile so that a flat Lambertian wall gives the same
            calibrated intensity regardless of divergence setting.
        peak_mode : str
            'all'       — return all detected local peaks (default).
            'strongest' — return only the per-channel strongest bin. For
                          histogram binning this skips full local-maximum
                          detection and is faster when only strongest echo
                          is needed.
        compute_tags : bool
            If True, also accumulate a bounce-0-only waveform in parallel
            and return a third value `all_fracs` aligned with `all_peaks`,
            giving each peak's real_frac = wf_b0[peak_bin] / wf_total[peak_bin].
            real_frac ≈ 1 ⇒ peak dominated by direct LiDAR→surface→LiDAR
            backscatter (real); real_frac ≈ 0 ⇒ peak dominated by
            multi-bounce paths (ghost). Default False keeps the legacy
            two-tuple return shape.

        Returns
        -------
        waveforms : ndarray, shape (n_pixels, num_bins)
        all_peaks : list of lists
            all_peaks[pixel_idx] = [(distance, amplitude), ...]
        all_fracs : list of lists (only when compute_tags=True)
            all_fracs[pixel_idx] = [real_frac, ...], aligned with all_peaks.
        """
        raydrop_threshold = None
        if raydrop is not None:
            if emitter_intensity is None:
                raise ValueError("raydrop requires emitter_intensity to be set")
            if not isinstance(raydrop, (list, tuple)) or len(raydrop) == 0:
                raise ValueError("raydrop must be a non-empty list of (rho, range) tuples")
            thresholds = [emitter_intensity * rho / (np.pi * r ** 2)
                          for rho, r in raydrop]
            raydrop_threshold = float(np.mean(thresholds))
            print(f"  Raydrop: threshold = {raydrop_threshold:.6e} "
                  f"(from {len(raydrop)} reference point(s))")

        # Beam divergence Gaussian normalization:
        # Fibonacci sub-rays sample a disk of radius 2σ with Gaussian weights.
        # Mean weight < 1, so raw radiance is attenuated. Compensate in calibration.
        if beam_divergence > 0 and spp > 0:
            sigma = beam_divergence
            cutoff = 2.0 * sigma
            fib_r = np.sqrt(np.arange(spp, dtype=np.float64) / spp) * cutoff
            gaussian_norm = float(np.mean(np.exp(-2.0 * fib_r**2 / sigma**2)))
        else:
            gaussian_norm = 1.0

        if method == 'binning':
            return self._reconstruct_binning(raw_aovs, spp, max_depth,
                                             sample_chunk, emitter_intensity,
                                             raydrop_threshold, gaussian_norm,
                                             peak_mode, compute_tags,
                                             robust_peaks)
        return self._reconstruct_kde(raw_aovs, spp, max_depth,
                                     sample_chunk, emitter_intensity,
                                     raydrop_threshold, gaussian_norm,
                                     compute_tags,
                                     robust_peaks)

    def _reconstruct_kde(self, raw_aovs: List[np.ndarray], spp: int,
                          max_depth: int, sample_chunk: int = 2_000_000,
                          emitter_intensity: float = None,
                          raydrop_threshold: float = None,
                          gaussian_norm: float = 1.0,
                          compute_tags: bool = False,
                          robust_peaks: bool = False,
                          ) -> Tuple[np.ndarray, list]:
        """
        GPU-accelerated sparse KDE waveform reconstruction for all channels.

        For each valid sample at distance d, only evaluates the Gaussian kernel
        at the ~K nearby bins (within ±4*bandwidth), then scatter_add's the
        weighted contributions into the per-channel waveform. This is exact KDE
        (no binning approximation) but O(N*K) instead of O(N*num_bins).

        Parameters
        ----------
        raw_aovs : list of ndarray
            Raw AOV arrays from integrator, length 3*max_depth.
            Each array has shape (n_pixels * spp,).
        spp : int
            Samples per pixel.
        max_depth : int
            Maximum bounce depth.
        sample_chunk : int
            Max valid samples per GPU scatter batch. Default: 2M.
        emitter_intensity : float, optional
            If provided, apply intensity calibration before KDE so that waveform
            amplitudes are in [0, 255] reflectivity units.

            Convention: a 100% Lambertian target (rho=1) maps to intensity 100
            regardless of range.  Derived from:

                L_100%(r) = I_e / (pi * r^2)         [Lambertian, rho=1, at range r]
                I_cal     = clip(100 * L_raw / L_100%, 0, 255)
                          = clip(100 * L_raw * pi * r^2 / I_e, 0, 255)

            where r = round_trip_dist / 2.

            If None (default), raw radiance values are used unchanged.

        Returns
        -------
        waveforms : ndarray, shape (n_pixels, num_bins)
            Reconstructed waveform per channel.  Units are raw radiance if
            emitter_intensity is None, or calibrated intensity [0, 255] otherwise.
        all_peaks : list of lists
            all_peaks[pixel_idx] = [(distance, amplitude), ...] sorted by amplitude desc.
        """
        if torch is None:
            raise ImportError("PyTorch is required for GPU KDE. Install with: pip install torch")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h_bw = self.bandwidth
        bin_width = self.distance_resolution

        # Kernel radius: how many bins away from center to evaluate
        # At 4*sigma, exp(-0.5*16) = 0.00034, effectively zero
        kernel_radius = int(np.ceil(4.0 * h_bw / bin_width))
        K = 2 * kernel_radius + 1  # total bins per sample
        offsets = torch.arange(-kernel_radius, kernel_radius + 1,
                               device=device, dtype=torch.long)  # (K,)

        # Flat waveform buffer on GPU
        wf_flat = torch.zeros(self.n_pixels * self.num_bins,
                              device=device, dtype=torch.float32)
        # Parallel bounce-0-only buffer for semantic tagging (real vs ghost).
        # Only allocated when caller requests tags — adds ~equal GPU memory.
        wf_flat_b0 = (torch.zeros_like(wf_flat) if compute_tags else None)

        t_start = pytime.time()

        for b in range(max_depth):
            rad_all = raw_aovs[3 * b]
            dist_all = raw_aovs[3 * b + 1]

            # Extract all valid samples for this bounce
            valid_mask = rad_all > 0
            if raydrop_threshold is not None:
                valid_mask = valid_mask & (rad_all >= raydrop_threshold)
            if not np.any(valid_mask):
                continue

            valid_idx = np.where(valid_mask)[0]
            pixel_ids_np = (valid_idx // spp).astype(np.int64)
            dists_np = dist_all[valid_idx].astype(np.float32)
            rads_np = rad_all[valid_idx].astype(np.float32)

            # Intensity calibration: convert raw radiance to calibrated intensity [0, 255].
            # Reference target: 100% Lambertian reflector at range r returns L = I_e / (pi * r^2).
            # Normalized intensity = 100 * L_raw * pi * r^2 / I_e, clipped to [0, 255].
            # Note: gaussian_norm is applied AFTER waveform aggregation (mean over spp),
            # not per-sample, to avoid clipping center sub-rays that have high Gaussian weight.
            if emitter_intensity is not None:
                r_one_way = dists_np / 2.0
                rads_np = np.clip(
                    100.0 * rads_np * np.pi * r_one_way ** 2 / emitter_intensity,
                    0.0, 255.0,
                ).astype(np.float32)

            N = len(valid_idx)

            # Process in chunks to control GPU memory
            # Each sample expands to K entries, memory ~ chunk * K * 12 bytes
            for c_start in range(0, N, sample_chunk):
                c_end = min(c_start + sample_chunk, N)

                px = torch.from_numpy(pixel_ids_np[c_start:c_end]).to(device)  # (C,)
                d = torch.from_numpy(dists_np[c_start:c_end]).to(device)       # (C,)
                r = torch.from_numpy(rads_np[c_start:c_end]).to(device)        # (C,)
                C = c_end - c_start

                # Center bin for each sample
                center_bins = torch.round(d / bin_width).long()  # (C,)

                # Expand: each sample -> K nearby bins
                bin_indices = center_bins[:, None] + offsets[None, :]  # (C, K)

                # Gaussian weights at each bin
                bin_centers_at = bin_indices.float() * bin_width       # (C, K)
                diff = (bin_centers_at - d[:, None]) / h_bw            # (C, K)
                weights = torch.exp(-0.5 * diff * diff) * r[:, None]   # (C, K)

                # Mask out-of-range bins
                valid_bins = (bin_indices >= 0) & (bin_indices < self.num_bins)
                weights = weights * valid_bins.float()
                bin_indices = bin_indices.clamp(0, self.num_bins - 1)

                # Flat index into (n_pixels, num_bins)
                flat_idx = px[:, None].expand_as(bin_indices) * self.num_bins + bin_indices
                flat_idx = flat_idx.reshape(-1)
                flat_w = weights.reshape(-1)

                wf_flat.scatter_add_(0, flat_idx, flat_w)
                if wf_flat_b0 is not None and b == 0:
                    wf_flat_b0.scatter_add_(0, flat_idx, flat_w)

        # Reshape and normalize on GPU (before peak detection)
        wf_2d = wf_flat.reshape(self.n_pixels, self.num_bins)
        wf_2d /= spp

        # Beam divergence compensation: divide by mean Gaussian weight after aggregation.
        # Per-sample radiance is attenuated by Gaussian(θ_i); the mean over spp bakes in
        # the mean weight. Dividing here restores calibrated intensity as if all sub-rays
        # had full weight (beam_divergence=0 gives gaussian_norm=1.0, no-op).
        if gaussian_norm != 1.0:
            wf_2d /= gaussian_norm

        wf_b0_2d = None
        if wf_flat_b0 is not None:
            wf_b0_2d = wf_flat_b0.reshape(self.n_pixels, self.num_bins)
            wf_b0_2d /= spp
            if gaussian_norm != 1.0:
                wf_b0_2d /= gaussian_norm

        t_kde = pytime.time() - t_start

        # Peak detection: GPU fast path (1% threshold + strict local max), or
        # CPU robust path (medfilt baseline + MAD prominence + scipy find_peaks).
        t_peak_start = pytime.time()
        if robust_peaks:
            peaks_out = self._find_peaks_robust_cpu(wf_2d, wf_b0_2d)
            _peak_label = "Peak detection (robust CPU)"
        else:
            peaks_out = self._find_peaks_gpu(wf_2d, wf_b0_2d)
            _peak_label = "Peak detection (GPU)"
        t_peak = pytime.time() - t_peak_start

        # Transfer to CPU for return
        waveforms = wf_2d.cpu().numpy()

        print(f"  KDE reconstruction: {t_kde*1000:.1f} ms")
        print(f"  {_peak_label}: {t_peak*1000:.1f} ms")
        print(f"  Total: {(t_kde + t_peak)*1000:.1f} ms")

        if compute_tags:
            all_peaks, all_fracs = peaks_out
            return waveforms, all_peaks, all_fracs
        return waveforms, peaks_out

    def _reconstruct_binning(self, raw_aovs: List[np.ndarray], spp: int,
                              max_depth: int, sample_chunk: int = 2_000_000,
                              emitter_intensity: float = None,
                              raydrop_threshold: float = None,
                              gaussian_norm: float = 1.0,
                              peak_mode: str = 'all',
                              compute_tags: bool = False,
                              robust_peaks: bool = False,
                              ) -> Tuple[np.ndarray, list]:
        """
        Fast histogram binning: each sample goes into its nearest distance bin.
        No Gaussian smoothing. Peak = per-channel argmax.

        Same calibration as KDE: I_cal = clip(100 * L_raw * pi * r^2 / (I_e * gaussian_norm), 0, 255).
        """
        if torch is None:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bin_width = self.distance_resolution

        wf_flat = torch.zeros(self.n_pixels * self.num_bins,
                              device=device, dtype=torch.float32)
        wf_flat_b0 = (torch.zeros_like(wf_flat) if compute_tags else None)

        t_start = pytime.time()

        for b in range(max_depth):
            rad_all = raw_aovs[3 * b]
            dist_all = raw_aovs[3 * b + 1]

            valid_mask = rad_all > 0
            if raydrop_threshold is not None:
                valid_mask = valid_mask & (rad_all >= raydrop_threshold)
            if not np.any(valid_mask):
                continue

            valid_idx = np.where(valid_mask)[0]
            pixel_ids_np = (valid_idx // spp).astype(np.int64)
            dists_np = dist_all[valid_idx].astype(np.float32)
            rads_np = rad_all[valid_idx].astype(np.float32)

            # Intensity calibration (same as KDE path, gaussian_norm applied after aggregation)
            if emitter_intensity is not None:
                r_one_way = dists_np / 2.0
                rads_np = np.clip(
                    100.0 * rads_np * np.pi * r_one_way ** 2 / emitter_intensity,
                    0.0, 255.0,
                ).astype(np.float32)

            N = len(valid_idx)

            for c_start in range(0, N, sample_chunk):
                c_end = min(c_start + sample_chunk, N)
                px = torch.from_numpy(pixel_ids_np[c_start:c_end]).to(device)
                d  = torch.from_numpy(dists_np[c_start:c_end]).to(device)
                r  = torch.from_numpy(rads_np[c_start:c_end]).to(device)

                bins = torch.round(d / bin_width).long()
                valid = (bins >= 0) & (bins < self.num_bins)
                bins = bins.clamp(0, self.num_bins - 1)

                flat_idx = px * self.num_bins + bins
                wf_flat.scatter_add_(0, flat_idx[valid], r[valid])
                if wf_flat_b0 is not None and b == 0:
                    wf_flat_b0.scatter_add_(0, flat_idx[valid], r[valid])

        # Reshape and normalize on GPU
        wf_2d = wf_flat.reshape(self.n_pixels, self.num_bins)
        wf_2d /= spp

        # Beam divergence compensation (same as KDE path)
        if gaussian_norm != 1.0:
            wf_2d /= gaussian_norm

        wf_b0_2d = None
        if wf_flat_b0 is not None:
            wf_b0_2d = wf_flat_b0.reshape(self.n_pixels, self.num_bins)
            wf_b0_2d /= spp
            if gaussian_norm != 1.0:
                wf_b0_2d /= gaussian_norm

        t_bin = pytime.time() - t_start

        # Peak extraction: GPU fast path, or CPU robust path (MAD + scipy)
        t_peak_start = pytime.time()
        if peak_mode == 'strongest':
            peaks_out = self._find_strongest_peaks_gpu(wf_2d, wf_b0_2d)
        elif peak_mode == 'all':
            if robust_peaks:
                peaks_out = self._find_peaks_robust_cpu(wf_2d, wf_b0_2d)
            else:
                peaks_out = self._find_peaks_gpu(wf_2d, wf_b0_2d)
        else:
            raise ValueError(f"Unknown peak_mode: {peak_mode}")
        t_peak = pytime.time() - t_peak_start

        # Transfer to CPU for return
        waveforms = wf_2d.cpu().numpy()

        print(f"  Binning reconstruction: {t_bin*1000:.1f} ms")
        print(f"  Peak detection: {t_peak*1000:.1f} ms")
        print(f"  Total: {(t_bin + t_peak)*1000:.1f} ms")

        if compute_tags:
            all_peaks, all_fracs = peaks_out
            return waveforms, all_peaks, all_fracs
        return waveforms, peaks_out

    def _find_strongest_peaks_gpu(self, wf_gpu, wf_b0_gpu=None):
        """
        GPU strongest-only extraction for histogram waveforms.

        Returns one peak per active channel using argmax(bin_value), which is
        sufficient when downstream code only needs the strongest echo.

        When `wf_b0_gpu` is provided, also returns `all_fracs[p]` = per-peak
        real_frac = wf_b0[bin] / wf_total[bin] at the selected bin.
        """
        wf_max, strongest_bins = wf_gpu.max(dim=1)
        active = wf_max > 0

        pixel_ids_gpu = torch.nonzero(active, as_tuple=False).squeeze(1)
        if pixel_ids_gpu.numel() == 0:
            empty_peaks = [[] for _ in range(self.n_pixels)]
            if wf_b0_gpu is not None:
                return empty_peaks, [[] for _ in range(self.n_pixels)]
            return empty_peaks

        strongest_bins_gpu = strongest_bins[pixel_ids_gpu]
        peak_amps_gpu = wf_max[pixel_ids_gpu].clamp(max=255.0)

        pixel_ids_np = pixel_ids_gpu.cpu().numpy()
        strongest_bins_np = strongest_bins_gpu.cpu().numpy()
        peak_amps_np = peak_amps_gpu.cpu().numpy().astype(np.float64)
        peak_dists_np = self.bin_centers[strongest_bins_np].astype(np.float64)

        # real_frac = wf_b0[peak_bin] / wf_total[peak_bin]. Denominator is the
        # *raw* total (pre-clamp) so the ratio is unaffected by intensity clipping.
        if wf_b0_gpu is not None:
            b0_at_peak = wf_b0_gpu[pixel_ids_gpu, strongest_bins_gpu]
            tot_at_peak = wf_gpu[pixel_ids_gpu, strongest_bins_gpu]
            fracs_np = (b0_at_peak / tot_at_peak.clamp_min(1e-12)).cpu().numpy().astype(np.float64)
        else:
            fracs_np = None

        all_peaks = [[] for _ in range(self.n_pixels)]
        all_fracs = ([[] for _ in range(self.n_pixels)] if fracs_np is not None else None)
        for k in range(len(pixel_ids_np)):
            all_peaks[pixel_ids_np[k]].append((peak_dists_np[k], peak_amps_np[k]))
            if all_fracs is not None:
                all_fracs[pixel_ids_np[k]].append(float(fracs_np[k]))

        if all_fracs is not None:
            return all_peaks, all_fracs
        return all_peaks

    def _find_peaks_robust_cpu(self, wf_gpu, wf_b0_gpu=None,
                                height_frac: float = 0.10,
                                prominence_frac: float = 0.05,
                                min_distance_bins: int = 5,
                                min_width_bins: int = 1):
        """Per-channel robust peak detection (Ghost-FWL vis_pred.py style).

        Uses scipy.signal.find_peaks with 4 constraints:
          1. height       = max(channel_waveform) * height_frac    (default 10% of max)
          2. prominence   = max(channel_waveform) * prominence_frac (default 5% of max)
          3. distance     = min_distance_bins (default 5 bins → ~0.5m at 0.1m bin)
          4. width        = min_width_bins (default 3 bins → ~0.3m)

        These four together filter out:
          - tiny wiggles (height < 10% of main peak)
          - prominence < 5% of main peak (peaks "merged" with the main peak's tail)
          - peaks too close together (< 5 bins apart)
          - single-bin spikes (numerical noise)

        Notably, this matches Ghost-FWL's `vis_pred.py:detect_peaks_in_voxel`
        algorithm; we considered MAD-based prominence (their `percent` method)
        but it produces MAD ≈ 0 on our mostly-zero waveforms (most bins are 0,
        median deviation is 0), making it permissive instead of strict.

        Slower than _find_peaks_gpu (CPU + scipy + per-channel loop), but
        rejects ringing/noise peaks that the simple GPU 1%-of-max threshold
        lets through.
        """
        from scipy.signal import find_peaks

        wf_np = wf_gpu.cpu().numpy().astype(np.float32)
        wf_b0_np = (wf_b0_gpu.cpu().numpy().astype(np.float32)
                    if wf_b0_gpu is not None else None)

        all_peaks = [[] for _ in range(self.n_pixels)]
        all_fracs = ([[] for _ in range(self.n_pixels)]
                     if wf_b0_np is not None else None)

        for i in range(self.n_pixels):
            hist = wf_np[i]
            wf_max = float(hist.max())
            if wf_max <= 0:
                continue
            peaks, _ = find_peaks(
                hist,
                height=wf_max * height_frac,
                prominence=wf_max * prominence_frac,
                distance=min_distance_bins,
                width=min_width_bins,
            )
            if len(peaks) == 0:
                continue
            peak_amps = hist[peaks]
            order = np.argsort(-peak_amps)  # descending by amplitude
            for idx in order:
                bin_idx = int(peaks[idx])
                dist = float(self.bin_centers[bin_idx])
                amp = float(hist[bin_idx])
                all_peaks[i].append((dist, amp))
                if all_fracs is not None:
                    tot = float(hist[bin_idx])
                    b0  = float(wf_b0_np[i, bin_idx])
                    all_fracs[i].append(b0 / max(tot, 1e-12))

        if all_fracs is not None:
            return all_peaks, all_fracs
        return all_peaks

    def _find_peaks_gpu(self, wf_gpu, wf_b0_gpu=None):
        """
        GPU-accelerated peak detection.  Operates on a GPU tensor to avoid
        transferring the full (n_pixels, num_bins) waveform to CPU.

        Parameters
        ----------
        wf_gpu : torch.Tensor, shape (n_pixels, num_bins), on CUDA
        wf_b0_gpu : torch.Tensor, optional
            Bounce-0-only waveform, same shape. When provided, per-peak
            real_frac is returned alongside all_peaks.

        Returns
        -------
        all_peaks : list of lists
            all_peaks[i] = [(distance, amplitude), ...] sorted by amplitude desc.
        all_fracs : list of lists, only when wf_b0_gpu is given
            all_fracs[i] = [real_frac, ...], aligned with all_peaks[i].
        """
        # Local-maximum test on GPU (all comparisons stay on device)
        wf_max, _ = wf_gpu.max(dim=1)                     # (n_pixels,)
        active = wf_max > 0

        interior = wf_gpu[:, 1:-1]                         # (n_pixels, num_bins-2)
        is_local_max = (interior > wf_gpu[:, :-2]) & (interior > wf_gpu[:, 2:])
        above_thresh = interior >= (wf_max * 0.01).unsqueeze(1)
        peak_mask = is_local_max & above_thresh & active.unsqueeze(1)

        # Sparse extraction on GPU
        pixel_ids_gpu, bin_offsets_gpu = torch.where(peak_mask)
        actual_bins_gpu = bin_offsets_gpu + 1
        peak_amps_gpu = wf_gpu[pixel_ids_gpu, actual_bins_gpu].clamp(max=255.0)

        # Only transfer sparse results to CPU (small)
        pixel_ids_np = pixel_ids_gpu.cpu().numpy()
        actual_bins_np = actual_bins_gpu.cpu().numpy()
        peak_amps_np = peak_amps_gpu.cpu().numpy().astype(np.float64)
        peak_dists_np = self.bin_centers[actual_bins_np].astype(np.float64)

        # real_frac at each peak bin, queried against the *raw* total (no clamp)
        if wf_b0_gpu is not None:
            b0_at_peak = wf_b0_gpu[pixel_ids_gpu, actual_bins_gpu]
            tot_at_peak = wf_gpu[pixel_ids_gpu, actual_bins_gpu]
            fracs_np = (b0_at_peak / tot_at_peak.clamp_min(1e-12)).cpu().numpy().astype(np.float64)
        else:
            fracs_np = None

        # Assembly: O(n_peaks) Python loop — fast since n_peaks << n_pixels * num_bins
        all_peaks = [[] for _ in range(self.n_pixels)]
        all_fracs = ([[] for _ in range(self.n_pixels)] if fracs_np is not None else None)
        for k in range(len(pixel_ids_np)):
            all_peaks[pixel_ids_np[k]].append((peak_dists_np[k], peak_amps_np[k]))
            if all_fracs is not None:
                all_fracs[pixel_ids_np[k]].append(float(fracs_np[k]))

        # Sort by amplitude desc; keep fracs aligned with peaks
        for p in range(self.n_pixels):
            if len(all_peaks[p]) > 1:
                if all_fracs is not None:
                    order = sorted(range(len(all_peaks[p])),
                                   key=lambda i: all_peaks[p][i][1], reverse=True)
                    all_peaks[p] = [all_peaks[p][i] for i in order]
                    all_fracs[p] = [all_fracs[p][i] for i in order]
                else:
                    all_peaks[p].sort(key=lambda x: x[1], reverse=True)

        if all_fracs is not None:
            return all_peaks, all_fracs
        return all_peaks

    def _find_peaks(self, waveforms: np.ndarray) -> list:
        """
        Find local maxima in each channel's waveform.

        Delegates to GPU when available, falls back to CPU NumPy.

        Parameters
        ----------
        waveforms : ndarray, shape (n_pixels, num_bins)

        Returns
        -------
        all_peaks : list of lists
            all_peaks[i] = [(distance, amplitude), ...] sorted by amplitude desc.
        """
        if torch is not None and torch.cuda.is_available():
            wf_gpu = torch.from_numpy(waveforms).cuda()
            return self._find_peaks_gpu(wf_gpu)

        # CPU fallback
        wf_max = waveforms.max(axis=1)
        active = wf_max > 0
        interior = waveforms[:, 1:-1]
        is_local_max = (interior > waveforms[:, :-2]) & (interior > waveforms[:, 2:])
        above_thresh = interior >= (wf_max * 0.01)[:, None]
        peak_mask = is_local_max & above_thresh & active[:, None]

        pixel_ids, bin_offsets = np.where(peak_mask)
        actual_bins = bin_offsets + 1
        peak_dists = self.bin_centers[actual_bins].astype(np.float64)
        peak_amps  = np.minimum(waveforms[pixel_ids, actual_bins].astype(np.float64), 255.0)

        all_peaks = [[] for _ in range(self.n_pixels)]
        for k in range(len(pixel_ids)):
            all_peaks[pixel_ids[k]].append((peak_dists[k], peak_amps[k]))
        for p in range(self.n_pixels):
            if len(all_peaks[p]) > 1:
                all_peaks[p].sort(key=lambda x: x[1], reverse=True)

        return all_peaks

    def extract_world_point_cloud(self, all_peaks: list,
                                  echo_mode: str = 'strongest',
                                  rotation=None,
                                  all_fracs: list = None,
                                  tag_threshold: float = 0.5) -> tuple:
        """
        Extract point cloud in WORLD coordinates.

            point_world = self.origin + one_way * (R @ local_dir)

        Parameters
        ----------
        all_peaks : list of lists
            Output from reconstruct_all_channels().
        echo_mode : str
            'strongest'      — single return: highest amplitude peak
            'first'          — single return: nearest peak
            'last'           — single return: farthest peak
            'dual_strongest' — dual return: strongest + 2nd strongest (by amplitude)
            'dual_nearest'   — dual return: nearest + 2nd nearest (by distance)
            'dual'           — alias for 'dual_strongest' (backwards compat)
        rotation : ndarray (3, 3) or None
            Rotation matrix R (local → world). If None, R = I (sensor has
            only a translation stored in self.origin, no rotation).
        all_fracs : list of lists, optional
            Aligned with all_peaks. When provided, also returns per-point
            (tags, fracs) arrays. Obtain via
            ``reconstruct_all_channels(..., compute_tags=True)``.
        tag_threshold : float
            real_frac > threshold → tag = 0 (real); otherwise tag = 1 (ghost).

        Returns
        -------
        points   : ndarray (N, 5)  [x, y, z, intensity, echo_index]
        metadata : ndarray (N, 4)  [channel_w, channel_h, one_way_dist, peak_amplitude]
        tags     : ndarray (N,) int8, only when all_fracs is given
        fracs    : ndarray (N,) float32, only when all_fracs is given

        echo_index = 0 for the first/strongest return, 1 for the second return.
        For 'strongest'/'first'/'last' all rows have echo_index = 0.
        """
        directions = self.compute_beam_directions()  # (N, 3) local
        if rotation is not None:
            directions = (np.asarray(rotation, dtype=np.float64) @ directions.T).T

        want_tags = all_fracs is not None
        pts, meta, fracs_out = [], [], []
        for p in range(self.n_pixels):
            peaks = all_peaks[p]
            if not peaks:
                continue
            fracs = all_fracs[p] if want_tags else None
            w, h = p % self.W, p // self.W
            if echo_mode in ('dual', 'dual_strongest'):
                idx_sel = list(range(min(2, len(peaks))))
            elif echo_mode == 'dual_nearest':
                idx_sel = sorted(range(len(peaks)), key=lambda i: peaks[i][0])[:2]
            elif echo_mode == 'strongest':
                idx_sel = [0]
            elif echo_mode == 'first':
                idx_sel = [min(range(len(peaks)), key=lambda i: peaks[i][0])]
            elif echo_mode == 'last':
                idx_sel = [max(range(len(peaks)), key=lambda i: peaks[i][0])]
            else:
                raise ValueError(f"Unknown echo_mode: {echo_mode}")
            for echo_idx, peak_i in enumerate(idx_sel):
                dist_rt, amp = peaks[peak_i]
                one_way = dist_rt / 2.0
                pt = self.origin + one_way * directions[p]
                pts.append([pt[0], pt[1], pt[2], amp, float(echo_idx)])
                meta.append([w, h, one_way, amp])
                if want_tags:
                    fracs_out.append(fracs[peak_i])

        if not pts:
            empty_pts = np.zeros((0, 5))
            empty_meta = np.zeros((0, 4))
            if want_tags:
                return empty_pts, empty_meta, np.zeros((0,), dtype=np.int8), np.zeros((0,), dtype=np.float32)
            return empty_pts, empty_meta

        pts_arr = np.array(pts, dtype=np.float64)
        meta_arr = np.array(meta, dtype=np.float64)
        if want_tags:
            fracs_arr = np.asarray(fracs_out, dtype=np.float32)
            tags_arr = np.where(fracs_arr > tag_threshold, 0, 1).astype(np.int8)
            return pts_arr, meta_arr, tags_arr, fracs_arr
        return pts_arr, meta_arr

    def extract_local_point_cloud(self, all_peaks: list,
                                  echo_mode: str = 'strongest',
                                  all_fracs: list = None,
                                  tag_threshold: float = 0.5) -> tuple:
        """
        Extract point cloud in SENSOR-LOCAL coordinates.

            point_local = one_way * local_dir

        No origin offset, no rotation applied. Use this when you want
        coordinates in the sensor's own frame (X-forward, Y-left, Z-up).

        Parameters
        ----------
        all_peaks : list of lists
            Output from reconstruct_all_channels().
        echo_mode : str
            'strongest'      — single return: highest amplitude peak
            'first'          — single return: nearest peak
            'last'           — single return: farthest peak
            'dual_strongest' — dual return: strongest + 2nd strongest (by amplitude)
            'dual_nearest'   — dual return: nearest + 2nd nearest (by distance)
            'dual'           — alias for 'dual_strongest' (backwards compat)
        all_fracs : list of lists, optional
            See extract_world_point_cloud.
        tag_threshold : float
            See extract_world_point_cloud.

        Returns
        -------
        points   : ndarray (N, 5)  [x, y, z, intensity, echo_index]
        metadata : ndarray (N, 4)  [channel_w, channel_h, one_way_dist, peak_amplitude]
        tags     : ndarray (N,) int8, only when all_fracs is given
        fracs    : ndarray (N,) float32, only when all_fracs is given
        """
        directions = self.compute_beam_directions()  # (N, 3) local

        want_tags = all_fracs is not None
        pts, meta, fracs_out = [], [], []
        for p in range(self.n_pixels):
            peaks = all_peaks[p]
            if not peaks:
                continue
            fracs = all_fracs[p] if want_tags else None
            w, h = p % self.W, p // self.W
            if echo_mode in ('dual', 'dual_strongest'):
                idx_sel = list(range(min(2, len(peaks))))
            elif echo_mode == 'dual_nearest':
                idx_sel = sorted(range(len(peaks)), key=lambda i: peaks[i][0])[:2]
            elif echo_mode == 'strongest':
                idx_sel = [0]
            elif echo_mode == 'first':
                idx_sel = [min(range(len(peaks)), key=lambda i: peaks[i][0])]
            elif echo_mode == 'last':
                idx_sel = [max(range(len(peaks)), key=lambda i: peaks[i][0])]
            else:
                raise ValueError(f"Unknown echo_mode: {echo_mode}")
            for echo_idx, peak_i in enumerate(idx_sel):
                dist_rt, amp = peaks[peak_i]
                one_way = dist_rt / 2.0
                pt = one_way * directions[p]
                pts.append([pt[0], pt[1], pt[2], amp, float(echo_idx)])
                meta.append([w, h, one_way, amp])
                if want_tags:
                    fracs_out.append(fracs[peak_i])

        if not pts:
            empty_pts = np.zeros((0, 5))
            empty_meta = np.zeros((0, 4))
            if want_tags:
                return empty_pts, empty_meta, np.zeros((0,), dtype=np.int8), np.zeros((0,), dtype=np.float32)
            return empty_pts, empty_meta

        pts_arr = np.array(pts, dtype=np.float64)
        meta_arr = np.array(meta, dtype=np.float64)
        if want_tags:
            fracs_arr = np.asarray(fracs_out, dtype=np.float32)
            tags_arr = np.where(fracs_arr > tag_threshold, 0, 1).astype(np.int8)
            return pts_arr, meta_arr, tags_arr, fracs_arr
        return pts_arr, meta_arr

    def statistics(self, all_peaks: list) -> dict:
        """
        Compute statistics from peak detection results.

        Parameters
        ----------
        all_peaks : list of lists
            Output from reconstruct_all_channels().

        Returns
        -------
        stats : dict
            n_channels_any_peak, n_channels_multi_peak,
            peak_count_histogram, multi_peak_channels
        """
        n_any = 0
        n_multi = 0
        peak_counts = []
        multi_channels = []

        for p in range(self.n_pixels):
            n = len(all_peaks[p])
            peak_counts.append(n)
            if n > 0:
                n_any += 1
            if n > 1:
                n_multi += 1
                w = p % self.W
                h = p // self.W
                multi_channels.append((w, h, n, all_peaks[p]))

        # Histogram of peak counts
        max_peaks = max(peak_counts) if peak_counts else 0
        histogram = {}
        for c in range(max_peaks + 1):
            count = peak_counts.count(c)
            if count > 0:
                histogram[c] = count

        return {
            'n_channels_any_peak': n_any,
            'n_channels_multi_peak': n_multi,
            'n_total_channels': self.n_pixels,
            'peak_count_histogram': histogram,
            'multi_peak_channels': multi_channels,
        }

    def get_channel_shape_ids(self, scene) -> tuple:
        """
        Query the first-hit shape for every channel via a single batch ray intersection.

        Shoots one ray per channel along its beam direction and records which shape
        is hit first (geometrically closest hit, independent of BSDF type).

        The bounce-0 filter is necessary because multi-bounce paths (e.g. ground -> cube)
        also produce echoes: without it, channels whose first hit is the ground but whose
        strongest echo comes from a cube bounce would be incorrectly included.

        Parameters
        ----------
        scene : mi.Scene
            Loaded Mitsuba scene.  Must share the same scene_dict as the render that
            produced the raw AOVs being analysed (same geometry, same shape names).

        Returns
        -------
        shape_ids : ndarray, shape (n_pixels,), dtype int32
            Per-channel shape index into shape_id_map.  -1 means no hit.
        shape_id_map : dict {int -> str}
            Maps shape index to the shape's id string (the key used in scene_dict,
            e.g. 'cube', 'ground').
        """
        import mitsuba as mi
        from ad.beam_config import get_beam_config

        dirs_np = get_beam_config(self.beam_config)   # (n_pixels, 3), float32

        ox = float(self.origin[0])
        oy = float(self.origin[1])
        oz = float(self.origin[2])

        # Build one ray per channel — all fired in a single vectorized DrJit call
        rays = mi.Ray3f(
            o=mi.Point3f(ox, oy, oz),
            d=mi.Vector3f(
                mi.Float(dirs_np[:, 0].copy()),
                mi.Float(dirs_np[:, 1].copy()),
                mi.Float(dirs_np[:, 2].copy()),
            ),
        )

        # Single vectorized intersection — one GPU kernel covering all channels
        si_batch = scene.ray_intersect(rays)
        valid_np  = np.array(si_batch.is_valid())     # (n_pixels,) bool

        # Assign shape indices.
        # Python loop over N_shapes (not over rays): each iteration is a vectorized
        # DrJit comparison covering all channels at once, then converted to numpy.
        shapes       = scene.shapes()
        shape_ids_np = np.full(self.n_pixels, -1, dtype=np.int32)

        for i, shape in enumerate(shapes):
            hit_np = np.array(si_batch.shape == shape)    # (n_pixels,) bool
            shape_ids_np[valid_np & hit_np] = i

        shape_id_map = {i: s.id() for i, s in enumerate(shapes)}
        return shape_ids_np, shape_id_map

    def plot_channel_waveform(self, raw_aovs, spp, channel_w, channel_h, max_depth,
                              save_path, scene_label='', emitter_intensity=None,
                              waveforms=None):
        """
        Plot and save a 3-panel per-channel waveform figure.

        Panels (top to bottom):
          [0] Time-binning (conditional average per bin)
          [1] Transient KDE via self.reconstruct_all_channels()
              — with intensity calibration when emitter_intensity is provided
          [2] Raw sample scatter (distance vs radiance)

        Parameters
        ----------
        raw_aovs : list of ndarray
            Per-bounce AOV arrays from integrator.raw_aovs (length 3*max_depth).
        spp : int
            Samples per pixel.
        channel_w, channel_h : int
            Channel index within the film grid.
        max_depth : int
            Number of bounces stored in raw_aovs.
        save_path : str
            Output PNG path.
        scene_label : str
            Optional label appended to the figure suptitle.
        emitter_intensity : float, optional
            Passed to reconstruct_all_channels(); enables intensity calibration
            so the KDE panel shows values in [0, 255] reflectivity units.
        waveforms : ndarray, shape (n_pixels, num_bins), optional
            Pre-computed waveform array from reconstruct_all_channels().
            If provided, skips the expensive full re-render and uses this directly.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        pixel_flat = channel_h * self.W + channel_w
        start = pixel_flat * spp
        end   = start + spp

        all_rads  = np.zeros((spp, max_depth), dtype=np.float32)
        all_dists = np.zeros((spp, max_depth), dtype=np.float32)
        for i in range(max_depth):
            all_rads[:, i]  = raw_aovs[3 * i][start:end]
            all_dists[:, i] = raw_aovs[3 * i + 1][start:end]

        valid_mask_2d = all_rads > 0
        n_valid = int(valid_mask_2d.sum())

        # ---- [0] Time-binning ----
        t0 = pytime.time()
        bin_centers_tb, wf_tb, counts_tb = _time_bin_waveform(
            all_dists, all_rads, self.num_bins, self.max_range)
        t_tb = pytime.time() - t0

        # ---- [1] Transient KDE via reconstruct_all_channels ----
        t0 = pytime.time()
        if waveforms is None:
            waveforms, _ = self.reconstruct_all_channels(
                raw_aovs, spp, max_depth, emitter_intensity=emitter_intensity)
        wf_kde = waveforms[channel_h * self.W + channel_w]
        t_kde = pytime.time() - t0

        # ---- print timing ----
        print(f"\n  Reconstruction timing  "
              f"(spp={spp}, n_valid={n_valid}, bins={self.num_bins}, h={self.bandwidth}m):")
        print(f"    [0] Time-binning        : {t_tb *1e3:8.3f} ms")
        print(f"    [1] Transient KDE (GPU) : {t_kde*1e3:8.3f} ms")

        # ---- figure layout ----
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                  '#ff7f00', '#a65628', '#f781bf', '#999999']

        fig, axes = plt.subplots(
            3, 1, figsize=(14, 14),
            gridspec_kw={'height_ratios': [1.5, 1.5, 1.0]})

        # shared x range
        flat_valid = all_rads.flatten() > 0
        if flat_valid.sum() > 0:
            dv = all_dists.flatten()[flat_valid]
            dmin, dmax = dv.min(), dv.max()
            margin = max(2.0, (dmax - dmin) * 0.3)
            xlim = (max(0.0, dmin - margin), min(self.max_range, dmax + margin))
        else:
            xlim = (0.0, self.max_range)

        bin_w = self.max_range / self.num_bins

        # ---- panel 0: time-binning ----
        ax = axes[0]
        ax.bar(bin_centers_tb, wf_tb, width=bin_w * 0.9,
               color='steelblue', alpha=0.6, label='conditional avg per bin')
        nz_bins = np.where(counts_tb > 0)[0]
        for b in nz_bins:
            if xlim[0] <= bin_centers_tb[b] <= xlim[1]:
                ax.text(bin_centers_tb[b], wf_tb[b], f'{counts_tb[b]}',
                        ha='center', va='bottom', fontsize=6, color='darkblue')
        ax.set_ylabel('Radiance')
        ax.set_title(
            f'Channel (w={channel_w}, h={channel_h})  —  '
            f'[0] Time-Bin: Σr_i / count   bin_w={bin_w:.3f}m  '
            f'[{t_tb*1e3:.2f} ms]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*xlim)

        # ---- panel 1: transient KDE ----
        ax = axes[1]
        cal_str = f'  I_e={emitter_intensity}' if emitter_intensity is not None else ''
        ylabel = 'Intensity [0-255]' if emitter_intensity is not None else 'Signal (MC units)'
        ax.plot(self.bin_centers, wf_kde, color='purple', linewidth=2.0)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f'[1] Transient KDE (reconstruct_all_channels)   '
            f'spp={spp}  n_valid={n_valid}  h={self.bandwidth}m{cal_str}  '
            f'[{t_kde*1e3:.2f} ms]')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*xlim)

        # ---- panel 2: raw scatter ----
        ax = axes[2]
        for i in range(max_depth):
            mask = all_rads[:, i] > 0
            nz = mask.sum()
            if nz > 0:
                ax.scatter(all_dists[mask, i], all_rads[mask, i],
                           s=3, alpha=0.4, color=colors[i % len(colors)],
                           label=f'Bounce {i} ({nz})', zorder=2 + i)
        ax.set_xlabel('Round-trip Distance (m)')
        ax.set_ylabel('Radiance')
        ax.set_title(f'[2] Raw Samples — spp={spp}, max_depth={max_depth}')
        ax.legend(fontsize=8, loc='upper right', markerscale=3)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*xlim)

        label_str = f'  —  {scene_label}' if scene_label else ''
        fig.suptitle(
            f'Single-Channel Waveform Reconstruction{label_str}',
            fontsize=13, y=1.002)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Waveform plot saved: {save_path}")






def _tag_sidecar_paths(point_cloud_path: str):
    """Return (tag.npy, frac.npy) paths for a given <name>.bin file."""
    if point_cloud_path.endswith('.bin'):
        base = point_cloud_path[:-4]
    else:
        base = point_cloud_path
    return base + '.tag.npy', base + '.frac.npy'


def save_point_cloud_tags(point_cloud_path: str, tags: np.ndarray, fracs: np.ndarray):
    """
    Save real/ghost semantic tags as sidecar files alongside an LPC2 .bin point cloud.

    Two files are written:
        <name>.tag.npy   — int8 array of length N, 0=real, 1=ghost
        <name>.frac.npy  — float32 array of length N, raw real_frac (0..1)

    Keeping these as sidecars (rather than mutating LPC2) means existing
    consumers of the .bin file are unaffected. Pair with load_point_cloud_tags.

    Parameters
    ----------
    point_cloud_path : str
        Path to the corresponding LPC2 .bin point cloud file. The sidecars
        are named by replacing the .bin suffix.
    tags : ndarray (N,) int8
        Binary real/ghost label (0=real, 1=ghost) from extract_*_point_cloud
        with all_fracs supplied.
    fracs : ndarray (N,) float32
        Raw real_frac values, same length as tags.
    """
    tag_path, frac_path = _tag_sidecar_paths(point_cloud_path)
    tags_arr = np.ascontiguousarray(tags, dtype=np.int8)
    fracs_arr = np.ascontiguousarray(fracs, dtype=np.float32)
    if len(tags_arr) != len(fracs_arr):
        raise ValueError(f"tags ({len(tags_arr)}) and fracs ({len(fracs_arr)}) length mismatch")
    np.save(tag_path, tags_arr)
    np.save(frac_path, fracs_arr)
    n_real = int(np.sum(tags_arr == 0))
    n_ghost = int(np.sum(tags_arr == 1))
    print(f"  Saved tags:  {tag_path} ({n_real} real + {n_ghost} ghost = {len(tags_arr)} pts)")
    print(f"  Saved fracs: {frac_path}")


def load_point_cloud_tags(point_cloud_path: str):
    """
    Load tag/frac sidecars saved by save_point_cloud_tags.

    Returns
    -------
    tags : ndarray (N,) int8
    fracs : ndarray (N,) float32
    """
    tag_path, frac_path = _tag_sidecar_paths(point_cloud_path)
    tags = np.load(tag_path)
    fracs = np.load(frac_path)
    if len(tags) != len(fracs):
        raise ValueError(f"tag/frac length mismatch: {len(tags)} vs {len(fracs)}")
    return tags, fracs


def plot_point_cloud(points: np.ndarray, metadata: np.ndarray,
                     all_peaks: list, W: int, H: int,
                     save_path: str, title: str = 'LiDAR Point Cloud'):
    """
    Plot point cloud visualizations.

    Row 1: Top-down (X-Y), Side (X-Z), 3D perspective — colored by intensity
    Row 2: Same views colored by num_peaks + peak-count histogram
    Row 3 (shown when second-return points exist): first vs second return overlay

    Parameters
    ----------
    points   : ndarray (N, 5) [x, y, z, intensity, echo_index]
    metadata : ndarray (N, 4) [w, h, one_way_dist, amplitude]
    all_peaks : list of lists
    W, H : int
    save_path : str
    title : str
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if len(points) == 0:
        print(f"  No points to plot, skipping {save_path}")
        return

    # Split by echo index
    mask1 = points[:, 4] == 0
    mask2 = points[:, 4] == 1
    pts1 = points[mask1]
    pts2 = points[mask2]
    meta1 = metadata[mask1]

    x, y, z, intensity = pts1[:, 0], pts1[:, 1], pts1[:, 2], pts1[:, 3]

    # Normalize intensity for coloring
    int_norm = intensity / (intensity.max() + 1e-12)

    # Build per-point peak count from metadata (w, h) -> all_peaks index
    n_peaks_per_point = np.zeros(len(pts1), dtype=int)
    for i in range(len(pts1)):
        w_i = int(meta1[i, 0])
        h_i = int(meta1[i, 1])
        p_idx = h_i * W + w_i
        if p_idx < len(all_peaks):
            n_peaks_per_point[i] = len(all_peaks[p_idx])

    has_second = len(pts2) > 0
    n_rows = 3 if has_second else 2
    fig = plt.figure(figsize=(18, 5 * n_rows))

    # Row 1: Intensity-colored views
    ax1 = fig.add_subplot(n_rows, 3, 1)
    sc1 = ax1.scatter(x, y, c=int_norm, cmap='hot', s=1, alpha=0.7)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_title('Top-Down (X-Y) — Intensity')
    ax1.set_aspect('equal')
    plt.colorbar(sc1, ax=ax1, label='Intensity')

    ax2 = fig.add_subplot(n_rows, 3, 2)
    sc2 = ax2.scatter(x, z, c=int_norm, cmap='hot', s=1, alpha=0.7)
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Z (m)')
    ax2.set_title('Side (X-Z) — Intensity')
    ax2.set_aspect('equal')
    plt.colorbar(sc2, ax=ax2, label='Intensity')

    ax3 = fig.add_subplot(n_rows, 3, 3, projection='3d')
    ax3.scatter(x, y, z, c=int_norm, cmap='hot', s=1, alpha=0.5)
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.set_title('3D Perspective — Intensity')

    # Row 2: Peak count views + histogram
    ax4 = fig.add_subplot(n_rows, 3, 4)
    sc4 = ax4.scatter(x, y, c=n_peaks_per_point, cmap='viridis', s=1, alpha=0.7,
                      vmin=0, vmax=max(n_peaks_per_point.max(), 1))
    ax4.set_xlabel('X (m)'); ax4.set_ylabel('Y (m)')
    ax4.set_title('Top-Down (X-Y) — # Peaks')
    ax4.set_aspect('equal')
    plt.colorbar(sc4, ax=ax4, label='# Peaks')

    ax5 = fig.add_subplot(n_rows, 3, 5)
    sc5 = ax5.scatter(x, z, c=n_peaks_per_point, cmap='viridis', s=1, alpha=0.7,
                      vmin=0, vmax=max(n_peaks_per_point.max(), 1))
    ax5.set_xlabel('X (m)'); ax5.set_ylabel('Z (m)')
    ax5.set_title('Side (X-Z) — # Peaks')
    ax5.set_aspect('equal')
    plt.colorbar(sc5, ax=ax5, label='# Peaks')

    ax6 = fig.add_subplot(n_rows, 3, 6)
    peak_counts_all = [len(p) for p in all_peaks]
    max_count = max(peak_counts_all) if peak_counts_all else 0
    if max_count > 0:
        bins = np.arange(0, max_count + 2) - 0.5
        ax6.hist(peak_counts_all, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax6.set_xlabel('# Peaks per Channel'); ax6.set_ylabel('Count')
    ax6.set_title('Peak Count Distribution')

    # Row 3: First vs second return overlay
    if has_second:
        x2, y2, z2 = pts2[:, 0], pts2[:, 1], pts2[:, 2]

        ax7 = fig.add_subplot(n_rows, 3, 7)
        ax7.scatter(x, y, c='blue', s=1, alpha=0.5, label='1st return')
        ax7.scatter(x2, y2, c='red', s=1, alpha=0.5, label='2nd return')
        ax7.set_xlabel('X (m)'); ax7.set_ylabel('Y (m)')
        ax7.set_title('Dual Return — Top-Down (X-Y)')
        ax7.set_aspect('equal')
        ax7.legend(markerscale=5, fontsize=8)

        ax8 = fig.add_subplot(n_rows, 3, 8)
        ax8.scatter(x, z, c='blue', s=1, alpha=0.5, label='1st return')
        ax8.scatter(x2, z2, c='red', s=1, alpha=0.5, label='2nd return')
        ax8.set_xlabel('X (m)'); ax8.set_ylabel('Z (m)')
        ax8.set_title('Dual Return — Side (X-Z)')
        ax8.set_aspect('equal')
        ax8.legend(markerscale=5, fontsize=8)

        ax9 = fig.add_subplot(n_rows, 3, 9, projection='3d')
        ax9.scatter(x, y, z, c='blue', s=1, alpha=0.4, label='1st')
        ax9.scatter(x2, y2, z2, c='red', s=1, alpha=0.4, label='2nd')
        ax9.set_xlabel('X'); ax9.set_ylabel('Y'); ax9.set_zlabel('Z')
        ax9.set_title('Dual Return — 3D')
        ax9.legend(fontsize=8)

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Point cloud plot saved: {save_path}")


def view_point_cloud_open3d(points: np.ndarray,
                            scene_meshes: Optional[list] = None,
                            point_size: float = 3.0,
                            window_name: str = 'LiDAR Point Cloud'):
    """
    Interactive 3D point cloud viewer using Open3D.

    Parameters
    ----------
    points : ndarray (N, 5) [x, y, z, intensity, echo_index]
        Unified point cloud. First return (echo_index=0) colored by intensity
        (hot colormap); second return (echo_index=1) shown in blue.
    scene_meshes : list of open3d.geometry.TriangleMesh, optional
    point_size : float
    window_name : str
    """
    import open3d as o3d

    if len(points) == 0:
        print("  No points to display.")
        return

    mask1 = points[:, 4] == 0
    mask2 = points[:, 4] == 1
    pts1 = points[mask1]
    pts2 = points[mask2]

    geometries = []

    if len(pts1) > 0:
        xyz = pts1[:, :3]
        intensity = pts1[:, 3]
        int_norm = intensity / (intensity.max() + 1e-12)
        colors = np.zeros((len(pts1), 3), dtype=np.float64)
        colors[:, 0] = np.clip(int_norm * 3.0, 0, 1)
        colors[:, 1] = np.clip(int_norm * 3.0 - 1.0, 0, 1)
        colors[:, 2] = np.clip(int_norm * 3.0 - 2.0, 0, 1)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(xyz)
        pcd1.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd1)

    if len(pts2) > 0:
        xyz2 = pts2[:, :3]
        colors2 = np.full((len(pts2), 3), [0.2, 0.4, 1.0])
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(xyz2)
        pcd2.colors = o3d.utility.Vector3dVector(colors2)
        geometries.append(pcd2)

    if scene_meshes:
        for mesh in scene_meshes:
            mesh.compute_vertex_normals()
            geometries.append(mesh)

    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    origin_sphere.paint_uniform_color([0.0, 1.0, 0.0])
    geometries.append(origin_sphere)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geometries.append(axes)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1280, height=960,
        point_show_normal=False,
    )


# ======================================================================
# Waveform reconstruction utilities (single-channel, CPU)
# ======================================================================

def transient_kde_estimation(spp, values, distances, valid_mask,
                              target_dist_grid, h):
    """
    Monte Carlo KDE waveform reconstruction (vectorized, CPU).

    Formula: I(d) = (1/N) * sum_i [ (f(x_i)/p(x_i)) * K_h(d - d_i) ]
    where the sum is over ALL valid hits and N = spp (total rays fired).

    Correct MC KDE estimator: denominator is ``spp`` (total fired rays),
    not ``n_valid`` (hit count).  Dividing by spp preserves the absolute
    scale of the MC estimator — a pixel that rarely hits a surface yields
    a lower signal than one that consistently hits it.
    Fully vectorized: constructs the (M, K) distance matrix in a single
    NumPy broadcast, no Python loop over bin centres.

    Parameters
    ----------
    spp : int
        Total number of primary rays fired for this pixel.
    values : ndarray, shape (spp, max_depth) or (K,)
        Radiance contributions f(x)/p(x) per sample.
    distances : ndarray, shape (spp, max_depth) or (K,)
        Round-trip distances (same shape as ``values``).
    valid_mask : ndarray, bool, same shape as ``values``
        True for every sample that produced a non-zero hit.
    target_dist_grid : ndarray, shape (M,)
        Distance values at which to evaluate the waveform (x-axis).
    h : float
        Gaussian kernel bandwidth in the same units as ``distances``.

    Returns
    -------
    estimated_signal : ndarray, shape (M,), float64
    """
    valid_values    = np.asarray(values)[valid_mask].astype(np.float64)
    valid_distances = np.asarray(distances)[valid_mask].astype(np.float64)
    grid = np.asarray(target_dist_grid, dtype=np.float64)

    if len(valid_values) == 0:
        return np.zeros(len(grid), dtype=np.float64)

    # K_h(u) = 1/(h * sqrt(2*pi)) * exp(-0.5 * (u/h)^2)
    norm = 1.0 / (h * np.sqrt(2.0 * np.pi))
    diff = grid[:, None] - valid_distances[None, :]       # (M, K)
    weights = norm * np.exp(-0.5 * (diff / h) ** 2)       # (M, K)
    return np.sum(valid_values * weights, axis=1) / spp   # (M,)


def _time_bin_waveform(all_dists, all_rads, num_bins, max_range):
    """Conditional average radiance per distance bin (baseline)."""
    bin_edges   = np.linspace(0.0, max_range, num_bins + 1, dtype=np.float32)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    d_flat  = all_dists.flatten()
    r_flat  = all_rads.flatten()
    valid   = r_flat > 0
    d_v     = d_flat[valid]
    r_v     = r_flat[valid]

    idx = np.clip(np.digitize(d_v, bin_edges) - 1, 0, num_bins - 1)
    wf  = np.zeros(num_bins, dtype=np.float64)
    cnt = np.zeros(num_bins, dtype=np.int64)
    np.add.at(wf,  idx, r_v)
    np.add.at(cnt, idx, 1)
    nz = cnt > 0
    wf[nz] /= cnt[nz]
    return bin_centers, wf.astype(np.float32), cnt

from ad.io import save_point_cloud_bin, load_point_cloud_bin
