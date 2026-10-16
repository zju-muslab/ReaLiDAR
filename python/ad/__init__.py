"""Explicit forward-only plugin registration for reviewed reproduction profiles."""
import importlib
import sys
from .beam_config import register_beam_config, get_beam_config, list_beam_configs

_profile = None


def configure(profile):
    global _profile
    if profile not in ('rs_m1_paper', 'vlp16_paper'):
        raise ValueError(f'Unknown reproduction profile: {profile}')
    if _profile is not None:
        if _profile != profile:
            raise RuntimeError('Use a fresh process when switching reproduction profiles')
        return
    import mitsuba as mi
    if mi.variant() != 'cuda_ad_rgb':
        raise RuntimeError('Set the validated cuda_ad_rgb variant before configuration')
    importlib.import_module('ad.sensors.lidar_sensor')
    importlib.import_module('ad.integrators.lidar_waveform')
    importlib.import_module(f'ad.models.{profile}.lidar_spot')
    # Independent XML names allow both material models in the same scene.
    importlib.import_module('ad.models.rs_m1_paper.retro_reflective')
    importlib.import_module('ad.models.vlp16_paper.retro_reflective')
    _profile = profile
