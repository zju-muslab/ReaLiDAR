"""
Velodyne VLP-16 (Puck) LiDAR configuration.

Specifications
--------------
- Channels         : 16 vertical laser lines
- Rotation freq    : 10 Hz
- Horizontal FOV   : 360°
- Points per second: 300,000
- Vertical FOV     : -15° to +15° (2.0° step, uniform spacing)
- Max range        : 200 m
- Wavelength       : 903 nm

Derived horizontal channels
----------------------------
  W = PPS / (FREQ * H) = 300,000 / (10 × 16) = 1875

References
----------
  Velodyne VLP-16 User Manual, Rev F (2018)
"""

from ad.lidar_configs import LiDARConfig

CONFIG = LiDARConfig(
    name='vlp16',

    # Scanner geometry
    H=16,
    FREQ=10.0,
    PPS=300_000,

    # Vertical FOV (uniform linspace from VFOV_MIN with VFOV_STEP)
    VFOV_MIN=-15.0,
    VFOV_STEP=2.0,

    # Render quality
    MAX_DEPTH=8,
    SPP=1024,

    # Emitter / receiver optics
    # The VLP-16 manual specifies a 3.0 mrad full divergence in the
    # horizontal/scan direction (and 1.5 mrad vertically).  The renderer
    # currently models an isotropic Gaussian beam; its parameter is the
    # 1/e² radius, so use half of the horizontal full-angle specification.
    # A future anisotropic emitter can represent the 0.75 mrad vertical value.
    BEAM_DIVERGENCE=0.0015,   # Gaussian 1/e² half-angle, radians
    EMITTER_INTENSITY=1000.0,

    # Physical mounting
    SENSOR_HEIGHT=0.8,  # metres above z=0 ground plane

    # Waveform reconstruction
    MAX_RANGE=200.0,       # metres (round-trip)
    PULSE_WIDTH=5.0,       # ns
    DIST_RESOLUTION=0.05,  # metres per waveform bin
)
