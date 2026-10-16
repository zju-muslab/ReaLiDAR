"""
RoboSense RS-M1 Plus solid-state LiDAR configuration.

Specifications (from C++ simulation reference)
-----------------------------------------------
- Type              : Solid-state (non-rotating)
- Channels (rings)  : 5  (ignored here — all beams treated as independent)
- Total beams/frame : 78,750   (5 rings × 15,750 beams each)
- Points per second : 787,500
- Frame rate        : 10 Hz
- Horizontal FOV    : ~120°  (approx. -61° to +62° azimuth)
- Vertical FOV      : ~26°   (approx. -14° to +12° elevation)
- Max range         : 200 m
- Noise std dev     : 0.02753913 m
- Wavelength        : 905 nm  (typical)

Beam layout
-----------
All 78,750 beams are flattened into a single independent list loaded from
``rs_m1plus_all_rings_angles.txt`` (ring_id column is ignored).

Film dimensions:
  H = 1   (single "row" — all beams independent)
  W = PPS / (FREQ × H) = 787,500 / (10 × 1) = 78,750

The beam_file path is relative to this package directory.

References
----------
  Angle table   : rs_m1plus_all_rings_angles.txt  (ring  horizontal  vertical)
"""

from ad.lidar_configs import LiDARConfig

CONFIG = LiDARConfig(
    name='rs_m1plus',

    # Scanner geometry — H=1 flattens all beams into a single independent list
    H=1,
    FREQ=10.0,
    PPS=787_500,           # → W = 787500 / (10 × 1) = 78,750 beams per frame

    # VFOV fields are unused for solid-state (beam_file overrides them)
    VFOV_MIN=0.0,
    VFOV_STEP=0.0,

    # Render quality
    MAX_DEPTH=8,
    SPP=1024,

    # Emitter / receiver optics
    BEAM_ANGLE=0.15,    # receiver half-cone angle, degrees
    BEAM_WIDTH=0.075,   # emitted beam half-width, degrees
    EMITTER_INTENSITY=1000.0,

    # Physical mounting
    SENSOR_HEIGHT=0.8,  # metres above z=0 ground plane

    # Waveform reconstruction
    MAX_RANGE=200.0,       # metres (round-trip)
    PULSE_WIDTH=5.0,       # ns
    DIST_RESOLUTION=0.05,  # metres per waveform bin

    # Solid-state beam pattern: load per-beam (horizontal, vertical) angles
    # Format per line:  ring_id  horizontal_deg  vertical_deg
    # ring_id is ignored — all beams are treated as independent parallel beams
    beam_file='rs_m1plus_all_rings_angles.txt',
)
