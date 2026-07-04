"""
Turns a 2D scalar field into an RGB array for pygame, without matplotlib
(matplotlib's redraw is too slow for a live interactive loop).
"""
import numpy as np

_SPEED_STOPS = np.array([
    [0.05, 0.05, 0.20],
    [0.10, 0.30, 0.60],
    [0.10, 0.65, 0.65],
    [0.85, 0.85, 0.20],
    [0.90, 0.20, 0.15],
])

_DIVERGING_STOPS = np.array([
    [0.10, 0.25, 0.70],
    [0.55, 0.65, 0.85],
    [0.95, 0.95, 0.95],
    [0.90, 0.55, 0.35],
    [0.70, 0.10, 0.10],
])


def _lut_from_stops(stops, n=256):
    xs = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, n)
    r = np.interp(t, xs, stops[:, 0])
    g = np.interp(t, xs, stops[:, 1])
    b = np.interp(t, xs, stops[:, 2])
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


_LUTS = {
    "speed": _lut_from_stops(_SPEED_STOPS),
    "diverging": _lut_from_stops(_DIVERGING_STOPS),
}


def apply_colormap(field, vmin, vmax, style="speed"):
    lut = _LUTS[style]
    norm = (field - vmin) / max(vmax - vmin, 1e-9)
    idx = np.clip((norm * 255).astype(np.int32), 0, 255)
    return lut[idx]
