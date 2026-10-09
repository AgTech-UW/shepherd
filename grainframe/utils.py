"""Small math helpers."""
import numpy as np


def wrapToPi(a):
    # Keeps angles between -pi and pi
    return (a + np.pi) % (2 * np.pi) - np.pi


def clampSpeeds(vx, vy, opts):
    # Rescale vector so its length is in [minSpeed, maxSpeed]
    s = np.hypot(vx, vy)
    s = np.maximum(s, 1e-9)
    sNew = np.clip(s, opts.minSpeed, opts.maxSpeed)
    scale = sNew / s
    return vx * scale, vy * scale
