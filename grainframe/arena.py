"""The arena: path bounding box grown by padding, plus the safe interior."""
from dataclasses import dataclass


@dataclass
class Arena:
    xFMin: float
    xFMax: float
    yFMin: float
    yFMax: float
    xSafeMin: float
    xSafeMax: float
    ySafeMin: float
    ySafeMax: float
    # spawn box = raw path bbox (boids get spawned inside this)
    xSpawnMin: float
    xSpawnMax: float
    ySpawnMin: float
    ySpawnMax: float


def make_arena(data, opts):
    xMin, xMax, yMin, yMax = data.bbox
    pad = 2.0 * opts.R
    xFMin = min(xMin, data.qi[0], data.qf[0]) - pad
    xFMax = max(xMax, data.qi[0], data.qf[0]) + pad
    yFMin = min(yMin, data.qi[1], data.qf[1]) - pad
    yFMax = max(yMax, data.qi[1], data.qf[1]) + pad

    return Arena(
        xFMin=xFMin, xFMax=xFMax, yFMin=yFMin, yFMax=yFMax,
        xSafeMin=xFMin + opts.safetyF, xSafeMax=xFMax - opts.safetyF,
        ySafeMin=yFMin + opts.safetyF, ySafeMax=yFMax - opts.safetyF,
        xSpawnMin=xMin, xSpawnMax=xMax, ySpawnMin=yMin, ySpawnMax=yMax,
    )
