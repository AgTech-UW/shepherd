"""Build synthetic straight-line paths (no MATLAB CSV needed).

For the two-captain crossing experiment we want dead-simple straight paths
that cross in an X, so that if nobody dodges, the captains collide at the
crossing point.
"""
import numpy as np
from .data_io import Dataset


def straightPath(x0, y0, x1, y1, nWayPts=9, nSamples=400):
    """A straight line from (x0,y0) to (x1,y1), as a Dataset.

    Returns the same Dataset shape the CSV loader produces, so everything
    downstream (arena, simulate, plotting) works unchanged.
    """
    heading = np.arctan2(y1 - y0, x1 - x0)      # the line's direction, constant the whole way

    # Dense sample of the line -> the "reference path" we score against
    ts = np.linspace(0.0, 1.0, nSamples)        # 0 -> 1 along the segment
    refPath = np.column_stack([x0 + ts * (x1 - x0),
                               y0 + ts * (y1 - y0)])

    # Sparse sample -> the waypoints the captain actually chases
    tw = np.linspace(0.0, 1.0, nWayPts)
    wayPts = np.column_stack([x0 + tw * (x1 - x0),
                              y0 + tw * (y1 - y0)])

    qi = np.array([x0, y0, heading])             # start pose
    qf = np.array([x1, y1, heading])             # goal pose (same heading, it's a line)

    refLen = float(np.hypot(x1 - x0, y1 - y0))   # arc length of a straight line = its length
    bbox = (refPath[:, 0].min(), refPath[:, 0].max(),
            refPath[:, 1].min(), refPath[:, 1].max())

    return Dataset(tag='synthetic', refPath=refPath, wayPts=wayPts,
                   qi=qi, qf=qf, R=5.0, v=1.0, dt=0.01,
                   refLen=refLen, bbox=bbox)


def crossingPaths(halfLen=40.0, nWayPts=9):
    """Two straight paths that cross in an X at the origin.

    Both captains start the same distance from the crossing and move at the
    same speed, so they arrive at the middle at the SAME time -- a guaranteed
    collision unless the avoidance rules kick in.
    """
    dataA = straightPath(-halfLen, -halfLen, +halfLen, +halfLen, nWayPts)  # SW -> NE
    dataB = straightPath(-halfLen, +halfLen, +halfLen, -halfLen, nWayPts)  # NW -> SE
    return dataA, dataB


def starPaths(nCaps=3, halfLen=40.0, nWayPts=9):
    """N straight paths all crossing at the origin, spokes of a star.

    Each captain starts on the rim and drives through the middle to the far
    side. With N captains they ALL arrive at the crossing together -- the
    same conflict as the X, just worse. This is the "if it works for 2, it
    works for N" check: nothing in the simulator is hard-coded to two.
    """
    datasets = []
    for k in range(nCaps):
        ang = np.pi * k / nCaps              # spread the spokes evenly over 180 deg
        x0, y0 = -halfLen * np.cos(ang), -halfLen * np.sin(ang)
        x1, y1 = +halfLen * np.cos(ang), +halfLen * np.sin(ang)
        datasets.append(straightPath(x0, y0, x1, y1, nWayPts))
    return datasets


def mergedArenaBox(datasets, opts):
    """One arena big enough to hold every fleet's path."""
    xMin = min(d.bbox[0] for d in datasets)
    xMax = max(d.bbox[1] for d in datasets)
    yMin = min(d.bbox[2] for d in datasets)
    yMax = max(d.bbox[3] for d in datasets)
    return xMin, xMax, yMin, yMax