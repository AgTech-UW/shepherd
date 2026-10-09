"""Load the MATLAB-exported Dubins CSV datasets."""
import os
from dataclasses import dataclass
import numpy as np


@dataclass
class Dataset:
    tag: str
    refPath: np.ndarray   # (M, >=2) reference Dubins path, NaN rows removed
    wayPts: np.ndarray    # (P, 2) waypoints
    qi: np.ndarray        # start pose (x, y, heading)
    qf: np.ndarray        # goal pose (x, y, heading)
    R: float
    v: float
    dt: float
    refLen: float         # total arc length of the path
    bbox: tuple           # (xmin, xmax, ymin, ymax) of the path


def load_dataset(tag='1', dataDir=None):
    if dataDir is None:
        # CSVs live in the data/ folder at the repo root
        repoRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataDir = os.path.join(repoRoot, 'data')

    p = lambda name: os.path.join(dataDir, 'saved_%s_%s.csv' % (name, tag))

    refPath = np.loadtxt(p('dubins_path'), delimiter=',')
    wayPts  = np.loadtxt(p('dubins_waypoints'), delimiter=',')
    qi      = np.loadtxt(p('qi'), delimiter=',')
    qf      = np.loadtxt(p('qf'), delimiter=',')
    pR, pV, pDt = np.loadtxt(p('params'), delimiter=',')

    # Remove any rows of the path that contain NaN.
    refPath = refPath[~np.isnan(refPath).any(axis=1)]

    # Total arc length of the path
    segLen = np.hypot(np.diff(refPath[:, 0]), np.diff(refPath[:, 1]))
    refLen = float(np.sum(segLen))

    bbox = (refPath[:, 0].min(), refPath[:, 0].max(),
            refPath[:, 1].min(), refPath[:, 1].max())

    return Dataset(tag=tag, refPath=refPath, wayPts=wayPts, qi=qi, qf=qf,
                   R=float(pR), v=float(pV), dt=float(pDt),
                   refLen=refLen, bbox=bbox)


def apply_dataset(opts, data):
    """Copy the vehicle params from the dataset into the options."""
    opts.R = data.R
    opts.v = data.v
    opts.dt = data.dt
    return opts