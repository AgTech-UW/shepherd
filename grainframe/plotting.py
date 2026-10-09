"""Plot helpers (the front end)."""
import numpy as np
from matplotlib import pyplot as plt


def plotBase(data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(data.refPath[:, 0], data.refPath[:, 1], 'b-', linewidth=2,
             label='Reference Dubins path')
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)


def addPose(pose, color, label=None, arrowLen=5.0):
    plt.quiver(pose[0], pose[1], arrowLen*np.cos(pose[2]), arrowLen*np.sin(pose[2]),
               color=color, angles='xy', scale_units='xy', scale=1, label=label)


def addFalcon(run, style='k--', label='Captain Falcon'):
    plt.plot(run['x'], run['y'], style, linewidth=1.5, label=label)


def addDistractors(run, label='Distractor boids'):
    if run['boidX'].size == 0:
        return
    nB = run['boidX'].shape[1]
    for b in range(nB):
        plt.plot(run['boidX'][:, b], run['boidY'][:, b], '-', color='0.8',
                 linewidth=0.6, zorder=1, label=label if b == 0 else None)
    plt.plot(run['boidX'][-1, :], run['boidY'][-1, :], '.', color='0.5',
             markersize=6, zorder=1)


def addFanboids(run, label='Fanboids', withStarts=True, withArrows=True):
    if run['fanX'].size == 0:
        return None
    nB = run['fanX'].shape[1]
    cols = plt.cm.viridis(np.linspace(0.15, 0.9, nB))
    for b in range(nB):
        plt.plot(run['fanX'][:, b], run['fanY'][:, b], '-', color=cols[b],
                 linewidth=0.8, zorder=1, label=label if b == 0 else None)
    if withStarts:
        plt.plot(run['fanX0'], run['fanY0'], 'o', color='0.4', mfc='none',
                 markersize=6, label='Fan start clumps')
    if withArrows:
        psiEnd = np.arctan2(run['fanVy'][-1], run['fanVx'][-1])
        plt.quiver(run['fanX'][-1], run['fanY'][-1],
                   2*np.cos(psiEnd), 2*np.sin(psiEnd),
                   color=cols, angles='xy', scale_units='xy', scale=1, width=0.004)
    return cols


def addWaypoints(data):
    plt.plot(data.wayPts[:, 0], data.wayPts[:, 1], 'rs', markersize=7,
             mfc='none', label='Waypoints')
