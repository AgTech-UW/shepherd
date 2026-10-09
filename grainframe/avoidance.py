"""Repulsive terms: the boids separation rule and the Khatib potential.

These are deliberately kept as two separate functions even though both are
repulsive artificial potential fields. The fanboids use the stock Reynolds
rule because that is the swarm model under study; the leaders use Khatib
because they are a controller we are free to design. Splitting them means a
change to the leader controller cannot silently alter the swarm results.

  separation -- Reynolds, "Flocks, Herds, and Schools", SIGGRAPH 1987.
  Khatib     -- "Real-Time Obstacle Avoidance for Manipulators and Mobile
                 Robots", IJRR 5(1), 1986.
"""
import numpy as np


def separationFlat(xb, yb, xOthers, yOthers, strength, range_):
    """STOCK Reynolds separation. Flat push proportional to displacement,
    cut off at the protected range. Do not change: this is the model.

    Note the shape of the failure this has. The force is LARGEST at the
    boundary (strength * range) and falls to zero at contact, and it jumps
    discontinuously from 0 the instant a neighbour crosses the range. That is
    backwards for an avoidance law, and is why the leaders use Khatib instead
    -- but the fanboids keep it, because reproducing the published model
    matters more than improving it.
    """
    dx = xb - xOthers
    dy = yb - yOthers
    D = np.hypot(dx, dy)
    tooClose = (D > 1e-9) & (D <= range_)
    return (float(np.sum(dx[tooClose]) * strength),
            float(np.sum(dy[tooClose]) * strength))


def repulsionKhatib(xb, yb, xOthers, yOthers, range_, gain, clip=0.25):
    """Khatib (1986) obstacle potential:  F = eta * (1/d - 1/range) / d^2.

    Zero at d = range and rising steeply toward contact -- the opposite
    profile to the flat rule, and continuous at the boundary so there is no
    step change when a neighbour comes into range.

    eta is scaled so that `gain` is literally the force magnitude at
    d = range/2, which makes the knob mean something physical. Note that a
    boids SF value does NOT transfer here: the two laws have completely
    different scales and gain is an independent parameter.

    clip: floor on d so the singularity at contact stays finite.
    """
    dx = xb - xOthers
    dy = yb - yOthers
    D = np.hypot(dx, dy)
    tooClose = (D > 1e-9) & (D <= range_)
    if not np.any(tooClose):
        return 0.0, 0.0

    d = np.clip(D[tooClose], clip, None)
    eta = gain * range_ ** 3 / 4.0
    w = eta * (1.0 / d - 1.0 / range_) / d ** 2       # magnitude
    ux, uy = dx[tooClose] / d, dy[tooClose] / d       # unit vector, away
    return float(np.sum(w * ux)), float(np.sum(w * uy))


def swirl(rx, ry, k):
    """Rotate a repulsion vector 90 degrees CCW in the WORLD frame and blend.

    The handedness tiebreaker. A symmetric head-on cannot be solved by
    repulsion alone: both agents sit on the mirror line, the repulsion points
    ALONG that line, and no gain breaks the tie -- they push each other
    backwards and lock.

    Check the sign on the head-on case. Red at the origin heading +x sees Gold
    at +d, so his repulsion is (-1, 0) and rot90 gives (0, -1): he goes to his
    own right. Gold's repulsion is (+1, 0) and rot90 gives (0, +1): also his
    own right. A starboard-to-starboard pass, which is what COLREGs Rule 14
    prescribes for a head-on.

    k = 0 disables it and restores the symmetric deadlock.
    """
    if k == 0.0:
        return rx, ry
    return rx + k * (-ry), ry + k * rx