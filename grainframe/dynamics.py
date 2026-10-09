"""Behavior rules and vehicle kinematics (the back end)."""
import numpy as np
from .utils import wrapToPi, clampSpeeds


def boidsRules(xb, yb, vxb, vyb, xAll, yAll, vxAll, vyAll, opts):
    # Vector FROM every other agent TO this boid, and the distances.
    dx = xb - xAll
    dy = yb - yAll
    D = np.sqrt(dx**2 + dy**2)

    idVisual    = (D > 1e-9) & (D <= opts.VR)   # neighbors I can see
    idProtected = (D > 1e-9) & (D <= opts.PR)   # neighbors that are TOO CLOSE

    # 1. Separation: push away from too-close neighbors.
    vxSeparation = np.sum(dx[idProtected]) * opts.SF
    vySeparation = np.sum(dy[idProtected]) * opts.SF

    if np.count_nonzero(idVisual) > 0:
        # Averages over the visible neighbors.
        xPosAvg = np.mean(xAll[idVisual])
        yPosAvg = np.mean(yAll[idVisual])
        xVelAvg = np.mean(vxAll[idVisual])
        yVelAvg = np.mean(vyAll[idVisual])

        # 2. Cohesion: steer toward the neighbors' center of mass.
        vxCohesion = (xPosAvg - xb) * opts.CF
        vyCohesion = (yPosAvg - yb) * opts.CF

        # 3. Alignment: steer toward the neighbors' average velocity.
        vxAlignment = (xVelAvg - vxb) * opts.AF
        vyAlignment = (yVelAvg - vyb) * opts.AF
    else:
        # Nobody in sight: no cohesion, no alignment.
        vxCohesion = vyCohesion = vxAlignment = vyAlignment = 0.0

    return (float(vxSeparation), float(vySeparation),
            float(vxAlignment),  float(vyAlignment),
            float(vxCohesion),   float(vyCohesion))


def fanLeaderRule(xb, yb, vxb, vyb, xCap, yCap, vxCap, vyCap, opts):
    # 1. Positional pull (Cohesion with the Captain)
    kx = (xCap - xb) * opts.fanLeaderFactor
    ky = (yCap - yb) * opts.fanLeaderFactor
    
    # 2. Velocity matching (Alignment with the Captain)
    # Note: Make sure to add 'fanLeaderVelocityFactor' to your opts class!
    kvx = (vxCap - vxb) * opts.fanLeaderVelocityFactor
    kvy = (vyCap - vyb) * opts.fanLeaderVelocityFactor
    
    # Return the combined forces
    return kx + kvx, ky + kvy

def falconSteering(thetai, vxTotal, vyTotal, opts):
    # Dubins car physics... straight, max-left, or max-right
    nrm = np.hypot(vxTotal, vyTotal)
    if nrm < 1e-12:
        return 0.0  # prevent dividing by zero

    # His current velocity vector.
    vi = (opts.v * np.cos(thetai), opts.v * np.sin(thetai))

    # Angle between current velocity and desired velocity.
    cosAng = np.dot(vi, (vxTotal, vyTotal)) / (opts.v * nrm)
    cosAng = np.clip(cosAng, -1, 1)
    totalAngle = np.arccos(cosAng)

    # Sign of the z component of the cross product tells LEFT vs RIGHT.
    crossZ = vi[0] * vyTotal - vi[1] * vxTotal

    if totalAngle < opts.dAng:
        controlAngle = 0        # go straight
    elif crossZ > 0:
        controlAngle = -1       # turn left
    else:
        controlAngle = 1        # turn right

    return -opts.v * controlAngle / opts.R


def ackermannClamp(vx, vy, vxDes, vyDes, opts):
    # The car constraint.. limits point-mass turning to a physical turning radius
    sDes = np.hypot(vxDes, vyDes)
    psi = np.arctan2(vy, vx)

    # Actual speed: clamped into [minSpeed, maxSpeed].
    s = np.clip(sDes, opts.minSpeed, opts.maxSpeed)

    # Maximum heading change allowed this step.
    dpsiMax = (s / opts.Rboid) * opts.dt

    # How far they want to turn, wrapped and clamped to what the car can do
    psiDes = np.arctan2(vyDes, vxDes)
    dpsi = wrapToPi(psiDes - psi)
    dpsi = np.clip(dpsi, -dpsiMax, +dpsiMax)

    # New heading and resulting velocities
    psiNew = psi + dpsi
    vxNew = s * np.cos(psiNew)
    vyNew = s * np.sin(psiNew)

    omega = dpsi / opts.dt     # heading rate for the final report
    return vxNew, vyNew, omega


class WaypointTracker:
    """Closest Point Approach (CPA) handover logic (was opts.minDist state)."""

    def __init__(self, opts):
        self.opts = opts
        self.minDist = np.inf

    def reset(self):
        self.minDist = np.inf

    def advance(self, xi, yi, idP, wayPts):
        opts = self.opts
        nP = wayPts.shape[0]
        if idP >= nP - 1:
            return idP

        d = np.hypot(wayPts[idP, 0] - xi, wayPts[idP, 1] - yi)
        self.minDist = min(self.minDist, d)

        closeEnough  = self.minDist < opts.passWindow
        movingAwayBy = d > (self.minDist + opts.hysteresis)

        # Hand over once he physically passes it and moves away
        if closeEnough and movingAwayBy:
            idP += 1
            self.minDist = np.inf
            # Skip duplicate waypoints
            while idP < nP - 1 and np.hypot(wayPts[idP, 0] - xi,
                                            wayPts[idP, 1] - yi) < 1e-5:
                idP += 1
        return idP


def initUniform(n, arena, opts):
    # Distractors: random positions all over the map
    x = np.random.uniform(arena.xSpawnMin, arena.xSpawnMax, n)
    y = np.random.uniform(arena.ySpawnMin, arena.ySpawnMax, n)

    vx = (np.random.rand(n) - 0.5) * 2 * opts.maxSpeed
    vy = (np.random.rand(n) - 0.5) * 2 * opts.maxSpeed

    vx, vy = clampSpeeds(vx, vy, opts)
    return x, y, vx, vy


def initClumped(n, arena, opts):
    # Fanboids: random clumps with gaussian noise
    k = np.random.randint(1, min(opts.maxClumps, n) + 1)

    clumpCentersX = np.random.uniform(arena.xSpawnMin, arena.xSpawnMax, k)
    clumpCentersY = np.random.uniform(arena.ySpawnMin, arena.ySpawnMax, k)

    clumpAssignments = np.random.randint(0, k, n)
    x = clumpCentersX[clumpAssignments] + np.random.normal(0, 2.0, n)
    y = clumpCentersY[clumpAssignments] + np.random.normal(0, 2.0, n)

    headings = np.random.uniform(0, 2 * np.pi, n)
    speeds = np.random.uniform(opts.minSpeed, opts.maxSpeed, n)
    vx = speeds * np.cos(headings)
    vy = speeds * np.sin(headings)

    return x, y, vx, vy
