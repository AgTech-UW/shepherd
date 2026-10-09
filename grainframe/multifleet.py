"""Multi-fleet simulation: N captains, each with their own path and fanboids.

The rules, per Adam's spec:
  - Each captain follows HIS OWN path, unerringly, most of the time.
  - BUT when another agent enters his protected range (PR), the SAC
    (separation / avoidance / collision) rule suddenly switches on and gets
    added to his path-following pull. He dodges, then -- because the leader
    pull never stopped -- he slides right back onto his path.
  - Fanboids stick with THEIR OWN captain (leader pull) and flock with their
    OWN fleet (cohesion + alignment), but separation applies CROSS-FLEET, so
    everybody dodges everybody regardless of team.

Bookkeeping note: every agent carries a fleet id. Same-fleet vs cross-fleet
is then just a mask comparison, which is what makes "if it works for 2, it
works for N" true -- nothing here is hard-coded to two.
"""
import numpy as np
from .utils import wrapToPi
from .dynamics import (boidsRules, falconSteering, ackermannClamp,
                       WaypointTracker, initClumped)


class Fleet:
    """One captain + his fanboids + the path he's trying to walk."""

    def __init__(self, name, data, nFan=0, color='C0'):
        self.name = name
        self.data = data          # his own Dataset (path, waypoints, poses)
        self.nFan = nFan          # how many fanboids ride with him
        self.color = color        # for plotting

        # Captain state, initialized at his start pose
        self.x = float(data.qi[0])
        self.y = float(data.qi[1])
        self.theta = float(data.qi[2])

        self.idP = 1                        # which waypoint he's chasing
        self.tracker = None                 # CPA waypoint handover (built at run time)
        self.done = False                   # has he finished his path?
        self.goalMin = np.inf               # closest he ever got to his last waypoint

        # Fanboid state (filled in by the runner)
        self.fanX = np.array([])
        self.fanY = np.array([])
        self.fanVx = np.array([])
        self.fanVy = np.array([])

        # Logs
        self.log = {'x': [], 'y': [], 'theta': [], 'avoiding': [],
                    'fanX': [], 'fanY': [], 'fanVx': [], 'fanVy': []}


def _separationFrom(xSelf, ySelf, xOthers, yOthers, opts, range_=None):
    """SAC rule: push away from anyone inside my protected range.

    Returns (pushX, pushY, isAvoiding). isAvoiding tells us whether the rule
    actually fired this step -- that's the "suddenly applies" switch.
    """
    if range_ is None:
        range_ = opts.PR                    # boids use PR; captains pass in capPR

    if xOthers.size == 0:
        return 0.0, 0.0, False, 0.0

    dx = xSelf - xOthers                    # vector FROM each other agent TO me
    dy = ySelf - yOthers
    D = np.hypot(dx, dy)

    tooClose = (D > 1e-9) & (D <= range_)   # inside my protected zone?
    if not np.any(tooClose):
        return 0.0, 0.0, False, 0.0         # nobody close -> rule stays OFF

    # Inverse-distance weighting: the closer they are, the harder the shove.
    w = (range_ / D[tooClose]) - 1.0        # 0 at the edge of the zone, big up close
    pushX = float(np.sum(dx[tooClose] / D[tooClose] * w) * opts.SF)
    pushY = float(np.sum(dy[tooClose] / D[tooClose] * w) * opts.SF)
    threat = float(np.sum(w))               # how much danger am I in right now?
    return pushX, pushY, True, threat


def runFleets(fleets, arena, opts, seed=None):
    """Run every fleet at once on a shared clock.

    Returns the fleets (with their logs filled in) plus a combined history
    that the collision metrics can chew on.
    """
    if seed is None:
        seed = opts.seed
    np.random.seed(seed)

    dt = opts.dt
    v = opts.v

    # ---- Set up each fleet ----
    for fl in fleets:
        fl.tracker = WaypointTracker(opts)              # fresh CPA memory
        if fl.nFan > 0:
            fx, fy, fvx, fvy = initClumped(fl.nFan, arena, opts)
            # Start each fleet's fanboids near THEIR captain, not scattered
            fx = fx * 0.15 + fl.x + np.random.normal(0, 3.0, fl.nFan)
            fy = fy * 0.15 + fl.y + np.random.normal(0, 3.0, fl.nFan)
            fl.fanX, fl.fanY, fl.fanVx, fl.fanVy = fx, fy, fvx, fvy

    # Longest path decides how much time everyone gets
    tMax = max(opts.tMax(fl.data.refLen) for fl in fleets)
    t = 0.0

    while t <= tMax:
        # ================= SNAPSHOT: where is everybody right now? =================
        # Captains
        capX = np.array([fl.x for fl in fleets])
        capY = np.array([fl.y for fl in fleets])

        # Fanboids, flattened, remembering which fleet each one belongs to
        allFanX, allFanY, allFanVx, allFanVy, fanFleet = [], [], [], [], []
        for k, fl in enumerate(fleets):
            if fl.nFan > 0:
                allFanX.append(fl.fanX)
                allFanY.append(fl.fanY)
                allFanVx.append(fl.fanVx)
                allFanVy.append(fl.fanVy)
                fanFleet.append(np.full(fl.nFan, k))     # <- the bookkeeping tag
        if allFanX:
            allFanX = np.concatenate(allFanX)
            allFanY = np.concatenate(allFanY)
            allFanVx = np.concatenate(allFanVx)
            allFanVy = np.concatenate(allFanVy)
            fanFleet = np.concatenate(fanFleet)
        else:
            allFanX = allFanY = allFanVx = allFanVy = fanFleet = np.array([])

        # ================= CAPTAINS =================
        newCapState = []
        for k, fl in enumerate(fleets):
            if fl.done:
                newCapState.append((fl.x, fl.y, fl.theta, False))
                continue

            wayPts = fl.data.wayPts
            nP = wayPts.shape[0]

            fl.idP = fl.tracker.advance(fl.x, fl.y, fl.idP, wayPts)
            xLeader = wayPts[fl.idP, 0]
            yLeader = wayPts[fl.idP, 1]
            distLeader = np.hypot(xLeader - fl.x, yLeader - fl.y)

            # Finished? (closest approach to his final waypoint, then moving away)
            if fl.idP == nP - 1:
                if distLeader < fl.goalMin:
                    fl.goalMin = distLeader
                passedIt = (fl.goalMin < opts.passWindow and
                            distLeader > fl.goalMin + opts.hysteresis)
                if passedIt:
                    fl.done = True
                    newCapState.append((fl.x, fl.y, fl.theta, False))
                    continue

            # 1. Follow the leader: pull toward my next waypoint (ALWAYS on)
            vxLeader = (xLeader - fl.x) * opts.leaderFactor
            vyLeader = (yLeader - fl.y) * opts.leaderFactor

            # 2. SAC: who else is near me? Other captains always count.
            #    Other fleets' fanboids count too. My OWN fanboids only if
            #    capAvoidsOwnFans -- otherwise my fleet keeps clear of me and I
            #    can walk my path unerringly.
            otherCapX = np.delete(capX, k)          # every captain but me
            otherCapY = np.delete(capY, k)

            if allFanX.size:
                if opts.capAvoidsOwnFans:
                    foreignFans = np.ones(allFanX.size, dtype=bool)
                else:
                    foreignFans = (fanFleet != k)   # <- only OTHER fleets' boids
                nearbyX = np.concatenate([otherCapX, allFanX[foreignFans]])
                nearbyY = np.concatenate([otherCapY, allFanY[foreignFans]])
            else:
                nearbyX, nearbyY = otherCapX, otherCapY

            pushX, pushY, isAvoiding, threat = _separationFrom(
                fl.x, fl.y, nearbyX, nearbyY, opts, range_=opts.capPR)

            # Rules of the road: also veer RIGHT of my own heading while avoiding.
            # This is the tiebreaker. Two captains in a symmetric crossing get
            # pushed along the SAME mirror line by plain repulsion, so they never
            # actually separate -- but if both consistently give way to the right,
            # the symmetry breaks and they pass cleanly.
            if isAvoiding:
                rightX = np.sin(fl.theta)       # unit vector 90 deg right of heading
                rightY = -np.cos(fl.theta)
                pushX += opts.capSwirl * opts.SF * threat * rightX
                pushY += opts.capSwirl * opts.SF * threat * rightY

            # Blend: path pull always, avoidance only when someone's in the zone
            vxTotal = vxLeader + opts.capAvoidWeight * pushX
            vyTotal = vyLeader + opts.capAvoidWeight * pushY

            # 3. Dubins steering + move
            vTheta = falconSteering(fl.theta, vxTotal, vyTotal, opts)
            newX = fl.x + v * np.cos(fl.theta) * dt
            newY = fl.y + v * np.sin(fl.theta) * dt
            newTheta = fl.theta + vTheta * dt
            newCapState.append((newX, newY, newTheta, isAvoiding))

        # ================= FANBOIDS (synchronous: decide, then move) =================
        newFanState = []
        for k, fl in enumerate(fleets):
            if fl.nFan == 0:
                newFanState.append(None)
                continue

            mine = (fanFleet == k)              # mask: which of the flattened fanboids are mine
            xb, yb = fl.fanX, fl.fanY
            vxb, vyb = fl.fanVx, fl.fanVy

            vxDes = np.empty(fl.nFan)
            vyDes = np.empty(fl.nFan)

            # Same-fleet neighbors get the full boid treatment (+ their captain)
            sameX = np.append(allFanX[mine], fl.x)
            sameY = np.append(allFanY[mine], fl.y)
            sameVx = np.append(allFanVx[mine], v * np.cos(fl.theta))
            sameVy = np.append(allFanVy[mine], v * np.sin(fl.theta))

            # EVERYBODY (all fleets + all captains) counts for separation only
            everyX = np.concatenate([allFanX, capX])
            everyY = np.concatenate([allFanY, capY])

            for i in range(fl.nFan):
                # Cohesion + alignment: my own fleet only
                (_, _, fA_x, fA_y, fC_x, fC_y) = boidsRules(
                    xb[i], yb[i], vxb[i], vyb[i],
                    sameX, sameY, sameVx, sameVy, opts)

                # Separation: cross-fleet, dodge anyone too close
                sepX, sepY, _, _ = _separationFrom(xb[i], yb[i], everyX, everyY, opts)

                # Leader pull: stick with MY captain
                leadX = (fl.x - xb[i]) * opts.fanLeaderFactor
                leadY = (fl.y - yb[i]) * opts.fanLeaderFactor

                kx = sepX + fA_x + fC_x + leadX
                ky = sepY + fA_y + fC_y + leadY

                vxDes[i] = vxb[i] + kx * dt
                vyDes[i] = vyb[i] + ky * dt

            # Wall turnback kicks
            vxDes[xb <= arena.xSafeMin] += opts.TF * dt
            vxDes[xb >= arena.xSafeMax] -= opts.TF * dt
            vyDes[yb <= arena.ySafeMin] += opts.TF * dt
            vyDes[yb >= arena.ySafeMax] -= opts.TF * dt

            vxNew, vyNew, _ = ackermannClamp(vxb, vyb, vxDes, vyDes, opts)
            newFanState.append((xb + vxNew * dt, yb + vyNew * dt, vxNew, vyNew))

        # ================= COMMIT the new state and log =================
        for k, fl in enumerate(fleets):
            fl.x, fl.y, fl.theta, isAvoiding = newCapState[k]
            fl.log['x'].append(fl.x)
            fl.log['y'].append(fl.y)
            fl.log['theta'].append(fl.theta)
            fl.log['avoiding'].append(isAvoiding)

            if newFanState[k] is not None:
                fl.fanX, fl.fanY, fl.fanVx, fl.fanVy = newFanState[k]
                fl.log['fanX'].append(fl.fanX.copy())
                fl.log['fanY'].append(fl.fanY.copy())
                fl.log['fanVx'].append(fl.fanVx.copy())
                fl.log['fanVy'].append(fl.fanVy.copy())

        t += dt
        if all(fl.done for fl in fleets):       # everybody home? stop early
            break

    # Convert logs to arrays
    for fl in fleets:
        fl.log = {key: np.array(val) for key, val in fl.log.items()}

    return fleets


# ---------------- Metrics for the multi-fleet case ----------------

def captainSeparation(fleets):
    """Distance between every pair of captains at every timestep.

    Returns (minDist, minDistTimeIdx, distSeries) for the closest pair.
    """
    n = len(fleets)
    T = min(len(fl.log['x']) for fl in fleets)      # trim to the shortest log

    best = np.inf
    bestIdx = -1
    bestSeries = None
    for i in range(n):
        for j in range(i + 1, n):
            dx = fleets[i].log['x'][:T] - fleets[j].log['x'][:T]
            dy = fleets[i].log['y'][:T] - fleets[j].log['y'][:T]
            d = np.hypot(dx, dy)
            if d.min() < best:
                best = float(d.min())
                bestIdx = int(np.argmin(d))
                bestSeries = d
    return best, bestIdx, bestSeries


def pathDeviation(fleet):
    """How far the captain strayed from his OWN reference path, over time."""
    px = fleet.log['x']
    py = fleet.log['y']
    ref = fleet.data.refPath[:, :2]
    dx = px[:, None] - ref[None, :, 0]
    dy = py[:, None] - ref[None, :, 1]
    return np.hypot(dx, dy).min(axis=1)