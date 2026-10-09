"""The main simulation loop."""
import numpy as np
from .dynamics import (boidsRules, fanLeaderRule, falconSteering,
                       ackermannClamp, WaypointTracker,
                       initUniform, initClumped)


def runSimulation(data, arena, opts, nDistract=0, nFan=0, seed=None, slots=None):
    """Simulate Captain Falcon + boids on the given dataset/arena.

    Returns a dict of logged arrays, truncated to the CPA frame at the goal.

    slots: optional (along, lat) arrays of length nFan, each fanboid's target
    in the captain's body frame (along = forward, lat = left). None keeps the
    original behaviour: every fanboid is pulled toward the captain himself.
    """
    if seed is None:
        seed = opts.seed
    np.random.seed(seed)

    wayPts = data.wayPts
    tracker = WaypointTracker(opts)  # fresh CPA memory every run

    xDis, yDis, vxDis, vyDis = initUniform(nDistract, arena, opts)
    if nFan > 0:
        xFan, yFan, vxFan, vyFan = initClumped(nFan, arena, opts)
    else:
        empty = np.array([])
        xFan, yFan, vxFan, vyFan = empty, empty, empty, empty

    xi = float(data.qi[0])
    yi = float(data.qi[1])
    thetai = float(data.qi[2])

    idP = 1
    nP = wayPts.shape[0]
    t  = 0.0
    dt = opts.dt
    v  = opts.v
    tMax = opts.tMax(data.refLen)

    log = {'t': [], 'x': [], 'y': [], 'theta': [],
           'boidX': [], 'boidY': [],
           'fanX': [], 'fanY': [], 'fanVx': [], 'fanVy': [], 'fanOmega': []}

    fanX0 = xFan.copy()
    fanY0 = yFan.copy()
    reachedGoal = False
    goalMin = np.inf
    cpaIdx = -1

    while t <= tMax:

        # ================= CAPTAIN =================
        idP = tracker.advance(xi, yi, idP, wayPts)
        xLeader = wayPts[idP, 0]
        yLeader = wayPts[idP, 1]
        distLeader = np.hypot(xLeader - xi, yLeader - yi)

        # Stop at closest approach to the final waypoint
        if idP == nP - 1:
            if distLeader < goalMin:
                goalMin = distLeader
                cpaIdx = len(log['x']) - 1  # marks closest point
            passedClosestPoint = (goalMin < opts.passWindow and
                                  distLeader > goalMin + opts.hysteresis)
            if passedClosestPoint:
                break

        vxi = v * np.cos(thetai)
        vyi = v * np.sin(thetai)

        # Pull toward current waypoint
        vxLeader = (xLeader - xi) * opts.leaderFactor
        vyLeader = (yLeader - yi) * opts.leaderFactor

        if opts.falconSeesFanboids and nFan > 0:
            xNb  = np.concatenate([xDis,  xFan])
            yNb  = np.concatenate([yDis,  yFan])
            vxNb = np.concatenate([vxDis, vxFan])
            vyNb = np.concatenate([vyDis, vyFan])
        else:
            xNb, yNb, vxNb, vyNb = xDis, yDis, vxDis, vyDis

        if xNb.size > 0:
            (fS_x, fS_y, fA_x, fA_y, fC_x, fC_y) = boidsRules(
                xi, yi, vxi, vyi, xNb, yNb, vxNb, vyNb, opts)
        else:
            fS_x = fS_y = fA_x = fA_y = fC_x = fC_y = 0

        vxTotal = vxLeader + opts.falconBoidsWeight * (fS_x + fA_x + fC_x)
        vyTotal = vyLeader + opts.falconBoidsWeight * (fS_y + fA_y + fC_y)

        vThetai = falconSteering(thetai, vxTotal, vyTotal, opts)
        t += dt
        xi += vxi * dt
        yi += vyi * dt
        thetai += vThetai * dt

        log['t'].append(t)
        log['x'].append(xi)
        log['y'].append(yi)
        log['theta'].append(thetai)

        # ================= SMALL BOIDS (synchronous two-pass) =================
        nSmall = nDistract + nFan
        if nSmall > 0:
            # Combine into master list for the snapshot
            xAll  = np.concatenate([xDis,  xFan])
            yAll  = np.concatenate([yDis,  yFan])
            vxAll = np.concatenate([vxDis, vxFan])
            vyAll = np.concatenate([vyDis, vyFan])

            # Fanboids see captain, distractors do not
            xAllF  = np.append(xAll,  xi)
            yAllF  = np.append(yAll,  yi)
            vxAllF = np.append(vxAll, v * np.cos(thetai))
            vyAllF = np.append(vyAll, v * np.sin(thetai))

            # ----- Pass 1: desired velocities (no movement yet) -----
            vxDes = np.empty(nSmall)
            vyDes = np.empty(nSmall)
            
            # Pre-calculate Captain's current velocity for the loop
            vxCap = v * np.cos(thetai)
            vyCap = v * np.sin(thetai)

            for i in range(nSmall):
                isFan = i >= nDistract

                if isFan:
                    neighborhood = (xAllF, yAllF, vxAllF, vyAllF)
                else:
                    neighborhood = (xAll, yAll, vxAll, vyAll)

                (fS_x, fS_y,
                 fA_x, fA_y,
                 fC_x, fC_y) = boidsRules(xAll[i], yAll[i], vxAll[i], vyAll[i],
                                          *neighborhood, opts)

                kx = fS_x + fA_x + fC_x
                ky = fS_y + fA_y + fC_y

                if isFan:
                    # Target: the captain himself, or this fanboid's own slot
                    if slots is None:
                        tx, ty = xi, yi
                    else:
                        j = i - nDistract
                        c, sn = np.cos(thetai), np.sin(thetai)
                        tx = xi + c * slots[0][j] - sn * slots[1][j]
                        ty = yi + sn * slots[0][j] + c * slots[1][j]
                    # Pass the positions AND the velocities into the new rule
                    kx4, ky4 = fanLeaderRule(
                        xAll[i], yAll[i], vxAll[i], vyAll[i], 
                        tx, ty, vxCap, vyCap, opts
                    )
                    kx += kx4
                    ky += ky4

                vxDes[i] = vxAll[i] + kx * dt
                vyDes[i] = vyAll[i] + ky * dt

            # Wall turnback kicks
            vxDes[xAll <= arena.xSafeMin] += opts.TF * dt
            vxDes[xAll >= arena.xSafeMax] -= opts.TF * dt
            vyDes[yAll <= arena.ySafeMin] += opts.TF * dt
            vyDes[yAll >= arena.ySafeMax] -= opts.TF * dt

            # ----- Pass 2: desire -> motion through Ackermann constraint -----
            vxNew, vyNew, omega = ackermannClamp(vxAll, vyAll, vxDes, vyDes, opts)
            xAll = xAll + vxNew * dt
            yAll = yAll + vyNew * dt

            # Split back into flocks
            xDis,  xFan  = xAll[:nDistract],  xAll[nDistract:]
            yDis,  yFan  = yAll[:nDistract],  yAll[nDistract:]
            vxDis, vxFan = vxNew[:nDistract], vxNew[nDistract:]
            vyDis, vyFan = vyNew[:nDistract], vyNew[nDistract:]

            if nDistract > 0:
                log['boidX'].append(xDis.copy())
                log['boidY'].append(yDis.copy())
            if nFan > 0:
                log['fanX'].append(xFan.copy())
                log['fanY'].append(yFan.copy())
                log['fanVx'].append(vxFan.copy())
                log['fanVy'].append(vyFan.copy())
                log['fanOmega'].append(omega[nDistract:].copy())

    # Rewind logs to exact CPA frame
    if cpaIdx >= 0:
        for key in log:
            log[key] = log[key][:cpaIdx + 1]
        reachedGoal = goalMin < opts.collectRadius

    out = {key: np.array(val) for key, val in log.items()}
    out['goalMiss'] = float(goalMin)
    out['fanX0'] = fanX0
    out['fanY0'] = fanY0
    out['reachedGoal'] = reachedGoal
    return out
