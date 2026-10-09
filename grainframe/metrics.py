"""Error metrics and reports."""
import numpy as np
from .utils import wrapToPi


def crossTrackError(px, py, pathXY):
    dxToPath = px[:, None] - pathXY[None, :, 0]
    dyToPath = py[:, None] - pathXY[None, :, 1]
    d = np.hypot(dxToPath, dyToPath)
    return d.min(axis=1)


# ---------------- Fanboid metrics (errors vs the Captain) ----------------

def fanFinalErrors(run):
    dx = run['fanX'][-1] - run['x'][-1]
    dy = run['fanY'][-1] - run['y'][-1]
    d = np.hypot(dx, dy)
    fanTheta = np.arctan2(run['fanVy'][-1], run['fanVx'][-1])
    dth = wrapToPi(fanTheta - run['theta'][-1])
    return dx, dy, d, dth


def fanChi2(run, opts):
    # Split position and heading errors
    dx, dy, _, dth = fanFinalErrors(run)
    chi2_pos = float(np.sum(dx**2 / opts.sigmaPos**2 + dy**2 / opts.sigmaPos**2))
    chi2_ang = float(np.sum(dth**2 / opts.sigmaTh**2))
    return chi2_pos, chi2_ang


# ---------------- Captain metrics (errors vs the GOAL pose) ----------------

def falconFinalErrors(run, data):
    """Captain Falcon's end pose vs the goal pose qf."""
    dx  = float(run['x'][-1] - data.qf[0])
    dy  = float(run['y'][-1] - data.qf[1])
    dth = float(wrapToPi(run['theta'][-1] - data.qf[2]))
    return dx, dy, np.hypot(dx, dy), dth


def falconChi2(run, data, opts):
    """chi^2 of Captain Falcon's end pose vs goal (per Adam's request)."""
    dx, dy, _, dth = falconFinalErrors(run, data)
    chi2_pos = (dx**2 + dy**2) / opts.sigmaPos**2
    chi2_ang = dth**2 / opts.sigmaTh**2
    return float(chi2_pos), float(chi2_ang)


# ---------------- Reports ----------------

def flockFinalReport(run, opts):
    # 1. Smoothness: how hard they were cranking the wheel
    omega = run['fanOmega']  # heading rate, not desired angle
    rms_omega = np.sqrt(np.mean(omega**2))
    max_omega = np.max(np.abs(omega))

    x = run['fanX'][-1]
    y = run['fanY'][-1]
    vx = run['fanVx'][-1]
    vy = run['fanVy'][-1]

    # 2. Cohesion: average distance from their own center of mass
    cm_x, cm_y = np.mean(x), np.mean(y)
    cohesion_dist = np.mean(np.hypot(x - cm_x, y - cm_y))

    # 3. Alignment: standard deviation of headings
    headings = np.arctan2(vy, vx)
    alignment_std = np.std(headings)

    # 4. Separation: minimum distance between any two boids
    n = len(x)
    min_sep = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            min_sep = min(min_sep, d)

    print('\n--- Fanboid Final-State Report ---')
    print('Smoothness (Omega) : RMS = %.3f rad/s, Max = %.3f rad/s' % (rms_omega, max_omega))
    print('Cohesion (Spread)  : Mean distance from center of mass = %.3f' % cohesion_dist)
    print('Alignment (Heading): Std dev of headings = %.3f deg' % np.degrees(alignment_std))
    print('Separation (Crash) : Minimum distance between boids = %.3f (Protected Range = %.1f)'
          % (min_sep, opts.PR))


# ---------------- Collision counting ----------------

def _smallAgentPositions(run):
    """Stack fanboid and distractor positions into one (T, nAgents) pair.
    Works whether the run had fanboids, distractors, or both.
    """
    parts_x = []                              # we'll collect however many groups exist
    parts_y = []
    if run['fanX'].size > 0:                  # did this run have fanboids? add 'em
        parts_x.append(run['fanX'])
        parts_y.append(run['fanY'])
    if run['boidX'].size > 0:                 # did it have distractors? add those too
        parts_x.append(run['boidX'])
        parts_y.append(run['boidY'])
    if not parts_x:                           # neither? hand back empty arrays, don't crash
        return np.zeros((0, 0)), np.zeros((0, 0))
    return np.hstack(parts_x), np.hstack(parts_y)   # glue groups side by side -> one big (T, nAgents) block


def count_collisions(run, opts, radius=None, includeCaptain=True):
    """Count collisions over the ENTIRE run (every logged timestep).
    Returns a dict with:
      - pairSteps    : number of (pair, timestep) boid-boid overlaps
      - uniquePairs  : how many distinct boid pairs EVER collided
      - capSteps     : number of timesteps any boid overlapped the captain
      - capBoids     : how many distinct boids EVER hit the captain
      - minSepPair   : closest any two boids got over the whole run
      - minSepCap    : closest any boid got to the captain over the whole run
    """
    if radius is None:                        # caller didn't say? fall back to the config default
        radius = opts.collisionRadius

    X, Y = _smallAgentPositions(run)          # every small agent's whole trajectory
    T = X.shape[0]                            # number of timesteps (rows)
    n = X.shape[1] if X.ndim > 1 else 0       # number of agents (cols); 0 if there were none

    pairSteps = 0                             # running tally: boid-boid overlaps, counted every frame
    capSteps = 0                              # running tally: boid-captain overlaps, every frame
    minSepPair = np.inf                       # track the closest two boids EVER got (start "infinitely far")
    minSepCap = np.inf                        # closest any boid got to Cap
    everCollidedPairs = set()                 # a set = no duplicates, so this counts DISTINCT pairs
    everHitCaptain = set()                    # distinct boids that ever touched Cap

    for t in range(T):                        # walk forward through time, frame by frame
        xs = X[t]                             # all agents' x at this instant
        ys = Y[t]

        # boid <-> boid
        for i in range(n):                    # for each agent...
            for j in range(i + 1, n):         # ...check every OTHER agent once (j starts past i, no repeats)
                d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])   # straight-line distance between them
                minSepPair = min(minSepPair, d)              # is this the new record-closest? keep it
                if d < radius:                # closer than a car-width? that's a collision
                    pairSteps += 1            # bump the "how long were they overlapping" counter
                    everCollidedPairs.add((i, j))   # remember this pair (set ignores if already logged)

        # boid <-> captain
        if includeCaptain:                    # only if we care about hitting Cap (we do here)
            cx = run['x'][t]                  # Cap's x at this instant
            cy = run['y'][t]
            for i in range(n):                # check every boid against Cap
                d = np.hypot(xs[i] - cx, ys[i] - cy)
                minSepCap = min(minSepCap, d)              # closest anyone's gotten to Cap so far
                if d < radius:                # overlapping Cap?
                    capSteps += 1             # count the frame
                    everHitCaptain.add(i)     # remember which boid it was

    return {
        'pairSteps':   pairSteps,             # total overlap-frames (long clinch = big number)
        'uniquePairs': len(everCollidedPairs),   # how many DIFFERENT pairs crashed (the honest headline)
        'capSteps':    capSteps,
        'capBoids':    len(everHitCaptain),   # how many different boids clipped Cap
        'minSepPair':  float(minSepPair),     # the single closest boid-boid moment
        'minSepCap':   float(minSepCap),      # the single closest boid-Cap moment
    }


def collisions_vs_radius(run, opts, radii):
    """Re-score ONE run at several collision radii (no re-simulation needed).

    "The smaller that radius, the more close the approach" -- this shows
    exactly how the collision counts shrink as the vehicles get smaller.
    Returns lists (uniquePairs, capBoids) matching the given radii.
    """
    pairCounts = []                           # one collision-count per radius we test
    capCounts = []
    for r in radii:                           # try each vehicle size in turn
        c = count_collisions(run, opts, radius=r)   # SAME trajectories, just a different yardstick
        pairCounts.append(c['uniquePairs'])   # log how many pairs count as crashed at this size
        capCounts.append(c['capBoids'])       # and how many hit Cap at this size
    return pairCounts, capCounts              # caller plots these against the radii