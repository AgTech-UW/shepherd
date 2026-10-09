"""Formations the agent can switch between, and the geometry it needs to
choose one.

A formation is a set of slots (along, lateral) in the leader's body frame:
along < 0 is behind, lateral > 0 is left. The env feeds the slots to the
paper's flockStep, which pulls each boid toward its own slot instead of
toward the leader. The boids are still boids: separation, cohesion and
alignment are unchanged, so a formation whose slots sit inside each other's
protected range fights itself (the paper's spacing condition, delta > P).

Spec (what an agent sends), all keys optional except shape:
  {"shape": "none"|"vee"|"echelon"|"column"|"line"|"custom",
   "spacing": metres between neighbouring slots (default P + 2),
   "standoff": distance of the nearest slot behind the leader (default P + 2),
   "openingDeg": vee/echelon half-angle from the centreline, 0 = single file,
                 90 = abreast (default 45, the paper's operating point),
   "ranks": for "column", how many abreast per rank (default 1),
   "side": "left"|"right" for "echelon",
   "slots": [[along, lat], ...] for "custom",
   "blend": seconds to morph from where the boids are to the new slots (default 3),
   "frame": "body" (default, the paper's: slots rigid behind the leader's
            current heading) | "track" (slot at `along` metres back ALONG
            THE PATH THE LEADER ACTUALLY DROVE, `lat` to its side, so a
            column follows him round a corner instead of swinging across it)}

"none" is the leader pull: every boid aims at the leader himself.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

SHAPES = ('none', 'vee', 'echelon', 'column', 'line', 'custom')


def slotsFor(spec, n, P):
    """Spec -> (along, lat) arrays of length n, or None for 'none'."""
    shape = str(spec.get('shape', 'none')).lower()
    if shape == 'none' or n == 0:
        return None
    d = float(np.clip(spec.get('spacing', P + 2.0), 1.0, 60.0))
    r0 = float(np.clip(spec.get('standoff', P + 2.0), 1.0, 60.0))
    op = np.deg2rad(float(np.clip(spec.get('openingDeg', 45.0), 0.0, 90.0)))
    along, lat = [], []
    if shape == 'vee':
        for i in range(n):
            arm = r0 + (i // 2) * d
            side = 1.0 if i % 2 == 0 else -1.0
            along.append(-arm * np.cos(op))
            lat.append(side * arm * np.sin(op))
    elif shape == 'echelon':
        sgn = -1.0 if str(spec.get('side', 'left')).lower() == 'right' else 1.0
        for i in range(n):
            arm = r0 + i * d
            along.append(-arm * np.cos(op))
            lat.append(sgn * arm * np.sin(op))
    elif shape in ('column', 'line'):
        k = n if shape == 'line' else int(np.clip(spec.get('ranks', 1), 1, n))
        for i in range(n):
            row, col = divmod(i, k)
            m = min(k, n - row * k)                  # boids in this rank
            along.append(-(r0 + row * d))
            lat.append((col - (m - 1) / 2.0) * d)
    elif shape == 'custom':
        pts = [p for p in spec.get('slots', []) if len(p) == 2][:n]
        for a, l in pts:
            along.append(float(np.clip(a, -80, 80)))
            lat.append(float(np.clip(l, -80, 80)))
        # not enough slots: the rest trail in single file
        tail = min(along) if along else -r0
        for i in range(n - len(pts)):
            along.append(tail - (i + 1) * d)
            lat.append(0.0)
    else:
        raise ValueError('unknown formation shape %r' % shape)
    return np.array(along), np.array(lat)


def geometry(along, lat, P):
    """Numbers that matter for choosing a formation."""
    if along is None:
        return dict(leftHalfWidth=None, rightHalfWidth=None, length=None,
                    minSlotGap=None, satisfiesSpacing=None)
    pa = np.concatenate([[0.0], along])            # include the leader at origin
    pl = np.concatenate([[0.0], lat])
    D = np.hypot(pa[:, None] - pa[None, :], pl[:, None] - pl[None, :])
    gap = float(D[np.triu_indices(len(pa), 1)].min())
    return dict(leftHalfWidth=round(float(max(0.0, lat.max())), 1),
                rightHalfWidth=round(float(max(0.0, -lat.min())), 1),
                length=round(float(-along.min()), 1),
                minSlotGap=round(gap, 1), satisfiesSpacing=bool(gap > P))


def assignIndex(relA, relL, along, lat):
    """Slot index for each boid, minimum total squared travel."""
    C = (relA[:, None] - along[None, :]) ** 2 + (relL[:, None] - lat[None, :]) ** 2
    rows, cols = linear_sum_assignment(C)
    perm = np.empty(len(relA), dtype=int)
    perm[rows] = cols
    return perm


def assign(relA, relL, along, lat):
    """Which boid takes which slot: minimum total squared travel (Hungarian).
    Index-order assignment makes boids cross through each other on every
    reshuffle, which is a collision generator of our own making."""
    C = (relA[:, None] - along[None, :]) ** 2 + (relL[:, None] - lat[None, :]) ** 2
    rows, cols = linear_sum_assignment(C)
    perm = np.empty(len(relA), dtype=int)
    perm[rows] = cols
    return along[perm], lat[perm]


def corridorAhead(env, stations=(0, 10, 20, 30, 45, 60), rayMax=40.0):
    """Free half-width left and right of the leader's intended path, sampled
    at arc distances ahead along (leader -> remaining route -> goal).
    Distances are to the nearest obstacle surface, minus body radius;
    rayMax means open."""
    pts = np.array([(env.x, env.y)] + list(env.remainingRoute()) + [env.course.goal[:2]], float)
    seg = np.diff(pts, axis=0)
    segL = np.hypot(seg[:, 0], seg[:, 1])
    keep = segL > 1e-6
    pts = np.vstack([pts[:1], pts[1:][keep]])
    seg, segL = seg[keep], segL[keep]
    out = []
    ob = env.course.obstacles
    if len(segL) == 0:
        return out
    cum = np.concatenate([[0.0], np.cumsum(segL)])
    for s in stations:
        if s > cum[-1]:
            break
        k = min(np.searchsorted(cum, s, side='right') - 1, len(segL) - 1)
        u = seg[k] / segL[k]
        p = pts[k] + u * (s - cum[k])
        nL = np.array([-u[1], u[0]])
        row = dict(ahead=s)
        for name, n in (('left', nL), ('right', -nL)):
            free = rayMax
            if len(ob):
                c = ob[:, :2] - p
                r = ob[:, 2] + env.opts.bodyRadius
                b = c @ n
                disc = b * b - (np.sum(c * c, 1) - r * r)
                t = b - np.sqrt(np.maximum(disc, 0.0))
                hit = (disc >= 0) & (t > -r)
                if np.any(hit):
                    free = float(np.clip(t[hit].min(), 0.0, rayMax))
            row[name] = round(free, 1)
        out.append(row)
    return out


def trackToBody(env, along, lat):
    """Track-frame slots -> equivalent body-frame offsets for flockStep.
    env.trail holds the leader's breadcrumbs (x, y, heading), newest last,
    spaced env.trailStep metres apart."""
    tr = env.trail
    n = len(tr)
    # the trail only grows every trailStep metres, so cache its array form
    cache = getattr(env, '_trailCache', None)
    if cache is None or cache[0] != (n, tr[-1]):
        cache = ((n, tr[-1]), np.asarray(tr))
        env._trailCache = cache
    k = np.clip(np.round(-along / env.trailStep).astype(int), 0, n - 1)
    pts = cache[1][n - 1 - k]
    bx = pts[:, 0] - lat * np.sin(pts[:, 2])
    by = pts[:, 1] + lat * np.cos(pts[:, 2])
    # slots ahead of the leader (along > 0) stay in the body frame
    ahead = along > 0
    c, s = np.cos(env.th), np.sin(env.th)
    dx, dy = bx - env.x, by - env.y
    A = np.where(ahead, along, dx * c + dy * s)
    L = np.where(ahead, lat, -dx * s + dy * c)
    return A, L
