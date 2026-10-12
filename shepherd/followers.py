"""Follower models: how a follower gets to and holds its slot.

'boids' (the paper): the slot term is a spring toward a moving point,
    v += fanLeaderFactor * (slot - p) * dt
For a vehicle with a minimum turning radius and a minimum speed this is the
waypoint law of guidance.py applied to every follower: it nulls position
error and ignores heading, so a displaced follower overshoots and loops.

'vfield': the slot term is a Reynolds (1999) steering behaviour,
    v += kSteer * (vDesired - v) * dt
with vDesired from the paper's vector field (Nelson et al. 2007) along a
line through the slot:
  - the line runs through the follower's slot in the direction the slot is
    moving: the leader's heading for body-frame slots, the heading of the
    leader's breadcrumb trail at that point for track-frame slots
  - cross-track error sets the desired heading, exactly as for the leader
  - along-track error sets the desired speed: slow down when ahead of the
    slot, speed up when behind, around the leader's speed
Separation, alignment and cohesion are unchanged (flockStep with the slot
term switched off), so the separation rule still acts as the safety layer.
Only the slot-holding rule changes.

'reflex': 'vfield' plus an obstacle reflex, for the leader too. Machines still
do not plan around obstacles, but each one checks whether the heading its
vector field asks for runs into a free-standing obstacle (a rock or a pole) within 10 m,
and if so bends it just enough to clear it (and slows in proportion).
Obstacles it would miss anyway are ignored, and so are fences and tree
lines: a reflex that will not cross a fence traps a follower on the wrong
side of it whenever the formation puts its slot there, so boundaries stay
the planner's job. Once the obstacle is behind it the follower is back on the vector
field, which returns it to its slot line without overshoot. There is no mode
switch. A subsumption-style reflex under the planner (Brooks 1986).
"""
import numpy as np

from . import formation as F


def slotReference(env):
    """World position and direction of travel of every follower's slot."""
    sl = env.currentSlots()
    c, s = np.cos(env.th), np.sin(env.th)
    sx = env.x + sl[0] * c - sl[1] * s
    sy = env.y + sl[0] * s + sl[1] * c
    head = np.full(env.nFan, env.th)
    if env.formation.get('frame', 'body') == 'track' and env.slotsTo is not None:
        along = env.slotsTo[0]
        tr = env.trail
        n = len(tr)
        F.trackToBody(env, *env.slotsTo)            # refreshes the trail cache
        arr = env._trailCache[1]
        k = np.clip(np.round(-along / env.trailStep).astype(int), 0, n - 1)
        head = np.where(along > 0, env.th, arr[n - 1 - k, 2])
    return sx, sy, head


def vfieldDesired(env, kCross=0.5, chiInfDeg=60.0, kAlong=0.4):
    """Desired velocity of every follower from the vector field on its slot line."""
    sx, sy, h = slotReference(env)
    ux, uy = np.cos(h), np.sin(h)
    ex, ey = env.fanX - sx, env.fanY - sy
    eLat = -ex * uy + ey * ux                        # + = left of the slot line
    eAl = ex * ux + ey * uy                          # + = ahead of the slot
    chi = h - np.deg2rad(chiInfDeg) * (2.0 / np.pi) * np.arctan(kCross * eLat)
    o = env.opts
    hi = o.maxSpeed if getattr(env, 'catchUp', None) is None else min(o.maxSpeed, env.speed + env.catchUp)
    sp = np.clip(env.speed - kAlong * eAl, o.minSpeed, max(hi, o.minSpeed))
    return sp * np.cos(chi), sp * np.sin(chi), eLat, eAl


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _obstacleTree(env):
    """The env's spatial index over obstacle centres (built here if the env
    has not built it yet, with the same reach the env uses)."""
    tree = getattr(env.course, '_obsTree', None)
    if tree is None:
        from scipy.spatial import cKDTree
        ob = env.course.obstacles
        tree = env.course._obsTree = cKDTree(ob[:, :2])
        env.course._obsReach = float(ob[:, 2].max()) + env.opts.bodyRadius + 10.0
    return tree


def _isolated(env, gap=3.0):
    """Obstacles standing on their own (rocks, poles, posts): no other
    obstacle centre within `gap` metres. Fences and tree lines are rows of
    closely spaced obstacles and are left to the planner: a reflex that will
    not let a follower cross a fence traps it on the wrong side whenever the
    formation puts its slot across that fence."""
    c = env.course
    cache = getattr(c, '_reflexObs', None)
    if cache is None:
        from scipy.spatial import cKDTree
        ob = c.obstacles
        if len(ob) < 2:
            iso = ob
        else:
            d, _ = cKDTree(ob[:, :2]).query(ob[:, :2], k=2)
            iso = ob[d[:, 1] > gap]
        cache = c._reflexObs = (iso, cKDTree(iso[:, :2]) if len(iso) else None)
    return cache


def reflexHeading(env, P, h, sp, look=None, margin=None, k=4):
    """The reflex for any machines at positions P (n, 2) that want heading h
    (n,) at speed sp (n,): returns the corrected (h, sp, w).

    Only obstacles standing on their own count (_isolated: rocks, poles).
    A machine only reacts to one that lies on the path it wants to take:
    each obstacle within `look` metres blocks a cone of headings around its
    bearing, half-angle asin((r + body + margin) / distance). If the wanted
    heading falls inside a cone, it is moved to the nearer edge of that cone
    (nearest first). The correction fades in between `look` and look / 2,
    and the speed drops with the size of the correction. Obstacles it would
    miss anyway are ignored. w in [0, 1] is the size of the correction
    (1 = 90 degrees). look and margin default to env.reflexLook (10 m) and
    env.reflexMargin (1.5 m)."""
    look = getattr(env, 'reflexLook', 10.0) if look is None else look
    margin = getattr(env, 'reflexMargin', 1.5) if margin is None else margin
    ob, tree = _isolated(env)
    n = len(P)
    if tree is None or n == 0:
        return h, sp, np.zeros(n)
    br = env.opts.bodyRadius
    D, idx = tree.query(P, k=min(k, len(ob)), distance_upper_bound=look + float(ob[:, 2].max()))
    D, idx = np.atleast_2d(D), np.atleast_2d(idx)
    h0 = np.asarray(h, float)
    h = h0.copy()
    for c in range(D.shape[1]):                       # nearest first
        ok = (idx[:, c] < len(ob)) & (D[:, c] < look)
        if not ok.any():
            break
        ii = np.minimum(idx[:, c], len(ob) - 1)
        dist = np.maximum(D[:, c], 1e-6)
        b = np.arctan2(ob[ii, 1] - P[:, 1], ob[ii, 0] - P[:, 0])
        half = np.arcsin(np.clip((ob[ii, 2] + br + margin) / dist, 0.0, 1.0))
        d = _wrap(h - b)
        inside = ok & (np.abs(d) < half)
        side = np.where(np.abs(d) > 1e-3, np.sign(d), 1.0)
        fade = np.clip((look - dist) / (look / 2), 0.0, 1.0)
        h = np.where(inside, h + fade * _wrap(b + side * half - h), h)
    w = np.clip(np.abs(_wrap(h - h0)) / (np.pi / 2), 0.0, 1.0)
    return h, np.asarray(sp, float) * (1.0 - 0.5 * w), w


def reflexDesired(env, vfx, vfy, **kw):
    """The reflex applied to the followers' desired velocities (vfx, vfy).
    Returns (vfx, vfy, w)."""
    P = np.column_stack([env.fanX, env.fanY])
    h, sp, w = reflexHeading(env, P, np.arctan2(vfy, vfx), np.hypot(vfx, vfy), **kw)
    return sp * np.cos(h), sp * np.sin(h), w


# The follower models compared in compare_followers.py, as
# (SimOptions overrides, env keyword arguments). Separation, alignment and
# cohesion are the paper's rules in all three.
FOLLOWER_MODELS = {
    'paper':  (dict(), dict()),
    'damped': (dict(fanLeaderFactor=2.0, SEP_PROFILE='norm', VR=20.0, CF=0.0, AF=4.0), dict()),
    'vfield': (dict(SEP_PROFILE='norm', minSpeed=0.3), dict(followerModel='vfield', reassignEvery=2.0)),
    'reflex': (dict(SEP_PROFILE='norm', minSpeed=0.3), dict(followerModel='reflex', reassignEvery=2.0)),
    # reflex + "pearls on a string" (A. Rogers): the leader waits for a follower that has fallen behind
    # (say, crawling through a wet spot) and followers make up ground gently, so the crew stays strung
    # together and speeds back up as a team
    'pearls': (dict(SEP_PROFILE='norm', minSpeed=0.3),
               dict(followerModel='reflex', reassignEvery=2.0, leaderWait=(8.0, 25.0, 0.3), catchUp=0.5)),
}

FOLLOWER_TEXT = {
    'paper': 'Followers cannot see obstacles; each steers to hold its slot in the formation. When disturbed '
             '(turns, formation changes, crowding) they swing in loops of several metres around their slot '
             'before settling, so leave room.',
    'damped': 'Followers cannot see obstacles; each steers to hold its slot in the formation and settles onto '
              'it, but after a turn or a formation change they drift a few metres before settling, so leave room.',
    'vfield': 'Followers cannot see obstacles; each drives along a line through its slot, like autosteer, and '
              'speeds up or slows down to keep its place. They hold their lines well, but forming up after a '
              'formation change takes time (a long single file needs a minute or more), and in single file '
              'they cannot pass one another.',
    'reflex': 'Followers do not plan around obstacles; each drives along a line through its slot, like autosteer, '
              'and speeds up or slows down to keep its place. As a last resort, a follower about to run into a rock '
              'or a pole swerves and slows to miss it, then returns to its line; this reflex does not cover fences '
              'or trees, can fail, and leaves gaps in the work, so plan as if it were not there. Forming up after a '
              'formation change takes time (a long single file needs a minute or more), and in single file they '
              'cannot pass one another.',
}


def followerSetup(name):
    """(SimOptions, env kwargs) for a follower model."""
    from grainframe import SimOptions
    over, kw = FOLLOWER_MODELS[name]
    o = SimOptions()
    for k, v in over.items():
        setattr(o, k, v)
    return o, dict(kw)
