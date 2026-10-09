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
    sp = np.clip(env.speed - kAlong * eAl, o.minSpeed, o.maxSpeed)
    return sp * np.cos(chi), sp * np.sin(chi), eLat, eAl


# The three follower models compared in compare_followers.py, as
# (SimOptions overrides, env keyword arguments). Separation, alignment and
# cohesion are the paper's rules in all three.
FOLLOWER_MODELS = {
    'paper':  (dict(), dict()),
    'damped': (dict(fanLeaderFactor=2.0, SEP_PROFILE='norm', VR=20.0, CF=0.0, AF=4.0), dict()),
    'vfield': (dict(SEP_PROFILE='norm', minSpeed=0.3), dict(followerModel='vfield', reassignEvery=2.0)),
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
}


def followerSetup(name):
    """(SimOptions, env kwargs) for a follower model."""
    from grainframe import SimOptions
    over, kw = FOLLOWER_MODELS[name]
    o = SimOptions()
    for k, v in over.items():
        setattr(o, k, v)
    return o, dict(kw)
