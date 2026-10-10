"""ShepherdEnv: one Dubins leader, a flock of plain boids, an obstacle course.

The leader is the only thing an agent controls. The fanboids are the paper's
followers unchanged: grainframe.flocking.flockStep with the leader pull
(slotAlong=None, i.e. every fanboid pulled toward the leader himself, the
configuration in which the paper measured 88% leader hits), then the
Ackermann clamp. They do not see obstacles unless boidsSeeObstacles=True.
The leader therefore has to shepherd them past obstacles, not just himself.

Time runs at the paper's dt = 0.01. An agent is consulted every
`decisionDt` sim-seconds (default 2 s), which is the only sane rate for an
LLM: it plans, the physics executes. Between consultations the env runs the
agent's plan, a list of segments:

  {"type": "arc",    "turn": u in [-1,1], "speed": s, "duration": T}
        constant curvature u/R at speed s for T seconds
  {"type": "goto",   "x":.., "y":.., "heading": rad, "speed": s}
        shortest Dubins path to that pose (obstacles ignored), open loop
  {"type": "follow", "waypoints": [[x,y],...], "speed": s}
        the paper's vector-field guidance along the polyline
  {"type": "seek",   "x":.., "y":.., "speed": s}
        bang-bang steer at a point, ends within 3 m
  {"type": "formation", "shape": .., "spacing": .., ...}
        instantaneous: re-shape the flock (see formation.py). Takes effect
        when the plan reaches it, so formation changes can be scheduled
        between motion segments.

A plan that runs out is followed by driving straight. An agent may return
None to keep running its current plan.

The episode ends at closest approach to the goal position once the leader is
within captureRadius (the same CPA rule the paper uses at its last
waypoint), or at tMax. Pose error is measured at that instant.
"""
import copy
import json
import numpy as np

from grainframe import SimOptions
from grainframe.flocking import flockStep
from grainframe.dynamics import ackermannClamp, falconSteering
from grainframe.guidance import guidanceVectorField
from grainframe.utils import wrapToPi
from .dubins import dubinsShortest
from . import formation as F
from .followers import vfieldDesired, reflexDesired, reflexHeading

CATS = ('fan-leader', 'fan-fan', 'fan-obstacle', 'leader-obstacle')


# ---------------------------------------------------------------------
#  Segment executors. Small classes, not generators, so the whole env can
#  be deep-copied for rollouts.
# ---------------------------------------------------------------------

class _Arc:
    def __init__(self, turn, speed, duration):
        self.turn, self.speed, self.left = float(turn), float(speed), float(duration)

    def control(self, env):
        if self.left <= 1e-12:
            return None
        self.left -= env.opts.dt
        return self.turn, self.speed


class _Goto:
    """Dubins pieces solved once from the pose at activation."""
    def __init__(self, x, y, heading, speed):
        self.goal = (float(x), float(y), float(heading))
        self.speed = float(speed)
        self.pieces = None

    def control(self, env):
        if self.pieces is None:
            _, pcs, _ = dubinsShortest((env.x, env.y, env.th), self.goal, env.opts.R)
            self.pieces = [[t, L] for t, L in pcs]
        while self.pieces and self.pieces[0][1] <= 1e-9:
            self.pieces.pop(0)
        if not self.pieces:
            return None
        ds = self.speed * env.opts.dt
        self.pieces[0][1] -= ds
        return self.pieces[0][0], self.speed


class _Follow:
    """Vector-field guidance with the paper's CPA waypoint handover."""
    def __init__(self, waypoints, speed, kCross=0.5, chiInfDeg=70.0, start=None):
        self.wp = np.asarray(waypoints, dtype=float).reshape(-1, 2)
        self.speed = float(speed)
        self.kCross, self.chiInf = kCross, np.deg2rad(chiInfDeg)
        self.idx = 0
        self.minD = np.inf
        # Start of the first line segment. Without it the line is drawn from
        # wherever the leader is when the segment starts, so every re-plan
        # moves the reference line and the leader wanders off the route.
        self.prev = None if start is None else np.asarray(start, dtype=float)

    def control(self, env):
        if self.idx >= len(self.wp):
            return None
        if self.prev is None:
            self.prev = np.array([env.x, env.y])
        tgt = self.wp[self.idx]
        d = np.hypot(tgt[0] - env.x, tgt[1] - env.y)
        self.minD = min(self.minD, d)
        # The paper's handover: closest approach within passWindow, then moving
        # away. If the leader misses the window (a re-plan or an offset line
        # near a corner), the vector field would follow this leg's line for
        # ever, so as a fallback also hand over once the leader crosses the
        # perpendicular to the leg at the waypoint. When the paper's rule
        # applies the fallback never fires, so earlier results are unchanged.
        cpa = self.minD < env.opts.passWindow and d > self.minD + 0.025
        seg = tgt - self.prev
        crossed = (self.minD >= env.opts.passWindow and d < 60.0 and float(seg @ seg) > 1e-9 and
                   float((env.x - tgt[0]) * seg[0] + (env.y - tgt[1]) * seg[1]) >= 0.0)
        if cpa or crossed:
            self.prev, self.idx, self.minD = tgt.copy(), self.idx + 1, np.inf
            if self.idx >= len(self.wp):
                return None
            tgt = self.wp[self.idx]
        out = guidanceVectorField(env.x, env.y, self.prev, tgt, 1.0,
                                  self.kCross, self.chiInf, 10.0)
        if out is None:
            out = (tgt[0] - env.x, tgt[1] - env.y)
        speed = self.speed
        if env.followerModel == 'reflex':                # the leader has the reflex too
            h, sp, w = reflexHeading(env, np.array([[env.x, env.y]]), np.array([np.arctan2(out[1], out[0])]),
                                     np.array([speed]))
            out, speed = (np.cos(h[0]), np.sin(h[0])), float(sp[0])
            env._noteLeaderReflex(w[0])
        return env.steerToward(out[0], out[1], speed), speed


class _Seek:
    def __init__(self, x, y, speed):
        self.p, self.speed = (float(x), float(y)), float(speed)

    def control(self, env):
        dx, dy = self.p[0] - env.x, self.p[1] - env.y
        if np.hypot(dx, dy) < 3.0:
            return None
        return env.steerToward(dx, dy, self.speed), self.speed


class _Formation:
    """Instantaneous: switches the formation when the plan reaches it."""
    def __init__(self, spec):
        self.formation = dict(spec)

    def control(self, env):
        env.setFormation(self.formation)
        return None


def _specKey(seg):
    try:
        return json.dumps(seg, sort_keys=True)
    except Exception:
        return None


def compileSegment(seg, env):
    ex = _compile(seg, env)
    ex.spec = _specKey(seg)
    return ex


def _compile(seg, env):
    """JSON-ish dict -> executor. Clamps everything into the feasible set,
    so a sloppy plan is degraded, never rejected."""
    t = str(seg.get('type', 'arc')).lower()
    s = env.clampSpeed(seg.get('speed', env.vNominal))
    if t == 'arc':
        return _Arc(np.clip(float(seg.get('turn', 0.0)), -1, 1), s,
                    np.clip(float(seg.get('duration', env.decisionDt)), 0, 60))
    if t == 'goto':
        h = seg.get('heading', None)
        if h is None:
            h = np.arctan2(float(seg['y']) - env.y, float(seg['x']) - env.x)
        return _Goto(seg['x'], seg['y'], float(h), s)
    if t == 'follow':
        return _Follow(seg['waypoints'], s, start=seg.get('start'))
    if t == 'seek':
        return _Seek(seg['x'], seg['y'], s)
    if t == 'formation':
        spec = {k: v for k, v in seg.items() if k != 'type'}
        F.slotsFor(spec, env.nFan, env.opts.PR)       # validate now, not mid-run
        return _Formation(spec)
    raise ValueError('unknown segment type %r' % t)


# ---------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------

class ShepherdEnv:
    def __init__(self, course, nFan=8, seed=0, opts=None,
                 speedRange=(0.5, 1.5), vNominal=1.0, decisionDt=2.0,
                 tMax=None, captureRadius=6.0, lostRadius=40.0,
                 boidsSeeObstacles=False, logEvery=10, formation=None,
                 followerModel='boids', kSteer=5.0, reassignEvery=0.0):
        self.course = course
        self.nFan = nFan
        self.seed = seed
        self.opts = copy.deepcopy(opts) if opts is not None else SimOptions()
        self.speedRange = tuple(speedRange)
        self.vNominal = vNominal
        self.decisionDt = decisionDt
        self.captureRadius = captureRadius
        self.lostRadius = lostRadius
        self.boidsSeeObstacles = boidsSeeObstacles
        self.followerModel = followerModel        # 'boids' (paper) | 'vfield' | 'reflex' (see followers.py)
        self.kSteer = kSteer
        self.reassignEvery = reassignEvery        # s; 0 = assign only on formation change
        self.logEvery = logEvery
        self.initialFormation = formation or dict(shape='none')
        if tMax is None:
            pts = [course.start[:2]] + list(course.route) + [course.goal[:2]]
            L = float(np.sum(np.hypot(*np.diff(np.asarray(pts, float), axis=0).T)))
            tMax = 3.0 * (L + 4 * np.pi * self.opts.R) / vNominal
        self.tMax = tMax
        self.reset()

    # ---- helpers the executors use ----
    def clampSpeed(self, s):
        return float(np.clip(float(s), *self.speedRange))

    def steerToward(self, vx, vy, speed):
        """Paper's bang-bang Dubins steering, returned as turn in {-1,0,1}."""
        self.opts.v = speed
        om = falconSteering(self.th, vx, vy, self.opts)
        return float(np.sign(om))

    # ---- lifecycle ----
    def reset(self):
        rng = np.random.default_rng(self.seed)
        self.x, self.y, self.th = map(float, self.course.start)
        self.speed = self.vNominal
        self.t = 0.0
        self.steps = 0
        # fanboids: a loose clump behind the leader, min spacing 4 m
        fx, fy = [], []
        while len(fx) < self.nFan:
            r = rng.uniform(8.0, 18.0)
            a = self.th + np.pi + rng.uniform(-np.pi / 3, np.pi / 3)
            px, py = self.x + r * np.cos(a), self.y + r * np.sin(a)
            if all(np.hypot(px - qx, py - qy) > 4.0 for qx, qy in zip(fx, fy)):
                fx.append(px)
                fy.append(py)
        self.fanX, self.fanY = np.array(fx), np.array(fy)
        self.fanVx = np.full(self.nFan, self.speed * np.cos(self.th))
        self.fanVy = np.full(self.nFan, self.speed * np.sin(self.th))
        self.squad = np.zeros(self.nFan, dtype=int)

        # leader breadcrumbs for track-frame slots; seeded with a straight
        # line behind the start pose so there is a track from t = 0
        self.trailStep, self.trailMax = 0.5, 400
        self.trail = [(self.x - k * self.trailStep * np.cos(self.th),
                       self.y - k * self.trailStep * np.sin(self.th), self.th)
                      for k in range(self.trailMax, 0, -1)] + [(self.x, self.y, self.th)]
        self._trailAcc = 0.0
        self.formation = dict(shape='none')
        self.slotsTo = None           # target slots (along, lat), or None = leader pull
        self.slotsFrom = None
        self.blendT = self.blendLeft = 0.0
        self.formationSwitches = 0
        self.slotErrSum, self.slotErrN = 0.0, 0
        self.routeIdx = 0             # course waypoints passed so far
        self.routeRadius = 12.0
        self.plan = []
        self.active = None
        self.collisions = []          # dicts: t, cat, a, b, x, y
        self.reflexEvents = []        # dicts: t, i, x, y (followerModel 'reflex')
        self._reflexOn = np.zeros(self.nFan, bool)
        self._contact = set()
        self.minFanLeader = np.inf
        self.minFanFan = np.inf
        self.minObsClear = np.inf     # leader or fan surface clearance
        self.goalMin = np.inf
        self.capture = None           # pose at CPA when captured
        self.done = False
        self.log = {'t': [], 'x': [], 'y': [], 'th': [], 's': [], 'fanX': [], 'fanY': []}
        self.setFormation(self.initialFormation, blend=0.0, count=False)
        self._log()
        return self.observe()

    def _bodyFrame(self):
        c, sn = np.cos(self.th), np.sin(self.th)
        dx, dy = self.fanX - self.x, self.fanY - self.y
        return dx * c + dy * sn, -dx * sn + dy * c

    def currentSlots(self):
        """Slots for this step, as body-frame offsets (what flockStep takes)."""
        if self.slotsTo is None:
            return None
        to = self.slotsTo
        if self.formation.get('frame', 'body') == 'track':
            to = F.trackToBody(self, *to)
        if self.blendLeft <= 0 or self.slotsFrom is None:
            return to
        w = 1.0 - self.blendLeft / self.blendT
        w = w * w * (3 - 2 * w)                       # smoothstep
        return (self.slotsFrom[0] + w * (to[0] - self.slotsFrom[0]),
                self.slotsFrom[1] + w * (to[1] - self.slotsFrom[1]))

    def setFormation(self, spec, blend=None, count=True):
        """Switch formation. Boids are assigned to the new slots by minimum
        total travel, and the slots morph from where each boid currently is
        to its new slot over `blend` seconds."""
        spec = dict(spec)
        if spec == self.formation:
            return
        new = F.slotsFor(spec, self.nFan, self.opts.PR)
        if count:
            self.formationSwitches += 1
        self.formation = spec
        if new is None:
            self.slotsTo = self.slotsFrom = None
            return
        relA, relL = self._bodyFrame()
        if spec.get('frame', 'body') == 'track':
            eqA, eqL = F.trackToBody(self, *new)
            perm = F.assignIndex(relA, relL, eqA, eqL)
        else:
            perm = F.assignIndex(relA, relL, *new)
        self.slotsTo = (new[0][perm], new[1][perm])
        b = float(spec.get('blend', 3.0)) if blend is None else blend
        self.blendT = self.blendLeft = max(0.0, b)
        self.slotsFrom = (relA, relL) if self.blendT > 0 else None

    def _reassign(self):
        """Re-match followers to the current slots by minimum total squared
        distance. In single file followers cannot pass one another, so a
        follower behind the wrong neighbour would otherwise wait for ever."""
        relA, relL = self._bodyFrame()
        eq = F.trackToBody(self, *self.slotsTo) if self.formation.get('frame', 'body') == 'track' else self.slotsTo
        perm = F.assignIndex(relA, relL, *eq)
        self.slotsTo = (self.slotsTo[0][perm], self.slotsTo[1][perm])

    def formationInfo(self):
        g = F.geometry(*(self.slotsTo if self.slotsTo is not None else (None, None)), self.opts.PR)
        info = dict(spec=self.formation, **g)
        sl = self.currentSlots()
        if sl is not None and self.nFan:
            a, l = self._bodyFrame()
            info['meanSlotError'] = round(float(np.mean(np.hypot(a - sl[0], l - sl[1]))), 2)
        return info

    def setPlan(self, plan):
        """Replace the plan. None keeps the current one."""
        if plan is None:
            return
        # Re-sending the segment that is already running (an LLM that says
        # "goto goal" every turn) continues it rather than restarting it.
        if (plan and self.active is not None and
                getattr(self.active, 'spec', None) == _specKey(plan[0])):
            segs = [self.active]
            plan = plan[1:]
        else:
            segs = []
        self.active = None
        for seg in plan:
            try:
                segs.append(compileSegment(seg, self))
            except Exception as e:     # bad segment from an LLM: drop it, say so
                self.lastPlanError = '%s: %r' % (e, seg)
        self.plan = segs

    # ---- one dt ----
    def _control(self):
        while True:
            if self.active is None:
                if not self.plan:
                    return 0.0, self.speed            # coast straight
                self.active = self.plan.pop(0)
            u = self.active.control(self)
            if u is not None:
                return u
            self.active = None

    def stepDt(self):
        o = self.opts
        dt = o.dt
        turn, s = self._control()
        s = min(self.clampSpeed(s), self.leaderSpeedCap())
        self.speed = s
        # leader: exact integration of a constant-curvature arc, so an
        # open-loop Dubins plan lands where the solver said it would and
        # re-solving mid-path reproduces the remainder instead of a loop
        ds = s * dt
        if abs(turn) < 1e-12:
            self.x += ds * np.cos(self.th)
            self.y += ds * np.sin(self.th)
        else:
            k = turn / o.R
            thN = self.th + k * ds
            self.x += (np.sin(thN) - np.sin(self.th)) / k
            self.y -= (np.cos(thN) - np.cos(self.th)) / k
            self.th = thN
        self.th = float(wrapToPi(self.th))
        self._trailAcc += ds
        if self._trailAcc >= self.trailStep:
            self._trailAcc -= self.trailStep
            self.trail.append((self.x, self.y, self.th))
            if len(self.trail) > self.trailMax + 1:
                del self.trail[0]

        # fanboids: paper's flockStep, leader pull, no slots
        if self.nFan:
            o.v = s                    # flockStep reads leader speed from opts.v
            sl = self.currentSlots()
            if self.followerModel in ('vfield', 'reflex') and sl is not None:
                # boid rules with the slot spring switched off, plus a Reynolds
                # steering term toward the vector-field velocity on the slot line
                lf, o.fanLeaderFactor = o.fanLeaderFactor, 0.0
                vxD, vyD = flockStep(self.fanX, self.fanY, self.fanVx, self.fanVy,
                                     self.squad, np.array([self.x]), np.array([self.y]),
                                     np.array([self.th]), o, sepOnlyOnEncounter=False,
                                     slotAlong=sl[0], slotLateral=sl[1])
                o.fanLeaderFactor = lf
                vfx, vfy, _, _ = vfieldDesired(self)
                if self.followerModel == 'reflex':
                    vfx, vfy, wr = reflexDesired(self, vfx, vfy)
                    on = wr > 0.25                     # a reflex "save": swerving hard
                    for i in np.nonzero(on & ~self._reflexOn)[0]:
                        self.reflexEvents.append(dict(t=round(self.t, 2), i=int(i), x=float(self.fanX[i]),
                                                      y=float(self.fanY[i])))
                    self._reflexOn = on
                vxD = vxD + self.kSteer * (vfx - self.fanVx) * dt
                vyD = vyD + self.kSteer * (vfy - self.fanVy) * dt
            else:
                vxD, vyD = flockStep(self.fanX, self.fanY, self.fanVx, self.fanVy,
                                     self.squad, np.array([self.x]), np.array([self.y]),
                                     np.array([self.th]), o, sepOnlyOnEncounter=False,
                                     slotAlong=None if sl is None else sl[0],
                                     slotLateral=None if sl is None else sl[1])
            if self.blendLeft > 0:
                self.blendLeft = max(0.0, self.blendLeft - dt)
            if self.boidsSeeObstacles and len(self.course.obstacles):
                ob = self.course.obstacles
                dx = self.fanX[:, None] - ob[None, :, 0]
                dy = self.fanY[:, None] - ob[None, :, 1]
                D = np.hypot(dx, dy)
                gap = D - ob[None, :, 2]
                near = gap < o.PR
                w = np.where(near, (o.PR - gap) / np.maximum(D, 1e-6), 0.0)
                vxD += (dx * w).sum(1) * o.SF * dt
                vyD += (dy * w).sum(1) * o.SF * dt
            self.fanVx, self.fanVy, _ = ackermannClamp(self.fanVx, self.fanVy, vxD, vyD, o)
            self.capFanSpeeds()
            self.fanX = self.fanX + self.fanVx * dt
            self.fanY = self.fanY + self.fanVy * dt

        self.t += dt
        self.steps += 1
        if (self.reassignEvery > 0 and self.slotsTo is not None and self.blendLeft <= 0 and
                self.steps % max(1, int(round(self.reassignEvery / dt))) == 0):
            self._reassign()
        rt = self.course.route
        while self.routeIdx < len(rt) and self._passed(self.routeIdx):
            self.routeIdx += 1
        self._collisions()
        self._goal()
        self.postStep()
        if self.steps % self.logEvery == 0:
            self._log()
        if self.t >= self.tMax:
            self.done = True

    def _noteLeaderReflex(self, w):
        on = w > 0.25
        if on and not getattr(self, '_leaderReflexOn', False):
            self.reflexEvents.append(dict(t=round(self.t, 2), i=-1, x=float(self.x), y=float(self.y)))
        self._leaderReflexOn = on

    # ---- hooks for subclasses (the farm): terrain and metrics ----
    def leaderSpeedCap(self):
        return np.inf

    def capFanSpeeds(self):
        pass

    def postStep(self):
        pass

    def _log(self):
        L = self.log
        L['t'].append(self.t); L['x'].append(self.x); L['y'].append(self.y)
        L['th'].append(self.th); L['s'].append(self.speed)
        L['fanX'].append(self.fanX.copy()); L['fanY'].append(self.fanY.copy())

    def _collisions(self):
        r = self.opts.collisionRadius
        br = self.opts.bodyRadius
        now = set()
        hits = []
        if self.nFan:
            dl = np.hypot(self.fanX - self.x, self.fanY - self.y)
            self.minFanLeader = min(self.minFanLeader, float(dl.min()))
            for i in np.nonzero(dl < r)[0]:
                hits.append(('fan-leader', 'L', 'f%d' % i, self.fanX[i], self.fanY[i]))
            if self.nFan > 1:
                dx = self.fanX[:, None] - self.fanX[None, :]
                dy = self.fanY[:, None] - self.fanY[None, :]
                D = np.hypot(dx, dy)
                iu = self._iu if hasattr(self, '_iu') else np.triu_indices(self.nFan, 1)
                self._iu = iu
                self.minFanFan = min(self.minFanFan, float(D[iu].min()))
                close = D[iu] < r
                for a, b in zip(iu[0][close], iu[1][close]):
                        hits.append(('fan-fan', 'f%d' % a, 'f%d' % b, self.fanX[a], self.fanY[a]))
        ob = self.course.obstacles
        if len(ob):
            ax = np.concatenate([[self.x], self.fanX])
            ay = np.concatenate([[self.y], self.fanY])
            # spatial index over obstacle centres, built once per course;
            # only obstacles within reach of some agent are measured
            tree = getattr(self.course, '_obsTree', None)
            if tree is None:
                from scipy.spatial import cKDTree
                tree = self.course._obsTree = cKDTree(ob[:, :2])
                self.course._obsReach = float(ob[:, 2].max()) + br + 10.0
            # 8 nearest centres per agent: obstacles are at most a few metres
            # across, so anything an agent can be touching is among them
            D, idx = tree.query(np.column_stack([ax, ay]), k=min(8, len(ob)),
                                distance_upper_bound=self.course._obsReach)
            D, idx = np.atleast_2d(D), np.atleast_2d(idx)
            valid = idx < len(ob)
            gap = np.where(valid, D - ob[np.minimum(idx, len(ob) - 1), 2] - br, np.inf)
            self.minObsClear = min(self.minObsClear, float(gap.min()))
            for a, j in zip(*np.nonzero(gap < 0)):
                k = idx[a, j]
                cat = 'leader-obstacle' if a == 0 else 'fan-obstacle'
                who = 'L' if a == 0 else 'f%d' % (a - 1)
                hits.append((cat, who, 'o%d' % k, ax[a], ay[a]))
        for cat, a, b, hx, hy in hits:
            # one episode per agent per continuous obstacle contact, so a boid
            # scraping along a wall of circles counts once, not once per circle
            key = (a, 'obs') if cat.endswith('obstacle') else (a, b)
            if key in now:
                continue
            now.add(key)
            if key not in self._contact:      # new episode, like collisions.py
                self.collisions.append(dict(t=round(self.t, 2), cat=cat, a=a, b=b,
                                            x=float(hx), y=float(hy)))
        self._contact = now

    def _goal(self):
        gx, gy, gth = self.course.goal
        d = np.hypot(gx - self.x, gy - self.y)
        # A pass only counts when roughly aligned with the goal heading, so
        # crossing the goal sideways in the middle of a loop does not end the
        # episode. Within the disk the CPA is recorded; leaving it ends it.
        aligned = abs(wrapToPi(self.th - gth)) < np.deg2rad(30.0)
        if d < self.captureRadius and (aligned or np.isfinite(self.goalMin)):
            if d < self.goalMin:
                self.goalMin = d
                self._best = (self.x, self.y, self.th, self.t)
            elif d > self.goalMin + 0.025:
                self.capture = self._best
                self.done = True

    # ---- the agent loop ----
    def runFor(self, seconds):
        n = int(round(seconds / self.opts.dt))
        nCol = len(self.collisions)
        for _ in range(n):
            if self.done:
                break
            self.stepDt()
        return self.collisions[nCol:]

    def observe(self):
        """Everything an agent may use, as plain Python numbers."""
        gx, gy, gth = self.course.goal
        fx, fy = self.fanX, self.fanY
        dl = np.hypot(fx - self.x, fy - self.y) if self.nFan else np.array([])
        counts = {c: sum(1 for k in self.collisions if k['cat'] == c) for c in CATS}
        ob = self.course.obstacles
        obs = []
        if len(ob):
            d = np.hypot(ob[:, 0] - self.x, ob[:, 1] - self.y) - ob[:, 2]
            for k in np.argsort(d)[:8]:
                if d[k] < 60:
                    obs.append([round(float(ob[k, 0]), 1), round(float(ob[k, 1]), 1),
                                round(float(ob[k, 2]), 1), round(float(d[k]), 1)])
        return dict(
            t=round(self.t, 2), tMax=round(self.tMax, 1), done=self.done,
            leader=dict(x=round(self.x, 2), y=round(self.y, 2),
                        heading=round(self.th, 3), speed=round(self.speed, 2)),
            goal=dict(x=gx, y=gy, heading=round(gth, 3),
                      distance=round(float(np.hypot(gx - self.x, gy - self.y)), 2),
                      bearing=round(float(np.arctan2(gy - self.y, gx - self.x)), 3)),
            flock=dict(
                n=self.nFan,
                centroid=[round(float(fx.mean()), 2), round(float(fy.mean()), 2)] if self.nFan else None,
                meanDistToLeader=round(float(dl.mean()), 2) if self.nFan else None,
                minDistToLeader=round(float(dl.min()), 2) if self.nFan else None,
                positions=[[round(float(a), 1), round(float(b), 1)] for a, b in zip(fx, fy)],
                headings=[round(float(np.arctan2(b, a)), 2) for a, b in zip(self.fanVx, self.fanVy)],
            ),
            obstaclesNearest=obs,                   # [x, y, r, surface distance]
            corridorAhead=F.corridorAhead(self),    # free half-width left/right along your route
            formation=self.formationInfo(),
            routeRemaining=[list(map(float, p)) for p in self.remainingRoute()],
            routeLegStart=list(map(float, self.lastRoutePoint())),
            collisions=counts,
            limits=dict(R=self.opts.R, speedRange=list(self.speedRange),
                        fanSpeedRange=[self.opts.minSpeed, self.opts.maxSpeed],
                        PR=self.opts.PR, VR=self.opts.VR,
                        collisionRadius=self.opts.collisionRadius),
        )

    def result(self):
        gx, gy, gth = self.course.goal
        if self.capture is not None:
            x, y, th, tc = self.capture
            captured = True
        else:
            x, y, th, tc = self.x, self.y, self.th, self.t
            captured = False
        lost = self.nLost()
        counts = {c: sum(1 for k in self.collisions if k['cat'] == c) for c in CATS}
        return dict(
            course=self.course.name, nFan=self.nFan, seed=self.seed,
            captured=captured, time=round(tc, 2),
            posErr=float(np.hypot(gx - x, gy - y)),
            headErrDeg=float(abs(np.rad2deg(wrapToPi(th - gth)))),
            collisions=sum(counts.values()), **counts,
            fansLost=lost, formationSwitches=self.formationSwitches,
            minFanLeader=self.minFanLeader, minFanFan=self.minFanFan,
            minObsClear=self.minObsClear, reflexEvents=len(getattr(self, 'reflexEvents', [])),
        )

    def targetsWorld(self):
        """Where each boid is supposed to be: its slot, or the leader."""
        sl = self.currentSlots()
        if sl is None:
            return np.full(self.nFan, self.x), np.full(self.nFan, self.y)
        c, sn = np.cos(self.th), np.sin(self.th)
        return self.x + sl[0] * c - sl[1] * sn, self.y + sl[0] * sn + sl[1] * c

    def nLost(self):
        """Boids more than lostRadius from where they are supposed to be."""
        if not self.nFan:
            return 0
        tx, ty = self.targetsWorld()
        return int(np.sum(np.hypot(self.fanX - tx, self.fanY - ty) > self.lostRadius))

    def _passed(self, i):
        """Route waypoint i counts as passed once the leader is within
        routeRadius of it AND past the line through it perpendicular to the
        outgoing leg (or within 3 m). Distance alone dropped corners early
        and every re-plan then cut them."""
        rt = self.course.route
        w = np.asarray(rt[i], float)
        d = np.hypot(w[0] - self.x, w[1] - self.y)
        if d < 3.0:
            return True
        if d > self.routeRadius:
            return False
        nxt = np.asarray(rt[i + 1] if i + 1 < len(rt) else self.course.goal[:2], float)
        u = nxt - w
        return float((self.x - w[0]) * u[0] + (self.y - w[1]) * u[1]) >= 0.0

    def lastRoutePoint(self):
        """Start of the current route leg: last passed waypoint, or start."""
        return list(self.course.route[self.routeIdx - 1]) if self.routeIdx > 0 \
            else list(self.course.start[:2])

    def remainingRoute(self):
        """Course waypoints not yet passed (see _passed)."""
        return list(self.course.route[self.routeIdx:])

    def clone(self):
        """Copy for rollouts. The trajectory log is not copied (it is the
        expensive part and rollouts do not need it)."""
        log, self.log = self.log, {k: [] for k in self.log}
        # the course (and any static arrays a subclass lists in _static) is
        # read-only during a run: share it instead of copying it per rollout
        memo = {id(self.course): self.course}
        for name in getattr(self, '_static', ()):
            memo[id(getattr(self, name))] = getattr(self, name)
        try:
            return copy.deepcopy(self, memo)
        finally:
            self.log = log


def score(res, w=None):
    """One scalar for ranking. Lower is better. Weights are a choice, keep
    them visible: every comparison should report the components too."""
    w = w or dict(pos=1.0, head=0.2, col=10.0, crash=100.0, lost=25.0, time=0.02, miss=200.0)
    return (w['pos'] * res['posErr'] + w['head'] * res['headErrDeg'] +
            w['col'] * res['collisions'] + w['crash'] * res['leader-obstacle'] +
            w['lost'] * res['fansLost'] +
            w['time'] * res['time'] + (0 if res['captured'] else w['miss']))
