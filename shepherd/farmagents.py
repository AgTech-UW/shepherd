"""Deciders for the farm mission. All of them choose the same three things
at the same moments, and the physics executes them:

  formation   one of FARM_PRESETS (or, for the language model, a custom spec)
  speed       leader speed, m/s
  offset      sideways shift of the pass being driven, m (+ = left of travel),
              limited so the working band stays inside the field

The leader follows the route with the paper's vector-field guidance and
parks with a Dubins path, whoever decides.

  FixedColumn     single file all the way. Safe and unproductive.
  Schedule        the obvious program: single file on lanes, gates and
                  headlands, the full working line on passes. Hazard-blind.
  FarmRule        what an engineer would add: per-hazard handling (shift so a
                  rock sits between machine lanes, fold the wet side away from
                  a slough on a wet day, plough through on a weather day). It
                  knows two instructions, 'wet' and 'weather'; anything else
                  falls back to 'wet'.
  FarmSearch      oracle: simulates every candidate on a copy of the true
                  state for `horizon` s and scores it with the instruction's
                  true cost. A reference, not a deployable controller.
"""
import numpy as np

from . import formation as F
from .agents import baselinePlan, rollout
from .farmenv import INSTRUCTIONS, instructionCost

# ---------------------------------------------------------------------
#  Formations. Slots are (along, lateral) in the leader's frame, lateral +
#  is left. Every working formation keeps neighbouring machines 12 m apart
#  (> P = 10), so all of them satisfy the spacing condition.
# ---------------------------------------------------------------------

def _custom(pts, frame='body'):
    return dict(shape='custom', slots=[list(map(float, p)) for p in pts], frame=frame)

_LINE = [(-12, l) for l in (12, 24, 36, 48, -12, -24, -36, -48)]
_LEFT = [(-12, l) for l in (12, 24, 36, 48)] + [(-24, l) for l in (12, 24, 36, 48)]
_RIGHT = [(a, -l) for a, l in _LEFT]

FARM_PRESETS = {
    'line':      _custom(_LINE),
    'lineLeft':  _custom(_LEFT),
    'lineRight': _custom(_RIGHT),
    'vee':       dict(shape='vee', spacing=12, openingDeg=45, standoff=12),
    'column2':   dict(shape='column', spacing=12, ranks=2, frame='track'),
    'column':    dict(shape='column', spacing=12, frame='track'),
}

FARM_DESCRIPTIONS = {
    'line':      'working line: all 9 machines abreast, 12 m apart',
    'lineLeft':  'half line folded LEFT: followers in two ranks on your left only',
    'lineRight': 'half line folded RIGHT: followers in two ranks on your right only',
    'vee':       'vee, 45 degrees, 12 m spacing (the paper\'s formation)',
    'column2':   'column two abreast, following your track',
    'column':    'single file, following your track',
}


def presetLanes(name, n=8, P=10.0):
    """Lateral positions of all machines (leader at 0) and formation length."""
    a, l = F.slotsFor(FARM_PRESETS[name], n, P)
    lanes = np.concatenate([[0.0], l])
    return lanes, float(-a.min())


def presetTable(swath=12.0):
    rows = []
    for name in FARM_PRESETS:
        lanes, length = presetLanes(name)
        u = np.unique(np.round(lanes, 1))
        band = (u.max() - u.min()) + swath
        rows.append('  %-9s %-62s working width %3.0f m, %d distinct lanes, length %3.0f m'
                    % (name, FARM_DESCRIPTIONS[name], band, len(u), length))
    return '\n'.join(rows)


def resolveFormation(f, n=8, P=10.0):
    """Preset name or custom spec -> (name, spec). Raises on nonsense."""
    if isinstance(f, str) and f in FARM_PRESETS:
        return f, dict(FARM_PRESETS[f])
    if isinstance(f, dict):
        spec = {k: v for k, v in f.items() if k != 'type'}
        F.slotsFor(spec, n, P)
        return 'custom', spec
    raise ValueError('unknown formation %r' % (f,))


def specLanes(spec, n=8, P=10.0):
    a, l = F.slotsFor(spec, n, P) or (np.zeros(n), np.zeros(n))
    return np.concatenate([[0.0], l]), float(-a.min())


# ---------------------------------------------------------------------
#  Plans
# ---------------------------------------------------------------------

def farmPlan(env, spec, speed, offset=0.0):
    """Formation switch, then follow the remaining route (the current pass
    shifted sideways by `offset`), then park."""
    rem = [list(p) for p in env.remainingRoute()]
    start = list(env.lastRoutePoint())
    fr = env.passFrame()
    if fr is not None and rem:
        lanes, _ = specLanes(spec, env.nFan, env.opts.PR)
        lo, hi = env.offsetLimits(lanes)
        offset = float(np.clip(offset, lo, hi)) if lo <= hi else 0.0
        n = fr[3]
        rem[0] = [rem[0][0] + offset * n[0], rem[0][1] + offset * n[1]]
        start = [start[0] + offset * n[0], start[1] + offset * n[1]]
    else:
        offset = 0.0
    env.passOffset = offset
    g = env.course.goal
    plan = [dict(type='formation', **spec)]
    if rem:
        plan.append(dict(type='follow', waypoints=rem, speed=speed, start=start))
    plan.append(dict(type='goto', x=g[0], y=g[1], heading=g[2], speed=speed))
    return plan


def passProgress(env):
    """(metres done, metres left) along the current pass, or None."""
    fr = env.passFrame()
    if fr is None:
        return None
    a, b, u, n = fr
    s = float((np.array([env.x, env.y]) - a) @ u)
    return s, float(np.hypot(*(b - a))) - s


def hazardsPassFrame(env, behind=110.0, ahead=90.0):
    """Hazards near the current pass, in pass coordinates: along (from the
    leader, + ahead) and lateral (from the pass centre line, + left)."""
    fr = env.passFrame()
    if fr is None:
        return []
    a, b, u, n = fr
    out = []
    lead = np.array([env.x, env.y])
    for h in env.course.hazards:
        p = np.array([h['x'], h['y']])
        along = float((p - lead) @ u)
        if -behind <= along <= ahead:
            out.append(dict(type=h['type'], along=along, lat=float((p - a) @ n), r=h['r']))
    return out


def laneConflicts(lanes, offset, hz, swath=12.0, body=0.5, margin=1.5):
    """For one hazard in pass coordinates: which machine lanes would touch it.
    Solid hazards: the machine body plus a margin. Sloughs: the implement."""
    pos = lanes + offset
    if hz['type'] == 'slough':
        return [float(x) for x in pos if abs(hz['lat'] - x) < hz['r'] + swath / 2]
    return [float(x) for x in pos if abs(hz['lat'] - x) < hz['r'] + body + margin]



def _fitContext(env, reach=60.0, turnDeg=30.0):
    """What every formation is checked against: the free width either side
    of the route over the next `reach` metres (off the passes; sampled every
    5 m so a fence post between samples does not make it flicker), and
    whether the route turns by more than `turnDeg` within that distance."""
    inPass = env.passFrame() is not None
    fl = fr = np.inf
    if not inPass:
        rows = F.corridorAhead(env, stations=tuple(range(0, int(reach) + 1, 5)), rayMax=120.0)
        fl = min([r['left'] for r in rows] or [120.0])
        fr = min([r['right'] for r in rows] or [120.0])
    return dict(fl=fl, fr=fr, turning=F.routeTurnAhead(env, reach) > turnDeg)


def fitsSpec(env, spec, ctx=None, margin=4.0):
    """Can formation `spec` be taken up here? It can when
      - every follower can reach its new slot in a straight line without
        touching an obstacle (followers cannot see obstacles, so widening
        while the tail is still in a lane or a gate sends them through it),
      - off the passes, the machines beside the leader's track fit between
        the obstacles on the route over the next 60 m with `margin` to spare
        (the leader's own track is the autopilot's, not a formation choice;
        on a pass the allowed offsets keep the machines inside the fences),
      - for body-frame formations, the route does not turn by more than 30
        degrees within 60 m, because they swing sideways in turns.
    The hand-written rule encodes the same facts as a schedule (single file
    off the passes and near their ends).

    Version 2 (env.fitVersion = 2): the first condition counts only fences,
    gate posts and tree lines. Rocks and poles are left to the reflex; in
    version 1 a rock between the followers and their new slots froze the
    formation next to it, exactly when a decider was changing it to get past."""
    ctx = ctx or _fitContext(env)
    lanes, _ = specLanes(spec, env.nFan, env.opts.PR)
    hiL, hiR = float(lanes.max()), float(-lanes.min())
    if hiL > 0 and hiL + margin > ctx['fl']:
        return False
    if hiR > 0 and hiR + margin > ctx['fr']:
        return False
    if ctx['turning'] and spec.get('frame', 'body') != 'track':
        return False
    return F.pathsToSlotsClear(env, spec, boundaryOnly=getattr(env, 'fitVersion', 1) >= 2)


def formationFit(env):
    """{preset name: fitsSpec} for every preset."""
    ctx = _fitContext(env)
    return {name: bool(fitsSpec(env, spec, ctx)) for name, spec in FARM_PRESETS.items()}


def _width(spec, env):
    lanes, _ = specLanes(spec, env.nFan, env.opts.PR)
    return float(lanes.max() - lanes.min())


def presetName(spec):
    for k, v in FARM_PRESETS.items():
        if v == spec:
            return k
    return 'custom'


def shieldSpec(env, spec, currentSpec=None):
    """The safety filter: the fit check made binding. Returns (spec to take
    up, blocked). A formation that fails `fitsSpec` is not taken up; the
    current formation is kept if it still fits, otherwise the widest preset
    that fits, otherwise the current one (or single file) is held."""
    ctx = _fitContext(env)
    if fitsSpec(env, spec, ctx):
        return spec, False
    if currentSpec is not None and fitsSpec(env, currentSpec, ctx):
        return currentSpec, True
    fits = [v for v in FARM_PRESETS.values() if fitsSpec(env, v, ctx)]
    if fits:
        return dict(max(fits, key=lambda v: _width(v, env))), True
    return (currentSpec if currentSpec is not None else dict(FARM_PRESETS['column'])), True


def _nLanes(spec, env):
    lanes, _ = specLanes(spec, env.nFan, env.opts.PR)
    return len(np.unique(np.round(lanes, 1)))


# ---------------------------------------------------------------------
#  Decision points, shared by the search and the language model
# ---------------------------------------------------------------------

_CLASSES = (('open', 30.0), ('wide', 12.0), ('narrow', 0.0))


class Triggers:
    """When to decide: start, a new phase, a hazard coming into sensor range,
    the corridor ahead changing width class, the formation being held no
    longer fitting (two checks running), a wider formation having fitted for
    `roomFor` s (once per phase and formation), a collision (at most every
    `collisionGap` s), otherwise every `period` s. See fitsSpec."""
    def __init__(self, period=30.0, collisionGap=8.0, roomFor=6.0):
        self.period, self.collisionGap, self.roomFor = period, collisionGap, roomFor

    def reset(self):
        self.last = -np.inf
        self.phase = None
        self.seen = set()
        self.width = None
        self.nCol = 0
        self.misfit = 0
        self.since = {}
        self.offered = set()

    def _width(self, env):
        rows = [r for r in F.corridorAhead(env) if r['ahead'] <= 45]
        w = min([min(r['left'], r['right']) for r in rows] or [40.0])
        c = next(n for n, lo in _CLASSES if w >= lo)
        if self.width is None or c == self.width:
            return c
        lo = dict(_CLASSES)[self.width]
        names = [n for n, _ in _CLASSES]
        hi = dict(_CLASSES)[names[names.index(self.width) - 1]] if self.width != names[0] else np.inf
        return c if (w < lo - 1.5 or w >= hi + 1.5) else self.width

    def check(self, env):
        why = []
        ph = env.phase()
        if self.phase is None:
            why.append('start')
        else:
            if ph != self.phase:
                why.append('entering %s' % ph)
            new = [h['id'] for h in env.hazardsInRange() if h['id'] not in self.seen]
            if new:
                why.append('hazard in sensor range')
            # corridor width only matters off the field; inside it, hazards
            # have their own trigger and rays clipping a rock would flicker
            if not ph.startswith('pass'):
                w = self._width(env)
                if self.width is not None and w != self.width:
                    why.append('corridor ahead now %s' % w)
                self.width = w
            ctx = _fitContext(env)
            cur = env.formation if env.formation.get('shape', 'none') != 'none' else None
            self.misfit = (self.misfit + 1) if (cur is not None and not fitsSpec(env, cur, ctx)) else 0
            if self.misfit == 2:
                why.append('formation no longer fits')
            if ph != self.phase:
                self.offered = set()
            nCur = _nLanes(cur, env) if cur is not None else 1
            for name, spec in FARM_PRESETS.items():
                if fitsSpec(env, spec, ctx):
                    self.since.setdefault(name, env.t)
                else:
                    self.since.pop(name, None)
            room = [k for k, t0 in self.since.items() if env.t - t0 >= self.roomFor and k not in self.offered
                    and _nLanes(FARM_PRESETS[k], env) > nCur]
            if room and env.t - self.last >= 10.0:
                why.append('room for a wider formation')
                self.offered |= set(room)
            if len(env.collisions) > self.nCol and env.t - self.last >= self.collisionGap:
                why.append('collision')
            if env.t - self.last >= self.period:
                why.append('periodic')
        if self.phase is None:
            self.width = self._width(env)
        self.phase = ph
        self.seen |= {h['id'] for h in env.hazardsInRange()}
        if why:
            self.last = env.t
            self.nCol = len(env.collisions)
        return why


# ---------------------------------------------------------------------
#  Comparators
# ---------------------------------------------------------------------

class _Base:
    name = 'base'
    shield = False              # safety filter on (see shieldSpec)

    def begin(self, env):
        self.current = None
        self.nBlocked = 0

    def _apply(self, env, name, speed, offset=0.0):
        if self.shield and (self.current is None or name != self.current[0]):   # only a change is checked
            cur = dict(FARM_PRESETS[self.current[0]]) if self.current and self.current[0] in FARM_PRESETS else None
            spec, blocked = shieldSpec(env, dict(FARM_PRESETS[name]), cur)
            if blocked:
                self.nBlocked += 1
                name = presetName(spec)
        key = (name, round(speed, 2), round(offset, 1))
        if key == self.current:
            return None
        self.current = key
        self.choices.append(dict(t=round(env.t, 1), formation=name, speed=speed, offset=offset))
        return farmPlan(env, dict(FARM_PRESETS[name]), speed, offset)

    def end(self, env, result):
        pass


def _scheduleChoice(env, workName='line', fold=35.0, unfold=20.0):
    """Single file except on a pass, where the work formation is used once
    the column has straightened out and until it is time to fold."""
    pp = passProgress(env)
    if pp is None:
        return 'column'
    done, left = pp
    return workName if (done > unfold and left > fold) else 'column'


class FixedColumn(_Base):
    name = 'column'

    def begin(self, env):
        super().begin(env)
        self.choices = []

    def act(self, env, obs):
        return self._apply(env, 'column', 1.0)


class Schedule(_Base):
    name = 'schedule'

    def begin(self, env):
        super().begin(env)
        self.choices = []

    def act(self, env, obs):
        return self._apply(env, _scheduleChoice(env), 1.0)


class FarmRule(_Base):
    """Hand-written per-hazard rule. mode: 'wet' or 'weather'."""
    name = 'rule'

    def __init__(self, mode='wet'):
        self.mode = mode if mode in ('wet', 'weather') else 'wet'

    def begin(self, env):
        super().begin(env)
        self.choices = []

    def _score(self, env, name, off, hz):
        lanes, length = presetLanes(name, env.nFan, env.opts.PR)
        sc = 3.0 * len(np.unique(np.round(lanes, 1))) * (1.6 if self.mode == 'weather' else 1.0)
        sc -= 0.05 * abs(off)
        for h in hz:
            if not (-(length + 10) <= h['along'] <= 70):
                continue
            hit = laneConflicts(lanes, off, h, env.course.spec.swath)
            if h['type'] == 'slough':
                sc -= len(hit) * (20.0 if self.mode == 'wet' else 0.5)
            elif hit:
                sc -= 1000.0
        return sc

    def act(self, env, obs):
        speed = 1.3 if self.mode == 'weather' else 1.0
        sched = _scheduleChoice(env)
        if sched == 'column':
            return self._apply(env, 'column', 1.0)
        hz = hazardsPassFrame(env)
        best, bestSc = ('line', 0.0), -np.inf
        for name in FARM_PRESETS:
            lanes, _ = presetLanes(name, env.nFan, env.opts.PR)
            lo, hi = env.offsetLimits(lanes)
            if lo > hi:
                continue
            for off in np.arange(-24.0, 24.01, 3.0):
                if lo <= off <= hi:
                    s = self._score(env, name, off, hz)
                    if s > bestSc + 1e-9:
                        best, bestSc = (name, float(off)), s
        if self.current is not None and self.current[0] in FARM_PRESETS:
            cur = (self.current[0], self.current[2])
            lanes, _ = presetLanes(cur[0], env.nFan, env.opts.PR)
            lo, hi = env.offsetLimits(lanes)
            if lo <= cur[1] <= hi and self._score(env, cur[0], cur[1], hz) >= bestSc - 1.0:
                best = cur
        return self._apply(env, best[0], speed, best[1])


def rolloutMetrics(env, plan, horizon, planDt=0.05):
    """Run a plan on a copy; return the change in every farm metric."""
    sim = env.clone()
    sim.opts.dt = planDt
    sim.logEvery = 10 ** 9
    m0 = dict(collisions=len(sim.collisions), slough=sim.sloughExposure, north=sim.northExposure,
              coverage=sim.coverage(), t=sim.t)
    sim.setPlan(plan)
    for _ in range(int(round(horizon / planDt))):
        if sim.done:
            break
        sim.stepDt()
    new = sim.collisions[m0['collisions']:]
    fences = _fenceDistances(sim)
    return dict(collisions=len(new),
                leaderHits=sum(1 for c in new if c['cat'] == 'leader-obstacle'),
                sloughExposure=sim.sloughExposure - m0['slough'],
                northExposure=sim.northExposure - m0['north'],
                coverageGain=sim.coverage() - m0['coverage'],
                time=sim.t - m0['t'], closestToFence=fences)


def _fenceDistances(env):
    L, W = env.course.field[1], env.course.field[3]
    ax = np.concatenate([[env.x], env.fanX])
    ay = np.concatenate([[env.y], env.fanY])
    inside = (ax > 0) & (ax < L) & (ay > 0) & (ay < W)
    if not inside.any():
        return None
    ax, ay = ax[inside], ay[inside]
    return dict(north=round(float((W - ay).min()), 1), south=round(float(ay.min()), 1),
                east=round(float((L - ax).min()), 1), west=round(float(ax.min()), 1))


def deltaCost(d, key):
    m = dict(collisions=d['collisions'], sloughExposure=d['sloughExposure'],
             northExposure=d['northExposure'], coverage=d['coverageGain'], time=d['time'])
    return instructionCost(m, key) + 500.0 * d['leaderHits']


class FarmSearch(_Base):
    """Oracle: same decision points and options as the language model, scored
    by simulating each on the true state with the instruction's true cost."""
    name = 'search'

    def __init__(self, instruction='wet', horizon=40.0, planDt=0.05, speeds=(1.0, 1.3),
                 offsets=(-24.0, -12.0, 0.0, 12.0, 24.0), hysteresis=2.0, period=30.0, verbose=False):
        self.instruction = instruction
        self.horizon, self.planDt = horizon, planDt
        self.speeds, self.offsets, self.hyst = speeds, offsets, hysteresis
        self.triggers = Triggers(period=period)
        self.verbose = verbose

    def begin(self, env):
        super().begin(env)
        self.choices = []
        self.triggers.reset()
        self.nDecisions = 0

    def candidates(self, env):
        out = []
        inPass = env.passFrame() is not None
        ok = None
        if self.shield:
            ctx = _fitContext(env)
            ok = {n for n, v in FARM_PRESETS.items() if fitsSpec(env, v, ctx)} | {'column'}
        for name in FARM_PRESETS:
            if ok is not None and name not in ok:
                continue
            lanes, _ = presetLanes(name, env.nFan, env.opts.PR)
            lo, hi = env.offsetLimits(lanes)
            offs = sorted({float(np.clip(o, lo, hi)) for o in self.offsets}) if inPass and lo <= hi else [0.0]
            if inPass and lo > hi:
                continue
            for o in offs:
                for s in self.speeds:
                    out.append((name, s, o))
        return out

    def act(self, env, obs):
        why = self.triggers.check(env)
        if not why:
            return None
        self.nDecisions += 1
        costs = {}
        for c in self.candidates(env):
            plan = farmPlan(env, dict(FARM_PRESETS[c[0]]), c[1], c[2])
            costs[c] = deltaCost(rolloutMetrics(env, plan, self.horizon, self.planDt), self.instruction)
        env.passOffset = self.current[2] if self.current else 0.0
        best = min(costs, key=costs.get)
        if self.current in costs and costs[best] > costs[self.current] - self.hyst:
            best = self.current
        if self.verbose:
            print('  t=%6.1f %-28s %-40s cost %.1f (%d candidates)' % (env.t, ','.join(why)[:28], best, costs[best], len(costs)))
        if best == self.current:
            return None
        return self._apply(env, *best)
