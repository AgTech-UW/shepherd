"""Agents that drive the leader. All share one interface:

    agent.begin(env)            once per episode
    plan = agent.act(env, obs)  every decisionDt; list of segments or None
    agent.end(env, result)      once per episode

Three of them, deliberately spanning the axis the comparison is about:

  BaselineAgent  the paper's controller: vector-field guidance along the
                 course route, then a Dubins path onto the goal pose. Never
                 looks at the flock. The physics comparator.
  CEMAgent       model-predictive search with a perfect copy of the
                 simulator: perturbs the baseline with a short manoeuvre
                 prefix and keeps whatever the rollouts say collides least.
                 No language, no priors, just search. This is the yardstick
                 for "can the boid rules be gamed at all".
  LLMAgent       an Ollama model reading a text observation and writing a
                 JSON plan from the same segment vocabulary. Optional
                 'whatIf' mode lets it propose several plans and have the
                 simulator score them before one is executed, and a
                 reflection step carries lessons across episodes.
"""
import json
import multiprocessing as mp
import os
import time
import urllib.request
import numpy as np

from .env import CATS
from .dubins import dubinsShortest
from . import formation as F


# ---------------------------------------------------------------------
#  Shared: the baseline plan from wherever the leader is now
# ---------------------------------------------------------------------

def baselinePlan(env, remaining, speed=None):
    g = env.course.goal
    s = env.vNominal if speed is None else speed
    plan = []
    if remaining:
        plan.append(dict(type='follow', waypoints=[list(p) for p in remaining], speed=s,
                         start=env.lastRoutePoint()))
    plan.append(dict(type='goto', x=g[0], y=g[1], heading=g[2], speed=s))
    return plan


def remainingLength(env, remaining):
    """Cost-to-go estimate: polyline through the remaining route, then the
    Dubins length from the last route point (approach heading) to the goal."""
    g = env.course.goal
    pts = [(env.x, env.y)] + list(remaining)
    L = float(np.sum(np.hypot(*np.diff(np.asarray(pts, float), axis=0).T))) if len(pts) > 1 else 0.0
    if remaining:
        hx, hy = (pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])
        q0 = (pts[-1][0], pts[-1][1], np.arctan2(hy, hx))
    else:
        q0 = (env.x, env.y, env.th)
    L += dubinsShortest(q0, g, env.opts.R)[2]
    return L


# ---------------------------------------------------------------------
#  Rollout scoring, shared by CEM and the LLM's what-if tool
# ---------------------------------------------------------------------

def rollout(env, plan, horizon, planDt=None):
    """Run `plan` on a copy of env for `horizon` sim-seconds.
    Returns (cost, summary). planDt coarsens the copy's timestep for speed."""
    sim = env.clone()
    if planDt:
        sim.opts.dt = planDt
    sim.logEvery = 10 ** 9
    sim.setPlan(plan)
    n0 = len(sim.collisions)
    minFL, minObs = np.inf, np.inf
    steps = int(round(horizon / sim.opts.dt))
    for _ in range(steps):
        if sim.done:
            break
        sim.stepDt()
        if sim.nFan:
            minFL = min(minFL, float(np.min(np.hypot(sim.fanX - sim.x, sim.fanY - sim.y))))
    new = sim.collisions[n0:]
    ncol = len(new)
    lost = sim.nLost()
    rem = sim.remainingRoute()
    if sim.capture is not None:
        r = sim.result()
        term = r['posErr'] + 0.2 * r['headErrDeg']
        togo = 0.0
    else:
        term = 0.0
        togo = remainingLength(sim, rem)
    nLeadObs = sum(1 for k in new if k['cat'] == 'leader-obstacle')
    # the leader hitting an obstacle is a crash, not a trade-off: without this
    # the search learns that one collision is cheaper than a long detour and
    # drives straight through walls
    cost = (10.0 * ncol + 500.0 * nLeadObs + 25.0 * lost + term + 0.1 * togo +
            (2.0 * max(0.0, 2.5 - minFL) if np.isfinite(minFL) else 0.0))
    summary = dict(predictedCollisions=ncol,
                   byType={c: sum(1 for k in new if k['cat'] == c) for c in CATS},
                   fansLost=lost, reachedGoal=sim.capture is not None,
                   minFanLeader=round(minFL, 2) if np.isfinite(minFL) else None,
                   distanceToGoAfter=round(togo, 1), cost=round(cost, 2))
    return cost, summary


def _rolloutCost(env, plan, horizon, planDt):
    return rollout(env, plan, horizon, planDt)[0]


# ---------------------------------------------------------------------
#  Baseline
# ---------------------------------------------------------------------

class BaselineAgent:
    name = 'baseline'

    def __init__(self, speed=None):
        self.speed = speed

    def begin(self, env):
        self.sent = False

    def act(self, env, obs):
        if self.sent:
            return None
        self.sent = True
        return baselinePlan(env, env.remainingRoute(), self.speed)

    def end(self, env, result):
        pass


# ---------------------------------------------------------------------
#  CEM / shooting MPC
# ---------------------------------------------------------------------

class CEMAgent:
    """Each decision: search over a prefix of K constant-curvature arcs
    (turn, speed, duration) followed by the baseline tail. A prefix of
    total duration 0 IS the baseline, so the search can only match or beat
    it under its own model. Warm-started from the previous best."""
    name = 'cem'

    def __init__(self, K=3, pop=24, elites=6, iters=3, horizon=16.0,
                 planDt=0.04, maxArc=3.0, seed=0, verbose=False, workers=None):
        self.K, self.pop, self.elites, self.iters = K, pop, elites, iters
        self.horizon, self.planDt, self.maxArc = horizon, planDt, maxArc
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        if workers is None:
            workers = max(1, (os.cpu_count() or 1) - 1)
        self.pool = mp.get_context('fork').Pool(workers) if workers > 1 else None

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    def begin(self, env):
        lo, hi = env.speedRange
        self.lo = np.tile([-1.0, lo, 0.0], self.K)
        self.hi = np.tile([1.0, hi, self.maxArc], self.K)
        self.mu = np.tile([0.0, env.vNominal, 0.0], self.K)
        self.decisions = 0

    def _plan(self, env, z, remaining):
        plan = []
        for k in range(self.K):
            u, s, d = z[3 * k:3 * k + 3]
            if d > 0.05:
                plan.append(dict(type='arc', turn=float(u), speed=float(s), duration=float(d)))
        tailSpeed = float(z[1]) if plan else env.vNominal
        return plan + baselinePlan(env, remaining, speed=np.clip(tailSpeed, *env.speedRange))

    def act(self, env, obs):
        remaining = env.remainingRoute()
        sig = (self.hi - self.lo) * 0.35
        mu = self.mu.copy()
        best = (np.inf, None)
        zero = mu.copy(); zero[2::3] = 0.0           # the baseline itself
        for it in range(self.iters):
            Z = mu + sig * self.rng.standard_normal((self.pop, mu.size))
            Z = np.clip(Z, self.lo, self.hi)
            if it == 0:
                Z[0], Z[1] = zero, mu
            plans = [self._plan(env, z, remaining) for z in Z]
            if self.pool is not None:
                costs = np.array(self.pool.starmap(_rolloutCost, [(env, pl, self.horizon, self.planDt) for pl in plans]))
            else:
                costs = np.array([_rolloutCost(env, pl, self.horizon, self.planDt) for pl in plans])
            order = np.argsort(costs)
            if costs[order[0]] < best[0]:
                best = (costs[order[0]], Z[order[0]].copy())
            E = Z[order[:self.elites]]
            mu, sig = E.mean(0), E.std(0) + 0.05 * (self.hi - self.lo)
        z = best[1]
        # warm start: drop decisionDt of prefix time
        shift = z.copy()
        left = env.decisionDt
        for k in range(self.K):
            take = min(left, shift[3 * k + 2]); shift[3 * k + 2] -= take; left -= take
        self.mu = shift
        self.decisions += 1
        if self.verbose:
            print('  t=%6.1f cost=%.2f prefix=%s' % (env.t, best[0], np.round(z, 2)))
        return self._plan(env, z, remaining)

    def end(self, env, result):
        pass



# ---------------------------------------------------------------------
#  Formation agents
# ---------------------------------------------------------------------

PRESETS = {
    'none':       dict(shape='none'),
    'vee':        dict(shape='vee', spacing=12, openingDeg=45),           # the paper's
    'veeTrack':   dict(shape='vee', spacing=12, openingDeg=45, frame='track'),
    'column2':    dict(shape='column', spacing=12, ranks=2, frame='track'),
    'column':     dict(shape='column', spacing=12, frame='track'),
    'columnBody': dict(shape='column', spacing=12),
}


def preset(nameOrSpec):
    if isinstance(nameOrSpec, dict):
        return dict(nameOrSpec)
    s = str(nameOrSpec)
    if s.strip().startswith('{'):
        return json.loads(s)
    return dict(PRESETS[s])


def _freeAhead(env, reach=45.0):
    rows = [r for r in F.corridorAhead(env) if r['ahead'] <= reach]
    if not rows:
        return 40.0, 40.0
    return min(r['left'] for r in rows), min(r['right'] for r in rows)


class AdaptiveFormationAgent:
    """Hand-written rule, the physics comparator for formation control.

    Motion is the baseline. Formation: the widest of [vee, 2-abreast
    column, single column] whose slots satisfy the spacing condition
    (delta = 12 > P = 10) and fit the free corridor width ahead with a 3 m
    margin. Narrowing happens as soon as the corridor ahead requires it.
    Widening waits until the leader has been in wide space long enough for
    the tail of the current formation to clear the narrow part; otherwise
    the tail is swung through the wall."""
    name = 'adaptive'
    ladder = ('vee', 'column2', 'column')

    def __init__(self, margin=3.0, speed=None):
        self.margin, self.speed = margin, speed

    def begin(self, env):
        self.current = None
        self.wideFor = 0.0
        self.lastT = 0.0

    def _fits(self, env, name, fl, fr):
        g = F.geometry(*F.slotsFor(PRESETS[name], env.nFan, env.opts.PR), env.opts.PR)
        return g['leftHalfWidth'] + self.margin <= fl and g['rightHalfWidth'] + self.margin <= fr

    def act(self, env, obs):
        fl, fr = _freeAhead(env)
        want = next((n for n in self.ladder if self._fits(env, n, fl, fr)), self.ladder[-1])
        dt, self.lastT = env.t - self.lastT, env.t
        if self.current is None or self.ladder.index(want) >= self.ladder.index(self.current):
            choice = want                                     # same or narrower: now
            self.wideFor = 0.0
        else:                                                 # wider: wait for the tail
            self.wideFor += dt
            curLen = F.geometry(*F.slotsFor(PRESETS[self.current], env.nFan, env.opts.PR),
                                env.opts.PR)['length']
            choice = want if self.wideFor * env.speed > curLen + 10 else self.current
            if choice == want:
                self.wideFor = 0.0
        if choice == self.current:
            return None
        self.current = choice
        return [dict(type='formation', **PRESETS[choice])] + \
            baselinePlan(env, env.remainingRoute(), self.speed)

    def end(self, env, result):
        pass


class FormationSearchAgent:
    """Formation and speed chosen by rollouts; motion is the baseline path.
    Every decision it simulates each (formation preset, speed) pair for
    `horizon` seconds and switches only if the best beats the current
    choice by `hysteresis` cost, because every switch disturbs the flock."""
    name = 'formsearch'

    def __init__(self, presets=('none', 'vee', 'veeTrack', 'column2', 'column', 'columnBody'),
                 speeds=(1.0, 1.25), horizon=25.0, planDt=0.05, hysteresis=5.0, verbose=False):
        self.presets, self.speeds = presets, speeds
        self.horizon, self.planDt, self.hyst = horizon, planDt, hysteresis
        self.verbose = verbose

    def begin(self, env):
        self.current = None

    def _plan(self, env, name, sp):
        return [dict(type='formation', **PRESETS[name])] + baselinePlan(env, env.remainingRoute(), sp)

    def act(self, env, obs):
        costs = {(n, sp): rollout(env, self._plan(env, n, sp), self.horizon, self.planDt)[0]
                 for n in self.presets for sp in self.speeds}
        best = min(costs, key=costs.get)
        if self.current is not None and costs[best] > costs[self.current] - self.hyst:
            best = self.current
        if self.verbose:
            print('  t=%6.1f %s %.1f' % (env.t, best, costs[best]))
        if best == self.current:
            return None
        self.current = best
        return self._plan(env, *best)

    def end(self, env, result):
        pass

# ---------------------------------------------------------------------
#  LLM (Ollama)
# ---------------------------------------------------------------------

SYSTEM = """You are the LEADER of a flock of {n} autonomous followers ("fanboids") on a 2-D field. You control only yourself. Your job: arrive EXACTLY at the goal pose (position AND heading) while causing as few collisions as possible.

PHYSICS (fixed, you cannot change it)
- You: a Dubins car. Minimum turn radius R={R} m. Speed between {smin} and {smax} m/s, never zero, never reverse. Heading in radians, 0 = +x, counter-clockwise positive.
- Fanboids: also turn-limited (radius {R} m), speed {fmin}-{fmax} m/s. Each one, every 0.01 s, adds to its velocity:
    separation: pushes away from ANY agent (including you) closer than PR={PR} m. Force = SF*(offset), so it is STRONGEST at the edge of PR and VANISHES at contact. Close up, nothing stops them.
    cohesion + alignment: toward the average position/velocity of flockmates within VR={VR} m.
    leader pull: toward YOUR position, proportional to distance. They are always trying to reach you.
  They do NOT see obstacles. If they cut a corner through a wall, that is a collision. Turn-limited followers overshoot when you turn or slow.
- Collision: two centres closer than {cr} m, or any body touching an obstacle.

SCORE (lower is better): posErr[m] + 0.2*headingErr[deg] + 10*collisions + 100 more if YOU hit an obstacle + 25*fansLost(>40 m from their slot, or from you if no formation) + 0.02*time[s] + 200 if you never reach the goal.

ACTIONS. Reply with ONE JSON object. Either
  {{"keep": true}}                              keep executing your current plan
or
  {{"thought": "<one or two sentences>", "plan": [segment, ...]}}
Segments run in order; when the plan runs out you drive straight. Segment types:
  {{"type":"arc","turn":u,"speed":s,"duration":T}}   u in [-1,1]: +1 hard left, -1 hard right, 0 straight
  {{"type":"goto","x":X,"y":Y,"heading":H,"speed":s}} shortest turn-limited path to that pose (IGNORES obstacles). Ending with a goto to the goal pose is how you hit it exactly.
  {{"type":"follow","waypoints":[[x,y],...],"speed":s,"start":[x,y]}} smooth path-following along the polyline start->waypoints. Give "start" (e.g. observation.routeLegStart) or the first leg is drawn from wherever you are, and re-sending it moves the line
  {{"type":"seek","x":X,"y":Y,"speed":s}}            steer straight at a point
  {{"type":"formation","shape":..., ...}}               re-shape the flock, instantly, when the plan reaches it (put it between motion segments to schedule a change)

FORMATION. You also decide where every follower should be. A formation gives each follower its own slot relative to you; instead of chasing YOU they chase their slot. Keys:
  shape: "none" (all chase you: the default, crowded) | "vee" | "echelon" | "column" | "line" | "custom"
  spacing: metres between neighbouring slots. Slots closer than PR={PR} m sit inside each other's separation range and fight (measured: 2-3x the collision rate). spacing > PR is the known-good regime.
  standoff: distance of the nearest slot behind you (default PR+2)
  openingDeg: vee/echelon half-angle from your centreline: 0 = single file, 90 = abreast (default 45)
  ranks: for "column", followers abreast per rank (1 = single file)
  side: "left"/"right" for "echelon";  slots: [[along, lateral], ...] for "custom" (along<0 is behind you, lateral>0 is your left)
  frame: "body" = slots fixed behind your CURRENT heading (they swing sideways when you turn) | "track" = slots placed along the path you ACTUALLY drove (a column follows you round corners)
  blend: seconds to morph to the new slots (default 3). Followers are re-assigned to the nearest new slots automatically.
observation.formation gives the current formation's half-widths, length and smallest slot gap.
observation.corridorAhead gives the free half-width left and right of your route at distances ahead (40 = open). Followers oscillate around their slots by several metres, so leave margin. Followers cannot fly slower than {fmin} m/s.
You are consulted every {ddt} s of sim time. You may replan every time or keep.
A safe default plan (the textbook controller) is: follow routeRemaining (course waypoints you have not yet passed), then goto the goal. It hits the pose but its flock collides. Beat it, by how you drive and how you shape the flock.
{whatif}{memory}"""

WHATIF = """
WHAT-IF TOOL. Instead of "plan" you may send {"thought": "...", "candidates": [{"name": "...", "plan": [...]}, ...]} with up to 4 candidates. The simulator runs each for the next {h} s and the lowest predicted cost is executed. You will see every candidate's predicted outcome next turn. Use it to test ideas about how the flock reacts.
"""


class OllamaClient:
    def __init__(self, model='llama3.1:8b', host='http://localhost:11434',
                 temperature=0.3, timeout=600, numCtx=8192):
        self.model, self.host = model, host.rstrip('/')
        self.temperature, self.timeout, self.numCtx = temperature, timeout, numCtx

    def chat(self, messages, json_mode=True):
        body = dict(model=self.model, messages=messages, stream=False,
                    options=dict(temperature=self.temperature, num_ctx=self.numCtx))
        if json_mode:
            body['format'] = 'json'
        req = urllib.request.Request(self.host + '/api/chat',
                                     data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())['message']['content']


def _parseJSON(text):
    try:
        return json.loads(text)
    except Exception:
        a, b = text.find('{'), text.rfind('}')
        if a >= 0 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except Exception:
                pass
    return None


def compactObs(obs, maxFans=12):
    """Trim the observation so a small model's context isn't flooded."""
    o = json.loads(json.dumps(obs))
    o.pop('limits', None)
    fl = o['flock']
    if fl['n'] > maxFans:
        fl['positions'] = fl['positions'][:maxFans]
        fl['headings'] = fl['headings'][:maxFans]
        fl['note'] = 'first %d of %d shown' % (maxFans, fl['n'])
    return o


class LLMAgent:
    name = 'llm'

    def __init__(self, client, whatIf=False, whatIfHorizon=12.0, memoryPath=None,
                 historyTurns=4, verbose=True, log=None):
        self.client = client
        self.whatIf = whatIf
        self.whatIfHorizon = whatIfHorizon
        self.memoryPath = memoryPath
        self.historyTurns = historyTurns
        self.verbose = verbose
        self.transcript = [] if log is None else log

    def _lessons(self):
        if not self.memoryPath:
            return []
        try:
            with open(self.memoryPath) as f:
                return json.load(f).get('lessons', [])
        except Exception:
            return []

    def begin(self, env):
        o = env.opts
        lessons = self._lessons()
        mem = ''
        if lessons:
            mem = '\nLESSONS FROM YOUR EARLIER EPISODES:\n' + '\n'.join('- ' + l for l in lessons) + '\n'
        wi = WHATIF.replace('{h}', '%g' % self.whatIfHorizon) if self.whatIf else ''
        self.system = SYSTEM.format(n=env.nFan, R=o.R, smin=env.speedRange[0], smax=env.speedRange[1],
                                    fmin=o.minSpeed, fmax=o.maxSpeed, PR=o.PR, VR=o.VR,
                                    cr=o.collisionRadius, ddt=env.decisionDt,
                                    whatif=wi, memory=mem)
        self.history = []
        self.lastEvents = []
        self.lastWhatIf = None
        self.nCalls, self.nBad, self.wall = 0, 0, 0.0
        self.desc = env.course.description

    def act(self, env, obs):
        user = dict(observation=compactObs(obs),
                    collisionsSinceLastDecision=self.lastEvents[-10:])
        if env.t < 1e-9:
            user['course'] = self.desc
        if self.lastWhatIf:
            user['whatIfResults'] = self.lastWhatIf
            self.lastWhatIf = None
        msg = dict(role='user', content=json.dumps(user))
        msgs = [dict(role='system', content=self.system)] + self.history[-2 * self.historyTurns:] + [msg]
        t0 = time.time()
        try:
            text = self.client.chat(msgs)
        except Exception as e:
            text = '{"error": "%s"}' % str(e).replace('"', "'")
        self.wall += time.time() - t0
        self.nCalls += 1
        reply = _parseJSON(text) or {}
        self.history += [msg, dict(role='assistant', content=text)]
        self.transcript.append(dict(t=env.t, obs=user, reply=text))

        plan = None
        if reply.get('keep'):
            plan = None
        elif self.whatIf and isinstance(reply.get('candidates'), list) and reply['candidates']:
            scored = []
            for c in reply['candidates'][:4]:
                p = c.get('plan') if isinstance(c, dict) else None
                if not isinstance(p, list):
                    continue
                cost, summ = rollout(env, p, self.whatIfHorizon, planDt=0.04)
                scored.append((cost, c.get('name', '?'), p, summ))
            if scored:
                scored.sort(key=lambda s: s[0])
                plan = scored[0][2]
                self.lastWhatIf = [dict(name=n, executed=(i == 0), **s) for i, (_, n, _, s) in enumerate(scored)]
        elif isinstance(reply.get('plan'), list):
            plan = reply['plan']
        else:
            self.nBad += 1
            if env.t < 1e-9:                      # never leave it rudderless
                plan = baselinePlan(env, env.remainingRoute())
        if self.verbose:
            print('  t=%6.1f llm %.1fs  %s' % (env.t, time.time() - t0,
                  (reply.get('thought') or ('keep' if reply.get('keep') else text[:100])).replace('\n', ' ')))
        return plan

    def observeEvents(self, events):
        self.lastEvents = [dict(t=e['t'], type=e['cat'], x=round(e['x'], 1), y=round(e['y'], 1))
                           for e in events]

    def end(self, env, result):
        if not self.memoryPath:
            return
        lessons = self._lessons()
        ask = dict(role='user', content=json.dumps(dict(
            episodeOver=True, result={k: (round(v, 2) if isinstance(v, float) else v) for k, v in result.items()},
            collisionLog=[dict(t=c['t'], type=c['cat'], x=round(c['x'], 1), y=round(c['y'], 1))
                          for c in env.collisions[:30]],
            previousLessons=lessons,
            instruction='Write the updated list of at most 6 short, concrete lessons about how '
                        'to lead THIS flock (what made them collide, what manoeuvres worked). '
                        'Keep lessons that still hold, drop wrong ones. Reply {"lessons": [...]}')))
        try:
            text = self.client.chat([dict(role='system', content=self.system)] + self.history[-4:] + [ask])
            new = (_parseJSON(text) or {}).get('lessons')
            if isinstance(new, list) and new:
                with open(self.memoryPath, 'w') as f:
                    json.dump(dict(lessons=[str(l) for l in new][:6]), f, indent=1)
        except Exception as e:
            print('reflection failed:', e)


class MockLLM:
    """Stands in for Ollama in tests: replies like a cautious model would,
    exercising plan, keep and what-if paths."""
    def __init__(self, whatIf=False):
        self.whatIf, self.n = whatIf, 0

    def chat(self, messages, json_mode=True):
        self.n += 1
        last = json.loads(messages[-1]['content'])
        if last.get('episodeOver'):
            return json.dumps(dict(lessons=['slow down before turns so followers do not overrun']))
        ob = last['observation']
        g = ob['goal']
        tail = [dict(type='goto', x=g['x'], y=g['y'], heading=g['heading'], speed=1.0)]
        route = ob['routeRemaining']
        if route:
            tail = [dict(type='follow', waypoints=route, speed=1.0, start=ob.get('routeLegStart'))] + tail
        if self.n % 3 == 2:
            return '{"keep": true}'
        rows = [r for r in ob.get('corridorAhead', []) if r['ahead'] <= 45]
        free = min([min(r['left'], r['right']) for r in rows] or [40])
        form = dict(type='formation', shape='vee', spacing=12) if free > 37 else \
            dict(type='formation', shape='column', spacing=12, frame='track')
        tail = [form] + tail
        if self.whatIf:
            return json.dumps(dict(thought='compare speeds', candidates=[
                dict(name='slow', plan=[dict(t, speed=0.7) for t in tail]),
                dict(name='fast', plan=[dict(t, speed=1.4) for t in tail])]))
        return json.dumps(dict(thought='follow then goto', plan=tail))
