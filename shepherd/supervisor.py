"""LLM as a supervisory planner over the physics controller.

The language model never steers. The leader follows the route with the
paper's vector-field guidance and lands the goal pose with a Dubins path,
exactly as the baseline does. The model chooses only what the hand-written
rule (AdaptiveFormationAgent) and the rollout search (FormationSearchAgent)
choose: the flock's formation and the leader's speed. Same action space,
three different deciders. That is the comparison.

It is consulted at decision points, not on a clock:
  start; a new route leg; the corridor ahead changing width class; a
  collision since the last call; otherwise at most every `maxGap` seconds.
That keeps an episode to roughly 10-20 calls, so a local model is usable.

Conditions:
  zero-shot      the prompt and the observation, nothing else
  whatIf         the model proposes up to 4 (formation, speed) candidates;
                 the simulator rolls each forward `whatIfHorizon` seconds
                 and the cheapest runs. It sees every prediction next call.
  memory         after each episode the model writes up to 6 lessons,
                 which are prepended to the prompt of the next episode.
"""
import json
import time

import numpy as np

from . import formation as F
from .agents import (PRESETS, baselinePlan, rollout, OllamaClient, _parseJSON)
from .env import CATS


CLASSES = (('open', 30.0), ('wide', 12.0), ('narrow', 0.0))   # free half-width ahead, m


def _freeWidth(env, reach=45.0):
    rows = [r for r in F.corridorAhead(env) if r['ahead'] <= reach]
    return min([min(r['left'], r['right']) for r in rows] or [40.0])


def _widthClass(w, prev=None, hyst=1.5):
    """Width class with hysteresis, so a corridor sitting on a boundary does
    not trigger a call every step."""
    def cls(x):
        return next(n for n, lo in CLASSES if x >= lo)
    c = cls(w)
    if prev is None or c == prev:
        return c
    # only change if it still changes with the boundary moved against it
    lo = dict(CLASSES)[prev]
    hi = dict(CLASSES).get(CLASSES[[n for n, _ in CLASSES].index(prev) - 1][0], np.inf) \
        if prev != CLASSES[0][0] else np.inf
    return c if (w < lo - hyst or w >= hi + hyst) else prev


def presetTable(n, P):
    rows = []
    for name, spec in PRESETS.items():
        sl = F.slotsFor(spec, n, P)
        if sl is None:
            rows.append('  %-10s all followers chase YOU (no slots). Crowded: they jam at %g m around you.' % (name, P))
            continue
        g = F.geometry(*sl, P)
        rows.append('  %-10s %-34s half-width L/R %4.1f/%4.1f m, length %5.1f m, smallest slot gap %4.1f m'
                    % (name, _describe(spec), g['leftHalfWidth'], g['rightHalfWidth'], g['length'], g['minSlotGap']))
    return '\n'.join(rows)


def _describe(spec):
    s = spec['shape']
    fr = ' (track frame)' if spec.get('frame') == 'track' else ''
    if s == 'vee':
        return 'vee, 45 deg, spacing %g%s' % (spec['spacing'], fr)
    if s == 'column':
        return 'column %d abreast, spacing %g%s' % (spec.get('ranks', 1), spec['spacing'], fr)
    return s + fr


SYSTEM = """You supervise a robot LEADER that leads {n} turn-limited follower robots ("boids") through a course. An autopilot already steers the leader along the route and parks it exactly on the goal pose. You decide only two things: the FORMATION the followers hold, and the leader's SPEED.

PHYSICS (fixed)
- Leader: speed {smin}-{smax} m/s, minimum turning radius {R} m.
- Followers: speed {fmin}-{fmax} m/s (they cannot go slower than {fmin}), minimum turning radius {R} m. Each follower is pulled toward its own slot in the formation, pushed away from anything within PR={PR} m, and aligns with neighbours within VR={VR} m.
- Followers CANNOT see walls. A follower that drifts out of a corridor hits the wall. Because they cannot turn tightly, a displaced follower swings in loops around its slot before settling; loops are wider when the formation is disturbed (turns, formation changes, crowding).
- Slots closer than PR={PR} m fight each other (separation pushes them out of their slots).

FORMATIONS you can choose (computed for {n} followers):
{table}
"body" slots sit behind the leader's current heading and swing sideways when the leader turns. "track" slots sit on the path the leader actually drove, so they follow round corners.
You may instead give a custom spec: {{"shape": "vee"|"echelon"|"column"|"line", "spacing": m, "openingDeg": 0-90, "ranks": k, "frame": "body"|"track"}}.

SCORE (lower is better): 10 per collision (follower-wall, follower-follower, follower-leader) + 25 per follower lost (>40 m from its slot) + 0.02 per second.

You are called at decision points: the start, a new route leg, a change in corridor width ahead, after a collision, or every {gap:g} s. Between calls your choice stays in force.
observation.corridorAhead gives the free half-width (m) left and right of the route at distances ahead; 40 means open.

Reply with ONE JSON object:
  {{"thought": "<one or two sentences>", "formation": "<name>" or {{custom spec}}, "speed": <number>}}
{whatif}{memory}"""

WHATIF = """
Instead you may test ideas: {{"thought": "...", "candidates": [{{"formation": ..., "speed": ...}}, ...]}} with 2 to 4 candidates. A simulator runs each for {h:g} s from the current state and the cheapest is applied. You will see every candidate's predicted collisions next call.
"""


class LLMSupervisor:
    name = 'llmsup'

    def __init__(self, client, whatIf=False, whatIfHorizon=25.0, memoryPath=None,
                 maxGap=20.0, minCollisionGap=6.0, historyTurns=3, verbose=True, learn=True):
        self.client = client
        self.whatIf, self.whatIfHorizon = whatIf, whatIfHorizon
        self.memoryPath = memoryPath
        self.maxGap = maxGap
        self.minCollisionGap = minCollisionGap
        self.learn = learn               # False: read lessons, never rewrite them
        self.historyTurns = historyTurns
        self.verbose = verbose
        self.transcript = []

    # ---- memory ----
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
        mem = ('\nLESSONS FROM YOUR EARLIER RUNS:\n' + '\n'.join('- ' + l for l in lessons) + '\n') if lessons else ''
        wi = WHATIF.format(h=self.whatIfHorizon) if self.whatIf else ''
        self.system = SYSTEM.format(n=env.nFan, smin=env.speedRange[0], smax=env.speedRange[1], R=o.R,
                                    fmin=o.minSpeed, fmax=o.maxSpeed, PR=o.PR, VR=o.VR,
                                    table=presetTable(env.nFan, o.PR), gap=self.maxGap,
                                    whatif=wi, memory=mem)
        self.history = []
        self.lastCall = -np.inf
        self.lastLeg = -1
        self.lastWidth = None
        self.lastNCol = 0
        self.current = None              # (formationSpec, speed)
        self.lastWhatIf = None
        self.nCalls = self.nBad = 0
        self.wall = 0.0
        self.choices = []
        self.transcript = []

    # ---- when to ask ----
    def _trigger(self, env):
        w = _widthClass(_freeWidth(env), self.lastWidth)
        reasons = []
        if self.current is None:
            reasons.append('start')
        else:
            if env.routeIdx != self.lastLeg:
                reasons.append('new route leg')
            if self.lastWidth is not None and w != self.lastWidth:
                reasons.append('corridor ahead changed from %s to %s' % (self.lastWidth, w))
            if len(env.collisions) > self.lastNCol and env.t - self.lastCall >= self.minCollisionGap:
                reasons.append('collision')
            if env.t - self.lastCall >= self.maxGap:
                reasons.append('periodic')
        self.lastLeg, self.lastWidth = env.routeIdx, w
        return reasons

    def _observation(self, env, reasons):
        new = env.collisions[self.lastNCol:]
        ob = env.course.obstacles
        nearWall, minClear = 0, None
        if len(ob) and env.nFan:
            gap = (np.hypot(env.fanX[:, None] - ob[None, :, 0], env.fanY[:, None] - ob[None, :, 1])
                   - ob[None, :, 2] - env.opts.bodyRadius).min(1)
            nearWall, minClear = int(np.sum(gap < 3.0)), round(float(gap.min()), 1)
        fi = env.formationInfo()
        gx, gy, _ = env.course.goal
        return dict(
            reason=reasons, t=round(env.t, 1),
            leader=dict(x=round(env.x, 1), y=round(env.y, 1),
                        headingDeg=round(float(np.rad2deg(env.th)), 0), speed=round(env.speed, 2)),
            goalDistance=round(float(np.hypot(gx - env.x, gy - env.y)), 1),
            routeLeg='%d of %d' % (env.routeIdx + 1, len(env.course.route) + 1),
            corridorAhead=F.corridorAhead(env),
            formation=dict(current=self._name(fi['spec']), halfWidthLR=[fi['leftHalfWidth'], fi['rightHalfWidth']],
                           length=fi['length'], minSlotGap=fi['minSlotGap'],
                           meanDistanceFromSlot=fi.get('meanSlotError')),
            followersWithin3mOfWall=nearWall, closestFollowerToWall=minClear,
            collisionsSoFar={c: sum(1 for k in env.collisions if k['cat'] == c) for c in CATS if c != 'leader-obstacle'},
            collisionsSinceLastCall=[dict(t=k['t'], type=k['cat'], x=round(k['x'], 0), y=round(k['y'], 0)) for k in new[:6]],
        )

    @staticmethod
    def _name(spec):
        for k, v in PRESETS.items():
            if v == spec:
                return k
        return spec

    def _resolve(self, env, f, s):
        spec = dict(PRESETS[f]) if isinstance(f, str) and f in PRESETS else dict(f) if isinstance(f, dict) else None
        if spec is None:
            raise ValueError('unknown formation %r' % (f,))
        spec.pop('type', None)
        F.slotsFor(spec, env.nFan, env.opts.PR)            # validates
        return spec, env.clampSpeed(s if s is not None else env.vNominal)

    def _plan(self, env, spec, speed):
        return [dict(type='formation', **spec)] + baselinePlan(env, env.remainingRoute(), speed)

    def act(self, env, obs):
        reasons = self._trigger(env)
        if not reasons:
            return None
        user = self._observation(env, reasons)
        self.lastNCol = len(env.collisions)
        self.lastCall = env.t
        if self.lastWhatIf:
            user['whatIfResults'] = self.lastWhatIf
            self.lastWhatIf = None
        if env.t < 1e-9:
            user['course'] = env.course.description
        msg = dict(role='user', content=json.dumps(user))
        msgs = [dict(role='system', content=self.system)] + self.history[-2 * self.historyTurns:] + [msg]
        t0 = time.time()
        try:
            text = self.client.chat(msgs)
        except Exception as e:
            text = '{"error": "%s"}' % str(e).replace('"', "'")
        dtCall = time.time() - t0
        self.wall += dtCall
        self.nCalls += 1
        reply = _parseJSON(text) or {}
        self.history += [msg, dict(role='assistant', content=text)]

        choice, err = None, None
        try:
            if self.whatIf and isinstance(reply.get('candidates'), list) and reply['candidates']:
                scored = []
                for c in reply['candidates'][:4]:
                    spec, sp = self._resolve(env, c.get('formation'), c.get('speed'))
                    cost, summ = rollout(env, self._plan(env, spec, sp), self.whatIfHorizon, planDt=0.05)
                    scored.append((cost, spec, sp, summ))
                scored.sort(key=lambda r: r[0])
                choice = scored[0][1:3]
                self.lastWhatIf = [dict(formation=self._name(sp_), speed=v, executed=(i == 0),
                                        predictedCollisions=sm['predictedCollisions'], fansLost=sm['fansLost'],
                                        cost=sm['cost']) for i, (_, sp_, v, sm) in enumerate(scored)]
            elif 'formation' in reply:
                choice = self._resolve(env, reply.get('formation'), reply.get('speed'))
        except Exception as e:
            err = str(e)
        if choice is None:
            self.nBad += 1
            if self.current is None:                     # never start rudderless
                choice = (dict(PRESETS['none']), env.vNominal)
        rec = dict(t=round(env.t, 1), reasons=reasons, reply=text, error=err, seconds=round(dtCall, 2),
                   choice=None if choice is None else dict(formation=self._name(choice[0]), speed=choice[1]))
        self.transcript.append(dict(observation=user, **rec))
        if self.verbose:
            print('  t=%6.1f [%s] %.1fs -> %s | %s' % (env.t, ', '.join(reasons), dtCall,
                  rec['choice'], (reply.get('thought') or err or text[:80]).replace('\n', ' ')[:140]))
        if choice is None or (self.current is not None and choice[0] == self.current[0]
                              and abs(choice[1] - self.current[1]) < 1e-9):
            return None
        self.current = choice
        self.choices.append(dict(t=round(env.t, 1), formation=self._name(choice[0]), speed=choice[1]))
        return self._plan(env, *choice)

    def end(self, env, result):
        if not self.memoryPath or not self.learn:
            return
        lessons = self._lessons()
        ask = dict(role='user', content=json.dumps(dict(
            episodeOver=True,
            course=env.course.name,
            result={k: (round(v, 2) if isinstance(v, float) else v) for k, v in result.items()
                    if k in ('collisions', 'fan-fan', 'fan-leader', 'fan-obstacle', 'fansLost', 'time', 'score')},
            yourChoices=self.choices,
            collisionLog=[dict(t=c['t'], type=c['cat'], x=round(c['x'], 0), y=round(c['y'], 0)) for c in env.collisions[:25]],
            previousLessons=lessons,
            instruction='Update your lessons: at most 6 short, concrete rules about which formation and speed to '
                        'use where, and what caused collisions. Keep lessons that still hold, fix or drop wrong ones. '
                        'Reply {"lessons": [...]}')))
        try:
            text = self.client.chat([dict(role='system', content=self.system), ask])
            new = (_parseJSON(text) or {}).get('lessons')
            if isinstance(new, list) and new:
                with open(self.memoryPath, 'w') as f:
                    json.dump(dict(lessons=[str(l) for l in new][:6]), f, indent=1)
        except Exception as e:
            print('reflection failed:', e)


class MockSupervisorLLM:
    """Plumbing test without Ollama: a width rule that sometimes proposes
    candidates and sometimes returns junk."""
    def __init__(self):
        self.n = 0

    def chat(self, messages, json_mode=True):
        self.n += 1
        last = json.loads(messages[-1]['content'])
        if last.get('episodeOver'):
            return json.dumps(dict(lessons=['use column in narrow corridors', 'keep speed near 1.2']))
        if self.n % 7 == 0:
            return 'I think a column would be good.'          # malformed on purpose
        rows = [r for r in last.get('corridorAhead', []) if r['ahead'] <= 45]
        w = min([min(r['left'], r['right']) for r in rows] or [40])
        f = 'vee' if w >= 37 else 'column2' if w >= 9 else 'column'
        if 'candidates' in messages[0]['content'] and 'WHATIF' not in messages[0]['content'] and self.n % 2:
            return json.dumps(dict(thought='test speeds', candidates=[dict(formation=f, speed=1.0),
                                                                      dict(formation=f, speed=1.25)]))
        return json.dumps(dict(thought='width %.0f -> %s' % (w, f), formation=f, speed=1.2))
