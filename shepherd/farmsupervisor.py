"""The language-model supervisor for the farm mission.

It decides what the comparators decide (formation, speed, pass offset), at
the same decision points as the oracle search, from the same perception the
hand-written rule uses (hazard positions, which machine lanes would touch a
hazard, corridor widths), plus the operator's instruction in plain words.
It is never told the scoring weights.

Conditions:
  zeroshot   prompt + observation
  whatif     it may propose up to 4 candidates; the simulator predicts each
             one's outcome over the next 40 s (collisions, wet-ground
             machine-seconds, coverage gained, closest machine to each
             fence); it then makes its final choice from those predictions.
             The simulator does not choose: there is no cost function here.
  lessons    after each training episode it rewrites up to 6 lessons;
             frozen at test
  memory     a generative-agents memory stream (memory.py): outcome records
             written after every decision in training, reflections when
             importance accumulates, top-k retrieval into every prompt;
             frozen at test
"""
import json
import time

import numpy as np

from . import formation as F
from .agents import _parseJSON
from .farmagents import (FARM_PRESETS, Triggers, farmPlan, presetTable, presetLanes,
                         resolveFormation, specLanes, hazardsPassFrame, laneConflicts,
                         passProgress, rolloutMetrics, formationFit)
from .farmenv import INSTRUCTIONS
from .memory import MemoryStream
from .followers import FOLLOWER_TEXT

# Bumped whenever what the model is shown changes, so results from an older
# prompt are never resumed into a newer run (see FARM_STUDY.md).
#   1  pilot 1
#   2  compass meaning of the offset; which formations fit the corridor ahead
#   3  offsets in fixed compass terms; fit check covers the tail, the paths
#      to the new slots and turns; what-if consults the simulator every call
PROMPT_VERSION = 3

PHASE_TEXT = {
    'lane': 'in the lane between the shelterbelts (narrow)',
    'gate': 'going through the field gate (narrow)',
    'approach': 'in the field, lining up for pass 1',
    'pass1': 'working pass 1',
    'headland': 'turning on the east headland',
    'pass2': 'working pass 2',
    'return': 'running down the west headland back to the gate',
    'park': 'out of the gate, parking',
}

# short tags for memory records (the full instruction is in every prompt)
DAY_LABEL = {'wet': 'wet day, protect the soil', 'weather': 'rain coming, get ground done',
             'north': 'keep back from the north fence'}

SYSTEM = """You supervise a team of {N} autonomous farm machines: one LEADER and {n} FOLLOWERS, each pulling a {sw:g} m seed drill. An autopilot steers the leader along the planned route and parks it at the end. You decide three things: the FORMATION the followers hold, the leader's SPEED, and the sideways OFFSET of the pass being worked.

TODAY'S INSTRUCTION FROM THE OPERATOR:
"{instruction}"

THE FARM (x = east, y = north, metres). The field covers x 0-{L:g}, y 0-{W:g} and is fenced, with a gate on the west side at y = {gy:g}. Route: out of the yard down a lane between two shelterbelts (about 20 m wide, with bends) -> through the gate -> pass 1 heading east along y = {y1:g} -> turn on the east headland -> pass 2 heading west along y = {y2:g} -> back down the west headland -> out the gate -> park. Hazards are reported when they come within {sr:g} m:
  slough     a wet low spot. Machines can drive through it, but they slow to {ws:g} m/s and rut and compact the soil.
  rock pile, power pole   solid.
Nothing may touch a fence, tree, rock, pole or another machine.

THE MACHINES. Leader speed {smin:g}-{smax:g} m/s. Followers: {fmin:g}-{fmax:g} m/s, turning radius {R:g} m. {followerText}

FORMATIONS. A follower's "lane" is its sideways position relative to the leader (+ = left of travel). Ground is worked where an implement passes, so wide formations get more done per pass and single file works one {sw:g} m strip.
{table}
"body" formations are fixed to the leader's heading and swing sideways in turns; the two columns follow the leader's actual track, so they are the ones that fit lanes, gates and headland turns. You may give a custom formation instead: {{"shape": "custom", "slots": [[along, lane], ... {n} pairs], "frame": "body"|"track"}}, along < 0 behind the leader, keeping every pair of machines more than 10 m apart.

OFFSET applies only while working a pass: it moves the leader's line {offPlus} (+) or {offMinus} (-) of the planned pass line, in metres, on every pass, whichever way the machines are travelling, within the range the observation gives for each formation. It resets at the next pass. Lanes and the lineLeft/lineRight formations are relative to the direction of travel; observation.pass.leftOfTravelIs says which compass side is left on the current pass.

observation.formationsThatFit says, for each formation, whether it can be taken up now: every follower can get to its new place without crossing a fence, tree, post, rock or pole; off the passes, the machines beside the leader's track fit between the obstacles on the route over the next 60 m with 4 m to spare; and, for the body formations (line, lineLeft, lineRight, vee), the route does not turn within the next 60 m. Speed does not change how wide a formation is.

You are called at the start, at each new phase, when a hazard comes into range, when the corridor ahead changes width, when the formations that fit change, after a collision, and otherwise every {period:g} s. Your choice stays in force until you change it.

{reply}{lessons}"""

REPLY = """Reply with ONE JSON object:
  {"thought": "<one or two sentences>", "formation": "<name>" or {custom}, "speed": <m/s>, "offset": <m>}
"""

WHATIF = """At every call you first TEST options: reply {{"thought": "...", "candidates": [{{"formation": ..., "speed": ..., "offset": ...}}, ...]}} with 1 to 4 options. A simulator runs each one from the current state for {h:g} s and reports what happens (collisions, machine-seconds on wet ground, field coverage gained, closest machine to each fence). You then make the final choice with ONE JSON object:
  {{"thought": "<one or two sentences>", "formation": "<name>" or {{custom}}, "speed": <m/s>, "offset": <m>}}
"""


def _compass(th):
    d = (np.rad2deg(th) + 360) % 360
    return ['east', 'north-east', 'north', 'north-west', 'west', 'south-west', 'south', 'south-east'][int((d + 22.5) // 45) % 8]


def _offsetAxis(course):
    """Compass words for + and - offsets: north/south when the passes run
    east-west, east/west when they run north-south."""
    legs = list(course.passLegs.values())
    if not legs:
        return 'north', 'south'
    a, b = np.asarray(legs[0][0], float), np.asarray(legs[0][1], float)
    return ('north', 'south') if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else ('east', 'west')


def _offsetSign(env):
    """+1 if left of travel is the + compass direction on the current pass,
    -1 if it is the - direction, 0 off the passes. The model speaks in
    compass offsets; the env takes offsets to the left of travel; the
    conversion is the same both ways."""
    fr = env.passFrame()
    if fr is None:
        return 0.0
    n = np.asarray(fr[3], float)
    plus, _ = _offsetAxis(env.course)
    return float(np.sign(n[1] if plus == 'north' else n[0]) or 1.0)


class FarmSupervisor:
    name = 'llm'

    def __init__(self, client, instruction='wet', condition='zeroshot', lessonsPath=None,
                 memoryPath=None, embedder=None, learn=False, period=30.0, whatIfHorizon=40.0,
                 historyTurns=2, memoryK=6, verbose=True):
        self.client = client
        self.instruction = instruction
        self.condition = condition
        self.lessonsPath, self.memoryPath = lessonsPath, memoryPath
        self.embedder = embedder
        self.learn = learn
        self.triggers = Triggers(period=period)
        self.whatIfHorizon = whatIfHorizon
        self.historyTurns = historyTurns
        self.memoryK = memoryK
        self.verbose = verbose

    # ---- setup ----
    def _lessons(self):
        try:
            with open(self.lessonsPath) as f:
                return json.load(f).get('lessons', [])
        except Exception:
            return []

    def begin(self, env):
        sp, o = env.course.spec, env.opts
        lessons = ''
        if self.condition == 'lessons':
            ls = self._lessons()
            if ls:
                lessons = '\nLESSONS FROM YOUR EARLIER DAYS IN THE FIELD:\n' + '\n'.join('- ' + l for l in ls) + '\n'
        if self.condition == 'memory':
            self.memory = MemoryStream.load(self.memoryPath, self.embedder)
            lessons = ('\nYou have a memory of earlier days in the field. The most relevant memories for the '
                       'current situation are included in each observation as "relevantMemories".\n')
        reply = WHATIF.format(h=self.whatIfHorizon) if self.condition == 'whatif' else REPLY
        offPlus, offMinus = _offsetAxis(env.course)
        self.system = SYSTEM.format(
            N=env.nFan + 1, n=env.nFan, sw=sp.swath, instruction=INSTRUCTIONS[self.instruction]['text'],
            L=env.course.field[1], W=env.course.field[3], gy=sp.gateY, y1=sp.passY(0), y2=sp.passY(1),
            sr=sp.sensorRange, ws=sp.wetSpeed, smin=env.speedRange[0], smax=env.speedRange[1],
            fmin=o.minSpeed, fmax=o.maxSpeed, R=o.R, table=presetTable(sp.swath),
            period=self.triggers.period, reply=reply, lessons=lessons, offPlus=offPlus, offMinus=offMinus,
            followerText=FOLLOWER_TEXT[env.followerModel if env.followerModel in ('vfield', 'reflex') else
                                       ('damped' if env.opts.CF == 0.0 and env.opts.VR >= 20 else 'paper')])
        self.triggers.reset()
        self.history = []
        self.current = None
        self.pending = None
        self.nCalls = self.nBad = 0
        self.wall = 0.0
        self.choices = []
        self.transcript = []
        self._nCol = 0

    # ---- perception ----
    def _situation(self, env, obsHz):
        """Short text of the situation, used as the memory query and record."""
        hz = ', '.join('%s %.0f m ahead %.0f m %s' % (h['type'], h['aheadM'], abs(h['leftM']),
                       'left' if h['leftM'] > 0 else 'right') for h in obsHz) or 'no hazards in range'
        return '[%s] Phase %s, heading %s, %s.' % (
            DAY_LABEL.get(self.instruction, self.instruction), env.phase(), _compass(env.th), hz)

    def _observation(self, env, why):
        ph = env.phase()
        spec = env.formation
        lanes, length = specLanes(spec, env.nFan, env.opts.PR) if spec.get('shape') != 'none' else (np.zeros(1), 0.0)
        fi = env.formationInfo()
        hzs = env.hazardsInRange()
        inPass = env.passFrame() is not None
        if inPass:
            pf = {round(h['along'], 3): h for h in hazardsPassFrame(env, behind=200, ahead=200)}
            for h in hzs:
                hp = next((v for v in hazardsPassFrame(env, behind=200, ahead=200)
                           if v['type'] == h['type'] and abs(v['r'] - h['radiusM']) < 0.06
                           and abs(v['along'] - h['aheadM']) < 1.5), None)
                if hp is not None:
                    hit = laneConflicts(lanes, env.passOffset, hp, env.course.spec.swath)
                    h['yourLanesThatWouldTouchIt'] = [round(x - env.passOffset, 1) for x in hit]
        note = {'slough': 'wet ground: passable, slows machines, ruts the soil',
                'rock': 'solid', 'pole': 'solid'}
        world = {i: hz for i, hz in enumerate(env.course.hazards)}
        for h in hzs:
            h['note'] = note[h['type']]
            w = world.get(h.pop('id', None))
            if w is not None:
                h['x'], h['y'] = round(float(w['x']), 0), round(float(w['y']), 0)
        ax = np.concatenate([[env.x], env.fanX])
        ay = np.concatenate([[env.y], env.fanY])
        ob = dict(
            instruction=INSTRUCTIONS[self.instruction]['text'],
            calledBecause=why, timeS=round(env.t, 0),
            phase='%s: %s' % (ph, PHASE_TEXT.get(ph, ph)),
            leader=dict(x=round(env.x, 1), y=round(env.y, 1), heading=_compass(env.th),
                        speed=round(env.speed, 2)),
            formation=dict(current=self._name(spec), lanes=sorted(set(round(float(x), 1) for x in lanes)),
                           lengthM=round(length, 0),
                           meanDistanceOfFollowersFromSlotsM=fi.get('meanSlotError')),
            hazardsInRange=hzs,
            machinesSpanX=[round(float(ax.min()), 0), round(float(ax.max()), 0)],
            machinesSpanY=[round(float(ay.min()), 0), round(float(ay.max()), 0)],
            progress=dict(fieldWorkedPct=round(100 * env.coverage(), 1),
                          wetGroundMachineSeconds=round(env.sloughExposure, 0),
                          collisionsSoFar=len(env.collisions)),
            collisionsSinceLastCall=[dict(t=round(c['t'], 0), type=c['cat'], x=round(c['x'], 0),
                                          y=round(c['y'], 0)) for c in env.collisions[self._nCol:]][:6],
        )
        if inPass:
            done, left = passProgress(env)
            a, b, u, nrm = env.passFrame()
            sg = _offsetSign(env)
            plus, minus = _offsetAxis(env.course)
            allowed = {}
            for k in FARM_PRESETS:
                lo, hi = env.offsetLimits(presetLanes(k)[0])
                if lo <= hi:
                    lo, hi = sorted((sg * lo, sg * hi))
                    allowed[k] = [round(lo, 0) + 0.0, round(hi, 0) + 0.0]
            ob['pass'] = dict(heading=_compass(np.arctan2(u[1], u[0])),
                              leftOfTravelIs=_compass(np.arctan2(nrm[1], nrm[0])),
                              metresDone=round(done, 0), metresLeft=round(left, 0),
                              offsetMeaning='+ moves the working line %s, - moves it %s' % (plus, minus),
                              currentOffsetM=round(sg * env.passOffset, 1) + 0.0,
                              allowedOffsetM=allowed)
        else:
            ob['corridorAhead'] = F.corridorAhead(env)
        ob['formationsThatFit'] = formationFit(env)
        return ob, self._situation(env, hzs)

    @staticmethod
    def _name(spec):
        for k, v in FARM_PRESETS.items():
            if v == spec:
                return k
        return spec

    # ---- decisions ----
    def _snapshot(self, env):
        return dict(t=env.t, col=len(env.collisions), slough=env.sloughExposure, cov=env.coverage())

    def _closePending(self, env):
        """Turn the last decision and what followed into a memory record."""
        p = self.pending
        if p is None or self.condition != 'memory' or not self.learn:
            return
        s0, s1 = p['snap'], self._snapshot(env)
        dc, ds = s1['col'] - s0['col'], s1['slough'] - s0['slough']
        dcov = 100 * (s1['cov'] - s0['cov'])
        text = ('%s I chose %s at %.1f m/s, offset %+.0f m. Over the next %.0f s: %d collisions, '
                '%.0f machine-seconds on wet ground, %.1f%% more of the field worked.'
                % (p['situation'], p['choice'][0], p['choice'][1], p['choice'][2], s1['t'] - s0['t'],
                   dc, ds, dcov))
        importance = float(min(10.0, 2.0 + 3.0 * dc + ds / 15.0))
        self.memory.add(text, importance)
        self.pending = None

    def _chat(self, msgs):
        t0 = time.time()
        try:
            text = self.client.chat(msgs)
        except Exception as e:
            text = '{"error": "%s"}' % str(e).replace('"', "'")
        self.wall += time.time() - t0
        self.nCalls += 1
        return text, time.time() - t0

    def _resolve(self, env, rep):
        name, spec = resolveFormation(rep.get('formation'), env.nFan, env.opts.PR)
        speed = env.clampSpeed(float(rep.get('speed', env.vNominal)))
        off = float(rep.get('offset', 0.0) or 0.0) * _offsetSign(env)   # compass -> left of travel
        return name, spec, speed, off

    def act(self, env, obs):
        why = self.triggers.check(env)
        if not why:
            return None
        if not hasattr(self, '_nCol'):
            self._nCol = 0
        self._closePending(env)
        ob, situation = self._observation(env, why)
        self._nCol = len(env.collisions)
        if self.condition == 'memory':
            self.memory.tick()
            ob['relevantMemories'] = self.memory.retrieve(situation, self.memoryK)
        msg = dict(role='user', content=json.dumps(ob))
        msgs = [dict(role='system', content=self.system)] + self.history[-2 * self.historyTurns:] + [msg]
        text, sec = self._chat(msgs)
        rep = _parseJSON(text) or {}
        rec = dict(t=round(env.t, 1), why=why, observation=ob, reply=text, seconds=round(sec, 2))
        final, err = None, None
        try:
            if self.condition == 'whatif':
                # every call goes through the simulator; a plain choice is
                # tested as the only candidate
                cands = rep.get('candidates')
                if not (isinstance(cands, list) and cands):
                    cands = [rep] if 'formation' in rep else []
                preds = []
                sg = _offsetSign(env)
                for c in cands[:4]:
                    if not isinstance(c, dict):
                        continue
                    try:
                        name, spec, sp, off = self._resolve(env, c)
                    except Exception:
                        continue
                    sim = env.clone()
                    plan = farmPlan(sim, spec, sp, off)
                    applied = sim.passOffset                     # after clipping to the allowed range
                    m = rolloutMetrics(env, plan, self.whatIfHorizon)
                    preds.append(dict(formation=name if name != 'custom' else spec, speed=sp,
                                      offset=round(sg * applied, 1) + 0.0,
                                      collisions=m['collisions'],
                                      wetGroundMachineSeconds=round(m['sloughExposure'], 0),
                                      fieldWorkedGainPct=round(100 * m['coverageGain'], 2),
                                      closestMachineToFenceM=m['closestToFence']))
                rec['predictions'] = preds
            if self.condition == 'whatif' and rec['predictions']:
                ask = dict(role='user', content=json.dumps(dict(
                    simulatorPredictions=preds, seconds=self.whatIfHorizon,
                    instruction='Make your final choice. Reply {"thought": "...", "formation": ..., "speed": ..., "offset": ...}')))
                text2, sec2 = self._chat(msgs + [dict(role='assistant', content=text), ask])
                rec['reply2'], rec['seconds2'] = text2, round(sec2, 2)
                rep = _parseJSON(text2) or {}
                text = text + '\n' + text2
            if 'formation' in rep:
                final = self._resolve(env, rep)
        except Exception as e:
            err = str(e)
        if final is None:
            self.nBad += 1
            if self.current is None:                       # never start rudderless
                final = ('column', dict(FARM_PRESETS['column']), env.vNominal, 0.0)
        self.history += [msg, dict(role='assistant', content=text)]
        sg = _offsetSign(env)
        rec['choice'] = None if final is None else dict(formation=final[0], speed=final[2], offset=final[3],
                                                        offsetCompass=round(sg * final[3], 1) + 0.0)
        rec['error'] = err
        self.transcript.append(rec)
        if self.verbose:
            print('    t=%5.0f %-30s %-22s %.1fs  %s' % (env.t, ','.join(why)[:30], rec['choice'] and
                  '%s %.1f %+.0f' % (rec['choice']['formation'] if isinstance(rec['choice']['formation'], str) else 'custom',
                                     rec['choice']['speed'], rec['choice']['offsetCompass']), sec,
                  (rep.get('thought') or err or '')[:90].replace('\n', ' ')), flush=True)
        if final is None:
            return None
        name, spec, speed, off = final
        self.pending = dict(situation=situation, choice=(name if isinstance(name, str) else 'custom', speed, sg * off),
                            snap=self._snapshot(env))
        key = (json.dumps(spec, sort_keys=True), round(speed, 2), round(off, 1))
        if key == self.current:
            return None
        self.current = key
        self.choices.append(dict(t=round(env.t, 1), formation=name, speed=speed, offset=off,
                                 offsetCompass=round(sg * off, 1)))
        return farmPlan(env, spec, speed, off)

    # ---- learning ----
    def end(self, env, result):
        if not self.learn:
            return
        if self.condition == 'memory':
            self._closePending(env)
            if self.memory.needsReflection() or self.memory.sinceReflection > 0:
                ins = self.memory.reflect(lambda m: self._chat(m)[0], self.system)
                self.transcript.append(dict(reflection=ins))
            self.memory.save(self.memoryPath)
        elif self.condition == 'lessons':
            ask = dict(role='user', content=json.dumps(dict(
                dayOver=True, instruction=INSTRUCTIONS[self.instruction]['text'],
                result=dict(collisions=result['collisions'], fieldWorkedPct=round(100 * result['coverage'], 1),
                            wetGroundMachineSeconds=round(result['sloughExposure'], 0), timeS=round(result['time'], 0)),
                yourChoices=self.choices[:40],
                collisionLog=[dict(t=round(c['t'], 0), type=c['cat'], x=round(c['x'], 0), y=round(c['y'], 0))
                              for c in env.collisions[:25]],
                previousLessons=self._lessons(),
                task='Update your lessons: at most 6 short, general rules about choosing formation, speed and '
                     'offset that would help on a different field and a different day. Keep the ones that '
                     'still hold, fix or drop wrong ones. Reply {"lessons": [...]}')))
            text, _ = self._chat([dict(role='system', content=self.system), ask])
            new = (_parseJSON(text) or {}).get('lessons')
            if isinstance(new, list) and new:
                with open(self.lessonsPath, 'w') as f:
                    json.dump(dict(lessons=[str(x) for x in new][:6]), f, indent=1)
            self.transcript.append(dict(lessonsReply=text))


class MockFarmLLM:
    """Offline stand-in: a crude width-and-hazard policy that speaks the
    protocol (choices, candidates, final choices, lessons, insights) and is
    sometimes malformed on purpose."""
    def __init__(self, seed=0):
        self.n = 0
        self.rng = np.random.default_rng(seed)

    def chat(self, messages, json_mode=True):
        self.n += 1
        last = messages[-1]['content']
        if '"insights"' in last:
            return json.dumps(dict(insights=['fold to single file before gates and headlands',
                                             'fold away from sloughs on wet days', 'thread rocks between lanes']))
        try:
            d = json.loads(last)
        except Exception:
            d = {}
        if d.get('dayOver'):
            return json.dumps(dict(lessons=['single file in lanes and turns', 'fold the side nearest a slough']))
        if 'simulatorPredictions' in d:
            p = min(d['simulatorPredictions'], key=lambda c: (c['collisions'], -c['fieldWorkedGainPct']))
            return json.dumps(dict(thought='fewest collisions', formation=p['formation'], speed=p['speed'], offset=p['offset']))
        if self.n % 9 == 0:
            return 'Let me think about this.'               # malformed
        ph = d.get('phase', '')
        if not ph.startswith('pass'):
            return json.dumps(dict(thought='narrow ground', formation='column', speed=1.0, offset=0))
        f = 'line'
        for h in d.get('hazardsInRange', []):
            if h['type'] == 'slough' and h.get('yourLanesThatWouldTouchIt'):
                f = 'lineRight' if h['leftM'] > 0 else 'lineLeft'
        left = d.get('pass', {}).get('metresLeft', 100)
        if left < 35:
            f = 'column'
        if 'candidates' in messages[0]['content'] and 'TEST options' in messages[0]['content'] and self.n % 2:
            return json.dumps(dict(thought='test', candidates=[dict(formation=f, speed=1.0, offset=0),
                                                                dict(formation='column2', speed=1.0, offset=0)]))
        return json.dumps(dict(thought='work', formation=f, speed=1.1, offset=0))
