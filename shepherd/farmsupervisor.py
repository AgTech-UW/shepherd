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
                         passProgress, rolloutMetrics)
from .farmenv import INSTRUCTIONS
from .memory import MemoryStream
from .followers import FOLLOWER_TEXT

# Bumped whenever what the model is shown changes, so results from an older
# prompt are never resumed into a newer run (see FARM_STUDY.md).
#   1  pilot 1
#   2  compass meaning of the offset; which formations fit the corridor ahead
PROMPT_VERSION = 2

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

OFFSET applies only while working a pass: it shifts the leader's line sideways (metres, + = left of travel), within the range the observation gives for each formation. Because left depends on the direction of travel, observation.pass.offsetMeaning says which compass direction a positive and a negative offset move the line on the current pass. It resets at the next pass.

Off the passes, observation.fitsCorridorAhead says, for each formation, whether its machines fit between the obstacles on the route over the next 60 m with 3 m to spare on each side. Speed does not change how wide a formation is.

You are called at the start, at each new phase, when a hazard comes into range, when the corridor ahead changes width, after a collision, and otherwise every {period:g} s. Your choice stays in force until you change it.

Reply with ONE JSON object:
  {{"thought": "<one or two sentences>", "formation": "<name>" or {{custom}}, "speed": <m/s>, "offset": <m>}}
{whatif}{lessons}"""

WHATIF = """
You may instead TEST options first: reply {{"thought": "...", "candidates": [{{"formation": ..., "speed": ..., "offset": ...}}, ...]}} with 2 to 4 options. A simulator will run each from the current state for {h:g} s and report what happens (collisions, machine-seconds on wet ground, field coverage gained, closest machine to each fence). You then make the final choice.
"""


def formationFit(env, reach=60.0, margin=3.0):
    """For each preset: do its machines fit between the obstacles on the
    route over the next `reach` metres, with `margin` to spare on each side?
    The same corridor widths and formation sizes the hand-written rule uses."""
    rows = [r for r in F.corridorAhead(env, rayMax=120.0) if r['ahead'] <= reach]
    fl = min([r['left'] for r in rows] or [120.0])
    fr = min([r['right'] for r in rows] or [120.0])
    out = {}
    for name in FARM_PRESETS:
        lanes, _ = presetLanes(name, env.nFan, env.opts.PR)
        out[name] = bool(max(0.0, lanes.max()) + margin <= fl and max(0.0, -lanes.min()) + margin <= fr)
    return out


def _compass(th):
    d = (np.rad2deg(th) + 360) % 360
    return ['east', 'north-east', 'north', 'north-west', 'west', 'south-west', 'south', 'south-east'][int((d + 22.5) // 45) % 8]


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
        wi = WHATIF.format(h=self.whatIfHorizon) if self.condition == 'whatif' else ''
        self.system = SYSTEM.format(
            N=env.nFan + 1, n=env.nFan, sw=sp.swath, instruction=INSTRUCTIONS[self.instruction]['text'],
            L=env.course.field[1], W=env.course.field[3], gy=sp.gateY, y1=sp.passY(0), y2=sp.passY(1),
            sr=sp.sensorRange, ws=sp.wetSpeed, smin=env.speedRange[0], smax=env.speedRange[1],
            fmin=o.minSpeed, fmax=o.maxSpeed, R=o.R, table=presetTable(sp.swath),
            period=self.triggers.period, whatif=wi, lessons=lessons,
            followerText=FOLLOWER_TEXT['vfield' if env.followerModel == 'vfield' else
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
        for h in hzs:
            h['note'] = note[h['type']]
            h.pop('id', None)
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
            nrm = env.passFrame()[3]
            plus = _compass(np.arctan2(nrm[1], nrm[0]))
            minus = _compass(np.arctan2(-nrm[1], -nrm[0]))
            ob['pass'] = dict(metresDone=round(done, 0), metresLeft=round(left, 0),
                              offsetMeaning='on this pass a positive offset moves the working line %s, '
                                            'a negative offset moves it %s' % (plus, minus),
                              currentOffsetM=round(env.passOffset, 1),
                              allowedOffsetM={k: [round(v, 0) for v in env.offsetLimits(presetLanes(k)[0])]
                                              for k in FARM_PRESETS if env.offsetLimits(presetLanes(k)[0])[0]
                                              <= env.offsetLimits(presetLanes(k)[0])[1]})
        else:
            ob['corridorAhead'] = F.corridorAhead(env)
            ob['fitsCorridorAhead'] = formationFit(env)
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
        text = ('%s I chose %s at %.1f m/s, offset %.0f m. Over the next %.0f s: %d collisions, '
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
        off = float(rep.get('offset', 0.0) or 0.0)
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
            if self.condition == 'whatif' and isinstance(rep.get('candidates'), list) and rep['candidates']:
                preds = []
                for c in rep['candidates'][:4]:
                    name, spec, sp, off = self._resolve(env, c)
                    m = rolloutMetrics(env, farmPlan(env.clone(), spec, sp, off), self.whatIfHorizon)
                    preds.append(dict(formation=name if name != 'custom' else spec, speed=sp, offset=off,
                                      collisions=m['collisions'],
                                      wetGroundMachineSeconds=round(m['sloughExposure'], 0),
                                      fieldWorkedGainPct=round(100 * m['coverageGain'], 2),
                                      closestMachineToFenceM=m['closestToFence']))
                rec['predictions'] = preds
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
        rec['choice'] = None if final is None else dict(formation=final[0], speed=final[2], offset=final[3])
        rec['error'] = err
        self.transcript.append(rec)
        if self.verbose:
            print('    t=%5.0f %-30s %-22s %.1fs  %s' % (env.t, ','.join(why)[:30], rec['choice'] and
                  '%s %.1f %+.0f' % (rec['choice']['formation'] if isinstance(rec['choice']['formation'], str) else 'custom',
                                     rec['choice']['speed'], rec['choice']['offset']), sec,
                  (rep.get('thought') or err or '')[:90].replace('\n', ' ')), flush=True)
        if final is None:
            return None
        name, spec, speed, off = final
        self.pending = dict(situation=situation, choice=(name if isinstance(name, str) else 'custom', speed, off),
                            snap=self._snapshot(env))
        key = (json.dumps(spec, sort_keys=True), round(speed, 2), round(off, 1))
        if key == self.current:
            return None
        self.current = key
        self.choices.append(dict(t=round(env.t, 1), formation=name, speed=speed, offset=off))
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
