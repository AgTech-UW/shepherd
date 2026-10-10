"""Follower models compared on the corridors and the farm.

    python -m shepherd.compare_followers [nSeeds] [out.csv]

Models (rules of separation/alignment/cohesion unchanged in all three):
  paper    the published boid parameters and the slot spring
  damped   boid parameters that damp the spring (VR 20, CF 0, AF 4,
           slot pull 2, 1/r separation)
  vfield   slot spring replaced by vector-field slot following (followers.py),
           1/r separation, followers may slow to 0.3 m/s, slots re-matched
           every 2 s
Weaving: mean absolute heading rate of the followers, deg/s, over the run.
"""
import csv, sys, time
import os
import multiprocessing as mp
import numpy as np

from grainframe import SimOptions
from .courses import get_course
from .env import ShepherdEnv
from .agents import baselinePlan, PRESETS
from .farm import FarmSpec, buildFarm
from .farmenv import FarmEnv
from .farmagents import Schedule, FarmRule
from .run import runEpisode

from .followers import FOLLOWER_MODELS as MODELS


class _Fixed:
    def __init__(self): pass
    def begin(self, env): self.sent = False
    def act(self, env, obs):
        if self.sent: return None
        self.sent = True; return baselinePlan(env, env.remainingRoute())
    def end(self, env, r): pass


def _instrument(env):
    """Wrap stepDt to accumulate weaving and slot error."""
    st = dict(h=np.arctan2(env.fanVy, env.fanVx), turn=0.0, n=0, err=0.0, ne=0)
    orig = env.stepDt
    def step():
        orig()
        h = np.arctan2(env.fanVy, env.fanVx)
        st['turn'] += float(np.mean(np.abs(np.angle(np.exp(1j * (h - st['h'])))))) ; st['n'] += 1
        st['h'] = h
        if env.steps % 50 == 0 and env.slotsTo is not None:
            sl = env.currentSlots(); a, l = env._bodyFrame()
            st['err'] += float(np.mean(np.hypot(a - sl[0], l - sl[1]))); st['ne'] += 1
    env.stepDt = step
    return st


def one(job):
    model, scen, seed = job
    over, kw = MODELS[model]
    o = SimOptions()
    for k, v in over.items():
        setattr(o, k, v)
    if scen.startswith('farm'):
        mode = scen.split('-')[1]
        env = FarmEnv(buildFarm(FarmSpec.load('farms/default.json'), seed), instruction='wet', opts=o, **kw)
        ag = Schedule() if mode == 'schedule' else FarmRule('wet')
    else:
        course, form = scen.split('-')
        env = ShepherdEnv(get_course(course), nFan=8, seed=seed, opts=o, formation=dict(PRESETS[form]), **kw)
        ag = _Fixed()
    st = _instrument(env)
    t0 = time.time()
    r = runEpisode(env, ag)
    dt = env.opts.dt
    return dict(model=model, scenario=scen, seed=seed, collisions=r['collisions'],
                fanObs=r['fan-obstacle'], fanFan=r['fan-fan'], fanLeader=r['fan-leader'],
                leaderObs=r['leader-obstacle'], captured=r['captured'], time=r['time'],
                coverage=r.get('coverage', np.nan), slough=r.get('sloughExposure', np.nan),
                weavingDegPerS=float(np.rad2deg(st['turn'] / max(1, st['n']) / dt)),
                slotErr=st['err'] / max(1, st['ne']), wall=time.time() - t0)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = sys.argv[2] if len(sys.argv) > 2 else 'results/shepherd/followers.csv'
    scen = ['dogleg-column', 'narrows-column', 'dogleg-vee', 'narrows-vee', 'farm-schedule', 'farm-rule']
    jobs = [(m, s, k) for s in scen for m in MODELS for k in range(n if not s.startswith('farm') else min(n, 4))]
    with mp.get_context(os.environ.get('SHEPHERD_MP') or ('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')).Pool() as p:
        rows = p.map(one, jobs)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('%-15s %-7s %3s %10s %8s %8s %9s %8s %8s' % ('scenario', 'model', 'n', 'collisions', 'walls', 'f-f', 'weave/s', 'slotErr', 'cover%'))
    for s in scen:
        for m in MODELS:
            rs = [r for r in rows if r['scenario'] == s and r['model'] == m]
            g = lambda k: np.mean([r[k] for r in rs])
            print('%-15s %-7s %3d %10.1f %8.1f %8.1f %9.1f %8.1f %8.0f' % (s, m, len(rs), g('collisions'), g('fanObs'), g('fanFan'),
                  g('weavingDegPerS'), g('slotErr'), 100 * g('coverage') if s.startswith('farm') else float('nan')))


if __name__ == '__main__':
    main()
