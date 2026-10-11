"""A decider that opens up too early (the AI's main failure), with and without the safety filter,
next to the schedule and the hand-written rule. Reflex followers, wet day, 8 layouts.

    python experiments/eager_decider_test.py
"""
import sys, multiprocessing as mp
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from shepherd.farm import FarmSpec, buildFarm
from shepherd.farmenv import FarmEnv
from shepherd.farmagents import _Base, _scheduleChoice, Schedule, FarmRule
from shepherd.followers import followerSetup
from shepherd.run import runEpisode

class Eager(_Base):
    """Full line from the moment the leader starts a pass, even with the tail still in the gate,
    and two abreast in the lane."""
    name = 'eager'
    def begin(self, env):
        super().begin(env); self.choices = []
    def act(self, env, obs):
        return self._apply(env, 'line' if _scheduleChoice(env, fold=35.0, unfold=0.0) == 'line' else 'column2', 1.0)

def job(a):
    dec, seed, shield = a
    sp = FarmSpec.load(os.path.join(ROOT, 'farms', 'default.json'))
    opts, kw = followerSetup('reflex')
    env = FarmEnv(buildFarm(sp, seed), instruction='wet', opts=opts, **kw)
    ag = {'eager': Eager, 'schedule': Schedule, 'rule': lambda: FarmRule('wet')}[dec]()
    ag.shield = shield
    r = runEpisode(env, ag)
    lane = sum(1 for c in env.collisions if c['x'] < 3)
    return dec, seed, shield, r['collisions'], lane, round(r['coverage'], 3), ag.nBlocked, round(r['time'])

if __name__ == '__main__':
    jobs = [(d, s, sh) for d in ('eager', 'schedule', 'rule') for s in range(8) for sh in (False, True)]
    with mp.get_context(os.environ.get('SHEPHERD_MP') or ('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')).Pool(2) as p:
        res = p.map(job, jobs)
    import pandas as pd
    df = pd.DataFrame(res, columns=['decider', 'seed', 'shield', 'collisions', 'laneGate', 'coverage', 'blocked', 'time'])
    os.makedirs(os.path.join(ROOT, 'results', 'shepherd', 'layer_tests'), exist_ok=True)
    df.to_csv(os.path.join(ROOT, 'results', 'shepherd', 'layer_tests', 'eager.csv'), index=False)
    print(df.groupby(['decider', 'shield'])[['collisions', 'laneGate', 'coverage', 'blocked', 'time']].mean().round(2))
