"""The two layers under the decider, tested on the deterministic deciders: the safety
filter (binding fit check) and the leader wait ('pearls on a string'), each on and off,
for the schedule, the hand-written rule and a test decider that widens too early.
Reflex followers, wet day, the eight test layouts.

    python experiments/layer_tests.py [--workers N] [--deciders eager,schedule,rule]
                                      [--models reflex,pearls] [--shield off|on|both] [--out layers.csv]

Writes results/shepherd/layer_tests/<out> and prints the means. Every episode is deterministic,
so rows can be run in parts and joined.
surge: share of follower time spent faster than the leader plus 0.5 m/s.
leaderSlowed: share of time the leader is held back by the wait.
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from shepherd.farm import FarmSpec, buildFarm                      # noqa: E402
from shepherd.farmenv import FarmEnv                               # noqa: E402
from shepherd.farmagents import _Base, _scheduleChoice, Schedule, FarmRule   # noqa: E402
from shepherd.followers import followerSetup                       # noqa: E402
from shepherd.run import runEpisode                                # noqa: E402


class Eager(_Base):
    """Full line from the moment the leader starts a pass, even with the tail still in the gate,
    and two abreast in the lane."""
    name = 'eager'

    def begin(self, env):
        super().begin(env)
        self.choices = []

    def act(self, env, obs):
        return self._apply(env, 'line' if _scheduleChoice(env, fold=35.0, unfold=0.0) == 'line' else 'column2', 1.0)


class MeteredFarm(FarmEnv):
    """FarmEnv that also records follower surges and how often the leader waits."""

    def reset(self, *a, **k):
        r = super().reset(*a, **k)
        self.mSteps = self.mSurge = self.mSlow = 0
        self.mFan = 0
        self.mVmax = 0.0
        return r

    def postStep(self):
        super().postStep()
        v = np.hypot(self.fanVx, self.fanVy)
        self.mSteps += 1
        self.mFan += len(v)
        self.mSurge += int((v > self.speed + 0.5).sum())
        self.mVmax = max(self.mVmax, float(v.max()) if len(v) else 0.0)
        if self.leaderWait:
            self.mSlow += int(self.lag > self.leaderWait[0])


def job(a):
    dec, model, shield, seed = a
    sp = FarmSpec.load(os.path.join(ROOT, 'farms', 'default.json'))
    opts, kw = followerSetup(model)
    env = MeteredFarm(buildFarm(sp, seed), instruction='wet', opts=opts, **kw)
    ag = {'eager': Eager, 'schedule': Schedule, 'rule': lambda: FarmRule('wet')}[dec]()
    ag.shield = shield
    r = runEpisode(env, ag)
    lane = sum(1 for c in env.collisions if c['x'] < 3)
    return dict(decider=dec, model=model, shield=shield, seed=seed, collisions=r['collisions'], laneGate=lane,
                coverage=round(r['coverage'], 4), time=round(r['time'], 1), wet=round(r['sloughExposure'], 1),
                blocked=getattr(ag, 'nBlocked', 0), surge=round(env.mSurge / max(env.mFan, 1), 4),
                maxFollowerSpeed=round(env.mVmax, 3), leaderSlowed=round(env.mSlow / max(env.mSteps, 1), 4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--deciders', default='eager,schedule,rule')
    p.add_argument('--models', default='reflex,pearls')
    p.add_argument('--shield', default='both', choices=('off', 'on', 'both'))
    p.add_argument('--out', default='layers.csv')
    a = p.parse_args()
    shields = {'off': (False,), 'on': (True,), 'both': (False, True)}[a.shield]
    jobs = [(d, m, sh, s) for d in a.deciders.split(',') for m in a.models.split(',')
            for sh in shields for s in range(8)]
    ctx = mp.get_context(os.environ.get('SHEPHERD_MP') or ('fork' if 'fork' in mp.get_all_start_methods() else 'spawn'))
    with ctx.Pool(a.workers) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            res.append(r)
            print('%3d/%d %-8s %-6s shield=%d seed %d | col %d lane %d cov %.2f t %4.0f' % (
                i, len(jobs), r['decider'], r['model'], r['shield'], r['seed'], r['collisions'], r['laneGate'],
                r['coverage'], r['time']), flush=True)
    import pandas as pd
    df = pd.DataFrame(res).sort_values(['decider', 'model', 'shield', 'seed'])
    out = os.path.join(ROOT, 'results', 'shepherd', 'layer_tests')
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, a.out), index=False)
    cols = ['collisions', 'laneGate', 'coverage', 'time', 'wet', 'blocked', 'surge', 'leaderSlowed']
    print(df.groupby(['decider', 'model', 'shield'])[cols].mean().round(3).to_string())


if __name__ == '__main__':
    main()
