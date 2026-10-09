"""Which boid settings stop the slot oscillation? Boid rules unchanged, only
their parameters. Baseline motion, single-file track column (spacing 12),
the paper's settings vs one-at-a-time changes and a few combinations.

  python -m shepherd.sweep_damping [nSeeds] [out.csv]

Oscillation is the RMS distance from each boid to its slot after t = 40 s
(formed up), on narrows and dogleg.
"""
import csv
import copy
import sys
import multiprocessing as mp

import numpy as np

from grainframe import SimOptions
from .courses import get_course
from .env import ShepherdEnv
from .agents import baselinePlan, PRESETS

VARIANTS = {
    'paper (VR 8)':             {},
    'VR 14':                    dict(VR=14),
    'VR 20':                    dict(VR=20),
    'VR 26':                    dict(VR=26),
    'AF 4':                     dict(AF=4.0),
    'CF 0':                     dict(CF=0.0),
    'leader pull 0.5':          dict(fanLeaderFactor=0.5),
    'leader pull 2':            dict(fanLeaderFactor=2.0),
    'minSpeed 0.5':             dict(minSpeed=0.5),
    'sep 1/r (norm)':           dict(SEP_PROFILE='norm'),
    'VR 20 + CF 0':             dict(VR=20, CF=0.0),
    'VR 20 + CF 0 + AF 4':      dict(VR=20, CF=0.0, AF=4.0),
    'VR 26 + CF 0 + AF 4':      dict(VR=26, CF=0.0, AF=4.0),
    'VR 20 + CF 0 + minSpd 0.5': dict(VR=20, CF=0.0, minSpeed=0.5),
}


ROUND2 = {
    'paper (VR 8)':                      {},
    'leader pull 2':                     dict(fanLeaderFactor=2.0),
    'leader pull 3':                     dict(fanLeaderFactor=3.0),
    'sep 1/r (norm)':                    dict(SEP_PROFILE='norm'),
    'pull 2 + sep 1/r':                  dict(fanLeaderFactor=2.0, SEP_PROFILE='norm'),
    'VR 20 + CF 0 + AF 4':               dict(VR=20, CF=0.0, AF=4.0),
    'pull 2 + sep 1/r + VR20 CF0 AF4':   dict(fanLeaderFactor=2.0, SEP_PROFILE='norm', VR=20, CF=0.0, AF=4.0),
}


def one(args):
    name, over, course, seed = args
    opts = SimOptions()
    for k, v in over.items():
        setattr(opts, k, v)
    env = ShepherdEnv(get_course(course), nFan=8, seed=seed, opts=opts,
                      formation=dict(PRESETS['column']))
    env.setPlan(baselinePlan(env, env.remainingRoute()))
    err = []
    while not env.done:
        env.runFor(1.0)
        if env.t > 40 and env.nFan:
            sl = env.currentSlots()
            a, l = env._bodyFrame()
            err.append(np.mean((a - sl[0]) ** 2 + (l - sl[1]) ** 2))
    r = env.result()
    return dict(variant=name, course=course, seed=seed, collisions=r['collisions'],
                fanObs=r['fan-obstacle'], fanFan=r['fan-fan'], fanLeader=r['fan-leader'],
                slotRMS=float(np.sqrt(np.mean(err))) if err else np.nan,
                lost=r['fansLost'])


def main():
    global VARIANTS
    if '--round2' in sys.argv:
        sys.argv.remove('--round2')
        VARIANTS = ROUND2
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = sys.argv[2] if len(sys.argv) > 2 else 'results/shepherd/damping_sweep.csv'
    jobs = [(nm, ov, c, s) for nm, ov in VARIANTS.items() for c in ('narrows', 'dogleg') for s in range(n)]
    with mp.get_context('fork').Pool() as pool:
        rows = pool.map(one, jobs)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('%-28s %8s %8s %8s %8s %8s' % ('variant', 'narrows', 'dogleg', 'slotRMS', 'fan-fan', 'lost'))
    for nm in VARIANTS:
        rs = [r for r in rows if r['variant'] == nm]
        m = lambda c, k: np.mean([r[k] for r in rs if c is None or r['course'] == c])
        print('%-28s %8.1f %8.1f %8.2f %8.2f %8.2f' % (nm, m('narrows', 'collisions'), m('dogleg', 'collisions'),
                                                     m(None, 'slotRMS'), m(None, 'fanFan'), m(None, 'lost')))


if __name__ == '__main__':
    main()
