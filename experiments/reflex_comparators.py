"""Hand-written deciders on the 8 test layouts with and without the obstacle
reflex (follower model 'reflex' vs 'vfield'). Writes one row per run.

    python experiments/reflex_comparators.py results/shepherd/reflex_vs_vfield.csv
"""
import os, sys, time, csv, numpy as np, multiprocessing as mp
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from shepherd.farm import FarmSpec, buildFarm
from shepherd.farmenv import FarmEnv
from shepherd.farmagents import FixedColumn, Schedule, FarmRule
from shepherd.followers import followerSetup
from shepherd.compare_followers import _instrument
from shepherd.run import runEpisode
def one(job):
    seed, agent, model = job
    spec = FarmSpec.load('farms/default.json'); opts, kw = followerSetup(model)
    env = FarmEnv(buildFarm(spec, seed), instruction='wet', opts=opts, **kw)
    ag = {'column': FixedColumn, 'schedule': Schedule, 'rule': lambda: FarmRule('wet')}[agent]()
    st = _instrument(env); r = runEpisode(env, ag)
    return dict(seed=seed, agent=agent, model=model, col=r['collisions'], fanObs=r['fan-obstacle'], fanFan=r['fan-fan'],
                leader=r['leader-obstacle'], reflex=r.get('reflexEvents', 0), cov=r['coverage'], wet=r['sloughExposure'],
                weave=float(np.rad2deg(st['turn'] / max(1, st['n']) / env.opts.dt)), lost=r['fansLost'])
if __name__ == '__main__':
    jobs = [(s, a, m) for a in ('column', 'schedule', 'rule') for s in range(8) for m in ('vfield', 'reflex')]
    with mp.get_context(os.environ.get('SHEPHERD_MP') or ('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')).Pool(2) as p:
        rows = []
        for r in p.imap_unordered(one, jobs):
            rows.append(r); print(r, flush=True)
    with open(sys.argv[1], 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('DONE')
