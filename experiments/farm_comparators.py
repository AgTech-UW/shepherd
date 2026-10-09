"""Run the non-LLM deciders on the farm, resumably, in parallel.

    python experiments/farm_comparators.py --seeds 0-7 [--workers 2]

Writes results/farm_study/comparators/episodes.csv (one row per decider x
instruction x seed) and a trajectory .npz per episode. Deciders whose
behaviour does not depend on the instruction (column, schedule; the rule
for an instruction it was not written for) are run once and scored under
each instruction.
"""
import argparse, csv, os, sys, time
import multiprocessing as mp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from shepherd.farm import FarmSpec, buildFarm                    # noqa: E402
from shepherd.farmenv import FarmEnv, INSTRUCTIONS                # noqa: E402
from shepherd.farmagents import FixedColumn, Schedule, FarmRule, FarmSearch   # noqa: E402
from shepherd.run import runEpisode                               # noqa: E402
from shepherd.followers import followerSetup                      # noqa: E402

COLS = ['agent', 'instruction', 'seed', 'captured', 'posErr', 'headErrDeg', 'collisions', 'fan-leader',
        'fan-fan', 'fan-obstacle', 'leader-obstacle', 'fansLost', 'coverage', 'sloughExposure',
        'northExposure', 'time', 'cost', 'cost_wet', 'cost_weather', 'cost_north', 'formationSwitches',
        'decisions', 'wall_s', 'llmCalls', 'llmBadReplies', 'llmWall_s', 'phase']


def job(args):
    agent, mode, seed, spec, out, followers = args
    course = buildFarm(spec, seed)
    instr = mode if mode in INSTRUCTIONS else 'wet'
    opts, kw = followerSetup(followers)
    env = FarmEnv(course, instruction=instr, opts=opts, **kw)
    ag = {'column': FixedColumn, 'schedule': Schedule}.get(agent)
    ag = ag() if ag else (FarmRule(mode) if agent == 'rule' else FarmSearch(mode))
    t0 = time.time()
    r = runEpisode(env, ag)
    r['decisions'] = getattr(ag, 'nDecisions', len(getattr(ag, 'choices', [])))
    r['wall_s'] = round(time.time() - t0, 1)
    np.savez_compressed(os.path.join(out, 'traj', '%s_%s_s%d.npz' % (agent, mode, seed)),
                        x=env.log['x'], y=env.log['y'], fanX=np.array(env.log['fanX']),
                        fanY=np.array(env.log['fanY']), covered=env.covered,
                        col=np.array([[c['x'], c['y']] for c in env.collisions]).reshape(-1, 2))
    return agent, mode, seed, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='0-7')
    p.add_argument('--spec', default=os.path.join(ROOT, 'farms', 'default.json'))
    p.add_argument('--out', default=None, help='default results/farm_study[_<followers>]/comparators')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    p.add_argument('--agents', default='column,schedule,rule,search')
    p.add_argument('--followers', default='vfield', choices=['paper', 'damped', 'vfield'])
    a = p.parse_args()
    if a.out is None:
        a.out = os.path.join(ROOT, 'results', 'farm_study' + ('' if a.followers == 'paper' else '_' + a.followers), 'comparators')
    lo, hi = map(int, a.seeds.split('-'))
    seeds = range(lo, hi + 1)
    spec = FarmSpec.load(a.spec)
    os.makedirs(os.path.join(a.out, 'traj'), exist_ok=True)
    path = os.path.join(a.out, 'episodes.csv')
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(r['agent'], r['instruction'], r['seed']) for r in csv.DictReader(f)}
    jobs = []
    for ag in a.agents.split(','):
        modes = {'column': ['any'], 'schedule': ['any'], 'rule': ['wet', 'weather'],
                 'search': list(INSTRUCTIONS)}[ag]
        for m in modes:
            for s in seeds:
                targets = list(INSTRUCTIONS) if m == 'any' else (['wet', 'north'] if (ag == 'rule' and m == 'wet') else [m])
                if all((ag, t, str(s)) in done for t in targets):
                    continue
                jobs.append((ag, m, s, spec, a.out, a.followers))
    jobs.sort(key=lambda j: j[0] != 'search')          # long jobs first
    print('%d jobs' % len(jobs), flush=True)
    with mp.get_context('fork').Pool(a.workers) as pool:
        for agent, mode, seed, r in pool.imap_unordered(job, jobs):
            targets = list(INSTRUCTIONS) if mode == 'any' else (['wet', 'north'] if (agent == 'rule' and mode == 'wet') else [mode])
            new = not os.path.exists(path)
            with open(path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
                if new:
                    w.writeheader()
                for t in targets:
                    row = dict(r, agent=agent, instruction=t, seed=seed, cost=r['cost_' + t], phase='test')
                    w.writerow(row)
            print('%-9s %-8s seed %d | col %3d cov %.2f slough %5.0f north %5.0f t %4.0f | %s | %.0fs'
                  % (agent, mode, seed, r['collisions'], r['coverage'], r['sloughExposure'], r['northExposure'],
                     r['time'], ' '.join('%s %.0f' % (t, r['cost_' + t]) for t in targets), r['wall_s']), flush=True)


if __name__ == '__main__':
    main()
