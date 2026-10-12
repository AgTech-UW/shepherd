"""Counterfactual check of fit check version 2 before spending a model run on it: replay the
recorded replies of the safety-filter run (fit check version 1) under version 2, in order, with
no model. Once a run diverges, later replies were written for a different state, so this is an
approximation of what the model would do; the real test is a new model run.

    python experiments/fit2_replay.py [--workers N] [--which rock|all]

--which rock: the runs that had a rock or pole collision; all: every test run.
Writes results/farm_study_reflex_shield/fit2_replay.csv.
"""
import argparse
import glob
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from shepherd.replay import replay                 # noqa: E402
from reflex_llm_compare import where, TAG          # noqa: E402

SDIR = os.path.join(ROOT, 'results', 'farm_study_reflex_shield')


def job(a):
    path, fitVersion = a
    env, info = replay(path, spec=os.path.join(ROOT, 'farms', 'default.json'), followers='reflex',
                       fitVersion=fitVersion)
    kinds = {}
    for c in env.collisions:
        k = where(dict(type=c.get('cat'), x=c['x'], y=c['y']), env.course)
        kinds[k] = kinds.get(k, 0) + 1
    cond, instr, _, s = os.path.basename(path)[:-5].split('_')
    r = info['result']
    return dict(cond=cond, instr=instr, seed=int(s[1:]), fitVersion=fitVersion, exact=info['exact'],
                collisions=r['collisions'], rockPole=kinds.get('rock/pole', 0), laneGate=kinds.get('lane/gate', 0),
                fence=kinds.get('field fence', 0), machine=kinds.get('machine', 0), coverage=r['coverage'],
                wet=r['sloughExposure'], north=r['northExposure'], time=r['time'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--which', default='rock', choices=('rock', 'all'))
    a = p.parse_args()
    runs = pd.read_csv(os.path.join(SDIR, 'shield_vs_reflex_runs.csv'))
    runs = runs[runs.study == 'shield']
    if a.which == 'rock':
        runs = runs[runs['rock/pole'] > 0]
    paths = [os.path.join(SDIR, 'transcripts', TAG, '%s_%s_test_s%d.json' % (r.cond, r.instr, r.seed))
             for r in runs.itertuples()]
    out = os.path.join(SDIR, 'fit2_replay.csv')
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    have = set(zip(done.cond, done.instr, done.seed)) if len(done) else set()

    def key(q):
        cond, instr, _, s = os.path.basename(q)[:-5].split('_')
        return cond, instr, int(s[1:])
    todo = [(q, 2) for q in paths if key(q) not in have]
    print('%d runs, %d to do' % (len(paths), len(todo)), flush=True)
    ctx = mp.get_context('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')
    with ctx.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, todo), 1):
            done = pd.concat([done, pd.DataFrame([r])], ignore_index=True)
            done.to_csv(out, index=False)
            print('%3d/%d %-8s %-7s s%d | col %d rock %d lane %d cov %.2f' % (
                i, len(todo), r['cond'], r['instr'], r['seed'], r['collisions'], r['rockPole'], r['laneGate'],
                r['coverage']), flush=True)
    m = done.merge(runs[['cond', 'instr', 'seed', 'total', 'rock/pole', 'lane/gate', 'coverage']],
                   on=['cond', 'instr', 'seed'], suffixes=('', '_v1'))
    print('\nrecorded (fit check 1) -> replayed under fit check 2, %d runs:' % len(m))
    print('  collisions %.2f -> %.2f | rock/pole %.2f -> %.2f | lane/gate %.2f -> %.2f | worked %.1f%% -> %.1f%%' % (
        m.total.mean(), m.collisions.mean(), m['rock/pole'].mean(), m.rockPole.mean(), m['lane/gate'].mean(),
        m.laneGate.mean(), 100 * m.coverage_v1.mean(), 100 * m.coverage.mean()))


if __name__ == '__main__':
    main()
