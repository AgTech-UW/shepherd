"""Counterfactual check of the reflex side rule before spending a model run on it: replay the
recorded replies of a model run, in order, with no model, under reflex version 2 (committed side of
least resistance) or 3 (as 2, and the flock rules may not turn a follower toward the obstacle it is
passing). Once a run diverges, later replies were written for a different state, so this
approximates what the model would do; the real test is a new model run. --version 1 checks that the
recorded run replays exactly.

    python experiments/reflex_side_replay.py [--run farm_study_reflex_shield_fit4_rL15M2.5]
        [--version 2] [--which rock|rock2|all|NAME,NAME] [--workers 2]

--which rock: runs with a rock or pole hit as recorded; rock2: also those with one in the version 2
replay. Writes results/<run>/reflex_side_v<version>.csv (resumes) and prints recorded vs replayed.
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


def job(a):
    path, version = a
    env, info = replay(path, spec=os.path.join(ROOT, 'farms', 'default.json'), followers='reflex',
                       envAttrs=dict(reflexVersion=version))
    kinds = {}
    for c in env.collisions:
        k = where(dict(type=c.get('cat'), x=c['x'], y=c['y']), env.course)
        kinds[k] = kinds.get(k, 0) + 1
    cond, instr, _, s = os.path.basename(path)[:-5].split('_')
    r = info['result']
    return dict(cond=cond, instr=instr, seed=int(s[1:]), version=version, exact=info['exact'],
                collisions=r['collisions'], rockPole=kinds.get('rock/pole', 0), laneGate=kinds.get('lane/gate', 0),
                fence=kinds.get('field fence', 0), machine=kinds.get('machine', 0), coverage=r['coverage'],
                wet=r['sloughExposure'], north=r['northExposure'], time=r['time'], reflexEvents=r.get('reflexEvents'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='farm_study_reflex_shield_fit4_rL15M2.5')
    p.add_argument('--version', type=int, default=2, choices=(1, 2, 3))
    p.add_argument('--which', default='rock')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    rdir = os.path.join(ROOT, 'results', a.run)
    tdir = os.path.join(rdir, 'transcripts', TAG)
    rec = pd.read_csv(os.path.join(rdir, 'filter_runs.csv'))
    rec = rec[rec.study == rec.study.iloc[-1]]
    if a.which == 'rock':
        rec = rec[rec['rock/pole'] > 0]
    elif a.which == 'rock2':
        v2 = pd.read_csv(os.path.join(rdir, 'reflex_side_v2.csv'))
        more = set(zip(v2[v2.rockPole > 0].cond, v2[v2.rockPole > 0].instr, v2[v2.rockPole > 0].seed))
        rec = rec[(rec['rock/pole'] > 0).values | np.array([k in more for k in zip(rec.cond, rec.instr, rec.seed)])]
    elif a.which != 'all':
        keep = set(a.which.split(','))
        rec = rec[[('%s_%s_test_s%d' % (r.cond, r.instr, r.seed)) in keep for r in rec.itertuples()]]
    paths = [os.path.join(tdir, '%s_%s_test_s%d.json' % (r.cond, r.instr, r.seed)) for r in rec.itertuples()]
    out = os.path.join(rdir, 'reflex_side_v%d.csv' % a.version)
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    have = set(zip(done.cond, done.instr, done.seed)) if len(done) else set()

    def key(q):
        cond, instr, _, s = os.path.basename(q)[:-5].split('_')
        return cond, instr, int(s[1:])
    todo = [(q, a.version) for q in paths if key(q) not in have]
    print('%d runs, %d to do' % (len(paths), len(todo)), flush=True)
    ctx = mp.get_context('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')
    with ctx.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, todo), 1):
            done = pd.concat([done, pd.DataFrame([r])], ignore_index=True)
            done.to_csv(out, index=False)
            print('%3d/%d %-8s %-7s s%d | col %d rock %d lane %d fence %d machine %d cov %.2f' % (
                i, len(todo), r['cond'], r['instr'], r['seed'], r['collisions'], r['rockPole'], r['laneGate'],
                r['fence'], r['machine'], r['coverage']), flush=True)
    m = done.merge(rec[['cond', 'instr', 'seed', 'total', 'rock/pole', 'lane/gate', 'field fence', 'coverage']],
                   on=['cond', 'instr', 'seed'], suffixes=('', '_rec'))
    print('\n%d runs, recorded -> replayed with reflex version %d (exact replays: %d):' % (
        len(m), a.version, int(m.exact.sum())))
    print('  collisions %.2f -> %.2f | rock/pole %.2f -> %.2f | lane/gate %.2f -> %.2f | fence %.2f -> %.2f | '
          'worked %.1f%% -> %.1f%%' % (m.total.mean(), m.collisions.mean(), m['rock/pole'].mean(), m.rockPole.mean(),
                                      m['lane/gate'].mean(), m.laneGate.mean(), m['field fence'].mean(),
                                      m.fence.mean(), 100 * m.coverage_rec.mean(), 100 * m.coverage.mean()))
    print('  machine-machine %d | rock/pole hits %d -> %d' % (m.machine.sum(), m['rock/pole'].sum(), m.rockPole.sum()))


if __name__ == '__main__':
    main()
