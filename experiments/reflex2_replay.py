"""Counterfactual check of a stronger reflex: replay the recorded replies of the reflex run (no
filter) for the runs that had a rock or pole collision, with the reflex looking further ahead and
keeping a wider margin. No model; once a run diverges, later replies were written for a different
state, so this approximates what the model would do.

    python experiments/reflex2_replay.py [--look 15] [--margin 2.5] [--workers 2]

Writes results/farm_study_reflex/reflex2_replay_L<look>_M<margin>.csv.
"""
import argparse
import multiprocessing as mp
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from shepherd.replay import replay                 # noqa: E402
from reflex_llm_compare import where, TAG          # noqa: E402

RDIR = os.path.join(ROOT, 'results', 'farm_study_reflex')


def job(a):
    path, look, margin = a
    env, info = replay(path, spec=os.path.join(ROOT, 'farms', 'default.json'), followers='reflex',
                       envAttrs=dict(reflexLook=look, reflexMargin=margin))
    kinds = {}
    for c in env.collisions:
        k = where(dict(type=c.get('cat'), x=c['x'], y=c['y']), env.course)
        kinds[k] = kinds.get(k, 0) + 1
    cond, instr, _, s = os.path.basename(path)[:-5].split('_')
    r = info['result']
    return dict(cond=cond, instr=instr, seed=int(s[1:]), look=look, margin=margin, collisions=r['collisions'],
                rockPole=kinds.get('rock/pole', 0), laneGate=kinds.get('lane/gate', 0), coverage=r['coverage'],
                time=r['time'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--look', type=float, default=15.0)
    p.add_argument('--margin', type=float, default=2.5)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    runs = pd.read_csv(os.path.join(ROOT, 'results', 'farm_study_reflex_shield', 'shield_vs_reflex_runs.csv'))
    runs = runs[(runs.study == 'reflex') & (runs['rock/pole'] > 0)]
    jobs = [(os.path.join(RDIR, 'transcripts', TAG, '%s_%s_test_s%d.json' % (r.cond, r.instr, r.seed)),
             a.look, a.margin) for r in runs.itertuples()]
    print('%d runs' % len(jobs), flush=True)
    ctx = mp.get_context('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')
    rows = []
    with ctx.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r)
            print('%3d/%d %-8s %-7s s%d | col %d rock %d lane %d cov %.2f' % (
                i, len(jobs), r['cond'], r['instr'], r['seed'], r['collisions'], r['rockPole'], r['laneGate'],
                r['coverage']), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RDIR, 'reflex2_replay_L%g_M%g.csv' % (a.look, a.margin)), index=False)
    m = df.merge(runs[['cond', 'instr', 'seed', 'total', 'rock/pole', 'coverage']], on=['cond', 'instr', 'seed'],
                 suffixes=('', '_rec'))
    print('\nrecorded -> replayed with look %g m, margin %g m, %d runs: collisions %.2f -> %.2f | rock/pole %d -> %d | '
          'worked %.1f%% -> %.1f%%' % (a.look, a.margin, len(m), m.total.mean(), m.collisions.mean(),
                                      m['rock/pole'].sum(), m.rockPole.sum(), 100 * m.coverage_rec.mean(),
                                      100 * m.coverage.mean()))


if __name__ == '__main__':
    main()
