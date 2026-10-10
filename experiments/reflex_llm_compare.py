"""Compare the language-model runs with and without the obstacle reflex,
paired by field layout, and split every collision by where it happened.

    python experiments/reflex_llm_compare.py [NONE_DIR] [REFLEX_DIR]

Defaults: results/farm_study_vfield and results/farm_study_reflex. Collisions
are read from the transcripts (each decision reports the collisions since the
last one, with position), so a collision after the final decision is counted
in the totals but not classified. Classes: within 6 m of a rock pile or power
pole; in the yard lane or the gate (x < 3 m); against the field fence; between
machines.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from shepherd.farm import FarmSpec, buildFarm  # noqa: E402

TAG = 'qwen2-5_7b-p3'
KINDS = ('rock/pole', 'lane/gate', 'field fence', 'machine', 'other')


def where(c, course):
    if c['type'] in ('fan-fan', 'fan-leader'):
        return 'machine'
    x, y = c['x'], c['y']
    L, W = course.field[1], course.field[3]
    for h in course.hazards:
        if h['type'] in ('rock', 'pole') and np.hypot(x - h['x'], y - h['y']) < h['r'] + 6:
            return 'rock/pole'
    if x < 3:
        return 'lane/gate'
    if x > L - 3 or y < 3 or y > W - 3:
        return 'field fence'
    return 'other'


def load(study_dir, label, courses):
    rows = []
    for f in glob.glob(os.path.join(study_dir, 'transcripts', TAG, '*_test_s*.json')):
        cond, instr, _, s = os.path.basename(f)[:-5].split('_')
        seed = int(s[1:])
        d = json.load(open(f))
        cnt = dict.fromkeys(KINDS, 0)
        for t in d['turns']:
            for c in t['observation'].get('collisionsSinceLastCall') or []:
                cnt[where(c, courses[seed])] += 1
        r = d['result']
        rows.append(dict(study=label, cond=cond, instr=instr, seed=seed, total=r['collisions'],
                         north=r['northExposure'], wet=r['sloughExposure'], coverage=r['coverage'], cost=r['cost'],
                         **cnt))
    return rows


def boot(x, B=10000, seed=0):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    none_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'farm_study_vfield')
    refl_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'results', 'farm_study_reflex')
    sp = FarmSpec.load(os.path.join(ROOT, 'farms', 'default.json'))
    courses = {s: buildFarm(sp, s) for s in range(8)}
    df = pd.DataFrame(load(none_dir, 'none', courses) + load(refl_dir, 'reflex', courses))
    df.to_csv(os.path.join(refl_dir, 'reflex_vs_none_runs.csv'), index=False)
    seen = df[list(KINDS)].sum(axis=1).groupby(df.study).sum()
    print('runs', df.groupby('study').size().to_dict(), '| collisions classified / total',
          seen.to_dict(), df.groupby('study').total.sum().to_dict())

    print('\nper run, mean (none -> reflex); zero-collision runs; paired difference by layout [95% CI]')
    p = df.pivot_table(index=['cond', 'instr', 'seed'], columns='study',
                       values=['total', 'rock/pole', 'lane/gate', 'field fence'])
    for cond in ('zeroshot', 'whatif', 'lessons', 'memory', 'ALL'):
        q = p if cond == 'ALL' else p.loc[cond]
        sub = df if cond == 'ALL' else df[df.cond == cond]
        parts = []
        for v in ('total', 'rock/pole', 'lane/gate', 'field fence'):
            dd = (q[(v, 'reflex')] - q[(v, 'none')]).groupby(level='seed').mean()
            m, lo, hi = boot(dd.values)
            parts.append('%s %.2f -> %.2f (%+.2f [%+.2f, %+.2f])' % (
                v, sub[sub.study == 'none'][v].mean(), sub[sub.study == 'reflex'][v].mean(), m, lo, hi))
        z = sub.assign(z=sub.total == 0).groupby('study').z.agg(['sum', 'size'])
        print('%-9s zero-collision runs %d/%d -> %d/%d' % (cond, z.loc['none', 'sum'], z.loc['none', 'size'],
                                                          z.loc['reflex', 'sum'], z.loc['reflex', 'size']))
        for s in parts:
            print('            ' + s)
    print('\ncattle strip on the north-fence day (machine-s): mean, fields with zero')
    n = df[df.instr == 'north'].groupby(['cond', 'study']).north
    print(pd.concat([n.mean().round(1), n.apply(lambda x: int((x == 0).sum()))], axis=1,
                    keys=['mean', 'zero']).to_string())


if __name__ == '__main__':
    main()
