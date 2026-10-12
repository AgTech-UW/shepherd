"""Reflex only, safety filter with fit check 1, and with fit check 2, side by side (same desktop,
same prompt and model), paired by field layout.

    python experiments/fit2_compare.py

Writes results/farm_study_reflex_shield_fit2/fit2_vs_v1_vs_reflex_runs.csv.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from shepherd.farm import FarmSpec, buildFarm           # noqa: E402
from reflex_llm_compare import load, boot, KINDS        # noqa: E402
from shield_compare import fitStats                     # noqa: E402

DIRS = {'reflex': 'farm_study_reflex', 'v1': 'farm_study_reflex_shield', 'v2': 'farm_study_reflex_shield_fit2'}


def ci(x):
    m, lo, hi = boot(x)
    return '%+.2f [%+.2f, %+.2f]' % (m, lo, hi)


def paired(sub, col, a, b):
    p = sub.pivot_table(index=['cond', 'instr', 'seed'], columns='study', values=col)
    return (p[b] - p[a]).groupby(level='seed').mean().values


def main():
    sp = FarmSpec.load(os.path.join(ROOT, 'farms', 'default.json'))
    courses = {s: buildFarm(sp, s) for s in range(8)}
    dirs = {k: os.path.join(ROOT, 'results', v) for k, v in DIRS.items()}
    df = pd.concat([pd.DataFrame(load(d, k, courses)) for k, d in dirs.items()])
    fs = pd.concat([fitStats(d).assign(study=k) for k, d in dirs.items()])
    df = df.merge(fs, on=['study', 'cond', 'instr', 'seed'])
    df.to_csv(os.path.join(dirs['v2'], 'fit2_vs_v1_vs_reflex_runs.csv'), index=False)
    print('columns: reflex only | filter, fit check 1 | filter, fit check 2')
    for cond in ('zeroshot', 'whatif', 'lessons', 'memory', 'ALL'):
        sub = df if cond == 'ALL' else df[df.cond == cond]
        g = sub.groupby('study')
        row = lambda col, f='%.2f': ' '.join(f % g[col].mean()[k] for k in DIRS)
        print('%-8s total %s | rock %s | lane %s | fence %s | 3+ %s | none %s | worked %s' % (
            cond, row('total'), row('rock/pole'), row('lane/gate'), row('field fence'),
            ' '.join('%d' % (sub[sub.study == k].total >= 3).sum() for k in DIRS),
            ' '.join('%d' % (sub[sub.study == k].total == 0).sum() for k in DIRS),
            ' '.join('%.1f' % (100 * g.coverage.mean()[k]) for k in DIRS)))
    for col in ('total', 'rock/pole', 'lane/gate'):
        print('fit check 2 minus 1, %-9s %s | minus reflex only %s' % (
            col, ci(paired(df, col, 'v1', 'v2')), ci(paired(df, col, 'reflex', 'v2'))))
    print('fit check 2 minus 1, worked', ci(100 * paired(df, 'coverage', 'v1', 'v2')))
    for k in DIRS:
        s = df[df.study == k]
        print('%-6s chose a non-fitting formation %.1f%% of decisions; filter holds per run %.1f' % (
            k, 100 * s.notFitting.sum() / s.decisions.sum(), s.held.mean()))
    n = df[df.instr == 'north']
    for k in DIRS:
        print('north-fence day, layouts with no time in the strip,', k,
              {c: int((n[(n.study == k) & (n.cond == c)].north == 0).sum()) for c in ('zeroshot', 'whatif', 'lessons', 'memory')})


if __name__ == '__main__':
    main()
