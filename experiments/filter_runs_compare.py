"""Every model run with the reflex, side by side (same desktop, same prompt and model), paired by
field layout: reflex only; safety filter (fit check 1); filter with fit check 2; filter with fit
check 2 and the stronger reflex (15 m, 2.5 m). Runs that are not there yet are skipped.

    python experiments/filter_runs_compare.py [--latex]

Prints the comparison, the model against the hand-written rule and the oracle in the last run,
and (--latex) the rows of a LaTeX table. Writes filter_runs.csv into the last run's folder.
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from shepherd.farm import FarmSpec, buildFarm           # noqa: E402
from reflex_llm_compare import load, boot, KINDS        # noqa: E402
from shield_compare import fitStats                     # noqa: E402

RUNS = [('reflex', 'farm_study_reflex', 'Reflex only'),
        ('filter', 'farm_study_reflex_shield', 'Safety filter'),
        ('fit2', 'farm_study_reflex_shield_fit2', 'Filter, fit check 2'),
        ('final', 'farm_study_reflex_shield_fit3_rL15M2.5', 'Filter 3 + stronger reflex')]
CONDS = ('zeroshot', 'whatif', 'lessons', 'memory')
NAMES = dict(zeroshot='Zero-shot', whatif='What-if', lessons='Lessons', memory='Memory')


def ci(x, f='%+.2f'):
    m, lo, hi = boot(x)
    return (f + ' [' + f + ', ' + f + ']') % (m, lo, hi)


def paired(df, col, a, b, keys=('cond', 'instr', 'seed')):
    p = df.pivot_table(index=list(keys), columns='study', values=col)
    return (p[b] - p[a]).groupby(level='seed').mean().values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latex', action='store_true')
    a = ap.parse_args()
    runs = [r for r in RUNS if os.path.isdir(os.path.join(ROOT, 'results', r[1], 'transcripts'))]
    keys = [r[0] for r in runs]
    sp = FarmSpec.load(os.path.join(ROOT, 'farms', 'default.json'))
    courses = {s: buildFarm(sp, s) for s in range(8)}
    parts = []
    for k, d, _ in runs:
        path = os.path.join(ROOT, 'results', d)
        x = pd.DataFrame(load(path, k, courses)).merge(fitStats(path).assign(study=k),
                                                       on=['study', 'cond', 'instr', 'seed'])
        parts.append(x)
    df = pd.concat(parts)
    last = keys[-1]
    df.to_csv(os.path.join(ROOT, 'results', runs[-1][1], 'filter_runs.csv'), index=False)

    def stat(sub, k, what):
        s = sub[sub.study == k]
        return dict(total=s.total.mean(), lane=s['lane/gate'].mean(), rock=s['rock/pole'].mean(),
                    fence=s['field fence'].mean(), bad=int((s.total >= 3).sum()), none=int((s.total == 0).sum()),
                    worked=100 * s.coverage.mean(), n=len(s))[what]

    print('runs:', ', '.join('%s (%d)' % (r[2], (df.study == r[0]).sum()) for r in runs))
    print('%-22s' % '' + ''.join('%-22s' % r[2] for r in runs))
    for label, what, f in (('collisions', 'total', '%.2f'), ('lane or gate', 'lane', '%.2f'),
                           ('rock or pole', 'rock', '%.2f'), ('field fence', 'fence', '%.2f'),
                           ('runs with 3+', 'bad', '%d'), ('runs with none', 'none', '%d'),
                           ('field worked %', 'worked', '%.1f')):
        print('%-22s' % label + ''.join('%-22s' % (f % stat(df, k, what)) for k in keys))
    for c in CONDS:
        print('%-22s' % (c + ' collisions') + ''.join('%-22s' % ('%.2f' % stat(df[df.cond == c], k, 'total'))
                                                      for k in keys))
    for k in keys:
        s = df[df.study == k]
        print('%-8s non-fitting choices %.1f%% of decisions, filter holds %.1f per run' % (
            k, 100 * s.notFitting.sum() / max(s.decisions.sum(), 1), s.held.mean()))
    if len(keys) > 1:
        prev = keys[-2]
        print('\n%s minus %s, paired by layout:' % (last, prev))
        for col in ('total', 'rock/pole', 'lane/gate'):
            print('   %-10s %s' % (col, ci(paired(df, col, prev, last))))
        print('   worked     %s points' % ci(100 * paired(df, 'coverage', prev, last)))
        print('%s minus reflex only: collisions %s, lane/gate %s, rock/pole %s' % (
            last, ci(paired(df, 'total', 'reflex', last)), ci(paired(df, 'lane/gate', 'reflex', last)),
            ci(paired(df, 'rock/pole', 'reflex', last))))
    n = df[df.instr == 'north']
    for k in keys:
        print('north-fence day, layouts with no time in the strip, %-7s %s' % (
            k, {c: int((n[(n.study == k) & (n.cond == c)].north == 0).sum()) for c in CONDS}))

    # the last run against its own baselines
    cs = pd.read_csv(os.path.join(ROOT, 'results', runs[-1][1], 'comparators', 'episodes.csv'))
    s = df[df.study == last]
    print('\nbaselines in the last run (collisions, worked %, cost):')
    print(cs.groupby(['instruction', 'agent'])[['collisions', 'coverage', 'cost']].mean().round(2).to_string())
    rule = cs[cs.agent == 'rule'].set_index(['instruction', 'seed'])
    d = [(sd, row.total - rule.loc[(i, sd)].collisions) for (i, sd), row in s.set_index(['instr', 'seed']).iterrows()]
    dd = pd.DataFrame(d, columns=['seed', 'd']).groupby('seed').d.mean().values
    print('model (all four) %.2f vs rule %.2f collisions per run: %s' % (
        s.total.mean(), cs[cs.agent == 'rule'].collisions.mean(), ci(dd)))
    z = s[(s.cond == 'zeroshot') & s.instr.isin(['wet', 'weather'])].set_index(['instr', 'seed']).total
    dz = pd.Series([z[(i, sd)] - rule.loc[(i, sd)].collisions for i, sd in z.index], index=z.index)
    print('zero-shot on the trained instructions %.2f vs rule %.2f: %s' % (
        z.mean(), rule.loc[['wet', 'weather']].collisions.mean(), ci(dz.groupby(level='seed').mean().values)))
    for instr in ('wet', 'weather', 'north'):
        for c in CONDS:
            m = s[(s.cond == c) & (s.instr == instr)].set_index('seed').sort_index()
            r = rule.loc[instr].sort_index()
            print('   %-7s %-9s cost minus rule %s' % (instr, c, ci((m.cost - r.cost).values, '%+.0f')))
    for c in CONDS:
        w = s[(s.cond == c) & (s.instr == 'wet')].set_index('seed').sort_index()
        r_ = s[(s.cond == c) & (s.instr == 'weather')].set_index('seed').sort_index()
        print('   %-9s wet-day minus rain-day wet ground %s; rain minus wet worked %s pts' % (
            c, ci((w.wet - r_.wet).values, '%+.0f'), ci(100 * (r_.coverage - w.coverage).values, '%+.1f')))

    if a.latex:
        print('\n% LaTeX rows: one column per run')
        for label, what, f in (('All collisions', 'total', '%.2f'), ('Lane or gate', 'lane', '%.2f'),
                               ('Rock or pole', 'rock', '%.2f'), ('Field fence', 'fence', '%.2f'),
                               ('Runs with 3 or more', 'bad', '%d'), ('Runs with none', 'none', '%d'),
                               ('Field worked (\\%)', 'worked', '%.1f')):
            print(label + ' & ' + ' & '.join(f % stat(df, k, what) for k in keys) + ' \\\\')
        print('\\hline')
        for c in CONDS:
            print('%s & ' % NAMES[c] + ' & '.join('%.2f' % stat(df[df.cond == c], k, 'total') for k in keys) + ' \\\\')


if __name__ == '__main__':
    main()
