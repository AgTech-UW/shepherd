"""The safety-filter run against the reflex run: language-model conditions and baselines,
paired by field layout.

    python experiments/shield_compare.py [REFLEX_DIR] [SHIELD_DIR]

Defaults: results/farm_study_reflex and results/farm_study_reflex_shield. Collisions are
classified by place as in reflex_llm_compare.py. From the transcripts it also counts how often
the model chose a formation listed as not fitting, and how often the filter held a choice back.
Writes shield_vs_reflex_runs.csv into SHIELD_DIR.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from shepherd.farm import FarmSpec, buildFarm   # noqa: E402
from reflex_llm_compare import load, boot, KINDS, TAG   # noqa: E402

CONDS = ('zeroshot', 'whatif', 'lessons', 'memory')
NAMES = dict(zeroshot='zero-shot', whatif='what-if', lessons='lessons', memory='memory',
             column='single file', schedule='schedule', rule='rule', search='oracle')


def fitStats(study_dir):
    rows = []
    for f in glob.glob(os.path.join(study_dir, 'transcripts', TAG, '*_test_s*.json')):
        cond, instr, _, s = os.path.basename(f)[:-5].split('_')
        d = json.load(open(f))
        n = bad = 0
        for t in d['turns']:
            ch = (t.get('choice') or {}).get('formation')
            fit = t['observation'].get('formationsThatFit') or {}
            if ch and ch in fit:
                n += 1
                bad += not fit[ch]
        sh = d.get('shield') or []
        rows.append(dict(cond=cond, instr=instr, seed=int(s[1:]), decisions=n, notFitting=bad,
                         held=sum('held' in e for e in sh), released=sum('released' in e for e in sh),
                         time=d['result']['time']))
    return pd.DataFrame(rows)


def ci(x):
    m, lo, hi = boot(x)
    return '%+.2f [%+.2f, %+.2f]' % (m, lo, hi)


def paired(df, col, a='reflex', b='shield', keys=('cond', 'instr', 'seed')):
    p = df.pivot_table(index=list(keys), columns='study', values=col)
    return (p[b] - p[a]).groupby(level='seed').mean().values


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'farm_study_reflex')
    sdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'results', 'farm_study_reflex_shield')
    sp = FarmSpec.load(os.path.join(ROOT, 'farms', 'default.json'))
    courses = {s: buildFarm(sp, s) for s in range(8)}
    df = pd.DataFrame(load(rdir, 'reflex', courses) + load(sdir, 'shield', courses))
    fs = pd.concat([fitStats(rdir).assign(study='reflex'), fitStats(sdir).assign(study='shield')])
    df = df.merge(fs, on=['study', 'cond', 'instr', 'seed'])
    df['classified'] = df[list(KINDS)].sum(axis=1)
    df.to_csv(os.path.join(sdir, 'shield_vs_reflex_runs.csv'), index=False)
    print('runs', df.groupby('study').size().to_dict(), '| collisions classified / total',
          df.groupby('study').classified.sum().to_dict(), df.groupby('study').total.sum().to_dict())

    print('\n== Language model: reflex -> reflex + filter, per run (paired difference by layout [95% CI])')
    for cond in CONDS + ('ALL',):
        sub = df if cond == 'ALL' else df[df.cond == cond]
        g = sub.groupby('study')
        z = sub.assign(z=sub.total == 0).groupby('study').z.sum()
        print('%-9s collisions %.2f -> %.2f %s | zero-collision runs %d -> %d of %d' % (
            cond, g.total.mean()['reflex'], g.total.mean()['shield'], ci(paired(sub, 'total')),
            z['reflex'], z['shield'], len(sub) // 2))
        for k in KINDS:
            print('            %-11s %.2f -> %.2f' % (k, g[k].mean()['reflex'], g[k].mean()['shield']))
        print('            worked %.1f%% -> %.1f%% %s pts | time %.0f -> %.0f s' % (
            100 * g.coverage.mean()['reflex'], 100 * g.coverage.mean()['shield'],
            ci(100 * paired(sub, 'coverage')), g.time.mean()['reflex'], g.time.mean()['shield']))
        print('            chose a non-fitting formation %.1f%% -> %.1f%% of decisions | filter holds per run %.1f, releases %.1f' % (
            100 * g.notFitting.sum()['reflex'] / g.decisions.sum()['reflex'],
            100 * g.notFitting.sum()['shield'] / g.decisions.sum()['shield'],
            g.held.mean()['shield'], g.released.mean()['shield']))

    print('\n== By instruction, filter run: collisions, worked %, wet m-s, north m-s, cost')
    s = df[df.study == 'shield']
    print(s.groupby(['instr', 'cond'])[['total', 'coverage', 'wet', 'north', 'cost']].mean().round(2).to_string())
    print('\nnorth-fence day, fields with no time in the strip:')
    for st in ('reflex', 'shield'):
        n = df[(df.study == st) & (df.instr == 'north')]
        print('  ', st, {c: int((n[n.cond == c].north == 0).sum()) for c in CONDS})
    print('\nwet day, wet ground (machine-s), reflex -> filter, paired:')
    for c in CONDS:
        sub = df[(df.cond == c) & (df.instr == 'wet')]
        g = sub.groupby('study').wet.mean()
        print('   %-9s %.0f -> %.0f %s' % (c, g['reflex'], g['shield'], ci(paired(sub, 'wet'))))

    print('\n== Baselines: reflex -> reflex + filter (per run, all three instructions)')
    cr = pd.read_csv(os.path.join(rdir, 'comparators', 'episodes.csv')).assign(study='reflex')
    cs = pd.read_csv(os.path.join(sdir, 'comparators', 'episodes.csv')).assign(study='shield')
    cb = pd.concat([cr, cs])
    cb = cb[cb.phase == 'test'] if 'phase' in cb else cb
    for ag in ('column', 'schedule', 'rule', 'search'):
        sub = cb[cb.agent == ag]
        g = sub.groupby('study')
        p = sub.pivot_table(index=['instruction', 'seed'], columns='study', values=['collisions', 'coverage'])
        dcol = (p[('collisions', 'shield')] - p[('collisions', 'reflex')]).groupby(level='seed').mean().values
        dcov = 100 * (p[('coverage', 'shield')] - p[('coverage', 'reflex')]).groupby(level='seed').mean().values
        print('%-9s collisions %.2f -> %.2f %s | worked %.1f%% -> %.1f%% %s | time %.0f -> %.0f s' % (
            NAMES[ag], g.collisions.mean()['reflex'], g.collisions.mean()['shield'], ci(dcol),
            100 * g.coverage.mean()['reflex'], 100 * g.coverage.mean()['shield'], ci(dcov),
            g.time.mean()['reflex'], g.time.mean()['shield']))
    print('\nbaselines with filter, by instruction (collisions, worked, wet, north, cost):')
    print(cs.groupby(['instruction', 'agent'])[['collisions', 'coverage', 'sloughExposure', 'northExposure', 'cost']]
          .mean().round(2).to_string())

    print('\n== Filter run: model condition minus hand-written rule, paired by layout')
    rule = cs[cs.agent == 'rule'].set_index(['instruction', 'seed'])
    for instr in ('wet', 'weather', 'north'):
        for c in CONDS:
            m = s[(s.cond == c) & (s.instr == instr)].set_index('seed').sort_index()
            r = rule.loc[instr].sort_index()
            print('  %-7s %-9s collisions %s | cost %s' % (
                instr, c, ci((m.total - r.collisions).values), ci((m.cost - r.cost).values)))
    print('  all     all four  collisions model %.2f vs rule %.2f per run' % (
        s.total.mean(), cs[cs.agent == 'rule'].collisions.mean()))


if __name__ == '__main__':
    main()
