"""Table of the two layers on the deterministic deciders (wet day, reflex followers, 8 layouts):
collisions, field worked and time to finish for reflex only, + safety filter, + leader wait, + both.

    python experiments/layer_table.py

Every episode is deterministic, so the cells come from whichever run produced them:
  schedule, rule, reflex only / + filter  results/farm_study_reflex[_shield]/comparators/episodes.csv
  schedule, rule, + leader wait           results/shepherd/layer_tests/pearls.csv
  widens too early, reflex only           results/shepherd/layer_tests/eager.csv (shield off rows)
  widens too early, + leader wait         results/shepherd/layer_tests/eager2.csv (shield off rows)
  every cell with the filter and the
  test decider or the leader wait         results/shepherd/layer_tests/layers_*_on.csv
(The filter-on rows of eager.csv and eager2.csv came from an earlier version of the filter and are
not used.) experiments/layer_tests.py regenerates any cell.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
LT = os.path.join(R, 'shepherd', 'layer_tests')


def cell(df):
    assert len(df) == 8, len(df)
    return dict(collisions=df.collisions.mean(), laneGate=df.laneGate.mean() if 'laneGate' in df else float('nan'),
                coverage=100 * df.coverage.mean(), time=df.time.mean())


def main():
    cells = {}
    for st, key in (('farm_study_reflex', 'reflex'), ('farm_study_reflex_shield', 'filter')):
        e = pd.read_csv(os.path.join(R, st, 'comparators', 'episodes.csv'))
        e = e[e.instruction == 'wet']
        for ag in ('schedule', 'rule'):
            cells[(ag, key)] = cell(e[e.agent == ag])
    p = pd.read_csv(os.path.join(LT, 'pearls.csv'))
    for ag in ('schedule', 'rule'):
        cells[(ag, 'wait')] = cell(p[(p.decider == ag) & (p.model == 'pearls')])
    e1 = pd.read_csv(os.path.join(LT, 'eager.csv'))
    cells[('eager', 'reflex')] = cell(e1[(e1.decider == 'eager') & (~e1.shield)])
    e2 = pd.read_csv(os.path.join(LT, 'eager2.csv'))
    cells[('eager', 'wait')] = cell(e2[(e2.decider == 'eager') & (~e2.shield)])
    parts = [pd.read_csv(os.path.join(LT, f)) for f in ('layers_eager_on.csv', 'layers_pearls_on.csv')
             if os.path.exists(os.path.join(LT, f))]
    for (dec, model), g in (pd.concat(parts).groupby(['decider', 'model']) if parts else []):
        cells[(dec, 'filter' if model == 'reflex' else 'both')] = cell(g)
    names = dict(rule='Hand-written rule', schedule='Schedule (hazard-blind)', eager='Widens too early (test)')
    print('%-26s %-22s %-22s %-22s %-22s' % ('', 'reflex only', '+ filter', '+ leader wait', '+ both'))
    for dec in ('rule', 'schedule', 'eager'):
        row = []
        for k in ('reflex', 'filter', 'wait', 'both'):
            c = cells.get((dec, k))
            row.append('--' if c is None else '%.2f, %.0f\\%%, %.0f s' % (c['collisions'], c['coverage'], c['time']))
        print('%s & %s \\\\' % (names[dec], ' & '.join(row)))
    print()
    for k, c in sorted(cells.items()):
        print(k, {a: round(b, 2) for a, b in c.items()})


if __name__ == '__main__':
    main()
