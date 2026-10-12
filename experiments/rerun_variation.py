"""How much do two runs of the same model study differ? The safety-filter run (fit check 1) on the
desktop against the same configuration on the laptop (67 of 96 test runs; it stopped when the
embedding model timed out), matched cell by cell (condition, instruction, layout).

    python experiments/rerun_variation.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from shepherd.farm import FarmSpec, buildFarm      # noqa: E402
from reflex_llm_compare import load, boot          # noqa: E402


def main():
    sp = FarmSpec.load(os.path.join(ROOT, 'farms', 'default.json'))
    courses = {s: buildFarm(sp, s) for s in range(8)}
    a = pd.DataFrame(load(os.path.join(ROOT, 'results', 'farm_study_reflex_shield'), 'desktop', courses))
    b = pd.DataFrame(load(os.path.join(ROOT, 'results', 'farm_study_reflex_shield_laptop'), 'laptop', courses))
    m = a.merge(b, on=['cond', 'instr', 'seed'], suffixes=('_d', '_l'))
    m.to_csv(os.path.join(ROOT, 'results', 'farm_study_reflex_shield_laptop', 'rerun_pairs.csv'), index=False)
    print('matched cells:', len(m), m.groupby('cond').size().to_dict(), m.groupby('instr').size().to_dict())
    for cond in ('zeroshot', 'whatif', 'lessons', 'memory', 'ALL'):
        s = m if cond == 'ALL' else m[m.cond == cond]
        if not len(s):
            continue
        d = (s.total_l - s.total_d).groupby(s.seed).mean().values
        mm, lo, hi = boot(d)
        print('%-8s n %2d | collisions desktop %.2f laptop %.2f (laptop - desktop %+.2f [%+.2f, %+.2f]) | '
              'same count in %d of %d cells | mean |diff| %.2f | worked %.1f vs %.1f | north %.0f vs %.0f' % (
                  cond, len(s), s.total_d.mean(), s.total_l.mean(), mm, lo, hi, int((s.total_d == s.total_l).sum()),
                  len(s), (s.total_d - s.total_l).abs().mean(), 100 * s.coverage_d.mean(), 100 * s.coverage_l.mean(),
                  s.north_d.mean(), s.north_l.mean()))
    for k in ('rock/pole', 'lane/gate', 'field fence'):
        print('   %-11s desktop %.2f laptop %.2f' % (k, m[k + '_d'].mean(), m[k + '_l'].mean()))
    print('runs with 3+: desktop %d laptop %d; with none: desktop %d laptop %d' % (
        (m.total_d >= 3).sum(), (m.total_l >= 3).sum(), (m.total_d == 0).sum(), (m.total_l == 0).sum()))
    print('correlation of per-cell collisions %.2f, of field worked %.2f' % (
        np.corrcoef(m.total_d, m.total_l)[0, 1], np.corrcoef(m.coverage_d, m.coverage_l)[0, 1]))
    n = m[m.instr == 'north']
    print('north-fence day, cells with no time in the strip: desktop %d laptop %d of %d' % (
        (n.north_d == 0).sum(), (n.north_l == 0).sum(), len(n)))


if __name__ == '__main__':
    main()
