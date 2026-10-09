"""Summarise episodes.csv files: mean per agent x course.

  python -m shepherd.compare results/shepherd/episodes.csv results/shepherd/cem*/episodes.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np

COLS = ['captured', 'posErr', 'headErrDeg', 'collisions', 'fan-leader', 'fan-fan',
        'fan-obstacle', 'leader-obstacle', 'fansLost', 'time', 'score', 'wall_s']


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    return rows


def summarise(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r['course'], r['agent'])].append(r)
    out = []
    for (course, agent), rs in sorted(g.items()):
        d = dict(course=course, agent=agent, n=len(rs))
        for c in COLS:
            v = [float(r[c] == 'True') if c == 'captured' else float(r[c]) for r in rs if r.get(c) not in (None, '')]
            d[c] = np.mean(v) if v else np.nan
        out.append(d)
    return out


def main():
    rows = load(sys.argv[1:])
    S = summarise(rows)
    hdr = ['course', 'agent', 'n'] + COLS
    w = [8, 22, 3] + [9] * len(COLS)
    print(' '.join(h[:x].ljust(x) for h, x in zip(hdr, w)))
    for d in S:
        cells = [d['course'], d['agent'], str(d['n'])] + ['%.2f' % d[c] for c in COLS]
        print(' '.join(c[:x].ljust(x) for c, x in zip(cells, w)))
    return S


if __name__ == '__main__':
    main()
