"""Animate one farm-study episode from its saved trajectory and transcript.

    python -m shepherd.animate_farm TRANSCRIPT.json TRAJ.npz out.gif [--every 25] [--fps 15]

The trajectory (.npz, logged every 0.1 s) gives the machines; the transcript
gives the instruction, the field layout (seed) and every decision with the
model's stated reason. Worked ground is rebuilt the way FarmEnv scores it:
every 0.5 s, field cells within half a swath of a machine are marked.
"""
import argparse
import json
import textwrap

import numpy as np

from .farm import FarmSpec, buildFarm
from .farmenv import INSTRUCTIONS

LOG_DT = 0.1
NAMES = {'line': 'full line', 'lineLeft': 'half line, left', 'lineRight': 'half line, right', 'vee': 'vee',
         'column2': 'two abreast', 'column': 'single file', 'custom': 'custom'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('transcript')
    p.add_argument('traj')
    p.add_argument('out')
    p.add_argument('--spec', default='farms/default.json')
    p.add_argument('--every', type=int, default=25, help='log samples per frame (0.1 s each)')
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--title', default='AI foreman (qwen2.5:7b, zero-shot)')
    a = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle, Circle

    d = json.load(open(a.transcript))
    res = d['result']
    seed, instr = int(res['seed']), res['instruction']
    spec = FarmSpec.load(a.spec)
    course = buildFarm(spec, seed)
    tr = np.load(a.traj)
    X, Y, FX, FY = tr['x'], tr['y'], tr['fanX'], tr['fanY']
    n = len(X)
    T = np.arange(n) * LOG_DT
    L, W = course.field[1], course.field[3]
    strip = spec.northStrip

    # decisions in force over time
    turns = [t for t in d['turns'] if t.get('choice')]
    dT = np.array([t['t'] for t in turns])

    def decisionAt(t):
        k = np.searchsorted(dT, t, side='right') - 1
        return turns[k] if k >= 0 else None

    # worked ground, rebuilt as FarmEnv does (every 0.5 s = every 5 log samples)
    cs = 2.0
    gx, gy = np.meshgrid(np.arange(cs / 2, L, cs), np.arange(cs / 2, W, cs))
    cx, cy = gx.ravel(), gy.ravel()
    r = spec.swath / 2
    coveredAt = np.full(cx.size, np.inf)
    for i in range(0, n, 5):
        ax_ = np.concatenate([[X[i]], FX[i]])
        ay_ = np.concatenate([[Y[i]], FY[i]])
        m = (ax_ > 0) & (ax_ < L) & (ay_ > 0) & (ay_ < W)
        if not m.any():
            continue
        ax_, ay_ = ax_[m], ay_[m]
        todo = np.isinf(coveredAt)
        idx = np.nonzero(todo)[0]
        d2 = (cx[idx, None] - ax_[None, :]) ** 2 + (cy[idx, None] - ay_[None, :]) ** 2
        coveredAt[idx[(d2 < r * r).any(1)]] = T[i]
    # running totals for the counters
    northS = np.cumsum([(np.sum((np.concatenate([[X[i]], FX[i]]) > 0) & (np.concatenate([[X[i]], FX[i]]) < L)
                                & (np.concatenate([[Y[i]], FY[i]]) > W - strip) & (np.concatenate([[Y[i]], FY[i]]) < W)))
                        * LOG_DT for i in range(n)])
    colT = np.array(sorted(c['t'] for t in d['turns'] for c in (t['observation'].get('collisionsSinceLastCall') or [])))
    colXY = np.array([[c['x'], c['y']] for t in d['turns'] for c in (t['observation'].get('collisionsSinceLastCall') or [])]).reshape(-1, 2)
    nCell = cx.size

    # ---- figure ----
    fig = plt.figure(figsize=(8.0, 7.6))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.70])
    hx = fig.add_axes([0.02, 0.73, 0.96, 0.26]); hx.axis('off')
    xl, xh = X.min() - 8, max(L + 8, X.max() + 8)
    ax.set_xlim(xl, xh); ax.set_ylim(-48, W + 12); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(Rectangle((0, 0), L, W, fc='#f3efe4', ec='none', zorder=0))
    ax.add_patch(Rectangle((0, W - strip), L, strip, fc='#f4d6cf', ec='none', zorder=0.5))
    ax.text(L / 2, W - strip / 2, 'cattle side: 25 m strip along the north fence', ha='center', va='center',
            fontsize=8, color='#a0473a', zorder=1)
    img = ax.imshow(np.zeros((gy.shape[0], gy.shape[1], 4)), extent=(0, L, 0, W), origin='lower', zorder=1,
                    interpolation='nearest')
    for h in course.hazards:
        if h['type'] == 'slough':
            ax.add_patch(Circle((h['x'], h['y']), h['r'], fc='#9cc3e6', ec='#5b8fc0', lw=0.8, alpha=0.8, zorder=1.5))
            ax.text(h['x'], h['y'], 'slough', ha='center', va='center', fontsize=7, color='#2f5f8a', zorder=1.6)
    ob = course.obstacles
    ax.scatter(ob[:, 0], ob[:, 1], s=np.clip((ob[:, 2] * 2.2) ** 2, 1.5, 40), c='#8d8a82', lw=0, zorder=2)
    for h in course.hazards:
        if h['type'] in ('rock', 'pole'):
            ax.text(h['x'] + h['r'] + 2, h['y'] + 2, 'rock pile' if h['type'] == 'rock' else 'power pole',
                    fontsize=7, color='#55534d', zorder=2)
    ax.text(L / 2, W + 4, 'north fence', ha='center', fontsize=8, color='#55534d')
    ax.text(-60, -42, 'yard lane between shelterbelts', ha='center', fontsize=8, color='#55534d')
    trails = [ax.plot([], [], color='#2a78d6', lw=0.7, alpha=0.35, zorder=3)[0] for _ in range(FX.shape[1])]
    lt, = ax.plot([], [], color='#1d1c1a', lw=1.0, alpha=0.5, zorder=3)
    fans = ax.scatter(FX[0], FY[0], s=26, color='#2a78d6', edgecolor='white', lw=0.7, zorder=5)
    lead = ax.scatter([X[0]], [Y[0]], s=46, marker='s', color='#1d1c1a', edgecolor='white', lw=0.7, zorder=6)
    hits = ax.scatter([], [], s=60, marker='x', color='#c8372d', lw=1.6, zorder=7)

    hx.text(0.0, 0.97, a.title, fontsize=12, weight='bold', va='top', transform=hx.transAxes)
    hx.text(0.0, 0.80, 'Operator: "%s"' % INSTRUCTIONS[instr]['text'], fontsize=9.2, va='top', style='italic',
            transform=hx.transAxes, wrap=True)
    status = hx.text(0.0, 0.58, '', fontsize=9.5, va='top', family='monospace', transform=hx.transAxes)
    thought = hx.text(0.0, 0.30, '', fontsize=9, va='top', color='#3d3b36', transform=hx.transAxes)
    hx.text(1.0, 0.0, 'black square: leader (autopilot, Part I vector field)   blue: 8 followers   '
            'green: ground worked   x: collision', fontsize=7, ha='right', va='bottom', color='#6b6862',
            transform=hx.transAxes)

    frames = np.arange(0, n, a.every)
    green = np.array([0.42, 0.69, 0.36, 0.55])

    def draw(fi):
        j = frames[fi]
        t = T[j]
        a0 = max(0, j - 600)
        for k, trl in enumerate(trails):
            trl.set_data(FX[a0:j + 1, k], FY[a0:j + 1, k])
        lt.set_data(X[a0:j + 1], Y[a0:j + 1])
        fans.set_offsets(np.column_stack([FX[j], FY[j]]))
        lead.set_offsets([[X[j], Y[j]]])
        done = (coveredAt <= t).reshape(gy.shape)
        rgba = np.zeros(done.shape + (4,)); rgba[done] = green
        img.set_data(rgba)
        nh = int(np.sum(colT <= t))
        hits.set_offsets(colXY[:nh] if nh else np.empty((0, 2)))
        dc = decisionAt(t)
        if dc is not None:
            c = dc['choice']
            f = c['formation'] if isinstance(c['formation'], str) else 'custom'
            off = c.get('offsetCompass', c['offset'])
            lim = ((dc['observation'].get('pass') or {}).get('allowedOffsetM') or {}).get(f)
            if lim:                                    # what was applied: clipped to the allowed range
                off = float(np.clip(off, lim[0], lim[1]))
            elif not dc['observation'].get('pass'):
                off = 0.0                              # offsets only apply on a pass
            offs = '%.0f m %s' % (abs(off), 'north' if off > 0 else 'south') if abs(off) >= 0.5 else 'none'
            status.set_text('t = %4.0f s   %-18s  speed %.1f m/s   pass offset: %s\n'
                            'field worked %3.0f%%   time in cattle strip %4.0f machine-s   collisions %d'
                            % (t, NAMES.get(f, f), c['speed'], offs, 100 * np.sum(coveredAt <= t) / nCell,
                               northS[j], nh))
            import re
            m = re.findall(r'"thought"\s*:\s*"([^"]*)"', dc.get('reply2') or dc['reply'])
            thought.set_text('\n'.join(textwrap.wrap('AI: "%s"' % (m[-1] if m else ''), 105)[:2]))
        return trails + [lt, fans, lead, img, hits, status, thought]

    FuncAnimation(fig, draw, frames=len(frames), blit=False).save(a.out, writer=PillowWriter(fps=a.fps), dpi=72)
    print('wrote', a.out, len(frames), 'frames')


if __name__ == '__main__':
    main()
