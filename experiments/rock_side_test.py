"""One follower, one rock: the reflex side rule on its own. The follower's slot line runs at a
sideways offset from the rock's centre and it starts 40 m before the rock, with the same steering,
reflex (15 m look, 2.5 m margin) and turn limit (5 m radius) as the farm, no neighbours, at 0.8 m/s.
Optionally the wanted heading jitters (5 degrees, new value every 0.5 s), as slot reassignment does.
Reflex version 1 against version 2 (committed side); version 3 only adds a rule about neighbours, so
with one machine it is version 2.

    python experiments/rock_side_test.py [--out results/rock_side/fig_rock_side.pdf]

Prints the closest clearance (body to rock surface) and the number of side switches for each offset,
and draws the figure.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from grainframe import SimOptions                          # noqa: E402
from grainframe.dynamics import ackermannClamp             # noqa: E402
from shepherd.followers import reflexHeading               # noqa: E402

ROCK = (40.0, 0.0, 2.9)
OFFSETS = np.arange(-4.0, 4.01, 0.5)


class _Course:
    def __init__(self, ob):
        self.obstacles = np.array(ob, float)


class _Env:
    def __init__(self, version):
        self.course = _Course([ROCK, (500.0, 500.0, 0.8)])
        self.opts = SimOptions()
        self.opts.minSpeed = 0.3
        self.nFan = 1
        self.x, self.y = -1e4, 0.0                         # leader far away
        self.reflexLook, self.reflexMargin, self.reflexVersion = 15.0, 2.5, version


def run(version, off, jitter=0.0, speed=0.8, seed=0, dt=0.01):
    rng = np.random.default_rng(seed)
    env = _Env(version)
    x, y, vx, vy = 0.0, off, speed, 0.0
    gmin, switches, last, jit = np.inf, 0, 0, 0.0
    for k in range(int(120 / dt)):
        env.fanX, env.fanY = np.array([x]), np.array([y])
        env.fanVx, env.fanVy = np.array([vx]), np.array([vy])
        if jitter and k % 50 == 0:
            jit = rng.normal(0.0, np.deg2rad(jitter))
        chi = -np.deg2rad(60) * (2 / np.pi) * np.arctan(0.5 * (y - off)) + jit     # vector field, Eq. (vfield)
        h, sp, w = reflexHeading(env, np.array([[x, y]]), np.array([chi]), np.array([speed]),
                                 ids=np.array([0]), hAct=np.array([np.arctan2(vy, vx)]))
        side = int(np.sign((h[0] - chi + np.pi) % (2 * np.pi) - np.pi)) if w[0] > 0.02 else 0
        if side and last and side != last:
            switches += 1
        last = side or last
        vfx, vfy = sp[0] * np.cos(h[0]), sp[0] * np.sin(h[0])
        nvx, nvy, _ = ackermannClamp(np.array(vx), np.array(vy), np.array(vx + 5.0 * (vfx - vx) * dt),
                                     np.array(vy + 5.0 * (vfy - vy) * dt), env.opts)
        vx, vy = float(nvx), float(nvy)
        x, y = x + vx * dt, y + vy * dt
        gmin = min(gmin, np.hypot(x - ROCK[0], y - ROCK[1]) - ROCK[2] - env.opts.bodyRadius)
        if x > ROCK[0] + 30:
            break
    return gmin, switches


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=os.path.join(ROOT, 'results', 'rock_side', 'fig_rock_side.pdf'))
    a = p.parse_args()
    res = {}
    for version in (1, 2):
        for jitter in (0.0, 5.0):
            r = [run(version, o, jitter, seed=int(10 * o + 100)) for o in OFFSETS]
            res[(version, jitter)] = r
            print('version %d, jitter %g deg: clearance %s | hits %d, side switches %d' % (
                version, jitter, ' '.join('%.2f' % g for g, _ in r), sum(g < 0 for g, _ in r),
                sum(s for _, s in r)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted = '#0b0b0b', '#52514e'
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ax.axhspan(-3, 0, color='#f2f1ee', lw=0)
    ax.axhline(0, color=muted, lw=0.8)
    ax.text(3.9, -2.75, 'contact', ha='right', va='bottom', fontsize=8, color=muted)
    for (version, jitter), col, ls, mk, lab in (
            ((1, 0.0), '#eb6834', '-', 'o', 'version 1'),
            ((1, 5.0), '#eb6834', '--', 's', 'version 1, heading jitter'),
            ((2, 0.0), '#2a78d6', '-', 'o', 'version 2 (with or without jitter)')):
        ax.plot(OFFSETS, [g for g, _ in res[(version, jitter)]], color=col, ls=ls, marker=mk, ms=3.5, lw=1.6,
                label=lab)
    ax.set_xlabel('slot line offset from the rock centre (m)', color=ink)
    ax.set_ylabel('closest clearance (m)', color=ink)
    ax.set_ylim(-2.8, 3.2)
    ax.axhline(2.5, color=muted, lw=0.6, ls=':')
    ax.text(-3.9, 2.6, 'margin 2.5 m', fontsize=8, color=muted, va='bottom')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
