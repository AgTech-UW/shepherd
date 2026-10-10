"""Gap in a wall: a vee folds to single file for a 16 m gap and opens back
up once the tail is through (hand-written width rule), with a choice of
follower model. Prints collisions over 5 seeds and writes an animation.

    python -m shepherd.demo_gap [vfield|paper|damped] [out.gif]
"""
import sys
import numpy as np

from .courses import Course, _wall
from .env import ShepherdEnv
from .agents import AdaptiveFormationAgent, PRESETS
from .followers import followerSetup

COURSE = Course(name='gap', start=(0.0, 0.0, 0.0), goal=(250.0, 0.0, 0.0),
                obstacles=np.vstack([_wall(70, 8, 70, 45), _wall(70, -45, 70, -8)]),
                route=[(40.0, 0.0), (100.0, 0.0), (220.0, 0.0)], bounds=(-30, 260, -50, 50),
                description='Wall with a 16 m gap at x = 70')
NAME = {'vee': 'vee', 'column2': 'two abreast', 'column': 'single file'}


def run(model, seed):
    o, kw = followerSetup(model)
    env = ShepherdEnv(COURSE, nFan=8, seed=seed, opts=o, formation=dict(PRESETS['vee']), logEvery=10, **kw)
    ag, forms = AdaptiveFormationAgent(), []
    obs = env.reset()
    ag.begin(env)
    while not env.done:
        env.setPlan(ag.act(env, obs))
        for _ in range(int(env.decisionDt / env.opts.dt)):
            if env.done:
                break
            env.stepDt()
            if env.steps % env.logEvery == 0:
                f = env.formation
                forms.append(next((NAME[k] for k, v in PRESETS.items() if v == f and k in NAME), f.get('shape')))
        obs = env.observe()
    return env, forms


def animate(env, forms, path, model, dpi=110, poster=2.0):
    """Poster frame first (single file in the gap), then the run; formation in large type."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    L = env.log
    X, Y, T = np.array(L['x']), np.array(L['y']), np.array(L['t'])
    FX, FY = np.array(L['fanX']), np.array(L['fanY'])
    n = min(len(T) - 1, len(forms))
    idx = list(np.arange(0, n, 20))
    pj = int(np.argmax(X >= 150))                      # leader well past the wall, tail coming through the gap
    idx = [pj] * int(poster * 15) + idx
    change = [0] + [i for i in range(1, n) if forms[i] != forms[i - 1]]
    fig = plt.figure(figsize=(8.0, 3.7))
    fig.text(0.012, 0.975, 'A vee folds into single file for a 16 m gap, then opens back up', fontsize=12.5,
             weight='bold', va='top', color='#1d1c1a')
    fig.text(0.012, 0.885, 'Every tractor is steered by the Part I vector field; the boid separation rule keeps '
             'them apart', fontsize=9.5, va='top', color='#55534d')
    ax = fig.add_axes([0.01, 0.07, 0.98, 0.72])
    ax.set_facecolor('#f6f2e7')
    for (x, y, r) in COURSE.obstacles:
        ax.add_patch(plt.Circle((x, y), r, color='#8d8a82', lw=0))
    ax.set_xlim(-25, 255); ax.set_ylim(-50, 56); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color('#d8d3c4')
    ax.text(70, 49, 'wall with a 16 m gap', ha='center', va='center', fontsize=9.5, color='#55534d', weight='bold')
    trails = [ax.plot([], [], color='#2a78d6', lw=0.8, alpha=0.4)[0] for _ in range(FX.shape[1])]
    lt, = ax.plot([], [], color='#1d1c1a', lw=1.0, alpha=0.5)
    fans = ax.scatter(FX[0], FY[0], s=34, color='#2a78d6', edgecolor='white', lw=0.8, zorder=4)
    lead = ax.scatter([X[0]], [Y[0]], s=60, marker='s', color='#1d1c1a', edgecolor='white', lw=0.8, zorder=5)
    form = ax.text(0.012, 0.965, '', transform=ax.transAxes, va='top', fontsize=15, weight='bold', zorder=9,
                   bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#1d1c1a', lw=1.0))
    fig.text(0.985, 0.012, 'black square: lead tractor   blue: 8 follower tractors and their tracks   '
             '%s followers, %d collisions' % (model, len(env.collisions)), fontsize=7, ha='right', color='#6b6862')

    def draw(i):
        j = idx[i]; a = max(0, j - 400)
        for k, tr in enumerate(trails):
            tr.set_data(FX[a:j + 1, k], FY[a:j + 1, k])
        lt.set_data(X[a:j + 1], Y[a:j + 1])
        fans.set_offsets(np.column_stack([FX[j], FY[j]]))
        lead.set_offsets([[X[j], Y[j]]])
        f = forms[min(j, len(forms) - 1)]
        form.set_text(f.upper())
        last = max(c for c in change if c <= j)
        form.get_bbox_patch().set_facecolor('#ffd35c' if last > 0 and j - last < 60 else 'white')
        return trails + [lt, fans, lead, form]
    FuncAnimation(fig, draw, frames=len(idx)).save(path, writer=PillowWriter(fps=15), dpi=dpi)


if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else 'vfield'
    out = sys.argv[2] if len(sys.argv) > 2 else 'gap_%s.gif' % model
    runs = [run(model, s) for s in range(5)]
    print(model, 'collisions per seed', [len(e.collisions) for e, _ in runs],
          'formation switches', [e.formationSwitches for e, _ in runs])
    animate(*runs[0], out, model)
    print('wrote', out)
