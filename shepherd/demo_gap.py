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


def animate(env, forms, path, model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    L = env.log
    X, Y, T = np.array(L['x']), np.array(L['y']), np.array(L['t'])
    FX, FY = np.array(L['fanX']), np.array(L['fanY'])
    idx = np.arange(0, min(len(T) - 1, len(forms)), 8)
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for (x, y, r) in COURSE.obstacles:
        ax.add_patch(plt.Circle((x, y), r, color='#b9b8b3', lw=0))
    ax.set_xlim(-25, 255); ax.set_ylim(-48, 48); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    trails = [ax.plot([], [], color='#2a78d6', lw=0.8, alpha=0.35)[0] for _ in range(FX.shape[1])]
    lt, = ax.plot([], [], color='#0b0b0b', lw=1.2, alpha=0.5)
    fans = ax.scatter(FX[0], FY[0], s=34, color='#2a78d6', edgecolor='white', lw=0.8, zorder=4)
    lead = ax.scatter([X[0]], [Y[0]], s=60, marker='s', color='#0b0b0b', zorder=5)
    txt = ax.text(0.01, 0.97, '', transform=ax.transAxes, va='top', fontsize=10)
    ax.text(0.99, 0.02, 'leader (square) + 8 followers, %s followers, %d collisions' % (model, len(env.collisions)),
            transform=ax.transAxes, ha='right', fontsize=8, color='#52514e')

    def draw(i):
        j = idx[i]; a = max(0, j - 400)
        for k, tr in enumerate(trails):
            tr.set_data(FX[a:j + 1, k], FY[a:j + 1, k])
        lt.set_data(X[a:j + 1], Y[a:j + 1])
        fans.set_offsets(np.column_stack([FX[j], FY[j]]))
        lead.set_offsets([[X[j], Y[j]]])
        txt.set_text('t = %3.0f s    formation: %s' % (T[j], forms[min(j, len(forms) - 1)]))
        return trails + [lt, fans, lead, txt]
    FuncAnimation(fig, draw, frames=len(idx)).save(path, writer=PillowWriter(fps=15), dpi=80)


if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else 'vfield'
    out = sys.argv[2] if len(sys.argv) > 2 else 'gap_%s.gif' % model
    runs = [run(model, s) for s in range(5)]
    print(model, 'collisions per seed', [len(e.collisions) for e, _ in runs],
          'formation switches', [e.formationSwitches for e, _ in runs])
    animate(*runs[0], out, model)
    print('wrote', out)
