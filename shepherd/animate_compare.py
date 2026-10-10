"""Side-by-side animation of the same farm job under two set-ups.

    python -m shepherd.animate_compare followers out.gif [dpi] [poster s]   # Part I boids vs vector-field followers
    python -m shepherd.animate_compare reflex out.gif      # vector field without and with the reflex

Each panel runs a non-LLM decider on the same field layout, so the only
difference is the one named in the panel titles.
"""
import sys

import numpy as np

from .farm import FarmSpec, buildFarm
from .farmenv import FarmEnv
from .farmagents import FixedColumn, Schedule, FarmRule
from .followers import followerSetup
from .run import runEpisode

PRESETS = {
    # name: (seed, decider, [(panel title, follower model), ...], time window s,
    #        view, layout, headline). view is (width following the leader, y low, y high)
    #        or ('fixed', x low, x high, y low, y high)
    'followers': (0, 'schedule', [('Before: Part I followers', 'paper'),
                                  ('After: vector-field followers', 'vfield')],
                  (215, 390), (125, 2, 118), 'cols',
                  'Same field, same plan: only how the followers hold their place changes'),
    'reflex': (3, 'column', [('Without reflex', 'vfield'), ('With reflex', 'reflex')],
               (262, 410), ('fixed', 30, 180, 38, 82), 'rows',
               'Single file down a pass with a rock pile right on the line'),
}


def runEnv(seed, decider, model, instruction='wet', spec='farms/default.json'):
    opts, kw = followerSetup(model)
    env = FarmEnv(buildFarm(FarmSpec.load(spec), seed), instruction=instruction, opts=opts, **kw)
    ag = {'column': FixedColumn, 'schedule': Schedule, 'rule': lambda: FarmRule('wet')}[decider]()
    runEpisode(env, ag)
    return env


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'followers'
    out = sys.argv[2] if len(sys.argv) > 2 else 'compare_%s.gif' % name
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 72
    poster = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    every = 20                                   # log samples (0.1 s) per frame
    seed, decider, panels, (t0, t1), view, layout, headline = PRESETS[name]
    fixed = view[0] == 'fixed'
    if fixed:
        _, fx0, fx1, yl, yh = view
        vw = fx1 - fx0
    else:
        vw, yl, yh = view

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle, Circle

    envs = [runEnv(seed, decider, m) for _, m in panels]
    course = envs[0].course
    L, W = course.field[1], course.field[3]
    swath = course.spec.swath
    logs = []
    for env in envs:
        lg = env.log
        T = np.array(lg['t']); X = np.array(lg['x']); Y = np.array(lg['y'])
        FX = np.array(lg['fanX']); FY = np.array(lg['fanY'])
        # heading rate of the followers, deg/s, smoothed over 10 s
        vx, vy = np.gradient(FX, axis=0), np.gradient(FY, axis=0)
        hd = np.unwrap(np.arctan2(vy, vx), axis=0)
        rate = np.abs(np.gradient(hd, axis=0)).mean(1) / (T[1] - T[0])
        rate = np.convolve(np.rad2deg(rate), np.ones(100) / 100, mode='same')
        cols = sorted((c['t'], c['x'], c['y']) for c in env.collisions)
        refl = sorted((e['t'], e['x'], e['y']) for e in getattr(env, 'reflexEvents', []))
        sl = env.course.sloughs                      # machines in the wet spot: slowed, and the ruts they leave
        AX = np.concatenate([X[:, None], FX], 1); AY = np.concatenate([Y[:, None], FY], 1)
        inS = (np.hypot(AX[..., None] - sl[:, 0], AY[..., None] - sl[:, 1]) < sl[:, 2]).any(-1) if len(sl) else \
            np.zeros(AX.shape, bool)
        k = np.arange(0, len(T), 5)
        rutT = np.repeat(T[k], inS[k].sum(1))
        rutXY = np.column_stack([AX[k][inS[k]], AY[k][inS[k]]])
        logs.append(dict(T=T, X=X, Y=Y, FX=FX, FY=FY, rate=rate, cols=cols, refl=refl, inS=inS, rutT=rutT,
                         rutXY=rutXY))
    # worked ground, as FarmEnv marks it, per panel
    cs = 2.0
    gx, gy = np.meshgrid(np.arange(cs / 2, L, cs), np.arange(cs / 2, W, cs))
    cx, cy = gx.ravel(), gy.ravel()
    for lg in logs:
        at = np.full(cx.size, np.inf)
        for i in range(0, len(lg['T']), 5):
            ax_ = np.concatenate([[lg['X'][i]], lg['FX'][i]]); ay_ = np.concatenate([[lg['Y'][i]], lg['FY'][i]])
            m = (ax_ > 0) & (ax_ < L) & (ay_ > 0) & (ay_ < W)
            if not m.any():
                continue
            idx = np.nonzero(np.isinf(at))[0]
            d2 = (cx[idx, None] - ax_[m][None, :]) ** 2 + (cy[idx, None] - ay_[m][None, :]) ** 2
            at[idx[(d2 < (swath / 2) ** 2).any(1)]] = lg['T'][i]
        lg['covAt'] = at

    if layout == 'cols':
        fig = plt.figure(figsize=(8.0, 4.9))
        boxes = [[0.01, 0.05, 0.48, 0.72], [0.51, 0.05, 0.48, 0.72]]
    else:
        fig = plt.figure(figsize=(8.0, 5.6))
        boxes = [[0.02, 0.50, 0.96, 0.32], [0.02, 0.06, 0.96, 0.32]]
    fig.text(0.02, 0.975, headline, fontsize=12, weight='bold', va='top')
    axes, art = [], []
    for k, ((title, model), lg) in enumerate(zip(panels, logs)):
        ax = fig.add_axes(boxes[k])
        axes.append(ax)
        ax.set_xlim(0, vw); ax.set_ylim(yl, yh); ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.add_patch(Rectangle((0, 0), L, W, fc='#f3efe4', ec='none', zorder=0))
        img = ax.imshow(np.zeros(gy.shape + (4,)), extent=(0, L, 0, W), origin='lower', zorder=1,
                        interpolation='nearest')
        for h in course.hazards:
            if h['type'] == 'slough':
                ax.add_patch(Circle((h['x'], h['y']), h['r'], fc='#9cc3e6', ec='#5b8fc0', lw=0.8, alpha=0.8, zorder=1.5))
                if yl < h['y'] < yh:
                    ax.text(h['x'], h['y'] + h['r'] + 1.5, 'wet spot (slough)', ha='center', va='bottom', fontsize=8,
                            color='#2f5f8a', zorder=2)
        ob = course.obstacles
        ax.scatter(ob[:, 0], ob[:, 1], s=np.clip((ob[:, 2] * 2.2) ** 2, 1.5, 40), c='#8d8a82', lw=0, zorder=2)
        for h in course.hazards:
            if h['type'] in ('rock', 'pole') and yl < h['y'] < yh:
                ax.text(h['x'] + h['r'] + 2, h['y'] + 2, 'rock pile' if h['type'] == 'rock' else 'power pole',
                        fontsize=8, color='#55534d', zorder=2)
        if layout == 'cols':          # narrow panels: title above, numbers below it
            ax.text(0.0, 1.10, title, transform=ax.transAxes, fontsize=10.5, weight='bold', va='bottom')
            stat = ax.text(0.0, 1.02, '', transform=ax.transAxes, fontsize=9, va='bottom', ha='left',
                           family='monospace', color='#3d3b36')
        else:
            ax.text(0.0, 1.02, title, transform=ax.transAxes, fontsize=10.5, weight='bold', va='bottom')
            stat = ax.text(1.0, 1.02, '', transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
                           family='monospace', color='#3d3b36')
        ruts = ax.scatter([], [], s=6, marker='s', color='#6b4423', alpha=0.45, lw=0, zorder=2.5)
        slow = ax.text(0, 0, '', fontsize=8.5, color='#a0581c', weight='bold', ha='center', va='top', zorder=8,
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))
        trails = [ax.plot([], [], color='#2a78d6', lw=0.9, alpha=0.55, zorder=3)[0] for _ in range(lg['FX'].shape[1])]
        fans = ax.scatter([], [], s=24, color='#2a78d6', edgecolor='white', lw=0.7, zorder=5)
        lead = ax.scatter([], [], s=40, marker='s', color='#1d1c1a', edgecolor='white', lw=0.7, zorder=6)
        hits = ax.scatter([], [], s=55, marker='x', color='#c8372d', lw=1.6, zorder=7)
        dodge = ax.scatter([], [], s=70, facecolor='none', edgecolor='#e0a100', lw=1.4, zorder=7)
        art.append(dict(img=img, stat=stat, trails=trails, fans=fans, lead=lead, hits=hits, dodge=dodge, ruts=ruts,
                        slow=slow))
    fig.text(0.98, 0.012, 'black square: leader   blue: followers and their tracks   green: ground worked   x: collision'
             + ('   orange ring: reflex swerve' if any(m == 'reflex' for _, m in panels) else '')
             + ('   brown: slowed in the wet spot, and its ruts' if any(l['inS'].any() for l in logs) else ''),
             fontsize=7, ha='right', color='#6b6862')

    green = np.array([0.42, 0.69, 0.36, 0.5])
    T = logs[0]['T']
    frames = np.nonzero((T >= t0) & (T <= t1))[0][::every]
    if poster > 0:                   # what a still preview shows: the outcome, then the replay
        frames = np.concatenate([np.full(int(poster * 15), frames[-1]), frames])

    def draw(fi):
        j = frames[fi]
        t = T[j]
        out = []
        cx0 = fx0 if fixed else logs[0]['X'][j] - 0.62 * vw   # camera fixed, or following the leader
        for ax in axes:
            ax.set_xlim(cx0, cx0 + vw)
        for lg, a in zip(logs, art):
            jj = min(j, len(lg['T']) - 1)
            a0 = max(0, jj - 500)
            for k, tr in enumerate(a['trails']):
                tr.set_data(lg['FX'][a0:jj + 1, k], lg['FY'][a0:jj + 1, k])
            ins = lg['inS'][jj]
            a['fans'].set_offsets(np.column_stack([lg['FX'][jj], lg['FY'][jj]]))
            a['fans'].set_facecolors(['#a0581c' if v else '#2a78d6' for v in ins[1:]])
            a['lead'].set_offsets([[lg['X'][jj], lg['Y'][jj]]])
            a['lead'].set_facecolors(['#a0581c' if ins[0] else '#1d1c1a'])
            m = lg['rutT'] <= t
            a['ruts'].set_offsets(lg['rutXY'][m] if m.any() else np.empty((0, 2)))
            sl = course.sloughs
            if ins.any() and len(sl):
                a['slow'].set_position((sl[0][0], sl[0][1] - sl[0][2] - 1.5))
                a['slow'].set_text('%d slowed to %.1f m/s' % (ins.sum(), course.spec.wetSpeed))
            else:
                a['slow'].set_text('')
            done = (lg['covAt'] <= t).reshape(gy.shape)
            rgba = np.zeros(done.shape + (4,)); rgba[done] = green
            a['img'].set_data(rgba)
            ch = [c for c in lg['cols'] if c[0] <= t]
            a['hits'].set_offsets(np.array([[c[1], c[2]] for c in ch]) if ch else np.empty((0, 2)))
            rf = [e for e in lg['refl'] if t - 8 <= e[0] <= t]
            a['dodge'].set_offsets(np.array([[e[1], e[2]] for e in rf]) if rf else np.empty((0, 2)))
            a['stat'].set_text('weaving %4.1f deg/s   collisions %d' % (lg['rate'][jj], len(ch)))
            out += a['trails'] + [a['fans'], a['lead'], a['img'], a['hits'], a['dodge'], a['stat'], a['ruts'],
                                  a['slow']]
        return out

    FuncAnimation(fig, draw, frames=len(frames)).save(out, writer=PillowWriter(fps=15), dpi=dpi)
    print('wrote', out, len(frames), 'frames;', ', '.join('%s: %d collisions' % (p[1], len(l['cols']))
                                                      for p, l in zip(panels, logs)))


if __name__ == '__main__':
    main()
