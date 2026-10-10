"""Story animations for the demo deck: one idea per animation.

    python -m shepherd.animate_story formation out.gif [--dpi 110] [--poster 2] [--cache DIR]
    python -m shepherd.animate_story wetspot   out.gif
    python -m shepherd.animate_story cattle    out.gif

Each panel is either a recorded language-model run from the study, replayed
exactly from its transcript (replay.py), or a non-LLM decider run on the same
field layout. What is drawn:

  - the formation in force, written large, lit up for a few seconds after a change
  - the AI's own stated reason for its latest formation change (or latest decision)
  - the wet spot (slough): a machine inside it is slowed to the spec's wet
    speed and drawn brown, and the ground it churns is left marked
  - the cattle strip along the north fence on cattle-day panels
  - counters for the one thing the story is about

Two panels can be aligned on the moment each leader passes the wet spot
(align='slough'), so runs that got there at different times play side by side.
"""
import argparse
import json
import os
import pickle
import re
import textwrap

import numpy as np

from .farm import FarmSpec, buildFarm
from .farmenv import FarmEnv, INSTRUCTIONS
from .farmagents import FARM_PRESETS, FixedColumn, Schedule, FarmRule
from .followers import followerSetup
from .replay import _Replayer
from .run import runEpisode

TR = 'results/farm_study_vfield/transcripts/qwen2-5_7b-p3/'
LABEL = {'line': 'FULL LINE', 'lineLeft': 'HALF LINE', 'lineRight': 'HALF LINE', 'vee': 'VEE',
         'column2': 'TWO ABREAST', 'column': 'SINGLE FILE', 'custom': 'CUSTOM SHAPE'}

# view: (x low, x high, y low, y high), fixed camera
# segments: [(t start, t end, caption), ...] played one after the other; with
#           align='slough' the times are relative to each leader passing the wet spot
# counters: any of 'collisions', 'wet', 'north', 'worked'
# thought: 'change' (the AI's reason for its latest formation change) or 'latest'
# panel note: shown where the AI's words would be, for a panel without an AI
PRESETS = {
    'formation': dict(
        headline='The AI folds the crew to fit the gate, then opens it up in the field',
        panels=[dict(transcript=TR + 'lessons_wet_test_s4.json', title='AI foreman, wet day')],
        view=(-75, 135, -40, 165), layout='single', align=None,
        segments=[(140, 272, 'GOING IN'), (960, 1196, 'HEADING HOME')],
        counters=('collisions',), thought='change', posterT=268, every=2.0),
    'wetspot': dict(
        headline='A wet spot slows every machine in it to a crawl and ruts the field',
        panels=[dict(decider='schedule', instruction='wet', seed=2, title='Fixed plan: ignores the wet spot',
                     note='Full line at full speed. The machines that hit the wet spot drop to 0.6 m/s and fall '
                          'out of line.'),
                dict(transcript=TR + 'lessons_wet_test_s2.json', title='AI foreman, told "it\'s wet"')],
        view=(116, 224, -6, 92), layout='cols', align='slough', segments=[(-80, 70, '')], every=1.5,
        counters=('wet',), thought='change', posterT=None),
    'cattle': dict(
        headline='"Neighbour\'s got cattle along the north fence": nobody programmed for that',
        panels=[dict(decider='rule', instruction='north', seed=0, title='Hand-written rule',
                     note='Has no idea what cattle are, so it works the field as usual.'),
                dict(transcript=TR + 'lessons_north_test_s0.json', title='AI foreman, with practice notes')],
        view=(-12, 252, -12, 240), layout='cols', align=None, segments=[(170, 1276, '')], every=6.0,
        counters=('north', 'collisions'), thought='latest', posterT=None),
}


def nameOf(spec):
    for k, v in FARM_PRESETS.items():
        if v == spec:
            return k
    return 'custom'


def runPanel(p, spec='farms/default.json', cache=None):
    """Run (or replay) one panel's episode and keep what the animation needs."""
    key = None
    if cache:
        os.makedirs(cache, exist_ok=True)
        key = os.path.join(cache, re.sub(r'\W+', '_', '%s_%s_%s_%s' % (
            os.path.basename(p.get('transcript', '')), p.get('decider', ''), p.get('instruction', ''),
            p.get('seed', ''))) + '.pkl')
        if os.path.exists(key):
            return pickle.load(open(key, 'rb'))
    sp = FarmSpec.load(spec)
    thoughts = []
    if 'transcript' in p:
        d = json.load(open(p['transcript']))
        seed, instr = int(d['result']['seed']), d['result']['instruction']
        agent = _Replayer(d['turns'], instr)
        for t in d['turns']:
            m = re.findall(r'"thought"\s*:\s*"([^"]*)"', t.get('reply2') or t['reply'])
            thoughts.append((t['t'], m[-1] if m else ''))
    else:
        seed, instr = p['seed'], p['instruction']
        agent = {'rule': lambda: FarmRule(instr), 'schedule': Schedule, 'column': FixedColumn}[p['decider']]()
    opts, kw = followerSetup(p.get('followers', 'vfield'))
    env = FarmEnv(buildFarm(sp, seed), instruction=instr, opts=opts, **kw)
    forms = []
    setF = env.setFormation

    def record(s, *a, **k):                    # every formation the decider puts in force
        forms.append((env.t, nameOf(s)))
        return setF(s, *a, **k)
    env.setFormation = record
    runEpisode(env, agent)
    L = env.log
    out = dict(seed=seed, instruction=instr, T=np.array(L['t']), X=np.array(L['x']), Y=np.array(L['y']),
               FX=np.array(L['fanX']), FY=np.array(L['fanY']), forms=forms, thoughts=thoughts,
               cols=sorted((c['t'], c['x'], c['y']) for c in env.collisions),
               exact=(not getattr(agent, 'mismatch', None)),
               result=dict(collisions=len(env.collisions), wet=env.sloughExposure, north=env.northExposure,
                           coverage=env.coverage()))
    if key:
        pickle.dump(out, open(key, 'wb'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preset', choices=sorted(PRESETS))
    ap.add_argument('out')
    ap.add_argument('--dpi', type=int, default=110)
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--every', type=float, default=None, help='simulated seconds per frame')
    ap.add_argument('--poster', type=float, default=2.0, help='open on the poster frame for this many seconds')
    ap.add_argument('--cache', default=None, help='directory to keep replays in between renders')
    ap.add_argument('--spec', default='farms/default.json')
    ap.add_argument('--view', default=None, help='x0,x1,y0,y1 to override the preset camera')
    ap.add_argument('--segments', default=None, help='t0:t1:caption;... to override the preset')
    ap.add_argument('--still', default=None, help='write one PNG at this (relative) time instead of the GIF')
    a = ap.parse_args()
    P = dict(PRESETS[a.preset])
    a.every = a.every or P.get('every', 2.0)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle, Circle

    runs = [runPanel(p, a.spec, a.cache) for p in P['panels']]
    for p, r in zip(P['panels'], runs):
        print('%-40s layout %d %-8s exact %s  %s' % (p['title'][:40], r['seed'], r['instruction'], r['exact'],
                                                    ', '.join('%s %.2f' % kv for kv in r['result'].items())))
    sp = FarmSpec.load(a.spec)
    course = buildFarm(sp, runs[0]['seed'])
    L, W = course.field[1], course.field[3]
    sl = course.sloughs
    if a.view:
        P['view'] = tuple(float(v) for v in a.view.split(','))
    if a.segments:
        P['segments'] = [(float(s.split(':')[0]), float(s.split(':')[1]), ':'.join(s.split(':')[2:]))
                         for s in a.segments.split(';')]
    x0, x1, y0, y1 = P['view']

    # ---- per-run derived series ----
    cs = 2.0
    gx, gy = np.meshgrid(np.arange(cs / 2, L, cs), np.arange(cs / 2, W, cs))
    cx, cy = gx.ravel(), gy.ravel()
    workable = ~((np.hypot(cx[:, None] - sl[:, 0], cy[:, None] - sl[:, 1]) < sl[:, 2]).any(1)) if len(sl) else \
        np.ones(cx.size, bool)
    for rr in runs:
        T = rr['T']
        AX = np.concatenate([rr['X'][:, None], rr['FX']], 1)
        AY = np.concatenate([rr['Y'][:, None], rr['FY']], 1)
        dt = T[1] - T[0]
        rr['dt'] = dt
        inS = (np.hypot(AX[..., None] - sl[:, 0], AY[..., None] - sl[:, 1]) < sl[:, 2]).any(-1) if len(sl) else \
            np.zeros(AX.shape, bool)
        rr['inS'] = inS
        rr['wetS'] = np.cumsum(inS.sum(1)) * dt
        inF = (AX > 0) & (AX < L) & (AY > 0) & (AY < W)
        rr['northS'] = np.cumsum((inF & (AY > W - sp.northStrip)).sum(1)) * dt
        k = np.arange(0, len(T), 5)
        sel = inS[k]
        rr['rutT'] = np.repeat(T[k], sel.sum(1))
        rr['rutXY'] = np.column_stack([AX[k][sel], AY[k][sel]])
        at = np.full(cx.size, np.inf)
        rad = sp.swath / 2
        for i in k:
            m = inF[i]
            if not m.any():
                continue
            idx = np.nonzero(np.isinf(at))[0]
            d2 = (cx[idx, None] - AX[i, m][None, :]) ** 2 + (cy[idx, None] - AY[i, m][None, :]) ** 2
            at[idx[(d2 < rad * rad).any(1)]] = T[i]
        rr['covAt'] = at
        f = []                                  # formation timeline, deduplicated
        for t, n in rr['forms']:
            if f and abs(f[-1][0] - t) < 1e-9:
                f[-1] = (t, n)
            elif not f or f[-1][1] != n:
                f.append((t, n))
        rr['fT'] = np.array([t for t, _ in f]); rr['fN'] = [n for _, n in f]
        rr['dT'] = np.array([t for t, _ in rr['thoughts']])
        rr['base'] = 0.0
        if P.get('align') == 'slough':          # the moment the leader is closest to the wet spot
            rr['base'] = float(T[np.argmin(np.hypot(rr['X'] - sl[0][0], rr['Y'] - sl[0][1]))])
    for p, rr in zip(P['panels'], runs):
        if P.get('align'):
            print('%s: passes the wet spot at t = %.0f s' % (p['title'], rr['base']))

    # ---- figure ----
    lay = P['layout']
    asp = (y1 - y0) / (x1 - x0)
    if lay == 'single':                         # sizes in inches: header, panel, footer
        pw, head = 7.84, 1.15
        ph = min(pw * asp, 5.6)
        pw = ph / asp
    else:
        pw, head = 3.84, 1.38
        ph = pw * asp
    H = head + ph + 0.24
    FW = max(6.4, pw + 0.16) if lay == 'single' else 8.0
    fig = plt.figure(figsize=(FW, H))
    b = 0.2 / H
    if lay == 'single':
        boxes = [[0.08 / FW, b, pw / FW, ph / H]]
    else:
        boxes = [[0.01, b, pw / 8.0, ph / H], [0.51, b, pw / 8.0, ph / H]]
    titleY = 1 - 0.56 / H
    thoughtY = 1 - 0.6 / H if lay == 'single' else 1 - 0.62 / H
    ctrY = b + ph / H + 0.05 / H
    fig.patch.set_facecolor('white')
    fig.text(0.012, 1 - 0.08 / H, P['headline'], fontsize=12.5, weight='bold', va='top', color='#1d1c1a')
    green = np.array([0.42, 0.69, 0.36, 0.5])
    BLUE, BROWN, RUT = '#2a78d6', '#a0581c', '#6b4423'
    art = []
    for kk, (p, rr) in enumerate(zip(P['panels'], runs)):
        bx = boxes[kk]
        ax = fig.add_axes(bx)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal', adjustable='box')
        ax.set_anchor('NW')
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color('#d8d3c4')
        ax.set_facecolor('#e9e4d4')
        ax.add_patch(Rectangle((0, 0), L, W, fc='#f6f2e7', ec='#b9b3a2', lw=0.8, zorder=0))
        if rr['instruction'] == 'north':
            ax.add_patch(Rectangle((0, W - sp.northStrip), L, sp.northStrip, fc='#f4d0c8', ec='none', zorder=0.5))
            ax.text(L * 0.8, W - sp.northStrip / 2, 'cattle side', ha='center', va='center', fontsize=9,
                    color='#a0473a', zorder=1.2, weight='bold')
        img = ax.imshow(np.zeros(gy.shape + (4,)), extent=(0, L, 0, W), origin='lower', zorder=1,
                        interpolation='nearest')
        for (sx, sy, sr) in sl:
            ax.add_patch(Circle((sx, sy), sr, fc='#9cc3e6', ec='#5b8fc0', lw=1.0, alpha=0.85, zorder=1.5))
            if y0 < sy + sr + 4 < y1 and x0 < sx < x1:
                ax.text(sx, sy + sr + 1.5, 'wet spot', ha='center', va='bottom', fontsize=9, color='#2f5f8a',
                        weight='bold', zorder=2)
        ob = course.obstacles
        ax.scatter(ob[:, 0], ob[:, 1], s=np.clip((ob[:, 2] * 2.2) ** 2, 1.5, 40), c='#8d8a82', lw=0, zorder=2)
        for h in course.hazards:
            if h['type'] in ('rock', 'pole') and x0 < h['x'] < x1 - 25 and y0 < h['y'] < y1:
                ax.text(h['x'] + h['r'] + 2, h['y'] + 2, 'rock pile' if h['type'] == 'rock' else 'power pole',
                        fontsize=7.5, color='#55534d', zorder=2)
        if x0 < -5 and y0 < sp.gateY < y1:
            ax.text(3, sp.gateY + sp.gateWidth / 2 + 4, 'gate, 16 m wide', fontsize=9, color='#55534d', ha='left',
                    zorder=2, weight='bold')
        if x0 < -60:
            ax.text(-63, 6, 'yard lane', fontsize=9, color='#55534d', ha='center', va='center', rotation=65,
                    zorder=2, weight='bold')
        ruts = ax.scatter([], [], s=6, marker='s', color=RUT, alpha=0.45, lw=0, zorder=2.5)
        trails = [ax.plot([], [], color=BLUE, lw=0.8, alpha=0.4, zorder=3)[0] for _ in range(rr['FX'].shape[1])]
        mach = ax.scatter([], [], s=30, color=BLUE, edgecolor='white', lw=0.7, zorder=5)
        lead = ax.scatter([], [], s=50, marker='s', color='#1d1c1a', edgecolor='white', lw=0.7, zorder=6)
        hits = ax.scatter([], [], s=60, marker='x', color='#c8372d', lw=1.8, zorder=7)
        slow = ax.text(0, 0, '', fontsize=9, color=BROWN, weight='bold', ha='center', va='top', zorder=8,
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))
        form = ax.text(0.02, 0.975, '', transform=ax.transAxes, fontsize=15 if lay == 'single' else 12.5,
                       weight='bold', va='top', ha='left', zorder=9, color='#1d1c1a',
                       bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#1d1c1a', lw=1.0))
        seg = ax.text(0.98, 0.975, '', transform=ax.transAxes, fontsize=12, weight='bold', va='top', ha='right',
                      color='#55534d', zorder=9)
        tx = bx[0] + 0.005
        fig.text(tx, titleY, p['title'], fontsize=11.5, weight='bold', va='bottom', color='#1d1c1a')
        ctr = fig.text(tx, ctrY, '', fontsize=10, va='bottom', family='monospace', color='#3d3b36', weight='bold')
        th = fig.text(tx, thoughtY, '', fontsize=9.2, va='top', color='#3d3b36', style='italic')
        if p.get('note') and not rr['thoughts']:
            wrapW = 110 if lay == 'single' else 60
            th.set_text('\n'.join(textwrap.wrap(p['note'], wrapW)[:3]))
            th.set_style('normal'); th.set_color('#6b6862')
        art.append(dict(ax=ax, img=img, ruts=ruts, trails=trails, mach=mach, lead=lead, hits=hits, slow=slow,
                        form=form, seg=seg, ctr=ctr, th=th))
    legend = 'black square: lead machine   blue: followers   green: ground worked'
    if len(sl) and x0 < sl[0][0] < x1 and y0 < sl[0][1] < y1:
        legend += '   brown: slowed in the wet spot, and its ruts'
    if 'collisions' in P['counters'] or any(rr['cols'] for rr in runs):
        legend += '   x: collision'
    fig.text(0.985, 0.006, legend, fontsize=7, ha='right', va='bottom', color='#6b6862')

    # ---- frames: times relative to each run's base ----
    tau, segOf = [], []
    for si, (t0, t1, cap) in enumerate(P['segments']):
        ts = list(np.arange(t0, t1 + 1e-9, a.every))
        if si > 0:                              # a short pause marks the jump in time
            ts = [ts[0]] * int(0.8 * a.fps) + ts
        tau += ts
        segOf += [si] * len(ts)
    if a.poster > 0:
        if P.get('posterT') is not None:
            pt = P['posterT']
            ps = next(i for i, (t0, t1, _) in enumerate(P['segments']) if t0 <= pt <= t1)
        else:
            pt, ps = tau[-1], segOf[-1]
        tau = [pt] * int(a.poster * a.fps) + tau
        segOf = [ps] * int(a.poster * a.fps) + segOf
    nCell = int(workable.sum())

    def draw(fi):
        out = []
        for p, rr, A in zip(P['panels'], runs, art):
            jj = int(np.clip(round((rr['base'] + tau[fi]) / rr['dt']), 0, len(rr['T']) - 1))
            t = rr['T'][jj]
            a0 = max(0, jj - 450)
            for k, tr in enumerate(A['trails']):
                tr.set_data(rr['FX'][a0:jj + 1, k], rr['FY'][a0:jj + 1, k])
            ins = rr['inS'][jj]
            A['mach'].set_offsets(np.column_stack([rr['FX'][jj], rr['FY'][jj]]))
            A['mach'].set_facecolors([BROWN if v else BLUE for v in ins[1:]])
            A['mach'].set_sizes([48 if v else 30 for v in ins[1:]])
            A['lead'].set_offsets([[rr['X'][jj], rr['Y'][jj]]])
            A['lead'].set_facecolors([BROWN if ins[0] else '#1d1c1a'])
            done = (rr['covAt'] <= t).reshape(gy.shape)
            rgba = np.zeros(done.shape + (4,)); rgba[done] = green
            A['img'].set_data(rgba)
            m = rr['rutT'] <= t
            A['ruts'].set_offsets(rr['rutXY'][m] if m.any() else np.empty((0, 2)))
            ch = [c for c in rr['cols'] if c[0] <= t]
            A['hits'].set_offsets(np.array([[c[1], c[2]] for c in ch]) if ch else np.empty((0, 2)))
            if ins.any():
                sx, sy, sr = sl[0]
                A['slow'].set_position((sx, sy - sr - 2))
                A['slow'].set_text('%d slowed to %.1f m/s' % (ins.sum(), sp.wetSpeed))
            else:
                A['slow'].set_text('')
            fk = np.searchsorted(rr['fT'], t, side='right') - 1
            if fk >= 0:
                A['form'].set_text(LABEL.get(rr['fN'][fk], rr['fN'][fk]))
                fresh = fk > 0 and t - rr['fT'][fk] < 6
                A['form'].get_bbox_patch().set_facecolor('#ffd35c' if fresh else 'white')
            A['seg'].set_text(P['segments'][segOf[fi]][2])
            parts = []
            for c in P['counters']:
                if c == 'collisions':
                    parts.append('collisions %d' % len(ch))
                elif c == 'wet':
                    parts.append('in the wet spot: %3.0f machine-seconds' % rr['wetS'][jj])
                elif c == 'north':
                    parts.append('cattle side %3.0f machine-s' % rr['northS'][jj])
                elif c == 'worked':
                    parts.append('field worked %2.0f%%' % (100 * np.sum((rr['covAt'] <= t) & workable) / nCell))
            A['ctr'].set_text('   '.join(parts))
            if P['thought'] and len(rr['dT']):
                if P['thought'] == 'change' and len(rr['fT']):
                    tk = rr['fT'][max(fk, 0)]
                    dk = int(np.argmin(np.abs(rr['dT'] - tk)))
                else:
                    dk = np.searchsorted(rr['dT'], t, side='right') - 1
                wrapW = int(FW * 13.5) if lay == 'single' else 58
                txt = rr['thoughts'][dk][1] if dk >= 0 else ''
                ln = textwrap.wrap('AI: "%s"' % txt, wrapW)
                if len(ln) > 3:
                    ln = ln[:3]; ln[2] = ln[2].rstrip(' ,.') + '…"'
                A['th'].set_text('\n'.join(ln) if txt else '')
            out += A['trails'] + [A[k] for k in ('img', 'ruts', 'mach', 'lead', 'hits', 'slow', 'form', 'seg',
                                                 'ctr', 'th')]
        return out

    if a.still is not None:
        tau = [float(a.still)]; segOf = [next((i for i, (t0, t1, _) in enumerate(P['segments'])
                                               if t0 <= float(a.still) <= t1), 0)]
        draw(0)
        fig.savefig(a.out, dpi=a.dpi)
        print('wrote', a.out)
        return
    FuncAnimation(fig, draw, frames=len(tau)).save(a.out, writer=PillowWriter(fps=a.fps), dpi=a.dpi)
    print('wrote', a.out, len(tau), 'frames')


if __name__ == '__main__':
    main()
