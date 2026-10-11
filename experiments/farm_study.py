"""Farm study: a language model supervising a turn-limited flock of farm
machines, against a hand-written rule and an oracle search.
See experiments/FARM_STUDY.md for the design and the hypotheses.

    # 1. pilot (about 15 min): plumbing, parse rate, timing
    python experiments/farm_study.py --pilot

    # 2. the run (about 4 h on an RTX 5050 laptop; resumable)
    python experiments/farm_study.py

    # 3. the non-LLM deciders (CPU only, about 1 h; resumable)
    python experiments/farm_comparators.py --workers 4

    # report only, from whatever has run (also merges the comparators)
    python experiments/farm_study.py --report-only

    # whole pipeline offline, with a scripted stand-in model
    python experiments/farm_study.py --mock --seeds 0-1 --train-episodes 2

Needs: ollama serve; ollama pull qwen2.5:7b; ollama pull nomic-embed-text
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shepherd.farm import FarmSpec, buildFarm                       # noqa: E402
from shepherd.farmenv import FarmEnv, INSTRUCTIONS                   # noqa: E402
from shepherd.farmsupervisor import FarmSupervisor, MockFarmLLM, PROMPT_VERSION   # noqa: E402
from shepherd.agents import OllamaClient                             # noqa: E402
from shepherd.memory import OllamaEmbedder, HashEmbedder             # noqa: E402
from shepherd.run import runEpisode                                  # noqa: E402
from shepherd.followers import followerSetup                         # noqa: E402

COLS = ['agent', 'instruction', 'seed', 'phase', 'captured', 'posErr', 'headErrDeg', 'collisions',
        'fan-leader', 'fan-fan', 'fan-obstacle', 'leader-obstacle', 'fansLost', 'coverage',
        'sloughExposure', 'northExposure', 'time', 'cost', 'cost_wet', 'cost_weather', 'cost_north',
        'formationSwitches', 'decisions', 'wall_s', 'llmCalls', 'llmBadReplies', 'llmWall_s']

DECIDERS = [  # csv agent, display name, family
    ('column',       'Single file throughout',       'fixed'),
    ('schedule',     'Schedule (hazard-blind)',      'fixed'),
    ('rule',         'Hand-written rule',            'rule'),
    ('search',       'Oracle search',                'search'),
    ('llm-zeroshot', 'LLM zero-shot',                'llm'),
    ('llm-lessons',  'LLM + lessons',                'llm'),
    ('llm-memory',   'LLM + memory stream',          'llm'),
    ('llm-whatif',   'LLM + what-if simulator',      'llm'),
]
COLOR = {'fixed': '#8a8984', 'rule': '#2a78d6', 'search': '#eb6834', 'llm': '#1baf7a'}
INSTR_TITLE = {'wet': '"It\'s wet ... stay out of the low spots"',
               'weather': '"Rain\'s coming ... get as much done as you can"',
               'north': '"Cattle along the north fence ..." (held out)'}


def parseSeeds(s):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


def appendRow(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def doneKeys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r['agent'], r['instruction'], r['seed'], r['phase']) for r in csv.DictReader(f)}


class Study:
    def __init__(self, a):
        self.a = a
        self.spec = FarmSpec.load(a.spec)
        self.dirs = {k: os.path.join(a.out, k, a.tag) for k in ('traj', 'transcripts')}
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
        self.csv = os.path.join(a.out, 'llm_%s.csv' % a.tag)
        self.lessons = os.path.join(a.out, 'lessons_%s.json' % a.tag)
        self.memory = os.path.join(a.out, 'memory_%s.json' % a.tag)
        if a.mock:
            self.client, self.embedder = MockFarmLLM(), HashEmbedder()
        else:
            self.client = OllamaClient(model=a.model, host=a.host, temperature=a.temperature, numCtx=a.num_ctx)
            self.embedder = OllamaEmbedder(model=a.embed_model, host=a.host)

    def episode(self, cond, instr, seed, phase, learn):
        a = self.a
        ag = FarmSupervisor(self.client, instruction=instr, condition=cond, lessonsPath=self.lessons,
                            memoryPath=self.memory, embedder=self.embedder, learn=learn,
                            verbose=a.verbose, shield=a.shield)
        opts, kw = followerSetup(a.followers)
        env = FarmEnv(buildFarm(self.spec, seed), instruction=instr, opts=opts, **kw)
        t0 = time.time()
        r = runEpisode(env, ag)
        r.update(agent='llm-' + cond, instruction=instr, seed=seed, phase=phase, decisions=len(ag.transcript),
                 llmCalls=ag.nCalls, llmBadReplies=ag.nBad, llmWall_s=round(ag.wall, 1),
                 wall_s=round(time.time() - t0, 1))
        stem = '%s_%s_%s_s%d' % (cond, instr, phase, seed)
        np.savez_compressed(os.path.join(self.dirs['traj'], stem + '.npz'), x=env.log['x'], y=env.log['y'],
                            fanX=np.array(env.log['fanX']), fanY=np.array(env.log['fanY']), covered=env.covered,
                            col=np.array([[c['x'], c['y']] for c in env.collisions]).reshape(-1, 2))
        with open(os.path.join(self.dirs['transcripts'], stem + '.json'), 'w') as f:
            json.dump(dict(system=ag.system, turns=ag.transcript, choices=ag.choices,
                           shield=ag.shieldLog if a.shield else None,
                           result={k: v for k, v in r.items() if not isinstance(v, np.ndarray)}),
                      f, indent=1, default=float)
        appendRow(self.csv, r)
        print('%-8s %-8s %-5s seed %3d | col %3d cov %4.0f%% wet %5.0f north %5.0f | cost %7.1f | '
              'calls %3d bad %2d %4.1f s/call | %4.0f s'
              % (cond, instr, phase, seed, r['collisions'], 100 * r['coverage'], r['sloughExposure'],
                 r['northExposure'], r['cost'], ag.nCalls, ag.nBad, ag.wall / max(1, ag.nCalls),
                 time.time() - t0), flush=True)
        return r

    def run(self):
        a = self.a
        done = doneKeys(self.csv)
        conds = a.conditions.split(',')
        instrs = a.instructions.split(',')
        trainInstr = [i for i in ('wet', 'weather') if not INSTRUCTIONS[i]['heldOut']]
        # training phases: lessons and memory, on held-out seeds, trained instructions only
        for cond in [c for c in conds if c in ('lessons', 'memory')]:
            for k in range(a.train_episodes):
                instr, seed = trainInstr[k % len(trainInstr)], 100 + k
                if ('llm-' + cond, instr, str(seed), 'train') in done:
                    continue
                self.episode(cond, instr, seed, 'train', learn=True)
                src = self.lessons if cond == 'lessons' else self.memory
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(self.dirs['transcripts'], '%s_after_train%02d.json' % (cond, k)))
        # test: seed-major order, so a partial run is still balanced
        for seed in parseSeeds(a.seeds):
            for instr in instrs:
                for cond in conds:
                    if ('llm-' + cond, instr, str(seed), 'test') in done:
                        continue
                    self.episode(cond, instr, seed, 'test', learn=False)


# ---------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------

def bootCI(x, n=4000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    return tuple(np.percentile(rng.choice(x, (n, len(x))).mean(1), [2.5, 97.5]))


def loadRows(a):
    rows = []
    p = os.path.join(a.out, 'comparators', 'episodes.csv')
    if os.path.exists(p):
        with open(p) as f:
            rows += list(csv.DictReader(f))
    p = os.path.join(a.out, 'llm_%s.csv' % a.tag)
    if os.path.exists(p):
        with open(p) as f:
            rows += list(csv.DictReader(f))
    keep = {}
    for r in rows:
        keep[(r['agent'], r['instruction'], r['seed'], r.get('phase', 'test'))] = r
    seeds = {str(s) for s in parseSeeds(a.seeds)}
    test = [r for r in keep.values() if r.get('phase', 'test') == 'test' and r['seed'] in seeds]
    train = [r for r in keep.values() if r.get('phase') == 'train']
    return test, train


def f(r, k):
    return float(r[k])


def summarise(test, instrs):
    S = []
    for instr in instrs:
        for key, name, fam in DECIDERS:
            rs = [r for r in test if r['agent'] == key and r['instruction'] == instr]
            if not rs:
                continue
            row = dict(instruction=instr, key=key, name=name, family=fam, n=len(rs))
            for m in ('cost', 'collisions', 'coverage', 'sloughExposure', 'northExposure', 'time', 'posErr', 'headErrDeg'):
                x = np.array([f(r, m) for r in rs])
                row[m], (row[m + 'Lo'], row[m + 'Hi']) = x.mean(), bootCI(x)
            calls = [f(r, 'llmCalls') for r in rs if r.get('llmCalls') not in (None, '')]
            if calls:
                row['calls'] = np.mean(calls)
                row['badPct'] = 100 * sum(f(r, 'llmBadReplies') for r in rs) / max(1, sum(calls))
                row['secPerCall'] = sum(f(r, 'llmWall_s') for r in rs) / max(1, sum(calls))
            S.append(row)
    return S


def adherence(test):
    """Does behaviour move the way the instruction asks? Paired by seed."""
    out = []
    for key, name, fam in DECIDERS:
        by = {}
        for r in test:
            if r['agent'] == key:
                by.setdefault(r['seed'], {})[r['instruction']] = r
        pairsWW = [(v['wet'], v['weather']) for v in by.values() if 'wet' in v and 'weather' in v]
        pairsNW = [(v['north'], v['wet']) for v in by.values() if 'north' in v and 'wet' in v]
        if not pairsWW and not pairsNW:
            continue
        row = dict(key=key, name=name, family=fam)
        if pairsWW:
            ds = np.array([f(w, 'sloughExposure') - f(h, 'sloughExposure') for w, h in pairsWW])
            dc = np.array([100 * (f(h, 'coverage') - f(w, 'coverage')) for w, h in pairsWW])
            row.update(nWW=len(pairsWW), wetMinusWeatherSlough=ds.mean(), wetMinusWeatherSloughCI=bootCI(ds),
                       weatherMinusWetCoverage=dc.mean(), weatherMinusWetCoverageCI=bootCI(dc))
        if pairsNW:
            dn = np.array([f(n, 'northExposure') - f(w, 'northExposure') for n, w in pairsNW])
            row.update(nNW=len(pairsNW), northMinusWetNorth=dn.mean(), northMinusWetNorthCI=bootCI(dn))
        out.append(row)
    return out


def _style(plt):
    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#52514e', 'xtick.color': '#52514e',
                         'ytick.color': '#0b0b0b', 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def _dotPanel(ax, rows, val, lo, hi, names, fmt='%.0f'):
    for i, nm in enumerate(names):
        r = next((r for r in rows if r['name'] == nm), None)
        if r is None:
            continue
        y = len(names) - 1 - i
        c = COLOR[r['family']]
        if np.isfinite(r[lo]):
            ax.plot([r[lo], r[hi]], [y, y], color=c, lw=2, solid_capstyle='round')
        ax.plot(r[val], y, 'o', ms=6.5, color=c, mec='white', mew=1.4, zorder=3)
        ax.annotate(fmt % r[val], (r[hi] if np.isfinite(r[hi]) else r[val], y), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=7.5, color='#52514e')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.grid(axis='x', color='#e6e5e1', lw=0.8)
    ax.set_axisbelow(True)


def figures(a, S, test, train):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    instrs = [i for i in INSTRUCTIONS if any(r['instruction'] == i for r in S)]
    names = [n for _, n, _ in DECIDERS if any(r['name'] == n for r in S)]
    # 1. instruction cost
    fig, axs = plt.subplots(1, len(instrs), figsize=(3.3 * len(instrs) + 1.4, 0.36 * len(names) + 1.3),
                            sharey=True, squeeze=False)
    for ax, ins in zip(axs[0], instrs):
        _dotPanel(ax, [r for r in S if r['instruction'] == ins], 'cost', 'costLo', 'costHi', names)
        ax.set_title(INSTR_TITLE[ins], fontsize=8)
        ax.set_xlabel('instruction cost (lower is better)')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, 'fig_cost_%s.%s' % (a.tag, ext)), dpi=200)
    plt.close(fig)
    # 2. what each instruction asks for, measured directly
    panels = [('wet', 'sloughExposure', 'machine-seconds on wet ground (lower = obeyed)', '%.0f', 1),
              ('weather', 'coverage', 'field worked, % (higher = obeyed)', '%.0f', 100),
              ('north', 'northExposure', 'machine-seconds near north fence (lower = obeyed)', '%.0f', 1)]
    panels = [p for p in panels if p[0] in instrs]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.3 * len(panels) + 1.4, 0.36 * len(names) + 1.3),
                            sharey=True, squeeze=False)
    for ax, (ins, m, lab, fmt, k) in zip(axs[0], panels):
        rows = [dict(r, v=r[m] * k, vLo=r[m + 'Lo'] * k, vHi=r[m + 'Hi'] * k) for r in S if r['instruction'] == ins]
        _dotPanel(ax, rows, 'v', 'vLo', 'vHi', names, fmt)
        ax.set_title(INSTR_TITLE[ins], fontsize=8)
        ax.set_xlabel(lab)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, 'fig_obey_%s.%s' % (a.tag, ext)), dpi=200)
    plt.close(fig)
    # 3. training curves: collisions per training day (the one measure on a
    #    common scale across wet and rain-coming days), day type labelled
    if train:
        fig, ax = plt.subplots(figsize=(4.4, 2.6))
        nmax = 0
        for key, mk in (('llm-lessons', 'o'), ('llm-memory', 's')):
            rs = sorted([r for r in train if r['agent'] == key], key=lambda r: int(r['seed']))
            if not rs:
                continue
            xs = np.arange(1, len(rs) + 1)
            nmax = max(nmax, len(rs))
            ax.plot(xs, [f(r, 'collisions') for r in rs], '-' + mk, color=COLOR['llm'], ms=5, mec='white',
                    mew=1.2, lw=1.6, alpha=1.0 if key == 'llm-memory' else 0.55, label=dict(DECIDERS_NAME)[key])
            if key == 'llm-lessons' or nmax == len(rs):
                labels = ['%d\n%s' % (i, 'wet' if r['instruction'] == 'wet' else 'rain') for i, r in zip(xs, rs)]
        ax.set_xticks(range(1, nmax + 1))
        ax.set_xticklabels(labels)
        ax.set_xlabel('training day')
        ax.set_ylabel('collisions')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis='y', color='#e6e5e1', lw=0.8)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(a.out, 'fig_training_%s.%s' % (a.tag, ext)), dpi=200)
        plt.close(fig)
    # 4. one field, wet day: rule vs search vs the best LLM condition (median-cost run)
    trajFigure(a, S, test)


DECIDERS_NAME = [(k, n) for k, n, _ in DECIDERS]


def trajFigure(a, S, test, instr='wet'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    llm = [r for r in S if r['family'] == 'llm' and r['instruction'] == instr]
    picks = ['rule', 'search'] + ([min(llm, key=lambda r: r['cost'])['key']] if llm else [])
    seedsAll = sorted({r['seed'] for r in test if r['instruction'] == instr and r['agent'] in picks},
                      key=int)
    if not seedsAll:
        return
    seed = seedsAll[0]
    spec = FarmSpec.load(a.spec)
    course = buildFarm(spec, int(seed))
    fig, axs = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 3.9), squeeze=False)
    for ax, key in zip(axs[0], picks):
        if key.startswith('llm-'):
            p = os.path.join(a.out, 'traj', a.tag, '%s_%s_test_s%s.npz' % (key[4:], instr, seed))
        else:
            mode = {'rule': 'wet' if instr in ('wet', 'north') else instr}.get(key, instr)
            p = os.path.join(a.out, 'comparators', 'traj', '%s_%s_s%s.npz' % (key, mode, seed))
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for (x, y, rr) in course.obstacles:
            ax.add_patch(plt.Circle((x, y), rr, color='#b9b8b3', lw=0))
        for (x, y, rr) in course.sloughs:
            ax.add_patch(plt.Circle((x, y), rr, color='#9ec5f4', lw=0))
        if os.path.exists(p):
            d = np.load(p)
            for i in range(d['fanX'].shape[1]):
                ax.plot(d['fanX'][:, i], d['fanY'][:, i], lw=0.45, color='#2a78d6', alpha=0.4)
            ax.plot(d['x'], d['y'], color='#0b0b0b', lw=1.3)
            if len(d['col']):
                ax.plot(d['col'][:, 0], d['col'][:, 1], 'x', color='#d03b3b', ms=4, mew=1.3)
        r = next((r for r in test if r['agent'] == key and r['instruction'] == instr and r['seed'] == seed), None)
        nm = dict(DECIDERS_NAME).get(key, key)
        ax.set_title(('%s\n%s collisions, %.0f%% worked, %.0f wet machine-s' % (
            nm, r['collisions'], 100 * f(r, 'coverage'), f(r, 'sloughExposure'))) if r else
            '%s\n(not run yet)' % nm, fontsize=7.5)
        b = course.bounds
        ax.set_xlim(b[0], b[1]); ax.set_ylim(b[2], b[3])
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, 'fig_traj_%s.%s' % (a.tag, ext)), dpi=200)
    plt.close(fig)


def tables(a, S, adh):
    with open(os.path.join(a.out, 'summary_%s.csv' % a.tag), 'w', newline='') as fh:
        keys = sorted({k for r in S for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(S)
    with open(os.path.join(a.out, 'adherence_%s.json' % a.tag), 'w') as fh:
        json.dump(adh, fh, indent=1, default=float)
    L = [r'\begin{table}[h!]', r'\centering', r'\small', r'\resizebox{\linewidth}{!}{%',
         r'\begin{tabular}{llcccccc}', r'\hline',
         r'Instruction & Decider & Cost & Collisions & Worked (\%) & Wet ground (m-s) & North strip (m-s) & LLM calls \\',
         r'\hline']
    for ins in dict.fromkeys(r['instruction'] for r in S):
        first = True
        for r in [r for r in S if r['instruction'] == ins]:
            L.append('%s & %s & %.0f [%.0f, %.0f] & %.1f & %.0f & %.0f & %.0f & %s \\\\' % (
                {'wet': 'Wet', 'weather': 'Rain coming', 'north': 'North fence$^*$'}[ins] if first else '',
                r['name'], r['cost'], r['costLo'], r['costHi'], r['collisions'], 100 * r['coverage'],
                r['sloughExposure'], r['northExposure'], '%.0f' % r['calls'] if 'calls' in r else '--'))
            first = False
        L.append(r'\hline')
    n = max(r['n'] for r in S)
    L += [r'\end{tabular}}',
          r'\caption{Farm mission, %d seeds (field layouts) per row, %d machines, paper boid parameters. Cost: '
          r"the instruction's cost (lower is better), mean with 95\%% bootstrap interval. m-s: machine-seconds. "
          r'$^*$Held-out instruction: the rule has no handler for it and runs its wet-day behaviour; the '
          r'language models never saw it in training. Oracle search simulates the true state with the true '
          r'cost. LLM: %s.}' % (n, 9, a.model.replace('_', r'\_')),
          r'\label{tab:farm}', r'\end{table}']
    with open(os.path.join(a.out, 'summary_%s.tex' % a.tag), 'w') as fh:
        fh.write('\n'.join(L) + '\n')


def report(a):
    test, train = loadRows(a)
    S = summarise(test, [i for i in INSTRUCTIONS])
    if not S:
        print('nothing to report yet')
        return
    adh = adherence(test)
    tables(a, S, adh)
    figures(a, S, test, train)
    print('\n%-8s %-26s %3s %18s %8s %6s %7s %7s %6s %5s %6s' % (
        'instr', 'decider', 'n', 'cost [95% CI]', 'collis.', 'work%', 'wet m-s', 'north', 'calls', 'bad%', 's/call'))
    for r in S:
        print('%-8s %-26s %3d %7.0f [%4.0f,%4.0f] %8.1f %6.0f %7.0f %7.0f %6s %5s %6s' % (
            r['instruction'], r['name'], r['n'], r['cost'], r['costLo'], r['costHi'], r['collisions'],
            100 * r['coverage'], r['sloughExposure'], r['northExposure'],
            '%.0f' % r['calls'] if 'calls' in r else '-', '%.0f' % r['badPct'] if 'badPct' in r else '-',
            '%.1f' % r['secPerCall'] if 'secPerCall' in r else '-'))
    print('\nInstruction following (paired by field layout):')
    for r in adh:
        parts = []
        if 'nWW' in r:
            parts.append('wet-day minus rain-day wet ground %+.0f m-s [%.0f, %.0f]; rain-day minus wet-day worked %+.1f%% [%.1f, %.1f]'
                         % (r['wetMinusWeatherSlough'], *r['wetMinusWeatherSloughCI'],
                            r['weatherMinusWetCoverage'], *r['weatherMinusWetCoverageCI']))
        if 'nNW' in r:
            parts.append('north-fence minus wet-day north strip %+.0f m-s [%.0f, %.0f]'
                         % (r['northMinusWetNorth'], *r['northMinusWetNorthCI']))
        print('  %-26s %s' % (r['name'], ' | '.join(parts)))
    print('\nwrote summary/adherence/figures to %s' % a.out)


def preflight(a):
    try:
        OllamaClient(model=a.model, host=a.host, timeout=180).chat([dict(role='user', content='Reply {"ok": true}')])
    except Exception as e:
        sys.exit('Cannot use Ollama model %r at %s: %s\n  ollama serve; ollama pull %s' % (a.model, a.host, e, a.model))
    if 'memory' in a.conditions:
        try:
            OllamaEmbedder(model=a.embed_model, host=a.host)('test')
        except Exception as e:
            sys.exit('Cannot use embedding model %r: %s\n  ollama pull %s' % (a.embed_model, e, a.embed_model))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='qwen2.5:7b')
    p.add_argument('--embed-model', default='nomic-embed-text')
    p.add_argument('--host', default='http://localhost:11434')
    p.add_argument('--temperature', type=float, default=0.3)
    p.add_argument('--num-ctx', type=int, default=8192)
    p.add_argument('--spec', default=os.path.join(ROOT, 'farms', 'default.json'))
    p.add_argument('--seeds', default='0-7')
    p.add_argument('--instructions', default='wet,weather,north')
    p.add_argument('--conditions', default='zeroshot,whatif,lessons,memory')
    p.add_argument('--train-episodes', type=int, default=6)
    p.add_argument('--out', default=None, help='default results/farm_study[_<followers>]')
    p.add_argument('--followers', default='vfield', choices=['paper', 'damped', 'vfield', 'reflex', 'pearls'],
                   help='follower model (shepherd/followers.py); default vector-field slot following')
    p.add_argument('--shield', action='store_true',
                   help='safety filter: a formation that fails the fit check is held back until it fits; '
                        'output goes to a separate folder (..._shield)')
    p.add_argument('--pilot', action='store_true', help='2 seeds, wet + north, zero-shot + what-if, separate output')
    p.add_argument('--mock', action='store_true')
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--verbose', action='store_true', help='print every decision')
    a = p.parse_args()
    if a.out is None:
        a.out = os.path.join(ROOT, 'results', 'farm_study' + ('' if a.followers == 'paper' else '_' + a.followers)
                             + ('_shield' if a.shield else ''))
    if a.pilot:
        a.seeds, a.instructions, a.conditions, a.train_episodes = '0-1', 'wet,north', 'zeroshot,whatif', 0
        a.verbose = True
    a.tag = ('mock' if a.mock else a.model.replace(':', '_').replace('/', '_').replace('.', '-')) + \
            '-p%d' % PROMPT_VERSION + ('-pilot' if a.pilot else '')
    if not a.mock and not a.report_only:
        preflight(a)
    if not a.report_only:
        Study(a).run()
    report(a)


if __name__ == '__main__':
    main()
