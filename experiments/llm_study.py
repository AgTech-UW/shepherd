"""LLM supervisor study: one command, resumable, writes everything the
study needs.

    # on the machine with Ollama (ollama serve running, model pulled)
    python experiments/llm_study.py --model qwen2.5:14b

    # dry run of the whole pipeline without a model
    python experiments/llm_study.py --mock --seeds 0-1 --train-episodes 2

    # figures and tables only, from whatever has been run so far
    python experiments/llm_study.py --report-only --model qwen2.5:14b

Conditions (same action space as the comparators: formation + speed; the
paper's vector-field controller steers, a Dubins path lands the pose):
  zeroshot   prompt + observation only
  whatif     the model proposes 2-4 candidates, the simulator scores them,
             the cheapest is applied
  memory     a training phase of --train-episodes on held-out seeds (100+)
             in which the model writes lessons after every episode; then
             evaluation on the test seeds with the lessons frozen

Comparators (no LLM) are read from results/llm_study/comparators/, which
ships with the code (10 seeds, narrows + dogleg, paper boids). Pass
--run-comparators to regenerate them on this machine.

Outputs in results/llm_study/:
  llm_<model>.csv                one row per episode (phase train/test)
  traj/<cond>_<course>_s<seed>.npz   trajectories for figures
  transcripts/...json            every prompt and reply
  lessons_<model>.json           the memory condition's lessons
  summary_<model>.csv / .tex     the summary table
  fig_collisions_<model>.pdf/png dot plot, mean collisions per run, 95% CI
  fig_learning_<model>.pdf/png   memory condition, training curve
  fig_traj_<model>.pdf/png       median-run trajectories, LLM vs rule
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

from grainframe import SimOptions                      # noqa: E402
from shepherd.courses import get_course                # noqa: E402
from shepherd.env import ShepherdEnv, score            # noqa: E402
from shepherd.agents import OllamaClient               # noqa: E402
from shepherd.supervisor import LLMSupervisor, MockSupervisorLLM   # noqa: E402
from shepherd.run import runEpisode                    # noqa: E402

COLS = ['agent', 'phase', 'course', 'nFan', 'seed', 'captured', 'posErr', 'headErrDeg',
        'collisions', 'fan-leader', 'fan-fan', 'fan-obstacle', 'leader-obstacle', 'fansLost',
        'formationSwitches', 'time', 'score', 'wall_s', 'llmCalls', 'llmBadReplies', 'llmWall_s']

DISPLAY = [   # (agent label in CSV, display name, family)
    ('baseline',        'Leader pull (no formation)', 'fixed'),
    ('baseline-vee',    'Fixed vee (paper)',          'fixed'),
    ('baseline-column', 'Fixed single-file column',   'fixed'),
    ('adaptive',        'Width rule',                 'rule'),
    ('formsearch',      'Rollout search',             'search'),
    ('llm-zeroshot',    'LLM zero-shot',              'llm'),
    ('llm-memory',      'LLM + lessons',              'llm'),
    ('llm-whatif',      'LLM + what-if simulator',    'llm'),
    ('baseline-tuned-column', 'Ref.: fixed column, damped boids', 'reference'),
]
FAMILY_COLOR = {'fixed': '#8a8984', 'rule': '#2a78d6', 'search': '#eb6834', 'llm': '#1baf7a',
                'reference': '#8a8984'}


def parseSeeds(s):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


def doneKeys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r['agent'], r['phase'], r['course'], r['seed']) for r in csv.DictReader(f)}


def appendRow(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def makeClient(a):
    if a.mock:
        return MockSupervisorLLM()
    return OllamaClient(model=a.model, host=a.host, temperature=a.temperature)


def episode(a, cond, phase, course, seed, memoryPath=None, learn=True):
    agent = LLMSupervisor(makeClient(a), whatIf=(cond == 'whatif'),
                          memoryPath=memoryPath, learn=learn, verbose=a.verbose)
    env = ShepherdEnv(get_course(course), nFan=a.nfan, seed=seed, opts=SimOptions())
    t0 = time.time()
    res = runEpisode(env, agent)
    res.update(agent='llm-' + cond, phase=phase, llmCalls=agent.nCalls,
               llmBadReplies=agent.nBad, llmWall_s=round(agent.wall, 1))
    stem = '%s_%s_%s_s%d' % (cond, phase, course, seed)
    np.savez_compressed(os.path.join(a.out, 'traj', a.tag, stem + '.npz'),
                        x=env.log['x'], y=env.log['y'], fanX=np.array(env.log['fanX']),
                        fanY=np.array(env.log['fanY']),
                        col=np.array([[c['x'], c['y']] for c in env.collisions]).reshape(-1, 2))
    with open(os.path.join(a.out, 'transcripts', a.tag, stem + '.json'), 'w') as f:
        json.dump(dict(system=agent.system, turns=agent.transcript, result=res), f, indent=1, default=float)
    print('%-8s %-5s %-8s seed %3d | collisions %3d (wall %d) lost %d | calls %d bad %d | %.0f s'
          % (cond, phase, course, seed, res['collisions'], res['fan-obstacle'], res['fansLost'],
             agent.nCalls, agent.nBad, time.time() - t0), flush=True)
    return res


def runStudy(a):
    csvPath = os.path.join(a.out, 'llm_%s.csv' % a.tag)
    done = doneKeys(csvPath)
    courses = a.courses.split(',')
    seeds = parseSeeds(a.seeds)
    conds = a.conditions.split(',')
    lessons = os.path.join(a.out, 'lessons_%s.json' % a.tag)

    if 'memory' in conds:
        # training: alternate courses, held-out seeds, lessons written every episode
        for k in range(a.train_episodes):
            course, seed = courses[k % len(courses)], 100 + k
            if ('llm-memory', 'train', course, str(seed)) in done:
                continue
            res = episode(a, 'memory', 'train', course, seed, memoryPath=lessons, learn=True)
            res['trainIndex'] = k
            appendRow(csvPath, res)
            if os.path.exists(lessons):
                shutil.copy(lessons, os.path.join(a.out, 'transcripts', a.tag, 'lessons_after_%02d.json' % k))
    for cond in conds:
        for course in courses:
            for seed in seeds:
                if ('llm-' + cond, 'test', course, str(seed)) in done:
                    continue
                res = episode(a, cond, 'test', course, seed,
                              memoryPath=lessons if cond == 'memory' else None, learn=False)
                appendRow(csvPath, res)


def runComparators(a):
    import subprocess
    d = os.path.join(a.out, 'comparators')
    for course in a.courses.split(','):
        for extra in (['--agent', 'baseline'], ['--agent', 'baseline', '--formation', 'vee'],
                      ['--agent', 'baseline', '--formation', 'column'], ['--agent', 'adaptive'],
                      ['--agent', 'formsearch']):
            subprocess.run([sys.executable, '-m', 'shepherd.run', *extra, '--course', course,
                            '--seeds', a.seeds, '--no-plot', '--out', d], cwd=ROOT, check=True)


# ---------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------

def loadRows(a):
    rows = []
    for p in glob.glob(os.path.join(a.out, 'comparators', '**', 'episodes.csv'), recursive=True):
        with open(p) as f:
            for r in csv.DictReader(f):
                r['phase'] = 'test'
                rows.append(r)
    p = os.path.join(a.out, 'llm_%s.csv' % a.tag)
    if os.path.exists(p):
        with open(p) as f:
            rows += list(csv.DictReader(f))
    seeds = set(str(s) for s in parseSeeds(a.seeds))
    # de-duplicate (agent, phase, course, seed), keep last
    keep = {}
    for r in rows:
        keep[(r['agent'], r['phase'], r['course'], r['seed'])] = r
    return [r for r in keep.values() if r['phase'] == 'train' or r['seed'] in seeds], \
           [r for r in keep.values() if r['phase'] == 'train']


def bootCI(x, n=4000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def summarise(rows, courses):
    out = []
    for course in courses:
        for key, name, fam in DISPLAY:
            rs = [r for r in rows if r['agent'] == key and r['course'] == course and r['phase'] == 'test']
            if not rs:
                continue
            col = np.array([float(r['collisions']) for r in rs])
            free = int(np.sum(col == 0))
            calls = [float(r['llmCalls']) for r in rs if r.get('llmCalls') not in (None, '')]
            bad = [float(r['llmBadReplies']) for r in rs if r.get('llmBadReplies') not in (None, '')]
            wall = [float(r['llmWall_s']) for r in rs if r.get('llmWall_s') not in (None, '')]
            out.append(dict(
                course=course, key=key, name=name, family=fam, n=len(rs),
                collisions=col.mean(), ciLo=bootCI(col)[0], ciHi=bootCI(col)[1],
                wall=np.mean([float(r['fan-obstacle']) for r in rs]),
                freeFrac=free / len(rs), freeLo=wilson(free, len(rs))[0], freeHi=wilson(free, len(rs))[1],
                lost=np.mean([float(r['fansLost']) for r in rs]),
                posErr=np.mean([float(r['posErr']) for r in rs]),
                headErr=np.mean([float(r['headErrDeg']) for r in rs]),
                time=np.mean([float(r['time']) for r in rs]),
                switches=np.mean([float(r['formationSwitches']) for r in rs if r.get('formationSwitches') not in (None, '')] or [np.nan]),
                calls=np.mean(calls) if calls else np.nan,
                badFrac=(np.sum(bad) / np.sum(calls)) if calls and np.sum(calls) else np.nan,
                secPerCall=(np.sum(wall) / np.sum(calls)) if calls and np.sum(calls) else np.nan))
    return out


def writeTables(a, S):
    with open(os.path.join(a.out, 'summary_%s.csv' % a.tag), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(S[0]))
        w.writeheader()
        w.writerows(S)
    L = [r'\begin{table}[h!]', r'\centering', r'\small',
         r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{llccccc}', r'\hline',
         r'Course & Decider & Collisions per run & Collision-free runs & Lost & Pose error (m, $^\circ$) & LLM calls \\',
         r'\hline']
    for course in dict.fromkeys(r['course'] for r in S):
        first = True
        for r in [r for r in S if r['course'] == course]:
            L.append('%s & %s & %.1f [%.1f, %.1f] & %.2f [%.2f, %.2f] & %.1f & %.2f, %.1f & %s \\\\' % (
                course if first else '', r['name'], r['collisions'], r['ciLo'], r['ciHi'],
                r['freeFrac'], r['freeLo'], r['freeHi'], r['lost'], r['posErr'], r['headErr'],
                '%.1f' % r['calls'] if np.isfinite(r['calls']) else '--'))
            first = False
        L.append(r'\hline')
    n = max(r['n'] for r in S)
    L += [r'\end{tabular}}',
          r'\caption{Formation-and-speed supervision on the corridor courses, %d boids, paper boid '
          r'parameters, %d seeds per row. Every decider shares one action space (formation preset or '
          r'custom spec, leader speed) and one low-level controller (vector-field guidance, Dubins '
          r'terminal path). The reference row uses damped boid parameters (see text) and is not one of the '
          r'deciders. Collisions per run with 95\%% bootstrap intervals; collision-free fraction '
          r'with 95\%% Wilson intervals. LLM: %s.}' % (a.nfan, n, a.model.replace('_', r'\_')),
          r'\label{tab:llm}', r'\end{table}']
    with open(os.path.join(a.out, 'summary_%s.tex' % a.tag), 'w') as f:
        f.write('\n'.join(L) + '\n')


def _style(plt):
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#0b0b0b', 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def figCollisions(a, S):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    courses = list(dict.fromkeys(r['course'] for r in S))
    names = [n for _, n, _ in DISPLAY if any(r['name'] == n for r in S)]
    fig, axs = plt.subplots(1, len(courses), figsize=(3.4 * len(courses), 0.38 * len(names) + 1.1),
                            sharey=True, sharex=True, squeeze=False)
    for ax, course in zip(axs[0], courses):
        for i, nm in enumerate(names):
            r = next((r for r in S if r['course'] == course and r['name'] == nm), None)
            if r is None:
                continue
            y = len(names) - 1 - i
            c = FAMILY_COLOR[r['family']]
            if np.isfinite(r['ciLo']):
                ax.plot([r['ciLo'], r['ciHi']], [y, y], color=c, lw=2, solid_capstyle='round')
            if r['family'] == 'reference':        # different boid parameters: hollow, set apart
                ax.plot(r['collisions'], y, 'o', ms=7, mfc='white', mec=c, mew=1.8, zorder=3)
                ax.axhline(y + 0.5, color='#d4d3ce', lw=0.8, ls=(0, (3, 3)))
            else:
                ax.plot(r['collisions'], y, 'o', ms=7, color=c, mec='white', mew=1.5, zorder=3)
            ax.text((r['ciHi'] if np.isfinite(r['ciHi']) else r['collisions']) + 0.6, y, '%.1f' % r['collisions'], va='center', fontsize=8, color='#52514e')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1])
        top = max([r['ciHi'] for r in S if np.isfinite(r['ciHi'])] + [r['collisions'] for r in S])
        ax.set_xlim(-1.0, 1.18 * top + 4)
        ax.set_title(course, fontsize=10, color='#0b0b0b')
        ax.set_xlabel('collisions per run (mean, 95% CI)')
        ax.grid(axis='x', color='#e6e5e1', lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, 'fig_collisions_%s.%s' % (a.tag, ext)), dpi=200)
    plt.close(fig)


def figLearning(a, train):
    if not train:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    train = sorted(train, key=lambda r: int(r['seed']))
    k = np.arange(1, len(train) + 1)
    col = np.array([float(r['collisions']) for r in train])
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(k, col, '-o', color=FAMILY_COLOR['llm'], lw=2, ms=6, mec='white', mew=1.5)
    for kk, r in zip(k, train):
        ax.annotate(r['course'][0], (kk, float(r['collisions'])), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=7, color='#52514e')
    ax.set_xlabel('training episode (lessons rewritten after each)')
    ax.set_ylabel('collisions')
    ax.set_xticks(k)
    ax.grid(axis='y', color='#e6e5e1', lw=0.8)
    ax.set_title('LLM + lessons: training (n = narrows, d = dogleg)', fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, 'fig_learning_%s.%s' % (a.tag, ext)), dpi=200)
    plt.close(fig)


def figTrajectories(a, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    courses = a.courses.split(',')
    conds = [c for c in a.conditions.split(',')]
    panels = []
    for course in courses:
        for cond in conds:
            rs = [r for r in rows if r['agent'] == 'llm-' + cond and r['course'] == course and r['phase'] == 'test']
            if not rs:
                continue
            rs.sort(key=lambda r: float(r['collisions']))
            med = rs[len(rs) // 2]
            p = os.path.join(a.out, 'traj', a.tag, '%s_test_%s_s%s.npz' % (cond, course, med['seed']))
            if os.path.exists(p):
                panels.append((course, cond, med, p))
    if not panels:
        return
    fig, axs = plt.subplots(len(courses), len(conds), figsize=(3.3 * len(conds), 3.0 * len(courses)), squeeze=False)
    for ax in axs.ravel():
        ax.axis('off')
    for course, cond, med, p in panels:
        ax = axs[courses.index(course)][conds.index(cond)]
        ax.axis('on')
        d = np.load(p)
        crs = get_course(course)
        for (x, y, r) in crs.obstacles:
            ax.add_patch(plt.Circle((x, y), r, color='#b9b8b3', lw=0))
        for i in range(d['fanX'].shape[1]):
            ax.plot(d['fanX'][:, i], d['fanY'][:, i], lw=0.5, color=FAMILY_COLOR['rule'], alpha=0.45)
        ax.plot(d['x'], d['y'], color='#0b0b0b', lw=1.6)
        if len(d['col']):
            ax.plot(d['col'][:, 0], d['col'][:, 1], 'x', color='#d03b3b', ms=5, mew=1.5)
        b = crs.bounds
        ax.set_xlim(b[0], b[1]); ax.set_ylim(b[2], b[3]); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('%s, %s: %s collisions (median run)' % (course, cond, med['collisions']), fontsize=8)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, 'fig_traj_%s.%s' % (a.tag, ext)), dpi=200)
    plt.close(fig)


def report(a):
    rows, train = loadRows(a)
    S = summarise(rows, a.courses.split(','))
    if not S:
        print('nothing to report yet')
        return
    writeTables(a, S)
    figCollisions(a, S)
    figLearning(a, train)
    figTrajectories(a, rows)
    print('\n%-9s %-28s %3s %18s %16s %5s %6s %7s %6s' % ('course', 'decider', 'n', 'collisions [95%CI]',
                                                         'free [95%CI]', 'lost', 'calls', 'bad', 's/call'))
    for r in S:
        print('%-9s %-28s %3d %6.1f [%4.1f,%4.1f] %5.2f [%.2f,%.2f] %5.1f %6s %7s %6s' % (
            r['course'], r['name'], r['n'], r['collisions'], r['ciLo'], r['ciHi'], r['freeFrac'],
            r['freeLo'], r['freeHi'], r['lost'],
            '%.1f' % r['calls'] if np.isfinite(r['calls']) else '-',
            '%.0f%%' % (100 * r['badFrac']) if np.isfinite(r['badFrac']) else '-',
            '%.1f' % r['secPerCall'] if np.isfinite(r['secPerCall']) else '-'))
    print('\nwrote summary_%s.csv/.tex and figures to %s' % (a.tag, a.out))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='qwen2.5:14b')
    p.add_argument('--host', default='http://localhost:11434')
    p.add_argument('--temperature', type=float, default=0.3)
    p.add_argument('--courses', default='narrows,dogleg')
    p.add_argument('--seeds', default='0-9')
    p.add_argument('--conditions', default='zeroshot,memory,whatif')
    p.add_argument('--train-episodes', type=int, default=6)
    p.add_argument('--nfan', type=int, default=8)
    p.add_argument('--out', default=os.path.join(ROOT, 'results', 'llm_study'))
    p.add_argument('--mock', action='store_true', help='no model: a scripted stand-in, to test the pipeline')
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--run-comparators', action='store_true')
    p.add_argument('--verbose', action='store_true', help='print every LLM decision')
    a = p.parse_args()
    # no dots: LaTeX's graphicx treats the first dot as the extension
    a.tag = 'mock' if a.mock else a.model.replace(':', '_').replace('/', '_').replace('.', '-')
    for sub in ('traj', 'transcripts'):
        os.makedirs(os.path.join(a.out, sub, a.tag), exist_ok=True)
    if not a.mock and not a.report_only:
        try:
            OllamaClient(model=a.model, host=a.host, timeout=120).chat(
                [dict(role='user', content='Reply with {"ok": true}')])
        except Exception as e:
            sys.exit('Cannot reach Ollama model %r at %s: %s\n(ollama serve; ollama pull %s)'
                     % (a.model, a.host, e, a.model))
    if a.run_comparators:
        runComparators(a)
    if not a.report_only:
        runStudy(a)
    report(a)


if __name__ == '__main__':
    main()
