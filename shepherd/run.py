"""Run agents on courses and compare them.

  python -m shepherd.run --agent baseline --course uturn --nfan 8 --seeds 0-4
  python -m shepherd.run --agent cem      --course uturn --nfan 8 --seeds 0-4
  python -m shepherd.run --agent llm --model qwen2.5:14b --course uturn --seeds 0-4 \
        --whatif --memory results/shepherd/memory_uturn.json

Every episode appends one row to results/shepherd/episodes.csv and writes a
trajectory plot; LLM episodes also write the full transcript as JSON.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from grainframe import SimOptions

from .courses import get_course, COURSES

# Boid parameters that damp the slot oscillation without changing any rule.
# See README "Damping the boids" and shepherd/sweep_damping.py.
TUNED_BOIDS = dict(fanLeaderFactor=2.0, SEP_PROFILE='norm', VR=20.0, CF=0.0, AF=4.0)
from .env import ShepherdEnv, score
from .agents import (BaselineAgent, CEMAgent, LLMAgent, OllamaClient, MockLLM,
                     AdaptiveFormationAgent, FormationSearchAgent, preset)


def runEpisode(env, agent):
    obs = env.reset()
    agent.begin(env)
    t0 = time.time()
    while not env.done:
        env.setPlan(agent.act(env, obs))
        events = env.runFor(env.decisionDt)
        if hasattr(agent, 'observeEvents'):
            agent.observeEvents(events)
        obs = env.observe()
    res = env.result()
    res['score'] = score(res)
    res['wall_s'] = round(time.time() - t0, 1)
    agent.end(env, res)
    return res


def plotEpisode(env, res, path, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    L = env.log
    fx, fy = np.array(L['fanX']), np.array(L['fanY'])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (x, y, r) in env.course.obstacles:
        ax.add_patch(plt.Circle((x, y), r, color='0.55', zorder=1))
    if fx.size:
        for i in range(fx.shape[1]):
            ax.plot(fx[:, i], fy[:, i], lw=0.6, alpha=0.55, zorder=2)
        k = np.linspace(0, len(L['t']) - 1, 7).astype(int)
        ax.scatter(fx[k].ravel(), fy[k].ravel(), s=6, c='C0', alpha=0.6, zorder=3)
    ax.plot(L['x'], L['y'], 'k', lw=2, zorder=4, label='leader')
    if env.course.route:
        r = np.array(env.course.route)
        ax.plot(r[:, 0], r[:, 1], 'k+', ms=10, label='route')
    for q, c, lab in [(env.course.start, 'g', 'start'), (env.course.goal, 'm', 'goal pose')]:
        ax.arrow(q[0], q[1], 6 * np.cos(q[2]), 6 * np.sin(q[2]), width=0.8, color=c, zorder=5, label=lab)
    cols = {'fan-leader': 'r', 'fan-fan': 'orange', 'fan-obstacle': 'purple', 'leader-obstacle': 'k'}
    for cat, c in cols.items():
        pts = [(e['x'], e['y']) for e in env.collisions if e['cat'] == cat]
        if pts:
            p = np.array(pts)
            ax.scatter(p[:, 0], p[:, 1], marker='x', s=50, c=c, zorder=6, label='%s (%d)' % (cat, len(p)))
    ax.set_aspect('equal')
    b = env.course.bounds
    ax.set_xlim(b[0], b[1]); ax.set_ylim(b[2], b[3])
    ax.set_title('%s\npos err %.2f m, heading err %.1f deg, collisions %d, lost %d, t %.0f s, score %.1f'
                 % (title, res['posErr'], res['headErrDeg'], res['collisions'], res['fansLost'],
                    res['time'], res['score']), fontsize=9)
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parseSeeds(s):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


def makeAgent(a):
    if a.agent == 'baseline':
        return BaselineAgent()
    if a.agent == 'adaptive':
        return AdaptiveFormationAgent()
    if a.agent == 'formsearch':
        return FormationSearchAgent(verbose=a.verbose)
    if a.agent == 'cem':
        return CEMAgent(pop=a.cem_pop, iters=a.cem_iters, horizon=a.cem_horizon,
                        planDt=a.cem_dt, verbose=a.verbose, workers=a.workers)
    if a.agent in ('llmsup', 'mocksup'):
        from .supervisor import LLMSupervisor, MockSupervisorLLM
        client = MockSupervisorLLM() if a.agent == 'mocksup' else \
            OllamaClient(model=a.model, host=a.host, temperature=a.temperature)
        return LLMSupervisor(client, whatIf=a.whatif, memoryPath=a.memory, verbose=True)
    if a.agent in ('llm', 'mock'):
        client = MockLLM(whatIf=a.whatif) if a.agent == 'mock' else \
            OllamaClient(model=a.model, host=a.host, temperature=a.temperature)
        return LLMAgent(client, whatIf=a.whatif, memoryPath=a.memory, verbose=True)
    raise ValueError(a.agent)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent', default='baseline', choices=['baseline', 'adaptive', 'formsearch', 'cem', 'llm', 'mock', 'llmsup', 'mocksup'])
    p.add_argument('--course', default='uturn', choices=list(COURSES) + ['all'])
    p.add_argument('--nfan', type=int, default=8)
    p.add_argument('--seeds', default='0')
    p.add_argument('--speed-min', type=float, default=0.5)
    p.add_argument('--speed-max', type=float, default=1.5)
    p.add_argument('--decision-dt', type=float, default=2.0)
    p.add_argument('--boids-see-obstacles', action='store_true')
    p.add_argument('--boids', default='paper', choices=['paper', 'tuned'],
                   help="boid parameters: 'paper' (SimOptions defaults) or 'tuned' (damped: "
                        "fanLeaderFactor 2, 1/r separation, VR 20, CF 0, AF 4; see README)")
    p.add_argument('--formation', default='none',
                   help='initial formation: preset (none, vee, veeTrack, column, column2, columnBody) or a JSON spec')
    p.add_argument('--model', default='llama3.1:8b')
    p.add_argument('--host', default='http://localhost:11434')
    p.add_argument('--temperature', type=float, default=0.3)
    p.add_argument('--whatif', action='store_true')
    p.add_argument('--memory', default=None, help='JSON file of lessons carried across episodes')
    p.add_argument('--cem-pop', type=int, default=24)
    p.add_argument('--cem-iters', type=int, default=2)
    p.add_argument('--cem-horizon', type=float, default=12.0)
    p.add_argument('--cem-dt', type=float, default=0.05, help='rollout timestep (model is coarser than the plant)')
    p.add_argument('--workers', type=int, default=None, help='rollout processes (default: cores-1)')
    p.add_argument('--out', default='results/shepherd')
    p.add_argument('--tag', default='')
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--verbose', action='store_true')
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    courses = list(COURSES) if a.course == 'all' else [a.course]
    label = a.agent + ('-tuned' if a.boids == 'tuned' else '') + ('' if a.formation == 'none' else '-' + str(a.formation)[:20]) + ('-whatif' if a.whatif else '') + (('-' + a.model) if a.agent in ('llm', 'llmsup') else '') + a.tag
    csvPath = os.path.join(a.out, 'episodes.csv')
    for cname in courses:
        for seed in parseSeeds(a.seeds):
            agent = makeAgent(a)
            opts = SimOptions()
            if a.boids == 'tuned':
                for k, v in TUNED_BOIDS.items():
                    setattr(opts, k, v)
            env = ShepherdEnv(get_course(cname), nFan=a.nfan, seed=seed, opts=opts,
                              speedRange=(a.speed_min, a.speed_max), decisionDt=a.decision_dt,
                              boidsSeeObstacles=a.boids_see_obstacles,
                              formation=preset(a.formation))
            print('== %s | %s | nFan %d | seed %d' % (label, cname, a.nfan, seed))
            res = runEpisode(env, agent)
            if hasattr(agent, 'close'):
                agent.close()
            res['agent'] = label
            if hasattr(agent, 'nCalls') and hasattr(agent, 'nBad'):
                res['llmCalls'], res['llmBadReplies'], res['llmWall_s'] = agent.nCalls, agent.nBad, round(agent.wall, 1)
            print('   ' + ', '.join('%s=%s' % (k, round(v, 2) if isinstance(v, float) else v)
                                    for k, v in res.items() if k not in ('course', 'seed', 'agent')))
            stem = os.path.join(a.out, '%s_%s_n%d_s%d' % (label.replace(':', '_'), cname, a.nfan, seed))
            if not a.no_plot:
                plotEpisode(env, res, stem + '.png', '%s on %s, %d fanboids, seed %d' % (label, cname, a.nfan, seed))
            if hasattr(agent, 'transcript') and hasattr(agent, 'system'):
                with open(stem + '_transcript.json', 'w') as f:
                    json.dump(dict(system=agent.system, turns=agent.transcript), f, indent=1)
            cols = ['agent', 'course', 'nFan', 'seed', 'captured', 'posErr', 'headErrDeg', 'collisions',
                    'fan-leader', 'fan-fan', 'fan-obstacle', 'leader-obstacle', 'fansLost', 'formationSwitches', 'time',
                    'minFanLeader', 'minFanFan', 'minObsClear', 'score', 'wall_s',
                    'llmCalls', 'llmBadReplies', 'llmWall_s']
            new = not os.path.exists(csvPath)
            with open(csvPath, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
                if new:
                    w.writeheader()
                w.writerow(res)


if __name__ == '__main__':
    main()
