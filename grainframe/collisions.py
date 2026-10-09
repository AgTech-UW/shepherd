"""Collision reporting across EVERY pair of agents, whatever their type.

Like animate.py, this is deliberately dumb about where the data came from.
You hand it a list of Agent objects (each with an x,y history, a name, a kind,
and a team) and it checks every pair over the whole run, then prints a
breakdown by category:

    leader-leader     two captains hit each other
    leader-fan        a fanboid hit a captain (own squad or the other one)
    fan-fan (same)    two fanboids from the SAME squadron hit
    fan-fan (cross)   two fanboids from DIFFERENT squadrons hit

A "collision" is any moment two agents' centres come within `radius`
(default opts.collisionRadius = 2 * bodyRadius, i.e. their disks overlap).
Contiguous frames get grouped into ONE episode, so a long clinch counts once.
"""
import numpy as np


class Agent:
    """One agent's trajectory, tagged so collisions can be categorized.

    x, y : equal-length position histories
    name : display name, e.g. 'Red Leader' or 'Red fan 3'
    kind : 'leader' or 'fan'
    team : squadron id (anything hashable: 0/1, 'Red'/'Gold', ...)
    """
    def __init__(self, x, y, name, kind, team=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.name = name
        self.kind = kind
        self.team = team


def _episodes(d, radius):
    # Group contiguous below-radius frames into (startIdx, endIdx) episodes
    below = d < radius
    if not np.any(below):
        return []
    idx = np.flatnonzero(below)
    splits = np.flatnonzero(np.diff(idx) > 1)      # gaps = episode boundaries
    groups = np.split(idx, splits + 1)
    return [(int(g[0]), int(g[-1])) for g in groups]


def _category(a, b):
    if a.kind == 'leader' and b.kind == 'leader':
        return 'leader-leader'
    if a.kind != b.kind:                            # one leader, one fan
        fan = a if a.kind == 'fan' else b
        lead = b if a.kind == 'fan' else a
        return ('leader-fan (own squad)' if fan.team == lead.team
                else 'leader-fan (other squad)')
    return ('fan-fan (same squad)' if a.team == b.team
            else 'fan-fan (cross squad)')


CATEGORIES = ['leader-leader',
              'leader-fan (own squad)', 'leader-fan (other squad)',
              'fan-fan (same squad)', 'fan-fan (cross squad)']


def collision_report(agents, radius, dt=0.01, verbose=True, listEpisodes=True):
    """Check every pair of agents over the whole run and print a breakdown.

    Returns a dict: category -> list of episode dicts, plus 'closest' giving
    the nearest approach seen in each category (even when nothing collided).
    """
    byCat = {c: [] for c in CATEGORIES}
    closest = {c: np.inf for c in CATEGORIES}

    T = min(len(a.x) for a in agents)
    n = len(agents)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = agents[i], agents[j]
            d = np.hypot(a.x[:T] - b.x[:T], a.y[:T] - b.y[:T])
            cat = _category(a, b)
            closest[cat] = min(closest[cat], float(d.min()))

            for t0, t1 in _episodes(d, radius):
                tw = t0 + int(np.argmin(d[t0:t1 + 1]))   # deepest moment
                byCat[cat].append({
                    'a': a.name, 'b': b.name,
                    'tStart': t0 * dt, 'tEnd': t1 * dt,
                    'tWorst': tw * dt,
                    'minDist': float(d[tw]),
                    'x': 0.5 * (a.x[tw] + b.x[tw]),
                    'y': 0.5 * (a.y[tw] + b.y[tw]),
                })

    if verbose:
        total = sum(len(v) for v in byCat.values())
        print('\n' + '=' * 62)
        print('COLLISION REPORT   radius %.2f   %d agents   %d timesteps'
              % (radius, n, T))
        print('=' * 62)
        for c in CATEGORIES:
            eps = byCat[c]
            near = closest[c]
            nearStr = '  n/a' if not np.isfinite(near) else '%5.2f' % near
            flag = '  <-- COLLISION' if eps else ''
            print('%-24s : %2d episode(s)   closest %s%s'
                  % (c, len(eps), nearStr, flag))
            if listEpisodes:
                for e in eps:
                    print('      %s <-> %s   t=%.1f-%.1fs   min %.3f  at (%.1f, %.1f)'
                          % (e['a'], e['b'], e['tStart'], e['tEnd'],
                             e['minDist'], e['x'], e['y']))
        print('-' * 62)
        print('TOTAL: %d collision episode(s)' % total)
        print('=' * 62 + '\n')

    byCat['closest'] = closest
    return byCat


# ---------------- glue helpers for the usual data shapes ----------------

def agents_from_arrays(leaders, fans, fanTeams, leaderNames=None,
                       teamNames=None):
    """Build agents from flat arrays (the sandbox style).

    leaders    : list of (x, y) per leader, in team order 0, 1, ...
    fans       : list of (x, y) per fanboid
    fanTeams   : team id for each fanboid (same length as fans)
    leaderNames: optional display names per leader
    teamNames  : optional display names per team, used to name the fans
    """
    agents = []
    for k, (lx, ly) in enumerate(leaders):
        nm = leaderNames[k] if leaderNames else 'Leader %d' % k
        agents.append(Agent(lx, ly, nm, 'leader', team=k))
    for i, (fx, fy) in enumerate(fans):
        tm = fanTeams[i]
        tn = teamNames[tm] if teamNames else 'Team%s' % tm
        agents.append(Agent(fx, fy, '%s fan %d' % (tn, i), 'fan', team=tm))
    return agents


def agents_from_fleets(fleets):
    """Build agents from Fleet objects (the multifleet style)."""
    agents = []
    for k, fl in enumerate(fleets):
        agents.append(Agent(fl.log['x'], fl.log['y'], fl.name, 'leader', team=k))
        if fl.log['fanX'].size > 0:
            fx, fy = fl.log['fanX'], fl.log['fanY']
            for b in range(fx.shape[1]):
                agents.append(Agent(fx[:, b], fy[:, b],
                                    '%s fan %d' % (fl.name, b), 'fan', team=k))
    return agents
