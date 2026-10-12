"""Replay a recorded language-model episode without the model.

The simulator is deterministic, so feeding the recorded replies back to the
supervisor at the same decision points reproduces the episode exactly. Every
decision time is checked against the transcript. This rebuilds trajectories
from transcripts alone (the study's zip leaves the bulky .npz files out),
and lets the same decisions be replayed with a different follower model.

    from shepherd.replay import replay
    env, info = replay('transcripts/.../zeroshot_north_test_s1.json')
"""
import json

import numpy as np

from .farm import FarmSpec, buildFarm
from .farmenv import FarmEnv
from .farmsupervisor import FarmSupervisor
from .followers import followerSetup
from .run import runEpisode


class _Recorded:
    """Stands in for the model: returns each decision's final reply in turn."""
    def __init__(self, turns):
        self.replies = [t.get('reply2') or t['reply'] for t in turns]
        self.k = 0

    def chat(self, messages, json_mode=True):
        r = self.replies[self.k] if self.k < len(self.replies) else '{}'
        self.k += 1
        return r


class _Replayer(FarmSupervisor):
    """The supervisor as recorded: same triggers, same parsing, zero-shot
    plumbing (one call per decision), so what-if, lessons and memory
    transcripts replay without a simulator search, lessons file or embedder."""
    def __init__(self, turns, instruction, shield=False):
        super().__init__(_Recorded(turns), instruction=instruction, condition='zeroshot', verbose=False,
                         shield=shield)
        self.turns = turns
        self.mismatch = []

    def act(self, env, obs):
        k = len(self.transcript)
        out = super().act(env, obs)
        if len(self.transcript) > k and k < len(self.turns):
            if abs(self.transcript[-1]['t'] - self.turns[k]['t']) > 1e-6:
                self.mismatch.append((k, self.turns[k]['t'], self.transcript[-1]['t']))
        return out


def replay(transcript, spec='farms/default.json', followers='vfield', fitVersion=None, shield=None, envAttrs=None):
    """Re-run a recorded episode. Returns (env, info); info['exact'] is True
    when every decision fell at the recorded time. fitVersion and shield
    default to what the transcript recorded; setting them replays the same
    replies, in order, under a different fit check or filter (a counterfactual:
    once the run diverges, later replies were written for a different state)."""
    d = json.load(open(transcript)) if isinstance(transcript, str) else transcript
    res = d['result']
    sp = FarmSpec.load(spec)
    opts, kw = followerSetup(followers)
    env = FarmEnv(buildFarm(sp, int(res['seed'])), instruction=res['instruction'], opts=opts, **kw)
    env.fitVersion = fitVersion if fitVersion is not None else d.get('fitVersion', 1)
    for k, v in (envAttrs or {}).items():     # e.g. reflexLook, reflexMargin for a counterfactual
        setattr(env, k, v)
    ag = _Replayer(d['turns'], res['instruction'], shield=(d.get('shield') is not None) if shield is None else shield)
    r = runEpisode(env, ag)
    info = dict(result=r, decisions=len(ag.transcript), recorded=len(d['turns']), mismatch=ag.mismatch,
                exact=(not ag.mismatch and len(ag.transcript) == len(d['turns'])))
    return env, info


def trajectory(env):
    L = env.log
    return np.array(L['x']), np.array(L['y']), np.array(L['fanX']), np.array(L['fanY'])
