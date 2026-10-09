# shepherd: an agent leads a flock of plain boids

One leader, a Dubins car with turn radius R = 5, is driven by an agent. It
leads N fanboids, which are the paper's followers unchanged:
`grainframe.flocking.flockStep` with the leader pull, then the Ackermann
clamp. The agent's job is to arrive **exactly** on a goal pose (position and
heading) while the flock causes as few collisions as possible. The fanboids
are still just boids. They don't see obstacles, so the leader has to herd
them around obstacles, not just steer itself.

`grainframe/` is an unmodified copy of the Part I simulator (see the
top-level README). `shepherd/` only imports from it.

## The comparison

All three agents use the same action vocabulary and the same physics. They
differ only in where the decisions come from:

| agent      | decides by                                                        | knows the boid rules?        |
|------------|-------------------------------------------------------------------|------------------------------|
| `baseline` | the paper's controller: vector-field route following, then a Dubins path to the goal pose | no, ignores the flock |
| `cem`      | model-predictive search: perturbs the baseline with a short manoeuvre, scores candidates on a copy of the simulator | implicitly, through rollouts |
| `llm`      | an Ollama model reads a text observation and writes a JSON plan  | only the description in its prompt |
| `llm --whatif` | the LLM proposes up to 4 plans, the simulator scores them, and the best one runs. The LLM sees every predicted outcome on its next turn | prompt plus experiments |

`baseline` is the physics comparator. `cem` is the yardstick for whether the
boid rules can be gamed at all, using search with no language. The question
for the LLM is where it falls between the two, and whether `--memory`
(lessons carried across episodes) moves it.

## Run

```bash
cd shepherd          # the repository root
python -m shepherd.run --agent baseline --course all --seeds 0-4
python -m shepherd.run --agent cem      --course uturn --seeds 0-4          # minutes per episode
python -m shepherd.run --agent mock     --course gate --seeds 0             # LLM plumbing, no Ollama needed

# Ollama (ollama serve running; pull a model first, e.g. `ollama pull qwen2.5:14b`)
python -m shepherd.run --agent llm --model qwen2.5:14b --course uturn --seeds 0-4
python -m shepherd.run --agent llm --model qwen2.5:14b --course uturn --seeds 0-9 \
       --whatif --memory results/shepherd/lessons_uturn.json

python -m shepherd.compare results/shepherd/episodes.csv results/shepherd/*/episodes.csv
```

Every episode appends one row to `episodes.csv` and saves a trajectory PNG.
LLM episodes also save `*_transcript.json`, which holds the system prompt and
every observation/reply pair. The reasoning to read for "new equilibria" is in
there.

Useful flags: `--nfan`, `--speed-min/--speed-max` (default 0.5 to 1.5;
set both to 1 for the paper's constant-speed leader), `--decision-dt`
(how often the agent is consulted, in sim-seconds, default 2),
`--boids-see-obstacles` (gives the boids an obstacle-separation term),
`--temperature`, `--host`.

## Pieces

- `env.py`: `ShepherdEnv`. The step is dt = 0.01. It counts collisions as
  episodes, the same way `collisions.py` does: fan-leader, fan-fan,
  fan-obstacle and leader-obstacle. A boid scraping along a wall counts once.
  The goal is captured at closest approach, the paper's CPA rule, but only on
  a pass aligned within 30° of the goal heading. Pose error is measured at
  that instant.
- Plan segments: `arc` (turn in [-1,1], speed, duration), `goto` (shortest
  Dubins path to a pose; this is how anyone hits the pose exactly), `follow`
  (vector-field guidance through waypoints, the paper's law) and `seek`.
  Out-of-range values are clamped, not rejected. If an agent re-sends the
  segment that is already running, it continues instead of restarting.
- `dubins.py`: a shortest-path solver over all six words. It is checked on
  3,000 random pose pairs, and the endpoint error is about 1e-11.
- `courses.py`: `open`, `gate` (16 m gap), `slalom` (4 posts), `uturn`
  (go round a divider and come back pointing home, the headland case),
  `narrows` and `dogleg` (corridors, see Formation control).
- `formation.py`: slot layouts, track-frame slots, Hungarian assignment,
  corridor ray casts.
- `agents.py`: the agents, the Ollama client (standard library only, no new
  dependencies), the prompt, and the rollout cost.
- Score (lower is better): posErr + 0.2·headErr° + 10·collisions +
  100 extra per leader-obstacle hit + 25·fansLost + 0.02·time + 200 if the goal is never reached. The weights
  are a choice. Always report the components alongside the score.

## Things to watch for (they're results, not bugs)

- **Abandoning the flock is a valid exploit.** A leader that outruns or
  shakes off its boids has nobody to collide with. `fansLost` (more than
  40 m from the leader at the end) is penalised so it shows up instead of
  hiding inside a low collision count.
- **CEM has a perfect model.** Rollouts run on a copy of the true state at
  dt = 0.05. That makes it an upper bound on what search can find, not a
  fair opponent for an LLM that only reads text. `--whatif` puts the LLM on
  the same footing.
- **The LLM plans and the physics executes.** At 1 call per 2 sim-seconds,
  the LLM never closes the control loop. The `goto` primitive is what makes
  "nail the pose exactly" achievable for it at all.
- The baseline's collisions come almost entirely from boids cutting corners
  through obstacles they can't see, plus the occasional leader hit that the
  paper describes.

## Hooking into ROS 2

The agent only needs pose and flock state in, and a plan out. The cleanest
bridge is a node that wraps `LLMAgent.act()`. It would subscribe to
`/<name>/gps/pose` and the fanboid poses, call the agent every
`decisionDt`, and execute the segments locally at 100 Hz into
`/<leader>/auto/cmd_vel` (v = speed, ω = turn·speed/R). The executors in
`env.py` are plain classes with a `control(env)` method, so they can be
reused with a small adapter.

## Formation control

The agent can also reshape the flock in the middle of a run. A plan segment
`{"type": "formation", ...}` gives each boid its own slot, using the
paper's slot mechanism in `flockStep`. The boids are still boids: separation,
cohesion and alignment are unchanged, so slots closer together than P fight
each other. The segment takes effect when the plan reaches it, so changes
can be scheduled ("straight 20 s, then column, then follow").

- Shapes: `none` (leader pull), `vee`, `echelon`, `column` (with `ranks`
  abreast), `line`, and `custom` (any list of slots). Each takes
  `spacing`, `standoff`, `openingDeg` and `blend`.
- `frame: "body"` (the paper's) keeps slots rigid behind the leader's
  current heading. `frame: "track"` places them along the path the leader
  actually drove.
- On every switch, boids are assigned to the new slots by minimum total
  travel (Hungarian algorithm), and targets morph over `blend` seconds.
  Assigning by index made boids cross through each other.
- The observation includes `corridorAhead` (free half-width left and right
  along the route ahead, from ray casts) and the current formation's
  half-widths, length, smallest slot gap, and whether it satisfies δ > P.
- `fansLost` now means "more than 40 m from its own slot", so long
  formations aren't penalised for being long.
- New courses: `narrows` (44 m → 14 m → 44 m channel) and `dogleg`
  (20 m corridor with a 90° turn).
- New agents: `adaptive`, a hand-written rule that picks the widest
  formation that satisfies δ > P and fits the corridor ahead, and waits for
  the tail to clear before widening. This is the comparator for formation
  control. `formsearch` picks a formation preset and a speed by rollouts.
- `--formation vee|column|...` or a JSON spec sets a fixed initial
  formation for any agent.

Corridors, 8 boids, 5 seeds, baseline motion:

| course | leader pull (`none`) | paper's vee | column (track) | adaptive rule | column @ 1.25 m/s | **formsearch** |
|---|---|---|---|---|---|---|
| narrows | 44.0 | 27.8 | 10.4 | 9.8  | 5.0 | **0.4** |
| dogleg  | 48.8 | 39.0 | 8.2  | 10.8 | 6.4 | **0.8** |

(mean collisions per run; formsearch switches formation 4 to 9 times per run)

- **Formation is the biggest lever measured so far,** worth 4–5× in
  corridors. Speed and path tweaks never came close.
- **The spacing condition holds here too.** A track column at δ = 6
  collides 3–4× more than at δ = 12.
- **Track-frame slots help at corners.** Dogleg collisions roughly halve
  (13 → 6 average over 3 seeds) compared with body-frame slots.
- **The remaining corridor collisions are slot oscillation.** Displaced
  boids orbit their slot instead of settling, and in a 14 m channel the
  orbit is wider than the free space. Slowing below the boids' minimum
  speed (0.8) makes it far worse, because they overrun their slots.
- **Adapting by geometry buys nothing, but adapting by state buys about
  10×.** The rule that matches formation width to corridor width only ties
  a fixed column. Rollout search over the same six presets and two speeds
  gets 0.4 and 0.8 collisions per run. Its strategy is legible: a vee in
  the open, then two abreast, then single file before the channel, at
  1.25 m/s. Speed alone accounts for some of the gain (column at 1.25 m/s:
  5 to 6), but not most of it. The rest comes from **when** it switches.
  It re-forms when its rollouts see a boid about to graze a wall.
  Hypothesis, not yet tested: each re-form re-targets the slots from where
  the boids currently are, which damps the slot oscillation. That would be
  a use of the formation as a control input rather than a shape. Check it
  by comparing against periodic re-forms on a timer.
- So the formation decision has to be closed-loop on the flock's state, not
  only on the map. An LLM reading `formation.meanSlotError` and the
  positions of boids near walls has something real to reason about, and
  formsearch sets the bar it has to reach.
- **Static obstacles alone make single file nearly sufficient.** A column
  that follows the leader's own track at δ > P stays clear wherever the
  leader is clear. To make shape choice a harder trade-off, charge for what
  a column is bad at: swath coverage (the agricultural case: abreast covers
  N rows, a column one), whole-flock arrival time, or other traffic (the
  paper's second squadron).

## Leader tracking (the wobble)

The `follow` segment is the paper's vector field (`guidanceVectorField`,
kCross = 0.5, chiInf = 70°) with `falconSteering` bang-bang steering and
the paper's closest-approach waypoint handover. `goto` is an open-loop
Dubins path integrated exactly, so it doesn't wobble.

- **Fixed: re-plan wobble.** A re-sent `follow` used to draw its first line
  from wherever the leader was at that moment, and route waypoints counted
  as passed anywhere within 12 m. Every re-plan, from formation switches,
  the LLM or CEM, moved the reference line and cut corners: up to 6.2 m off
  route on the dogleg. `follow` now takes `start` (the agents pass
  `observation.routeLegStart`), and a waypoint only counts as passed once
  the leader crosses the perpendicular through it. Re-planning every 2 s
  now stays within 0.36 m. Formation results on the dogleg didn't change
  (formsearch 0.8, column 8.2, adaptive 10.8).
- **Remaining: the paper's corner overshoot.** Planned once, the leader
  overshoots the dogleg corner by about 1.5 m in an S-bend, and the slalom
  legs by up to 5 m. The cause is the handover: it switches to the next leg
  only after closest approach, once the leader is already moving away, so
  every corner is entered late at full turn rate. Handing over at the
  perpendicular line instead removes most of it (1.5 → 0.36 m). If this
  matters for the paper, the fix is a lead distance of about
  R·tan(Δχ/2) before the corner.
- The boids' loops in the plots are a separate effect: the undamped slot
  spring (see Formation control).

## Damping the boids (settings only, rules unchanged)

The boids circle their slots mostly because of the **turn limit**, not a
lack of damping. On a straight line with no walls, boids at the paper's
R = 5 take about 100 s to form a column and still sit 2–4 m off their slots.
Near point-mass boids (R = 0.5) form up in about 20 s and hold within 1 m. A
follower that can't turn tightly has to fly a loop to correct any error.
That's the paper's premise, so R stays.

What can be changed is the boid parameters. With the paper's settings
(VR = 8 m, slots 12 m apart), a boid in formation sees **nobody**, so
alignment, the one rule that damps, never acts. Raising VR alone makes
things worse, because cohesion then pulls boids off their slots toward
their neighbours. Sweep, single-file track column, baseline motion, mean
collisions (`python -m shepherd.sweep_damping 10 out.csv --round2`):

| boid settings (10 seeds) | narrows | dogleg | slot error RMS (m) |
|---|---|---|---|
| paper (VR 8, CF 0.8, AF 2, pull 1, flat separation) | 11.0 | 7.1 | 7.1 |
| pull 2 | 8.4 | 4.6 | 6.7 |
| pull 3 | 4.5 | 4.1 | 5.2 |
| 1/r separation | 10.2 | 4.4 | 6.0 |
| VR 20, CF 0, AF 4 | 4.0 | 4.7 | 8.2 |
| **pull 2 + 1/r separation + VR 20, CF 0, AF 4** (`--boids tuned`) | **0.5** | **0.5** | **4.3** |

The parts only work together. Seeing far gives alignment something to
match. Cohesion off stops seeing far from wrecking the slots. A stiffer
pull with 1/r separation holds the slot while keeping contact repulsion
strong. With the tuned boids, a **fixed** column matches what formsearch
needed constant re-forming to reach.

Across the other courses (5 seeds, collisions per run, paper → tuned):
open with leader pull 1.0 → 0.0, gate with leader pull 7.6 → 3.4, narrows
with a vee 27.8 → 6.0, U-turn column 2.6 → 0.6, slalom column 2.2 → 1.0.
The cost: in single file, the stiffer pull occasionally drives a boid into
the one ahead (fan-fan 0.2–0.4 → 0.6–1.0 per run). The `slotPullCap` comment
in `flocking.py` describes this trapdoor.

Since these settings change the baseline itself, the agent comparison
should be re-run on `--boids tuned` before anything is claimed. The paper
settings stay the default.

## First numbers (8 fanboids, 3 seeds each, so anecdotal)

| course | agent | collisions | of which boid→obstacle | leader crashes | pos err (m) | heading err (°) | time (s) |
|---|---|---|---|---|---|---|---|
| open   | baseline | 1.3  | 0    | 0    | 0.01 | 0.27 | 87  |
| open   | cem      | 0.3  | 0    | 0    | 0.01 | 0.10 | 64  |
| gate   | baseline | 6.7  | 6.0  | 0    | 0.04 | 0.00 | 103 |
| gate   | cem      | 5.3  | 5.0  | 0    | 0.00 | 0.02 | 78  |
| slalom | baseline | 10.3 | 9.7  | 0    | 0.04 | 0.00 | 207 |
| slalom | cem      | 11.0 | 10.0 | 0.33 | 0.01 | 0.11 | 171 |
| uturn  | baseline | 8.7  | 8.0  | 0    | 0.11 | 0.14 | 237 |
| uturn  | cem      | 9.7  | 9.3  | 0    | 0.00 | 0.13 | 196 |

What this says so far:

- Hitting the pose is solved by the `goto` primitive for everyone, at about
  0.1 m and 0.3°. The interesting variable is collisions.
- A 12 s search horizon finds the cheap wins: going faster and avoiding
  fan-leader hits on open ground. It does **not** find a way to keep blind
  boids off obstacles. Those collisions come from the flock cutting corners
  many seconds after the leader has committed to a line, which is beyond
  what a short horizon can see. This gap is where a strategic planner (an
  LLM, a longer horizon, or learned value) could show something.
- The gate is partly a capacity problem. A flock held apart by PR = 10 is
  wider than the 16 m gap, which echoes the paper's spacing condition. Some
  of those collisions are likely forced no matter how the leader drives, so
  a lower bound for each course is worth estimating before claiming any
  agent is near-optimal.
- Collisions vary widely between seeds (U-turn baseline: 4, 7, 15). Use
  20 or more seeds before reporting anything.
- Early on, the search drove the leader **through** the divider, because
  one collision cost less than a long detour. Leader crashes are now
  priced at 100 in the score and 500 in the rollouts. The search still
  trapped itself once on the slalom: a car that can't stop has to see far
  enough ahead.

## The farm mission (LLM study)

`farm.py`, `farmenv.py`, `farmagents.py`, `farmsupervisor.py`, `memory.py`,
and `experiments/farm_study.py`. The design and hypotheses, written before
any LLM run, are in `experiments/FARM_STUDY.md`.

- **The world is a JSON spec** (`farms/default.json`): field size, implement
  width, lane shape and width, gate, hazard counts and size ranges. Each seed
  randomises where the slough, rock pile and pole go. Copy the file, edit it,
  and pass `--spec`.
- **What every decider chooses:** formation (`line`, `lineLeft`,
  `lineRight`, `vee`, `column2`, `column`, or a custom spec for the LLM),
  speed, and the sideways offset of the pass being worked. The paper's
  controller still drives.
- **Metrics:** collisions, field worked (coverage), machine-seconds on wet
  ground, machine-seconds in the north strip, time, and final pose error.
- **Instructions:** `wet` and `weather` are trained; `north` is held out.
  The model sees words only. The scoring weights are in `farmenv.INSTRUCTIONS`.

```bash
python experiments/farm_comparators.py --seeds 0-7      # rule, search, baselines (hours on 2 cores)
python experiments/farm_study.py --pilot                 # LLM pilot
python experiments/farm_study.py                         # LLM run, resumable
python experiments/farm_study.py --report-only           # tables + figures
```
