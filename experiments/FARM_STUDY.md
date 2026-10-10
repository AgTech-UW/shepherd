# Farm study: design and hypotheses

Written on 9 October 2026, before any language-model run. The prompt, the
decision points, the conditions and the analysis below are frozen once the
pilot has passed. Anything changed after the pilot is listed at the bottom
with the reason.

## Question

Can a small local language model, reading an operator's instruction in
plain words, supervise a turn-limited robot flock through a field operation
as safely as a hand-written controller? Does it do what it is told,
including an instruction nobody programmed for? What does that cost?

The language model never steers. Every decider chooses the same three
things at decision points: formation, leader speed, and sideways offset of
the pass being worked. The paper's vector-field guidance and a Dubins
terminal path drive the leader, and the followers hold their slots by the
same vector field (see "Follower model" below).

## The world

`farms/default.json`: a yard lane between two shelterbelts with two bends,
a field gate, and a 240 x 228 m fenced field worked in two passes with a
headland turn, then back out the gate to park on a pose. There are 9
machines (a leader and 8 followers), each with a 12 m implement. Per field
layout (seed), one slough (a wet low spot that slows machines and ruts the
soil), one rock pile and one power pole are placed at random. Test layouts
are seeds 0–7, and training layouts for the memory conditions are seeds
100–105.

## Instructions (what the operator says)

| key | words given to the model | how it is scored |
|---|---|---|
| wet | "It's wet out there. Stay out of the low spots and don't rut up the field, even if it means leaving some ground undone." | 10 per collision, 2 per machine-second on wet ground, −100 per unit of coverage, 0.01 per second |
| weather | "Rain's coming in tonight. Get as much of this field done as you can before it hits." | 10, 0.1, −300, 0.05 |
| north (held out) | "Neighbour's got cattle along the north fence. Keep the machines well back from that side." | 10, 0.5, −150, 0.01, plus 1 per machine-second in the 25 m strip along the north fence |

The models never see the weights, only the words. The weights are our
operationalisation of the instruction and are reported as such. Every
component is reported separately as well.

## Deciders

- **Single file throughout**: safe and unproductive.
- **Schedule (hazard-blind)**: single file off the field, the full working line on passes.
- **Hand-written rule**: the schedule plus per-hazard handling (thread rocks between machine lanes, fold the wet side away from sloughs on a wet day, plough through on a rain day). It knows `wet` and `weather`. For `north` it runs its wet-day behaviour, because nobody wrote a handler.
- **Oracle search**: at the same decision points as the LLM, it simulates every option (6 formations × 5 offsets × 2 speeds) for 40 s on the true state and scores each with the true instruction cost. It is a reference, not a deployable controller.
- **LLM zero-shot**: the instruction, the farm, the physics in plain words, and the formations with their dimensions.
- **LLM + what-if**: at every decision it proposes 1–4 options. The simulator reports what each would do over 40 s (collisions, wet-ground machine-seconds, coverage gained, closest machine to each fence), and the model then chooses. The simulator does not choose. (Changed after pilot 2; see below.)
- **LLM + lessons**: 6 training days (alternating wet and rain, never the held-out instruction). After each day it rewrites at most 6 lessons. The lessons are frozen for test.
- **LLM + memory stream**: generative-agents memory (Park et al. 2023), trained on the same 6 days. Outcome records are written after every decision, importance comes from the outcome, reflections are made when importance accumulates, and recency + importance + relevance retrieval puts the top 6 into each prompt. The memory is frozen for test.

Every decider sees the same perception: hazard positions, which of its
machine lanes would touch each hazard, corridor widths, the formation's
dimensions, and the machines' extent.

Decision points (LLM and search): the start, each new phase, a hazard
coming into range, the corridor ahead changing width (off the field), the
formation being held no longer fitting, a wider formation having fitted for
6 s, a collision (at most every 8 s), and otherwise every 30 s. ("Fitting"
is defined under pilot 2 below; the two fit triggers were added then.)

## Hypotheses

- **H1 Safety.** LLM zero-shot collisions are within 25% of the hand-written rule's on the trained instructions.
- **H2 Instruction following.** For each LLM condition, paired by layout, wet-ground exposure is lower on wet days than on rain days, and coverage is higher on rain days than on wet days.
- **H3 Held-out instruction.** For each LLM condition, paired by layout, north-strip exposure under the north-fence instruction is lower than under the wet-day instruction. The rule, which has no handler, shows no such difference.
- **H4 Simulator access.** LLM + what-if has lower instruction cost than LLM zero-shot.
- **H5 Experience.** LLM + lessons and LLM + memory have lower instruction cost than zero-shot. Lessons vs memory is exploratory, with no directional prediction.
- **H6 Ceiling.** The oracle search has the lowest instruction cost on every instruction. This is a sanity check of the setup, not a finding.

All comparisons are reported with 95% bootstrap intervals over the 8
layouts. With n = 8 these are estimates, not significance tests. Every
result is reported, including failures.

## Protocol

1. Pilot: `python experiments/farm_study.py --pilot`. Two layouts, wet and north, zero-shot and what-if. Check the parse rate (target below 10% bad replies), seconds per call, and that choices are sensible. Fix plumbing only, not the prompt's content, unless the pilot shows the model cannot understand it at all; any such change is recorded below.
2. Freeze. Record the model tag (`ollama list`) and the commit.
3. Run: `python experiments/farm_study.py`. It is resumable. Do not change anything mid-run.
4. Report: `python experiments/farm_study.py --report-only`.

## Fixes before any LLM run (9 Oct)

Found by the comparators and by a stress test in which a random decider
changes formation, speed and offset every 1–7 s on 14 layouts. All 42
missions now complete.
- Path following could miss a waypoint when the pass was offset and then follow that leg's line forever. Added a fallback handover once the leader crosses the perpendicular at the waypoint. It never fires when the paper's rule works, and the corridor results reproduce exactly.
- The end of a pass now counts as reached when the leader crosses the end line, whatever the offset.
- Offsets are capped at ±48 m, so a pass stays on its own strip.
- The farm's waypoint radius (45 m) was being reset to the base value (12 m) at the start of each episode.

## Follower model (decided 9 Oct, before any LLM run)

The paper's slot spring makes turn-limited followers weave continuously:
10–12°/s of heading change even on a straight. That is unusable for field
machines, so the study uses vector-field slot following
(`shepherd/followers.py`). Each follower tracks a line through its slot with
the paper's vector field, holds its place along the line with speed, may slow
to 0.3 m/s, and keeps the boid separation rule (1/r profile) as a safety
layer; slots are re-matched every 2 s. Compared over 5 corridor seeds and 4
farm seeds (`python -m shepherd.compare_followers`):

| follower model | weaving (°/s) | collisions, single file in corridors | farm collisions (schedule / rule) |
|---|---|---|---|
| paper boids | 10–12 | 8–10 | 13.5 / 13.5 |
| damped boids (parameters only) | 3–5 | 0.6 | 1.8 / 0.8 |
| vector-field slot following | 1.3–2.7 | 0 | 2.5 / 2.0 |

`--followers paper|damped` reruns the whole study with another model.

## Changes after the pilot

Pilot 1 (9 Oct, qwen2.5:7b, seeds 0–1, wet and north, zero-shot and
what-if). Plumbing worked, but the model misread two things in the
observation, so these two changes were made to what it is shown. The
decision points, conditions, instructions, costs and analysis are unchanged.

- **Offset direction.** Offsets were described only as "+ = left of
  travel". On the north-fence day the zero-shot model moved the pass the
  wrong way (756 machine-seconds in the north strip on one layout), because
  left is north on the eastbound pass and south on the westbound one. The
  pass observation now includes `offsetMeaning`, which says which compass
  direction a positive and a negative offset move the line on the current
  pass (east pass: + north; west pass: + south).
- **Which formations fit.** Off the field the model was shown corridor
  widths and formation widths separately and had to compare them itself. It
  chose the full working line in the yard lane (22–33 collisions) and
  answered the collisions by slowing to 0.5 m/s instead of narrowing the
  formation.
  The observation now includes `fitsCorridorAhead`: for each formation,
  whether its machines fit between the obstacles on the route over the next
  60 m with 3 m to spare on each side. The prompt adds one sentence: speed
  does not change how wide a formation is.

Both are facts the hand-written rule and the search already had in usable
form; the change gives the model the same facts in words. The study is then
rerun from scratch (pilot 2, then the full run); no pilot-1 replies are
reused. Outputs carry the prompt version in their names (`-p2`,
`PROMPT_VERSION` in `shepherd/farmsupervisor.py`), so an older prompt's
results can never be resumed into a newer run.

Pilot 2 (9 Oct, same model and design, prompt version 2). Plumbing was
good: 0–3% unusable replies, about 2 s per call, 15 min of compute for the
8 episodes. Both pilot-1 fixes worked (zero-shot, wet, layout 0: 33 → 2
collisions; north fence, layout 0: 756 → 0 north-strip machine-seconds).
Three problems remained, in what the model is shown and how it is asked.
Prompt version 3:

- **"Fits" was wrong in two ways, and the model trusted it.** The fit check
  looked only at the route ahead of the leader. As soon as the leader was
  through the gate it reported that the half line fitted, while the single
  file's tail was still in the lane; the followers then cut across to their
  new places through the gate posts and the fence (7–8 collisions in 7 of
  the 8 episodes, all follower–obstacle). It also reported a half line as
  fitting on the headland, where body formations swing out in the U-turn
  (east-fence collisions). `formationsThatFit` now holds a formation as
  fitting only when (a) every follower's straight path to its new place
  clears every obstacle, (b) off the passes, the machines beside the
  leader's track fit between the obstacles over the next 60 m with 4 m to
  spare (sampled every 5 m; the leader's own track is not checked, because
  no formation choice changes it), and (c) for body formations, the route
  does not turn by more than 30° within 60 m. It is shown at every call,
  on the passes too. Two decision points go with it: the formation being
  held no longer fitting (two checks running), and a wider formation having
  fitted for 6 s (once per formation per phase, at least 10 s after the
  previous call). The search uses the same decision points. The rule needs
  neither: it encodes the same facts as its schedule (single file off the
  passes, wide only from 20 m into a pass to 35 m before its end).
- **Offset sign on the westbound pass.** With "on this pass a positive
  offset moves the line south", the model still reasoned "keep back from
  the north fence, so a negative offset" on pass 2 in 3 of the 4
  north-fence episodes (zero-shot layout 1: 413 north-strip
  machine-seconds). Offsets are now in fixed compass terms on
  every pass (+ north, − south, since all passes run east–west), in the
  reply, the observation and the what-if predictions. The pass observation
  also gives its heading and which compass side is left of travel, for the
  lanes and the lineLeft/lineRight formations, and every hazard now carries
  its x, y position.
- **What-if never used the simulator.** It was described as optional and
  the reply format at the end of the prompt asked for a plain choice; the
  model tested options at 1 of 160 decisions, so the condition was
  zero-shot under another name. Now every what-if decision goes through
  the simulator: the model proposes 1–4 options (a plain choice is tested
  as the only option), sees the predictions, and makes the final choice.

The comparators are rerun with the new decision points (the search uses
them; the other three do not).
