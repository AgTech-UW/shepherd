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

Pilot 3 (9 Oct, prompt version 3, commit 4b25994). 0 unusable replies in
8 episodes; 20 min of compute (zero-shot about 1.5 min an episode, what-if
about 3.5 min, now that every what-if decision is simulated). Collisions
fell from 98 to 27 over the same 8 episodes, 20 of them in one zero-shot
episode in which the model chose line formations at the gate exit although
`formationsThatFit` listed only the columns. That and the remaining
north-strip time (what-if, layout 1: a full line on pass 2 with no offset)
are the model's decisions, not the interface's, so the pilot passed and the
design is frozen at prompt version 3 for the full run.


## Results (main study, run 9–10 Oct 2026)

qwen2.5:7b (Ollama id 845dbda0ea48), prompt version 3, on an RTX 5050 laptop
GPU. 96 test episodes (8 layouts × 3 instructions × 4 conditions) plus 12
training episodes; 96 comparator episodes. Unusable replies 0.1–1.2%; 1.6–2.4 s
per call; 1.6–3.9 min of compute per episode. Files:
`results/farm_study_vfield/*qwen2-5_7b-p3*`, comparators in
`results/farm_study_vfield/comparators/`, analysis below by paired bootstrap
over the 8 layouts (95% intervals; with n = 8 these are estimates, not tests).

| instruction | decider | cost | collisions | worked | wet m-s | north m-s |
|---|---|---|---|---|---|---|
| wet | hand-written rule | 46 | 1.6 | 56% | 38 | 178 |
| wet | oracle search | −33 | 3.0 | 70% | 0 | 202 |
| wet | LLM zero-shot | 150 | 2.5 | 61% | 89 | 111 |
| wet | LLM + what-if | 123 | 2.4 | 61% | 75 | 223 |
| wet | LLM + lessons | 184 | 0.6 | 60% | 114 | 175 |
| wet | LLM + memory | 296 | 8.6 | 65% | 131 | 206 |
| rain | hand-written rule | −119 | 1.2 | 61% | 107 | 127 |
| rain | oracle search | −143 | 3.2 | 73% | 74 | 237 |
| rain | LLM zero-shot | −76 | 4.9 | 63% | 152 | 84 |
| rain | LLM + what-if | −104 | 1.8 | 59% | 104 | 109 |
| rain | LLM + lessons | −75 | 5.4 | 63% | 110 | 211 |
| rain | LLM + memory | −71 | 5.8 | 68% | 102 | 279 |
| north (held out) | hand-written rule | 138 | 1.6 | 56% | 38 | 178 |
| north (held out) | oracle search | −53 | 3.1 | 64% | 8 | 0 |
| north (held out) | LLM zero-shot | 56 | 2.2 | 58% | 198 | 10 |
| north (held out) | LLM + what-if | 33 | 2.1 | 54% | 118 | 24 |
| north (held out) | LLM + lessons | −9 | 1.8 | 52% | 82 | 0 |
| north (held out) | LLM + memory | 110 | 5.6 | 60% | 96 | 83 |

(Single file and the hazard-blind schedule are in `summary_qwen2-5_7b-p3.csv`.)

- **H1 Safety: not supported as stated.** On the trained instructions the
  zero-shot model averaged 3.7 collisions per episode against the rule's 1.4
  (+2.2 [+0.4, +4.2]). The median episode is the same (1 collision, as for
  the rule, what-if and lessons); the excess comes from a minority of bad
  episodes (zero-shot: 6 of 24 with 5 or more; rule: 0 of 24). About three
  quarters of LLM collisions were followers crossing the lane trees, the gate
  or the fence when the model widened the formation while the tail was still
  in the lane, which `formationsThatFit` advised against; every condition
  chose a formation listed as not fitting in 10–14% of decisions. What-if was
  closest to the rule (2.1, +0.6 [0.0, +1.4]).
- **H2 Instruction following (wet vs rain): not supported.** The rule and the
  search separate the two days clearly (wet ground −69 [−117, −18] and −74;
  work +4.8 [+2.4, +7.4] and +2.7 points). Zero-shot points the right way but
  both intervals include zero (−63 [−141, +14]; +1.9 [−4.4, +7.7]); what-if is
  right on wet ground and wrong on work; lessons shows no difference; memory
  drove more on wet ground on wet days (+29 [+6, +61]).
- **H3 Held-out instruction: supported for zero-shot, lessons and memory.**
  North-strip exposure, north-fence day minus wet day: zero-shot −101
  [−159, −41], lessons −175 [−255, −93], memory −124 [−201, −43]; what-if
  −199 [−541, +27], right direction with an interval that includes zero
  because of one wet-day outlier (1300 machine-seconds). On the north-fence day
  zero-shot spent 0 in the strip on 5 of 8 layouts (mean 10), lessons 0 on all
  8, what-if 0 on 6 of 8 (mean 24), memory 83; the rule, as predicted, spent
  exactly what it spent on its wet day (178).
- **H4 Simulator access: not supported at n = 8.** What-if minus zero-shot
  cost was negative on all three instructions (wet −27 [−101, +39], rain −29
  [−76, +11], north −23 [−80, +40]), with intervals that include zero. The
  model sent 1.1 options per decision on average: it used the simulator to
  check its own choice, not to compare alternatives.
- **H5 Experience: not supported.** Lessons: wet +35 [−78, +163], rain 0, north
  −65 [−144, −4] (better, on the instruction it never trained on). Memory:
  wet +146 [+17, +289] (worse), rain +5, north +54 [−48, +160], and 6.7
  collisions per episode (median 6). Its reflections were over-general and
  sometimes counterproductive ("Keep the offset at 0 m to avoid unnecessary
  deviations", "Choose a formation that maximizes working width"). Lessons vs
  memory (exploratory): lessons lower cost on all three instructions.
- **H6 Ceiling: supported.** Oracle search had the lowest cost on every
  instruction (minus the best other decider: wet −79 [−173, −5], rain −16
  [−24, −10], north −44 [−68, −22]). Its 3 collisions per episode all happen
  in the yard lane in the first minute, identically on every layout: at the
  start all options tie over the 40 s horizon, the first in the list (vee)
  is taken, and its cost (followers clipping the shelterbelt at a bend) falls
  outside the horizon. A greedy receding horizon is myopic; reported as is.

**Rule vs language model.** Where the rule was written for the instruction,
it beats every LLM condition (zero-shot minus rule: wet +104 [+25, +177], rain
+43 [+2, +97]). On the held-out instruction the LLMs beat it (what-if −105
[−164, −55], lessons −147 [−227, −76], zero-shot −83 [−165, +1]) and close
part of the gap between the rule and the oracle: zero-shot 43%, what-if 55%,
lessons 77%, memory 15%.

**Reading.** A small local model, told in plain words, followed an instruction
nobody programmed for, and in a typical episode was as safe as the
hand-written rule. It did not reliably turn the two trained instructions into
different behaviour, its failures are rare but large (ignoring a stated fact
about what fits), and neither a simulator it was not made to compare options
with, nor experience distilled into lessons or a memory stream, reliably
improved it. The obvious complements are a rule or reflex layer for the
known cases and the failure modes, with the language model for the cases no
rule covers.

## Follow-up: an obstacle reflex (exploratory, added after the main run started)

Not part of the frozen study above; reported separately as exploratory.

The machines do not avoid anything on their own: every brain avoids hazards
only through formation, speed and offset. The follow-up adds a reflex layer
underneath (follower model `reflex`, `shepherd/followers.py`), in the spirit
of subsumption (Brooks 1986): a machine about to run into a free-standing
obstacle (a rock or a pole) within 10 m bends the heading its vector field
asks for just enough to clear it, and slows in proportion; once past, the
vector field returns it to its line. The leader has it too. Fences and tree
lines are excluded on purpose: in testing, a reflex that would not let a
follower cross a fence trapped it on the wrong side whenever the formation
put its slot there (it scraped along outside the field for the rest of the
episode). Boundaries stay the planner's job.

Hand-written deciders on the 8 test layouts (`experiments/reflex_comparators.py`,
`results/shepherd/reflex_vs_vfield.csv`), mean per run:

| decider | collisions, vector field | collisions, + reflex | reflex swerves | coverage | weaving (°/s) |
|---|---|---|---|---|---|
| single file | 2.25 | 0.00 | 0.2 | 14.8% → 14.8% | 1.41 → 1.44 |
| schedule (hazard-blind) | 2.12 | 1.12 | 1.4 | 67.6% → 67.8% | 1.63 → 1.73 |
| hand-written rule | 1.62 | 1.12 | 0.5 | 56.1% → 56.1% | 2.27 → 2.28 |

Every collision with a rock or a pole is gone, at no cost in coverage or
smoothness; what remains (about one per run) is fence and gate scrapes when
a plan widens the formation too early. The LLM conditions are rerun with
`--followers reflex` (zero-shot and what-if), and the prompt says the reflex
exists but to plan as if it did not. The comparison of interest is how
often the reflex has to save each brain.
