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
  quarters of all LLM collisions (from the collisions reported in the
  transcripts) were followers crossing the lane trees, the gate or the fence
  when the model widened the formation while the tail was still in the lane,
  which `formationsThatFit` advised against; for what-if, half were the rock
  pile instead. The conditions chose a formation listed as not fitting in
  9.6–14.3% of decisions. What-if was
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

### Reflex rerun of all four AI conditions (10 Oct, Windows PC)

The whole LLM study was rerun with `--followers reflex`: all four conditions,
lessons and memory retrained from scratch on the same six practice days, 96
test runs, the same prompt (v3) and model (`qwen2.5:7b`), on a Windows PC with
an RTX 3070 (3.4–4.1 s per call; the Ollama version there was not recorded).
The comparators were rerun on that machine too and match the laptop results
above exactly. The run stopped once at the comparators (an accidental Ctrl+C)
and was resumed; no finished job was repeated. Results are in
`results/farm_study_reflex/`, the breakdown below from
`python experiments/reflex_llm_compare.py`.

Collisions are classified by where they happened, from the positions the
transcripts report (240 of 244 reflex-run collisions; 4 fell after the last
decision): within 6 m of a rock pile or pole, in the yard lane or gate,
against the field fence, or between machines. Mean per run, without → with
the reflex; in brackets the paired difference by layout with its 95%
bootstrap interval:

| condition | all collisions | rock or pole | lane or gate | field fence | runs with none |
|---|---|---|---|---|---|
| zero-shot | 3.21 → 0.71 (−2.50 [−4.42, −1.12]) | 0.75 → 0.21 | 1.67 → 0.50 | 0.79 → 0.00 | 7 → 15 of 24 |
| what-if | 2.08 → 2.75 (+0.67 [−2.29, +3.83]) | 1.21 → 0.21 | 0.50 → 2.38 | 0.38 → 0.12 | 4 → 12 of 24 |
| lessons | 2.58 → 2.38 (−0.21 [−2.21, +1.54]) | 0.88 → 0.12 | 1.46 → 1.92 | 0.25 → 0.29 | 7 → 14 of 24 |
| memory | 6.67 → 4.33 (−2.33 [−4.29, −0.25]) | 1.04 → 0.12 | 5.58 → 3.83 | 0.04 → 0.25 | 0 → 9 of 24 |
| all four | 3.64 → 2.54 (−1.09 [−2.56, +0.41]) | 0.97 → 0.17 (−0.80 [−1.06, −0.53]) | 2.30 → 2.16 (−0.15 [−1.45, +1.21]) | 0.36 → 0.17 | 18 → 50 of 96 |

- The reflex does what it was built for. Rock and pole collisions fell by
  82% overall and in every condition (72–88%).
- It does nothing for the lane and the gate, by design, and these are now
  about 85% of what is left. The lane and gate counts move up for what-if and
  down for zero-shot and memory. These are rerun-to-rerun swings of a
  stochastic model, not an effect of the reflex. The zero-shot drop in total
  collisions is therefore part reflex (rocks and poles) and part a luckier
  rerun (lane, gate, fence).
- More than half of all AI runs (50 of 96) had no collisions, against 18 of
  96 without the reflex. The median per run is 0 for zero-shot and lessons.
- Cattle day: the AI on its own stayed out of the north strip on 7 of 8
  fields (2 machine-seconds on the eighth); with practice notes on all 8, as
  before. What-if got worse (135 machine-seconds on average, one bad field).
- Instruction following is unchanged: no condition spends less time on wet
  ground on wet days than on rain days (zero-shot +8 [−41, 64] machine-s),
  so H2 still fails with the reflex.

Next: a safety filter that makes the fit check binding (a formation that does
not fit the space ahead is not applied until it does). That targets the lane
and gate collisions directly. It also covers the oracle search, whose three
collisions per run are all in the lane at the start, with or without the reflex.

## Follow-up 2: safety filter and leader wait (exploratory, baselines only, 10 Oct)

Two further layers, tested so far on the deterministic deciders with the
reflex on (8 test layouts, wet-day instruction). To stand in for the AI's main
failure, a test decider widens too early on purpose: two abreast in the lane,
full line from the first metre of each pass with the tail still in the gate
(`experiments/eager_decider_test.py`).

- **Safety filter** (`--shield`, `farmagents.shieldSpec`): the fit check is
  binding. A formation that fails it is not taken up. The current one is kept
  if it still fits, otherwise the widest that fits, and the held-back choice
  is applied as soon as it fits. The search only considers fitting formations.
- **Leader wait, "pearls on a string"** (A. Rogers; follower model `pearls`):
  the leader's speed is scaled by clip(1 − (ℓ − 8)/(25 − 8), 0.3, 1), where
  ℓ is how far the furthest-behind follower trails its slot, and a follower
  making up ground may not exceed the leader's speed + 0.5 m/s.

Collisions, field worked, time to finish, mean per run (`python
experiments/layer_table.py`; any cell can be regenerated with
`experiments/layer_tests.py`):

| decider | reflex only | + safety filter | + leader waits | + both |
|---|---|---|---|---|
| hand-written rule | 1.12, 56%, 877 s | 0.38, 56%, 877 s | 0.25, 57%, 1043 s | 0.38, 56%, 1034 s |
| schedule | 1.12, 68%, 868 s | 0.25, 68%, 868 s | 0.12, 68%, 983 s | 0.12, 68%, 982 s |
| widens too early (test) | 5.12, 73%, 869 s | 1.12, 73%, 868 s | 9.00, 73%, 976 s | 2.12, 73%, 960 s |

*Corrected 11 Oct.* The filter-on cells first reported here (rule 0.25, 49%;
schedule 0.50, 60%; widens too early 0.50, 65%) came from an earlier, stricter
draft of the filter, not the committed one: it held back far more choices (206
per run for the test decider, against 118 now). Rerun with the committed filter (the one
the AI run used), the filter costs no ground.

- The filter cuts the test decider from 5.12 to 1.12 collisions, with no loss
  of ground. The one that remains is a gate-post contact as the crew comes back
  through the gate at the very end of the run.
- Leader wait removes most remaining collisions of the sensible deciders with
  no loss of ground and 13–19% more time. Followers racing to catch up
  (faster than leader + 0.5 m/s) drop from 10–12% of the time to 1–2%. Most of
  the waiting is while the crew forms up at the start of each pass. In the
  slough the followers fall only a few metres behind, under the 8 m threshold.
- Leader wait does not rescue a bad plan. With early widening, the leader
  waits while followers press against the trees, and collisions rise.
  The filter stops a bad choice. The wait smooths a good one. Together they do
  no better than the better of the two for the schedule and the rule.

Next: the AI with the reflex and the filter (`experiments/RUN_SHIELD.md`).

## Follow-up 3: the AI with the reflex and the safety filter (11 Oct, Windows PC)

All four AI conditions rerun with the reflex and the safety filter on, lessons
and memory relearned from scratch on the same practice days, the same prompt
(v3) and model (`qwen2.5:7b`, id 845dbda0ea48), 96 test runs, on the same
desktop as the reflex rerun (RTX 3070 8 GB, Ollama 0.40.2). The model is not
told about the filter. Baselines rerun with the filter too. Results in
`results/farm_study_reflex_shield/`; comparison with the reflex rerun from
`python experiments/shield_compare.py` (writes `shield_vs_reflex_runs.csv`).
The first 44 test runs waited about 2 s per request for `localhost` to fall back
from IPv6 to IPv4 (fixed in 84c8513 by connecting to 127.0.0.1); this changes
run time only.

Language model, reflex only -> reflex + filter, per run:

| condition | collisions | lane/gate | rock/pole | runs with 3+ | runs with none | field worked |
|---|---|---|---|---|---|---|
| zero-shot | 0.71 -> 0.67 | 0.50 -> 0.25 | 0.21 -> 0.33 | 2 -> 1 | 15 -> 15 of 24 | 56.3 -> 57.2% |
| what-if | 2.75 -> 0.54 | 2.38 -> 0.21 | 0.21 -> 0.33 | 5 -> 1 | 12 -> 13 | 58.1 -> 58.5% |
| lessons | 2.38 -> 0.75 | 1.92 -> 0.33 | 0.12 -> 0.42 | 4 -> 0 | 14 -> 7 | 57.1 -> 57.6% |
| memory | 4.33 -> 0.62 | 3.83 -> 0.33 | 0.12 -> 0.29 | 10 -> 1 | 9 -> 13 | 62.5 -> 62.9% |
| all four | 2.54 -> 0.65 | 2.16 -> 0.28 | 0.17 -> 0.34 | 21 -> 3 of 96 | 50 -> 48 of 96 | 58.5 -> 59.0% |

- Paired by layout: all collisions −1.90 [−3.25, −0.73]; lane/gate −1.88
  [−3.22, −0.72]; field fence −0.15 [−0.27, −0.04]; field worked +0.55
  [−0.88, +1.97] points. Worst run 24 -> 3 collisions.
- The model still chose a formation listed as not fitting in 8.2% of its
  decisions (8.2% in the reflex run). The filter held a choice back 421 times
  in 96 runs (4.4 per run); 118 were applied later, the rest were superseded.
- **Side effect: rock and pole collisions rose**, 0.17 -> 0.34 per run, +0.18
  [+0.05, +0.32]. 26 of the 33 came within 40 s after the filter held a
  choice back. Condition (a) of the fit check (every follower's straight path
  to its new slot is clear) fails next to a rock, so the filter freezes the
  formation exactly when the model is changing it to get past. Fix for the
  next version: apply condition (a) only off the passes, or always let a
  change through that narrows the crew.
- North-fence day, layouts with no time in the strip: zero-shot 8/8, lessons
  8/8, what-if 6/8, memory 0/8 (reflex run: 7, 8, 7, 5). North minus wet-day
  strip: zero-shot −76 [−146, −18], lessons −136 [−208, −65], what-if −38
  [−90, +12], memory −54 [−237, +90].
- Wet against rain is still not separated by any model condition (e.g.
  lessons wet-ground −29 [−62, +1], what-if −59 [−142, 0]); the rule and the
  oracle separate them clearly.

Baselines, reflex only -> reflex + filter, all three instructions:

| decider | collisions | field worked |
|---|---|---|
| single file | 0.00 -> 0.00 | 14.8 -> 14.8% |
| schedule | 1.12 -> 0.25 | 67.9 -> 67.7% |
| hand-written rule | 1.12 -> 0.33 | 57.7 -> 57.0% |
| oracle search | 3.04 -> 0.38 | 70.4 -> 64.6% |

The oracle's first-minute lane collisions are gone and its cost improves on
every instruction (wet −52, rain −158, north −73).

Model against rule, both with the filter, paired by layout: all four conditions
0.65 against 0.33 collisions per run, +0.31 [−0.06, +0.64]; zero-shot on the
trained instructions 0.44 against 0.31, +0.13 [−0.19, +0.44]. Cost: rain-day
zero-shot ties the rule (−0.3 [−24, +27]); north-fence zero-shot −100 [−160,
−41] and lessons −122 [−187, −55] better than the rule; on the wet day every
model condition is worse (+115 to +246).

Next: the same run on the laptop (second replicate), then the filter with the
rock fix and the leader wait together.

## Follow-up 4: fit check 2 and a stronger reflex (11–12 Oct, Windows PC)

The first filter run raised rock and pole collisions (0.17 -> 0.34 per run):
condition (a) of the fit check froze the formation when a rock lay between the
followers and their new slots. Two changes, each first checked by replaying the
recorded replies of the 28 first-filter-run episodes with a rock or pole
collision, with no model (`fit2_replay.py`; once a run diverges the replies were
written for another state, so this is a guide only):

| replayed under | rock/pole hits | episodes with one | lane/gate | worked |
|---|---|---|---|---|
| as recorded (fit check 1) | 33 | 28 | 4 | 58.5% |
| fit check 2 | 30 | 25 | 4 | 57.4% |
| fit check 2 + reflex 15 m / 2.5 m | 21 | 17 | 4 | 57.3% |
| fit check 2 + reflex 20 m / 3.5 m (27 episodes) | 23 of 32 | 20 | 4 | 57.2% |

- **Fit check 2** (`--fit-version 2`): the path condition counts only
  fences, gate posts and tree lines (`formation.boundaryObstacles`); rocks and
  poles are the reflex's. Also changes the `formationsThatFit` the model sees.
- **Stronger reflex** (`--reflex-look 15 --reflex-margin 2.5`; was 10 m,
  1.5 m). 20 m / 3.5 m was no better, so 15 / 2.5.

Model run with fit check 2 (normal reflex), `results/farm_study_reflex_shield_fit2/`,
compared by `python experiments/fit2_compare.py` and
`python experiments/filter_runs_compare.py`:

| | reflex only | filter (fit 1) | filter (fit 2) |
|---|---|---|---|
| collisions per run | 2.54 | 0.65 | 0.66 |
| lane/gate | 2.16 | 0.28 | 0.30 |
| rock/pole | 0.17 | 0.34 | 0.26 |
| field fence | 0.17 | 0.02 | 0.09 |
| runs with 3+ / with none (of 96) | 21 / 50 | 3 / 48 | 7 / 61 |
| field worked | 58.5% | 59.0% | 56.6% |
| zero-shot / what-if / lessons / memory | 0.71 / 2.75 / 2.38 / 4.33 | 0.67 / 0.54 / 0.75 / 0.62 | 0.58 / 0.38 / 0.38 / 1.29 |

- Fit 2 minus fit 1, paired: collisions +0.01 [−0.16, +0.19]; rock/pole −0.08
  [−0.23, +0.07]; worked −2.5 [−5.1, −0.3] points, nearly all memory (62.9 ->
  53.6%). Non-fitting choices 8.2% -> 5.3% of decisions (fewer formations are
  listed as not fitting); filter holds 4.4 -> 2.8 per run.
- Memory, relearned from scratch, was the worst condition this time (1.29 per
  run; one run with 8: it switched formation and offset at almost every
  decision and the re-forming crew struck the south fence). Its rerun-to-rerun
  variation is larger than any effect of the layers.
- North-fence day, layouts with no time in the strip: zero-shot 8, lessons 8,
  memory 7, what-if 6 (of 8).
- Baselines with fit check 2: the oracle has 0.0–0.1 collisions per run.

Final run (fit check 2 + stronger reflex): `experiments/run_final_windows.ps1`,
output `results/farm_study_reflex_shield_fit2_rL15M2.5/`.
