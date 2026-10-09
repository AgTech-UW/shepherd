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
terminal path drive the leader, and the followers are the paper's boids
with the published parameters.

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
- **LLM + what-if**: it may propose up to 4 options. The simulator reports what each would do over 40 s (collisions, wet-ground machine-seconds, coverage gained, closest machine to each fence), and the model then chooses. The simulator does not choose.
- **LLM + lessons**: 6 training days (alternating wet and rain, never the held-out instruction). After each day it rewrites at most 6 lessons. The lessons are frozen for test.
- **LLM + memory stream**: generative-agents memory (Park et al. 2023), trained on the same 6 days. Outcome records are written after every decision, importance comes from the outcome, reflections are made when importance accumulates, and recency + importance + relevance retrieval puts the top 6 into each prompt. The memory is frozen for test.

Every decider sees the same perception: hazard positions, which of its
machine lanes would touch each hazard, corridor widths, the formation's
dimensions, and the machines' extent.

Decision points (LLM and search): the start, each new phase, a hazard
coming into range, the corridor ahead changing width (off the field), a
collision (at most every 8 s), and otherwise every 30 s.

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

## Changes after the pilot

(none yet)
