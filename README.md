# shepherd

A language model supervising a turn-limited robot flock: formation control
through corridors and gaps, and a farm field operation driven by an
operator's instructions in plain words.

It builds on the vector-field guidance, the boid followers and the spacing
condition in [AgTech-UW/grainframe_captainFalcon](https://github.com/AgTech-UW/grainframe_captainFalcon).
`grainframe/` here is an unmodified copy of that package at commit
`dcfcaf3`, so this repository runs on its own. `shepherd/` only imports
from it.

## Setup

```bash
pip install -r requirements.txt
# for the language-model runs: Ollama running locally, then
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

## The farm study

Design, hypotheses and every change made after the pilot:
[`experiments/FARM_STUDY.md`](experiments/FARM_STUDY.md).

```bash
python experiments/farm_study.py --pilot                  # ~20 min: plumbing, parse rate, timing
python experiments/farm_study.py                          # the run, ~4 h on an RTX 5050 laptop; resumable
python experiments/farm_comparators.py --workers 4        # rule, schedule, single file, oracle search; ~1 h, CPU only
python experiments/farm_study.py --report-only            # tables and figures from whatever has run
python experiments/farm_study.py --mock --seeds 0-1 --train-episodes 2   # whole pipeline, no model needed
```

Results go to `results/farm_study_vfield/`. Output names carry the model and
the prompt version, so results from an older prompt are never resumed into a
newer run.

## Follow-up: obstacle reflex

`--followers reflex` gives every machine a last-resort swerve around rocks
and poles (see the end of `experiments/FARM_STUDY.md`).

```bash
python experiments/farm_study.py --followers reflex --conditions zeroshot,whatif
python experiments/farm_comparators.py --followers reflex --workers 4
python experiments/reflex_comparators.py results/shepherd/reflex_vs_vfield.csv
```

## Animations

```bash
python -m shepherd.animate_farm TRANSCRIPT.json out.gif           # any LLM episode, replayed exactly from its transcript
python -m shepherd.animate_compare followers out.gif              # Part I followers vs vector-field followers
python -m shepherd.animate_compare reflex out.gif                 # single file past a rock, without and with the reflex
python -m shepherd.animate_story formation out.gif --cache c/     # AI folds for the gate, opens up in the field (replayed run)
python -m shepherd.animate_story wetspot out.gif --cache c/       # wet spot: fixed plan ploughs through vs AI told "it's wet"
python -m shepherd.animate_story cattle out.gif --cache c/        # cattle day: hand-written rule vs AI with practice notes
```

In every animation a machine inside a wet spot is drawn brown (it is slowed to
`wetSpeed`, 0.6 m/s) and the ground it churns stays marked. `animate_story`
draws the formation in force in large type and the AI's own stated reason
under the title; its presets name the transcripts they replay.
Rendered copies are in `results/shepherd/figures/`.

## Other experiments

```bash
python -m shepherd.compare_followers        # paper boids vs damped boids vs vector-field slot following
python -m shepherd.demo_gap vfield          # gap in a wall: vee -> single file -> vee (hand-written width rule, no AI)
python experiments/llm_study.py --mock --seeds 0-1 --train-episodes 2   # the earlier corridor study
python -m shepherd.run --agent baseline --course all --seeds 0-4        # agent-led flock on the test courses
```

## Layout

| path | what |
|---|---|
| `shepherd/env.py` | the simulator: Dubins leader, followers, obstacles, collisions |
| `shepherd/followers.py` | follower models; vector-field slot following is the default; the obstacle reflex |
| `shepherd/replay.py` | re-run a recorded LLM episode from its transcript (exact) |
| `shepherd/animate_farm.py`, `animate_compare.py` | GIFs |
| `shepherd/formation.py` | formation slots, track-frame slots, slot assignment, corridor ray casts |
| `shepherd/farm.py`, `farmenv.py` | the farm world, instructions and their costs |
| `shepherd/farmagents.py` | non-LLM deciders: single file, schedule, hand-written rule, oracle search |
| `shepherd/farmsupervisor.py` | the LLM supervisor and its prompt |
| `shepherd/memory.py` | generative-agents memory stream (Park et al. 2023) |
| `shepherd/README.md` | the simulator and agents in detail |
| `experiments/` | the study scripts |
| `farms/default.json` | the farm layout |
| `results/` | outputs |
