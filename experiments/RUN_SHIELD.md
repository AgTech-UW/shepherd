# Safety-filter run (Linux laptop)

The safety filter makes the fit check binding. If a decider picks a formation
that does not fit the space ahead (followers would cut across trees, gate posts
or the fence to reach their new slots), it is not taken up. The crew keeps its
current formation, or the widest one that fits, and the choice is applied as
soon as it fits. Every hold and release is logged in the transcript under
`"shield"`. The prompt is unchanged (v3), and the reflex is on.

From the repository folder, with Ollama running and `qwen2.5:7b` and
`nomic-embed-text` pulled:

```bash
# 1. the AI on its own: 24 runs, about 1.5-2 hours
python experiments/farm_study.py --followers reflex --shield --conditions zeroshot

# 2. the baselines with the same filter (deterministic, about an hour with 4 workers)
python experiments/farm_comparators.py --followers reflex --shield --workers 4

# 3. tables and figures
python experiments/farm_study.py --followers reflex --shield --conditions zeroshot --report-only

# 4. zip for sending, without the bulky trajectory files
zip -r ~/Downloads/shield_results.zip results/farm_study_reflex_shield -x "*/traj/*"
```

For all four AI versions overnight, leave out `--conditions zeroshot` in steps
1 and 3. Lessons and memory then rerun their six practice days first.
Everything is resumable, so if it stops, run the same command again.

Output goes to `results/farm_study_reflex_shield/`, separate from the main
study (`farm_study_vfield`) and the reflex study (`farm_study_reflex`).
