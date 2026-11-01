# RUNLOG

## 2026-11-01, measurement-science lane

- Read benchdif, agent-validity-ceiling (AVC) and judge-reliability. AVC was the thin arm: its
  between-model variances were read off the leaderboard by eye and its coverage values were assumed.
- Built item response matrices from github.com/SWE-bench/experiments (commit in
  `src/benchmeasure/data/SOURCE`): 173 Verified submissions x 500 tasks, 14 Multilingual x 300.
  Two Verified submissions were dropped because their per-task files contradict their reported
  score (10.2 vs 52.8, and 0.0 vs 69.6).
- Wrote the library: G-study (p x i, p x i x r), exact McNemar pairs, validity ceiling and coverage
  estimate, invariance checks (excess gap variance, held-out subset swing, random subset stability,
  logistic DIF through benchdif). 14 tests.
- Finding that changed the plan: estimating the top k's reliability on the same tasks used to pick
  the top k is biased low (top 10: 0.19). Split-task estimate (pick on half, score on the other half,
  50 splits): top 10 G = 0.70, top 20 G = 0.69, whole board 0.994.
- Coverage measured, not assumed: Verified vs Multilingual on 12 matched mini-SWE-agent v2 models,
  true-score r = 0.978, C = 0.956 (95% CI 0.63 to 1.00). AVC assumed C = 0.25.
- Literature search (orx, ~11 queries): close 2026 prior work exists for each arm (Srinivasan,
  Hardy et al., Liu et al., Zheng and Yang). Notes in `paper/literature-notes.md`.
- benchdif: fast-forwarded harness-dif into main (46 tests pass), rebased onto one remote README
  commit, pushed.

Next: paper, READMEs across the three repos, site files, fresh-clone check, publish.

## 2026-11-01, later in the run

- Paper drafted (12 pages, tectonic). An adversarial review found real problems; all fixed in code,
  not just wording: split-task reliability now reports a halves cross-check (top 10: 0.69 split,
  0.61 halves, 0.19 naive); the DIF gap-variance null is computed within score bands (1.39x vs
  1.29x raw, so ability was not inflating it); the subset swing is reported as the one-sided shift
  (4.7 pp) and the adversarial range (8.2 pp), with base models kept on one side of each split;
  the ceiling is shown to equal the observed R^2 within a population and is used for attribution;
  coverage is reported with and without the weakest model (0.96 vs 0.80); SWE-bench Verified Mini
  is compared with random 50-task subsets (it is typical, 0.36 vs median 0.42).
- judge-reliability: the judge is now scored on the ceiling's own 249 comparisons
  (one order 83.4 vs ceiling 88.2, gap -4.8, CI -9.6 to +0.1; swap averaged -1.1).
- benchdif: headline numbers rerun from fresh data; flagged-item range corrected to 154 to 777.
- All four repos: gitleaks clean (allowlists for benchmark task names and prompt-hash cache keys),
  no em dashes in tracked files, READMEs at 100/100 on inmyvoice where checked.
- Published github.com/abhaymettu/benchmeasure (public); pushed benchdif, judge-reliability,
  agent-validity-ceiling.

### Fresh-clone proof

```
$ uv venv -p 3.12 venv && uv pip install git+https://github.com/abhaymettu/benchmeasure
$ venv/bin/benchmeasure demo
173 models x 500 tasks
1. Reliability (generalizability theory, models x tasks)
   whole board   G = 0.994   model SD 19.0 pp   score SEM 1.5 pp
   top 10        G = 0.690   model SD 1.6 pp   tasks for G = 0.9: 2,023
   top 20        G = 0.682   model SD 1.6 pp   tasks for G = 0.9: 2,096
   leaderboard neighbours in the top 20: 0 of 19 separable by exact McNemar at 0.05
2. Invariance: banded gap variance 1.39x; one 50-task subset moves the gap up to 4.7 pp
3. Validity ceiling: true-score r = 0.978; coverage C = 0.956
$ git clone https://github.com/abhaymettu/benchmeasure clone && cd clone && uv run pytest -q
15 passed in 27.01s
```

- Blind README judge (staff-engineer persona) scored 6 to 8 of 10; applied its five edits (hero leads with the winner's-curse result, G defined first, circularity moved to Limits, consistent coverage numbers, figure label fix).
