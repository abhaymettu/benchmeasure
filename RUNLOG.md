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
