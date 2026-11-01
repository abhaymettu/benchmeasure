# Benchmarks are measurement instruments

Every leaderboard comparison assumes three things nobody checks. The score would come out the same on a fresh sample of similar tasks. A task means the same thing for every model. And the score predicts something outside the benchmark. Psychometrics calls these reliability, invariance and validity, and it has had standard tests for each for decades. I come from that field, so I ran the tests on AI benchmarks, using only public item-level data.

## The short version

On SWE-bench Verified, using the per-task results of 173 public leaderboard submissions:

- **The leaderboard as a whole is reliable. Its top is not.** The ranking's reliability (a G coefficient, the share of score variance that reflects real differences between systems) is 0.994 across all submissions. Among the top 20 it is about 0.68. None of the 19 neighbouring pairs in the top 20 can be told apart by a paired test, and the smallest gap a test can pick up reliably is 4.3 points, nearly the whole top 20's spread.
- **The usual way of checking this is biased.** If you pick the top 10 and compute their reliability on the same tasks, you get 0.19, because the leaders were partly picked for lucky noise. Picking on half the tasks and measuring on the other half gives about 0.69, and a second method that avoids the bias gives 0.61.
- **What the benchmark covers can be measured, not assumed.** Twelve models were run with the same agent scaffold on Verified and on SWE-bench Multilingual. Once each score is corrected for its own noise, the two correlate 0.98, or 0.90 if you drop the weakest model. Either way the noise in the two scores costs more predictive power than what Verified leaves out. My earlier analysis assumed the benchmark covered a quarter of what matters and blamed coverage. With measured inputs, for this kind of near transfer, the evidence points at noise.
- **Tasks mostly mean the same thing for open and closed models, until you take a subset.** The full-score gap between open- and closed-weight submissions is 1.2 points. A 50-task subset chosen on other submissions moves it by up to 4.7, about four times its size. The published 50-task SWE-bench Verified Mini keeps the overall ranking, and ranks the top 20 about as well as any random 50 tasks would, which is to say barely: no 50 tasks can order the leaders.

The same pattern is stronger on the Open LLM Leaderboard. Item-level differences between model families are 6 to 10 times what noise predicts, they cancel almost exactly in full scores, and they come back in 100-item subsets ([benchdif](https://github.com/abhaymettu/benchdif)). And when the scorer is an LLM judge, it is a rater with its own reliability. On the MT-Bench comparisons where a fair human ceiling exists (one annotator against the majority of the others, 88.2%), GPT-4 agrees with the human majority 83.4% of the time in one presentation order and 87.1% when it judges both orders. The rate at which it flips when the answers are swapped (16.1%) matches how often it disagrees with itself on identical reruns (16.2%) ([judge-reliability](https://github.com/abhaymettu/judge-reliability)).

## The tool

`benchmeasure` runs all three checks on any results file: rows are models, columns are tasks.

```sh
pip install git+https://github.com/abhaymettu/benchmeasure
benchmeasure demo                       # the SWE-bench analysis above, in about two seconds
benchmeasure report results.csv --groups meta.csv --group-col open_weights --criterion other.csv
```

It reports the reliability of the whole board and of its top, which neighbouring pairs a paired test can separate, how much a grouping's gap depends on which tasks you keep, and the validity ceiling against a second benchmark.

## What I would ask a leaderboard to report

1. The reliability of the comparison being made, with selection and measurement on different tasks.
2. Repeated runs when a claimed gain is within a few standard errors.
3. Whether any subset used in place of the full benchmark preserves group gaps, checked on held-out models.
4. For an LLM judge, agreement against a like-for-like human ceiling and against itself.
5. Measured coverage wherever the same models were run on a second benchmark.

## Limits

Each public submission has one run, so run noise is bounded with published estimates rather than measured here. Coverage rests on twelve matched models, moves with the weakest one, and measures near transfer, not usefulness in deployment, which no public data measures. Prior work has applied generalizability theory to agent benchmarks and shown that the top of SWE-bench is not separable; the paper names it and says what is new.

Paper: [benchmeasure.pdf](https://github.com/abhaymettu/benchmeasure/blob/main/paper/benchmeasure.pdf). Code and data: [github.com/abhaymettu/benchmeasure](https://github.com/abhaymettu/benchmeasure).
