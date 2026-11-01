# Launch post drafts (for Abhay to send himself; nothing has been posted)

## X / short

benchmarks are measurement instruments, so i ran the three checks psychometrics runs on every test.

swe-bench verified: reliability 0.994 across the leaderboard, about 0.68 for the top 20. none of the 19 neighbouring top-20 pairs can be told apart.

`pip install git+https://github.com/abhaymettu/benchmeasure && benchmeasure demo`

## Longer (LinkedIn or a thread)

I come from psychometrics, and every leaderboard skips the three questions that field asks first. Does the score replicate? Does a task mean the same thing for every model? How much can the score predict?

I ran all three on public item-level data:

- SWE-bench Verified ranks its whole leaderboard almost perfectly (reliability 0.994) and its top 20 at about 0.68. A paired test needs a 4.3 point gap to see a difference there, and the top 20 span 4.8 points.
- The usual way of computing reliability for the leaders, on the same tasks that picked them, says 0.19 for the top 10. Picking on half the tasks and measuring on the other half says about 0.69. That bias is easy to fall into.
- Twelve models run on both SWE-bench Verified and SWE-bench Multilingual correlate 0.98 once each score's noise is removed. My own earlier analysis had assumed the benchmark covered a quarter of what matters. For this kind of transfer, noise is the bigger problem.
- When the grader is GPT-4, it lands 4.8 points under a fair human ceiling on the same items in one order and 1.1 under when it judges both orders, and it flips on swapped answers as often as it disagrees with itself on reruns.

The tool runs all of this on any results file in one command. Paper, code and data: https://github.com/abhaymettu/benchmeasure
