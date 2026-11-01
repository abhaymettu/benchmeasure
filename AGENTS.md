# AGENTS.md

## Layout
- `src/benchmeasure/`: `reliability.py` (G-study), `compare.py` (paired tests), `invariance.py`,
  `ceiling.py`, `report.py` (the one-call report), `cli.py`. Bundled matrices in `data/`.
- `experiments/swebench.py` writes `results/swebench.json` and `figures/`. Every SWE-bench number
  in README.md, the paper and `site/` comes from that JSON.
- `paper/benchmeasure.tex` builds with `tectonic -X compile paper/benchmeasure.tex`.

## Rules
- Run `uv run pytest` before committing. `tests/test_reproduce.py` regenerates the results JSON in
  a temp dir and compares bytes; if you change the experiment, regenerate and commit the JSON.
- A new estimator needs a test against an independent oracle (statsmodels, a closed form, or
  recovery of known parameters in simulation).
- Never type a number into prose that no script produces. If the JSON changes, update README.md,
  the paper and `site/` in the same commit.
- No em dashes anywhere. Plain commit messages, no attribution trailers.
- Item-level DIF lives in benchdif; add DIF methods there, not here.
