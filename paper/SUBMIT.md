# Submitting the paper

Nothing has been submitted anywhere. These steps need Abhay's accounts.

## Build

```sh
cd paper
tectonic -X compile benchmeasure.tex     # produces benchmeasure.pdf (12 pages)
```

## arXiv

1. Category: `cs.LG` primary, cross-list `cs.SE` and `stat.ME`. First-time submitters in cs.LG
   may need an endorsement; arXiv's submission form says so if it applies.
2. Upload a tarball of `benchmeasure.tex`, `refs.bib`, the generated `benchmeasure.bbl`
   (run `tectonic -X compile --keep-intermediates benchmeasure.tex` to keep it), and the two
   figures from `../figures/` (`fig_reliability_curve.png`, `fig_multilingual.png`). arXiv
   compiles with pdflatex; `\graphicspath` already looks in `figures/`, so put the PNGs in a
   `figures/` folder inside the tarball.
3. Title: Benchmarks Are Measurement Instruments: Reliability, Invariance and Validity Ceilings
   for LLM and Agent Evaluation. Abstract: copy from the `abstract` environment and strip the
   LaTeX (`\%` to `%`).
4. License: CC BY 4.0 suits a methods paper with open code.
5. Comments field: "12 pages, 2 figures. Code and data: https://github.com/abhaymettu/benchmeasure".

## Before submitting, check

- `refs.bib` entry `chowdhury2024verified` (OpenAI's SWE-bench Verified post) was not fetched
  first hand (the page returned 403); its author list comes from another paper's reference list.
  Open https://openai.com/index/introducing-swe-bench-verified/ and confirm or fix the authors.
- Several cited 2026 papers are recent arXiv preprints; check whether any now has a venue.
- Rerun `uv run python experiments/swebench.py` and `uv run pytest`; the numbers in the paper
  are the ones in `results/swebench.json`.

## Venues worth considering

Workshops on evaluation or measurement at NeurIPS or ICML, or a datasets and benchmarks track.
Check each call for page limits; the body is about 10 pages before references.
