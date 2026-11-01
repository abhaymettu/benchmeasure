# Literature notes: "Benchmarks are measurement instruments"

Compiled 2026-11-01. Tools: `orx` 0.2.19 (telemetry off) for discovery (`orx discover embedding|keyword`, about 11 queries) and reading (`orx paper <id>` report, `--full` for full text); arXiv export API for author lists and abstracts; Crossref and OpenAlex for DOI metadata. Raw search outputs were kept locally and are not committed. No paid backend, no spend.

"Read" levels: **full** = full text pulled with `orx paper --full` and searched for the claims below; **report** = alphaXiv's generated report plus abstract (numbers from a report are marked as such); **abstract** = arXiv or publisher abstract only; **metadata** = bibliographic record only.

## Verification status of the ids we were asked to check

| id | Exists? | Title as registered | Authors (first) |
|---|---|---|---|
| arXiv 2602.07150 | Yes (v3, 2026-02-06) | On Randomness in Agentic Evals | Bjarnason, Silva, Monperrus (KTH). ICLR 2026 Workshop on Agents in the Wild |
| arXiv 2608.00794 | Yes (v5, 2026-08-01) | Measurement Without Validity: The Compounding Reliability Problem in Agentic AI Evaluation | William Caban (single author; Red Hat) |
| arXiv 2606.19704 | Yes (v1, 2026-06-18) | Beyond Static Leaderboards: Predictive Validity for the Evaluation of LLM Agents | Dhaval C. Patel et al. (61 authors; IBM, Columbia) |
| arXiv 2501.10711 | Yes (v5, 2025-01-18) | Code Benchmarks Should Prioritize Rigor, Reliability, and Reproducibility | Jialun Cao et al. (15 authors) |

Not verified and kept out of refs.bib: the OpenAI "Introducing SWE-bench Verified" page itself (HTTP 403). It is in refs.bib as `chowdhury2024verified` with the author list taken from the SWE-smith reference list, flagged in its `note`. Drop it if you want only first-hand verified entries.

## 1. Variance, noise and reliability of LLM and agent evals

**bjarnason2026randomness** (full). 60,000 SWE-bench Verified trajectories: 3 models x 2 scaffolds x 2 temperatures x 10 independent runs of all 500 tasks. Single-run pass@1 varies by 2.2 to 6.0 pp depending on which run is picked. Table 1 (mean +/- run-to-run SD over 10 runs; min, max):

| Model | Temp | nano-agent | r2e-gym |
|---|---|---|---|
| Qwen3-32B | 0.6 | 16.4 +/- 0.7 (15.0, 17.2) | 23.9 +/- 1.4 (21.4, 26.4) |
| DeepSWE-preview | 1.0 | 31.4 +/- 1.0 (28.8, 32.4) | 34.4 +/- 1.5 (31.6, 37.0) |
| Devstral-2 | 0.2 | 63.5 +/- 1.1 (61.8, 65.0) | 34.9 +/- 1.5 (32.2, 37.0) |
| Qwen3-32B | 0.0 | 16.4 +/- 1.2 (14.4, 18.0) | 22.3 +/- 1.8 (19.8, 25.2) |
| DeepSWE-preview | 0.0 | 20.4 +/- 1.0 (18.2, 21.4) | 19.2 +/- 1.5 (17.0, 21.6) |
| Devstral-2 | 0.0 | 63.8 +/- 1.6 (60.6, 66.6) | 35.4 +/- 1.7 (32.0, 37.8) |

So run-to-run SD spans 0.7 to 1.8 pp across the 12 configurations. Their abstract says "standard deviations exceeding 1.5 percentage points even at temperature 0"; in the table, 3 of 6 temperature-0 cells exceed 1.5, one equals 1.5, and two are below (nano-agent Qwen3-32B 1.2, DeepSWE-preview 1.0), so quote the table rather than the abstract sentence. Also: pass@1 to pass@5 gap up to 24.9 pp; trajectories diverge within the first few percent of tokens. They recommend multiple runs per task and power analysis. **Relation:** this gives us an external estimate of the within-system (occasion) error that our G-study on 182 single-run submissions cannot separate from the model x item residual. Bjarnason et al. measure run variance but do not decompose model, item and interaction variance, and do not connect it to a validity ceiling.

**miller2024errorbars** (abstract). Treats eval questions as draws from a super-population; gives formulas for standard errors, paired differences between two models, clustered questions and power. **Relation:** the standard-error baseline our reliability framing extends; Miller does not do variance components or validity ceilings.

**madaan2024variance** (abstract). Measures seed variance across initialisations and monotonicity during training on many models; finds that reframing choice tasks as completion reduces variance for about 7B models, while item analysis and IRT "struggle to meaningfully reduce variance". **Relation:** training-seed noise, a different facet from our item and model facets.

**heineman2025signal** (abstract). Defines signal (spread between models) and noise (variability across training steps); higher signal-to-noise benchmarks give better small-scale decisions and lower scaling-law error. 30 benchmarks, 375 models, 900K results. **Relation:** SNR is a cousin of a G coefficient; they do not use G-theory or tie noise to a ceiling on external validity.

**srinivasan2026ddr** (full, searched). Four-facet G-theory on TheAgentCompany, tau2-bench and AppWorld traces (Henderson I, REML, Bayesian GLMM agree). Agent main effect is under 3% of total variance in every dataset; agent x task interaction 7 to 23%. E rho^2 on tau2 action_checks falls from 0.752 to 0.000 on the hardest task quartile. Cites Cronbach et al. 1972. Does not use SWE-bench. **Relation:** the closest prior G-study of agent benchmarks. Ours differs in the benchmark (SWE-bench Verified, 182 public submissions) and in carrying the reliability into an attenuation ceiling on validity.

**hardy2026agentreliability** (full, searched). Bayesian G-theory variance decomposition over 22 benchmarks from the Holistic Agent Leaderboard and Harbor Index, including SWE-bench Verified Mini (24 models, 50 tasks, 2 scaffolds, 26 model-scaffold pairs). Fixed model-scaffold systems are ranked reliably (0.935 to 0.994); underlying-model reliability is 0.148 to 0.841; infinitely many similar tasks raise model-ranking reliability by at most 0.097 when scaffold coverage dominates. Cites Cronbach et al. 1972 and Brennan 2001. **Relation:** very close. They already apply G-theory to a SWE-bench Verified subset with a scaffold facet. Our G-study uses the full 500-item Verified set and 182 submissions, and adds the attenuation ceiling and the cross-benchmark (DIF, judge) unification.

**liu2026converged** (full, searched). Audits 254 public SWE-bench submissions across four splits without running models. On Verified the top two each resolve 396 of 500; the top ten share 285 successes and 51 failures, leaving 164 distinguishing instances; within-model scaffold ranges reach 29.8 pp vs an 8.8 pp spread across the top thirty; exact McNemar tests separate none of 29 adjacent top-thirty Verified pairs at alpha 0.05 (the Test split separates 14 of 23). No G-theory or variance components found in the text. **Relation:** same data source as our SWE-bench G-study; their analysis is paired testing and nesting, ours is variance decomposition and a validity ceiling. Must be cited and positioned against.

**anand2026askwhich** (abstract). G-theory on 373,019 LLM-judge scores (system, item, judge, interactions). Under one judge, generalizability asymptotes to sigma2_s/(sigma2_s + sigma2_sj) regardless of item count; pairwise both-order judging raises the ceiling to 0.986; first-position advantage of 8.6 pp. **Relation:** close to our judge-reliability arm (a judge facet sets a ceiling). Ours compares the GPT-4 judge to a human-to-human ceiling on MT-Bench; theirs is judge-panel sizing.

**li2026measuredorreal** (abstract). Shows the textbook Spearman correction (divide by the square root of two reliabilities) can return correlations above 1 (70 of 100 CIFAR-100 classes) and has infinite MSE; proposes correcting only when both reliabilities pass a significance test, dividing by upper confidence bounds. Prompt settings reverse rankings (ARC: Qwen2-1.5B trails Llama-3.2-3B by 3.8 points at 0-shot, leads by 16.6 at 2-shot). **Relation:** directly relevant to how we present the ceiling R^2 <= rho_X * rho_Y * C: we should state it as an upper bound with uncertainty on rho, not as a disattenuated point estimate.

**caban2026measurement** (full). Proposes V_total <= V1 x V2 x V3 (task generation, human-simulator calibration, automated judgment), applied to five agentic benchmark configurations. Survey of 55 agentic evaluation papers: 26 report no chance-corrected inter-rater statistic (four raters agree at Krippendorff alpha 0.834); about 80% use mismatched, incomplete or absent IRR metrics (rater range 60.7 to 84.4%). V1 failures in 7 of 10 benchmarks; up to 9 pp inter-simulator variance. No Spearman attenuation or G-theory found in the text. **Relation:** a multiplicative validity bound like ours, but knowledge-based and survey-driven, not estimated from item-level data.

**patel2026beyond** (report plus abstract metadata). Position paper grounded in AssetOpsBench and 14 implementation studies. Proposes ranking agents by predictive validity (correlation of in-sample rank with out-of-sample rank) and a twelve-tier measurement apparatus. Per the alphaXiv report, the CODS-2025 competition had public-vs-hidden Spearman rho = -0.13 for the execution track (n = 13) and 0.69 for planning (n = 20); verify these against the PDF before citing. **Relation:** argues for criterion validity in the same spirit; no reliability decomposition.

**cao2025codebench** (abstract). Survey of 672 code benchmarks (2014 to 2025) and a 55-item checklist (HOW2BENCH) for rigor, reliability and reproducibility. **Relation:** background for code-benchmark quality; not quantitative psychometrics.

## 2. IRT, psychometrics and DIF for LLM benchmarks

**polo2024tinybenchmarks** (abstract). 100 curated examples suffice to estimate MMLU (14K items) performance; releases tiny versions of Open LLM Leaderboard, MMLU, HELM, AlpacaEval 2.0. ICML 2024. **Relation:** our benchdif item data source; IRT for efficiency, not for invariance.

**rodriguez2021evaluation** (metadata). ACL 2021 paper arguing for IRT-based leaderboards because items differ in difficulty and discrimination. **Relation:** early IRT-for-leaderboards reference.

**lalor2016irt** (metadata), **vania2021comparing** (abstract: IRT over 29 datasets and 18 Transformers; Quoref, HellaSwag, MC-TACO discriminate strong models best). Early IRT-in-NLP references.

**hofmann2025fluid** (abstract). IRT plus adaptive item selection; higher validity and less variance on MMLU with fifty times fewer items. COLM 2025.

**zheng2026familydif** (full, searched). Residual DIF across model families via a spectral MIRT approximation on MMLU-Pro, BBH, MMLU, HellaSwag and WinoGrande (RouterEval response matrices). Full-benchmark and low-DIF rankings correlate at tau_b 0.900 to 0.948, yet in four of five benchmarks 30.9 to 47.1% of cross-family pairs within one point reverse order, 16.9 to 28.6 pp above matched-random subtests. No Mantel-Haenszel or logistic-regression DIF found in the text; cites tinyBenchmarks. **Relation:** the closest prior to benchdif. Same qualitative finding (aggregate rankings robust, near-ties and subsets fragile). Ours uses classical DIF (MH and logistic regression) on Open LLM Leaderboard and tinyBenchmarks items and frames the result as cancellation at the test level vs breakage in small subsets.

**jiang2026trustirt** (abstract). 18,000 simulations: IRT estimators designed for human testing behave poorly in AI regimes (few models, many items, skewed abilities). **Relation:** caveat for any IRT-based DIF claim.

**sachdeva2026rasch** (abstract). Many-facet Rasch models on nine LLM raters of the Measuring Hate Speech corpus; LLMs differ from humans in severity, item calibration, order robustness and scale use. **Relation:** rater-facet psychometrics, close to our judge arm.

**zhu2026ctt** (abstract). With one judge and one prompt, KR-20, Phi(lambda) and Livingston-Lewis cannot isolate judge error (KR-20 0.5223 on judge verdicts vs 0.5231 on gold). **Relation:** supports our choice to benchmark the judge against a human-to-human ceiling rather than an internal-consistency statistic.

## 3. Construct validity

**bean2025measuring** (abstract). 29 expert reviewers, systematic review of 445 LLM benchmarks; finds recurring problems in phenomena definitions, tasks and scoring; eight recommendations. NeurIPS 2025 D&B. **Relation:** the construct-validity framing we inherit; it is qualitative, ours is quantitative (reliability and invariance as preconditions for validity).

## 4. LLM-as-a-judge

**zheng2023judging** (full). MT-Bench and Chatbot Arena. Table 5 (MT-Bench, first turn): GPT-4 pairwise vs human agreement 66% in setup S1 (ties and position-inconsistent votes counted as ties; random = 33%) and 85% in S2 (non-tie only; random = 50%); human vs human 63% (S1) and 81% (S2). Table 2: GPT-4 position consistency 65.0% (default prompt), 30.0% biased toward the first answer; Claude-v1 23.8% consistent. Repetitive-list verbosity attack failure: GPT-4 8.7%, Claude-v1 and GPT-3.5 91.3%. Also names self-enhancement bias. **Relation:** our judge-reliability data and the source of the "judge matches human-human agreement" claim we re-test with a proper ceiling.

**chiang2024arena** (abstract). Chatbot Arena platform, over 240K votes, Bradley-Terry ranking; crowd votes agree well with experts.

**wang2023notfair** (abstract). Position manipulation lets Vicuna-13B beat ChatGPT on 66 of 80 queries with ChatGPT as judge; proposes multiple-evidence, balanced-position and human-in-the-loop calibration. **Relation:** position-bias baseline.

**panickssery2024selfpref** (abstract). GPT-4 and Llama 2 recognise their own outputs above chance; fine-tuning shows a linear relation between self-recognition and self-preference. **wataoka2024selfpref** (abstract). Quantitative self-preference metric; GPT-4 shows significant self-preference; judges favour lower-perplexity outputs. **Relation:** self-preference references for our judge arm.

**thakur2025judging** (abstract). 13 judges on 9 exam-taker models in a high inter-human-agreement setting; only the best judges approach human alignment, still well below inter-human agreement, with scores up to 5 points off. GEM2 2025. **bavaresco2024judgebench** (abstract). JUDGE-BENCH, 20 datasets, 11 LLMs; judge reliability varies widely by task and property. ACL 2025. **Relation:** both compare judges to humans; neither frames human-human agreement as a formal reliability ceiling on judge validity.

**kumar2026verdicts** (abstract). Verdicts change across identical temperature-0 replications; position swaps flip most verdicts on hard tasks; the most deterministic judge agrees with ground truth 51% of the time; derives an accuracy upper bound from position bias. **jha2026whosegold** (abstract). On 246 unanimous MT-Bench cells, benchmark authors and recruited experts differ on 30.5% and reverse on 8.5%, while the six-model leaderboard is unchanged (tau 1.00); pool switch moves win rate by 1.9 pp (SD). **Relation:** both bear directly on our MT-Bench human-to-human ceiling.

## 5. SWE-bench

**jimenez2024swebench** (abstract). 2,294 issues from 12 Python repos; ICLR 2024. **chowdhury2024verified** (not fetched; see flag above). The 500-instance human-validated Verified subset, described in SWE-smith as "a human-curated subset of 500 instances". **yang2025swesmith** (full, searched). Introduces SWE-bench Multilingual: 300 task instances across 9 additional languages ("a new dataset introduced in this paper"). Cite this for SWE-bench Multilingual.

## 6. Classical methods (metadata only, DOIs verified via Crossref or OpenAlex)

spearman1904proof (DOI 10.2307/1412159; Crossref has first page 72 only, confirm the page range), cronbach1963theory, cronbach1972dependability (no DOI; existence confirmed by 1972 to 1974 reviews in Science, JASA and AERJ in OpenAlex and by the full reference in Srinivasan 2026), brennan2001generalizability, holland1986dif (ETS report) and holland1988dif (Test Validity, eds. Wainer and Braun; Crossref record is the 2013 Routledge reissue, pp. 149 to 166), swaminathan1990logistic, meredith1993measurement, krippendorff2004reliability and hayes2007krippendorff (alpha), cohen1960kappa, shrout1979icc, mcnemar1947note.

## Closest existing work and what is new

1. G-theory for agent benchmarks already exists (Srinivasan 2026 on TheAgentCompany, tau2, AppWorld; Hardy et al. 2026 on 22 HAL benchmarks including a 50-task SWE-bench Verified Mini). We cannot claim the first G-study of agent evals. We can claim a G-study on the full 500-item SWE-bench Verified across 182 public submissions, as far as these searches found.
2. Liu et al. 2026 already audit 254 public SWE-bench submissions (shared successes, McNemar, scaffold effects). Our SWE-bench contribution must be framed as variance components plus an attenuation-based validity ceiling, not as the first leaderboard audit.
3. DIF across model families with the "robust aggregate, fragile near-ties" pattern is in Zheng and Yang 2026 (MIRT residual DIF). benchdif's distinct angle is classical MH and logistic-regression DIF, test-level cancellation, and the tinyBenchmarks and Open LLM Leaderboard data.
4. Multiplicative validity bounds exist in qualitative form (Caban 2026), and Li and Ni 2026 warn that Spearman disattenuation is unstable. Using R^2 <= rho_X rho_Y C as an upper bound with estimated reliabilities on real agent data was not found in prior work; present it as a bound with intervals.
5. Judge reliability against humans is well studied (Zheng 2023; Thakur 2025; Bavaresco 2025; Anand 2026; Jha 2026). Our addition is treating human-to-human agreement as an explicit reliability ceiling and placing judges, items and runs in one measurement model.
6. Bjarnason et al. 2026 supply an occasion-facet estimate (run-to-run SD 0.7 to 1.8 pp on SWE-bench Verified) that our single-run G-study cannot estimate; cite it to bound the confounded residual.
7. The genuinely new part is the unification: one instrument-theory frame (reliability, invariance, validity ceiling) applied across static benchmarks, agent benchmarks and judges, with each arm's result reproducible from public item-level data.
