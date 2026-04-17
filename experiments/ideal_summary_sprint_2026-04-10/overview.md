# Ideal Summary Sprint Overview

This folder contains a four-agent comparison exercise focused on a narrower question than prompt tuning:

- What does an ideal one-sentence paper-feed summary actually look like?
- What changes when GPT-5.4 writes the summaries directly, without the current generator/reviewer pipeline?

## Shared Setup

- All four agents read the current baseline generator prompt.
- All four agents read the current baseline reviewer prompt.
- All four agents summarized the same 10 papers from `shared_papers_10.jsonl`.
- None of them used the repo pipeline or external model calls for the summaries.

## Where They Agreed

- The ideal summary is not a compressed abstract.
- The ideal summary should answer: `Why open this paper?`
- One clear payoff beats broad completeness.
- Title restatement is a major failure mode.
- Method-first phrasing is usually weaker than result-first or consequence-first phrasing.
- Technical papers do not need to be dumbed down, but they do need a clean hook.

## Where They Differed

- Agents A, B, and C lean more headline-like.
- Agent D is more skeptical of overly “sold” phrasing, especially for math, theory, and survey papers.
- Across the 10 papers, the main tension is not brevity versus accuracy. It is boldness versus trustworthiness.

## Cross-Agent Takeaway

The strongest shared view is:

`Ideal summary = concrete subject + strongest payoff + no wasted setup`

The main caution from the exercise is:

`Payoff-first does not always mean louder.`

For theory, statistics, and broad review papers, the best line is often calmer and more literal than the best line for systems or applied ML work.

## Reports

- `agent_a/report.md`
- `agent_b/report.md`
- `agent_c/report.md`
- `agent_d/report.md`

## Summary Comparison

### 2308.07193 — Task Offloading for Smart Glasses in Healthcare

- Agent A: `Offloading smart-glasses work cuts battery drain and speeds healthcare temperature screening.`
- Agent B: `Offloading smart-glasses vision tasks to nearby devices cuts battery drain for hospital fever screening.`
- Agent C: `Offloading smart-glasses vision tasks can save battery and speed healthcare screening.`
- Agent D: `Offloading fever detection from smart glasses can cut latency and battery strain in the field.`

### 2412.16453 — The Evolving Usage of GenAI by Computing Students

- Agent A: `Computing students now rely on GenAI more often, but use it less intensely.`
- Agent B: `Computing students increasingly use GenAI for help-seeking, but daily use may be overstated.`
- Agent C: `Computing students now rely on GenAI far more, but report less frequent use over time.`
- Agent D: `ChatGPT has become a mainstream help resource for computing students, nearly matching search.`

### 2602.12406 — Eyes on Many

- Agent A: `DoublePinch plus Gaze+Pinch is the best XR multi-object selection combo.`
- Agent B: `DoublePinch with Gaze+Pinch outperforms gaze, hand, and voice for XR multi-object selection.`
- Agent C: `DoublePinch and gaze selection perform best for selecting multiple objects in XR.`
- Agent D: `Double-pinch with gaze-plus-pinch was fastest for multi-object selection in XR.`

### 2603.11661 — Resonate

- Agent A: `Online RL and audio-language rewards push text-to-audio quality and alignment higher.`
- Agent B: `Online RL with audio-language feedback pushes text-to-audio models to new SOTA.`
- Agent C: `Online RL with audio-language rewards pushes text-to-audio models to new SOTA.`
- Agent D: `Online RL with audio-language feedback lifts text-to-audio quality and semantic alignment.`

### 2503.21304 — Euclid protocluster candidates

- Agent A: `Euclid is already finding protocluster candidates and probing the earliest cluster-building sites.`
- Agent B: `Euclid confirms two known protocluster candidates and extends the search for protoclusters to higher redshift.`
- Agent C: `Euclid already finds high-redshift protoclusters and can probe cluster seeds to about five and a half.`
- Agent D: `Euclid already recovers high-redshift protoclusters and could extend such searches to z about 5.5.`

### 2305.00168 — Structural, electronic, and magnetic properties of CrTe2

- Agent A: `Strain and band filling flip CrTe2 magnetism, showing how to tune its anisotropy.`
- Agent B: `DFT predicts 1T-CrTe2 as the ground state and shows strain can flip its magnetic orientation.`
- Agent C: `Strain and band filling flip CrTe2 magnetization, revealing tunable anisotropy in monolayers and bilayers.`
- Agent D: `Strain and band filling can flip CrTe2's magnetic orientation between in-plane and out-of-plane.`

### 2412.18794 — Entropic adapted Wasserstein distance on Gaussians

- Agent A: `Closed-form adapted Wasserstein formulas now extend to Gaussian processes in higher dimensions.`
- Agent B: `Closed-form formulas now extend entropic adapted Wasserstein distance to multidimensional Gaussian processes.`
- Agent C: `Closed-form formulas now extend adapted Wasserstein distance to Gaussian processes with entropic regularization.`
- Agent D: `Entropic adapted Wasserstein distances for Gaussian time series now have multidimensional closed-form solutions.`

### 1603.07714 — On tessellations of random maps and the t_g-recurrence

- Agent A: `Voronoi cell masses on random surfaces tie into Painleve I and the t_g recurrence.`
- Agent B: `Voronoi cell masses on random surfaces link to Painlevé I and a simple Dirichlet law.`
- Agent C: `Voronoi cell masses on random surfaces may follow a simple Dirichlet law.`
- Agent D: `Voronoi cell masses on Brownian surfaces connect map asymptotics to the t_g recurrence.`

### 2407.11435 — Genomic Language Models: Opportunities and Challenges

- Agent A: `Genomic language models could reshape DNA analysis, but scaling and evaluation remain hard.`
- Agent B: `Genomic language models could speed genome interpretation, but scaling them to complex species remains hard.`
- Agent C: `Genomic language models could predict function and design DNA, but scaling them remains hard.`
- Agent D: `Genomic language models could transform sequence design and function prediction, but scaling remains hard.`

### 2506.21353 — Bayesian Modeling for Aggregated Relational Data

- Agent A: `Stan-based Bayesian ARD tools make hidden-population inference and model checking more practical.`
- Agent B: `A Stan-based Bayesian toolkit improves hidden-population inference from aggregated relational data.`
- Agent C: `Stan implementations make Bayesian network-size models faster to fit and easier to compare.`
- Agent D: `Stan-based Bayesian ARD models improve fit diagnostics, runtime, and model comparison.`
