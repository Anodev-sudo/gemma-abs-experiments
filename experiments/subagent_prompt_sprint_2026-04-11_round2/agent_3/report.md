# Agent 3 Report

## Prompt idea

Direction: consequence-first / reader-value wording.

I kept the reviewer unchanged and changed only the generator:
- new generator: `deepseek_v32_generator_consequence_first_v1.json`
- reviewer: baseline `qwen3_235b_a22b_2507_reviewer_classifier_v2_headline_baseline.json`

Prompt thesis:
- answer `why should I care?` more explicitly
- name the consequence, implication, or practical reason to open
- keep sentence-case, natural feed prose
- do not drift back into title-like phrasing or label-first openings
- for theory papers, treat the consequence as a stronger limit, broader result, cleaner proof target, or useful guarantee rather than fake real-world hype

## Run results

### Run A
- seed: `20260411`
- output: `run_a/`
- result: `83 kept / 17 dropped`
- local filters: `5`
- model-reviewed rows: `95`
- average kept length: `11.96` words
- generic openings in kept rows: `10`

Drop mix:
- local: `5` over-15-word drops
- model: `11` `misses the main payoff`, `1` `too vague`

### Run B
- seed: `20260412`
- output: `run_b/`
- result: `84 kept / 16 dropped`
- local filters: `7`
- model-reviewed rows: `93`
- average kept length: `12.02` words
- generic openings in kept rows: `9`

Drop mix:
- local: `6` over-15-word drops, `1` near-duplicate title
- model: `7` `misses the main payoff`, `1` `too vague`, `1` `mostly repeats the title`

Note:
- the first `run_a` attempt had duplicated JSONL rows despite a sane manifest, so I retried once and used the clean rerun above

## Two-run read

Combined read:
- average kept rows: `83.5 / 100`
- average dropped rows: `16.5 / 100`
- average kept length: `11.99` words
- combined generic openings across both runs: `19`

What improved:
- the kept rows usually do make the consequence clearer than the current baseline prompt
- the prose stayed mostly natural and sentence-case
- title overlap stayed low
- it handled theory papers better than a purely headline-y style when the abstract offered a real structural payoff

What did not improve enough:
- the main failure mode was still `misses the main payoff`
- some summaries got consequence-ish wording without actually landing the strongest reason to open
- generic/article openings still showed up more than I want
- the branch still leaked a few label-first or mildly abstract-like lines, especially on toolkits, theory, and survey-ish papers

Style read on kept rows:
- best cases felt useful and reader-oriented, not title-cased and not overly punchy
- weaker accepts still sounded a little like: `X does Y in setting Z`, which is cleaner than a title but not always a compelling feed line

Representative kept lines:
- `LicenseGPT cuts license review time by 94% for commercial AI developers.`
- `TMR oscillates with barrier thickness due to interference, not quantum confinement.`
- `Nearest neighbors in Wasserstein space impute distributions better than entry samples alone.`
- `Splines that respect density constraints improve bivariate density estimation and dependence analysis.`

Representative misses:
- `A single Python library unifies evolutionary computation and scikit-learn workflows.`  
  This is clear, but the reviewer was right that it still does not really say why to open.
- `AI City Challenge 2023 set records for retail and traffic vision tasks across five tracks.`  
  Too event-like and not enough real payoff.

## Final recommendation

This is a solid direction, but not my top branch.

I would keep the idea, not this exact prompt:
- preserve the consequence-first framing
- keep the unchanged reviewer
- carry forward the better parts into a later hybrid

I would **not** promote this generator as the new baseline on its own.

Why:
- the two runs were stable, which is good
- but they stabilized in the `83-84 kept` range, not in standout territory
- the prompt improved reader-value wording somewhat, but not enough to consistently beat the stronger existing variants
- the biggest remaining problem is still payoff selection, not sentence mechanics
