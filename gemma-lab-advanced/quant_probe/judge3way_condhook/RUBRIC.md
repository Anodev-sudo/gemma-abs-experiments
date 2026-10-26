# Blind brief-judging rubric (3-way)

You are a strict, blind judge of one-sentence arXiv "feed briefs" (a single sentence giving a
reader a reason to open a paper). For each paper you score three anonymized briefs — A, B, C —
against the paper's abstract. You do NOT know which system produced which; judge only on merit.
Judge faithfulness against the ABSTRACT only — it is ground truth.

Score EACH of A, B, C on three dimensions, INTEGER 0–3:

**faithful** (most important):
- 3 = every claim supported by the abstract AND captures the paper's main contribution
- 2 = mostly faithful, only minor imprecision or slightly off emphasis
- 1 = misleading — overstates, wrong emphasis, or a claim not clearly supported
- 0 = hard hallucination — invented entity/number, inverted result, or fabricated specific

**readable**:
- 3 = clear, fluent, a non-specialist could follow
- 2 = mostly readable, some jargon
- 1 = clunky, jargon-heavy, or grammatically broken
- 0 = incoherent / garbled

**hook**:
- 3 = compelling, specific reason to open
- 2 = moderate interest
- 1 = flat / generic ("A new method improves X")
- 0 = no hook or off-putting

Be strict and consistent. A named specific (acronym, system, dataset) that is NOT in the abstract
is faithful=0 for that brief — never a bonus for specificity. Use the full 0–3 range; do not
flatline everything at 2 or 3.

## Output format (the aggregator depends on this EXACTLY)

Write a JSON file with the Write tool to the path you are given:
```
{"pass": <P>, "set": "<repr|stress>", "shard": <N>, "scores": {
  "<arxiv_id>": {"A": {"faithful": int, "readable": int, "hook": int},
                 "B": {"faithful": int, "readable": int, "hook": int},
                 "C": {"faithful": int, "readable": int, "hook": int}},
  ...
}}
```
Include EVERY paper in the shard. After writing, report papers-scored + mean faithful per slot.
