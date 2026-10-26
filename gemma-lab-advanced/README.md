# gemma-lab-advanced

Can Gemma 3 270M write a faithful one-sentence brief for an arXiv paper, and still fit in
under 100 MB on a phone? This lab says yes. The result is `faithful_only_q5`, a 97 MB model
that comes within about a tenth of a point of IBM's Granite 4.0 350M (374 MB at 8-bit) on
faithfulness, with no significant difference.

A brief is one plain sentence, about 10 to 14 words, that tells a reader why they might open
the paper. "Faithful" means every claim in it is backed by the abstract.

## Result

Same prompt, same MLX runtime, greedy decoding, blind judges. Scores are 0 to 3.

| | faithful_only_q5 (Gemma) | Granite 4.0 350M q8 |
|---|---|---|
| Size on disk | **97 MB** | 374 MB |
| Faithful, 240 typical papers | 2.08 | 2.20 |
| Faithful, 150 hard papers | 2.09 | 2.24 |
| Hallucinated briefs, typical / hard | 3.3% / 9.4% | 3.8% / 8.1% |
| Readable, typical papers | 1.71 | 1.80 |
| Hook, typical papers | 1.09 | 1.37 |

Gemma trails on faithfulness by 0.11 on typical papers (95% bootstrap interval −0.26 to +0.03)
and 0.14 on hard ones (−0.35 to +0.05). Neither gap is significant, and hallucination rates are
about the same. That is not proof the two are equal: the intervals leave room for a gap of
about a quarter point, which would fail the −0.10 non-inferiority bar this lab set for its last
experiment. It does put a model a quarter of the size close behind. Granite keeps a real lead on hook, the score for how
much a brief makes you want to read on. Three attempts to close that gap moved hook by nothing,
and two of them cost faithfulness, so the lab treats it as a limit of model size and leaves it.
Full numbers: [`quant_probe/VOCAB_STRIP_FINDINGS.md`](quant_probe/VOCAB_STRIP_FINDINGS.md) (top
section). The verdict and reasoning: [`CONCLUSION.md`](CONCLUSION.md).

## How

1. **Cut the vocabulary.** Gemma has a 262k-token vocabulary, and its tied embeddings are 63%
   of the weights. Tokenizing all 3.07M arXiv titles and abstracts showed 55% of those tokens
   never appear. The cut keeps tokens found in at least 100 papers, plus the byte alphabet and
   every BPE merge parent needed to rebuild them, so any input still tokenizes. That leaves
   65,465 tokens and takes the model from 268M to about 142M parameters. Code: `quant_probe/strip_vocab.py`, checked by
   `quant_probe/test_strip_tokenizer.py`.
2. **Distill for faithfulness, not flair.** A larger Gemma (gemma-4-26b-a4b-qat, served
   locally by LM Studio, prompt only) wrote plain briefs for recent papers. The same teacher,
   in a fresh context, checked each claim against the abstract, repaired what it could and
   dropped the rest: 5,182 rows. Teacher and student are both Gemma, so the license chain stays
   simple. Code: `quant_probe/distill/`.
3. **Fine-tune with LoRA on a MacBook.** Rank 16 on attention and MLP layers, in MLX, trained
   straight onto the cut model so it learns the new tokenization at the same time. Config:
   `training/configs/faithful_df100_stage1.yaml`.
4. **Quantize to 5 bits** (group size 64). 8-bit and 5-bit both stay close to full precision.
   4-bit loses too much, even with quantization-aware LoRA. Details: `quant_probe/QUANT_FINDINGS.md`.

What made the difference was step 2. Earlier rounds trained on catchier targets and then
drilled a "punchy" voice on top. At 270M parameters that teaches the model to invent a payoff
the abstract never states. Plain targets and no voice stage gave +0.31 faithfulness over the
previous best Gemma build, at the same size.

## How it was judged

- Two held-out sets: 240 recent papers drawn like the live feed, and 150 hard papers that
  contain words the vocabulary cut has to break into pieces.
- Each brief is scored for faithful, readable and hook by two independent blind judges
  (Claude subagents) against the abstract alone. Briefs are anonymized and shuffled per
  paper. Rubric: `quant_probe/judge3way_condhook/RUBRIC.md`.
- Paired per-paper deltas with bootstrap 95% intervals. The two judges gave the same
  faithfulness score 71% to 80% of the time.
- The harness is reusable: `quant_probe/gen_faith_vs_granite.sh`, `build_hg_shards.py`,
  `aggregate_hg.py`.

## Layout

```
CONCLUSION.md          the answer, and why the hook gap was left alone
HISTORY.md             the running lab log, as written during the work
SPEC_*.md              spec for the last experiment (grounded subject-fronting; null result)
recent_briefs_*.md     sample briefs from both models on recent papers
eval_v5_vs_granite/    round 1: the first Gemma vs Granite read, and its harness
quant_probe/           vocabulary cut, quantization, and every judged round
  distill/             the teacher pipeline: gather, generate, verify, repair, build
training/              LoRA configs, train scripts, and per-run notes
gold_pipeline/         notes on the 994-row human-style gold set (code is in ../scripts)
```

## Not in this repo

Paper pools, generated briefs, judge outputs and the Python venv. They live in a private
storage bucket. The shipped model itself is public on Hugging Face:
[peering-latent/gemma-abs-fit](https://huggingface.co/peering-latent/gemma-abs-fit)
(`q5/faithful_only_q5`). The scripts still name the paths they used, so they document the
work but will not run on a fresh clone without that data.

To run anything: Apple Silicon, Python 3.12 and `mlx-lm`. Shell scripts move to the lab root
themselves. The teacher scripts expect LM Studio at `127.0.0.1:1234`.
