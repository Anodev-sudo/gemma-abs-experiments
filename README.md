# gemma-abs-experiments

Teaching very small language models to write one-sentence briefs for arXiv papers, so a
phone can do it on-device with no server call.

The end result is a 97 MB build of Gemma 3 270M that comes within about a tenth of a point of
IBM's Granite 4.0 350M (374 MB) on faithfulness to the abstract, with no significant difference. It runs on-device in FirstPass, an arXiv reading app. All
training ran on one MacBook, with LoRA in [MLX](https://github.com/ml-explore/mlx).

A brief answers "why might I open this paper?" in one plain sentence of about 10 to 15 words.
It should name the subject and the main finding, skip filler like "This paper proposes", and
claim nothing the abstract doesn't support. That last rule is the hard one for a 270M model.

## The two phases

**Phase 1, April to May 2026 (most of this repo).** The first job was training data. A
writer/reviewer pipeline over OpenRouter turned balanced samples of recent arXiv abstracts into
training rows: one model writes, a stricter model reviews or rewrites, and local filters drop
the rest. DeepSeek V4 Pro did most of the writing and much of the reviewing, Qwen models
reviewed too, and later lanes used Kimi K2.6 to rewrite the cases the student got wrong. Gemma 3 270M then went through
many rounds of LoRA training on a 10.7k-row set: 4-bit hardening, 5-bit correction overlays,
DPO and token-weighted gold sets. Alongside that ran runtime tests (LiteRT, llama.cpp, MLX
Swift), a safety eval, and a 994-row "gold" set of natural-language briefs. Granite 4.0 350M,
trained on that gold set, read better and invented less than any Gemma build of the time, so
it shipped first, at 8-bit.

**Phase 2, June 2026 ([`gemma-lab-advanced/`](gemma-lab-advanced/)).** Closing the gap. Cut
Gemma's vocabulary to the tokens arXiv actually uses, distilled plain, verified briefs from a
larger Gemma, fine-tuned on those alone, and quantized to 5 bits. That build lands close behind
Granite on faithfulness at about a quarter of the size. Granite still writes catchier briefs; the lab
explains why that gap was left alone.

| | Gemma 3 270M, q5 | Granite 4.0 350M, q8 |
|---|---|---|
| Size | **97 MB** | 374 MB |
| Faithful (0 to 3), typical / hard papers | 2.08 / 2.09 | 2.20 / 2.24 |
| Hook (0 to 3), typical papers | 1.09 | 1.37 |

Neither faithfulness gap is significant (95% intervals −0.26 to +0.03 and −0.35 to +0.05), but
those intervals leave room for a gap of about a quarter point, so this is not proof the two are
equal. The hook gap is significant. Method and full numbers are in [`gemma-lab-advanced/README.md`](gemma-lab-advanced/README.md).

## Where to start

- [`gemma-lab-advanced/README.md`](gemma-lab-advanced/README.md) and its
  [`CONCLUSION.md`](gemma-lab-advanced/CONCLUSION.md): how the shipped model was built.
- [`notes/README.md`](notes/README.md): about 100 lab notes from phase 1, grouped by theme.
- [`experiments/README.md`](experiments/README.md): the April prompt sprints, where several
  coding agents each tried one change to the writer prompt.
- [`scripts/summary_generation/`](scripts/summary_generation/): the config-driven
  writer/reviewer pipeline.

## Layout

```
configs/
  synthetic_summary_generation/   one JSON per pipeline lane: models, prompts, filters, outputs
  mlx_lora/ sft/ dpo/             training configs
prompts/                          writer, reviewer, runtime and safety-eval prompts
scripts/                          dataset builders, review-packet tools, evals, exports
tools/
  run-tui/                        terminal browser for pipeline run outputs
  mlx-summary-probe/              macOS probe that runs a model through MLX Swift
  summary_safety_filter/          Swift package for output safety checks
docker/litert-export/             container for LiteRT export
tests/                            unit tests: python3 -m unittest discover -s tests
notes/  experiments/              write-ups
gemma-lab-advanced/               phase 2
hf_dataset/  hf_bucket/           READMEs only (see below)
```

## Data and weights

The shipped model is public on Hugging Face at
[peering-latent/gemma-abs-fit](https://huggingface.co/peering-latent/gemma-abs-fit), with a
model card and a code sample (use `q5/faithful_only_q5`).

Data is not in git. Paths that start with `hf_dataset/` or `hf_bucket/` in configs, scripts
and notes point to data kept in private Hugging Face storage ([`REMOTE_TARGETS.md`](REMOTE_TARGETS.md)).
The code and write-ups are complete, but most scripts need that data to run.

## License

Code, prompts and write-ups in this repo are under the [MIT License](LICENSE). Model weights
are not in this repo; they are fine-tunes of Gemma 3 and fall under Google's
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).
