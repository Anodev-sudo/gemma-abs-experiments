# Brief-distillation pipeline (bigger-Gemma teacher, prompt-only)

Goal: produce a large, high-quality, **brief-objective** training corpus by prompting a
bigger Gemma (no finetune — few-shot voice) and keeping only faithfulness-verified rows.
Teacher and student are both Gemma → same license, distillation is clean.

## Why prompt-only works here
The brief *voice* is carried by **few-shot exemplars** (the 894 gold briefs) + an explicit
rules block that contrasts brief-vs-summary (the axis rounds 2–5 showed is what matters).
No weight updates to the teacher; we just show it the objective.

## Stages
1. **gather_papers.py** — assemble a paper pool (title+abstract), dedup against the eval
   sets (repr-240, stress-150) and the gold 894 so we never train on eval material.
   Sources: local 10k grounding pool, or pull fresh from the VPS arxiv DB ("more real data").
2. **generate.py** — few-shot teacher generation. K gold exemplars + rules → one brief/paper.
   `--model <bigger-gemma>`. Over-generate (target ≥1.3× the rows you want to keep).
3. **verify.py** — independent faithfulness pass by the SAME teacher (fresh context, sees
   only abstract+brief): keep only FAITHFUL. Plus heuristic pre-filter (length/filler/dup).
4. **build_dataset.py** — assemble verified rows into the chat-format stage-1 grounding set
   (`training/data/curriculum_stage1_<name>`), reuse brief 50/50 valid/test.

Then: retrain DF≥100 (brief-grounding stage1 + voice stage2) on the new set → q5 → blind
double-judge vs the current DF≥100-q5 baseline (existing harness).

## Teacher: gemma-4-26b-a4b-qat via LMStudio (HTTP, OpenAI-compatible)
The teacher is served at `http://127.0.0.1:1234` — `client.py` talks to it (stdlib only).
It is a **reasoning** model: we always send `reasoning_effort=none`, otherwise it spends the
whole token budget on hidden thinking and returns empty content (and runs ~10x slower). With
reasoning off it's ~0.7–1s/paper on the full few-shot prompt — an 8k run is a couple of hours.

## Run order
```
D=quant_probe/distill
python $D/gather_papers.py --source vps --n 8000 --out $D/pool.json
python $D/generate.py --papers $D/pool.json --shots 6 --temp 0.3 --out $D/cand.jsonl   # --resume to continue
python $D/verify.py   --papers $D/pool.json --cand $D/cand.jsonl --out $D/verified.jsonl
python $D/build_dataset.py --papers $D/pool.json --verified $D/verified.jsonl --name gemma4xl_briefground_v1
```
Over-generate to absorb verify rejection: ~8k candidates → ~5–6k verified.
(`--model` defaults to the served model; pass it only to point at a different endpoint id.)
