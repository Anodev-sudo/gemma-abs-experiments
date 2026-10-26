import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab_common import TEST_SPLIT, build_prompt, clean, word_count
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
model_path, label = sys.argv[1], sys.argv[2]
model, tok = load(model_path)
sampler = make_sampler(temp=0.0)
out = Path("eval_v5_vs_granite/out")/f"{label}.jsonl"
rows=[json.loads(l) for l in TEST_SPLIT.read_text().splitlines() if l.strip()]
res=[]
with out.open("w") as fh:
    for i,row in enumerate(rows):
        ids=build_prompt(tok,row["messages"])
        raw=generate(model,tok,prompt=ids,max_tokens=40,sampler=sampler,verbose=False)
        b=clean(raw); res.append((i,b,word_count(b)))
        fh.write(json.dumps({"idx":i,"paper_id":row.get("paper_id"),"brief":b,"words":word_count(b),"raw":raw})+"\n")
n=len(res); wc=[w for _,_,w in res]; over=sum(w>14 for w in wc)
print(f"[{label}] {n} rows | avg {sum(wc)/n:.1f}w | <=14: {n-over}/{n}")
