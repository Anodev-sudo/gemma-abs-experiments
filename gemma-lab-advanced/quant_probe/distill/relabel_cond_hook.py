#!/usr/bin/env python3
"""Stage A of cond_hook: CONDITIONAL subject-fronting relabel of the faithful stage-1 targets.

Each row's current faithful brief (target_v1) is rewritten to FRONT a specific named mechanism
as its subject + grounded payoff -- but ONLY when that specific is clearly supported by the
abstract. Otherwise the generic-hedge form (target_v1) is kept verbatim. A fresh-context verify
gate (claim-decomposition + subject-NP-present + NO contrast clause) is the firewall: any fronted
candidate that fails is DEMOTED back to target_v1 (never dropped -- we preserve the row count so
the only delta vs the faithful set is surface form).

  python relabel_cond_hook.py --regenerated regenerated.jsonl --papers pool.json \
      --out cond_hook/relabeled.jsonl [--cap N] [--resume]

All teacher ops by the LMStudio-served gemma-4-26b-a4b-qat (HTTP) -- see client.py.
The Claude verifier-precision audit (Stage A step 2) is a SEPARATE, later step, not in this file.
"""
import sys, json, argparse, re, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat

# --- Relabeler: front a GROUNDED specific subject, else keep the generic brief unchanged. -------
# NB this DELIBERATELY inverts regenerate.py rule (4) ("do NOT name the system/method"): naming the
# grounded specific is the entire point here. The naming is still gated downstream by the verifier.
RSYS = (
 "You rewrite a one-sentence arXiv feed blurb. You are given an ABSTRACT and the CURRENT blurb "
 "(already faithful but often opening with a generic subject like 'A new method/framework/dataset'). "
 "Your job: when the abstract clearly names a SPECIFIC mechanism -- a named method, dataset, object, "
 "or concrete result -- rewrite the blurb to lead with THAT specific as its subject, followed by its "
 "payoff. Template: [specific subject] [payoff verb] [grounded object]. "
 "Front the specific ONLY IF it is verbatim or clearly entailed by the ABSTRACT. If no specific is "
 "clearly supported, return the CURRENT blurb unchanged. "
 "HARD RULES: (1) never invent an entity, number, system name, or result not in the abstract; "
 "(2) NEVER add a contrast/surprise clause ('even when ...', 'without ...', 'despite ...', "
 "'yet still ...') -- this is forbidden regardless of grounding; (3) one sentence with a verb, "
 "14 words or fewer; (4) no LaTeX, equations, or dense numbers; (5) no filler ('This paper', 'We'); "
 "(6) when in doubt, keep the generic form. "
 "Output ONLY a JSON object on a single line: "
 '{"frontable": true|false, "subject_np": "<the specific subject you fronted, or null>", '
 '"source_span": "<short abstract phrase that grounds it, or null>", "brief": "<the final sentence>"}'
)

# --- Verifier (fresh context): the firewall. Sees ONLY abstract + candidate, not the relabel. ----
# = regenerate.py VSYS + two cond_hook-specific automatic-FAIL checks (subject-NP, contrast clause).
VSYS = (
 "You are a meticulous fact-checker. Given an ABSTRACT and a one-sentence BLURB, check the blurb. "
 "First, on separate lines, list each distinct factual claim the blurb makes; after each write "
 "SUPPORTED (quote the abstract phrase) or UNSUPPORTED. "
 "Then check the BLURB's grammatical SUBJECT: write 'SUBJECT-GROUNDED' if the subject noun phrase is "
 "named or clearly entailed in the abstract, else 'SUBJECT-UNGROUNDED'. "
 "Then write 'CONTRAST-CLAUSE: YES' if the blurb contains any surprise/contrast clause "
 "('even when', 'without', 'despite', 'yet', 'while still'), else 'CONTRAST-CLAUSE: NO'. "
 "Also state whether the blurb captures the paper's MAIN contribution or a side point. "
 "Final line 'VERDICT: PASS' ONLY IF every claim is SUPPORTED, the subject is SUBJECT-GROUNDED, "
 "CONTRAST-CLAUSE is NO, and the main contribution is captured; otherwise "
 "'VERDICT: FAIL - <reason>'. Be strict: invented specifics, an ungrounded subject, a contrast "
 "clause, overstatement, inversion, or a misread of the main point all mean FAIL."
)

JSON_RE = re.compile(r"\{.*\}", re.S)
# Unambiguous "surprise/contrast tail" constructions forbidden by spec §3. Deterministic belt-and-
# suspenders to the teacher gate: any fronted candidate matching this is demoted, no second-guessing.
# NB intentionally NOT banning bare "without" -- it is often a benign grounded capability claim
# ("segments images without labels"); that case is left to the teacher verifier's grounding check.
SURPRISE_RE = re.compile(r"\b(even (when|if|though|as)|despite|yet still|while still|albeit)\b", re.I)


def relabel(ab, b):
    """Return (frontable_proposed, subject_np, source_span, brief). Falls back to keep-generic."""
    out = chat([{"role": "system", "content": RSYS},
                {"role": "user", "content": f"ABSTRACT:\n{ab}\n\nCURRENT blurb:\n{b}\n\nJSON:"}],
               temp=0.2, max_tokens=160)
    m = JSON_RE.search(out or "")
    if not m:
        return (None, None, None, None)  # parse fail -> caller keeps generic
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return (None, None, None, None)
    brief = (d.get("brief") or "").strip().strip('"')
    return (bool(d.get("frontable")), d.get("subject_np"), d.get("source_span"), brief)


def verify(ab, b):
    """Fresh-context gate. Returns (verdict 'PASS'|'FAIL', reason)."""
    out = chat([{"role": "system", "content": VSYS},
                {"role": "user", "content": f"ABSTRACT:\n{ab}\n\nBLURB: {b}"}],
               temp=0.0, max_tokens=360)
    m = re.search(r"VERDICT:\s*(PASS|FAIL)\s*-?\s*(.*)", out or "", re.I | re.S)
    if not m:
        return ("FAIL", "no-verdict")
    reason = re.split(r"[\r\n]", m.group(2).strip())[0][:120] if m.group(2) else ""
    return (m.group(1).upper(), reason or "")


def ok_len(b):
    return bool(b) and 4 <= len(b.split()) <= 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerated", required=True, help="faithful targets: {arxiv_id, brief}=target_v1")
    ap.add_argument("--papers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=0, help="pilot: only first N rows")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    papers = {p["arxiv_id"]: p for p in json.load(open(a.papers))}
    items = [json.loads(l) for l in open(a.regenerated) if l.strip()]
    if a.cap:
        items = items[:a.cap]

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    done, mode = set(), "w"
    if a.resume and outp.exists():
        for l in open(outp):
            done.add(json.loads(l)["arxiv_id"])
        mode = "a"

    cnt = collections.Counter()
    with open(outp, mode) as fh:
        for it in items:
            aid = it["arxiv_id"]
            if aid in done or aid not in papers:
                continue
            ab = papers[aid]["abstract"]
            v1 = it["brief"].strip()  # current faithful target = generic-hedge default

            proposed, subj, span, cand = relabel(ab, v1)

            if not proposed or not ok_len(cand) or cand.strip().lower() == v1.lower():
                # teacher kept generic, parse-failed, or produced an unusable/no-op candidate
                final, frontable, status, reason = v1, False, (
                    "parse_fail" if proposed is None else "kept_generic"), ""
                subj = subj if frontable else None
            elif SURPRISE_RE.search(cand):
                final, frontable, status, subj, reason = v1, False, "demoted", None, "surprise-clause"
            else:
                verdict, reason = verify(ab, cand)
                if verdict == "PASS":
                    final, frontable, status = cand, True, "fronted"
                else:
                    # one repair attempt, then demote to generic (never drop)
                    p2, s2, sp2, cand2 = relabel(ab, v1)
                    if p2 and ok_len(cand2) and cand2.strip().lower() != v1.lower() \
                            and not SURPRISE_RE.search(cand2) and verify(ab, cand2)[0] == "PASS":
                        final, frontable, status, subj, span = cand2, True, "fronted_repaired", s2, sp2
                    else:
                        final, frontable, status, subj = v1, False, "demoted", None

            cnt[status] += 1
            row = {"arxiv_id": aid, "brief": final, "frontable": frontable,
                   "subject_np": subj if frontable else None, "target_v1": v1, "status": status}
            if status in ("demoted",):
                row["fail_reason"] = reason
            fh.write(json.dumps(row) + "\n")
            fh.flush()

            n = sum(cnt.values())
            if n % 100 == 0:
                print(f"  {n}/{len(items)}  {dict(cnt)}", flush=True)

    total = sum(cnt.values())
    fronted = cnt["fronted"] + cnt["fronted_repaired"]
    proposed_front = fronted + cnt["demoted"]
    print(f"[relabel] total={total} fronted={fronted} demoted={cnt['demoted']} "
          f"kept_generic={cnt['kept_generic']} parse_fail={cnt['parse_fail']} "
          f"frontable_final={fronted/total:.1%} proposed={proposed_front/total:.1%} -> {a.out}")


if __name__ == "__main__":
    main()
