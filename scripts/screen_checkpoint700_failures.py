#!/usr/bin/env python3
"""Heuristic screen for numeric corruption and semantic-smoothing risks."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

STOP = {
    'the','a','an','and','or','of','in','on','to','for','with','from','by','as','at','is','are','be','this','that','these','those',
    'using','use','uses','used','can','may','could','will','new','paper','study','model','models','method','methods','approach','framework',
    'system','systems','result','results','problem','problems','data','based','show','shows','provide','provides','yield','yields','make','makes',
    'improve','improves','boost','boosts','enable','enables','reveal','reveals','find','finds','through','without','within','between','than','more',
    'less','high','low','different','important','efficient','robust','performance','algorithm','algorithms'
}
GENERIC_TERMS = {
    'improve','improves','boost','boosts','enable','enables','reveal','reveals','provide','provides','new','efficient','robust','framework',
    'approach','solution','method','system','performance','useful','significant','powerful','better','faster','novel','effective'
}
POLARITY_TERMS = {'not','without','cannot','unable','never','no','rather','instead','but','preventing','prevents'}
NUM_RE = re.compile(r'(?<![A-Za-z])(?:~|≈|about\s+)?\d+(?:,\d{3})*(?:\.\d+)?(?:\^\d+)?(?:\s*(?:%|percent|x|×|nm|km|kg/s|kg|g|K|TeV|MeV|Mpc|au))?', re.I)
WORD_NUM_RE = re.compile(r'\b(\d+(?:\.\d+)?)\s+(thousand|million|billion)\b', re.I)


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", text.lower())


def content_terms(text: str) -> set[str]:
    return {w.strip("'-") for w in words(text) if len(w.strip("'-")) >= 4 and w.strip("'-") not in STOP}


def canon_num(raw: str) -> str:
    s = raw.lower().strip()
    s = s.replace('about ', '').replace('~', '').replace('≈', '').strip()
    s = re.sub(r'\s*(percent|%)$', '', s)
    s = re.sub(r'\s*(x|×|nm|km|kg/s|kg|g|k|tev|mev|mpc|au)$', '', s)
    s = s.replace(',', '')
    s = re.sub(r'\s+', '', s)
    if '^' in s:
        # Keep both raw scientific and expanded if simple 10^n.
        return s
    return s


def expanded_nums(text: str) -> set[str]:
    vals = set()
    for m in WORD_NUM_RE.finditer(text):
        base = float(m.group(1))
        mult = {'thousand': 1_000, 'million': 1_000_000, 'billion': 1_000_000_000}[m.group(2).lower()]
        val = base * mult
        if abs(val - round(val)) < 1e-9:
            vals.add(str(int(round(val))))
        else:
            vals.add(str(val))
    return vals


def extract_nums(text: str) -> list[str]:
    return [canon_num(m.group(0)) for m in NUM_RE.finditer(text)]


def source_num_set(text: str) -> set[str]:
    nums = set(extract_nums(text)) | expanded_nums(text)
    extra = set()
    for n in list(nums):
        if n.startswith('10^') and n[3:].isdigit():
            extra.add(str(10 ** int(n[3:])))
        if '.' in n:
            extra.add(n.replace('.', ''))  # for detecting decimal compaction as related, not supported
    return nums | extra


def true_source_num_set(text: str) -> set[str]:
    nums = set(extract_nums(text)) | expanded_nums(text)
    extra = set()
    for n in list(nums):
        if n.startswith('10^') and n[3:].isdigit():
            extra.add(str(10 ** int(n[3:])))
    return nums | extra


def local_window(text: str, num: str, radius: int = 80) -> str:
    low = text.lower()
    # Try raw and comma variants.
    patterns = [num, num.replace(',', '')]
    for p in patterns:
        idx = low.find(p.lower())
        if idx >= 0:
            return low[max(0, idx-radius): idx+len(p)+radius]
    return low[:200]


def metric_role_risk(summary: str, source: str, out_nums: list[str]) -> list[str]:
    risks = []
    s_low = summary.lower()
    src_low = source.lower()
    for n in out_nums:
        if n not in true_source_num_set(source):
            continue
        win_s = local_window(s_low, n)
        win_src = local_window(src_low, n)
        if 'cost' in win_s and re.search(r'\b(more|higher|increase|increased)\b', win_s) and re.search(r'\b(smaller|lower|less|reduc|cut|decreas)\b', win_src):
            risks.append(n)
        if re.search(r'\b(lower|less|smaller|reduc|cut)\b', win_s) and re.search(r'\b(higher|more|increase|improv|outperform)\b', win_src):
            risks.append(n)
    return risks


def repeated_content(summary: str) -> bool:
    toks = [w for w in words(summary) if w not in STOP]
    c = Counter(toks)
    if any(v >= 2 for k, v in c.items() if len(k) >= 6):
        return True
    # repeated adjacent bigrams-ish
    low = summary.lower()
    return bool(re.search(r'\b(\w{4,})\b(?:\W+\w+){0,3}\W+\1\b', low))


def screen(row: dict, gen: dict, predpo: dict | None) -> dict:
    source = f"{row.get('title','')}\n{row.get('abstract','')}"
    abstract = row.get('abstract', '')
    summary = gen.get('normalized_output', '')
    out_nums = extract_nums(summary)
    src_true = true_source_num_set(source)
    src_related = source_num_set(source)

    unsupported = [n for n in out_nums if n not in src_true]
    decimal_compaction = []
    for n in unsupported:
        # If output number equals source decimal stripped of punctuation, high-risk corruption.
        for m in re.finditer(r'\d+\.\d+', source):
            if n == m.group(0).replace('.', ''):
                decimal_compaction.append(n)

    metric_risks = metric_role_risk(summary, source, out_nums)

    sum_terms = content_terms(summary)
    src_terms = content_terms(source)
    title_terms = content_terms(row.get('title', ''))
    overlap = len(sum_terms & src_terms)
    unsupported_terms = sorted(t for t in sum_terms if t not in src_terms and t not in title_terms)
    generic_hits = sorted(set(words(summary)) & GENERIC_TERMS)
    polarity_hits = sorted(set(words(summary)) & POLARITY_TERMS)
    wc = len(summary.split())

    semantic_tags = []
    if repeated_content(summary):
        semantic_tags.append('repetition_or_tautology')
    if len(generic_hits) >= 2 and overlap <= 3:
        semantic_tags.append('generic_low_anchor')
    if len(unsupported_terms) >= 3 and overlap <= 4:
        semantic_tags.append('unsupported_terms_low_anchor')
    if row.get('primary_category','').startswith(('math', 'physics')) and overlap <= 3 and any(g in generic_hits for g in ['new','framework','solution','approach','efficient','useful']):
        semantic_tags.append('technical_generic_smoothing')
    if "the paper" in summary.lower() or "paper's" in summary.lower():
        semantic_tags.append('paper_filler_style')
    if any(p in polarity_hits for p in ['not','without','preventing','prevents','but']) and overlap <= 4:
        semantic_tags.append('polarity_smoothing_risk')

    pre_summary = predpo.get('normalized_output') if predpo else ''
    if pre_summary:
        pre_terms = content_terms(pre_summary)
        dropped_pre_terms = sorted((pre_terms & src_terms) - sum_terms)
    else:
        dropped_pre_terms = []
    if pre_summary and len(dropped_pre_terms) >= 4 and len(unsupported_terms) >= 2:
        semantic_tags.append('changed_dropped_source_terms')

    numeric_tags = []
    if decimal_compaction:
        numeric_tags.append('decimal_compaction')
    if metric_risks:
        numeric_tags.append('metric_role_risk')
    if unsupported:
        numeric_tags.append('unsupported_number')
    if out_nums and len(out_nums) >= 3:
        numeric_tags.append('number_dense_summary')

    severity = 'none'
    if 'decimal_compaction' in numeric_tags or 'metric_role_risk' in numeric_tags:
        severity = 'high'
    elif 'unsupported_number' in numeric_tags and any(not (1900 <= int(re.sub(r'\D', '', n) or 0) <= 2030) for n in unsupported):
        severity = 'medium_high'
    elif len(semantic_tags) >= 2:
        severity = 'medium_high'
    elif numeric_tags or semantic_tags:
        severity = 'medium'

    return {
        'eval_id': row.get('eval_id',''),
        'paper_id': row['paper_id'],
        'title': row.get('title',''),
        'primary_category': row.get('primary_category',''),
        'summary': summary,
        'pre_dpo_summary': pre_summary,
        'word_count': wc,
        'numeric_tags': numeric_tags,
        'semantic_tags': sorted(set(semantic_tags)),
        'severity': severity,
        'output_numbers': out_nums,
        'unsupported_numbers': unsupported,
        'decimal_compaction_numbers': decimal_compaction,
        'metric_role_numbers': metric_risks,
        'source_numbers_sample': sorted(src_true)[:30],
        'summary_terms': sorted(sum_terms),
        'source_overlap_terms': sorted(sum_terms & src_terms),
        'unsupported_summary_terms': unsupported_terms[:20],
        'generic_hits': generic_hits,
        'polarity_hits': polarity_hits,
        'dropped_pre_source_terms': dropped_pre_terms[:20],
        'abstract': row.get('abstract',''),
    }


def load_by_id(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out[d['paper_id']] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--generations', required=True)
    ap.add_argument('--pre-dpo-generations', default=None)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    gens = load_by_id(Path(args.generations))
    pre = load_by_id(Path(args.pre_dpo_generations)) if args.pre_dpo_generations else {}
    rows = []
    with open(args.inputs) as f:
        for line in f:
            row = json.loads(line)
            gen = gens.get(row['paper_id'])
            if not gen or not gen.get('ok'):
                continue
            rows.append(screen(row, gen, pre.get(row['paper_id'])))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    screened = out_dir / 'screened_rows.jsonl'
    candidates = out_dir / 'repair_candidates.jsonl'
    numeric = out_dir / 'numeric_candidates.jsonl'
    semantic = out_dir / 'semantic_candidates.jsonl'

    with screened.open('w') as f_all, candidates.open('w') as f_c, numeric.open('w') as f_n, semantic.open('w') as f_s:
        for r in rows:
            f_all.write(json.dumps(r) + '\n')
            is_num = bool(r['numeric_tags'])
            is_sem = bool(r['semantic_tags'])
            if is_num or is_sem:
                f_c.write(json.dumps(r) + '\n')
            if is_num:
                f_n.write(json.dumps(r) + '\n')
            if is_sem:
                f_s.write(json.dumps(r) + '\n')

    tag_counts = Counter()
    sev_counts = Counter(r['severity'] for r in rows)
    num_tag_counts = Counter(t for r in rows for t in r['numeric_tags'])
    sem_tag_counts = Counter(t for r in rows for t in r['semantic_tags'])
    for r in rows:
        if r['numeric_tags']:
            tag_counts['numeric_any'] += 1
        if r['semantic_tags']:
            tag_counts['semantic_any'] += 1
        if r['numeric_tags'] or r['semantic_tags']:
            tag_counts['candidate_any'] += 1

    report = {
        'rows_screened': len(rows),
        'candidate_counts': dict(tag_counts),
        'severity_counts': dict(sev_counts),
        'numeric_tag_counts': dict(num_tag_counts),
        'semantic_tag_counts': dict(sem_tag_counts),
        'files': {
            'screened_rows': str(screened),
            'repair_candidates': str(candidates),
            'numeric_candidates': str(numeric),
            'semantic_candidates': str(semantic),
        }
    }
    (out_dir / 'heuristic_report.json').write_text(json.dumps(report, indent=2) + '\n')
    md = [
        '# Checkpoint 700 10k Failure Mining Heuristic Report', '',
        f'- Rows screened: {len(rows)}',
        f'- Any repair candidate: {tag_counts["candidate_any"]}',
        f'- Numeric candidates: {tag_counts["numeric_any"]}',
        f'- Semantic-smoothing candidates: {tag_counts["semantic_any"]}', '',
        '## Severity counts', '',
    ]
    for k,v in sev_counts.most_common():
        md.append(f'- `{k}`: {v}')
    md += ['', '## Numeric tag counts', '']
    for k,v in num_tag_counts.most_common():
        md.append(f'- `{k}`: {v}')
    md += ['', '## Semantic tag counts', '']
    for k,v in sem_tag_counts.most_common():
        md.append(f'- `{k}`: {v}')
    md += ['', '## Files', '']
    for k,v in report['files'].items():
        md.append(f'- {k}: `{v}`')
    (out_dir / 'heuristic_report.md').write_text('\n'.join(md) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
