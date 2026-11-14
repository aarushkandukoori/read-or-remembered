"""Where do the 257 human-confirmed Misstatement/Misreference claims come from (RQ1 provenance)?

Reads the claims and our deterministic features from outputs/provenance/batches/*.jsonl and the
LLM-annotated provenance labels from outputs/provenance/labels_A/*.jsonl (annotator A, all claims) and
outputs/provenance/labels_B/*.jsonl (an independent second annotator on a subset). Reports the class
distribution (counts, shares, 95% Wilson intervals), the split by model and by error form, flag rates,
the cross-tab of class x evidence_seen_before, and A-vs-B agreement on the claims both labeled.
Writes outputs/provenance/summary.json and prints Markdown tables (pasted into notes/provenance.md).
Usage: .venv/bin/python -I scripts/provenance_summary.py
"""
import collections as C
import glob
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / 'outputs/provenance'
CLASSES = ['Propagated', 'Recalled-other-version', 'Convention-guess', 'Read-but-misread', 'Unsupported']
FLAGS = ['contradicted_by_prior_output', 'named_before_seen', 'hedged', 'uses_upstream_fix']
MODELS = {'deepseek-v32': 'DeepSeek-V3.2', 'flash': 'DeepSeek-V4.1-Flash'}
FORMS = ['Misstatement', 'Misreference']
SEEN = ['all', 'some', 'none', 'no_repo_evidence']
MEMORY = {'Recalled-other-version'}   # memory-sourced class for the binary agreement


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 6), round(c + h, 6)]


def kappa(pairs, cats):
    """Cohen's kappa for (label_a, label_b) pairs; None if chance agreement is 1."""
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = C.Counter(a for a, _ in pairs), C.Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def boot_kappa(pairs, cats, reps=2000, seed=0):
    """Bootstrap 95% interval for kappa, resampling claims (undefined resamples are skipped)."""
    rng = random.Random(seed)
    ks = [kappa([rng.choice(pairs) for _ in pairs], cats) for _ in range(reps)]
    ks = sorted(k for k in ks if k is not None)
    return [round(ks[int(0.025 * len(ks))], 3), round(ks[int(0.975 * len(ks)) - 1], 3)], reps - len(ks)


def load_labels(who):
    """cid -> label dict for one annotator, plus problems found while reading."""
    labels, problems = {}, []
    for f in sorted(glob.glob(str(P / f'labels_{who}/*.jsonl'))):
        for i, line in enumerate(open(f), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f'{Path(f).name}:{i} bad JSON')
                continue
            if d.get('provenance') not in CLASSES:
                problems.append(f"{Path(f).name}:{i} unknown class {d.get('provenance')!r}")
                continue
            if d['cid'] in labels:
                problems.append(f"{Path(f).name}:{i} duplicate cid {d['cid']} (first label kept)")
                continue
            labels[d['cid']] = d
    return labels, problems


def dist(labels, cids):
    """Count, share and Wilson interval per class over the given claims."""
    n = len(cids)
    cnt = C.Counter(labels[c]['provenance'] for c in cids)
    return {'n': n, 'classes': {k: {'count': cnt[k], 'share': round(cnt[k] / n, 6) if n else None,
                                    'wilson95': wilson(cnt[k], n)} for k in CLASSES}}


def flag_rates(labels, cids):
    n = len(cids)
    out = {f: {'count': sum(bool(labels[c][f]) for c in cids), 'n': n} for f in FLAGS}
    for f in FLAGS:
        out[f]['share'] = round(out[f]['count'] / n, 6) if n else None
        out[f]['wilson95'] = wilson(out[f]['count'], n)
    out['channel'] = dict(C.Counter(labels[c]['channel'] for c in cids).most_common())
    out['confidence'] = dict(C.Counter(labels[c]['confidence'] for c in cids).most_common())
    return out


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    items = {}
    batch_of = {}
    for f in sorted(glob.glob(str(P / 'batches/*.jsonl'))):
        for line in open(f):
            if line.strip():
                d = json.loads(line)
                items[d['cid']] = d
                batch_of[d['cid']] = Path(f).stem
    A, probs_a = load_labels('A')
    B, probs_b = load_labels('B')
    stray = sorted(set(A) - set(items)) + sorted(set(B) - set(items))
    missing_a = sorted(set(items) - set(A))
    cids = sorted(c for c in items if c in A)

    out = C.OrderedDict()
    out['command'] = '.venv/bin/python -I scripts/provenance_summary.py'
    out['labels'] = 'LLM-annotated (Claude Opus 5.5 subagents); annotator A on all claims, B independent on a subset'
    out['coverage'] = {'n_claims': len(items), 'n_labeled_A': len(cids), 'missing_A': missing_a,
                       'missing_A_by_batch': dict(C.Counter(batch_of[c] for c in missing_a)),
                       'n_labeled_B': len(B), 'B_batches': sorted({batch_of[c] for c in B if c in items}),
                       'labels_not_in_batches': stray, 'problems_A': probs_a, 'problems_B': probs_b}
    out['all'] = dist(A, cids)
    out['by_model'] = {m: dist(A, [c for c in cids if items[c]['model'] == m]) for m in MODELS}
    out['by_error_form'] = {f: dist(A, [c for c in cids if items[c]['error_form'] == f]) for f in FORMS}
    out['flags'] = flag_rates(A, cids)
    out['flags_by_class'] = {k: flag_rates(A, [c for c in cids if A[c]['provenance'] == k]) for k in CLASSES}
    out['flags_by_error_form_and_class'] = {
        f: {k: flag_rates(A, [c for c in cids if items[c]['error_form'] == f and A[c]['provenance'] == k])
            for k in CLASSES} for f in FORMS}
    out['crosstab_class_x_evidence_seen_before'] = {
        k: {s: sum(1 for c in cids if A[c]['provenance'] == k and items[c]['evidence_seen_before'] == s)
            for s in SEEN} for k in CLASSES}
    out['crosstab_evidence_seen_before_totals'] = {s: sum(items[c]['evidence_seen_before'] == s for c in cids)
                                                   for s in SEEN}

    both = sorted(c for c in B if c in A)
    agree = {'n_overlap': len(both)}
    if both:
        pairs = [(A[c]['provenance'], B[c]['provenance']) for c in both]
        k5 = kappa(pairs, CLASSES)
        ci5, skip5 = boot_kappa(pairs, CLASSES)
        bin_pairs = [(a in MEMORY, b in MEMORY) for a, b in pairs]
        kb = kappa(bin_pairs, [True, False])
        cib, skipb = boot_kappa(bin_pairs, [True, False])
        agree.update({
            'raw_agreement': round(sum(a == b for a, b in pairs) / len(pairs), 4),
            'raw_agreement_wilson95': wilson(sum(a == b for a, b in pairs), len(pairs)),
            'n_agree': sum(a == b for a, b in pairs),
            'kappa_5_classes': round(k5, 3) if k5 is not None else None,
            'kappa_5_classes_boot95': ci5, 'boot_reps_skipped_5': skip5,
            'memory_vs_not': {'definition': 'Recalled-other-version vs the other four classes',
                              'raw_agreement': round(sum(a == b for a, b in bin_pairs) / len(bin_pairs), 4),
                              'kappa': round(kb, 3) if kb is not None else None,
                              'kappa_boot95': cib, 'boot_reps_skipped': skipb,
                              'A_memory': sum(a for a, _ in bin_pairs), 'B_memory': sum(b for _, b in bin_pairs),
                              'both_memory': sum(a and b for a, b in bin_pairs)},
            'A_classes_on_overlap': {k: sum(a == k for a, _ in pairs) for k in CLASSES},
            'B_classes_on_overlap': {k: sum(b == k for _, b in pairs) for k in CLASSES},
            'confusion_A_rows_B_cols': {ka: {kb_: sum(1 for a, b in pairs if a == ka and b == kb_)
                                             for kb_ in CLASSES} for ka in CLASSES},
            'flag_agreement': {f: round(sum(bool(A[c][f]) == bool(B[c][f]) for c in both) / len(both), 4)
                               for f in FLAGS},
            'disagreements': [{'cid': c, 'A': A[c]['provenance'], 'B': B[c]['provenance']}
                              for c in both if A[c]['provenance'] != B[c]['provenance']]})
    out['agreement_A_vs_B'] = agree
    (P / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')

    # ---- Markdown tables for notes/provenance.md ----
    cov = out['coverage']
    print(f"Coverage: {cov['n_labeled_A']} of {cov['n_claims']} claims labeled by A; missing {len(missing_a)} "
          f"{cov['missing_A_by_batch'] or ''}; B labeled {cov['n_labeled_B']} ({', '.join(cov['B_batches'])}); "
          f"problems A {len(probs_a)}, B {len(probs_b)}; stray labels {len(stray)}\n")
    cols = [('All', out['all'])] + [(MODELS[m], out['by_model'][m]) for m in MODELS] + \
           [(f, out['by_error_form'][f]) for f in FORMS]
    print('| class | ' + ' | '.join(f"{name} (n={d['n']})" for name, d in cols) + ' |')
    print('|---' * (len(cols) + 1) + '|')
    for k in CLASSES:
        cells = []
        for name, d in cols:
            x = d['classes'][k]
            cells.append(f"{x['count']} ({pct(x['share'])})" if d['n'] else '-')
        print(f'| {k} | ' + ' | '.join(cells) + ' |')
    print('\n95% Wilson intervals (all claims): ' + '; '.join(
        f"{k} {pct(out['all']['classes'][k]['wilson95'][0])}-{pct(out['all']['classes'][k]['wilson95'][1])}"
        for k in CLASSES))
    fl = out['flags']
    print('\n| flag (annotator A) | count | share | 95% CI |\n|---|---|---|---|')
    for f in FLAGS:
        print(f"| {f} | {fl[f]['count']}/{fl[f]['n']} | {pct(fl[f]['share'])} | "
              f"{pct(fl[f]['wilson95'][0])}-{pct(fl[f]['wilson95'][1])} |")
    print(f"| channel | {fl['channel']} | | |\n| confidence | {fl['confidence']} | | |")
    print('\n| class | ' + ' | '.join(FLAGS) + ' |\n|---' * 1 + '|---' * len(FLAGS) + '|')
    for k in CLASSES:
        fk = out['flags_by_class'][k]
        print(f'| {k} (n={fk[FLAGS[0]]["n"]}) | ' + ' | '.join(f"{fk[f]['count']}" for f in FLAGS) + ' |')
    print('\n| class x evidence_seen_before | ' + ' | '.join(SEEN) + ' |\n|---' + '|---' * len(SEEN) + '|')
    for k in CLASSES:
        print(f'| {k} | ' + ' | '.join(str(out['crosstab_class_x_evidence_seen_before'][k][s]) for s in SEEN) + ' |')
    print('| total | ' + ' | '.join(str(out['crosstab_evidence_seen_before_totals'][s]) for s in SEEN) + ' |')
    if both:
        m = agree['memory_vs_not']
        print(f"\nAgreement A vs B on {len(both)} claims: raw {pct(agree['raw_agreement'])} ({agree['n_agree']}/"
              f"{len(both)}), kappa (5 classes) {agree['kappa_5_classes']} (bootstrap 95% CI "
              f"{agree['kappa_5_classes_boot95'][0]} to {agree['kappa_5_classes_boot95'][1]}); memory vs not: raw "
              f"{pct(m['raw_agreement'])}, kappa {m['kappa']} (CI {m['kappa_boot95'][0]} to {m['kappa_boot95'][1]}, "
              f"{m['boot_reps_skipped']} undefined resamples skipped); A memory {m['A_memory']}, B memory "
              f"{m['B_memory']}, both {m['both_memory']}")
        print('\n| A \\ B | ' + ' | '.join(CLASSES) + ' |\n|---' + '|---' * len(CLASSES) + '|')
        for ka in CLASSES:
            print(f'| {ka} | ' + ' | '.join(str(agree['confusion_A_rows_B_cols'][ka][kb_]) for kb_ in CLASSES) + ' |')
        print('\nFlag agreement A vs B: ' + ', '.join(f'{f} {pct(v)}' for f, v in agree['flag_agreement'].items()))
    else:
        print('\nNo overlap between annotators A and B yet: agreement not computed.')
    print(f"\nwrote {(P / 'summary.json').relative_to(ROOT)}")


if __name__ == '__main__':
    main()
