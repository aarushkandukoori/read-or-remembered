"""Tables for RQ1 fact provenance (called by: rq1_fact_provenance.py summarize).

Reads outputs/rq1/facts_{swebv,deepswe}.jsonl (one row per fact, first mention per trajectory)
and the per-trajectory list in the cache folder, and writes:
  outputs/rq1/facts_summary.json      all headline numbers
  outputs/rq1/fig_nbs_rate_accuracy.csv   named-before-seen rate and accuracy, per group and fact type
  outputs/rq1/fig_wrong_breakdown.csv     wrong named-before-seen facts: stale / future / never / signature kinds
  outputs/rq1/fig_self_correction.csv     contradicted later? changed course? steps until contradiction
  outputs/rq1/fig_resolved_assoc.csv      resolve rate of trajectories with vs without a wrong recalled fact
  outputs/rq1/fig_first_field.csv         where named-before-seen facts first appear (reasoning / text / command)
Groups: model; reasoning visible vs hidden; open vs closed weights; (DeepSWE) source and language.
"""
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict

FACT_TYPES = ['path', 'module', 'symbol', 'code_name', 'sig_def', 'sig_call', 'kwarg']
RIGHT = ('real', 'match', 'valid')


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n (None if n == 0)."""
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - r) / d, 4), round((c + r) / d, 4)


def rate(k, n):
    return round(k / n, 4) if n else None


def load(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def model_visibility(trajs):
    """Model-level label: 'visible' if most of its trajectories carry reasoning text."""
    by = defaultdict(list)
    for t in trajs:
        by[t['model']].append(bool(t.get('reasoning_visible')))
    return {m: ('visible' if sum(v) / len(v) >= 0.5 else 'hidden') for m, v in by.items()}


def group_keys(row, vis):
    """The groups a row (or trajectory) belongs to."""
    keys = [('all', 'all'), ('model', row['model']), ('reasoning', vis[row['model']]),
            ('weights', row['weights'])]
    if row['bench'] == 'deepswe':
        keys += [('source', row['source']), ('language', row['lang'])]
    return keys


def nbs_table(rows, trajs, vis, bench):
    n_traj = Counter()
    for t in trajs:
        for g in group_keys(t, vis):
            n_traj[g] += 1
    cells = defaultdict(Counter)
    for r in rows:
        for g in group_keys(r, vis):
            for ft in (r['fact_type'], 'all'):
                c = cells[g + (ft,)]
                c['mentions'] += 1
                if r['nbs']:
                    c['nbs'] += 1
                    c['nbs_right'] += r['truth'] in RIGHT
                else:
                    c['seen_right'] += r['truth'] in RIGHT
    out = []
    for (gt, g, ft), c in sorted(cells.items(), key=lambda kv: [str(x) for x in kv[0]]):
        lo, hi = wilson(c['nbs_right'], c['nbs'])
        out.append({'bench': bench, 'group_type': gt, 'group': g, 'fact_type': ft,
                    'n_trajectories': n_traj[(gt, g)], 'n_mentions': c['mentions'], 'n_nbs': c['nbs'],
                    'nbs_rate': rate(c['nbs'], c['mentions']),
                    'nbs_per_trajectory': round(c['nbs'] / n_traj[(gt, g)], 3) if n_traj[(gt, g)] else None,
                    'nbs_accuracy': rate(c['nbs_right'], c['nbs']), 'nbs_accuracy_lo': lo, 'nbs_accuracy_hi': hi,
                    'seen_accuracy': rate(c['seen_right'], c['mentions'] - c['nbs'])})
    return out


def wrong_table(rows, vis, bench):
    cells = defaultdict(Counter)
    for r in rows:
        if not (r['nbs'] and r['wrong']):
            continue
        for gt, g in group_keys(r, vis):
            if gt not in ('all', 'model', 'reasoning', 'weights'):
                continue
            for ft in (r['fact_type'], 'all'):
                c = cells[(gt, g, ft)]
                c['wrong'] += 1
                kind = r['truth'] if r['truth'] in ('stale', 'future', 'never') else r['truth']
                c[kind] += 1
                if r.get('other_version_match') in ('stale', 'future'):
                    c['sig_or_kwarg_matches_' + r['other_version_match']] += 1
    out = []
    for (gt, g, ft), c in sorted(cells.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = {'bench': bench, 'group_type': gt, 'group': g, 'fact_type': ft, 'n_wrong_nbs': c['wrong']}
        for k in ('stale', 'future', 'never', 'order_wrong', 'names_differ', 'invalid',
                  'sig_or_kwarg_matches_stale', 'sig_or_kwarg_matches_future'):
            row[k] = c[k]
        for k in ('stale', 'future', 'never'):
            row[k + '_share'] = rate(c[k], c['wrong'])
        out.append(row)
    return out


def correction_table(rows, vis, bench):
    cells = defaultdict(list)
    for r in rows:
        if r['nbs'] and r['wrong']:
            for gt, g in group_keys(r, vis):
                if gt in ('all', 'model', 'reasoning', 'weights'):
                    cells[(gt, g, r['fact_type'])].append(r)
                    cells[(gt, g, 'all')].append(r)
    out = []
    for (gt, g, ft), rs in sorted(cells.items(), key=lambda kv: [str(x) for x in kv[0]]):
        n = len(rs)
        con = [r for r in rs if r.get('contradicted_step')]
        changed = sum(bool(r.get('changed_course')) for r in rs)
        steps = [r['steps_to_contradiction'] for r in con]
        out.append({'bench': bench, 'group_type': gt, 'group': g, 'fact_type': ft, 'n_wrong_nbs': n,
                    'contradicted': len(con), 'contradicted_share': rate(len(con), n),
                    'changed_course': changed, 'changed_course_share': rate(changed, n),
                    'median_steps_to_contradiction': statistics.median(steps) if steps else None,
                    'contradicted_same_step': sum(s == 0 for s in steps),
                    'never_contradicted': n - len(con),
                    'in_final_patch': sum(bool(r.get('in_final_patch')) for r in rs),
                    'never_contradicted_and_in_patch': sum(bool(r.get('in_final_patch')) and not r.get('contradicted_step')
                                                           for r in rs)})
    return out


def resolved_table(rows, trajs, vis, bench):
    """Association only: resolve rate of trajectories that contain at least one wrong
    named-before-seen fact vs those that contain none."""
    wrong_any, wrong_unfixed = set(), set()
    for r in rows:
        if r['nbs'] and r['wrong']:
            wrong_any.add(r['trial'])
            if not r.get('changed_course'):
                wrong_unfixed.add(r['trial'])
    cells = defaultdict(lambda: defaultdict(list))
    for t in trajs:
        if t.get('resolved') is None:
            continue
        for gt, g in group_keys(t, vis):
            if gt not in ('all', 'model', 'reasoning', 'weights'):
                continue
            cells[(gt, g)]['with' if t['trial'] in wrong_any else 'without'].append(bool(t['resolved']))
            cells[(gt, g)]['unfixed' if t['trial'] in wrong_unfixed else 'no_unfixed'].append(bool(t['resolved']))
    out = []
    for (gt, g), d in sorted(cells.items()):
        w, wo = d['with'], d['without']
        row = {'bench': bench, 'group_type': gt, 'group': g,
               'n_with_wrong_nbs': len(w), 'resolved_with': rate(sum(w), len(w)),
               'n_without': len(wo), 'resolved_without': rate(sum(wo), len(wo)),
               'n_with_unfixed_wrong_nbs': len(d['unfixed']), 'resolved_with_unfixed': rate(sum(d['unfixed']),
                                                                                            len(d['unfixed'])),
               'n_without_unfixed': len(d['no_unfixed']), 'resolved_without_unfixed': rate(sum(d['no_unfixed']),
                                                                                           len(d['no_unfixed']))}
        row['difference_with_minus_without'] = (round(row['resolved_with'] - row['resolved_without'], 4)
                                                if w and wo else None)
        out.append(row)
    return out


def field_table(rows, vis, bench):
    cells = defaultdict(Counter)
    for r in rows:
        if not r['nbs']:
            continue
        for gt, g in group_keys(r, vis):
            if gt in ('all', 'model', 'reasoning', 'weights'):
                cells[(gt, g)][r['field']] += 1
                cells[(gt, g)]['n'] += 1
                if r.get('hedged') is not None:
                    cells[(gt, g)]['text_mentions'] += 1
                    cells[(gt, g)]['hedged'] += bool(r['hedged'])
    out = []
    for (gt, g), c in sorted(cells.items()):
        out.append({'bench': bench, 'group_type': gt, 'group': g, 'n_nbs': c['n'],
                    'first_in_reasoning': c['reasoning'], 'first_in_text': c['visible'], 'first_in_command': c['command'],
                    'share_reasoning': rate(c['reasoning'], c['n']), 'share_text': rate(c['visible'], c['n']),
                    'share_command': rate(c['command'], c['n']),
                    'hedged_share_of_text_mentions': rate(c['hedged'], c['text_mentions'])})
    return out


def traj_table(rows, trajs, vis):
    """One row per trajectory: groups, resolved, and counts of facts / NBS facts / wrong NBS facts."""
    counts = defaultdict(Counter)
    for r in rows:
        c = counts[r['trial']]
        c['facts'] += 1
        c['nbs'] += r['nbs']
        c['wrong_nbs'] += bool(r['nbs'] and r['wrong'])
        c['wrong_nbs_uncorrected'] += bool(r['nbs'] and r['wrong'] and not r.get('changed_course'))
    out = []
    for t in trajs:
        c = counts[t['trial']]
        out.append({'bench': t['bench'], 'source': t['source'], 'model': t['model'], 'trial': t['trial'],
                    'task': t['task'], 'repo': t['repo'], 'lang': t['lang'], 'weights': t['weights'],
                    'reasoning_model_level': vis[t['model']], 'reasoning_in_trajectory': t.get('reasoning_visible'),
                    'effort': t.get('effort'), 'resolved': t.get('resolved'), 'n_steps': t['n_steps'],
                    'n_facts': c['facts'], 'n_nbs': c['nbs'], 'n_wrong_nbs': c['wrong_nbs'],
                    'n_wrong_nbs_uncorrected': c['wrong_nbs_uncorrected']})
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print('wrote', path, len(rows), 'rows')


def main(out_dir, cache_dir):
    summary = {'note': 'All numbers computed by scripts/rq1_fact_provenance.py + rq1_summarize.py; '
                       'resolve-rate comparisons are associations, not causal effects.'}
    tables = defaultdict(list)
    for bench in ('swebv', 'deepswe'):
        all_rows = load(os.path.join(out_dir, f'facts_{bench}.jsonl'))
        meta = {t['trial']: t for t in load(os.path.join(cache_dir, f'trajectories_{bench}.jsonl')) if 'skipped' not in t}
        for r in all_rows:  # join the per-trajectory fields (bench, source, lang, weights, resolved, ...)
            m = meta.get(r['trial'], {})
            for k in ('bench', 'source', 'task', 'repo', 'lang', 'resolved', 'reasoning_visible', 'weights', 'n_steps'):
                r.setdefault(k, m.get(k))
        # not judged: 'external' code names (library names), 'planned' mentions (the agent says it will
        # create the thing), 'outside' paths (under a folder this repo never had)
        rows = [r for r in all_rows if r.get('wrong') is not None]
        external = [r for r in all_rows if r.get('wrong') is None]
        trajs = [t for t in load(os.path.join(cache_dir, f'trajectories_{bench}.jsonl')) if 'skipped' not in t]
        skipped = [t for t in load(os.path.join(cache_dir, f'trajectories_{bench}.jsonl')) if 'skipped' in t]
        if not trajs:
            continue
        vis = model_visibility(trajs)
        nbs = nbs_table(rows, trajs, vis, bench)
        wrong = wrong_table(rows, vis, bench)
        corr = correction_table(rows, vis, bench)
        res = resolved_table(rows, trajs, vis, bench)
        fld = field_table(rows, vis, bench)
        for name, t in (('nbs_rate_accuracy', nbs), ('wrong_breakdown', wrong), ('self_correction', corr),
                        ('resolved_assoc', res), ('first_field', fld), ('trajectories', traj_table(rows, trajs, vis))):
            tables[name].extend(t)
        summary[bench] = {
            'n_trajectories': len(trajs), 'n_skipped': len(skipped),
            'skipped_reasons': dict(Counter(t['skipped'].split(':')[0] for t in skipped)),
            'n_fact_rows': len(all_rows), 'n_judged_rows': len(rows), 'n_unjudged_rows': len(external),
            'unjudged_by_kind': dict(Counter(r['truth'] for r in external)),
            'unjudged_nbs_by_kind': dict(Counter(r['truth'] for r in external if r['nbs'])),
            'models': sorted(vis), 'model_reasoning': vis,
            'fact_types': dict(Counter(r['fact_type'] for r in rows)),
            'overall_by_fact_type': {r['fact_type']: {k: r[k] for k in ('n_mentions', 'n_nbs', 'nbs_rate',
                                                                       'nbs_accuracy', 'nbs_accuracy_lo',
                                                                       'nbs_accuracy_hi', 'seen_accuracy')}
                                     for r in nbs if r['group_type'] == 'all'},
            'by_reasoning': [r for r in nbs if r['group_type'] == 'reasoning' and r['fact_type'] == 'all'],
            'by_weights': [r for r in nbs if r['group_type'] == 'weights' and r['fact_type'] == 'all'],
            'wrong_breakdown_all': [r for r in wrong if r['group_type'] == 'all'],
            'self_correction_all': [r for r in corr if r['group_type'] == 'all'],
            'resolved_assoc_all': [r for r in res if r['group_type'] == 'all'],
            'first_field_all': [r for r in fld if r['group_type'] == 'all'],
            'mean_base_coverage': round(statistics.mean([t['base_coverage'] for t in trajs
                                                         if t.get('base_coverage') is not None]), 4),
        }
    for name, t in tables.items():
        write_csv(os.path.join(out_dir, f'fig_{name}.csv'), t)
    hc = os.path.join(cache_dir, 'handcheck_judgments.json')
    if os.path.exists(hc):
        with open(hc) as f:
            summary['hand_check'] = json.load(f)
    with open(os.path.join(out_dir, 'facts_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print('wrote', os.path.join(out_dir, 'facts_summary.json'))
