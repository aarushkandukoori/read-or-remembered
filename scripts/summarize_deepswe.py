"""Summarize the DeepSWE data we downloaded with fetch_deepswe.py.

Two levels:
  1. The index (trials.json): every trial Datacurve published, per model + effort config.
  2. The sample (trajectory files on disk): how long trajectories are and, the key question
     for us, whether each step carries reasoning text ("reasoning_content") we can mine
     for claims, plus visible message text and the bash commands.

Usage (pandas is only needed for the optional SWE-bench Verified file):
  .venv/bin/python -I scripts/summarize_deepswe.py data/deepswe outputs/deepswe \
      data/meta/swebv/data/test-00000-of-00001.parquet

Writes CSV files to the output folder:
  index_by_config.csv   one row per (version, slice, source, model, effort) in the index
  sample_trials.csv     one row per downloaded trajectory
  sample_by_config.csv  sample_trials averaged per config
and prints the main tables.
"""
import csv
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def iter_steps(traj):
    """Yield one simple dict per agent step of a DeepSWE (Harbor ATIF) trajectory:
    reasoning text, visible message text, the bash commands, and their outputs.
    Reusable by later provenance scripts."""
    for s in traj.get('steps', []):
        if s.get('source') != 'agent':
            continue
        calls = s.get('tool_calls') or []
        results = (s.get('observation') or {}).get('results') or []
        msg = s.get('message')
        yield {
            'step_id': s.get('step_id'),
            'reasoning': (s.get('reasoning_content') or '').strip(),
            'message': msg.strip() if isinstance(msg, str) else '',
            'tools': [c.get('function_name') for c in calls],
            'commands': [(c.get('arguments') or {}).get('command', '') for c in calls],
            'outputs': [r.get('content') if isinstance(r.get('content'), str) else json.dumps(r.get('content'))
                        for r in results],
        }


def index_table(data_dir):
    """Per-config numbers straight from the published index files."""
    out = []
    for version in ['v1.1', 'v1']:
        with open(os.path.join(data_dir, 'meta', version, 'trials.json')) as f:
            rows = json.load(f)['rows']
        groups = defaultdict(list)
        for r in rows:
            groups[(version, r['eval_scope'], r['source'], r['model'], r.get('reasoning_effort'))].append(r)
        for key, rs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
            scored = [r for r in rs if r.get('included_in_score')]
            out.append({
                'version': key[0], 'scope': key[1], 'source': key[2], 'model': key[3],
                'reasoning_effort': key[4], 'harness': rs[0].get('harness'),
                'provider': Counter(r.get('provider') for r in rs).most_common(1)[0][0],
                'n_trials': len(rs), 'n_tasks': len({r['task_name'] for r in rs}),
                'n_with_trajectory': sum(bool(r.get('has_trajectory')) for r in rs),
                'n_scored': len(scored),
                'pass_rate': round(sum(bool(r['passed']) for r in scored) / len(scored), 4) if scored else None,
                'mean_steps': mean([r.get('n_agent_steps') for r in rs]),
                'mean_cost_usd': mean([r.get('cost_usd') for r in rs]),
                'mean_output_tokens': mean([r.get('n_output_tokens') for r in rs]),
                'n_with_reasoning_token_count': sum(r.get('n_reasoning_tokens') is not None for r in rs),
            })
    return out


def sample_table(data_dir):
    """Per-trajectory numbers for every trial listed in sample_manifest.csv."""
    with open(os.path.join(data_dir, 'sample_manifest.csv')) as f:
        manifest = list(csv.DictReader(f))
    out = []
    for m in manifest:
        folder = os.path.join(data_dir, 'trials', m['version'], m['trial_name'])
        path = os.path.join(folder, 'trajectory.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            traj = json.load(f)
        steps = list(iter_steps(traj))
        n = len(steps) or 1
        reward = {}
        if os.path.exists(os.path.join(folder, 'reward.json')):
            with open(os.path.join(folder, 'reward.json')) as f:
                reward = json.load(f)
        fm = traj.get('final_metrics') or {}
        out.append({
            'version': m['version'], 'scope': m['scope'], 'source': m['source'], 'model': m['model'],
            'reasoning_effort': m['reasoning_effort'], 'task_name': m['task_name'],
            'trial_name': m['trial_name'], 'passed': m['passed'],
            'schema_version': traj.get('schema_version'),
            'agent': f"{traj['agent'].get('name')} {traj['agent'].get('version')}",
            'model_name': traj['agent'].get('model_name'),
            'agent_steps': len(steps),
            'tool_calls': sum(len(s['tools']) for s in steps),
            'tool_names': ';'.join(sorted({t for s in steps for t in s['tools'] if t})),
            'frac_steps_with_reasoning': round(sum(bool(s['reasoning']) for s in steps) / n, 3),
            'reasoning_chars': sum(len(s['reasoning']) for s in steps),
            # some harness versions copy the visible message into reasoning_content
            # (seen for gemini-3-7-flash): then there is no separate reasoning text
            'steps_reasoning_equals_message': sum(bool(s['reasoning']) and s['reasoning'] == s['message']
                                                  for s in steps),
            'frac_steps_with_message': round(sum(bool(s['message']) for s in steps) / n, 3),
            'message_chars': sum(len(s['message']) for s in steps),
            'total_reasoning_tokens': (fm.get('extra') or {}).get('total_reasoning_tokens'),
            'total_cost_usd': fm.get('total_cost_usd'),
            'patch_bytes': os.path.getsize(os.path.join(folder, 'model.patch'))
            if os.path.exists(os.path.join(folder, 'model.patch')) else 0,
            'reward': reward.get('reward'), 'f2p': reward.get('f2p'), 'p2p': reward.get('p2p'),
        })
    return out


def by_config(sample):
    groups = defaultdict(list)
    for r in sample:
        groups[(r['version'], r['scope'], r['source'], r['model'], r['reasoning_effort'])].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        out.append({
            'version': key[0], 'scope': key[1], 'source': key[2], 'model': key[3],
            'reasoning_effort': key[4], 'n_trajectories': len(rs),
            'mean_agent_steps': mean([r['agent_steps'] for r in rs]),
            'share_trials_with_any_reasoning': round(sum(r['reasoning_chars'] > 0 for r in rs) / len(rs), 3),
            'mean_frac_steps_with_reasoning': mean([r['frac_steps_with_reasoning'] for r in rs]),
            'mean_reasoning_chars_per_trial': mean([r['reasoning_chars'] for r in rs]),
            'share_reasoning_steps_that_copy_message': round(
                sum(r['steps_reasoning_equals_message'] for r in rs)
                / max(1, sum(round(r['frac_steps_with_reasoning'] * r['agent_steps']) for r in rs)), 3),
            'mean_frac_steps_with_message': mean([r['frac_steps_with_message'] for r in rs]),
            'mean_message_chars_per_trial': mean([r['message_chars'] for r in rs]),
            # rough hint of raw vs summarized reasoning: raw text runs about 3-4 characters
            # per token; far fewer characters than tokens means we only got a summary
            'reasoning_chars_per_reasoning_token': round(
                sum(r['reasoning_chars'] for r in rs if r['total_reasoning_tokens'])
                / sum(r['total_reasoning_tokens'] for r in rs if r['total_reasoning_tokens']), 2)
            if any(r['total_reasoning_tokens'] for r in rs) else None,
            'pass_rate_in_sample': round(sum(r['passed'] == 'True' for r in rs) / len(rs), 3),
            'agent_versions': ';'.join(sorted({r['agent'] for r in rs})),
        })
    return out


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('wrote', path, len(rows), 'rows')


def repo_overlap(data_dir, swebv_parquet):
    """Which DeepSWE / SWE-bench Pro (cross-bench) repos are also SWE-bench Verified repos?"""
    with open(os.path.join(data_dir, 'meta', 'v1.1', 'tasks.json')) as f:
        deepswe = {t['repository'].lower() for t in json.load(f)['rows']}
    with open(os.path.join(data_dir, 'meta', 'v1', 'trials.json')) as f:
        rows = json.load(f)['rows']
    # SWE-bench Pro task names look like instance_<org>__<repo>-<40-char commit>[-v...]
    pro = set()
    for r in rows:
        if r['source'] == 'swebenchpro':
            m = re.match(r'instance_(.+?)__(.+?)-[0-9a-f]{40}', r['task_name'])
            pro.add(f'{m.group(1)}/{m.group(2)}'.lower())
    import pandas as pd  # only needed here
    verified = {r.lower() for r in pd.read_parquet(swebv_parquet, columns=['repo'])['repo']}
    print(f'\nSWE-bench Verified repos ({len(verified)}):', sorted(verified))
    print(f'DeepSWE repos: {len(deepswe)}; overlap with Verified:', sorted(deepswe & verified) or 'none')
    print(f'SWE-bench Pro repos in cross-bench sample ({len(pro)}):', sorted(pro))
    print('  overlap with Verified:', sorted(pro & verified) or 'none')


def main(data_dir, out_dir, swebv_parquet=None):
    os.makedirs(out_dir, exist_ok=True)
    index = index_table(data_dir)
    write_csv(os.path.join(out_dir, 'index_by_config.csv'), index)
    sample = sample_table(data_dir)
    write_csv(os.path.join(out_dir, 'sample_trials.csv'), sample)
    per_config = by_config(sample)
    write_csv(os.path.join(out_dir, 'sample_by_config.csv'), per_config)

    print('\nPer model (index, all versions/slices):')
    per_model = defaultdict(lambda: Counter())
    for r in index:
        k = (r['version'], r['scope'], r['source'], r['model'])
        per_model[k]['trials'] += r['n_trials']
        per_model[k]['traj'] += r['n_with_trajectory']
        per_model[k]['configs'] += 1
    for k, c in sorted(per_model.items()):
        print(f"  {k[0]:5} {k[1]:11} {k[2]:11} {k[3]:24} configs={c['configs']} trials={c['trials']} with_traj={c['traj']}")

    print('\nReasoning text in the sample, per model (all configs pooled):')
    pooled = defaultdict(list)
    for r in sample:
        pooled[(r['version'], r['scope'], r['source'], r['model'])].append(r)
    for k, rs in sorted(pooled.items()):
        print(f"  {k[0]:5} {k[1]:11} {k[2]:11} {k[3]:24} n={len(rs):3} steps={mean([r['agent_steps'] for r in rs]):7} "
              f"reasoning_steps={mean([r['frac_steps_with_reasoning'] for r in rs]):5} "
              f"reasoning_chars/trial={mean([r['reasoning_chars'] for r in rs]):9} "
              f"message_steps={mean([r['frac_steps_with_message'] for r in rs]):5}")
    if swebv_parquet:
        repo_overlap(data_dir, swebv_parquet)


if __name__ == '__main__':
    main(*sys.argv[1:4])
