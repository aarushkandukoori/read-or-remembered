"""Per-run behavior around the trapped fact (agent-targeted traps).

For each run: steps, cost, whether it ever wrote a call to the trapped function, the form of the first
written call (memory form = how the original repo / the model's memory has it; trap form = the
changed version), whether it had opened the definition or seen other call sites before that, errors
that followed a memory-form call (caught by its own runs), and the form used in the final patch.
Usage: python -I analyze_targeted.py IID
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
iid = sys.argv[1]
cfg = json.load(open(HERE / 'work' / iid / 'trap.json'))
func = cfg['func']
grades = {g['run']: g for g in json.load(open(HERE / 'runs' / iid / 'grades.json'))}


def form(snippet):
    """'memory', 'trap' or 'neutral' for one call text starting right after FUNC("""
    if cfg['kind'] == 'kwrename':
        if re.match(rf'[^)]*\b{cfg["param"]}=', snippet):
            return 'memory'
        if re.match(rf'[^)]*\b{cfg["newname"]}=', snippet):
            return 'trap'
        return 'neutral'
    first = snippet.lstrip()
    if re.match(r'''["']\w+__\w+["']|["']pk__in["']''', first):
        return 'memory'          # add_filter("pk__in", values): lookup first, as in the original repo
    if re.match(r'''[^,]+,\s*["']\w+["']''', first):
        return 'trap'            # add_filter(values, "pk__in")
    return 'neutral'


def calls(text):
    return [form(text[m.end():m.end() + 160]) for m in re.finditer(rf'(?<![\w]){func}\(', text)]


mem_words = re.compile(r'(from memory|recall|remember|upstream|modern Django|newer Django|Django \d\.\d+ has)', re.I)
rows = []
for f in sorted((HERE / 'runs' / iid).glob('*.traj.json')):
    run = f.name.replace('.traj.json', '')
    tr = json.load(open(f))
    msgs = tr['messages']
    step, first, saw_def, saw_calls, err_after_mem, mem_writes = 0, None, None, 0, False, 0
    hidden = visible = 0
    obs_so_far = ''
    for i, m in enumerate(msgs):
        if m.get('role') != 'assistant':
            if m.get('role') in ('tool', 'user') and i > 1:
                obs_so_far += '\n' + str(m.get('content') or '')
            continue
        step += 1
        hidden += bool(mem_words.search(m.get('reasoning_content') or ''))
        visible += bool(mem_words.search(m.get('content') or ''))
        cmds = '\n'.join(json.loads(tc['function']['arguments']).get('command', '')
                         for tc in (m.get('tool_calls') or []) if tc.get('function'))
        forms = calls(cmds)
        if forms:
            if first is None:
                first = (step, forms[0])
                saw_def = f'def {func}(' in obs_so_far
                saw_calls = len(re.findall(rf'(?<![\w]){func}\(', obs_so_far))
            if 'memory' in forms:
                mem_writes += 1
                nxt = ' '.join(str(n.get('content') or '') for n in msgs[i + 1:i + 4] if n.get('role') in ('tool', 'user'))
                if re.search(r'Traceback|Error|FAILED|unexpected keyword', nxt):
                    err_after_mem = True
    patch = tr.get('info', {}).get('submission') or ''
    added = '\n'.join(l for l in patch.split('\n') if l.startswith('+') and not l.startswith('+++'))
    g = grades.get(run, {})
    rows.append(dict(run=run, resolved=g.get('resolved'), tests=f"F2P {g.get('F2P')} P2P {g.get('P2P')}",
                     steps=step, cost=round(tr.get('info', {}).get('model_stats', {}).get('instance_cost', 0), 4),
                     first_call_written=first, saw_definition_before=saw_def, call_sites_seen_before=saw_calls,
                     memory_form_writes=mem_writes, error_after_memory_form=err_after_mem,
                     final_patch_forms=sorted(set(calls(added))) or ['no call'],
                     memory_talk_hidden_steps=hidden, memory_talk_visible_steps=visible))
    print(json.dumps(rows[-1]))
json.dump(rows, open(HERE / 'runs' / iid / 'analysis.json', 'w'), indent=1)
