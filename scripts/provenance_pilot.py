"""Pilot: when a coding agent names a repo file it has not seen yet in the run, is that file
real, stale (only in older versions), from the future (only in newer versions), or invented?

Usage:
  python3 -I scripts/provenance_pilot.py RUN_DIR SWEBV_PARQUET REPOS_DIR OUT_JSONL

RUN_DIR      one mini-swe-agent run from the SWE-bench submissions bucket (trajs/ + logs/)
REPOS_DIR    blobless clones named like django_django (git clone --filter=blob:none)
OUT_JSONL    one row per (task, file path) the agent mentions

How a mention is classified (first time the agent writes the path, in its hidden reasoning,
its visible THOUGHT, or its command):
  seen      the path, or at least its file name, already appeared in the task text or in an
            earlier tool output, so the agent could have read it
  derived   not seen, but its dotted module name (e.g. auth.validators) appeared earlier
  unseen    neither: the agent named the file before anything in the run showed it
Unseen paths are then checked against git:
  real      exists at the task's base commit
  stale     not at the base commit, but existed in an earlier commit
  future    not at the base commit, only exists in later commits
  invented  never existed in the repo's history
Files the agent writes itself (reproduce.py etc.) are dropped.
"""
import json, os, re, subprocess, sys, bisect
from collections import defaultdict

import pyarrow.parquet as pq

EXT = r'(?:py|pyx|pxd|pyi|c|h|cfg|ini|toml|txt|rst|md|json|yaml|yml|html|js|css|po|sh|in)'
# a path with at least one directory, e.g. django/contrib/auth/validators.py or ./sympy/core/expr.py
PATH_RE = re.compile(r'(?<![\w.\-/:])((?:\.{1,2}/|/testbed/)?(?:[A-Za-z_][\w.\-]*/)+[A-Za-z_][\w.\-]*\.' + EXT + r')(?![\w/])')
WRITE_RE = re.compile(r'(?:>\s*|tee\s+(?:-a\s+)?|touch\s+|cp\s+\S+\s+|mv\s+\S+\s+)(\S+)')
HEDGE = re.compile(r"\b(probably|likely|might|may be|maybe|perhaps|i think|i believe|should be|presumably|"
                   r"if i recall|i recall|i remember|from memory|typically|usually|i'm not sure|not sure|"
                   r"let me (?:check|find|look|verify|search|see|confirm|locate))\b", re.I)


def norm(p):
    p = p.strip('`\'"(),;:')
    for pre in ('/testbed/', './', 'testbed/'):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True).stdout


class Repo:
    """Path lookups at a commit and across the whole history (trees only, no file contents)."""

    def __init__(self, path):
        self.path = path
        self.trees = {}
        self.events = None  # path -> sorted list of commit times where it was added
        self.by_name = None

    def tree(self, commit):
        if commit not in self.trees:
            paths = set(git(self.path, 'ls-tree', '-r', '--name-only', commit).split('\n')) - {''}
            by_name = defaultdict(list)
            for p in paths:
                by_name[p.rsplit('/', 1)[-1]].append(p)
            self.trees[commit] = (paths, by_name)
        return self.trees[commit]

    def history(self):
        if self.events is None:
            cache = self.path.rstrip('/') + '.adds.json'
            if os.path.exists(cache):
                self.events = json.load(open(cache))
            else:
                out = git(self.path, 'log', '--all', '--no-renames', '--diff-filter=A', '--name-only', '--format=@@%ct')
                ev, t = defaultdict(list), 0
                for line in out.split('\n'):
                    if line.startswith('@@'):
                        t = int(line[2:])
                    elif line:
                        ev[line].append(t)
                self.events = {p: sorted(v) for p, v in ev.items()}
                json.dump(self.events, open(cache, 'w'))
            self.by_name = defaultdict(list)
            for p in self.events:
                self.by_name[p.rsplit('/', 1)[-1]].append(p)
        return self.events

    def commit_time(self, commit):
        return int(git(self.path, 'show', '-s', '--format=%ct', commit).strip() or 0)


def resolve(mention, paths, by_name):
    """Exact path, or a partial path that is the tail of exactly one real path."""
    if mention in paths:
        return mention
    hits = [p for p in by_name.get(mention.rsplit('/', 1)[-1], []) if p.endswith('/' + mention)]
    return hits[0] if hits else None


def steps_of(traj):
    """Yield (step_no, reasoning, visible_text, command, observation_after)."""
    msgs = traj['messages']
    out, step = [], 0
    for i, m in enumerate(msgs):
        if m.get('role') != 'assistant':
            continue
        step += 1
        resp = (m.get('extra') or {}).get('response') or {}
        msg = ((resp.get('choices') or [{}])[0].get('message') or {}) if isinstance(resp, dict) else {}
        reasoning = m.get('reasoning') or msg.get('reasoning_content') or msg.get('reasoning') or ''
        if not reasoning and isinstance(msg.get('thinking_blocks'), list):
            reasoning = '\n'.join(b.get('thinking', '') for b in msg['thinking_blocks'] if isinstance(b, dict))
        content = m.get('content') or ''
        if isinstance(content, list):
            content = '\n'.join(c.get('text', '') for c in content if isinstance(c, dict))
        cmds = re.findall(r'```(?:bash)?\n(.*?)```', content, re.S)
        for tc in (m.get('tool_calls') or msg.get('tool_calls') or []):
            try:
                cmds.append(json.loads(tc['function']['arguments']).get('command', ''))
            except Exception:
                pass
        visible = re.sub(r'```.*?```', ' ', content, flags=re.S)
        obs = []
        for n in msgs[i + 1:]:
            if n.get('role') == 'assistant':
                break
            c = n.get('content') or ''
            if isinstance(c, list):
                c = '\n'.join(x.get('text', '') for x in c if isinstance(x, dict))
            obs.append(c)
        # the model sometimes types a fake tool result into its own message
        fake = '<returncode>' in content or '<output>' in content
        out.append((step, reasoning, visible, '\n'.join(cmds), '\n'.join(obs), fake))
    return out


def dotted(path):
    mod = path.rsplit('.', 1)[0].replace('/', '.')
    parts = mod.split('.')
    return '.'.join(parts[-2:]) if len(parts) >= 2 else None


def sentence_around(text, needle):
    i = text.find(needle)
    if i < 0:
        return ''
    a = max(text.rfind('.', 0, i), text.rfind('\n', 0, i), i - 250)
    b = min([x for x in (text.find('.\n', i), text.find('. ', i + len(needle)), text.find('\n\n', i), i + 250) if x >= 0])
    return text[a + 1:b + 1]


def analyze(run_dir, tasks, repos_dir, out):
    repos = {}
    n_tasks = 0
    for iid in sorted(os.listdir(os.path.join(run_dir, 'trajs'))):
        f = os.path.join(run_dir, 'trajs', iid, iid + '.traj.json')
        if iid not in tasks or not os.path.exists(f):
            continue
        task = tasks[iid]
        rp = os.path.join(repos_dir, task['repo'].replace('/', '_'))
        if not os.path.isdir(rp):
            continue
        repo = repos.setdefault(rp, Repo(rp))
        paths, by_name = repo.tree(task['base_commit'])
        if not paths:
            continue
        rep_f = os.path.join(run_dir, 'logs', iid, 'report.json')
        resolved = json.load(open(rep_f)).get('resolved') if os.path.exists(rep_f) else None
        traj = json.load(open(f))
        steps = steps_of(traj)
        n_tasks += 1

        context = '\n'.join(str(m.get('content') or '') for m in traj['messages'][:2])  # system + task
        seen_text = context
        written = set()
        for _, _, _, cmd, _, _ in steps:
            for w in WRITE_RE.findall(cmd):
                written.add(norm(w))
        first = {}
        n_fake = sum(s[5] for s in steps)
        for step, reasoning, visible, cmd, obs, fake in steps:
            for field, text in (('reasoning', reasoning), ('thought', visible), ('command', cmd)):
                for raw in PATH_RE.findall(text):
                    p = norm(raw)
                    if p.startswith('/') or p in first or '/' not in p:
                        continue
                    name = p.rsplit('/', 1)[-1]
                    if p in seen_text:
                        prov = 'seen'
                    elif name in seen_text:
                        prov = 'seen_name'
                    elif dotted(p) and dotted(p) in seen_text:
                        prov = 'derived'
                    else:
                        prov = 'unseen'
                    first[p] = dict(step=step, field=field, prov=prov, in_fake_output=bool(fake and field == 'thought'),
                                    context=sentence_around(text, raw)[:400])
            seen_text += '\n' + obs  # tool output becomes visible from the next step on

        for p, info in first.items():
            at_base = resolve(p, paths, by_name)
            years = None
            if at_base:
                status = 'real'
            elif p in written or any(w.endswith(p) for w in written):
                continue  # the agent made this file itself
            else:
                ev = repo.history()
                hit = resolve(p, ev, repo.by_name)
                if not hit:
                    status = 'invented'
                else:
                    t0 = task.setdefault('_t', repo.commit_time(task['base_commit']))
                    adds = ev[hit]
                    status = 'stale' if adds and adds[0] < t0 else 'future'
                    years = round((adds[0] - t0) / 31557600, 2)  # negative = added before the base commit
            out.write(json.dumps(dict(
                instance_id=iid, repo=task['repo'], resolved=resolved, path=p, status=status, years=years,
                hedged=bool(HEDGE.search(info['context'])), n_steps=len(steps), n_fake_outputs=n_fake,
                **info)) + '\n')
    return n_tasks


def main():
    run_dir, parquet, repos_dir, out_path = sys.argv[1:5]
    t = pq.read_table(parquet).to_pandas()
    tasks = {r.instance_id: dict(repo=r.repo, base_commit=r.base_commit) for r in t.itertuples()}
    with open(out_path, 'w') as out:
        n = analyze(run_dir, tasks, repos_dir, out)
    print(f'{n} tasks analyzed -> {out_path}')


if __name__ == '__main__':
    main()
