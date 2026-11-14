"""Static trap coverage over all 500 SWE-bench Verified tasks (the RQ4 "refreshability" estimate).

For each task we take the GOLD patch, rebuild the source files it edits at the base commit, and list
every repo fact that the lines it ADDS rely on. This reuses the fact finder of
scripts/agent_targeted_candidates.py (which does the same for the agents' own patches) and adds the
facts that trap types T1, T3, T6, T7 and T8 need. A task counts as covered by a trap type if at least
one fact is a usable target for it:

  T1 argument order    f(a, b, ...) with 2+ positional args; the first two parameters are both required
                       or both have defaults (so they can be swapped in the def)
  T2 default value     f(...) leaves out a parameter whose default is a plain value other than None
  T3 rename            uses a repo function, class or method by name (dunder names excluded)
  T4 move              imports a function/class from the module that defines it (from X import Y with Y
                       defined in X itself, not re-exported), or reads it as X.Y through the module
  T5 exception swap    try: f(...) except E, where the repo function f itself raises E
  T6 return convention a, b = f(...) or f(...)[i], where every return in f is a tuple of that length
  T7 keyword rename    f(k=v), where k is a parameter of f that has a default
  T8 flag inversion    passes True/False (by keyword or position) to a parameter whose default is True/False

This is a static check only: it says a trap CAN be built on a symbol the gold patch uses, not that the
task's tests are sensitive to it (that needs the Docker validation). "cross_file" counts only targets
defined in a different file from the edit (trap-design rule 3: the fact should not sit next to the edit).

Outputs: outputs/traps/coverage.json (counts), work/catalog/gold_facts.jsonl (every fact, for picking
catalog targets).
Usage (from the project folder): .venv/bin/python -I scripts/trap_coverage.py
"""
import ast, json, os, sys, textwrap
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))  # -I leaves the script folder off sys.path
import agent_targeted_candidates as atc  # noqa: E402  (our own script: index of repo definitions + fact finder)
import pyarrow.parquet as pq  # noqa: E402

OUT_JSON = os.path.join(ROOT, 'outputs/traps/coverage.json')
OUT_FACTS = os.path.join(ROOT, 'work/catalog/gold_facts.jsonl')
TYPES = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']

atc.judge = lambda run, f: f  # there is no trajectory here, so keep every fact as found


class GoldFinder(atc.FactFinder):
    """The agent-patch fact finder plus the extra facts the trap types need."""

    def handle_call(self, node, only_keyword=None):
        super().handle_call(node, only_keyword)
        t, bound = self.callee(node.func)
        if t is None or t is atc.EXTERNAL or t.get('kind') not in ('function', 'method', 'class'):
            return
        if not only_keyword and isinstance(node.func, ast.Name):
            self.ref(node.func, t)  # f(...) by bare name: a rename target (x.f(...) is an 'attribute' fact)
        sig = self.signature_for(t, bound)
        if sig is None:
            return
        params, s, owner = sig
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            return  # f(*args) / f(**kwargs): cannot tell which parameters are filled
        for kw in ([only_keyword] if only_keyword else node.keywords):
            if kw.arg in params or kw.arg in s['kwonly']:
                d = s['defaults'].get(kw.arg)
                self.add('t7', node, owner, f"{owner['qual']}(..., {kw.arg}=...)", kw.arg,
                         line=getattr(kw, 'lineno', None), params=[kw.arg],
                         extra=dict(has_default=kw.arg in s['defaults'], default=d))
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool) and d in ('True', 'False'):
                    self.add('t8', node, owner, f"{owner['qual']}(..., {kw.arg}={kw.value.value}) (default {d})",
                             kw.arg, line=getattr(kw, 'lineno', None), params=[kw.arg],
                             extra=dict(default=d, passed=repr(kw.value.value), passed_by='keyword'))
        if only_keyword:
            return
        for i, a in enumerate(node.args):
            if i < len(params) and isinstance(a, ast.Constant) and isinstance(a.value, bool) \
                    and s['defaults'].get(params[i]) in ('True', 'False'):
                self.add('t8', node, owner, f"{owner['qual']} gets {a.value} by position for {params[i]}",
                         params[i], params=[params[i]],
                         extra=dict(default=s['defaults'][params[i]], passed=repr(a.value), passed_by='position'))
        if len(node.args) >= 2 and len(params) >= 2:
            both_req = params[0] not in s['defaults'] and params[1] not in s['defaults']
            both_def = params[0] in s['defaults'] and params[1] in s['defaults']
            self.add('t1', node, owner, f"{owner['qual']}({params[0]}, {params[1]}, ...) called with "
                     f"{len(node.args)} positional args", f'{params[0]},{params[1]}', params=params[:2],
                     extra=dict(swappable=both_req or both_def, n_pos=len(node.args)))

    def ref(self, node, t):
        """A use of a repo function/class by bare name (call or not)."""
        self.add('ref', node, t, f"uses {t['qual']} by name", t['qual'], use='name')

    def visit_Name(self, node):
        if node.lineno in self.added and isinstance(node.ctx, ast.Load) and node.id not in ('self', 'cls'):
            t = self.ctx.resolve_name(node.id)
            if t and t is not atc.EXTERNAL and t.get('kind') in ('function', 'class'):
                self.ref(node, t)

    def visit_Assign(self, node):
        tg = node.targets[0] if len(node.targets) == 1 else None
        if node.lineno in self.added and isinstance(node.value, ast.Call) and isinstance(tg, (ast.Tuple, ast.List)):
            self.unpack(node, node.value, n=len(tg.elts))
        self.generic_visit(node)

    def visit_Subscript(self, node):
        sl = node.slice
        if node.lineno in self.added and isinstance(node.value, ast.Call) and isinstance(sl, ast.Constant) \
                and isinstance(sl.value, int) and not isinstance(sl.value, bool):
            self.unpack(node, node.value, index=sl.value)
        self.generic_visit(node)

    def unpack(self, node, call, n=None, index=None):
        """a, b = f(...) or f(...)[i] where f always returns a tuple of fixed size: a T6 target."""
        t, bound = self.callee(call.func)
        if t is None or t is atc.EXTERNAL or t.get('kind') not in ('function', 'method'):
            return
        sizes = return_sizes(self.ctx.index, t)
        if not sizes or None in sizes or len(sizes) != 1:
            return
        k = next(iter(sizes))
        if k < 2 or (n is not None and n != k) or (index is not None and not -k <= index < k):
            return
        how = 'unpacks it' if n is not None else f'takes item [{index}]'
        self.add('t6', node, t, f"{t['qual']} returns a {k}-tuple; code {how}", str(k), use='unpack',
                 extra=dict(tuple_size=k, index=index))


def return_sizes(index, t):
    """Set of tuple sizes returned by function t (None for a return that is not a tuple literal);
    None if the function cannot be read or is a generator."""
    text = index.source(t['path'])
    if not text or not t.get('line') or not t.get('end'):
        return None
    seg = textwrap.dedent('\n'.join(text.split('\n')[t['line'] - 1:t['end']]))
    try:
        fn = ast.parse(seg).body[0]
    except (SyntaxError, IndexError, ValueError):
        return None
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    sizes, todo = set(), list(fn.body)
    while todo:
        x = todo.pop()
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(x, (ast.Yield, ast.YieldFrom)):
            return None
        if isinstance(x, ast.Return):
            sizes.add(len(x.value.elts) if isinstance(x.value, ast.Tuple) else None)
        todo.extend(ast.iter_child_nodes(x))
    return sizes


atc.FactFinder = GoldFinder  # facts_for_patch() builds its finder from this name


def name_counts(index):
    """name -> number of repo definitions (module-level functions/classes and methods) with that name."""
    c = Counter()
    for s in index.files.values():
        c.update(list(s['funcs']))  # list(): count the names (a dict would add its values)
        c.update(list(s['classes']))
        for cl in s['classes'].values():
            c.update(list(cl['methods']))
    return c


def usable(f, n_defs):
    """Which trap types (T1..T8) this fact is a usable target for."""
    ft, kind, name = f['fact_type'], f.get('target_kind'), f.get('def_name') or ''
    out = set()
    if ft == 't1' and f.get('swappable'):
        out.add('T1')
    if ft == 'default' and f.get('default') not in (None, 'None'):
        out.add('T2')
    if ft in ('ref', 'attribute') and kind in ('function', 'class', 'method') and not name.startswith('__'):
        out.add('T3')
    if ft == 'import' and kind in ('function', 'class') and f.get('target_file') \
            and atc.module_name(f['target_file']) == f.get('module'):
        out.add('T4')
    if ft == 'attribute' and f.get('how') == 'module' and kind in ('function', 'class'):
        out.add('T4')
    if ft == 'raises':
        out.add('T5')
    if ft == 't6':
        out.add('T6')
    if ft == 't7' and f.get('has_default'):
        out.add('T7')
    if ft == 't8':
        out.add('T8')
    return out


def main():
    tasks = pq.read_table(atc.PARQUET).to_pandas().sort_values(['repo', 'base_commit'])
    all_tasks = tasks
    per_task = {}  # instance_id -> {type: {'any': bool, 'cross_file': bool, 'unique_name': bool}}
    notes_count = Counter()
    os.makedirs(os.path.dirname(OUT_FACTS), exist_ok=True)
    with open(OUT_FACTS, 'w') as out:
        for repo, group in tasks.groupby('repo', sort=True):
            git = atc.Git(os.path.join(atc.REPOS, repo.replace('/', '_')))
            cache = {}
            extra = set()
            if repo == 'mwaskom/seaborn':  # same special case as the agent miner: seaborn calls matplotlib a lot
                mpl = all_tasks[all_tasks.repo == 'matplotlib/matplotlib'].sort_values('created_at').iloc[-1]
                extra = set(atc.CommitIndex(atc.Git(os.path.join(atc.REPOS, 'matplotlib_matplotlib')),
                                            mpl.base_commit, {}).members_index())
            for r in group.itertuples():
                index = atc.CommitIndex(git, r.base_commit, cache)
                index.extra_common = extra
                facts, notes = atc.facts_for_patch(index, r.patch, None)
                for n in notes:
                    notes_count[n.split(': ', 1)[1]] += 1
                counts = name_counts(index)
                cov = {t: dict(any=False, cross_file=False) for t in TYPES}
                for f in facts:
                    n_defs = counts.get(f.get('def_name') or '', 0)
                    types = usable(f, n_defs)
                    if not types:
                        continue
                    f.update(instance_id=r.instance_id, repo=repo, trap_types=sorted(types), n_defs_same_name=n_defs)
                    out.write(json.dumps(f) + '\n')
                    for t in types:
                        cov[t]['any'] = True
                        if f.get('cross_file'):
                            cov[t]['cross_file'] = True
                        if t == 'T3' and n_defs == 1:
                            cov[t]['unique_name'] = True
                per_task[r.instance_id] = dict(repo=repo, cov=cov)
            print(repo, len(group), 'tasks', flush=True)
    n = len(per_task)
    by_repo = defaultdict(lambda: defaultdict(lambda: Counter()))
    totals = {t: Counter() for t in TYPES}
    union = Counter()
    n_types_hist = Counter()
    for iid, d in per_task.items():
        k_any = 0
        for t in TYPES:
            for level in ('any', 'cross_file', 'unique_name'):
                if d['cov'][t].get(level):
                    totals[t][level] += 1
                    by_repo[d['repo']][t][level] += 1
            k_any += d['cov'][t]['any']
        union['any'] += k_any > 0
        union['cross_file'] += any(d['cov'][t]['cross_file'] for t in TYPES)
        n_types_hist[k_any] += 1
    repo_sizes = Counter(d['repo'] for d in per_task.values())
    result = dict(
        description='Static trap coverage on SWE-bench Verified: for each trap type, number of tasks whose gold '
                    "patch's added lines use at least one symbol the trap type can be built on (static check "
                    'only; not validated with tests). "cross_file": target defined in a different file than the '
                    'edit. "unique_name" (T3): the name has exactly one definition among repo source files.',
        command='.venv/bin/python -I scripts/trap_coverage.py',
        n_tasks=n,
        per_type={t: dict(tasks_any=totals[t]['any'], share_any=round(totals[t]['any'] / n, 4),
                          tasks_cross_file=totals[t]['cross_file'],
                          share_cross_file=round(totals[t]['cross_file'] / n, 4),
                          **({'tasks_unique_name': totals[t]['unique_name'],
                              'share_unique_name': round(totals[t]['unique_name'] / n, 4)} if t == 'T3' else {}))
                  for t in TYPES},
        any_type=dict(tasks_any=union['any'], tasks_cross_file=union['cross_file']),
        tasks_by_number_of_types=dict(sorted(n_types_hist.items())),
        per_repo={repo: dict(n_tasks=repo_sizes[repo],
                             **{t: dict(by_repo[repo][t]) for t in TYPES}) for repo in sorted(repo_sizes)},
        skipped_files=dict(notes_count),
    )
    json.dump(result, open(OUT_JSON, 'w'), indent=1)
    print(json.dumps(result['per_type'], indent=1))
    print('any type:', result['any_type'], '->', OUT_JSON)


if __name__ == '__main__':
    main()
