"""The eight memory-trap transforms (T1-T8) and their placebos.

Each transform changes ONE fact about a repo that a model has probably memorized, and carries the
change through the whole repo (definition, every call site, imports, doctests, docs) so the repo
still behaves the same. apply(repo, 'trap') makes the remembered fact false; apply(repo, 'placebo')
makes an edit of the same kind, at the same places, that keeps the remembered fact true.

  T1 ArgSwap         f(a, b) -> f(b, a): the first two parameters swap places; every positional
                     call is swapped too.          placebo: the same calls name their arguments
  T2 DefaultChange   p=X -> p=Y; calls that relied on the default now pass p=X.
                                                   placebo: the same p=X added, default unchanged
  T3 Rename          old name -> new name everywhere, no alias.
                                                   placebo: the same rename plus `old = new` alias
                                                   (files that imported old import both names)
  T4 Move            function/class moves from module a to module b, imports updated, no re-export.
                                                   placebo: b re-exports it from a, imports point at b
  T5 ExceptionSwap   f raises a new exception class instead of E; excepts/tests around f also accept it.
                                                   placebo: the same, but the new class subclasses E
  T6 ReturnSwap      f returns (b, a) instead of (a, b); every caller unpacks the other way.
                                                   placebo: the same lines, only parenthesized
  T7 KeywordRename   parameter old -> new; keyword calls updated.
                                                   placebo: the same rename, old keyword still accepted
  T8 FlagInvert      flag=True -> inverse_flag=False; calls pass the negated value.
                                                   placebo: the same edit, old flag still accepted

Name-based matching: calls are matched by the name they are called with (f(...) or obj.f(...)).
That is only safe for names with one definition in the repo, so prepare() refuses names that are
defined more than once (unless the definitions are given as a family) and reports uses it cannot
rewrite (the function passed around as a value, f(*args), f(**kwargs)) as warnings.
"""
import ast
import builtins
import re

import libcst as cst

from . import common as C


class TrapError(Exception):
    """The target cannot be trapped safely (e.g. its name is defined twice)."""


class Report:
    """What one apply() run changed and anything it could not handle."""

    def __init__(self):
        self.files = set()
        self.sites = []      # (file, kind, short code)
        self.warnings = []   # things a human should look at (not rewritten)

    def site(self, rel, kind, node=None, text=None):
        snippet = text if text is not None else (C.code_of(node).strip().replace('\n', ' ')[:140] if node else '')
        self.sites.append((rel, kind, snippet))

    def warn(self, rel, msg):
        self.warnings.append(f'{rel}: {msg}')

    def as_dict(self):
        kinds = {}
        for _, k, _ in self.sites:
            kinds[k] = kinds.get(k, 0) + 1
        return dict(files=sorted(self.files), n_sites=len(self.sites), sites_by_kind=kinds,
                    sites=[dict(file=f, kind=k, code=s) for f, k, s in self.sites], warnings=self.warnings)


# --------------------------------------------------------------------------- finding definitions

class _DefFinder(cst.CSTVisitor):
    """Collect every function/class definition of a module with its qualified name."""

    def __init__(self):
        self.stack, self.found = [], {}

    def visit_ClassDef(self, node):
        self.stack.append(node.name.value)
        self.found.setdefault('.'.join(self.stack), node)

    def leave_ClassDef(self, node):
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self.stack.append(node.name.value)
        self.found.setdefault('.'.join(self.stack), node)

    def leave_FunctionDef(self, node):
        self.stack.pop()


def find_def(repo, rel, qual):
    """The libcst node defining `qual` (e.g. 'f', 'Cls.method', 'Cls') in file `rel`."""
    return def_info(repo, rel, qual)[0]


def def_info(repo, rel, qual):
    """(node, is_method, is_static) for the definition `qual` in file `rel`. A method is a def directly
    inside a class; its first parameter (self/cls) is filled automatically unless it is a staticmethod."""
    mod = cst.parse_module(C.read(repo, rel))
    v = _DefFinder()
    mod.visit(v)
    if qual not in v.found:
        raise TrapError(f'{qual} is not defined in {rel}')
    node = v.found[qual]
    parent = v.found.get(qual.rpartition('.')[0]) if '.' in qual else None
    is_method = isinstance(node, cst.FunctionDef) and isinstance(parent, cst.ClassDef)
    static = isinstance(node, cst.FunctionDef) and 'staticmethod' in decorator_names(node)
    return node, is_method, static


def decorator_names(fn):
    out = set()
    for d in fn.decorators:
        e = d.decorator.func if isinstance(d.decorator, cst.Call) else d.decorator
        out.add(e.value if isinstance(e, cst.Name) else e.attr.value if isinstance(e, cst.Attribute) else '')
    return out


def positional_params(fn):
    return list(fn.params.posonly_params) + list(fn.params.params)


def count_definitions(repo, name):
    """Every def/class called `name` in the repo's .py files: list of (file, qualname)."""
    out = []
    for rel in C.iter_files(repo, ('.py',)):
        text = C.read(repo, rel)
        if not re.search(r'\b(?:def|class)\s+' + re.escape(name) + r'\b', text):
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue

        def walk(body, prefix):
            for n in body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    q = prefix + n.name
                    if n.name == name:
                        out.append((rel, q))
                    walk(n.body, q + '.')
                elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                    for field in ('body', 'orelse', 'finalbody', 'handlers'):
                        sub = getattr(n, field, None) or []
                        walk([h for h in sub] if field != 'handlers' else [x for h in sub for x in h.body], prefix)
        walk(tree.body, '')
    return out


class _BindingFinder(ast.NodeVisitor):
    """Places where a name is BOUND (assigned, a parameter, a loop variable, an import alias):
    a rename of that name would clash with those."""

    def __init__(self, name):
        self.name, self.lines = name, []

    def _hit(self, node):
        self.lines.append(getattr(node, 'lineno', 0))

    def visit_Name(self, node):
        if node.id == self.name and isinstance(node.ctx, (ast.Store, ast.Del)):
            self._hit(node)

    def visit_arg(self, node):
        if node.arg == self.name:
            self._hit(node)

    def visit_alias(self, node):
        # "import x as name" binds a different object to the name; "from m import name as name" does not
        if node.asname == self.name and node.name.split('.')[-1] != self.name:
            self._hit(node)

    def visit_ExceptHandler(self, node):
        if node.name == self.name:
            self._hit(node)
        self.generic_visit(node)


def bindings_of(repo, name, skip_defs=()):
    """Files/lines (outside the target definitions) where `name` is bound as a variable."""
    out = []
    for rel in C.iter_files(repo, ('.py',)):
        text = C.read(repo, rel)
        if name not in text:
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        f = _BindingFinder(name)
        f.visit(tree)
        out.extend((rel, ln) for ln in f.lines)
    return out


class _EscapeFinder(cst.CSTVisitor):
    """Uses of a function name that are not calls (passed as a value, stored, wrapped by partial...)."""

    def __init__(self, names):
        self.names, self.skip, self.hits = names, set(), []

    def visit_Call(self, node):
        self.skip.add(id(node.func))
        if isinstance(node.func, cst.Attribute):
            self.skip.add(id(node.func.attr))

    def visit_FunctionDef(self, node):
        self.skip.add(id(node.name))

    def visit_ClassDef(self, node):
        self.skip.add(id(node.name))

    def visit_ImportAlias(self, node):
        for n in (node.name, node.asname.name if node.asname else None):
            if isinstance(n, cst.Name):
                self.skip.add(id(n))
            elif isinstance(n, cst.Attribute):
                self.skip.add(id(n.attr))

    def visit_Arg(self, node):
        if node.keyword is not None:
            self.skip.add(id(node.keyword))

    def visit_Param(self, node):
        self.skip.add(id(node.name))

    def visit_Attribute(self, node):
        if isinstance(node.value, cst.Name):  # f.cache_clear() etc.: attribute of the function itself
            self.skip.add(id(node.value))
        if node.attr.value in self.names and id(node) not in self.skip and id(node.attr) not in self.skip:
            self.hits.append(C.code_of(node))
        self.skip.add(id(node.attr))

    def visit_Name(self, node):
        if node.value in self.names and id(node) not in self.skip:
            self.hits.append(node.value)


# --------------------------------------------------------------------------------- base class

class QualTransformer(cst.CSTTransformer):
    """A libcst transformer that knows the qualified name of the def/class it is inside."""

    def __init__(self, rel):
        super().__init__()
        self.rel, self.stack = rel, []

    def visit_ClassDef(self, node):
        self.stack.append(node.name.value)

    def leave_ClassDef(self, orig, upd):
        qual = '.'.join(self.stack)
        self.stack.pop()
        return self.on_class(orig, upd, qual)

    def visit_FunctionDef(self, node):
        self.stack.append(node.name.value)

    def leave_FunctionDef(self, orig, upd):
        qual = '.'.join(self.stack)
        self.stack.pop()
        return self.on_function(orig, upd, qual)

    def on_class(self, orig, upd, qual):
        return upd

    def on_function(self, orig, upd, qual):
        return upd


class Transform:
    """Base class. Subclasses fill in prepare(), the code transformer and the text rewriters."""
    type_id = ''
    title = ''
    loud_or_quiet = ''
    placebo_kind = ''

    def __init__(self, defs, call_names=None):
        if isinstance(defs, tuple):
            defs = [defs]
        self.defs = [tuple(d) for d in defs]          # [(file, qualname)], usually one definition
        self.def_file, self.qual = self.defs[0]
        parts = self.qual.split('.')
        self.is_ctor = parts[-1] in ('__init__', '__new__')
        self.name = parts[-2] if self.is_ctor else parts[-1]
        self.class_name = parts[-2] if len(parts) >= 2 and not self.is_ctor else None
        names = {q.split('.')[-2] if q.split('.')[-1] in ('__init__', '__new__') else q.split('.')[-1]
                 for _, q in self.defs}
        self.call_names = set(call_names) if call_names else names
        self.mode = None
        self.init = {} if not call_names else dict(call_names=sorted(call_names))  # extra constructor args
        self.extra_edits = []  # [(file, old_text, new_text)] literal prose edits, see add_prose_edit()

    # -- the steps of apply()
    def prepare(self, repo, report):
        """Read the definition, check the target is safe. Raise TrapError if not."""

    def mentions(self, text):
        return any(n in text for n in self.call_names)

    def scan(self, rel, mod):
        """Look at a parsed file before rewriting it (default: nothing)."""

    def code_transformer(self, rel, report):
        raise NotImplementedError

    def edit_strings(self, rel, code, report):
        """Rewrite code written inside strings/comments (doctests). Default: nothing."""
        return code

    def edit_doc(self, rel, text, report):
        """Rewrite a .rst/.txt doc file. Default: nothing."""
        return text

    def finish(self, repo, report):
        """Last step, after all files were rewritten (new modules, aliases...)."""

    def apply(self, repo, mode):
        assert mode in ('trap', 'placebo')
        self.mode = mode
        report = Report()
        self.prepare(repo, report)
        for rel in C.iter_files(repo, ('.py',)):
            text = C.read(repo, rel)
            if not self.mentions(text):
                continue
            try:
                mod = cst.parse_module(text)
            except Exception as e:  # e.g. Python 2 syntax
                report.warn(rel, f'could not parse ({type(e).__name__}), not rewritten')
                continue
            self.scan(rel, mod)
            new = mod.visit(self.code_transformer(rel, report)).code
            new = self.edit_strings(rel, new, report)
            if new != text:
                C.write(repo, rel, new)
                report.files.add(rel)
        for rel in C.iter_files(repo, C.DOC_EXTS):
            text = C.read(repo, rel)
            if not self.mentions(text):
                continue
            new = self.edit_doc(rel, text, report)
            if new != text:
                C.write(repo, rel, new)
                report.files.add(rel)
        self.finish(repo, report)
        self.apply_prose_edits(repo, report)
        return report

    def describe(self):
        """dict(original_fact=..., trapped_fact=...) in plain words."""
        raise NotImplementedError

    def spec(self):
        """Parameters needed to rebuild this transform (stored in spec.json): make(spec) rebuilds it."""
        return dict(type=self.type_id, defs=[list(d) for d in self.defs], init=dict(self.init))

    # -- shared checks
    def check_unique(self, repo, report):
        """All definitions with the target's name(s) must be part of the target family."""
        family = set(self.defs)
        if self.is_ctor:
            family = {(f, '.'.join(q.split('.')[:-1])) for f, q in self.defs}
        found = []
        for name in sorted({q.split('.')[-1] for _, q in family}):
            hits = count_definitions(repo, name)
            found += hits
            extra = [d for d in hits if d not in family]
            if extra:
                raise TrapError(f'{name} is also defined at {extra[:5]}; name-based rewriting is unsafe')
        return found

    def add_prose_edit(self, rel, old, new):
        """A literal text edit made in trap mode only (e.g. a docstring that spells out the old fact).
        Prose cannot be rewritten automatically, so these are listed explicitly in the catalog plan."""
        self.extra_edits.append((rel, old, new))
        self.init.setdefault('prose_edits', []).append([rel, old, new])

    def apply_prose_edits(self, repo, report):
        if self.mode != 'trap':
            return
        for rel, old, new in self.extra_edits:
            text = C.read(repo, rel)
            if text.count(old) != 1:
                report.warn(rel, f'prose edit matched {text.count(old)} times, not applied: {old[:60]!r}')
                continue
            C.write(repo, rel, text.replace(old, new))
            report.files.add(rel)
            report.site(rel, 'prose', text=new.strip()[:80])

    def check_no_subclasses(self, repo):
        """For a constructor target (Cls.__init__): calls to subclasses also run it, and name-based
        rewriting would miss them, so refuse classes that have subclasses in the repo."""
        if not self.is_ctor:
            return
        pat = re.compile(r'^\s*class\s+\w+\s*\([^)]*\b' + re.escape(self.name) + r'\b', re.M)
        subs = [rel for rel in C.iter_files(repo, ('.py',)) if self.name in C.read(repo, rel)
                and pat.search(C.read(repo, rel))]
        if subs:
            raise TrapError(f'{self.name} has subclasses (e.g. in {subs[:3]}); constructor traps would miss them')

    def check_escapes(self, repo, report, names=None):
        """Warn about uses of the name that are not plain calls."""
        names = names or self.call_names
        for rel in C.iter_files(repo, ('.py',)):
            text = C.read(repo, rel)
            if not any(n in text for n in names):
                continue
            try:
                mod = cst.parse_module(text)
            except Exception:
                continue
            v = _EscapeFinder(names)
            mod.visit(v)
            for h in v.hits:
                report.warn(rel, f'non-call use of {h} (not rewritten)')


def _bound_offset(call, transform, fn_is_method, static):
    """How many positional arguments to skip before the target's own parameters: 1 when a method is
    called through its class (Cls.method(obj, a, b)), else 0."""
    if fn_is_method and not static and isinstance(call.func, cst.Attribute) \
            and isinstance(call.func.value, cst.Name) and call.func.value.value == transform.class_name:
        return 1
    return 0


def _swap(args, i, j):
    """Swap the values of arguments i and j, keeping each slot's comma and spacing."""
    a, b = args[i], args[j]
    args[i] = b.with_changes(comma=a.comma, whitespace_after_arg=a.whitespace_after_arg)
    args[j] = a.with_changes(comma=b.comma, whitespace_after_arg=b.whitespace_after_arg)




# ======================================================================== T1 argument order

class ArgSwap(Transform):
    type_id = 'T1'
    title = 'argument order'
    loud_or_quiet = 'quiet'
    placebo_kind = 'the same calls pass their arguments by keyword (order unchanged)'

    def prepare(self, repo, report):
        self.check_unique(repo, report)
        self.check_no_subclasses(repo)
        fn, is_method, static = def_info(repo, self.def_file, self.qual)
        self.is_method, self.static = is_method, static
        params = positional_params(fn)
        skip = 1 if is_method and not static else 0
        self.skip = skip
        names = [p.name.value for p in params[skip:]]
        if len(names) < 2:
            raise TrapError(f'{self.qual} has fewer than 2 positional parameters')
        self.p0, self.p1 = names[0], names[1]
        self.pos_names = names
        for rel, qual in self.defs[1:]:
            fn2, m2, st2 = def_info(repo, rel, qual)
            names2 = [p.name.value for p in positional_params(fn2)[1 if m2 and not st2 else 0:]]
            if names2[:2] != names[:2]:
                raise TrapError(f'{qual} names its first parameters {names2[:2]}, not {names[:2]}')
        d0, d1 = params[skip].default is not None, params[skip + 1].default is not None
        if d0 != d1:
            raise TrapError(f'{self.p0} and {self.p1} cannot swap: only one has a default')
        if len(fn.params.posonly_params) > skip:
            raise TrapError('positional-only parameters: the placebo cannot pass them by keyword')
        self.check_escapes(repo, report)

    def describe(self):
        q = self.qual
        return dict(original_fact=f'{q}({self.p0}, {self.p1}, ...): {self.p0} comes first',
                    trapped_fact=f'{q}({self.p1}, {self.p0}, ...): {self.p1} comes first; callers pass '
                                 f'arguments in the new order')

    def edit_args(self, call, report, rel):
        args = list(call.args)
        offset = _bound_offset(call, self, self.is_method, self.static)
        pos = [i for i, a in enumerate(args) if C.is_positional(a)]
        star = [i for i, a in enumerate(args) if a.star == '*']
        if self.mode == 'trap':
            if star and (len(pos) < offset + 2 or star[0] < pos[offset + 1]):
                report.warn(rel, f'call with *args not rewritten: {C.code_of(call)[:100]}')
                return None
            if len(pos) >= offset + 2:
                _swap(args, pos[offset], pos[offset + 1])
                return args
            if len(pos) == offset + 1:  # only the old first argument by position: name it
                args[pos[offset]] = C.keywordize(args[pos[offset]], self.p0)
                return args
            return None
        # placebo: name every positional argument (same order)
        if len(pos) <= offset:
            return None
        if star or len(pos) - offset > len(self.pos_names):
            report.warn(rel, f'placebo: call not rewritten: {C.code_of(call)[:100]}')
            return None
        for k, i in enumerate(pos[offset:]):
            args[i] = C.keywordize(args[i], self.pos_names[k])
        return args

    def code_transformer(self, rel, report):
        t = self

        class X(QualTransformer):
            def on_function(self, orig, upd, qual):
                if t.mode == 'trap' and (rel, qual) in t.defs:
                    seq = 'posonly_params' if len(upd.params.posonly_params) > t.skip else 'params'
                    ps = list(getattr(upd.params, seq))
                    i = t.skip
                    a, b = ps[i], ps[i + 1]
                    ps[i] = b.with_changes(comma=a.comma, whitespace_after_param=a.whitespace_after_param)
                    ps[i + 1] = a.with_changes(comma=b.comma, whitespace_after_param=b.whitespace_after_param)
                    report.site(rel, 'definition', text=f'{qual}({t.p1}, {t.p0}, ...)')
                    return upd.with_changes(params=upd.params.with_changes(**{seq: ps}))
                return upd

            def leave_Call(self, orig, upd):
                if C.call_name(upd) not in t.call_names:
                    return upd
                args = t.edit_args(upd, report, rel)
                if args is None:
                    return upd
                new = upd.with_changes(args=args)
                report.site(rel, 'call', new)
                return new
        return X(rel)

    def edit_cores(self, cores, before):
        """Same rewrite for a call written as text (doctests, docs)."""
        if C.is_signature_context(before):
            if self.mode != 'trap':
                return None
            off = 1 if cores and cores[0] in ('self', 'cls') else 0
            if len(cores) >= off + 2 and not cores[off].startswith('*') and not cores[off + 1].startswith('*'):
                cores[off], cores[off + 1] = cores[off + 1], cores[off]
                return cores
            return None
        if any(c.startswith('*') for c in cores):
            return None
        pos = [i for i, c in enumerate(cores) if c and not C.is_keyword_core(c)]
        if self.mode == 'trap':
            if len(pos) >= 2:
                cores[pos[0]], cores[pos[1]] = cores[pos[1]], cores[pos[0]]
                return cores
            if len(pos) == 1:
                cores[pos[0]] = f'{self.p0}={cores[pos[0]]}'
                return cores
            return None
        if not pos or len(pos) > len(self.pos_names):
            return None
        for k, i in enumerate(pos):
            cores[i] = f'{self.pos_names[k]}={cores[i]}'
        return cores

    def _text(self, rel, text, report, kind):
        for n in self.call_names:
            text, k = C.rewrite_text_calls(text, n, self.edit_cores)
            for _ in range(k):
                report.site(rel, kind, text=f'{n}(...) in text')
        return text

    def edit_strings(self, rel, code, report):
        def edit(s):
            return self._text(rel, s, report, 'doctest/string') if any(n + '(' in s for n in self.call_names) else s
        return C.rewrite_string_tokens(code, edit)

    def edit_doc(self, rel, text, report):
        return self._text(rel, text, report, 'doc')

    def spec(self):
        return dict(super().spec(), p0=self.p0, p1=self.p1)


# ======================================================================== T2 default value

class DefaultChange(Transform):
    type_id = 'T2'
    title = 'default value'
    loud_or_quiet = 'quiet'
    placebo_kind = 'the same calls pass the old default explicitly; the default itself is unchanged'

    def __init__(self, defs, param, new_default, call_names=None):
        super().__init__(defs, call_names)
        self.param, self.new_default = param, new_default
        self.init.update(param=param, new_default=new_default)

    def prepare(self, repo, report):
        self.check_unique(repo, report)
        self.check_no_subclasses(repo)
        olds = set()
        for rel, qual in self.defs:
            fn, is_method, static = def_info(repo, rel, qual)
            params = positional_params(fn)
            skip = 1 if is_method and not static else 0
            names = [p.name.value for p in params[skip:]]
            kwonly = {p.name.value: p for p in fn.params.kwonly_params}
            if self.param in names:
                p = params[skip + names.index(self.param)]
                self.index = names.index(self.param)
            elif self.param in kwonly:
                p = kwonly[self.param]
                self.index = None
            else:
                raise TrapError(f'{qual} has no parameter {self.param}')
            if p.default is None:
                raise TrapError(f'{qual}: {self.param} has no default')
            olds.add(C.code_of(p.default).strip())
            self.is_method, self.static, self.skip = is_method, static, skip
        if len(olds) != 1:
            raise TrapError(f'definitions disagree on the default of {self.param}: {olds}')
        self.old_default = olds.pop()
        if self.old_default == self.new_default:
            raise TrapError('new default equals the old one')
        self.check_escapes(repo, report)

    def describe(self):
        return dict(original_fact=f'{self.qual}: {self.param} defaults to {self.old_default}',
                    trapped_fact=f'{self.qual}: {self.param} defaults to {self.new_default}; every call '
                                 f'that relied on the old default now passes {self.param}={self.old_default}')

    def passes_param(self, call, report, rel):
        """True/False whether the call fills the parameter; None if we cannot tell."""
        args = list(call.args)
        if any(a.star == '**' for a in args):
            return None
        if any(a.keyword is not None and a.keyword.value == self.param for a in args):
            return True
        if self.index is None:
            return False
        if any(a.star == '*' for a in args):
            return None
        offset = _bound_offset(call, self, self.is_method, self.static)
        npos = sum(1 for a in args if C.is_positional(a)) - offset
        return npos > self.index

    def code_transformer(self, rel, report):
        t = self

        class X(QualTransformer):
            def on_function(self, orig, upd, qual):
                if t.mode == 'trap' and (rel, qual) in t.defs:
                    def fix(p):
                        if p.name.value == t.param:
                            return p.with_changes(default=cst.parse_expression(t.new_default))
                        return p
                    ps = upd.params
                    new = ps.with_changes(params=[fix(p) for p in ps.params],
                                          posonly_params=[fix(p) for p in ps.posonly_params],
                                          kwonly_params=[fix(p) for p in ps.kwonly_params])
                    report.site(rel, 'definition', text=f'{qual}(..., {t.param}={t.new_default})')
                    return upd.with_changes(params=new)
                return upd

            def leave_Call(self, orig, upd):
                if C.call_name(upd) not in t.call_names:
                    return upd
                passed = t.passes_param(upd, report, rel)
                if passed is None:
                    report.warn(rel, f'cannot tell if {t.param} is passed: {C.code_of(upd)[:100]}')
                    return upd
                if passed:
                    return upd
                arg = cst.Arg(value=cst.parse_expression(t.old_default), keyword=cst.Name(t.param), equal=C.EQ)
                new = upd.with_changes(args=C.append_arg(upd, arg))
                report.site(rel, 'call', new)
                return new
        return X(rel)

    def edit_cores(self, cores, before):
        if C.is_signature_context(before):
            if self.mode != 'trap':
                return None
            out = []
            for c in cores:
                m = re.match(r'^(\**)' + re.escape(self.param) + r'(\s*(?::[^=]*)?=\s*)(.*)$', c)
                out.append(m.group(1) + self.param + m.group(2) + self.new_default if m else c)
            return out if out != cores else None
        if any(c.startswith('*') for c in cores):
            return None
        kws = {re.match(r'^(\w+)', c).group(1) for c in cores if c and C.is_keyword_core(c)}
        if self.param in kws:
            return None
        npos = sum(1 for c in cores if c and not C.is_keyword_core(c))
        if self.index is not None and npos > self.index:
            return None
        if cores and not cores[-1]:  # trailing comma
            return cores[:-1] + [f'{self.param}={self.old_default}', '']
        return cores + [f'{self.param}={self.old_default}']

    def _text(self, rel, text, report, kind):
        for n in self.call_names:
            text, k = C.rewrite_text_calls(text, n, self.edit_cores)
            for _ in range(k):
                report.site(rel, kind, text=f'{n}(...) in text')
        return text

    def edit_strings(self, rel, code, report):
        def edit(s):
            return self._text(rel, s, report, 'doctest/string') if ('>>>' in s or '::' in s) and \
                any(n + '(' in s for n in self.call_names) else s
        return C.rewrite_string_tokens(code, edit)

    def edit_doc(self, rel, text, report):
        return self._text(rel, text, report, 'doc')

    def spec(self):
        return dict(super().spec(), param=self.param, old_default=self.old_default, new_default=self.new_default)


# ======================================================================== T3 rename

class Rename(Transform):
    type_id = 'T3'
    title = 'rename'
    loud_or_quiet = 'loud'
    placebo_kind = ('the same rename everywhere, plus a one-line alias that keeps the old name working; files that '
                    'imported the old name import both names')

    def __init__(self, defs, new_name):
        super().__init__(defs)
        self.old = self.name
        self.new = new_name
        self.init.update(new_name=new_name)

    def mentions(self, text):
        return self.old in text

    def prepare(self, repo, report):
        if not C.valid_identifier(self.new):
            raise TrapError(f'{self.new} is not a valid name')
        self.check_unique(repo, report)
        if count_definitions(repo, self.new):
            raise TrapError(f'{self.new} is already defined in the repo')
        clash = [b for b in bindings_of(repo, self.old)]
        if clash:
            raise TrapError(f'{self.old} is also bound as a variable/parameter at {clash[:5]}')
        if bindings_of(repo, self.new):
            raise TrapError(f'{self.new} is already used as a variable/parameter')

    def describe(self):
        return dict(original_fact=f'{self.qual} is called {self.old}',
                    trapped_fact=f'{self.old} was renamed to {self.new} everywhere (no alias)')

    def code_transformer(self, rel, report):
        t = self

        class X(QualTransformer):
            def __init__(self, rel):
                super().__init__(rel)
                self.skip = set()  # Name nodes that are not references to the target

            def visit_Arg(self, node):  # keyword names in calls belong to the callee: keep
                if node.keyword is not None:
                    self.skip.add(id(node.keyword))

            def visit_Param(self, node):
                self.skip.add(id(node.name))

            def leave_Name(self, orig, upd):
                if upd.value == t.old and id(orig) not in self.skip:
                    report.site(rel, 'name', text=t.old)
                    return upd.with_changes(value=t.new)
                return upd

            def leave_ImportFrom(self, orig, upd):
                # Placebo only: a file that imported the old name keeps it bound as well
                # (`from m import old` -> `from m import new, old`; the alias in the defining module makes the
                # old name importable). Without this, code written from memory that uses the old name in a
                # module that already imported it fails in the placebo too (seen in sympy-14248: the fix calls
                # _keep_coeff in printing/str.py, which already imports it at the top).
                if t.mode != 'placebo' or isinstance(upd.names, cst.ImportStar):
                    return upd
                names, changed = [], False
                for o, a in zip(orig.names, upd.names):
                    names.append(a)
                    if isinstance(o.name, cst.Name) and o.name.value == t.old and o.asname is None:
                        names[-1] = a.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(' ')))
                        names.append(cst.ImportAlias(name=cst.Name(t.old), comma=a.comma))
                        changed = True
                if not changed:
                    return upd
                report.site(rel, 'import keeps old name', text=f'import {t.new}, {t.old}')
                return upd.with_changes(names=names)
        return X(rel)

    def edit_strings(self, rel, code, report):
        def edit(s):
            new = C.replace_word(s, self.old, self.new)
            if new != s:
                report.site(rel, 'string/comment', text=s.strip()[:80])
            return new
        return C.rewrite_string_tokens(code, edit, kinds=('STRING', 'COMMENT'))

    def edit_doc(self, rel, text, report):
        new = C.replace_word(text, self.old, self.new)
        if new != text:
            report.site(rel, 'doc', text=f'{self.old} -> {self.new}')
        return new

    def finish(self, repo, report):
        if self.mode != 'placebo':
            return
        # placebo: keep the old name as an alias right after the (renamed) definition
        rel, qual = self.defs[0]
        text = C.read(repo, rel)
        mod = cst.parse_module(text)
        parts = qual.split('.')
        alias = cst.parse_statement(f'{self.old} = {self.new}  # old name, kept as an alias\n')
        t = self

        class AddAlias(cst.CSTTransformer):
            def __init__(self):
                self.depth, self.done = 0, False

            def _insert(self, body_list):
                out = []
                for s in body_list:
                    out.append(s)
                    if not self.done and isinstance(s, (cst.FunctionDef, cst.ClassDef)) and s.name.value == t.new:
                        out.append(alias)
                        self.done = True
                return out

            def leave_Module(self, orig, upd):
                if len(parts) == 1:
                    return upd.with_changes(body=self._insert(upd.body))
                return upd

            def leave_ClassDef(self, orig, upd):
                if len(parts) == 2 and upd.name.value == parts[0] and isinstance(upd.body, cst.IndentedBlock):
                    return upd.with_changes(body=upd.body.with_changes(body=self._insert(upd.body.body)))
                return upd
        v = AddAlias()
        new = mod.visit(v).code
        if not v.done:
            raise TrapError('placebo: could not place the alias')
        C.write(repo, rel, new)
        report.files.add(rel)
        report.site(rel, 'alias', text=f'{self.old} = {self.new}')

    def spec(self):
        return dict(super().spec(), old_name=self.old, new_name=self.new)


# ======================================================================== T4 move

class Move(Transform):
    type_id = 'T4'
    title = 'move'
    loud_or_quiet = 'loud'
    PLACEBO_KINDS = {
        'alias': ('the same move and the same import rewrites, plus one re-export line in the old module, so '
                  'the old import path still works (same footprint as the trap)'),
        'reexport': ('the new module only re-exports the object from its old module; every import is pointed '
                     'at the new module the same way, the object stays where it was (the design-doc placebo; '
                     'much smaller than the trap when the object is large)'),
    }

    def __init__(self, defs, new_file, docstring=None, placebo_style='alias'):
        super().__init__(defs)
        self.new_file = new_file
        self.old_mod = C.module_name(self.def_file)
        self.new_mod = C.module_name(new_file)
        self.docstring = docstring
        self.placebo_style = placebo_style
        self.placebo_kind = self.PLACEBO_KINDS[placebo_style]
        self.init.update(new_file=new_file, docstring=docstring, placebo_style=placebo_style)

    def moves(self):
        """True if this run moves the definition (trap, or the 'alias' placebo)."""
        return self.mode == 'trap' or self.placebo_style == 'alias'

    def mentions(self, text):
        return self.name in text

    def describe(self):
        return dict(original_fact=f'{self.name} is defined in and importable from {self.old_mod}',
                    trapped_fact=f'{self.name} moved to {self.new_mod}; {self.old_mod} no longer has it and '
                                 f'every import was updated')

    # -- helpers
    def _resolve_from(self, rel, node):
        """Absolute module of an ImportFrom node written in file rel."""
        level = len(node.relative)
        mod = C.dotted(node.module) if node.module is not None else None
        return C.absolute_module(rel, level, mod)

    def _new_from_module(self, rel, node):
        """How to write the module part of 'from X import name' for the new module, in the same style
        (relative stays relative when the new module sits in the same package as the old one)."""
        level = len(node.relative)
        if level and self.old_mod.rpartition('.')[0] == self.new_mod.rpartition('.')[0]:
            mod = C.dotted(node.module) if node.module is not None else ''
            new_last = self.new_mod.rpartition('.')[2]
            head = mod.rpartition('.')[0] if '.' in mod else ''
            return '.' * level + (head + '.' if head else '') + new_last
        return self.new_mod

    def prepare(self, repo, report):
        if self.qual != self.name:
            raise TrapError('only module-level functions/classes can be moved')
        src = C.read(repo, self.def_file)
        tree = ast.parse(src)
        node = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == self.name), None)
        if node is None:
            raise TrapError(f'{self.name} is not a top-level definition in {self.def_file}')
        # names the definition needs from its old module
        bound_inside = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        bound_inside |= {a.arg for n in ast.walk(node) if isinstance(n, ast.arguments)
                         for a in n.args + n.kwonlyargs + n.posonlyargs + [x for x in (n.vararg, n.kwarg) if x]}
        bound_inside |= {n.name for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                          ast.ClassDef)) and n is not node}
        bound_inside |= {(a.asname or a.name).split('.')[0] for n in ast.walk(node)
                         if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        free = sorted(used - bound_inside - set(dir(builtins)) - {self.name})
        imports, from_old = [], []
        top_defs = {}
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                top_defs[n.name] = n
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                for tg in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    for x in ast.walk(tg):
                        if isinstance(x, ast.Name):
                            top_defs[x.id] = n
        import_of = {}
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)) and n.col_offset == 0:
                for a in n.names:
                    bound = (a.asname or a.name).split('.')[0]
                    import_of[bound] = (n, a)
        for name in free:
            if name in import_of:
                n, a = import_of[name]
                if isinstance(n, ast.ImportFrom):
                    mod = C.absolute_module(self.def_file, n.level, n.module)
                    imports.append(f'from {mod} import {a.name}' + (f' as {a.asname}' if a.asname else ''))
                else:
                    imports.append(f'import {a.name}' + (f' as {a.asname}' if a.asname else ''))
            elif name in top_defs:
                from_old.append(name)
            else:
                report.warn(self.def_file, f'{self.name} uses {name}, not found at module level')
        self.imports_needed = imports
        self.from_old = from_old
        if self.mode == 'placebo' and self.placebo_style == 'alias' and from_old:
            raise TrapError('alias placebo would make an import cycle (the moved object needs names from its old '
                            'module); use placebo_style="reexport"')
        # the old module must not use the moved object itself (it would have to import it back)
        rest_uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == self.name
                     and not (node.lineno <= n.lineno <= node.end_lineno)]
        if rest_uses:
            raise TrapError(f'{self.old_mod} itself uses {self.name} (line {rest_uses[0].lineno}); moving it '
                            'would need an import back (import cycle)')
        if any(isinstance(n, ast.ImportFrom) and n.names[0].name == '*' for n in ast.walk(tree)):
            report.warn(self.def_file, 'old module has a star import')
        # star imports of the old module elsewhere would silently lose the name
        for rel in C.iter_files(repo, ('.py',)):
            text = C.read(repo, rel)
            if 'import *' in text and self.old_mod.rpartition('.')[2] in text:
                try:
                    t2 = ast.parse(text)
                except (SyntaxError, ValueError):
                    continue
                for n in ast.walk(t2):
                    if isinstance(n, ast.ImportFrom) and n.names[0].name == '*' and \
                            C.absolute_module(rel, n.level, n.module) == self.old_mod:
                        report.warn(rel, f'star import of {self.old_mod}: check {self.name} is not used there')
        self.new_exists = __import__('os').path.exists(__import__('os').path.join(repo, self.new_file))
        self._moved_src = None

    def scan(self, rel, mod):
        """Which local names in this file refer to the old module object, and how NAME is used."""
        info = dict(aliases=set(), alias_uses=False, dotted_uses=False)
        t = self

        class V(cst.CSTVisitor):
            def visit_ImportFrom(self, node):
                if isinstance(node.names, cst.ImportStar):
                    return
                base = t._resolve_from(rel, node)
                for a in node.names:
                    if isinstance(a.name, cst.Name) and f'{base}.{a.name.value}' == t.old_mod:
                        info['aliases'].add(a.asname.name.value if a.asname else a.name.value)

            def visit_Import(self, node):
                for a in node.names:
                    if C.dotted(a.name) == t.old_mod and a.asname:
                        info['aliases'].add(a.asname.name.value)

            def visit_Attribute(self, node):
                if C.dotted(node) == f'{t.old_mod}.{t.name}':
                    info['dotted_uses'] = True
                if isinstance(node.value, cst.Name) and node.value.value in info['aliases'] \
                        and node.attr.value == t.name:
                    info['alias_uses'] = True
        mod.visit(V())
        new_last = self.new_mod.rpartition('.')[2]
        code = mod.code
        info['new_alias'] = new_last if not re.search(r'(?<![\w.])' + re.escape(new_last) + r'(?![\w])', code) \
            else new_last + '_module'
        self._scan = info

    def code_transformer(self, rel, report):
        t = self
        info = self._scan
        is_old = rel == self.def_file
        new_pkg, _, new_last = self.new_mod.rpartition('.')
        alias = info['new_alias']
        as_part = '' if alias == new_last else f' as {alias}'

        class X(cst.CSTTransformer):
            def leave_Module(self, orig, upd):
                if not (is_old and t.moves()):
                    return upd
                keep = []
                for st in upd.body:
                    if isinstance(st, (cst.FunctionDef, cst.ClassDef)) and st.name.value == t.name:
                        t._moved_src = cst.Module(body=[st.with_changes(leading_lines=[])]).code
                        report.site(rel, 'definition removed', text=f'{t.name} moved out of {rel}')
                        continue
                    keep.append(st)
                return upd.with_changes(body=keep)

            def leave_SimpleStatementLine(self, orig, upd):
                if len(upd.body) != 1:
                    return upd
                st = upd.body[0]
                extra = []
                if isinstance(st, cst.ImportFrom) and not isinstance(st.names, cst.ImportStar):
                    base = t._resolve_from(rel, st)
                    names = list(st.names)
                    # (a) from <old module> import a, NAME  ->  from <old module> import a
                    #                                           from <new module> import NAME
                    hit = [a for a in names if isinstance(a.name, cst.Name) and a.name.value == t.name]
                    if base == t.old_mod and hit and not is_old:
                        a = hit[0]
                        rest = [x for x in names if x is not a]
                        new_from = t._new_from_module(rel, st)
                        asn = f' as {a.asname.name.value}' if a.asname else ''
                        line = cst.parse_statement(f'from {new_from} import {t.name}{asn}\n')
                        report.site(rel, 'import', text=f'from {new_from} import {t.name}{asn}')
                        if not rest:
                            return line.with_changes(leading_lines=upd.leading_lines)
                        rest[-1] = rest[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
                        upd = upd.with_changes(body=[st.with_changes(names=rest)])
                        extra.append(line)
                    # (b) from <pkg> import <old module>  (NAME used as module.NAME): import the new module too
                    elif info['alias_uses'] and any(isinstance(a.name, cst.Name) and f'{base}.{a.name.value}' == t.old_mod
                                                    for a in names):
                        if base == new_pkg and alias == new_last:
                            last = names[-1]
                            names[-1] = last.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(' ')))
                            names.append(cst.ImportAlias(name=cst.Name(new_last), comma=last.comma))
                            report.site(rel, 'import', text=f'from {new_pkg} import ... {new_last}')
                            return upd.with_changes(body=[st.with_changes(names=names)])
                        extra.append(cst.parse_statement(f'from {new_pkg} import {new_last}{as_part}\n'))
                        report.site(rel, 'import', text=f'from {new_pkg} import {new_last}{as_part}')
                elif isinstance(st, cst.Import):
                    for a in st.names:
                        if C.dotted(a.name) == t.old_mod:
                            if a.asname and info['alias_uses']:
                                extra.append(cst.parse_statement(f'import {t.new_mod} as {alias}\n'))
                            elif not a.asname and info['dotted_uses']:
                                extra.append(cst.parse_statement(f'import {t.new_mod}\n'))
                    if extra:
                        report.site(rel, 'import', text=C.code_of(extra[0]).strip())
                return cst.FlattenSentinel([upd] + extra) if extra else upd

            def leave_Attribute(self, orig, upd):
                if C.dotted(upd) == f'{t.old_mod}.{t.name}':  # written out in full
                    report.site(rel, 'attribute', text=f'{t.new_mod}.{t.name}')
                    return cst.parse_expression(f'{t.new_mod}.{t.name}')
                if isinstance(upd.value, cst.Name) and upd.value.value in info['aliases'] \
                        and upd.attr.value == t.name:  # through a module alias: module.NAME
                    report.site(rel, 'attribute', text=f'{alias}.{t.name}')
                    return upd.with_changes(value=cst.Name(alias))
                return upd
        return X()

    def _dotted_text(self, text):
        """Rewrite old.module.NAME and `from old.module import ..., NAME` written as text."""
        text = re.sub(r'(?<![\w.])' + re.escape(f'{self.old_mod}.{self.name}') + r'(?![\w])',
                      f'{self.new_mod}.{self.name}', text)
        line_re = re.compile(r'^(?P<pre>[ \t]*(?:>>> )?)from\s+' + re.escape(self.old_mod) +
                             r'\s+import\s+(?P<names>[\w ,]+?)[ \t]*$', re.M)

        def fix(m):
            names = [x.strip() for x in m.group('names').split(',') if x.strip()]
            mine = [x for x in names if x.split(' as ')[0].strip() == self.name]
            if not mine:
                return m.group(0)
            rest = [x for x in names if x not in mine]
            new_line = f"{m.group('pre')}from {self.new_mod} import {mine[0]}"
            if not rest:
                return new_line
            return f"{m.group('pre')}from {self.old_mod} import {', '.join(rest)}\n{new_line}"
        return line_re.sub(fix, text)

    def edit_strings(self, rel, code, report):
        def edit(s):
            new = self._dotted_text(s)
            if new != s:
                report.site(rel, 'string/doctest', text=s.strip()[:80])
            return new
        return C.rewrite_string_tokens(code, edit, kinds=('STRING', 'COMMENT'))

    def edit_doc(self, rel, text, report):
        new = self._dotted_text(text)
        if new != text:
            report.site(rel, 'doc', text=f'{self.old_mod}.{self.name} -> {self.new_mod}.{self.name}')
        return new

    def finish(self, repo, report):
        import os
        path = os.path.join(repo, self.new_file)
        doc = self.docstring or f'{self.name}, moved here from {self.old_mod}.'
        if self.moves():
            if self._moved_src is None:
                raise TrapError('definition was not removed from the old module')
            text = C.read(repo, self.def_file)
            if self.mode == 'trap':
                # the old module's __all__ must not list the moved name any more
                text2 = re.sub(r"(__all__\s*=\s*[\[(][^\])]*?)\s*['\"]" + re.escape(self.name) + r"['\"]\s*,?",
                               r'\1', text)
                if text2 != text:
                    C.write(repo, self.def_file, text2)
                    report.site(self.def_file, '__all__', text=f'{self.name} removed from __all__')
            else:
                # alias placebo: the old module re-exports the moved object, so the old import still works
                C.write(repo, self.def_file, _add_top_import(
                    text, f'from {self.new_mod} import {self.name}  # noqa: F401 (old import path)'))
                report.site(self.def_file, 're-export', text=f'from {self.new_mod} import {self.name}')
            lines = list(self.imports_needed)
            if self.from_old:
                lines.append(f'from {self.old_mod} import {", ".join(self.from_old)}')
            body = ('\n'.join(lines) + '\n\n\n' if lines else '\n') + self._moved_src.rstrip('\n') + '\n'
        else:
            body = f'from {self.old_mod} import {self.name}  # noqa: F401\n'
        if self.new_exists:
            old = C.read(repo, self.new_file)
            if self.moves():
                # needed imports go after the module's last top-level import, skipping names the module
                # already binds and imports from the module itself
                bound = _module_level_names(old)
                lines = []
                for ln in list(self.imports_needed) + \
                        ([f'from {self.old_mod} import {", ".join(self.from_old)}'] if self.from_old else []):
                    m = re.match(r'from (\S+) import (.+)$', ln)
                    if m and m.group(1) == self.new_mod:
                        continue
                    names = [x.strip().split(' as ')[-1] for x in m.group(2).split(',')] if m else \
                        [ln.split()[-1].split('.')[0]]
                    if all(n in bound for n in names):
                        continue
                    lines.append(ln)
                for ln in lines:
                    old = _add_top_import(old, ln)
                body = self._moved_src.rstrip('\n') + '\n'
            C.write(repo, self.new_file, old.rstrip('\n') + '\n\n\n' + body)
        else:
            C.write(repo, self.new_file, f'"""{doc}"""\n' + body)
        report.files.add(self.new_file)
        report.site(self.new_file, 'new module' if not self.new_exists else 'module', text=self.new_file)

    def spec(self):
        return dict(super().spec(), name=self.name, old_module=self.old_mod, new_module=self.new_mod,
                    new_file=self.new_file)


# ======================================================================== T5 exception swap

EXC_ASSERTS = {'assertRaises', 'assertRaisesMessage', 'assertRaisesRegex', 'assertRaisesRegexp', 'raises',
               'assertWarns'}


class ExceptionSwap(Transform):
    type_id = 'T5'
    title = 'exception swap'
    loud_or_quiet = 'quiet'
    placebo_kind = 'the same edit, but the new exception class also subclasses the old one'

    def __init__(self, defs, old_exc, new_exc, new_base, exc_file=None, closure_depth=1, doc=None):
        super().__init__(defs)
        self.old_exc, self.new_exc, self.new_base = old_exc, new_exc, new_base
        self.exc_file = exc_file or self.def_file
        self.exc_mod = C.module_name(self.exc_file)
        self.closure_depth = closure_depth
        self.doc = doc or f'Raised by {self.name}() when it cannot do its job.'
        self.init.update(old_exc=old_exc, new_exc=new_exc, new_base=new_base, exc_file=self.exc_file,
                         closure_depth=closure_depth, doc=self.doc)

    def mentions(self, text):
        return self.old_exc in text or self.new_exc in text

    def describe(self):
        return dict(original_fact=f'{self.qual} raises {self.old_exc}',
                    trapped_fact=f'{self.qual} raises {self.new_exc} (a new {self.new_base} subclass, not a '
                                 f'{self.old_exc}); excepts and tests around it also accept {self.new_exc}')

    def prepare(self, repo, report):
        self.check_unique(repo, report)
        if count_definitions(repo, self.new_exc):
            raise TrapError(f'{self.new_exc} already exists')
        fn = find_def(repo, self.def_file, self.qual)
        code = C.code_of(fn)
        if not re.search(r'\braise\s+(?:[\w.]+\.)?' + re.escape(self.old_exc) + r'\b', code):
            raise TrapError(f'{self.qual} has no raise {self.old_exc}')
        # closure: functions that call f (or a function in the closure) without catching old_exc
        names = {self.name}
        for _ in range(self.closure_depth):
            more = set()
            for rel in C.iter_files(repo, ('.py',)):
                if C.is_test_path(rel):
                    continue
                text = C.read(repo, rel)
                if not any(n in text for n in names):
                    continue
                try:
                    tree = ast.parse(text)
                except (SyntaxError, ValueError):
                    continue
                for fdef in ast.walk(tree):
                    if isinstance(fdef, (ast.FunctionDef, ast.AsyncFunctionDef)) and fdef.name not in names \
                            and _calls_outside_handler(fdef, names, self.old_exc):
                        more.add(fdef.name)
            if not more:
                break
            names |= more
        self.closure = names
        report.warn(self.def_file, f'closure ({len(names)} names): {sorted(names)[:30]}')

    def _handler_catches(self, h):
        if h.type is None:
            return False
        types = [e.value for e in h.type.elements] if isinstance(h.type, cst.Tuple) else [h.type]
        return any((C.dotted(x) or '').split('.')[-1] == self.old_exc for x in types)

    def code_transformer(self, rel, report):
        t = self

        class X(QualTransformer):
            def __init__(self, rel):
                super().__init__(rel)
                self.used_new = False
                self.try_stack = []

            def on_function(self, orig, upd, qual):
                if (rel, qual) in t.defs:
                    class R(cst.CSTTransformer):
                        def leave_Raise(self, o, u):
                            exc = u.exc
                            target = exc.func if isinstance(exc, cst.Call) else exc
                            if (C.dotted(target) or '').split('.')[-1] == t.old_exc:
                                new_t = cst.Name(t.new_exc)
                                report.site(rel, 'raise', text=f'raise {t.new_exc}')
                                return u.with_changes(exc=exc.with_changes(func=new_t) if isinstance(exc, cst.Call)
                                                      else new_t)
                            return u
                    self.used_new = True
                    return upd.with_changes(body=upd.body.visit(R()))
                return upd

            def leave_Try(self, orig, upd):
                body_calls = _cst_call_names(orig.body)
                if not (body_calls & t.closure):
                    return upd
                handlers = []
                for h in upd.handlers:
                    if t._handler_catches(h):
                        types = list(h.type.elements) if isinstance(h.type, cst.Tuple) else \
                            [cst.Element(value=h.type)]
                        if not any((C.dotted(e.value) or '').split('.')[-1] == t.new_exc for e in types):
                            types = [e.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(' ')))
                                     for e in types[:-1]] + [types[-1].with_changes(
                                         comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(' ')))]
                            types.append(cst.Element(value=cst.Name(t.new_exc)))
                            h = h.with_changes(type=cst.Tuple(elements=types))
                            self.used_new = True
                            report.site(rel, 'except', h.with_changes(body=cst.IndentedBlock(body=[])))
                    handlers.append(h)
                return upd.with_changes(handlers=handlers)

            def leave_With(self, orig, upd):
                # with self.assertRaises(E): <calls>   /   with raises(E): <calls>
                body_calls = _cst_call_names(orig.body)
                if not (body_calls & t.closure):
                    return upd
                items = []
                for it in upd.items:
                    items.append(t._fix_assert_call(it.item, rel, report, self) if isinstance(it.item, cst.Call)
                                 else it.item)
                return upd.with_changes(items=[it.with_changes(item=x) for it, x in zip(upd.items, items)])

            def leave_Call(self, orig, upd):
                # self.assertRaises(E, f, ...) / raises(E, lambda: f(...)) : call form
                name = C.call_name(upd)
                if name in EXC_ASSERTS and len(upd.args) >= 2:
                    rest_calls = set()
                    for a in orig.args[1:]:
                        rest_calls |= _cst_call_names(a.value)
                        d = C.dotted(a.value)
                        if d:
                            rest_calls.add(d.split('.')[-1])
                    if rest_calls & t.closure:
                        return t._fix_assert_call(upd, rel, report, self)
                return upd
        return X(rel)

    def _fix_assert_call(self, call, rel, report, visitor):
        if C.call_name(call) not in EXC_ASSERTS or not call.args:
            return call
        first = call.args[0]
        if first.keyword is not None:
            return call
        v = first.value
        elems = list(v.elements) if isinstance(v, cst.Tuple) else [cst.Element(value=v)]
        names = [(C.dotted(e.value) or '').split('.')[-1] for e in elems]
        if self.old_exc not in names or self.new_exc in names:
            return call
        elems = [e.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(' '))) for e in elems]
        elems.append(cst.Element(value=cst.Name(self.new_exc)))
        visitor.used_new = True
        new = call.with_changes(args=[first.with_changes(value=cst.Tuple(elements=elems))] + list(call.args[1:]))
        report.site(rel, 'test assertion', text=C.code_of(new)[:100])
        return new

    def finish(self, repo, report):
        # 1. define the new class in exc_file, after the old exception's class (or after the imports)
        text = C.read(repo, self.exc_file)
        mod = cst.parse_module(text)
        bases = self.new_base
        if self.mode == 'placebo':
            bases = self.old_exc if self.new_base in _builtin_mro(self.old_exc) else f'{self.new_base}, {self.old_exc}'
        cls = cst.parse_statement(f'class {self.new_exc}({bases}):\n    """{self.doc}"""\n')
        body = list(mod.body)
        pos = None
        for i, s in enumerate(body):
            if isinstance(s, cst.ClassDef) and s.name.value == self.old_exc:
                pos = i + 1
        if pos is None:  # after the last top-level import
            for i, s in enumerate(body):
                if isinstance(s, cst.SimpleStatementLine) and isinstance(s.body[0], (cst.Import, cst.ImportFrom)):
                    pos = i + 1
            pos = pos or 0
        blank = cst.EmptyLine()
        cls = cls.with_changes(leading_lines=[blank, blank])
        body.insert(pos, cls)
        if pos < len(body) - 1 and isinstance(body[pos + 1], (cst.ClassDef, cst.FunctionDef)):
            body[pos + 1] = body[pos + 1].with_changes(leading_lines=[blank, blank] +
                                                       [x for x in body[pos + 1].leading_lines if x.comment])
        C.write(repo, self.exc_file, mod.with_changes(body=body).code)
        report.files.add(self.exc_file)
        report.site(self.exc_file, 'new class', text=f'class {self.new_exc}({bases})')
        # 2. every other file that now names the new exception needs it imported
        for rel in sorted(report.files):
            if rel == self.exc_file or not rel.endswith('.py'):
                continue
            code = C.read(repo, rel)
            if not re.search(r'(?<![\w.])' + re.escape(self.new_exc) + r'\b', code):
                continue
            new = _add_import_next_to(code, rel, self.exc_mod, self.new_exc, prefer=[self.old_exc] +
                                      sorted(self.closure))
            if new is None:
                report.warn(rel, f'could not find where to import {self.new_exc}; added a top-level import')
                new = _add_top_import(code, f'from {self.exc_mod} import {self.new_exc}')
            if new != code:
                C.write(repo, rel, new)
                report.site(rel, 'import', text=f'import {self.new_exc}')

    def spec(self):
        return dict(super().spec(), old_exc=self.old_exc, new_exc=self.new_exc, new_base=self.new_base,
                    exc_file=self.exc_file, closure=sorted(self.closure))


def _builtin_mro(name):
    """Names of the classes in the MRO of builtin exception `name` (empty if not a builtin)."""
    cls = getattr(builtins, name, None)
    return {c.__name__ for c in cls.__mro__} if isinstance(cls, type) else set()


def _calls_outside_handler(fdef, names, exc):
    """True if function fdef calls one of `names` outside a try that catches `exc`."""
    def caught(node_stack):
        for t in node_stack:
            for h in t.handlers:
                if h.type is None:
                    return True
                types = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
                if any((ast.unparse(x).split('.')[-1]) in (exc, 'Exception', 'BaseException') for x in types):
                    return True
        return False

    def walk(node, stack):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Call):
                f = child.func
                n = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                if n in names and not caught(stack):
                    return True
            if isinstance(child, ast.Try):
                inner = stack + [child]
                if any(walk(b, inner) for b in child.body):
                    return True
                for part in child.handlers + child.orelse + child.finalbody:
                    if walk(part, stack):
                        return True
                continue
            if walk(child, stack):
                return True
        return False
    return walk(fdef, [])


def _cst_call_names(node):
    """Names of all calls inside a libcst node (not looking into nested defs/classes/lambdas)."""
    out = set()

    class V(cst.CSTVisitor):
        def visit_FunctionDef(self, n):
            return False

        def visit_ClassDef(self, n):
            return False

        def visit_Call(self, n):
            nm = C.call_name(n)
            if nm:
                out.add(nm)

        def visit_Lambda(self, n):
            return True
    if isinstance(node, cst.CSTNode):
        node.visit(V())
    return out


def _add_import_next_to(code, rel, module, name, prefer):
    """Add `name` to an existing `from <module> import ...` statement (top-level or inside a function)
    that already imports one of `prefer` (or anything) from that module. None if there is none."""
    try:
        mod = cst.parse_module(code)
    except Exception:
        return None
    done = [False]

    class X(cst.CSTTransformer):
        def leave_ImportFrom(self, orig, upd):
            if done[0] or isinstance(upd.names, cst.ImportStar):
                return upd
            level = len(upd.relative)
            m = C.absolute_module(rel, level, C.dotted(upd.module) if upd.module is not None else None)
            if m != module:
                return upd
            have = [a.name.value for a in upd.names if isinstance(a.name, cst.Name)]
            if name in have:
                done[0] = True
                return upd
            names = list(upd.names)
            last = names[-1]
            trailing = not isinstance(last.comma, cst.MaybeSentinel)
            names[-1] = last.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(' ')))
            new_alias = cst.ImportAlias(name=cst.Name(name), comma=last.comma if trailing else cst.MaybeSentinel.DEFAULT)
            done[0] = True
            return upd.with_changes(names=names + [new_alias])
    new = mod.visit(X()).code
    return new if done[0] else None


def _module_level_names(code):
    """Names bound at the top level of a module (defs, classes, assignments, imports)."""
    out = set()
    for n in ast.parse(code).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split('.')[0] for a in n.names}
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for tg in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                out |= {x.id for x in ast.walk(tg) if isinstance(x, ast.Name)}
    return out


def _add_top_import(code, line):
    """Add an import line after the last top-level import of a module."""
    mod = cst.parse_module(code)
    body = list(mod.body)
    pos = 0
    for i, s in enumerate(body):
        if isinstance(s, cst.SimpleStatementLine) and isinstance(s.body[0], (cst.Import, cst.ImportFrom)):
            pos = i + 1
    body.insert(pos, cst.parse_statement(line + '\n'))
    return mod.with_changes(body=body).code


# ======================================================================== T6 return convention

class ReturnSwap(Transform):
    type_id = 'T6'
    title = 'return convention'
    loud_or_quiet = 'quiet'
    placebo_kind = 'the same return statements and unpacking lines, only wrapped in parentheses'

    def __init__(self, defs, size=2, i=0, j=1):
        super().__init__(defs)
        self.size, self.i, self.j = size, i, j
        self.init.update(size=size, i=i, j=j)

    def describe(self):
        return dict(original_fact=f'{self.qual} returns a {self.size}-tuple with item {self.i} first',
                    trapped_fact=f'{self.qual} returns the tuple with items {self.i} and {self.j} swapped; every '
                                 f'caller unpacks it the new way')

    def prepare(self, repo, report):
        self.check_unique(repo, report)
        for rel, qual in self.defs:
            fn = find_def(repo, rel, qual)
            tree = ast.parse(C.code_of(fn).strip())
            f = tree.body[0]
            for r in _returns(f):
                if r.value is None or not (isinstance(r.value, ast.Tuple) and len(r.value.elts) == self.size):
                    callee = r.value.func if isinstance(r.value, ast.Call) else None
                    cname = callee.id if isinstance(callee, ast.Name) else callee.attr if \
                        isinstance(callee, ast.Attribute) else None
                    if cname not in self.call_names:
                        raise TrapError(f'{qual} has a return that is not a {self.size}-tuple: line {r.lineno}')

    def _swap_elements(self, tup):
        els = list(tup.elements)
        a, b = els[self.i], els[self.j]
        els[self.i] = b.with_changes(comma=a.comma)
        els[self.j] = a.with_changes(comma=b.comma)
        return tup.with_changes(elements=els)

    def _paren(self, tup):
        if tup.lpar:
            return None
        return tup.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])

    def code_transformer(self, rel, report):
        t = self

        class X(QualTransformer):
            def __init__(self, rel):
                super().__init__(rel)
                self.in_target = 0

            def visit_FunctionDef(self, node):
                super().visit_FunctionDef(node)
                if (rel, '.'.join(self.stack)) in t.defs:
                    self.in_target += 1

            def on_function(self, orig, upd, qual):
                if (rel, qual) in t.defs:
                    self.in_target -= 1
                return upd

            def leave_Return(self, orig, upd):
                if self.in_target and isinstance(upd.value, cst.Tuple) and len(upd.value.elements) == t.size:
                    new_val = t._swap_elements(upd.value) if t.mode == 'trap' else t._paren(upd.value)
                    if new_val is not None:
                        report.site(rel, 'return', text=C.code_of(new_val)[:80])
                        return upd.with_changes(value=new_val)
                return upd

            def leave_Assign(self, orig, upd):
                # a, b = f(...)  ->  b, a = f(...)
                if len(upd.targets) == 1 and isinstance(upd.value, cst.Call) and \
                        C.call_name(upd.value) in t.call_names:
                    tg = upd.targets[0].target
                    if isinstance(tg, (cst.Tuple, cst.List)) and len(tg.elements) == t.size:
                        new_tg = t._swap_elements(tg) if t.mode == 'trap' else (
                            t._paren(tg) if isinstance(tg, cst.Tuple) else None)
                        if new_tg is not None:
                            report.site(rel, 'unpack', text=C.code_of(new_tg))
                            return upd.with_changes(targets=[upd.targets[0].with_changes(target=new_tg)])
                return upd

            def leave_Subscript(self, orig, upd):
                # f(...)[0]  ->  f(...)[1]
                if isinstance(upd.value, cst.Call) and C.call_name(upd.value) in t.call_names \
                        and len(upd.slice) == 1 and isinstance(upd.slice[0].slice, cst.Index):
                    idx = upd.slice[0].slice.value
                    if isinstance(idx, cst.Integer) and t.mode == 'trap':
                        k = int(idx.value)
                        new_k = t.j if k == t.i else t.i if k == t.j else k
                        if new_k != k:
                            report.site(rel, 'index', text=f'[{k}] -> [{new_k}]')
                            return upd.with_changes(slice=[upd.slice[0].with_changes(
                                slice=cst.Index(value=cst.Integer(str(new_k))))])
                    elif isinstance(idx, cst.Integer):  # placebo: index written with explicit parentheses
                        report.site(rel, 'index', text=f'[{idx.value}]')
                        return upd.with_changes(value=upd.value.with_changes(
                            lpar=[cst.LeftParen()], rpar=[cst.RightParen()]))
                return upd

            def leave_Comparison(self, orig, upd):
                # f(...) == (a, b)  ->  f(...) == (b, a)   (common in tests)
                if isinstance(upd.left, cst.Call) and C.call_name(upd.left) in t.call_names \
                        and len(upd.comparisons) == 1:
                    comp = upd.comparisons[0]
                    if isinstance(comp.operator, (cst.Equal, cst.NotEqual)) and \
                            isinstance(comp.comparator, cst.Tuple) and len(comp.comparator.elements) == t.size:
                        new_c = t._swap_elements(comp.comparator) if t.mode == 'trap' else None
                        if t.mode == 'placebo':  # same line touched: the call gets parentheses
                            if upd.left.lpar:
                                return upd
                            report.site(rel, 'compare', text=C.code_of(comp.comparator))
                            return upd.with_changes(left=upd.left.with_changes(
                                lpar=[cst.LeftParen()], rpar=[cst.RightParen()]))
                        report.site(rel, 'compare', text=C.code_of(new_c))
                        return upd.with_changes(comparisons=[comp.with_changes(comparator=new_c)])
                return upd

            def leave_Call(self, orig, upd):
                # self.assertEqual(f(...), (a, b))
                if C.call_name(upd) in ('assertEqual', 'assertEquals') and len(upd.args) >= 2:
                    a0, a1 = upd.args[0].value, upd.args[1].value
                    for k, (x, y) in enumerate(((a0, a1), (a1, a0))):
                        if isinstance(x, cst.Call) and C.call_name(x) in t.call_names and \
                                isinstance(y, cst.Tuple) and len(y.elements) == t.size:
                            if t.mode == 'placebo':  # same line touched: the call gets parentheses
                                if x.lpar:
                                    return upd
                                args = list(upd.args)
                                pos = 0 if k == 0 else 1
                                args[pos] = args[pos].with_changes(value=x.with_changes(
                                    lpar=[cst.LeftParen()], rpar=[cst.RightParen()]))
                                report.site(rel, 'assertEqual', text=C.code_of(y))
                                return upd.with_changes(args=args)
                            args = list(upd.args)
                            pos = 1 if k == 0 else 0
                            args[pos] = args[pos].with_changes(value=t._swap_elements(y))
                            report.site(rel, 'assertEqual', text=C.code_of(args[pos].value))
                            return upd.with_changes(args=args)
                return upd
        return X(rel)

    def _doctest_editors(self, rel, report):
        """(edit_source, edit_output) for doctest examples: `>>> a, b = f(x)` lines are rewritten like
        code, and the printed tuple after `>>> f(x)` is swapped."""
        def edit_source(src):
            if not any(n + '(' in src for n in self.call_names):
                return None
            if self.mode == 'placebo' and self._bare_call(src):
                return '(' + src + ')'  # same example touched as in the trap (whose output changes)
            try:
                m = cst.parse_module(src)
            except Exception:
                return None
            return m.visit(self.code_transformer(rel, report)).code

        def edit_output(src, outs):
            if self.mode != 'trap' or len(outs) != 1:
                return None
            if not self._bare_call(src):
                return None
            line = outs[0]
            ind = line[:len(line) - len(line.lstrip())]
            body = line.strip()
            if not (body.startswith('(') and body.endswith(')')):
                return None
            res = C.split_call_args(body, 1)
            if res is None or res[0] != len(body) - 1:
                return None
            cores = [p[1] for p in res[1]]
            if len(cores) != self.size:
                return None
            cores[self.i], cores[self.j] = cores[self.j], cores[self.i]
            report.site(rel, 'doctest output', text=body)
            return [ind + '(' + ', '.join(cores) + ')']
        return edit_source, edit_output

    def _bare_call(self, src):
        """True if a doctest example is just a call to the target, like `>>> f(x)`."""
        try:
            e = ast.parse(src.strip(), mode='eval').body
        except SyntaxError:
            return False
        return isinstance(e, ast.Call) and ((isinstance(e.func, ast.Name) and e.func.id in self.call_names) or
                                            (isinstance(e.func, ast.Attribute) and e.func.attr in self.call_names))

    def edit_strings(self, rel, code, report):
        new, _ = C.rewrite_docstring_doctests(code, *self._doctest_editors(rel, report))
        return new

    def edit_doc(self, rel, text, report):
        new, _ = C.rewrite_doctests_in_text(text, *self._doctest_editors(rel, report))
        return new

    def spec(self):
        return dict(super().spec(), size=self.size, i=self.i, j=self.j)


def _returns(fdef):
    """Return statements of a function (not of nested functions)."""
    out, todo = [], list(fdef.body)
    while todo:
        n = todo.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(n, ast.Return):
            out.append(n)
        todo.extend(ast.iter_child_nodes(n))
    return out


# ======================================================================== T7 keyword rename / T8 flag

class _ParamRenamer(cst.CSTTransformer):
    """Inside one function: rename parameter `old` to `new` (or, with invert=True, replace every use of
    `old` by `not new`). Keyword names of calls and attribute names are left alone."""

    def __init__(self, old, new, invert=False):
        super().__init__()
        self.old, self.new, self.invert = old, new, invert
        self.made = set()
        self.skip = set()  # Name nodes that are keyword names, attribute names or parameter names

    def visit_Arg(self, node):
        if node.keyword is not None:
            self.skip.add(id(node.keyword))

    def visit_Attribute(self, node):
        self.skip.add(id(node.attr))

    def visit_Param(self, node):
        self.skip.add(id(node.name))

    def leave_Param(self, orig, upd):
        if orig.name.value == self.old:
            ch = dict(name=cst.Name(self.new))
            if self.invert and upd.default is not None:
                ch['default'] = C.negate(upd.default)
            return upd.with_changes(**ch)
        return upd

    def leave_Name(self, orig, upd):
        if upd.value != self.old or id(orig) in self.skip:
            return upd
        if not self.invert:
            return upd.with_changes(value=self.new)
        node = cst.UnaryOperation(operator=cst.Not(), expression=cst.Name(self.new),
                                  lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        self.made.add(id(node))
        return node

    def leave_UnaryOperation(self, orig, upd):
        # not (not new)  ->  new
        e = upd.expression
        if isinstance(upd.operator, cst.Not) and id(e) in self.made:
            return cst.Name(self.new, lpar=upd.lpar, rpar=upd.rpar)
        return upd

    def _strip(self, node):
        if id(node) in self.made:
            new = node.with_changes(lpar=[], rpar=[])
            self.made.add(id(new))
            return new
        return node

    # contexts where `not new` needs no parentheses
    def leave_If(self, orig, upd):
        return upd.with_changes(test=self._strip(upd.test))

    def leave_While(self, orig, upd):
        return upd.with_changes(test=self._strip(upd.test))

    def leave_Assign(self, orig, upd):
        return upd.with_changes(value=self._strip(upd.value))

    def leave_Return(self, orig, upd):
        return upd.with_changes(value=self._strip(upd.value)) if upd.value is not None else upd

    def leave_IfExp(self, orig, upd):
        return upd.with_changes(test=self._strip(upd.test))

    def leave_BooleanOperation(self, orig, upd):
        return upd.with_changes(left=self._strip(upd.left), right=self._strip(upd.right))


def _insert_after_docstring(fn, stmts):
    body = fn.body
    if not isinstance(body, cst.IndentedBlock):
        raise TrapError('one-line function bodies are not supported')
    items = list(body.body)
    i = 0
    if items and isinstance(items[0], cst.SimpleStatementLine) and isinstance(items[0].body[0], cst.Expr) and \
            isinstance(items[0].body[0].value, (cst.SimpleString, cst.ConcatenatedString)):
        i = 1
    return fn.with_changes(body=body.with_changes(body=items[:i] + stmts + items[i:]))


def _add_kwonly(params, name, default='None'):
    """Add a keyword-only parameter `name=default` (after *args, or after a bare *)."""
    p = cst.Param(name=cst.Name(name), default=cst.parse_expression(default), equal=C.EQ)
    kw = list(params.kwonly_params)
    star = params.star_arg
    ch = {}
    if isinstance(star, cst.MaybeSentinel):
        ch['star_arg'] = cst.ParamStar()
    # keep commas consistent: libcst adds commas between parameters when they are missing
    kw = [k.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(' '))) if isinstance(k.comma, cst.MaybeSentinel)
          else k for k in kw]
    ch['kwonly_params'] = kw + [p]
    return params.with_changes(**ch)


class KeywordRename(Transform):
    type_id = 'T7'
    title = 'keyword rename'
    loud_or_quiet = 'loud'
    placebo_kind = 'the same rename everywhere, but the function still accepts the old keyword'

    def __init__(self, defs, param, new_param, call_names=None):
        super().__init__(defs, call_names)
        self.param, self.new_param = param, new_param
        self.init.update(param=param, new_param=new_param)

    def describe(self):
        return dict(original_fact=f'{self.qual} takes a keyword argument {self.param}=',
                    trapped_fact=f'{self.qual} calls it {self.new_param}= now; {self.param}= raises TypeError')

    def prepare(self, repo, report):
        self.check_unique(repo, report)
        for rel, qual in self.defs:
            fn = find_def(repo, rel, qual)
            names = [p.name.value for p in positional_params(fn)] + [p.name.value for p in fn.params.kwonly_params]
            if self.param not in names:
                raise TrapError(f'{qual} has no parameter {self.param}')
            if self.new_param in names:
                raise TrapError(f'{qual} already has a parameter {self.new_param}')
            body = C.code_of(fn.body)
            if re.search(r'(?<![\w.])' + re.escape(self.new_param) + r'(?![\w])', body):
                raise TrapError(f'{qual} already uses the name {self.new_param} in its body')
        self.check_escapes(repo, report)

    def _def_edit(self, upd, invert=False):
        new = upd.with_changes(params=upd.params.visit(_ParamRenamer(self.param, self.new_param, invert)),
                               body=upd.body.visit(_ParamRenamer(self.param, self.new_param, invert)))
        if self.mode == 'placebo':
            new = new.with_changes(params=_add_kwonly(new.params, self.param))
            line = (f'{self.new_param} = not {self.param}' if invert else f'{self.new_param} = {self.param}')
            stmt = cst.parse_statement(f'if {self.param} is not None:  # old name, still accepted\n    {line}\n')
            new = _insert_after_docstring(new, [stmt])
        return new

    def _docstring_edit(self, fn):
        """Backticked mentions of the old parameter name in the function's own docstring."""
        body = fn.body
        if not isinstance(body, cst.IndentedBlock) or not body.body:
            return fn
        first = body.body[0]
        if not (isinstance(first, cst.SimpleStatementLine) and isinstance(first.body[0], cst.Expr)
                and isinstance(first.body[0].value, cst.SimpleString)):
            return fn
        doc = first.body[0].value
        new_val = re.sub(r'(`+)' + re.escape(self.param) + r'(`+)', r'\g<1>' + self.new_param + r'\g<2>', doc.value)
        if new_val == doc.value:
            return fn
        new_first = first.with_changes(body=[first.body[0].with_changes(value=doc.with_changes(value=new_val))])
        return fn.with_changes(body=body.with_changes(body=[new_first] + list(body.body[1:])))

    def edit_call_args(self, call):
        changed = False
        args = []
        for a in call.args:
            if a.keyword is not None and a.keyword.value == self.param:
                a = a.with_changes(keyword=cst.Name(self.new_param))
                changed = True
            args.append(a)
        return args if changed else None

    def code_transformer(self, rel, report):
        t = self

        class X(QualTransformer):
            def on_function(self, orig, upd, qual):
                if (rel, qual) in t.defs:
                    report.site(rel, 'definition', text=f'{qual}(..., {t.new_param}=...)')
                    new = t._def_edit(upd)
                    try:
                        return t._docstring_edit(new)
                    except Exception:
                        return new
                return upd

            def leave_Call(self, orig, upd):
                if C.call_name(upd) not in t.call_names:
                    return upd
                if any(a.star == '**' for a in upd.args):
                    report.warn(rel, f'call with **kwargs: check {t.param} is not inside: {C.code_of(upd)[:100]}')
                args = t.edit_call_args(upd)
                if args is None:
                    return upd
                new = upd.with_changes(args=args)
                report.site(rel, 'call', new)
                return new
        return X(rel)

    def edit_cores(self, cores, before):
        out = []
        for c in cores:
            m = re.match(r'^(\**)' + re.escape(self.param) + r'(\s*(?::[^=]*)?=)', c)
            out.append(m.group(1) + self.new_param + c[m.end(1) + len(self.param):] if m else c)
        return out if out != cores else None

    def _text(self, rel, text, report, kind):
        for n in self.call_names:
            text, k = C.rewrite_text_calls(text, n, self.edit_cores)
            for _ in range(k):
                report.site(rel, kind, text=f'{n}(...) in text')
        return text

    def edit_strings(self, rel, code, report):
        def edit(s):
            return self._text(rel, s, report, 'doctest/string') if any(n + '(' in s for n in self.call_names) else s
        return C.rewrite_string_tokens(code, edit)

    def edit_doc(self, rel, text, report):
        return self._text(rel, text, report, 'doc')

    def spec(self):
        return dict(super().spec(), param=self.param, new_param=self.new_param)


class FlagInvert(KeywordRename):
    type_id = 'T8'
    title = 'flag inversion'
    loud_or_quiet = 'loud by name (old keyword raises TypeError), quiet by value (positional flags)'
    placebo_kind = 'the same edit everywhere, but the function still accepts the old flag (translated)'

    def describe(self):
        return dict(original_fact=f'{self.qual} has a flag {self.param}={self.old_default}',
                    trapped_fact=f'the flag is now {self.new_param}={self.new_default} with the opposite meaning; '
                                 f'callers pass the negated value')

    def prepare(self, repo, report):
        super().prepare(repo, report)
        self.index_by_name, defaults = {}, set()
        for rel, qual in self.defs:
            fn, is_method, static = def_info(repo, rel, qual)
            skip = 1 if is_method and not static else 0
            if (rel, qual) == self.defs[0]:
                self.is_method, self.static = is_method, static
            params = positional_params(fn)
            names = [p.name.value for p in params[skip:]]
            allp = {p.name.value: p for p in params + list(fn.params.kwonly_params)}
            p = allp[self.param]
            if p.default is not None:
                if C.bool_value(p.default) is None:
                    raise TrapError(f'{qual}: {self.param} does not default to True/False')
                defaults.add(C.code_of(p.default).strip())
            # where a caller passes the flag by position (None: keyword-only)
            self.index_by_name[qual.split('.')[-1]] = names.index(self.param) if self.param in names else None
            # the flag must not be reassigned inside the function (its meaning would be inverted twice)
            tree = ast.parse(C.code_of(fn).strip())
            if any(isinstance(n, ast.Name) and n.id == self.param and isinstance(n.ctx, ast.Store)
                   for n in ast.walk(tree)):
                raise TrapError(f'{self.param} is reassigned inside {qual}')
        if len(defaults) != 1:
            raise TrapError(f'the flag must default to the same True/False in the family: {defaults}')
        self.old_default = defaults.pop()
        self.new_default = 'False' if self.old_default == 'True' else 'True'

    def code_transformer(self, rel, report):
        t = self

        class X(QualTransformer):
            def on_function(self, orig, upd, qual):
                if (rel, qual) in t.defs:
                    report.site(rel, 'definition', text=f'{qual}(..., {t.new_param}={t.new_default})')
                    new = t._def_edit(upd, invert=True)
                    try:
                        return t._docstring_edit(new)
                    except Exception:
                        return new
                return upd

            def leave_Call(self, orig, upd):
                if C.call_name(upd) not in t.call_names:
                    return upd
                args = list(upd.args)
                changed = False
                offset = _bound_offset(upd, t, t.is_method, t.static)
                pos = [i for i, a in enumerate(args) if C.is_positional(a)]
                for i, a in enumerate(args):
                    if a.keyword is not None and a.keyword.value == t.param:
                        args[i] = a.with_changes(keyword=cst.Name(t.new_param), value=C.negate(a.value))
                        changed = True
                index = t.index_by_name.get(C.call_name(upd))
                if index is not None and len(pos) > offset + index and not any(a.star == '*' for a in args):
                    k = pos[offset + index]
                    args[k] = args[k].with_changes(value=C.negate(args[k].value))
                    changed = True
                if any(a.star == '**' for a in args):
                    report.warn(rel, f'call with **kwargs: check {t.param} is not inside: {C.code_of(upd)[:100]}')
                if not changed:
                    return upd
                new = upd.with_changes(args=args)
                report.site(rel, 'call', new)
                return new
        return X(rel)

    def edit_cores(self, cores, before):
        out = []
        sig = C.is_signature_context(before)
        for c in cores:
            m = re.match(r'^' + re.escape(self.param) + r'(\s*(?::[^=]*)?=\s*)(.*)$', c)
            if m:
                val = m.group(2).strip()
                neg = {'True': 'False', 'False': 'True'}.get(val, f'not ({val})' if not sig else val)
                out.append(self.new_param + m.group(1) + neg)
            else:
                out.append(c)
        return out if out != cores else None

    def spec(self):
        return dict(super().spec(), old_default=self.old_default, new_default=self.new_default)


TYPES = {'T1': ArgSwap, 'T2': DefaultChange, 'T3': Rename, 'T4': Move, 'T5': ExceptionSwap,
         'T6': ReturnSwap, 'T7': KeywordRename, 'T8': FlagInvert}


def make(spec):
    """Rebuild a transform from its spec(): {'type': 'T3', 'defs': [[file, qual]], 'init': {...}}.
    init may also hold 'prose_edits': [[file, old_text, new_text], ...] (trap mode only)."""
    init = dict(spec.get('init', {}))
    prose = init.pop('prose_edits', [])
    t = TYPES[spec['type']]([tuple(d) for d in spec['defs']], **init)
    for rel, old, new in prose:
        t.add_prose_edit(rel, old, new)
    return t
