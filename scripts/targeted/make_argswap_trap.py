"""Build an argument-order memory trap (or its placebo) in a checked-out repo, in place.

trap     swap the first two parameters of FUNC in its definition, and rewrite every call so behavior
         is unchanged: f(a, b) -> f(b, a); f(a, x=..) -> f(x=.., OLDFIRST=a). Docs that spell the
         old signature are updated too, so nothing in the repo still shows the old order.
placebo  touch the same call sites without contradicting memory: f(a, b) -> f(p0=a, p1=b), in the
         original order, and leave the definition alone. Same footprint as the trap, but every
         call written from memory is still correct.

Usage: python -I make_argswap_trap.py REPO_DIR DEF_FILE FUNC trap|placebo
"""
import io, os, re, sys, tokenize
import libcst as cst

repo, def_file, func, mode = sys.argv[1:5]
src = open(os.path.join(repo, def_file)).read()
mod = cst.parse_module(src)


class FindDef(cst.CSTVisitor):
    def __init__(self):
        self.params = None

    def visit_FunctionDef(self, node):
        if node.name.value == func and self.params is None:
            self.params = [p.name.value for p in node.params.params if p.name.value not in ('self', 'cls')]


fd = FindDef()
mod.visit(fd)
p0, p1 = fd.params[:2]
new_names = {p0: p0 + '_', p1: p1 + '_'} if mode == 'placebo' else {}


class FixDef(cst.CSTTransformer):
    """Swap (trap) or rename (placebo) the two parameters in the definition."""

    def __init__(self):
        self.depth = 0

    def leave_FunctionDef(self, orig, node):
        if orig.name.value != func:
            return node
        ps = list(node.params.params)
        off = 1 if ps and ps[0].name.value in ('self', 'cls') else 0
        if mode == 'trap':
            a, b = ps[off], ps[off + 1]
            ps[off], ps[off + 1] = b.with_changes(comma=a.comma), a.with_changes(comma=b.comma)
            return node.with_changes(params=node.params.with_changes(params=ps))
        ps = [p.with_changes(name=cst.Name(new_names.get(p.name.value, p.name.value))) for p in ps]

        class Rename(cst.CSTTransformer):
            def leave_Name(self, o, n):
                return n.with_changes(value=new_names[n.value]) if n.value in new_names else n

        body = node.body.visit(Rename())
        return node.with_changes(params=node.params.with_changes(params=ps), body=body)


class FixCalls(cst.CSTTransformer):
    def __init__(self):
        self.changed = 0

    def leave_Call(self, orig, node):
        f = node.func
        name = f.value if isinstance(f, cst.Name) else f.attr.value if isinstance(f, cst.Attribute) else None
        if name != func:
            return node
        args = list(node.args)
        pos = [a for a in args if a.keyword is None and not a.star]
        kw = [a for a in args if a.keyword is not None]
        if any(a.star for a in args):
            return node
        if mode == 'placebo':
            if not pos:
                return node
            eq = cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(''), whitespace_after=cst.SimpleWhitespace(''))
            names = [p0, p1]
            out = []
            for a in args:
                if a.keyword is None and not a.star and names:
                    a = a.with_changes(keyword=cst.Name(names.pop(0)), equal=eq)
                out.append(a)
            self.changed += 1
            return node.with_changes(args=out)
        if len(pos) >= 2:
            i, j = args.index(pos[0]), args.index(pos[1])
            args[i], args[j] = pos[1].with_changes(comma=pos[0].comma), pos[0].with_changes(comma=pos[1].comma)
        elif len(pos) == 1:  # the one positional value was the old first parameter: name it
            i = args.index(pos[0])
            args[i] = pos[0].with_changes(keyword=cst.Name(p0), equal=cst.AssignEqual(
                whitespace_before=cst.SimpleWhitespace(''), whitespace_after=cst.SimpleWhitespace('')))
            args.append(args.pop(i).with_changes(comma=cst.MaybeSentinel.DEFAULT))
            args = [a.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(' ')))
                    for a in args[:-1]] + [args[-1]]
        else:
            return node
        self.changed += 1
        return node.with_changes(args=args)


LEAD = re.compile(r'^(?:\s|\.\.\.(?=\s))*')


def rewrite_text_calls(text):
    """Rewrite FUNC(...) calls written as text (doctest examples, .rst/.txt docs) the same way
    the code transformer does. Handles calls that span doctest continuation lines."""
    pat = re.compile(rf'(?<![\w]){func}\(')
    out, i, n = [], 0, 0
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        j = start = m.end()
        depth, quote, args, cur = 0, None, [], start
        while j < len(text):
            c = text[j]
            if quote:
                if c == '\\':
                    j += 2
                    continue
                if c == quote:
                    quote = None
            elif c in '\'"':
                quote = c
            elif c in '([{':
                depth += 1
            elif c in ')]}':
                if depth == 0:
                    break
                depth -= 1
            elif c == ',' and depth == 0:
                args.append(text[cur:j])
                cur = j + 1
            j += 1
        else:
            out.append(text[i:])
            break
        args.append(text[cur:j])
        parts = []
        for a in args:
            lead = LEAD.match(a).group(0)
            core = a[len(lead):].rstrip()
            parts.append([lead, core, a[len(lead) + len(core):]])
        is_kw = lambda c: bool(re.match(r'^\w+\s*=(?!=)', c)) or c.startswith('*')
        if len(parts) >= 2 and parts[0][1] and parts[1][1] and not is_kw(parts[0][1]) and not is_kw(parts[1][1]):
            if mode == 'trap':
                parts[0][1], parts[1][1] = parts[1][1], parts[0][1]
            else:
                parts[0][1], parts[1][1] = f'{p0}={parts[0][1]}', f'{p1}={parts[1][1]}'
            n += 1
        out.append(text[i:start] + ','.join(''.join(x) for x in parts) + ')')
        i = j + 1
    return ''.join(out), n


def rewrite_doctests(code):
    """Apply rewrite_text_calls inside string tokens that hold doctest examples."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code, 0
    lines = code.splitlines(keepends=True)
    offs = [0]
    for l in lines:
        offs.append(offs[-1] + len(l))
    edits, total = [], 0
    for t in toks:
        if t.type == tokenize.STRING and '>>>' in t.string and f'{func}(' in t.string:
            new, n = rewrite_text_calls(t.string)
            if n:
                a = offs[t.start[0] - 1] + t.start[1]
                b = offs[t.end[0] - 1] + t.end[1]
                edits.append((a, b, new))
                total += n
    for a, b, new in reversed(edits):
        code = code[:a] + new + code[b:]
    return code, total


n_files = n_calls = 0
for root, dirs, files in os.walk(repo):
    dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', '__pycache__')]
    for fn in files:
        path = os.path.join(root, fn)
        rel = os.path.relpath(path, repo)
        if fn.endswith('.py'):
            text = open(path, encoding='utf-8', errors='surrogateescape').read()
            if func not in text:
                continue
            try:
                m = cst.parse_module(text)
            except Exception:
                continue
            if rel == def_file and mode == 'trap':
                m = m.visit(FixDef())
            fc = FixCalls()
            m = m.visit(fc)
            code, n_doc = rewrite_doctests(m.code)
            if code != text:
                open(path, 'w', encoding='utf-8', errors='surrogateescape').write(code)
                n_files += 1
                n_calls += fc.changed + n_doc
        elif fn.endswith(('.txt', '.rst')):
            text = open(path, encoding='utf-8', errors='surrogateescape').read()
            if f'{func}(' not in text:
                continue
            new, n_doc = rewrite_text_calls(text)
            n_calls += n_doc
            if new != text:
                open(path, 'w', encoding='utf-8', errors='surrogateescape').write(new)
                n_files += 1
print(f'{mode}: {func}({p0}, {p1}) -> files changed {n_files}, calls rewritten {n_calls}')
