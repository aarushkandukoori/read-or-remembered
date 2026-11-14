"""Shared helpers for the memory-trap transforms.

Walking a repo, reading and writing files without changing their line endings, small libcst helpers
for call arguments, and rewriting code that is written as text: doctest examples (inside docstrings
or in .rst/.txt docs) and strings/comments.
"""
import io
import keyword
import os
import re
import tokenize

import libcst as cst

SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.tox', '.eggs', 'build', 'dist', '.venv', 'venv',
             '.mypy_cache', '.pytest_cache'}
DOC_EXTS = ('.txt', '.rst')

# "a=b" with no spaces, the usual style for keyword arguments
EQ = cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(''), whitespace_after=cst.SimpleWhitespace(''))


def iter_files(repo, exts):
    """Repo-relative paths of all files ending in one of `exts`, in a fixed order."""
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.endswith('.egg-info'))
        for fn in sorted(files):
            if fn.endswith(exts):
                yield os.path.relpath(os.path.join(root, fn), repo)


def read(repo, rel):
    with open(os.path.join(repo, rel), encoding='utf-8', errors='surrogateescape', newline='') as f:
        return f.read()


def write(repo, rel, text):
    path = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8', errors='surrogateescape', newline='') as f:
        f.write(text)


def is_test_path(rel):
    """True for files in test folders, test_*.py files and conftest.py (same rule as the fact miner)."""
    parts = rel.split('/')
    return (any(p in ('tests', 'test', 'testing') for p in parts[:-1])
            or parts[-1].startswith('test_') or parts[-1] == 'conftest.py')


def module_name(rel):
    """django/db/models/query.py -> django.db.models.query; lib/x/__init__.py -> x"""
    p = rel[:-3]
    for prefix in ('src/', 'lib/'):
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    if p.endswith('/__init__'):
        p = p[:-len('/__init__')]
    return p.replace('/', '.')


def absolute_module(rel, level, module):
    """Absolute module name of a relative import `from ..x import y` written in file `rel`."""
    if not level:
        return module or ''
    mod = module_name(rel)
    pkg = mod if rel.endswith('__init__.py') else mod.rpartition('.')[0]
    parts = pkg.split('.') if pkg else []
    if level > 1:
        parts = parts[:max(0, len(parts) - (level - 1))]
    return '.'.join(parts + ([module] if module else []))


def dotted(node):
    """'a.b.c' for a Name/Attribute chain (libcst), else None."""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        head = dotted(node.value)
        return head + '.' + node.attr.value if head else None
    return None


def call_name(call):
    """The name a call is made with: f(...) -> 'f', obj.f(...) -> 'f'."""
    f = call.func
    if isinstance(f, cst.Name):
        return f.value
    if isinstance(f, cst.Attribute):
        return f.attr.value
    return None


def is_positional(arg):
    return arg.keyword is None and arg.star == ''


def has_star(args):
    """True if a call uses *args or **kwargs."""
    return any(a.star for a in args)


def keywordize(arg, name):
    """Turn a positional argument into name=value."""
    return arg.with_changes(keyword=cst.Name(name), equal=EQ)


def append_arg(call, new_arg):
    """Add an argument at the end of a call, keeping the call's comma and line-break style."""
    args = list(call.args)
    if not args:
        return [new_arg]
    last = args[-1]
    if isinstance(last.comma, cst.MaybeSentinel):  # no trailing comma: f(a, b) -> f(a, b, new)
        args[-1] = last.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(' ')))
        return args + [new_arg]
    # trailing comma (usually one argument per line): the new argument takes over the last comma,
    # and the old last argument gets the comma style of the argument before it
    if len(args) >= 2:
        prev_comma = args[-2].comma
    else:
        prev_comma = cst.Comma(whitespace_after=call.whitespace_before_args)
    args[-1] = last.with_changes(comma=prev_comma)
    return args + [new_arg.with_changes(comma=last.comma)]


def bool_value(node):
    """True/False for a literal True/False name, else None."""
    if isinstance(node, cst.Name) and node.value in ('True', 'False'):
        return node.value == 'True'
    return None


def negate(node):
    """Expression for `not node` (flips a True/False literal; removes an outer `not`)."""
    b = bool_value(node)
    if b is not None:
        return cst.Name('False' if b else 'True')
    if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Not):
        inner = node.expression
        return inner.with_changes(lpar=[], rpar=[]) if isinstance(inner, (cst.Name, cst.Attribute, cst.Call)) else inner
    simple = isinstance(node, (cst.Name, cst.Attribute, cst.Call, cst.Subscript))
    inner = node if simple else node.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
    return cst.UnaryOperation(operator=cst.Not(), expression=inner)


def code_of(node):
    """Source text of a libcst node."""
    return cst.Module(body=[]).code_for_node(node)


# ---------------------------------------------------------------- code written as text ----------

PROMPT = re.compile(r'^(?P<ind>[ \t]*)(?P<p>>>>|\.\.\.)(?P<sp> ?)(?P<code>.*)$')


def doctest_examples(text):
    """Find doctest examples in a block of text. Yields (first_line, last_line, prefixes, code_lines,
    output_lines) with 0-based line numbers; prefixes keep each line's indentation and prompt."""
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        m = PROMPT.match(lines[i])
        if not m or m.group('p') != '>>>':
            i += 1
            continue
        start, prefixes, code = i, [], []
        prefixes.append(m.group('ind') + m.group('p') + m.group('sp'))
        code.append(m.group('code'))
        i += 1
        while i < len(lines):
            m2 = PROMPT.match(lines[i])
            if not m2 or m2.group('p') != '...' or m2.group('ind') != m.group('ind'):
                break
            prefixes.append(m2.group('ind') + m2.group('p') + m2.group('sp'))
            code.append(m2.group('code'))
            i += 1
        out_start = i
        while i < len(lines) and lines[i].strip() and not PROMPT.match(lines[i]) \
                and lines[i].startswith(m.group('ind')) and lines[i].strip() not in ('"""', "'''"):
            i += 1  # expected output (stops at a blank line, the next prompt or a docstring's closing quotes)
        yield start, out_start - 1, prefixes, code, list(range(out_start, i))


def rewrite_doctests_in_text(text, edit_source, edit_output=None):
    """Apply edit_source(code) -> new code (or None) to every doctest example in `text`.
    The new code must have the same number of lines (it is put back line by line behind the same
    prompts). edit_output(code, output_lines) -> new output lines (or None) can rewrite the expected
    output of an example (used by the return-convention trap). Returns (new_text, n_examples_changed)."""
    if '>>>' not in text:
        return text, 0
    lines = text.split('\n')
    n = 0
    for start, end, prefixes, code, out_idx in list(doctest_examples(text)):
        src = '\n'.join(code)
        new = edit_source(src)
        changed = False
        if new is not None and new != src:
            new_lines = new.split('\n')
            if len(new_lines) == len(code):
                for k, (pre, nl) in enumerate(zip(prefixes, new_lines)):
                    lines[start + k] = pre + nl
                changed = True
        if edit_output is not None and out_idx:
            outs = [lines[k] for k in out_idx]
            new_out = edit_output(src, outs)
            if new_out is not None and new_out != outs and len(new_out) == len(outs):
                for k, line in zip(out_idx, new_out):
                    lines[k] = line
                changed = True
        n += changed
    return '\n'.join(lines), n


def rewrite_string_tokens(code, edit, kinds=('STRING',)):
    """Apply edit(token_text) -> new text to string (and/or comment) tokens of Python code.
    kinds: 'STRING' (also f-string parts on Python 3.12+), 'COMMENT'. Returns the new code."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code
    want = set()
    for k in kinds:
        want.add(getattr(tokenize, k))
        if k == 'STRING' and hasattr(tokenize, 'FSTRING_MIDDLE'):
            want.add(tokenize.FSTRING_MIDDLE)
    lines = code.splitlines(keepends=True)
    offs = [0]
    for line in lines:
        offs.append(offs[-1] + len(line))
    edits = []
    for t in toks:
        if t.type in want:
            new = edit(t.string)
            if new != t.string:
                a = offs[t.start[0] - 1] + t.start[1]
                b = offs[t.end[0] - 1] + t.end[1]
                if code[a:b] == t.string:  # f-string parts can be reported with odd offsets: skip those
                    edits.append((a, b, new))
    for a, b, new in reversed(edits):
        code = code[:a] + new + code[b:]
    return code


def rewrite_docstring_doctests(code, edit_source, edit_output=None):
    """Run rewrite_doctests_in_text inside every string token of Python code that holds '>>>'."""
    count = [0]

    def edit(s):
        if '>>>' not in s:
            return s
        new, n = rewrite_doctests_in_text(s, edit_source, edit_output)
        count[0] += n
        return new
    return rewrite_string_tokens(code, edit), count[0]


def split_call_args(text, start):
    """Parse the argument list of a call written as text. `start` is the index just after '('.
    Returns (end, parts) where text[end] is the closing ')' and parts are (lead, core, trail) strings
    for each top-level argument, or None if the parentheses never close."""
    depth, quote, j, cur, args = 0, None, start, start, []
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
        return None
    args.append(text[cur:j])
    lead_re = re.compile(r'^(?:\s|\.\.\.(?=\s))*')  # whitespace and doctest '...' continuation prompts
    parts = []
    for a in args:
        lead = lead_re.match(a).group(0)
        core = a[len(lead):].rstrip()
        parts.append([lead, core, a[len(lead) + len(core):]])
    if len(parts) == 1 and not parts[0][1]:
        parts = []  # f()
    return j, parts


def rewrite_text_calls(text, name, edit):
    """Find calls NAME(...) written as text (docs, doctests) and let edit(cores, before) rewrite the list
    of argument strings. `before` is the text in front of the name on its line (so callers can tell a
    signature like '.. function:: name(' or 'def name(' from a call). edit returns a new list of argument
    strings or None. Returns (new_text, n_changed)."""
    pat = re.compile(r'(?<![\w])' + re.escape(name) + r'\(')
    out, i, n = [], 0, 0
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        res = split_call_args(text, m.end())
        if res is None:
            out.append(text[i:])
            break
        end, parts = res
        line_start = text.rfind('\n', 0, m.start()) + 1
        before = text[line_start:m.start()]
        cores = [p[1] for p in parts]
        new = edit(list(cores), before)
        if new is None or new == cores:
            out.append(text[i:end + 1])
        else:
            n += 1
            rebuilt = []
            for k, core in enumerate(new):
                if k < len(parts):
                    rebuilt.append(parts[k][0] + core + parts[k][2])
                else:  # an added argument
                    rebuilt.append(' ' + core)
            body = ','.join(rebuilt)
            if len(new) > len(parts) and parts and not parts[-1][1]:
                # the call had a trailing comma: keep it at the end
                body = ','.join([parts[k][0] + new[k] + parts[k][2] for k in range(len(parts) - 1)]
                                + [' ' + c for c in new[len(parts) - 1:]] + [parts[-1][0] + parts[-1][2]])
            out.append(text[i:m.end()] + body + ')')
        i = end + 1
    return ''.join(out), n


def is_keyword_core(core):
    """'x=1' or '**kw' / '*a' style argument text."""
    return bool(re.match(r'^\w+\s*=(?!=)', core)) or core.startswith('*')


def is_signature_context(before):
    """True if the text before a NAME( looks like a definition or a Sphinx signature directive."""
    b = before.rstrip()
    return bool(re.search(r'(?:\bdef|\bclass|::)\s*(?:[\w.]+\.)?$', b))


def replace_word(text, old, new):
    """Replace whole-word occurrences of identifier `old` (not part of a longer name)."""
    return re.sub(r'(?<![\w])' + re.escape(old) + r'(?![\w])', new, text)


def valid_identifier(name):
    return name.isidentifier() and not keyword.iskeyword(name)


def count_changed_lines(diff_text):
    """Number of added + removed lines in a unified diff (file headers excluded)."""
    return sum(1 for line in diff_text.split('\n')
               if line[:1] in ('+', '-') and not line.startswith(('+++', '---')))


def diff_files(diff_text):
    """Files touched by a unified diff (b/ side, or a/ side for deleted files)."""
    files = []
    for line in diff_text.split('\n'):
        if line.startswith('diff --git '):
            m = re.match(r'diff --git a/(\S+) b/(\S+)', line)
            if m:
                files.append(m.group(2))
    return files
