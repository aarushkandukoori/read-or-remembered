"""Keyword-rename memory trap (or its placebo), in place.

trap     rename keyword parameter PARAM of FUNC to NEW in the definition (and its uses in the body),
         and rewrite every call that passes PARAM= to NEW=. A call written from memory with PARAM=
         now raises TypeError. Doctest/docs calls are rewritten too.
placebo  the same visible edit, but the definition still accepts the old keyword (a two-line alias),
         so every call written from memory still works.
Usage: python -I make_kwrename_trap.py REPO DEF_FILE FUNC PARAM NEW trap|placebo
"""
import os, re, sys
import libcst as cst

repo, def_file, func, param, new, mode = sys.argv[1:7]


class FixDef(cst.CSTTransformer):
    def __init__(self):
        self.done = False

    def leave_FunctionDef(self, orig, node):
        if orig.name.value != func or self.done:
            return node
        self.done = True

        class Rename(cst.CSTTransformer):
            def leave_Name(self, o, n):
                return n.with_changes(value=new) if n.value == param else n

        params = node.params.visit(Rename())
        body = node.body.visit(Rename())
        if mode == 'placebo':
            star = params.star_kwarg
            kw_name = star.name.value if star else '_legacy_kwargs'
            if not star:
                params = params.with_changes(star_kwarg=cst.Param(name=cst.Name(kw_name)))
            alias = cst.parse_statement(f"{new} = {kw_name}.pop('{param}', {new})")
            stmts = list(body.body)
            i = 1 if stmts and isinstance(stmts[0], cst.SimpleStatementLine) and \
                isinstance(stmts[0].body[0], cst.Expr) and isinstance(stmts[0].body[0].value, (cst.SimpleString, cst.ConcatenatedString)) else 0
            body = body.with_changes(body=stmts[:i] + [alias] + stmts[i:])
        return node.with_changes(params=params, body=body)


class FixCalls(cst.CSTTransformer):
    def __init__(self):
        self.changed = 0

    def leave_Call(self, orig, node):
        f = node.func
        name = f.value if isinstance(f, cst.Name) else f.attr.value if isinstance(f, cst.Attribute) else None
        if name != func:
            return node
        args = [a.with_changes(keyword=cst.Name(new)) if a.keyword is not None and a.keyword.value == param else a
                for a in node.args]
        if args != list(node.args):
            self.changed += 1
        return node.with_changes(args=args)


pat = re.compile(rf'(\b{func}\([^()]*?)\b{param}=')
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
            if rel == def_file:
                m = m.visit(FixDef())
                m = cst.parse_module(m.code.replace(f'`{param}`', f'`{new}`'))
            fc = FixCalls()
            m = m.visit(fc)
            code = m.code
            # doctest examples inside strings
            code2 = re.sub(rf'(>>>[^\n]*?\b{func}\([^()\n]*?)\b{param}=', rf'\1{new}=', code)
            if code2 != text:
                open(path, 'w', encoding='utf-8', errors='surrogateescape').write(code2)
                n_files += 1
                n_calls += fc.changed
        elif fn.endswith(('.txt', '.rst')):
            text = open(path, encoding='utf-8', errors='surrogateescape').read()
            new_text = pat.sub(rf'\1{new}=', text)
            if new_text != text:
                open(path, 'w', encoding='utf-8', errors='surrogateescape').write(new_text)
                n_files += 1
print(f'{mode}: {func}({param}=) -> {new}=; files changed {n_files}, calls rewritten {n_calls}')
