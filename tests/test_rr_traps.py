"""Unit tests for rr_traps on a tiny toy repo.

For every trap type T1-T8 we check three things:
  1. after the trap, the toy repo's own test suite (tests, doctests in the code, doctests in the docs)
     still passes: the transform rewrote every use consistently;
  2. after the placebo, the same suite still passes;
  3. a "memory" snippet written the OLD way (what a model would write from memory) works in the
     original and in the placebo, but fails in the trap.
Run: .venv/bin/python -m pytest -q tests/test_rr_traps.py
"""
import filecmp
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import rr_traps  # noqa: E402
from rr_traps import common as C  # noqa: E402

TOY = {
    'toypkg/__init__.py': '"""A toy package for the trap tests."""\n',
    'toypkg/util.py': '''
        """Small utilities."""
        import re


        class ParseError(Exception):
            """Bad input."""


        def power(base, exponent):
            """Raise base to exponent.

            >>> power(2, 3)
            8
            """
            return base ** exponent


        def clean(text, strict=False):
            """Strip text. With strict=True, empty text is an error.

            >>> clean('  hi ')
            'hi'
            """
            text = text.strip()
            if strict and not text:
                raise ParseError('empty text')
            return text


        def parse_pair(text):
            """'a=b' -> ('a', 'b'). Raises ValueError when there is no '='."""
            if '=' not in text:
                raise ValueError('no = in %r' % text)
            key, _, value = text.partition('=')
            return key.strip(), value.strip()


        def split_name(full):
            """Split a full name into (first, last).

            >>> split_name('Ada Lovelace')
            ('Ada', 'Lovelace')
            """
            first, _, last = full.partition(' ')
            return first, last


        def match(pattern, text, case_sensitive=True):
            """True if the pattern occurs in text."""
            flags = 0 if case_sensitive else re.IGNORECASE
            return re.search(pattern, text, flags) is not None


        def fetch(url, timeout=10):
            """Pretend to fetch a url."""
            return '%s?t=%s' % (url, timeout)
        ''',
    'toypkg/helpers.py': '''
        """Text helpers."""


        def slugify(text):
            """'Hello World' -> 'hello-world'"""
            return text.lower().replace(' ', '-')


        def shout(text):
            """Upper-case text with an exclamation mark."""
            return text.upper() + '!'
        ''',
    'toypkg/app.py': '''
        """An app that uses the utilities."""
        from toypkg import helpers
        from toypkg.helpers import shout
        from toypkg.util import clean, fetch, match, parse_pair, power, split_name


        def area(side):
            return power(side, 2)


        def settings(lines):
            out = {}
            for line in lines:
                try:
                    k, v = parse_pair(line)
                except ValueError:
                    continue
                out[k] = v
            return out


        def initials(full):
            first, last = split_name(full)
            return first[:1] + last[:1]


        def surname(full):
            return split_name(full)[1]


        def has_word(text, word):
            return match(word, text, case_sensitive=False)


        def title(text):
            return helpers.slugify(clean(text))


        def loud(text):
            return shout(text)


        def get(url):
            return fetch(url, timeout=5)
        ''',
    'tests/test_app.py': '''
        import pytest

        from toypkg import app
        from toypkg.helpers import shout, slugify
        from toypkg.util import clean, fetch, match, parse_pair, power, split_name


        def test_power():
            assert power(2, 10) == 1024
            assert app.area(3) == 9


        def test_clean():
            assert clean(' a ') == 'a'
            assert clean('', strict=False) == ''


        def test_settings():
            assert app.settings(['a=1', 'junk', 'b = 2']) == {'a': '1', 'b': '2'}
            with pytest.raises(ValueError):
                parse_pair('junk')


        def test_names():
            assert split_name('Ada Lovelace') == ('Ada', 'Lovelace')
            assert app.initials('Ada Lovelace') == 'AL'
            assert app.surname('Ada Lovelace') == 'Lovelace'


        def test_match():
            assert match('a', 'cat')
            assert not match('A', 'cat')
            assert match('A', 'cat', case_sensitive=False)
            assert app.has_word('Cat', 'cat')


        def test_text():
            assert slugify('A B') == 'a-b'
            assert app.title(' A B ') == 'a-b'
            assert shout('hi') == 'HI!'
            assert app.loud('x') == 'X!'


        def test_fetch():
            assert fetch('u', timeout=1) == 'u?t=1'
            assert app.get('v') == 'v?t=5'
        ''',
    'docs/usage.rst': '''
        Usage
        =====

        .. function:: power(base, exponent)

           Raise base to a power.

        >>> from toypkg.util import power, clean, fetch, match, split_name
        >>> power(2, 5)
        32
        >>> clean('  hi  ')
        'hi'
        >>> fetch('x', timeout=3)
        'x?t=3'
        >>> match('A', 'a', case_sensitive=False)
        True
        >>> split_name('Grace Hopper')
        ('Grace', 'Hopper')
        >>> from toypkg.helpers import slugify, shout
        >>> slugify('Hello World')
        'hello-world'
        >>> shout('hi')
        'HI!'
        ''',
}

# trap id -> (transform, memory snippet written the old way)
CASES = {
    'T1': (lambda: rr_traps.ArgSwap(('toypkg/util.py', 'power')),
           'from toypkg.util import power\nassert power(2, 3) == 8\n'),
    'T2': (lambda: rr_traps.DefaultChange(('toypkg/util.py', 'clean'), 'strict', 'True'),
           "from toypkg.util import clean\nassert clean('') == ''\n"),
    'T3': (lambda: rr_traps.Rename(('toypkg/helpers.py', 'shout'), 'exclaim'),
           "from toypkg.helpers import shout\nassert shout('a') == 'A!'\n"),
    'T4': (lambda: rr_traps.Move(('toypkg/helpers.py', 'slugify'), 'toypkg/text.py'),
           "from toypkg.helpers import slugify\nassert slugify('A B') == 'a-b'\n"),
    'T5': (lambda: rr_traps.ExceptionSwap(('toypkg/util.py', 'parse_pair'), 'ValueError', 'PairError', 'LookupError'),
           "from toypkg.util import parse_pair\ntry:\n    parse_pair('x')\nexcept ValueError:\n    pass\n"),
    'T6': (lambda: rr_traps.ReturnSwap(('toypkg/util.py', 'split_name')),
           "from toypkg.util import split_name\nfirst, last = split_name('Ada Lovelace')\nassert first == 'Ada'\n"),
    'T7': (lambda: rr_traps.KeywordRename(('toypkg/util.py', 'fetch'), 'timeout', 'timeout_s'),
           "from toypkg.util import fetch\nassert fetch('u', timeout=1) == 'u?t=1'\n"),
    'T8': (lambda: rr_traps.FlagInvert(('toypkg/util.py', 'match'), 'case_sensitive', 'ignore_case'),
           "from toypkg.util import match\nassert match('A', 'a', case_sensitive=False)\n"),
}


def make_toy(path):
    for rel, text in TOY.items():
        C.write(str(path), rel, textwrap.dedent(text).lstrip('\n'))
    return str(path)


def suite_passes(repo):
    """Run the toy repo's tests + doctests (code and docs). Returns (ok, output)."""
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--doctest-modules',
                        '--doctest-glob=*.rst', 'tests', 'toypkg', 'docs'],
                       cwd=repo, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=repo))
    return r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]


def memory_works(repo, snippet):
    r = subprocess.run([sys.executable, '-c', snippet], cwd=repo, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=repo))
    return r.returncode == 0


def changed_files(a, b):
    """Files that differ between two directory trees (or exist in only one)."""
    out = set()
    for rel in set(C.iter_files(a, ('.py', '.rst'))) | set(C.iter_files(b, ('.py', '.rst'))):
        pa, pb = os.path.join(a, rel), os.path.join(b, rel)
        if not (os.path.exists(pa) and os.path.exists(pb)) or not filecmp.cmp(pa, pb, shallow=False):
            out.add(rel)
    return out


def test_toy_repo_is_sane(tmp_path):
    repo = make_toy(tmp_path / 'orig')
    ok, out = suite_passes(repo)
    assert ok, out
    for tid, (_, snippet) in CASES.items():
        assert memory_works(repo, snippet), tid


@pytest.mark.parametrize('tid', sorted(CASES))
def test_trap_and_placebo(tmp_path, tid):
    make_transform, snippet = CASES[tid]
    orig = make_toy(tmp_path / 'orig')
    results = {}
    for mode in ('trap', 'placebo'):
        repo = make_toy(tmp_path / mode)
        report = make_transform().apply(repo, mode)
        ok, out = suite_passes(repo)
        assert ok, f'{tid} {mode}: toy test suite broke\n{out}\nwarnings: {report.warnings}'
        results[mode] = dict(memory=memory_works(repo, snippet), files=changed_files(orig, repo),
                             sites=len(report.sites))
    assert not results['trap']['memory'], f'{tid}: code written from memory still works in the trap'
    assert results['placebo']['memory'], f'{tid}: code written from memory broke in the placebo'
    # the placebo touches the same files as the trap
    assert results['placebo']['files'] == results['trap']['files'], results
    assert results['trap']['files'], 'trap changed nothing'


def test_refuses_duplicate_names(tmp_path):
    repo = make_toy(tmp_path / 'dup')
    C.write(repo, 'toypkg/other.py', 'def power(a, b):\n    return a * b\n')
    with pytest.raises(rr_traps.TrapError):
        rr_traps.ArgSwap(('toypkg/util.py', 'power')).apply(repo, 'trap')


def test_rename_placebo_keeps_imported_old_name(tmp_path):
    """Code written from memory can use the old name inside a module that already imports it (no import of
    its own). The T3 placebo must keep that working and the trap must not (found in the Docker validation of
    sympy-14248: the upstream fix calls _keep_coeff in printing/str.py, which imports it at the top)."""
    snippet = "import toypkg.app as a\nassert a.shout('x') == 'X!'\n"  # app.py has `from toypkg.helpers import shout`
    for mode, should_work in (('trap', False), ('placebo', True)):
        repo = make_toy(tmp_path / mode)
        rr_traps.Rename(('toypkg/helpers.py', 'shout'), 'exclaim').apply(repo, mode)
        ok, out = suite_passes(repo)
        assert ok, f'{mode}: toy test suite broke\n{out}'
        assert memory_works(repo, snippet) == should_work, mode


def test_rename_refuses_variable_clash(tmp_path):
    repo = make_toy(tmp_path / 'clash')
    C.write(repo, 'toypkg/other.py', 'def f():\n    shout = 1\n    return shout\n')
    with pytest.raises(rr_traps.TrapError):
        rr_traps.Rename(('toypkg/helpers.py', 'shout'), 'exclaim').apply(repo, 'trap')


def test_text_call_rewriter():
    text = ">>> f(a, g(b, c),\n...   d=1)\n"
    new, n = C.rewrite_text_calls(text, 'f', lambda cores, before: [cores[1], cores[0]] + cores[2:])
    assert n == 1 and new.startswith('>>> f(g(b, c), a,')


def test_make_from_spec_roundtrip(tmp_path):
    repo = make_toy(tmp_path / 'spec')
    t = rr_traps.DefaultChange(('toypkg/util.py', 'clean'), 'strict', 'True')
    t.apply(repo, 'trap')
    t2 = rr_traps.make(t.spec())
    assert isinstance(t2, rr_traps.DefaultChange)
    assert (t2.param, t2.new_default, t2.defs) == ('strict', 'True', t.defs)


def test_move_reexport_placebo(tmp_path):
    """The design-doc placebo for T4: the object stays in its module, a new module re-exports it."""
    orig = make_toy(tmp_path / 'orig')
    repo = make_toy(tmp_path / 'reexport')
    rr_traps.Move(('toypkg/helpers.py', 'slugify'), 'toypkg/text.py', placebo_style='reexport').apply(repo, 'placebo')
    ok, out = suite_passes(repo)
    assert ok, out
    assert memory_works(repo, CASES['T4'][1])
    assert 'toypkg/helpers.py' not in changed_files(orig, repo)  # the definition did not move
    assert C.read(repo, 'toypkg/text.py').strip().endswith('import slugify  # noqa: F401')
