"""RQ1 on You Wang et al.'s SWE-bench Pro claims: had the agent READ the code behind a claim before making it?

Claims: every refuted claim (8,540) plus a random control sample of 8,540 supported claims (seed 0),
from data/faultyclaims/claims_tidy.jsonl (made by load_faultyclaims.py). Claim id: cid = 'c%05d' % i,
where i is the 0-based row number in claims_tidy.jsonl (the annotation batches use the same ids).

Features per claim (one JSON row each in outputs/rq1/claims_features.jsonl):

  evidence_seen_before = all / some / none / no_repo_evidence
      You Wang et al.'s assessment agent backed every verdict with citations. For the citations of
      kind "repository" (a file + line range at the task's base commit) we read those lines from git and
      ask: did each cited line appear in a tool output (observation) BEFORE the claim's message?
      - Lines are compared after collapsing whitespace. Display prefixes that tools add are ignored:
        line numbers (cat -n, nl, grep -n), "path:12:" (grep -rn) and "+"/"-"/">" (diffs, pytest).
      - Trivial lines are skipped (blank, only punctuation, or only keywords/literals/numbers, like
        "}", "else:", "return nil, err"): they appear everywhere and prove nothing.
      - Main version ("anchored"): the observation must be about the same file, i.e. the file's name
        appears in that observation or in the command that produced it. evidence_seen_before_any is the
        looser version (the line appeared in any earlier observation).
      no_repo_evidence: no repository citation, or every cited line is trivial (see no_repo_reason).
  file_seen_before = all / some / none / no_repo_evidence
      Same question per cited file: was ANY non-trivial line of that file shown before the claim?
  identifiers: code names in the claim (backticked, or code-shaped like snake_case / camelCase /
      dotted.names / paths), checked with git at the task's base commit:
      base      exists at the base commit (symbols: as a whole word in the repo's code files, read
                from git; paths: git ls-tree)
      external  missing at base, but its first part is a standard-library/builtin name or a name the repo
                imports from a third-party package: git cannot check it
      stale / future / both   missing at base; found only in OLDER / only in NEWER / in both kinds of
                history snapshots (default-branch commits every 2 years + the newest commit; see
                rq1_pro_repos.py snapshots). This stands in for git log -S over all refs, which needs
                every historical file version and does not work on a blobless clone within 3 GB.
      never     found in none of these
      Flags per identifier: in_task (named in the task text), seen_before (in an earlier tool output),
      written_before (in an earlier command of the agent), in_own_command (in the claim message's own
      tool call: something the agent is about to write).

Usage (from the project root; it never downloads: run rq1_pro_repos.py first):
    .venv/bin/python -I scripts/rq1_claim_provenance.py features   # -> outputs/rq1/claims_features.jsonl
    .venv/bin/python -I scripts/rq1_claim_provenance.py summary    # -> outputs/rq1/claims_summary.json
    .venv/bin/python -I scripts/rq1_claim_provenance.py handcheck 15 0 > <scratch file>   # claims to check by hand

Inputs are downloaded data (You Wang et al.'s artifact, CC-BY-4.0): read as JSON only, never executed.
"""
import builtins
import collections
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile

# The shared repo list, file filter and git helper live in rq1_pro_repos.py next to this file.
# (-I mode does not put the script's folder on the import path, so we add it ourselves.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq1_pro_repos import (REPOS_DIR, SUBMODULES, MANIFEST_PATHSPEC, SNAPSHOTS_JSON,  # noqa: E402
                           NO_FETCH, git, repo_of, searchable_files)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT = os.path.join(PROJECT, "data", "faultyclaims", "reasoning_replication")
TIDY = os.path.join(PROJECT, "data", "faultyclaims", "claims_tidy.jsonl")
OUT_DIR = os.path.join(PROJECT, "outputs", "rq1")
FEATURES = os.path.join(OUT_DIR, "claims_features.jsonl")
SUMMARY = os.path.join(OUT_DIR, "claims_summary.json")
SEED = 0

# ---------------------------------------------------------------------------------------------
# 1. Text helpers: normalizing lines, trivial lines, tool display prefixes
# ---------------------------------------------------------------------------------------------

# Language keywords and constant literals (Python, JavaScript/TypeScript, Go). A line made only of
# these (plus punctuation and numbers) is "trivial".
KEYWORDS = set("""
False None True and as assert async await break class continue def del elif else except finally for
from global if import in is lambda nonlocal not or pass raise return try while with yield self cls
case catch const debugger default delete do export extends function instanceof let new null of static
super switch this throw true false typeof undefined var void enum implements interface package private
protected public readonly type declare namespace module abstract keyof any never unknown boolean number
string object symbol chan defer fallthrough func go goto map range select struct nil iota err error ok
bool int byte rune end fi done then elif esac
""".split())


def norm(line):
    """Collapse every run of spaces/tabs to one space and trim both ends."""
    return " ".join(line.split())


def is_trivial(n):
    """True for a normalized line that is too generic to show anything: shorter than 4 characters, or
    nothing but punctuation, numbers, keywords and literals (e.g. '}', 'else:', 'return nil, err')."""
    if len(n) < 4:
        return True
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", n)
    return all(t in KEYWORDS or t.isdigit() for t in tokens)


# Prefixes that tools put in front of a file's lines. We also index each output line with these removed.
DISPLAY_PREFIXES = [
    re.compile(r"^\d+\s?[:\-|]?\s?"),             # "12 code", "12:code", "12-code" (cat -n, nl -ba, grep -n)
    re.compile(r"^\S+?[:\-]\d+[:\-]\s?"),           # "src/a.js:12:code", "src/a.js-12-code" (grep -rn)
    re.compile(r"^[^\s:]+\.[A-Za-z0-9]{1,6}:\s?"),  # "src/a.js:code" (grep -r without -n)
    re.compile(r"^[+\-><]\s?"),                     # "+code", "-code" (diffs), "> code" (pytest)
]


def display_variants(n):
    """The normalized output line itself plus the line with one known display prefix removed."""
    out = {n}
    for rx in DISPLAY_PREFIXES:
        m = rx.match(n)
        if m and m.end() < len(n):
            out.add(n[m.end():].strip())
    return out


def word_regex(word):
    """Regex that finds `word` as a whole name (not inside a longer name)."""
    return re.compile(r"(?<![A-Za-z0-9_$])" + re.escape(word) + r"(?![A-Za-z0-9_$])")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------
# 2. Trajectories: what the agent had seen before each message
# ---------------------------------------------------------------------------------------------

def message_text(m):
    c = m.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # content given as parts
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return json.dumps(c)


def tool_calls(m):
    """[(call id, shell command)] of an assistant message."""
    out = []
    for call in m.get("tool_calls") or []:
        args = (call.get("function") or {}).get("arguments") or ""
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
            cmd = parsed.get("command", "") if isinstance(parsed, dict) else str(parsed)
        except (ValueError, AttributeError):
            cmd = str(args)
        out.append((call.get("id"), cmd if isinstance(cmd, str) else json.dumps(cmd)))
    return out


class Trajectory:
    """One agent run. Observations = the 'tool' messages (command outputs)."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.messages = json.load(f)["messages"]
        self.task = next(message_text(m) for m in self.messages if m["role"] == "user")
        self.obs = {}            # message index -> output text
        self.obs_cmd = {}        # message index -> the command that produced this output
        self.cmds = {}           # assistant message index -> its commands, joined
        cmd_by_id = {}
        for i, m in enumerate(self.messages):
            if m["role"] == "assistant":
                calls = tool_calls(m)
                self.cmds[i] = "\n".join(c for _, c in calls)
                cmd_by_id.update(calls)
            elif m["role"] == "tool":
                self.obs[i] = message_text(m)
                self.obs_cmd[i] = cmd_by_id.get(m.get("tool_call_id"), "")
        # index: every output line (and its prefix-free variants) -> the observation indices that show it
        self.index = collections.defaultdict(list)
        for i in sorted(self.obs):
            for raw in self.obs[i].splitlines():
                n = norm(raw)
                if not n:
                    continue
                for v in display_variants(n):
                    seen = self.index[v]
                    if not seen or seen[-1] != i:
                        seen.append(i)
        self.task_lines = set()
        for raw in self.task.splitlines():
            n = norm(raw)
            if n:
                self.task_lines |= display_variants(n)
        self._first_anchored = {}
        self._first_word = {}
        self._file_first = {}

    def anchored(self, i, file_name):
        """Is observation i about this file (its name in the output or in the command)?"""
        return file_name in self.obs[i] or file_name in self.obs_cmd[i]

    def first_seen(self, line, file_name):
        """Earliest observation index that shows this normalized line, anchored to the file (or None)."""
        key = (line, file_name)
        if key not in self._first_anchored:
            self._first_anchored[key] = next(
                (i for i in self.index.get(line, ()) if self.anchored(i, file_name)), None)
        return self._first_anchored[key]

    def file_first_seen(self, blob_oid, file_name, raw_lines):
        """Earliest observation index that shows ANY non-trivial line of this file version (or None)."""
        key = (blob_oid, file_name)
        if key not in self._file_first:
            firsts = [self.first_seen(n, file_name) for n in map(norm, raw_lines) if not is_trivial(n)]
            firsts = [i for i in firsts if i is not None]
            self._file_first[key] = min(firsts) if firsts else None
        return self._file_first[key]

    def first_seen_any(self, line):
        hits = self.index.get(line)
        return hits[0] if hits else None

    def first_word(self, word, where):
        """Earliest message index whose output ('obs') or command ('cmd') contains word as a name."""
        key = (word, where)
        if key not in self._first_word:
            texts = self.obs if where == "obs" else self.cmds
            rx = word_regex(word)
            self._first_word[key] = next((i for i in sorted(texts) if rx.search(texts[i])), None)
        return self._first_word[key]


# ---------------------------------------------------------------------------------------------
# 3. Identifiers named in a claim
# ---------------------------------------------------------------------------------------------

NAME = r"[A-Za-z_$][A-Za-z0-9_$]*"
DOTTED = rf"{NAME}(?:(?:\.|::|#){NAME})+"
BACKTICK = re.compile(r"`([^`\n]{1,200})`")
QUOTED = re.compile(r"'[^'\n]*'|\"[^\"\n]*\"")
PATH_RX = re.compile(r"(?<![\w.\-/])(?:\.{1,2}/|/)?(?:[\w.\-]+/)+[\w.\-]+")
FILE_EXT = re.compile(r"\.(?:py|pyi|js|jsx|mjs|cjs|ts|tsx|go|proto|rs|java|c|h|sh|ya?ml|toml|cfg|ini|json|"
                      r"html?|tpl|tmpl|jinja2?|j2|vue|sql|graphql|md|rst|txt|css|less|scss|lock|mod|sum)$")
CODE_SHAPES = [re.compile(p) for p in (
    rf"(?<![\w.$]){DOTTED}",                   # dotted.name, Class::method, Class#method
    r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b",       # snake_case
    r"(?<![\w.])_{1,2}[A-Za-z][A-Za-z0-9_]*\b",  # _private, __dunder__
    r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b",       # CONSTANT_CASE
    r"\b[a-z]+[A-Z][A-Za-z0-9]*\b",             # camelCase
    r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b",     # PascalCase with an inner capital
    r"\b[A-Z]{2,}[a-z][A-Za-z0-9]*\b",          # HTTPServer, IOError
    r"\b[A-Za-z_][A-Za-z0-9_]*(?=\()",          # name( : written like a call
)]
SKIP_LAST = {"com", "org", "net", "io", "www", "http", "https"}  # parts of web addresses


def extract_identifiers(text):
    """Code-like names in a claim, in order, without duplicates."""
    found = []
    for span in BACKTICK.findall(text):            # 1) inside `backticks`: everything code-like
        span = span.strip()
        if QUOTED.fullmatch(span):                # a string literal, not a name
            continue
        if " " not in span and (PATH_RX.fullmatch(span) or FILE_EXT.search(span)):
            found.append(span)
            continue
        found += re.findall(rf"{NAME}(?:(?:\.|::|#){NAME})*", QUOTED.sub(" ", span))
    rest = BACKTICK.sub(" ", text)                 # 2) outside backticks: only code-shaped words
    found += [m.group(0) for m in PATH_RX.finditer(rest) if "/" in m.group(0).strip("/")]
    rest = PATH_RX.sub(" ", rest)
    for rx in CODE_SHAPES:
        found += [m.group(0) for m in rx.finditer(rest)]
    out = []
    for tok in found:
        tok = tok.strip().strip(".,;:()[]{}'\"")
        if tok and tok not in out:
            out.append(tok)
    return out


def identifier_key(tok):
    """What to look for in git: ('path', 'src/a.js') or ('symbol', 'getQueuedPosts'), plus the first
    part of a dotted name (for the external check). None = not worth checking."""
    t = re.sub(r":\d+(?::\d+)?$", "", tok)        # drop ":12" line numbers after a path
    if t.startswith("/app/"):                       # the agents' container keeps the repo in /app
        t = t[len("/app/"):]
    t = re.sub(r"^(?:\.{1,2}/)+", "", t)
    if "*" in t or " " in t:
        return None
    if "/" in t or FILE_EXT.search(t):
        # only file names/paths with an extension; URL routes like /api/v3/chats and folders are skipped
        t = t.strip("/")
        if FILE_EXT.search(t) and re.fullmatch(r"[\w.\-/]+", t) and len(t) >= 3:
            return ("path", t, None)
        return None
    parts = re.split(r"\.|::|#", t)
    last, root = parts[-1].lstrip("$"), parts[0].lstrip("$")
    if (len(last) < 3 or last in KEYWORDS or last in SKIP_LAST or last.isdigit()
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", last)):
        return None
    return ("symbol", last, root if len(parts) > 1 else None)


# ---------------------------------------------------------------------------------------------
# 4. Git lookups (no downloads: NO_FETCH makes a missing file an error instead of a download)
# ---------------------------------------------------------------------------------------------

_blob_cache = {}


def blob_text(repo_dir, oid):
    """File content as text (UTF-8, bad bytes replaced), cached."""
    if oid not in _blob_cache:
        raw = subprocess.run(["git", "-C", repo_dir, "cat-file", "blob", oid], capture_output=True,
                             env=NO_FETCH, check=True).stdout
        _blob_cache[oid] = raw.decode("utf-8", errors="replace")
    return _blob_cache[oid]


def read_blobs(repo_dir, oids):
    """Yield (blob id, content bytes) for many file versions with one 'git cat-file --batch' call."""
    with tempfile.TemporaryFile("w+b") as ids:
        ids.write(("\n".join(oids) + "\n").encode())
        ids.seek(0)
        proc = subprocess.Popen(["git", "-C", repo_dir, "cat-file", "--batch"], stdin=ids,
                                stdout=subprocess.PIPE, env=NO_FETCH)
        for _ in oids:
            header = proc.stdout.readline().decode().split()   # "<id> blob <size>" or "<id> missing"
            if len(header) != 3:
                raise RuntimeError(f"file version not downloaded: {header} in {repo_dir}")
            data = proc.stdout.read(int(header[2]))
            proc.stdout.read(1)                                 # newline after each object
            yield header[0], data
        proc.wait()


def tree_paths(repo_dir, commit):
    """All file paths and folder paths at a commit (needs no file contents)."""
    out = git(repo_dir, "ls-tree", "-r", "--name-only", "-z", commit)
    files = [p for p in out.split("\0") if p]
    folders = set()
    for p in files:
        parts = p.split("/")
        for k in range(1, len(parts)):
            folders.add("/".join(parts[:k]))
    return files, folders


def path_exists(token, files, folders, file_set):
    """A path token matches a file/folder path exactly or as its tail ('posts/queue.js', 'queue.js')."""
    if token in file_set or token in folders:
        return True
    tail = "/" + token
    return any(p.endswith(tail) for p in files) or any(d.endswith(tail) for d in folders)


# Names that belong to the language, not to the repo.
PY_BUILTIN = set(sys.stdlib_module_names) | set(dir(builtins))
JS_BUILTIN = set("""Object Array String Number Boolean Symbol BigInt Math JSON Date RegExp Error TypeError
RangeError SyntaxError ReferenceError Promise Map Set WeakMap WeakSet Proxy Reflect Intl console window
document navigator location history localStorage sessionStorage fetch URL URLSearchParams Blob File
FormData Headers Request Response setTimeout clearTimeout setInterval clearInterval requestAnimationFrame
process Buffer require module exports global globalThis HTMLElement Element Node Event CustomEvent
KeyboardEvent MouseEvent crypto TextEncoder TextDecoder Uint8Array ArrayBuffer DataView atob btoa
encodeURIComponent decodeURIComponent parseInt parseFloat isNaN isFinite jest expect describe it test
beforeEach afterEach beforeAll afterAll React ReactDOM""".split())
GO_STDLIB = set("""fmt strings strconv errors context time os io ioutil bufio bytes net http url sync atomic
sort regexp path filepath json xml base64 hex sha256 sha1 sha512 md5 crypto tls x509 rand log slog
testing reflect runtime unicode utf8 math big exec signal syscall template html heap list ring flag gzip
zip tar sql driver embed fs mime multipart httptest httputil pprof debug trace slices maps cmp unsafe
cookiejar textproto netip pem rsa ecdsa ed25519 hmac subtle elliptic aes cipher""".split())


def go_package_name(path):
    """Default name of an imported Go package: 'gopkg.in/yaml.v3' -> 'yaml', 'x/otel/semconv/v1.4.0' -> 'semconv'."""
    parts = path.split("/")
    last = parts[-1]
    if re.fullmatch(r"v\d+(\.\d+)*", last) and len(parts) > 1:
        last = parts[-2]
    last = re.sub(r"\.v\d+$", "", last)
    last = re.sub(r"^go-", "", last)
    last = re.sub(r"[-.]go$", "", last)
    return last.replace("-", "_").replace(".", "_")


SOURCE_LANG = {".py": "py", ".pyi": "py", ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
               ".ts": "js", ".tsx": "js", ".go": "go"}
PY_FROM = re.compile(r"^[ \t]*from[ \t]+([A-Za-z_][\w.]*)[ \t]+import[ \t]+\(?([\w ,\t]+)", re.M)
PY_IMPORT = re.compile(r"^[ \t]*import[ \t]+([\w., \t]+)$", re.M)
JS_IMPORT = re.compile(r"""^[ \t]*import[ \t]+(?:type[ \t]+)?([^;'"`]*?)[ \t]*from[ \t]*['"]([^'"]+)['"]""", re.M)
JS_REQUIRE = re.compile(r"""(?:const|let|var)[ \t]+([^=;]*?)[ \t]*=[ \t]*require\([ \t]*['"]([^'"]+)['"][ \t]*\)""")
GO_IMPORT = re.compile(r'^[ \t]*(?:import[ \t]+)?([A-Za-z_]\w*|\.|_)?[ \t]*"([A-Za-z0-9_./\-]+)"[ \t]*$', re.M)


def parse_imports(text, lang):
    """Import statements of one source file as (language, imported module, names it binds)."""
    out = []
    if lang == "py":
        for module, items in PY_FROM.findall(text):
            out.append(("py", module, [i.split()[-1] for i in items.split(",") if i.split()]))
        for items in PY_IMPORT.findall(text):
            for item in items.split(","):
                bits = item.split()
                if bits:
                    out.append(("py", bits[0], [bits[-1] if len(bits) == 3 and bits[1] == "as"
                                                else bits[0].split(".")[0]]))
    elif lang == "js":
        for bound, spec in JS_IMPORT.findall(text) + JS_REQUIRE.findall(text):
            bound = re.sub(r"\b[\w$]+\s+as\s+", "", bound.replace("* as", " "))
            out.append(("js", spec, re.findall(r"[A-Za-z_$][\w$]*", bound)))
    elif lang == "go":
        for alias, path in GO_IMPORT.findall(text):
            out.append(("go", path, [alias if alias and alias not in (".", "_") else go_package_name(path)]))
    return out


class BlobScanner:
    """Reads every needed file version of one repo once, and remembers which of the query words it
    contains as whole words (like 'git grep -w') and its import statements. Binary files are skipped
    (a NUL byte in the first 8000 bytes, the same test git grep -I uses)."""

    def __init__(self, repo_dir, words):
        self.repo_dir, self.words = repo_dir, set(words)
        self.found, self.imports = {}, {}

    def scan(self, files):
        """files = {path: blob id}; reads the versions not seen yet."""
        todo = {}
        for path, oid in files.items():
            if oid not in self.found:
                todo.setdefault(oid, path)
        for oid, data in read_blobs(self.repo_dir, list(todo)):
            if b"\0" in data[:8000]:
                self.found[oid], self.imports[oid] = frozenset(), ()
                continue
            text = data.decode("utf-8", errors="replace")
            self.found[oid] = frozenset(set(re.findall(r"[A-Za-z0-9_]+", text)) & self.words)
            lang = SOURCE_LANG.get(os.path.splitext(todo[oid])[1])
            self.imports[oid] = tuple(parse_imports(text, lang)) if lang else ()

    def words_at(self, files):
        """Query words that occur in at least one of these files."""
        out = set()
        for oid in set(files.values()):
            out |= self.found[oid]
        return out


def external_names(repo_dir, commit, files, all_paths, scanner):
    """Names that come from outside the repo at this commit: builtins/standard library, plus every name
    the repo's code imports from a package that is not the repo's own (Python, JS/TS, Go)."""
    names = PY_BUILTIN | JS_BUILTIN | GO_STDLIB
    own_py = set()  # a Python module is the repo's own if its first part is a top-level folder/module
    for p in all_paths:
        if p.endswith(".py") and "vendor/" not in p:
            parts = p.split("/")
            own_py.add(parts[0][:-3] if len(parts) == 1 else parts[0])
            if parts[0] in ("lib", "src") and len(parts) > 1:
                own_py.add(parts[1][:-3] if len(parts) == 2 else parts[1])
    own_js, own_go = set(), set()  # JS packages / Go modules declared by the repo's own manifests
    for path, oid in searchable_files(repo_dir, commit, MANIFEST_PATHSPEC).items():
        if path.endswith("package.json"):
            try:
                own_js.add(json.loads(blob_text(repo_dir, oid)).get("name") or "")
            except (ValueError, AttributeError):
                pass
        elif path.endswith("go.mod"):
            m = re.search(r"^module\s+(\S+)", blob_text(repo_dir, oid), re.M)
            if m:
                own_go.add(m.group(1))
    for oid in set(files.values()):
        for lang, module, bound in scanner.imports[oid]:
            if lang == "py":
                if module.split(".")[0] in own_py:
                    continue
                names.add(module.split(".")[0])
            elif lang == "js":
                if module.startswith((".", "/")):
                    continue
                pkg = "/".join(module.split("/")[:2]) if module.startswith("@") else module.split("/")[0]
                if pkg in own_js:
                    continue
                names.add(pkg.split("/")[-1])
            elif any(module == mod or module.startswith(mod + "/") for mod in own_go):
                continue
            names.update(bound)
    return names


# ---------------------------------------------------------------------------------------------
# 5. Building the features
# ---------------------------------------------------------------------------------------------

def load_claims():
    with open(TIDY, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def select_claims(rows):
    """All refuted claims + a random sample (seed 0) of the same number of supported claims."""
    refuted = [i for i, r in enumerate(rows) if r["verdict"] == "refuted"]
    supported = [i for i, r in enumerate(rows) if r["verdict"] == "supported"]
    control = random.Random(SEED).sample(supported, len(refuted))
    return {i: "refuted" for i in refuted} | {i: "supported_control" for i in control}


def load_evidence_tables():
    """{(model label, instance id): (base commit, evidence table)} from You Wang et al.'s result.json files."""
    out = {}
    root = os.path.join(ARTIFACT, "rq2", "results")
    for label in os.listdir(root):
        for instance in os.listdir(os.path.join(root, label)):
            with open(os.path.join(root, label, instance, "result.json"), encoding="utf-8") as f:
                r = json.load(f)
            out[(label, instance)] = (r["base_commit"], r["evidence"])
    return out


def classify(n_seen, n_total):
    if n_total == 0:
        return "no_repo_evidence"
    return "all" if n_seen == n_total else ("some" if n_seen else "none")


def evidence_features(row, traj, base_commit, ev_table, checks):
    """Seen-before features of the claim's repository citations."""
    here = row["message_index"]
    repo = repo_of(row["instance_id"])
    citations, cited, files = [], {}, {}
    for ev in row["evidence"]:
        if ev["kind"] != "repository":
            continue
        full = ev_table[ev["id"]]
        prov = full["repository_provenance"]
        name = repo if prov["repository_path"] == "." else SUBMODULES[prov["repository_path"]][0]
        repo_dir = os.path.join(REPOS_DIR, name)
        text = blob_text(repo_dir, prov["blob_oid"])
        raw_lines = text.splitlines(keepends=True)   # same line splitting as the artifact's line_slice()
        excerpt = "".join(raw_lines[ev["start_line"] - 1:ev["end_line"]])
        sha_ok = sha256_text(text) == full["source_sha256"] and sha256_text(excerpt) == full["excerpt_sha256"]
        checks["citations"] += 1
        checks["sha_ok"] += sha_ok
        file_name = ev["source"].rsplit("/", 1)[-1]
        files[(ev["source"], prov["blob_oid"])] = (file_name, raw_lines)
        n_nontrivial = n_seen = 0
        for lineno in range(ev["start_line"], ev["end_line"] + 1):
            line = norm(raw_lines[lineno - 1]) if lineno <= len(raw_lines) else ""
            if is_trivial(line):
                continue
            n_nontrivial += 1
            first = traj.first_seen(line, file_name)
            n_seen += first is not None and first < here
            cited[(ev["source"], lineno)] = (line, file_name)
        citations.append({"file": ev["source"], "start": ev["start_line"], "end": ev["end_line"],
                          "n_nontrivial": n_nontrivial, "n_seen": n_seen, "sha_ok": sha_ok})

    # line level, over the union of cited lines (a line cited twice counts once)
    seen = seen_any = in_task = 0
    first_seen_msgs = []
    for line, file_name in cited.values():
        first = traj.first_seen(line, file_name)
        if first is not None and first < here:
            seen += 1
            first_seen_msgs.append(first)
        first_any = traj.first_seen_any(line)
        seen_any += first_any is not None and first_any < here
        in_task += line in traj.task_lines
    # file level: was any non-trivial line of each cited file shown before?
    files_seen = 0
    for (source, blob_oid), (file_name, raw_lines) in files.items():
        first = traj.file_first_seen(blob_oid, file_name, raw_lines)
        files_seen += first is not None and first < here
    n_total = len(cited)
    if not citations:
        reason = "no_repository_citation"
    elif n_total == 0:
        reason = "only_trivial_lines"
    else:
        reason = None
    return {
        "n_repo_citations": len(citations),
        "n_cited_lines": n_total,               # non-trivial, unique (file, line) pairs
        "n_lines_seen": seen,
        "n_lines_seen_any": seen_any,
        "n_lines_in_task_text": in_task,
        "evidence_seen_before": classify(seen, n_total),
        "evidence_seen_before_any": classify(seen_any, n_total),
        "no_repo_reason": reason,
        "n_cited_files": len(files),
        "n_cited_files_seen": files_seen,
        "file_seen_before": classify(files_seen, len(files)) if n_total else "no_repo_evidence",
        "first_seen_msg": min(first_seen_msgs) if first_seen_msgs else None,
        "citations": citations,
    }


def identifier_candidates(row, traj):
    """Identifiers of the claim with their trajectory flags (git status is filled in later)."""
    here = row["message_index"]
    own_cmd = traj.cmds.get(here, "")
    out = []
    for tok in extract_identifiers(row["claim"] + "\n" + row["quote"]):
        key = identifier_key(tok)
        if not key:
            continue
        kind, word, root = key
        if any(o["key"] == word and o["kind"] == kind for o in out):
            continue
        if kind == "path":
            in_task = word in traj.task
            seen = next((i for i in sorted(traj.obs) if i < here and word in traj.obs[i]), None) is not None
            written = next((i for i in sorted(traj.cmds) if i < here and word in traj.cmds[i]), None) is not None
            own = word in own_cmd
        else:
            in_task = word_regex(word).search(traj.task) is not None
            first_obs = traj.first_word(word, "obs")
            first_cmd = traj.first_word(word, "cmd")
            seen = first_obs is not None and first_obs < here
            written = first_cmd is not None and first_cmd < here
            own = word_regex(word).search(own_cmd) is not None
        out.append({"name": tok, "kind": kind, "key": word, "root": root, "in_task": in_task,
                    "seen_before": seen, "written_before": written, "in_own_command": own})
    return out


def resolve_identifiers(feature_rows):
    """Fill in the git status of every identifier: base / external / stale / future / both / never."""
    with open(SNAPSHOTS_JSON) as f:
        snapshots = json.load(f)
    for repo in sorted({fr["repo"] for fr in feature_rows}):
        repo_dir = os.path.join(REPOS_DIR, repo)
        rows = [fr for fr in feature_rows if fr["repo"] == repo]
        scanner = BlobScanner(repo_dir, {i["key"] for fr in rows for i in fr["identifiers"] if i["kind"] == "symbol"})
        missing_words, missing_paths = set(), set()
        for commit in sorted({fr["base_commit"] for fr in rows}):
            files = searchable_files(repo_dir, commit)       # {path: blob id} of the repo's code files
            scanner.scan(files)
            found = scanner.words_at(files)
            all_paths, folders = tree_paths(repo_dir, commit)
            path_set = set(all_paths)
            ext = None
            for fr in rows:
                if fr["base_commit"] != commit:
                    continue
                for ident in fr["identifiers"]:
                    if ident["kind"] == "symbol":
                        ident["at_base"] = ident["key"] in found
                    else:
                        ident["at_base"] = path_exists(ident["key"], all_paths, folders, path_set)
                    ident["status"] = "base" if ident["at_base"] else None
                    if ident["at_base"]:
                        continue
                    if ident["kind"] == "symbol":
                        if ext is None:
                            ext = external_names(repo_dir, commit, files, all_paths, scanner)
                        first = ident["root"] or ident["key"]
                        if first in ext:
                            ident["status"] = "external"
                            continue
                        missing_words.add(ident["key"])
                    else:
                        missing_paths.add(ident["key"])
        # history snapshots: where do the missing names exist?
        hits = collections.defaultdict(set)   # name -> commit times of the snapshots that have it
        for snap in snapshots[repo]:
            files = searchable_files(repo_dir, snap["commit"])
            scanner.scan(files)
            for word in scanner.words_at(files) & missing_words:
                hits[word].add(snap["time"])
            if missing_paths:
                all_paths, folders = tree_paths(repo_dir, snap["commit"])
                path_set = set(all_paths)
                for p in missing_paths:
                    if path_exists(p, all_paths, folders, path_set):
                        hits[p].add(snap["time"])
        for fr in rows:
            for ident in fr["identifiers"]:
                if ident["status"] is None:
                    older = any(t < fr["base_time"] for t in hits[ident["key"]])
                    newer = any(t > fr["base_time"] for t in hits[ident["key"]])
                    ident["status"] = ("both" if older and newer else "stale" if older
                                       else "future" if newer else "never")
        print(f"  {repo}: {len(scanner.found):,} file versions read, {len(missing_words)} names and "
              f"{len(missing_paths)} paths missing at some base commit", flush=True)


def identifier_summary(idents):
    """Claim-level view. 'Unexplained' = missing at base, not external, not named in the task, not seen
    or written earlier in the run, and not part of the claim's own command (i.e. not something the agent
    is about to create). These are the memory (or guess) candidates."""
    unexplained = [i for i in idents if i["status"] in ("stale", "future", "both", "never")
                   and not (i["in_task"] or i["seen_before"] or i["written_before"] or i["in_own_command"])]
    kinds = {i["status"] for i in unexplained}
    if not idents:
        cls = "no_identifiers"
    elif not unexplained:
        cls = "all_explained"
    elif kinds & {"stale", "future", "both"}:
        cls = "other_version"
    else:
        cls = "never"
    return {"n_identifiers": len(idents),
            "n_missing_at_base": sum(1 for i in idents if i["status"] != "base"),
            "n_unexplained": len(unexplained),
            "unexplained": [i["name"] for i in unexplained],
            "ident_class": cls}


def cmd_features():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_claims()
    chosen = select_claims(rows)
    tables = load_evidence_tables()
    by_traj = collections.defaultdict(list)
    for i in sorted(chosen):
        by_traj[rows[i]["trajectory_path"]].append(i)
    checks = collections.Counter()
    base_times = {}
    out_rows = []
    for t, (traj_path, idxs) in enumerate(sorted(by_traj.items())):
        traj = Trajectory(os.path.join(PROJECT, traj_path))
        for i in idxs:
            r = rows[i]
            base_commit, ev_table = tables[(r["model"], r["instance_id"])]
            repo = repo_of(r["instance_id"])
            if (repo, base_commit) not in base_times:
                base_times[(repo, base_commit)] = int(git(os.path.join(REPOS_DIR, repo), "-c", "log.mailmap=false",
                                                          "log", "-1", "--format=%ct", base_commit).strip())
            fr = {"cid": f"c{i:05d}", "unit_id": r["unit_id"], "group": chosen[i], "verdict": r["verdict"],
                  "model": r["model"], "network_none": r["network_none"],
                  "git_history_pruned": r["git_history_pruned"], "instance_id": r["instance_id"],
                  "repo": repo, "base_commit": base_commit, "base_time": base_times[(repo, base_commit)],
                  "message_index": r["message_index"], "step": r["step"], "n_steps": r["n_steps"],
                  "ogsf_type": r["ogsf_type"], "proof_pattern": r["proof_pattern"],
                  "manual_reviewed": r["manual_reviewed"], "manual_confirmed": r["manual_confirmed"],
                  "error_form": r["error_form"], "key_claim": r["key_claim"],
                  "trajectory_resolved": r["trajectory_resolved"],
                  "cites_dependency": any(e["kind"] == "dependency" for e in r["evidence"]),
                  "cites_web": any(e["kind"] == "web" for e in r["evidence"])}
            fr.update(evidence_features(r, traj, base_commit, ev_table, checks))
            fr["identifiers"] = identifier_candidates(r, traj)
            out_rows.append(fr)
        if (t + 1) % 50 == 0:
            print(f"trajectories {t + 1}/{len(by_traj)}", flush=True)
    print(f"evidence lines read: {checks['citations']} citations, {checks['sha_ok']} match the artifact's sha256",
          flush=True)
    resolve_identifiers(out_rows)
    for fr in out_rows:
        fr.update(identifier_summary(fr["identifiers"]))
    out_rows.sort(key=lambda fr: fr["cid"])
    with open(FEATURES, "w", encoding="utf-8") as f:
        for fr in out_rows:
            f.write(json.dumps(fr, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows):,} rows to {FEATURES}")


# ---------------------------------------------------------------------------------------------
# 6. Summary tables
# ---------------------------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n (as proportions)."""
    if n == 0:
        return [None, None]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


CLASSES = ["all", "some", "none", "no_repo_evidence"]
IDENT_CLASSES = ["no_identifiers", "all_explained", "other_version", "never"]


def table(rows, field="evidence_seen_before", classes=CLASSES):
    """Counts per class, plus proportions (with Wilson CIs) among claims that have repo evidence."""
    c = collections.Counter(r[field] for r in rows)
    with_ev = sum(c[k] for k in classes if k != "no_repo_evidence")
    out = {"n": len(rows), "counts": {k: c[k] for k in classes}}
    if field == "evidence_seen_before":
        some = [r["n_lines_seen"] / r["n_cited_lines"] for r in rows if r[field] == "some"]
        out["mean_share_of_lines_seen_when_some"] = round(sum(some) / len(some), 4) if some else None
    if "no_repo_evidence" in classes:
        out["n_with_repo_evidence"] = with_ev
        out["share_of_with_evidence"] = {
            k: {"p": round(c[k] / with_ev, 4) if with_ev else None, "ci95": wilson(c[k], with_ev)}
            for k in classes if k != "no_repo_evidence"}
    else:
        out["share"] = {k: {"p": round(c[k] / len(rows), 4) if rows else None, "ci95": wilson(c[k], len(rows))}
                        for k in classes}
    return out


def ident_table(rows):
    """Identifier statuses and claim-level classes."""
    status = collections.Counter(i["status"] for r in rows for i in r["identifiers"])
    unexplained_status = collections.Counter(
        i["status"] for r in rows for i in r["identifiers"]
        if i["status"] in ("stale", "future", "both", "never")
        and not (i["in_task"] or i["seen_before"] or i["written_before"] or i["in_own_command"]))
    with_ids = [r for r in rows if r["n_identifiers"]]
    k_unexpl = sum(1 for r in with_ids if r["n_unexplained"])
    k_other = sum(1 for r in with_ids if r["ident_class"] == "other_version")
    return {"claims": len(rows), "claims_with_identifiers": len(with_ids),
            "identifier_status": dict(status), "unexplained_identifier_status": dict(unexplained_status),
            "claim_class": {k: sum(1 for r in rows if r["ident_class"] == k) for k in IDENT_CLASSES},
            "p_claim_has_unexplained_missing": {"k": k_unexpl, "n": len(with_ids),
                                                "p": round(k_unexpl / len(with_ids), 4) if with_ids else None,
                                                "ci95": wilson(k_unexpl, len(with_ids))},
            "p_claim_has_other_version": {"k": k_other, "n": len(with_ids),
                                          "p": round(k_other / len(with_ids), 4) if with_ids else None,
                                          "ci95": wilson(k_other, len(with_ids))}}


def by(rows, key):
    out = collections.defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return dict(sorted(out.items()))


def cmd_summary():
    with open(FEATURES, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    ds = [r for r in rows if r["network_none"]]        # DeepSeek-V3.2 + DeepSeek-V4.1-Flash: no network
    qw = [r for r in rows if not r["network_none"]]    # Qwen3.6-27B + Qwen3.8-27B: network on, full git
    manual = [r for r in rows if r["manual_confirmed"]]
    mm = [r for r in manual if r["error_form"] in ("Misstatement", "Misreference")]
    mm_nonet = [r for r in mm if r["network_none"]]
    # controls for the M+M subset: other claims from the same reviewed trajectories
    mm_trajs = {(r["model"], r["instance_id"]) for r in mm_nonet}
    same_traj = [r for r in ds if (r["model"], r["instance_id"]) in mm_trajs]

    def both_views(sub):
        return {"anchored": table(sub), "any_observation": table(sub, "evidence_seen_before_any"),
                "file_seen_before": table(sub, "file_seen_before")}

    s = {
        "description": "RQ1 claim-level provenance on You Wang et al.'s SWE-bench Pro claims "
                       "(arXiv 2610.32338 artifact). See notes/rq1-claims.md for method and caveats.",
        "made_by": ".venv/bin/python -I scripts/rq1_claim_provenance.py features; then ... summary",
        "cells": "counts per class; share_of_with_evidence = share among claims with repository evidence, "
                 "with 95% Wilson intervals (claims in one trajectory are not independent, so these are too narrow)",
        "n_rows": len(rows),
        "groups": {g: len(v) for g, v in by(rows, lambda r: r["group"]).items()},
        "sha256_check": {"citations": sum(len(r["citations"]) for r in rows),
                         "sha_ok": sum(c["sha_ok"] for r in rows for c in r["citations"])},
        "no_repo_reason": dict(collections.Counter(r["no_repo_reason"] for r in rows if r["no_repo_reason"])),
        "verdict_x_evidence_seen": {
            "deepseek_no_network (main)": {v: both_views(sub) for v, sub in by(ds, lambda r: r["verdict"]).items()},
            "qwen_network_on": {v: both_views(sub) for v, sub in by(qw, lambda r: r["verdict"]).items()},
            "all_models": {v: both_views(sub) for v, sub in by(rows, lambda r: r["verdict"]).items()},
        },
        "manual_773_error_form_x_evidence_seen": {
            "all_models": {f: both_views(sub) for f, sub in by(manual, lambda r: r["error_form"]).items()},
            "deepseek_no_network": {f: both_views(sub) for f, sub in
                                    by([r for r in manual if r["network_none"]], lambda r: r["error_form"]).items()},
            "qwen_network_on": {f: both_views(sub) for f, sub in
                                by([r for r in manual if not r["network_none"]], lambda r: r["error_form"]).items()},
            "n_manual_confirmed_found": len(manual),
        },
        "model_x_network": {
            f"{m} (network_none={sub[0]['network_none']})": {v: both_views(s2) for v, s2 in
                                                              by(sub, lambda r: r["verdict"]).items()}
            for m, sub in by(rows, lambda r: r["model"]).items()},
        "misstatement_misreference_no_network": {
            "n": len(mm_nonet),
            "by_form": {f: len(sub) for f, sub in by(mm_nonet, lambda r: r["error_form"]).items()},
            "by_model": {m: len(sub) for m, sub in by(mm_nonet, lambda r: r["model"]).items()},
            "evidence": both_views(mm_nonet),
            "identifiers": ident_table(mm_nonet),
            "control_same_trajectories": {
                "refuted_not_MM": both_views([r for r in same_traj if r["verdict"] == "refuted"
                                              and r["error_form"] not in ("Misstatement", "Misreference")]),
                "supported_control": both_views([r for r in same_traj if r["verdict"] == "supported"]),
                "supported_control_identifiers": ident_table([r for r in same_traj if r["verdict"] == "supported"]),
            },
            "qwen_network_on_MM_for_comparison": {"evidence": both_views([r for r in mm if not r["network_none"]]),
                                                  "identifiers": ident_table([r for r in mm if not r["network_none"]])},
        },
        "identifiers_verdict": {
            "deepseek_no_network (main)": {v: ident_table(sub) for v, sub in by(ds, lambda r: r["verdict"]).items()},
            "qwen_network_on": {v: ident_table(sub) for v, sub in by(qw, lambda r: r["verdict"]).items()},
        },
        "evidence_seen_x_identifier_class (deepseek main)": {
            v: {e: dict(collections.Counter(r["ident_class"] for r in sub2))
                for e, sub2 in by(sub, lambda r: r["evidence_seen_before"]).items()}
            for v, sub in by(ds, lambda r: r["verdict"]).items()},
        "ogsf_x_evidence_seen (deepseek main)": {
            v: {t: table(sub2) for t, sub2 in by(sub, lambda r: r["ogsf_type"]).items()}
            for v, sub in by(ds, lambda r: r["verdict"]).items()},
        "ogsf_x_identifiers (deepseek main)": {
            v: {t: ident_table(sub2) for t, sub2 in by(sub, lambda r: r["ogsf_type"]).items()}
            for v, sub in by(ds, lambda r: r["verdict"]).items()},
        "mm_no_network_claims_with_unexplained_identifiers": [
            {"cid": r["cid"], "model": r["model"], "error_form": r["error_form"],
             "evidence_seen_before": r["evidence_seen_before"],
             "identifiers": [[i["name"], i["status"]] for i in r["identifiers"] if i["name"] in r["unexplained"]]}
            for r in mm_nonet if r["n_unexplained"]],
        "citations_with_sha_mismatch": [[r["cid"], c["file"]] for r in rows for c in r["citations"] if not c["sha_ok"]],
    }
    with open(SUMMARY, "w") as f:
        json.dump(s, f, indent=1)
    print(f"wrote {SUMMARY}")
    print_tables(s)


def fmt_view(t):
    sh = t.get("share_of_with_evidence", {})
    cells = []
    for k in CLASSES:
        n = t["counts"][k]
        if k == "no_repo_evidence":
            cells.append(f"{n}")
        else:
            p = sh[k]["p"]
            ci = sh[k]["ci95"]
            cells.append(f"{n} ({100 * p:.1f}% [{100 * ci[0]:.1f}-{100 * ci[1]:.1f}])" if p is not None else f"{n}")
    return " | ".join(cells)


def print_tables(s):
    """Markdown tables for the notes (counts; % of claims with repo evidence, 95% Wilson CI)."""
    print("\n## verdict x evidence_seen_before (anchored)  cells: n (% of claims with repo evidence [95% CI])")
    print("| subset | verdict | n | all | some | none | no_repo_evidence |\n|---|---|---|---|---|---|---|")
    for sub, d in s["verdict_x_evidence_seen"].items():
        for v, views in d.items():
            print(f"| {sub} | {v} | {views['anchored']['n']} | {fmt_view(views['anchored'])} |")
    print("\n## same, looser 'any observation' matching")
    print("| subset | verdict | n | all | some | none | no_repo_evidence |\n|---|---|---|---|---|---|---|")
    for sub, d in s["verdict_x_evidence_seen"].items():
        for v, views in d.items():
            print(f"| {sub} | {v} | {views['any_observation']['n']} | {fmt_view(views['any_observation'])} |")
    print("\n## file_seen_before")
    print("| subset | verdict | n | all | some | none | no_repo_evidence |\n|---|---|---|---|---|---|---|")
    for sub, d in s["verdict_x_evidence_seen"].items():
        for v, views in d.items():
            print(f"| {sub} | {v} | {views['file_seen_before']['n']} | {fmt_view(views['file_seen_before'])} |")
    print("\n## 773 manually confirmed faulty claims: error form x evidence_seen_before (anchored)")
    print("| subset | error form | n | all | some | none | no_repo_evidence |\n|---|---|---|---|---|---|---|")
    for sub in ("all_models", "deepseek_no_network", "qwen_network_on"):
        for f, views in s["manual_773_error_form_x_evidence_seen"][sub].items():
            print(f"| {sub} | {f} | {views['anchored']['n']} | {fmt_view(views['anchored'])} |")
    print("\n## model x network x verdict (anchored)")
    print("| model | verdict | n | all | some | none | no_repo_evidence |\n|---|---|---|---|---|---|---|")
    for m, d in s["model_x_network"].items():
        for v, views in d.items():
            print(f"| {m} | {v} | {views['anchored']['n']} | {fmt_view(views['anchored'])} |")
    mm = s["misstatement_misreference_no_network"]
    print(f"\n## Misstatement+Misreference, no-network (DeepSeek) subset: n={mm['n']} {mm['by_form']} {mm['by_model']}")
    print("| subset | n | all | some | none | no_repo_evidence |\n|---|---|---|---|---|---|")
    print(f"| M+M no-net, anchored | {mm['evidence']['anchored']['n']} | {fmt_view(mm['evidence']['anchored'])} |")
    print(f"| M+M no-net, any obs | {mm['evidence']['any_observation']['n']} | {fmt_view(mm['evidence']['any_observation'])} |")
    print(f"| M+M no-net, file seen | {mm['evidence']['file_seen_before']['n']} | {fmt_view(mm['evidence']['file_seen_before'])} |")
    c = mm["control_same_trajectories"]
    print(f"| same trajs: refuted not M+M | {c['refuted_not_MM']['anchored']['n']} | {fmt_view(c['refuted_not_MM']['anchored'])} |")
    print(f"| same trajs: supported control | {c['supported_control']['anchored']['n']} | {fmt_view(c['supported_control']['anchored'])} |")
    q = mm["qwen_network_on_MM_for_comparison"]["evidence"]["anchored"]
    print(f"| Qwen M+M (network on) | {q['n']} | {fmt_view(q)} |")
    print("\n## identifiers (claim level): has >=1 unexplained missing identifier; of those other-version")
    print("| subset | verdict | claims w/ ids | unexplained missing | other version (stale/future/both) | identifier statuses |")
    print("|---|---|---|---|---|---|")
    for sub, d in s["identifiers_verdict"].items():
        for v, t in d.items():
            u, o = t["p_claim_has_unexplained_missing"], t["p_claim_has_other_version"]
            print(f"| {sub} | {v} | {t['claims_with_identifiers']} | {u['k']} ({100 * u['p']:.1f}% "
                  f"[{100 * u['ci95'][0]:.1f}-{100 * u['ci95'][1]:.1f}]) | {o['k']} ({100 * o['p']:.1f}%) | "
                  f"{t['identifier_status']} |")
    for name, t in (("M+M no-net", mm["identifiers"]),
                    ("same trajs supported", c["supported_control_identifiers"]),
                    ("Qwen M+M", mm["qwen_network_on_MM_for_comparison"]["identifiers"])):
        u, o = t["p_claim_has_unexplained_missing"], t["p_claim_has_other_version"]
        if u["n"]:
            print(f"| {name} | - | {t['claims_with_identifiers']} | {u['k']} ({100 * u['p']:.1f}% "
                  f"[{100 * u['ci95'][0]:.1f}-{100 * u['ci95'][1]:.1f}]) | {o['k']} ({100 * o['p']:.1f}%) | "
                  f"{t['identifier_status']} |")


# ---------------------------------------------------------------------------------------------
# 7. Hand check: print random claims with what the script decided, to verify against the trajectory
# ---------------------------------------------------------------------------------------------

def cmd_handcheck(n, seed):
    with open(FEATURES, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    pool = [r for r in rows if r["evidence_seen_before"] != "no_repo_evidence"]
    pick = random.Random(seed).sample(pool, n)
    tidy = load_claims()
    tables = load_evidence_tables()
    for r in pick:
        t = tidy[int(r["cid"][1:])]
        traj = Trajectory(os.path.join(PROJECT, t["trajectory_path"]))
        _, ev_table = tables[(t["model"], t["instance_id"])]
        print("=" * 100)
        print(f"{r['cid']}  {r['model']}  {r['verdict']}  message_index={r['message_index']}  "
              f"script says: {r['evidence_seen_before']} ({r['n_lines_seen']}/{r['n_cited_lines']} lines), "
              f"any-obs: {r['evidence_seen_before_any']}, file_seen: {r['file_seen_before']}")
        print(f"trajectory: {t['trajectory_path']}")
        print(f"CLAIM: {t['claim']}")
        print("earlier commands (tool message index: command):")
        for i in sorted(traj.obs):
            if i < r["message_index"]:
                print(f"   [{i}] {traj.obs_cmd[i][:220]!r}")
        for ev in t["evidence"]:
            if ev["kind"] != "repository":
                continue
            prov = ev_table[ev["id"]]["repository_provenance"]
            name = r["repo"] if prov["repository_path"] == "." else SUBMODULES[prov["repository_path"]][0]
            lines = blob_text(os.path.join(REPOS_DIR, name), prov["blob_oid"]).splitlines()
            fname = ev["source"].rsplit("/", 1)[-1]
            print(f"--- cited {ev['source']}:{ev['start_line']}-{ev['end_line']}")
            for k in range(ev["start_line"], ev["end_line"] + 1):
                line = norm(lines[k - 1]) if k <= len(lines) else ""
                if is_trivial(line):
                    mark = "trivial"
                else:
                    first = traj.first_seen(line, fname)
                    mark = f"first seen at msg {first}" if first is not None else "never shown"
                print(f"   {k:>5} [{mark:>22}] {line[:150]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "features":
        cmd_features()
    elif sys.argv[1] == "summary":
        cmd_summary()
    elif sys.argv[1] == "handcheck":
        cmd_handcheck(int(sys.argv[2]), int(sys.argv[3]))
    else:
        sys.exit(__doc__)
