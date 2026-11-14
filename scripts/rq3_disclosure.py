"""RQ3 disclosure: does an agent's reasoning reveal that it relies on memory?

We read agent trajectories and look, step by step, for two kinds of phrases:
  * explicit RECALL statements ("I recall ...", "from memory", "the upstream fix", "in newer versions of Django")
  * HEDGES ("probably", "I think", "might", "should be in ...")
Each agent step has up to three text channels:
  hidden   the model's reasoning text (reasoning_content / reasoning), not shown to the user by the harness
  visible  the normal message text (with code blocks taken out)
  command  the shell command(s) the step ran
We then answer four questions:
  1. How often does each channel contain a recall statement (per step, per trajectory), per model?
  2. When an agent names a fact before it has seen it, is the sentence hedged / flagged, or asserted plainly?
     Separately for facts that turned out right and wrong (pilot file paths; You Wang et al.'s claims).
  3. How precise are the lexicons? (a fixed random sample, seed 0, judged by an LLM reader = Claude, not a human)
  4. In our own trap runs: how many steps talk about memory in hidden vs visible text?

Data (all read-only; downloads are treated as data, never run):
  data/raw/<run>/trajs/...           public mini-swe-agent runs on SWE-bench Verified (3 models x 500 tasks)
  data/faultyclaims/...              You Wang et al.'s 400 trajectories + claims_tidy.jsonl + claims_seen_before.jsonl
  data/deepswe/...                   DeepSWE trials (only trials with reasoning text; gemini-3-7-flash excluded
                                     because its reasoning text is a copy of the visible message)
  outputs/{deepseek-v3.2,kimi-k2.5,minimax-m2.5}.jsonl   pilot "named before seen" file paths
  outputs/agent_runs, outputs/runs/*                      our pilot trap runs
  ~/Projects/rr-targeted/runs/*                            the other session's targeted trap runs (read only)

Writes:
  outputs/rq3/disclosure_summary.json     all key numbers + the lexicons
  outputs/rq3/fig_*.csv                   tables behind every number
  data/rq3_validation/sample.jsonl        the validation snippets WITH text (stays under data/, gitignored,
                                          because DeepSWE text may not be redistributed)
  outputs/rq3/fig_lexicon_validation.csv  the same snippets as ids only, plus the LLM-judged labels
                                          (if this file already has labels, it is read, never overwritten)

Usage (from the project root):
  .venv/bin/python -I scripts/rq3_disclosure.py
"""

import collections
import csv
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs', 'rq3')
VAL_DIR = os.path.join(ROOT, 'data', 'rq3_validation')
TARGETED_RUNS = os.path.expanduser('~/Projects/rr-targeted/runs')  # other session: we only READ these files

N_BOOT = 2000  # bootstrap resamples (over trajectories)
SEED = 0

# ---------------------------------------------------------------------------------------------
# 1. The lexicons. Every pattern is listed with what it catches and what it deliberately skips.
#    All patterns are case-insensitive. (?-i:...) switches case-sensitivity back on for names.
# ---------------------------------------------------------------------------------------------

# Library / language names used by the version-talk patterns (case-sensitive, so "go" or "node" in
# ordinary English do not count; "Go 1.18+" and "Node 18+" do).
LIBS = (r"(?-i:Django|SymPy|sympy|Python|NumPy|numpy|pandas|pytest|Matplotlib|matplotlib|Sphinx|sphinx|"
        r"scikit-learn|sklearn|Astropy|astropy|xarray|requests|Flask|flask|seaborn|Pylint|pylint|Qt|PyQt\d?|"
        r"Go|Node(?:\.js)?|React|TypeScript|Rust|Java|Ansible|ansible)")
# Projects of the SWE-bench Verified tasks (used by "the actual Django fix").
PROJECTS = (r"(?-i:Django|SymPy|Sphinx|pytest|Matplotlib|matplotlib|Astropy|astropy|xarray|scikit-learn|sklearn|"
            r"Flask|requests|seaborn|Pylint|pylint)")

# (name, family, regex, what it is for)
# family A = explicit memory statements and references to the real upstream project (the main measure)
# family B = version-history talk (weaker: version facts can also be read from the task text or an error)
RECALL_PATTERNS = [
    ('i_recall', 'A',
     r"\bI(?:'m|\s+am)?\s+(?:(?:also|now|vaguely|strongly|clearly|definitely|distinctly|faintly|actually|really|do|"
     r"did|can|could|don't|do\s+not|didn't|can't|cannot|couldn't|seem\s+to)\s+){0,2}(?:recall|remember)(?:ed|ing)?\b"
     r"(?!\s+to\b)"  # "I remember to ..." is an intention, not a memory
     r"(?!,?\s+(?:that\s+)?(?:we|I)\s+(?:saw|ran|did|tried|added|changed|modified|made|wrote|created|looked|checked|"
     r"found|edited|had|have|were|was|got|used|fixed|applied|updated)\b)"  # "I remember we saw ..." = this run
     r"(?!\s+(?:the|our|my|this)\s+(?:earlier|previous|last|above|first|initial)\b)",
     '"I recall", "I remember", "I vaguely remember", "I can\'t recall"; skips "I remember to ..." and memories '
     'of this run ("I remember we saw ...", "I recall the earlier output")'),
    ('if_i_recall', 'A',
     r"\b(?:if|as\s+far\s+as|as)\s+I\s+(?:can\s+)?(?:recall|remember)\b|\bif\s+(?:my\s+)?memory\s+serves\b|\bIIRC\b|"
     r"\bAFAIK\b|\bas\s+far\s+as\s+I\s+know\b|\bto\s+(?:the\s+best\s+of\s+)?my\s+knowledge\b|"
     r"\bfrom\s+what\s+I\s+(?:can\s+)?(?:recall|remember|know)\b",
     '"if I recall correctly", "as far as I know", "to my knowledge", "IIRC", "AFAIK", "from what I remember"'),
    ('lets_recall', 'A',
     r"\blet(?:'s|\s+us|\s+me)\s+(?:try\s+to\s+|try\s+and\s+)?(?:recall|remember)\b"
     r"(?!\s*:?\s*(?:(?:what|where|how|why|that|when)\s+)?(?:I|we|I'm|we're|I've|we've|I'd|we'd|I'll|we'll)\b)"
     r"(?!\s*:?\s*(?:the|our|my|this|that)\s+(?:problem|issue|task|PR\s+description|pull\s+request\s+description|"
     r"description|current|state|context|conversation|situation|error|errors|output|outputs|traceback|"
     r"tests?\s+(?:output|results?|failures?)|results?|requirements?|plan|changes|steps?|instructions?|previous|"
     r"earlier|last|bug|failure|goal|objective|constraints?|format|rules?)\b)",
     '"let me recall ...", "let\'s try to remember ..."; skips recalling this run ("let me recall the issue / '
     'what I was doing / the current state")'),
    ('from_memory', 'A',
     r"(?<!load\s)(?<!loaded\s)(?<!read\s)(?<!free\s)(?<!freed\s)(?<!evict\s)(?<!delete\s)(?<!deleted\s)"
     r"(?<!remove\s)(?<!removed\s)(?<!release\s)(?<!released\s)(?<!fetch\s)(?<!retrieve\s)(?<!serve\s)"
     r"(?<!served\s)(?<!loads\s)(?<!reads\s)(?<!data\s)"
     r"\bfrom\s+(?:my\s+|the\s+model's\s+)?memory\b(?!\s+(?:allocat|usage|leak|address|buffer|map|cache|pool))",
     '"from memory", "from my memory"; skips computer-memory uses ("load from memory", "freed from memory")'),
    ('my_memory', 'A',
     r"\b(?:my|the\s+model's)\s+(?:memory|memories|recollection|knowledge|training(?:\s+data)?)\b|"
     r"\b(?:prior|background|parametric|pretrained|pre-trained|general)\s+knowledge\b|"
     r"\b(?:vague|faint|distinct|fuzzy|hazy)\s+(?:memory|recollection)\b|\bknowledge\s+cutoff\b",
     '"my memory", "my knowledge", "my training data", "prior/general knowledge", "a vague memory", '
     '"knowledge cutoff"'),
    ('search_memory', 'A',
     r"\bsearch(?:ing)?\s+(?:my\s+)?(?:memory|knowledge)\b",
     '"search memory", "search my knowledge" (DeepSeek\'s terse style)'),
    ('memorized', 'A',
     r"\bmemori[sz](?:e|ed|es|ing)\b",
     '"memorized", "memorize" (not "memoize", which is caching)'),
    ('upstream', 'A',
     r"(?<![\w/.\-])(?<!somewhere\s)(?<!further\s)(?<!farther\s)(?<!higher\s)(?<!way\s)upstream(?![\w/.\-])"
     r"(?!\s+(?:servers?|proxy|proxies|callers?|call\s+chain|nodes?|stages?|layers?|functions?|methods?|operations?|"
     r"logic|data|dependencies|dependency|components?|tasks?|jobs?|branch|remote|code\s+path|"
     r"in\s+the\s+(?:call|stack|pipeline|chain|data|flow|code\s+path))\b)",
     '"upstream" (the real project); skips paths/identifiers (plugins/upstream, x.upstream) and the call-chain '
     'sense ("somewhere upstream", "upstream callers/servers", "upstream in the call stack")'),
    ('actual_pr_commit', 'A',
     r"\b(?:the|an?)\s+(?:actual|real|upstream|merged|official)\s+(?:PR|PRs|pull\s+requests?|commits?)\b",
     '"the actual PR", "a real commit", "the merged PR" (the task\'s source change). Deliberately NOT "the actual '
     'fix/patch/change" or "the original PR/commit": those usually mean the agent\'s own fix, the PR description '
     'or a commit read with git'),
    ('based_on_real', 'A',
     r"\b(?:based\s+on|derived\s+from|taken\s+from|generated\s+from|comes?\s+from|came\s+from|extracted\s+from)\s+"
     r"(?:a|an|the)\s+(?:real|actual|specific\s+real|upstream|real-world|existing\s+upstream)\s+"
     r"(?:PR|pull\s+request|commit|patch|fix|change|issue|ticket|feature|bug|GitHub)\b",
     '"this task is based on a real PR", "derived from the actual commit"'),
    ('actual_project_fix', 'A',
     r"\b(?:actual|real|official|original|merged|released)\s+(?:" + PROJECTS + r"\s+|upstream\s+)"
     r"(?:fix|patch|commit|PR|pull\s+request|change|solution)\b|"
     r"\b(?:the\s+)?(?:actual|real|official|original|merged|released)\s+(?:fix|patch|change|solution|"
     r"implementation)\s+(?:in|from|by|on)\s+(?:the\s+)?(?:upstream|GitHub|" + PROJECTS + r")\b",
     '"the actual Django fix", "the real fix in Django", "the official upstream patch"'),
    # family B: version-history talk
    ('newer_versions', 'B',
     r"\b(?:in|on|with|for|from|since|until|of|before|after)\s+(?:the\s+|a\s+|an\s+)?(?:newer|later|more\s+recent|"
     r"recent|modern|future|upcoming|subsequent|latest|older|earlier)\s+(?:major\s+|minor\s+)?(?:versions?|releases?)\b",
     '"in newer versions", "in later releases", "from an older version"; needs a preposition, so "install an older '
     'version" (an action) does not count'),
    ('modern_lib', 'B',
     r"\b(?:in|on|with|for|from|since|of|than)\s+(?:the\s+)?(?:modern|newer|recent|later|older|latest|more\s+recent)\s+"
     r"(?:versions?\s+of\s+)?" + LIBS + r"\b",
     '"in modern Django", "in newer Python", "in older versions of numpy" (not "current Django": that is this repo)'),
    ('lib_version_range', 'B',
     LIBS + r"\s*v?\d+(?:\.\d+)*(?:\.x\b|\+(?![+\w])|\s+and\s+(?:later|above|newer|up)\b|"
     r"\s+or\s+(?:later|newer|above|higher)\b|\s+onwards?\b)",
     '"Django 4.x", "Python 3.8+", "sympy 1.12 and later" (a version RANGE of a named library)'),
    ('changed_in_version', 'B',
     r"\b(?:added|introduced|deprecated|removed|changed|fixed|renamed|moved|dropped|backported|merged|landed)\s+in\s+"
     r"(?:(?:" + LIBS + r"|version|v)\s*)?\d+(?:\.\d+)+\b",
     '"removed in Python 3.10", "introduced in Django 3.2", "fixed in 3.4.2"'),
    ('historically', 'B', r"\bhistorically\b", '"historically"'),
]

HEDGE_PATTERNS = [
    ('probably', r"\bprobabl[ey]\b", '"probably", "probable"'),
    ('likely', r"\b(?:un)?likely\b", '"likely", "most likely", "unlikely"'),
    ('maybe_perhaps', r"\b(?:maybe|perhaps|possibly|presumably|supposedly|apparently|conceivably)\b",
     '"maybe", "perhaps", "possibly", "presumably", "supposedly", "apparently"'),
    ('i_think', r"\bI\s+(?:think|believe|guess|suspect|assume|suppose|reckon|imagine)\b|"
                r"\bI'd\s+(?:guess|assume|say|think)\b|\bmy\s+(?:guess|assumption|hunch)\b",
     '"I think", "I believe", "I guess", "I suspect", "I assume", "my guess"'),
    ('not_sure', r"\b(?:not|n't)\s+(?:entirely\s+|completely\s+|totally\s+|quite\s+|100%\s+|fully\s+)?(?:sure|certain)\b|"
                 r"\bunsure\b|\buncertain\b|\bif\s+I'm\s+not\s+mistaken\b",
     '"not sure", "not certain", "unsure", "uncertain", "if I\'m not mistaken"'),
    ('seems', r"\bseem(?:s|ed|ingly)?\b|\bappears?\s+to\b|\bit\s+appears\b",
     '"seems", "seemingly", "appears to", "it appears"'),
    ('might_could', r"\bmight\b|\bcould\s+be\b|\bmay\s+(?:be|have)\b", '"might", "could be", "may be", "may have"'),
    ('should_be_loc', r"\bshould\s+be\s+(?:in|at|under|inside|within|located|defined|somewhere|found|there|here|"
                      r"available|present|importable|exported|declared|implemented)\b",
     '"should be in/at/defined/located ..." (only the location/existence sense of "should be")'),
]

# Broad "memory-ish" words. Only used to split the UNMATCHED sentences into two groups for the recall
# check (sentences with one of these words are where missed recall statements would hide).
BROAD = re.compile(
    r"\b(?:recall\w*|remember\w*|memor\w*|upstream|histor\w*|versions?|releases?|released|newer|older|later|"
    r"earlier|modern|deprecat\w*|introduc\w*|original|actual|real|official|known|familiar|commits?|PRs?|"
    r"pull\s+requests?|tickets?|changelog|knowledge|training|I\s+know|I've\s+seen|I\s+have\s+seen)\b", re.I)

# The regex the pilot used (scripts/trap_pipeline.py), kept only to recompute the pilot's numbers.
PILOT_MEM = re.compile(r'(from memory|recall|remember|upstream|modern (?:Django|SymPy|sympy)|newer version)', re.I)

RECALL_RE = {name: re.compile(rx, re.I) for name, fam, rx, _ in RECALL_PATTERNS}
FAMILY = {name: fam for name, fam, _, _ in RECALL_PATTERNS}
A_RE = re.compile('|'.join('(?:%s)' % rx for _, fam, rx, _ in RECALL_PATTERNS if fam == 'A'), re.I)
B_RE = re.compile('|'.join('(?:%s)' % rx for _, fam, rx, _ in RECALL_PATTERNS if fam == 'B'), re.I)
HEDGE_RE = {name: re.compile(rx, re.I) for name, rx, _ in HEDGE_PATTERNS}
H_RE = re.compile('|'.join('(?:%s)' % rx for _, rx, _ in HEDGE_PATTERNS), re.I)

# Rule 1 (first-person memory phrases only): "I recall ..." often means "I remember what happened earlier in
# this run" ("I recall the instructions", "I remember the earlier output"). If the SAME sentence also has
# one of these in-run cue words, the first-person phrase does not count.
CUE_FILTERED = {'i_recall', 'if_i_recall', 'lets_recall'}
IN_RUN_CUES = re.compile(
    r"\b(?:earlier|previous(?:ly)?|above|initial\s+output|original\s+file|just\s+(?:saw|ran|read|looked|checked|did|"
    r"created|wrote|edited|tried|used)|(?:I|we)\s+(?:saw|ran|read|looked\s+at|checked|did|created|wrote|edited|used|"
    r"tried|added|changed|modified|made|opened|viewed|printed|grepped)|instructions?|prompt|"
    r"PR\s+(?:says|said|description|statement)|problem\s+statement|issue\s+(?:says|said|description|statement|text)|"
    r"task\s+(?:says|said|description|statement)|spec(?:ification)?|the\s+output|traceback|error\s+message|"
    r"test\s+(?:output|results?|failures?|run)|this\s+(?:run|session|conversation))\b", re.I)
# Rule 2 (version patterns with a dotted number, e.g. "Python 3.8+", "removed in NumPy 2.0"): if every dotted
# version number in the match already appeared in the task text or in an earlier tool output of the same
# trajectory, the agent could have READ it, so it does not count as recall.
VERSION_CHECKED = {'lib_version_range', 'changed_in_version'}
DOTTED = re.compile(r'\d+(?:\.\d+)+')


def version_was_seen(match_text, seen):
    nums = DOTTED.findall(match_text)
    return bool(nums) and all(re.search(r'(?<![\d.])' + re.escape(n) + r'(?![\d])', seen) for n in nums)


def recall_hits(text, seen=None):
    """Names of the recall patterns that match this text (after rules 1 and 2).
    seen = task text + earlier tool outputs of the trajectory (None = skip rule 2)."""
    hits = []
    for n, rx in RECALL_RE.items():
        if n in VERSION_CHECKED and seen is not None:
            if any(not version_was_seen(m.group(0), seen) for m in rx.finditer(text)):
                hits.append(n)
        elif rx.search(text):
            if n in CUE_FILTERED and IN_RUN_CUES.search(text):
                continue
            hits.append(n)
    return hits


def recall_families(text, seen=None):
    """(family A matched, family B matched) for one sentence."""
    if not (A_RE.search(text) or B_RE.search(text)):  # fast pre-check
        return False, False, []
    hits = recall_hits(text, seen)
    return any(FAMILY[h] == 'A' for h in hits), any(FAMILY[h] == 'B' for h in hits), hits


def hedge_hits(text):
    return [n for n, rx in HEDGE_RE.items() if rx.search(text)]


# ---------------------------------------------------------------------------------------------
# 2. Text helpers
# ---------------------------------------------------------------------------------------------

FENCE = re.compile(r'```.*?(?:```|\Z)', re.S)  # fenced code block (or an unclosed one running to the end)
FAKE_OUTPUT = re.compile(r'<returncode>.*?</returncode>|<output>.*?(?:</output>|\Z)', re.S)
# a sentence ends at a line break, or at . ! ? followed by space and a capital/quote/bracket
SPLIT = re.compile(r'\n+|(?<=[.!?])\s+(?=[A-Z0-9`"\'(\[*_#-])')


def sentence_spans(text):
    """(start, end) character spans of the sentences in text."""
    spans, start = [], 0
    for m in SPLIT.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def sentences(text):
    """Unique sentences (in order) of a text; very short pieces are dropped."""
    out, seen = [], set()
    for a, b in sentence_spans(text):
        s = text[a:b].strip()
        if len(s) >= 3 and s not in seen:
            seen.add(s)
            out.append((s, a))
    return out


def as_text(c):
    if isinstance(c, list):
        return '\n'.join(x.get('text', '') for x in c if isinstance(x, dict))
    return c if isinstance(c, str) else ''


# ---------------------------------------------------------------------------------------------
# 3. Loaders. Each returns a list of steps; a step is a dict with the raw channel texts.
# ---------------------------------------------------------------------------------------------

SWEBV_RUNS = {  # run folder -> model name
    '20251201_mini-v1.17.1_deepseek-v3.2-reasoner': 'DeepSeek-V3.2',
    '20260217_mini-v2.0.0_kimi-k2-5-high': 'Kimi K2.5',
    '20260217_mini-v2.0.0_minimax-2-5-high': 'MiniMax M2.5',
}
FC_MODELS = {  # trajectory folder -> name used in You Wang et al.'s paper
    'deepseek-v3.2': 'DeepSeek-V3.2', 'deepseek-flash': 'DeepSeek-V4.1-Flash',
    'qwen3.6-27b': 'Qwen3.6-27B', 'qwen3.8-27b': 'Qwen3.8-27B',
}


def observations_after(msgs, i):
    """Text of the tool/user messages that follow assistant message i (what the agent saw next)."""
    out = []
    for n in msgs[i + 1:]:
        if n.get('role') == 'assistant':
            break
        out.append(as_text(n.get('content')))
    return '\n'.join(out)


def task_text(msgs):
    """System + user messages before the first agent step."""
    out = []
    for m in msgs:
        if m.get('role') == 'assistant':
            break
        out.append(as_text(m.get('content')))
    return '\n'.join(out)


def swebv_steps(traj):
    """Same channel split as scripts/provenance_pilot.py: (reasoning, visible text without ``` blocks, commands)."""
    steps = []
    msgs = traj['messages']
    for i, m in enumerate(msgs):
        if m.get('role') != 'assistant':
            continue
        resp = (m.get('extra') or {}).get('response') or {}
        msg = ((resp.get('choices') or [{}])[0].get('message') or {}) if isinstance(resp, dict) else {}
        reasoning = m.get('reasoning') or msg.get('reasoning_content') or msg.get('reasoning') or ''
        if not reasoning and isinstance(msg.get('thinking_blocks'), list):
            reasoning = '\n'.join(b.get('thinking', '') for b in msg['thinking_blocks'] if isinstance(b, dict))
        content = as_text(m.get('content'))
        cmds = re.findall(r'```(?:bash)?\n(.*?)```', content, re.S)
        for tc in (m.get('tool_calls') or msg.get('tool_calls') or []):
            try:
                cmds.append(json.loads(tc['function']['arguments']).get('command', ''))
            except Exception:
                pass
        visible = re.sub(r'```.*?```', ' ', content, flags=re.S)
        steps.append({'hidden': reasoning or '', 'visible': visible, 'command': '\n'.join(cmds),
                      'obs': observations_after(msgs, i)})
    return steps


def tool_call_commands(m):
    """Commands from an OpenAI-style tool_calls list (sometimes stored as a JSON string)."""
    tcs = m.get('tool_calls') or []
    if isinstance(tcs, str):
        try:
            tcs = json.loads(tcs)
        except Exception:
            tcs = []
    cmds = []
    for tc in tcs:
        try:
            a = tc['function']['arguments']
            cmds.append((json.loads(a) if isinstance(a, str) else a).get('command', ''))
        except Exception:
            pass
    return cmds


def chat_steps(traj):
    """mini-swe-agent tool-call trajectories (You Wang et al.'s runs and our trap runs)."""
    steps = []
    msgs = traj['messages']
    for i, m in enumerate(msgs):
        if m.get('role') != 'assistant':
            continue
        steps.append({'hidden': m.get('reasoning_content') or m.get('reasoning') or '',
                      'visible': as_text(m.get('content')), 'command': '\n'.join(tool_call_commands(m)),
                      'obs': observations_after(msgs, i)})
    return steps


def deepswe_steps(traj):
    """DeepSWE (Harbor ATIF) agent steps, same fields as scripts/summarize_deepswe.py iter_steps()."""
    steps = []
    for s in traj.get('steps', []):
        if s.get('source') != 'agent':
            continue
        msg = s.get('message')
        results = (s.get('observation') or {}).get('results') or []
        steps.append({'hidden': (s.get('reasoning_content') or '').strip(),
                      'visible': msg.strip() if isinstance(msg, str) else '',
                      'command': '\n'.join(str((c.get('arguments') or {}).get('command', ''))
                                           for c in (s.get('tool_calls') or [])),
                      'obs': '\n'.join(r.get('content') if isinstance(r.get('content'), str)
                                       else json.dumps(r.get('content')) for r in results)})
    return steps


def deepswe_task_text(traj):
    return '\n'.join(s.get('message') or '' for s in traj.get('steps', [])
                     if s.get('source') in ('system', 'user') and isinstance(s.get('message'), str))


def clean_channels(step, stats):
    """Text we actually search: code blocks and typed fake tool outputs removed from hidden/visible text.
    If the visible text is an exact copy of the hidden text, the visible copy is dropped (counted in stats)."""
    hidden = FENCE.sub(' ', step['hidden'])
    visible = step['visible']
    if FAKE_OUTPUT.search(visible):
        stats['steps_with_fake_tool_output_removed'] += 1
        visible = FAKE_OUTPUT.sub(' ', visible)
    visible = FENCE.sub(' ', visible)
    if step['hidden'].strip() and step['hidden'].strip() == step['visible'].strip():
        stats['steps_visible_equal_hidden_dropped'] += 1
        visible = ''
    return {'hidden': hidden, 'visible': visible, 'command': step['command']}


def iter_trajectories(stats):
    """Yield (dataset, model, traj_id, steps, task_text, extra) for every trajectory in the three big datasets."""
    # SWE-bench Verified
    for run, model in SWEBV_RUNS.items():
        for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'raw', run, 'trajs', '*', '*.traj.json'))):
            with open(f) as fh:
                traj = json.load(fh)
            yield ('swebv', model, os.path.basename(f)[:-len('.traj.json')], swebv_steps(traj),
                   task_text(traj['messages']), {})
    # You Wang et al. (SWE-bench Pro tasks)
    status = json.load(open(os.path.join(ROOT, 'data/faultyclaims/reasoning_replication/data/resolution_status.json')))
    for folder, model in FC_MODELS.items():
        for f in sorted(glob.glob(os.path.join(ROOT, 'data/faultyclaims/reasoning_replication/data', folder, '*.json'))):
            with open(f) as fh:
                traj = json.load(fh)
            iid = os.path.basename(f)[:-5]
            env = traj['info']['config']['environment']
            yield 'faultyclaims', model, iid, chat_steps(traj), task_text(traj['messages']), {
                'network_none': '--network=none' in (env.get('run_args') or []),
                'resolved': status.get(folder, {}).get(iid) == 'resolved'}
    # DeepSWE: trials with reasoning text only
    with open(os.path.join(ROOT, 'data/deepswe/sample_manifest.csv')) as fh:
        manifest = list(csv.DictReader(fh))
    for m in manifest:
        if m['model'] == 'gemini-3-7-flash':
            stats['deepswe_excluded_gemini_3_7_flash'] += 1
            continue
        p = os.path.join(ROOT, 'data/deepswe/trials', m['version'], m['trial_name'], 'trajectory.json')
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            traj = json.load(fh)
        steps = deepswe_steps(traj)
        if not any(s['hidden'] for s in steps):
            stats['deepswe_excluded_no_reasoning'] += 1
            continue
        yield 'deepswe', m['model'], m['trial_name'], steps, deepswe_task_text(traj), {
            'effort': m['reasoning_effort'], 'version': m['version'], 'scope': m['scope'], 'passed': m['passed']}


# ---------------------------------------------------------------------------------------------
# 4. Validation sample: reservoir sampling (uniform, seed 0) over unique sentences, by stratum.
# ---------------------------------------------------------------------------------------------

STRATA = {  # stratum -> (lexicon, per-source sizes or total size)
    'recall_A_matched': ('recall', {'swebv': 40, 'faultyclaims': 30, 'deepswe': 30}),
    'recall_B_only_matched': ('recall', {'swebv': 20, 'faultyclaims': 15, 'deepswe': 15}),
    'recall_unmatched_broad': ('recall', {'all': 50}),
    'recall_unmatched_other': ('recall', {'all': 50}),
    'hedge_matched': ('hedge', {'all': 80}),
    'hedge_unmatched': ('hedge', {'all': 40}),
}


class Reservoir:
    """Keep a uniform random sample of k items from a stream (Algorithm R), reproducible with a seed."""

    def __init__(self, k, seed):
        self.k, self.n, self.items, self.rng = k, 0, [], random.Random(seed)

    def add(self, item_fn):
        # item_fn builds the item only when it is kept (saves memory)
        if self.n < self.k:
            self.items.append(item_fn())
        else:
            j = self.rng.randrange(self.n + 1)
            if j < self.k:
                self.items[j] = item_fn()
        self.n += 1


def make_reservoirs():
    res, i = {}, 0
    for stratum, (_, sizes) in STRATA.items():
        for src, k in sizes.items():
            res[(stratum, src)] = Reservoir(k, SEED + i)
            i += 1
    return res


# ---------------------------------------------------------------------------------------------
# 5. Bootstrap helpers (resampling trajectories with replacement)
# ---------------------------------------------------------------------------------------------

def boot_mean(values):
    """Mean of per-trajectory values with a 95% percentile bootstrap CI."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return None, None, None
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    b = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_ratio(num, den):
    """sum(num)/sum(den) over trajectories, with a 95% bootstrap CI over trajectories."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    if den.sum() == 0:
        return None, None, None
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(num), size=(N_BOOT, len(num)))
    with np.errstate(invalid='ignore', divide='ignore'):
        b = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    b = b[~np.isnan(b)]
    return float(num.sum() / den.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_diff_independent(a, b):
    """Difference of two group means (a - b), groups resampled independently."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(SEED)
    ia = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    ib = rng.integers(0, len(b), size=(N_BOOT, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_group_diff(rows, unit_key, in_g1, in_g2, label):
    """Share of `label` in group 1 minus group 2, when both groups come from the same trajectories:
    resample trajectories once and recompute both shares in each resample (95% CI)."""
    units = sorted({r[unit_key] for r in rows if in_g1(r) or in_g2(r)})
    pos = {u: i for i, u in enumerate(units)}
    n1, k1, n2, k2 = (np.zeros(len(units)) for _ in range(4))
    for r in rows:
        if in_g1(r):
            n1[pos[r[unit_key]]] += 1
            k1[pos[r[unit_key]]] += r[label]
        if in_g2(r):
            n2[pos[r[unit_key]]] += 1
            k2[pos[r[unit_key]]] += r[label]
    rng = np.random.default_rng(SEED)
    b = rng.integers(0, len(units), size=(N_BOOT, len(units)))
    with np.errstate(invalid='ignore', divide='ignore'):
        d = k1[b].sum(axis=1) / n1[b].sum(axis=1) - k2[b].sum(axis=1) / n2[b].sum(axis=1)
    d = d[~np.isnan(d)]
    point = k1.sum() / n1.sum() - k2.sum() / n2.sum()
    return {'n_group1': int(n1.sum()), 'n_group2': int(n2.sum()), 'share_group1': r4(k1.sum() / n1.sum()),
            'share_group2': r4(k2.sum() / n2.sum()), 'diff': r4(point),
            'ci_lo': r4(float(np.percentile(d, 2.5))), 'ci_hi': r4(float(np.percentile(d, 97.5)))}


def r4(x):
    return None if x is None else round(x, 4)


def write_csv(name, rows, fields=None):
    path = os.path.join(OUT, name)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# ---------------------------------------------------------------------------------------------
# 6. Part 1: recall statements per step and channel
# ---------------------------------------------------------------------------------------------

CHANNELS = ('hidden', 'visible', 'command')


def analyze_channels(stats):
    """One pass over all trajectories: per-trajectory counts, per-pattern counts, validation sample."""
    traj_rows = []
    pattern_steps = collections.Counter()  # (dataset, model, channel, pattern) -> steps with a hit
    pattern_sents = collections.Counter()  # (dataset, model, channel, pattern) -> unique sentences with a hit
    stratum_sizes = collections.Counter()  # (stratum, source) -> number of sentences in the pool
    res = make_reservoirs()
    t0 = time.time()
    for k, (ds, model, tid, raw_steps, task, extra) in enumerate(iter_trajectories(stats)):
        row = {'dataset': ds, 'model': model, 'traj_id': tid, 'n_steps': len(raw_steps)}
        for ch in CHANNELS:
            for key in ('nonempty', 'A', 'B', 'any', 'hedge'):
                row[f'{ch}_{key}'] = 0
        seen = task  # what the agent could have read so far: the task, then every tool output
        for step_no, raw in enumerate(raw_steps, 1):
            st = clean_channels(raw, stats)
            for ch in CHANNELS:
                text = st[ch]
                if not text.strip():
                    continue
                row[f'{ch}_nonempty'] += 1
                got_a = got_b = got_h = False
                step_patterns = set()
                for sent_i, (s, offset) in enumerate(sentences(text)):
                    a, b, names = recall_families(s, seen)
                    h = bool(H_RE.search(s))
                    for n in names:
                        pattern_sents[(ds, model, ch, n)] += 1
                    step_patterns.update(names)
                    got_a |= a
                    got_b |= b
                    got_h |= h
                    if ch == 'command':
                        continue  # the validation sample uses statements only (hidden + visible text)
                    # --- validation strata (recall lexicon) ---
                    if a:
                        stratum = 'recall_A_matched'
                    elif b:
                        stratum = 'recall_B_only_matched'
                    elif BROAD.search(s):
                        stratum = 'recall_unmatched_broad'
                    else:
                        stratum = 'recall_unmatched_other'
                    stratum_sizes[(stratum, ds)] += 1

                    def build(s=s, offset=offset, text=text, step_no=step_no, ch=ch, sent_i=sent_i, names=names, h=h):
                        return {'source': ds, 'model': model, 'traj_id': tid, 'step': step_no, 'channel': ch,
                                'sentence_index': sent_i, 'sentence': s[:700],
                                'before': text[max(0, offset - 300):offset], 'after': text[offset + len(s):offset + len(s) + 200],
                                'recall_patterns': names, 'hedge_patterns': hedge_hits(s), 'hedged': h}
                    key = (stratum, ds) if (stratum, ds) in res else (stratum, 'all')
                    res[key].add(build)
                    # --- validation strata (hedge lexicon) ---
                    hs = 'hedge_matched' if h else 'hedge_unmatched'
                    stratum_sizes[(hs, ds)] += 1
                    res[(hs, 'all')].add(build)
                for n in step_patterns:
                    pattern_steps[(ds, model, ch, n)] += 1
                row[f'{ch}_A'] += got_a
                row[f'{ch}_B'] += got_b
                row[f'{ch}_any'] += (got_a or got_b)
                row[f'{ch}_hedge'] += got_h
            seen += '\n' + raw.get('obs', '')  # this step's tool output is readable from the next step on
        row.update({f'extra_{kk}': vv for kk, vv in extra.items()})
        traj_rows.append(row)
        if (k + 1) % 500 == 0:
            print(f'  {k + 1} trajectories ({time.time() - t0:.0f}s)', flush=True)
    return traj_rows, pattern_steps, pattern_sents, stratum_sizes, res


def rate_tables(traj_rows):
    """Per (dataset, model, channel, measure): mean per-trajectory step rate with CIs, and more."""
    groups = collections.defaultdict(list)
    for r in traj_rows:
        groups[(r['dataset'], r['model'])].append(r)
        groups[(r['dataset'], 'ALL')].append(r)
    rates, contrasts = [], []
    for (ds, model), rs in sorted(groups.items()):
        n_steps = np.array([r['n_steps'] for r in rs], dtype=float)
        n_steps[n_steps == 0] = np.nan
        for measure in ('A', 'B', 'any', 'hedge'):
            per_ch = {}
            for ch in CHANNELS:
                hits = np.array([r[f'{ch}_{measure}'] for r in rs], dtype=float)
                nonempty = np.array([r[f'{ch}_nonempty'] for r in rs], dtype=float)
                ok = ~np.isnan(n_steps)
                rate = hits[ok] / n_steps[ok]  # share of ALL agent steps whose channel has a hit
                per_ch[ch] = (rate, hits[ok] > 0)
                m, lo, hi = boot_mean(rate)
                pm, plo, phi = boot_ratio(hits[ok], nonempty[ok])  # share of steps WITH text in this channel
                sm, slo, shi = boot_mean((hits[ok] > 0).astype(float))
                rates.append({
                    'dataset': ds, 'model': model, 'measure': measure, 'channel': channel_name(ch),
                    'n_traj': int(ok.sum()), 'n_steps': int(np.nansum(n_steps)),
                    'n_steps_with_text': int(nonempty.sum()), 'n_steps_with_hit': int(hits.sum()),
                    'mean_traj_rate': r4(m), 'ci_lo': r4(lo), 'ci_hi': r4(hi),
                    'rate_among_steps_with_text': r4(pm), 'rate_text_ci_lo': r4(plo), 'rate_text_ci_hi': r4(phi),
                    'share_traj_with_hit': r4(sm), 'share_ci_lo': r4(slo), 'share_ci_hi': r4(shi)})
            if measure == 'hedge':
                continue
            for c1, c2 in (('hidden', 'visible'), ('hidden', 'command'), ('visible', 'command')):
                d = per_ch[c1][0] - per_ch[c2][0]
                m, lo, hi = boot_mean(d)
                h1 = per_ch[c1][1]
                both = per_ch[c2][1][h1]  # among trajectories with a hit in c1, share with a hit in c2 too
                contrasts.append({
                    'dataset': ds, 'model': model, 'measure': measure,
                    'contrast': f'{channel_name(c1)} minus {channel_name(c2)}', 'n_traj': len(d),
                    'mean_diff_traj_rate': r4(m), 'ci_lo': r4(lo), 'ci_hi': r4(hi),
                    f'n_traj_with_hit_in_{c1}': int(h1.sum()),
                    'share_of_those_with_hit_in_other_channel': r4(float(both.mean())) if h1.sum() else None})
    return rates, contrasts


def channel_name(ch):
    return {'hidden': 'hidden reasoning', 'visible': 'visible message', 'command': 'command'}[ch]


def model_differences(traj_rows):
    """Pairwise model differences of the hidden-channel family-A rate, within a dataset."""
    out = []
    by = collections.defaultdict(list)
    for r in traj_rows:
        if r['n_steps']:
            by[(r['dataset'], r['model'])].append(r['hidden_A'] / r['n_steps'])
    for ds in ('swebv', 'faultyclaims'):
        models = sorted(m for d, m in by if d == ds)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m, lo, hi = boot_diff_independent(by[(ds, models[i])], by[(ds, models[j])])
                out.append({'dataset': ds, 'model_a': models[i], 'model_b': models[j], 'measure': 'A',
                            'channel': 'hidden reasoning', 'diff_a_minus_b': r4(m), 'ci_lo': r4(lo), 'ci_hi': r4(hi)})
    return out


# ---------------------------------------------------------------------------------------------
# 7. Part 2: calibration. Facts named before they were seen: hedged/flagged or asserted plainly?
# ---------------------------------------------------------------------------------------------

def label_sentence(s):
    """Disclosure labels for one sentence (rule 2, the seen-version check, is not applied here)."""
    a, b, _ = recall_families(s)
    h = bool(H_RE.search(s))
    return {'recall_A': a, 'version_B': b, 'hedged': h, 'flagged_any': a or b or h, 'plain': not (a or b or h)}


def enclosing_sentence(text, start, end):
    """The sentence(s) of text that overlap the span [start, end)."""
    parts = [(a, b) for a, b in sentence_spans(text) if a < end and b > start]
    if not parts:
        return text[start:end]
    return text[parts[0][0]:parts[-1][1]].strip()


def inside_fence(text, pos):
    """True if character position pos is inside a ``` code block."""
    return any(m.start() <= pos < m.end() for m in FENCE.finditer(text))


PATH_MODELS = {'deepseek-v3.2': ('20251201_mini-v1.17.1_deepseek-v3.2-reasoner', 'DeepSeek-V3.2'),
               'kimi-k2.5': ('20260217_mini-v2.0.0_kimi-k2-5-high', 'Kimi K2.5'),
               'minimax-m2.5': ('20260217_mini-v2.0.0_minimax-2-5-high', 'MiniMax M2.5')}


def calibration_paths():
    """Pilot file paths named before seen (prov == unseen, not inside a typed fake output).
    The sentence is re-extracted from the raw trajectory (the pilot's saved context can be cut inside a path)."""
    rows = []
    for short, (run, model) in PATH_MODELS.items():
        recs = [json.loads(l) for l in open(os.path.join(ROOT, 'outputs', short + '.jsonl'))]
        recs = [r for r in recs if r['prov'] == 'unseen' and not r.get('in_fake_output')]
        cache = {}
        for r in recs:
            status = 'correct (exists at base commit)' if r['status'] == 'real' else \
                'wrong: other version (stale/future)' if r['status'] in ('stale', 'future') else 'wrong: never existed'
            out = {'model': model, 'instance_id': r['instance_id'], 'step': r['step'], 'path': r['path'],
                   'status': r['status'], 'status_group': status,
                   'channel': {'reasoning': 'hidden reasoning', 'thought': 'visible message', 'command': 'command'}[r['field']]}
            if r['field'] == 'command':
                out.update({'sentence_kind': 'command only (no sentence)', 'recall_A': False, 'version_B': False,
                            'hedged': False, 'flagged_any': False, 'plain': False})
                rows.append(out)
                continue
            iid = r['instance_id']
            if iid not in cache:
                f = os.path.join(ROOT, 'data', 'raw', run, 'trajs', iid, iid + '.traj.json')
                cache = {iid: swebv_steps(json.load(open(f)))}  # one trajectory at a time
            st = cache[iid][r['step'] - 1]
            text = st['hidden'] if r['field'] == 'reasoning' else st['visible']
            m = re.search(r'(?<![\w/.\-])(?:/testbed/|\./)?' + re.escape(r['path']) + r'(?![\w/])', text)
            if not m:
                sent, kind = r['context'], 'sentence (pilot context, path not re-found)'
            else:
                sent = enclosing_sentence(text, m.start(), m.end())
                kind = 'inside a code block' if inside_fence(text, m.start()) else 'sentence'
            out.update({'sentence_kind': kind, **label_sentence(sent)})
            rows.append(out)
    return rows


def summarize_calibration(rows, group_keys, unit_key):
    """Shares of flagged/hedged/plain sentences per group, CIs by resampling unit_key (trajectories)."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in group_keys)].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        units = collections.defaultdict(list)
        for r in rs:
            units[r[unit_key]].append(r)
        rec = dict(zip(group_keys, key))
        rec['n'] = len(rs)
        rec['n_traj'] = len(units)
        den = [len(v) for v in units.values()]
        for lab in ('recall_A', 'version_B', 'hedged', 'flagged_any', 'plain'):
            num = [sum(x[lab] for x in v) for v in units.values()]
            m, lo, hi = boot_ratio(num, den)
            rec[f'share_{lab}'] = r4(m)
            rec[f'{lab}_ci_lo'] = r4(lo)
            rec[f'{lab}_ci_hi'] = r4(hi)
        out.append(rec)
    return out


def calibration_claims():
    """You Wang et al.'s claims: sentence around the CoT quote, by verdict and by seen/unseen names."""
    seen = {}
    with open(os.path.join(ROOT, 'data/faultyclaims/claims_seen_before.jsonl')) as fh:
        for line in fh:
            r = json.loads(line)
            seen[r['unit_id']] = r
    rows, cache_path, cache_msgs = [], None, None
    stats = collections.Counter()
    with open(os.path.join(ROOT, 'data/faultyclaims/claims_tidy.jsonl')) as fh:
        for line in fh:
            c = json.loads(line)
            if c['trajectory_path'] != cache_path:
                cache_path = c['trajectory_path']
                cache_msgs = json.load(open(os.path.join(ROOT, cache_path)))['messages']
            msg = cache_msgs[c['message_index']]
            rc = msg.get('reasoning_content') or ''
            q, qs, qe = c['quote'], c['quote_start'], c['quote_end']
            if qs is not None and rc[qs:qe] == q:
                sent, how = enclosing_sentence(rc, qs, qe), 'offsets'
            elif q and q in rc:
                i = rc.find(q)
                sent, how = enclosing_sentence(rc, i, i + len(q)), 'search in reasoning'
            elif q and q in as_text(msg.get('content')):
                txt = as_text(msg.get('content'))
                i = txt.find(q)
                sent, how = enclosing_sentence(txt, i, i + len(q)), 'search in visible text'
            else:
                sent, how = q, 'quote only'
            stats['located_by_' + how] += 1
            sb = seen[c['unit_id']]
            if sb['identifiers_unseen'] or sb['files_unseen']:
                names = 'named before seen'
            elif sb['identifiers'] or c['related_files']:
                names = 'all names seen before'
            else:
                names = 'no code names'
            lab = label_sentence(sent)
            qlab = label_sentence(q)
            rows.append({'model': c['model_name'], 'traj': c['trajectory_path'], 'verdict': c['verdict'],
                         'names': names, 'manual_confirmed_faulty': bool(c['manual_confirmed']),
                         'error_form': c['error_form'], **lab,
                         **{'quote_' + k: v for k, v in qlab.items()}})
    return rows, stats


# ---------------------------------------------------------------------------------------------
# 8. Part 4: our trap runs (pilot + the other session's targeted runs, read only)
# ---------------------------------------------------------------------------------------------

def trap_run_files():
    sets = [('pilot django-11149 (network on)', os.path.join(ROOT, 'outputs/agent_runs')),
            ('pilot sympy-13615 (network on)', os.path.join(ROOT, 'outputs/runs/sympy__sympy-13615')),
            ('pilot sympy-13615 (network off)', os.path.join(ROOT, 'outputs/runs/sympy__sympy-13615-nonet'))]
    for d in sorted(glob.glob(os.path.join(TARGETED_RUNS, '*'))):
        if os.path.isdir(d):
            sets.append((f'targeted {os.path.basename(d)} (network off)', d))
    for name, d in sets:
        for f in sorted(glob.glob(os.path.join(d, '*.traj.json'))):
            yield name, f


def analyze_trap_runs():
    rows, skipped = [], []
    stats = collections.Counter()
    for set_name, f in trap_run_files():
        try:
            traj = json.load(open(f))
        except Exception as e:  # a run still being written by the other session
            skipped.append(f'{f}: {e}')
            continue
        raw = chat_steps(traj)
        run = os.path.basename(f)[:-len('.traj.json')]
        r = {'run_set': set_name, 'run': run, 'condition': run.split('-')[0], 'steps': len(raw),
             'exit_status': (traj.get('info') or {}).get('exit_status')}
        for ch in ('hidden', 'visible'):
            r[f'{ch}_steps_with_text'] = sum(bool(s[ch].strip()) for s in raw)
            # the pilot's own regex, on the raw text, exactly as scripts/trap_pipeline.py counted
            r[f'pilot_regex_{ch}'] = sum(bool(PILOT_MEM.search(s[ch] or '')) for s in raw)
        for ch in ('hidden', 'visible'):
            for key in ('A', 'B', 'any', 'hedge'):
                r[f'{key}_{ch}'] = 0
        seen = task_text(traj['messages'])
        for s in raw:
            st = clean_channels(s, stats)
            for ch in ('hidden', 'visible'):
                a = b = h = False
                for sent, _ in sentences(st[ch]):  # same sentence-level matching as part 1
                    sa, sb, _ = recall_families(sent, seen)
                    a, b, h = a or sa, b or sb, h or bool(H_RE.search(sent))
                r[f'A_{ch}'] += a
                r[f'B_{ch}'] += b
                r[f'any_{ch}'] += (a or b)
                r[f'hedge_{ch}'] += h
            seen += '\n' + s['obs']
        rows.append(r)
    return rows, skipped


# ---------------------------------------------------------------------------------------------
# 9. Validation: write the sample, read the LLM-judged labels if they exist
# ---------------------------------------------------------------------------------------------

LABEL_FIELDS = ['sample_id', 'lexicon', 'stratum', 'source', 'model', 'traj_id', 'step', 'channel',
                'sentence_index', 'recall_patterns', 'hedge_patterns', 'label', 'note']


def write_validation_sample(res):
    os.makedirs(VAL_DIR, exist_ok=True)
    sample = []
    for (stratum, src), r in res.items():
        for item in r.items:
            sample.append(dict(item, stratum=stratum, lexicon=STRATA[stratum][0]))
    # stable order and ids
    sample.sort(key=lambda x: (list(STRATA).index(x['stratum']), x['source'], x['model'], x['traj_id'],
                               x['step'], x['channel'], x['sentence_index']))
    for i, x in enumerate(sample):
        x['sample_id'] = f'v{i:03d}'
    with open(os.path.join(VAL_DIR, 'sample.jsonl'), 'w') as fh:
        for x in sample:
            fh.write(json.dumps(x, ensure_ascii=False) + '\n')
    return sample


def read_or_init_labels(sample):
    """Labels live in outputs/rq3/fig_lexicon_validation.csv (ids only, no text). Never overwrite labels."""
    path = os.path.join(OUT, 'fig_lexicon_validation.csv')
    ids = [x['sample_id'] for x in sample]
    if os.path.exists(path):
        with open(path) as fh:
            labels = list(csv.DictReader(fh))
        filled = any(l['label'] for l in labels)
        same = [l['sample_id'] for l in labels] == ids and all(
            l['traj_id'] == x['traj_id'] and str(l['step']) == str(x['step']) and l['channel'] == x['channel']
            and str(l['sentence_index']) == str(x['sentence_index']) for l, x in zip(labels, sample))
        if filled and not same:
            sys.exit('fig_lexicon_validation.csv has labels but does not match the regenerated sample; '
                     'not touching it.')
        if filled:
            return labels
    rows = [{'sample_id': x['sample_id'], 'lexicon': x['lexicon'], 'stratum': x['stratum'], 'source': x['source'],
             'model': x['model'], 'traj_id': x['traj_id'], 'step': x['step'], 'channel': x['channel'],
             'sentence_index': x['sentence_index'], 'recall_patterns': ';'.join(x['recall_patterns']),
             'hedge_patterns': ';'.join(x['hedge_patterns']), 'label': '', 'note': ''} for x in sample]
    write_csv('fig_lexicon_validation.csv', rows, LABEL_FIELDS)
    return rows


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - h) / d, (c + h) / d


RECALL_LABELS = {  # label -> meaning (LLM-judged; see notes/rq3-disclosure.md)
    'mem_repo': 'states or uses remembered knowledge about the task\'s own project (code, behavior, history, '
                'its real fix/PR/commit), incl. "I can\'t remember the upstream code"',
    'mem_general': 'states remembered knowledge about other software, languages or math (e.g. Python version facts)',
    'upstream_ref': 'refers to the real upstream fix/PR/commit/version as something to match, find or guess, '
                    'without claiming to remember its content',
    'read_upstream': 'refers to upstream content the agent actually obtained in this run (network, installed '
                     'newer package): read, not remembered',
    'no': 'not a memory statement (memory of this run, its own fix, code/quoted text, version facts read from '
          'the task or a tool output)',
    'unclear': 'cannot tell from the snippet',
}
STRICT = {'mem_repo', 'mem_general'}
LENIENT = STRICT | {'upstream_ref'}


def score_validation(labels, stratum_sizes):
    """Precision of each lexicon part, and an estimated recall, from the LLM-judged sample.
    Recall lexicon: strict positive = mem_repo/mem_general; lenient positive = + upstream_ref.
    Hedge lexicon: positive = hedge (an epistemic hedge or tentative claim/plan), else no/unclear."""
    if not any(l['label'] for l in labels):
        return {'status': 'no labels yet'}
    by = collections.defaultdict(list)
    for l in labels:
        # strata sampled per source are scored per source; strata sampled over all sources are scored pooled
        src = l['source'] if l['source'] in STRATA[l['stratum']][1] else 'all'
        by[(l['stratum'], src)].append(l['label'])
    result = {'status': 'LLM-judged (Claude read every snippet; not human-labeled)', 'recall_labels': RECALL_LABELS,
              'strata': {}}
    for (stratum, src), labs in sorted(by.items()):
        lexicon = STRATA[stratum][0]
        rec = {'n': len(labs), 'labels': dict(collections.Counter(labs))}
        defs = {'strict': STRICT, 'lenient': LENIENT} if lexicon == 'recall' else {'hedge': {'hedge'}}
        for name, pos in defs.items():
            k = sum(x in pos for x in labs)
            lo, hi = wilson(k, len(labs))
            rec[f'share_{name}'] = r4(k / len(labs))
            rec[f'{name}_ci_lo'] = r4(lo)
            rec[f'{name}_ci_hi'] = r4(hi)
        if lexicon == 'recall':
            rec['share_mem_repo'] = r4(sum(x == 'mem_repo' for x in labs) / len(labs))
        rec['pool_size'] = sum(v for (s, d), v in stratum_sizes.items() if s == stratum and (src == 'all' or d == src))
        result['strata'][f'{stratum}|{src}'] = rec

    def weighted(strata_names, field):
        """Pool-size-weighted estimate over strata: sum of (sample share x pool size), and total pool size."""
        tot = est = 0.0
        for key, rec in result['strata'].items():
            if key.split('|')[0] in strata_names:
                tot += rec['pool_size']
                est += rec[field] * rec['pool_size']
        return est, tot

    summ = {}
    for name in ('strict', 'lenient'):
        a_tp, a_n = weighted(['recall_A_matched'], f'share_{name}')
        b_tp, b_n = weighted(['recall_B_only_matched'], f'share_{name}')
        fn, _ = weighted(['recall_unmatched_broad', 'recall_unmatched_other'], f'share_{name}')
        summ[f'precision_A_{name}'] = r4(a_tp / a_n)
        summ[f'precision_B_only_{name}'] = r4(b_tp / b_n)
        summ[f'precision_A_or_B_{name}'] = r4((a_tp + b_tp) / (a_n + b_n))
        summ[f'estimated_true_statements_matched_{name}'] = round(a_tp + b_tp)
        summ[f'estimated_true_statements_missed_{name}'] = round(fn)
        summ[f'estimated_recall_A_or_B_{name}'] = r4((a_tp + b_tp) / (a_tp + b_tp + fn)) if (a_tp + b_tp + fn) else None
        # pessimistic version: each unmatched stratum at the upper end of its 95% Wilson interval
        fn_hi, _ = weighted(['recall_unmatched_broad', 'recall_unmatched_other'], f'{name}_ci_hi')
        summ[f'recall_A_or_B_{name}_if_misses_at_upper_95'] = r4((a_tp + b_tp) / (a_tp + b_tp + fn_hi))
    a_repo, a_n = weighted(['recall_A_matched'], 'share_mem_repo')
    summ['precision_A_mem_repo_only'] = r4(a_repo / a_n)
    h_tp, h_n = weighted(['hedge_matched'], 'share_hedge')
    h_fn, _ = weighted(['hedge_unmatched'], 'share_hedge')
    summ['precision_hedge'] = r4(h_tp / h_n)
    summ['estimated_recall_hedge'] = r4(h_tp / (h_tp + h_fn)) if (h_tp + h_fn) else None
    summ['note'] = ('Weighted by each stratum\'s pool size (unique hidden/visible sentences in swebv, faultyclaims, '
                    'deepswe). Recall is an estimate: missed statements = share judged positive in each unmatched '
                    'stratum x its pool size. The "other" unmatched stratum is huge, so even 0 positives in 50 leaves '
                    'a wide upper bound (Wilson upper bound ~7% of that stratum): recall could be much lower.')
    result['summary'] = summ
    return result


# ---------------------------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    started = time.strftime('%Y-%m-%d %H:%M:%S')
    stats = collections.Counter()

    print('Part 1: recall statements per channel ...', flush=True)
    traj_rows, pattern_steps, pattern_sents, stratum_sizes, res = analyze_channels(stats)
    rates, contrasts = rate_tables(traj_rows)
    model_diffs = model_differences(traj_rows)

    traj_fields = list(dict.fromkeys(k for r in traj_rows for k in r))
    write_csv('fig_traj_recall_counts.csv', traj_rows, traj_fields)
    write_csv('fig_recall_rates.csv', rates)
    write_csv('fig_channel_contrasts.csv', contrasts, list(dict.fromkeys(k for r in contrasts for k in r)))
    write_csv('fig_model_differences.csv', model_diffs)
    prow = [{'dataset': ds, 'model': m, 'channel': channel_name(ch), 'pattern': p, 'family': FAMILY[p],
             'steps_with_hit': v, 'unique_sentences_with_hit': pattern_sents[(ds, m, ch, p)]}
            for (ds, m, ch, p), v in sorted(pattern_steps.items())]
    write_csv('fig_pattern_counts.csv', prow)

    print('Part 2: calibration ...', flush=True)
    path_rows = calibration_paths()
    path_tab = summarize_calibration(path_rows, ['model', 'channel', 'status_group', 'sentence_kind'], 'instance_id')
    path_tab_nl = summarize_calibration([r for r in path_rows if r['sentence_kind'] == 'sentence'],
                                        ['model', 'status_group'], 'instance_id')
    for r in path_tab_nl:
        r['channel'] = 'hidden or visible text, plain sentences only'
        r['sentence_kind'] = 'sentence'
    write_csv('fig_calibration_paths.csv', path_tab + path_tab_nl,
              list(dict.fromkeys(k for r in path_tab + path_tab_nl for k in r)))
    claim_rows, claim_stats = calibration_claims()
    for r in claim_rows:
        r['model_all'] = 'ALL'
    claim_tab = summarize_calibration(claim_rows, ['model', 'names', 'verdict'], 'traj') + \
        summarize_calibration(claim_rows, ['model_all', 'names', 'verdict'], 'traj')
    # quote-only robustness check
    qrows = [dict(model_all='ALL', names=r['names'], verdict=r['verdict'], traj=r['traj'],
                  **{k: r['quote_' + k] for k in ('recall_A', 'version_B', 'hedged', 'flagged_any', 'plain')})
             for r in claim_rows]
    claim_tab_q = summarize_calibration(qrows, ['model_all', 'names', 'verdict'], 'traj')
    for r in claim_tab_q:
        r['text_used'] = 'quote only'
    for r in claim_tab:
        r['text_used'] = 'enclosing sentence'
        if 'model_all' in r:
            r['model'] = r.pop('model_all')
    for r in claim_tab_q:
        r['model'] = r.pop('model_all')
    manual = summarize_calibration([dict(r, model_all='ALL') for r in claim_rows if r['manual_confirmed_faulty']],
                                   ['model_all', 'names'], 'traj')
    for r in manual:
        r['model'] = r.pop('model_all')
        r['verdict'] = 'refuted + manually confirmed (40 reviewed trajectories)'
        r['text_used'] = 'enclosing sentence'
    allc = claim_tab + claim_tab_q + manual
    write_csv('fig_calibration_claims.csv', allc, list(dict.fromkeys(k for r in allc for k in r)))

    # wrong vs right: is a wrong fact hedged/flagged more often than a right one? (same-trajectory bootstrap)
    contrasts_cal = []
    for names in ('named before seen', 'all names seen before'):
        for lab in ('hedged', 'recall_A', 'flagged_any'):
            rec = boot_group_diff(claim_rows, 'traj',
                                  lambda r, names=names: r['names'] == names and r['verdict'] == 'refuted',
                                  lambda r, names=names: r['names'] == names and r['verdict'] == 'supported', lab)
            contrasts_cal.append({'data': 'You Wang et al. claims (all 4 models)', 'subset': names,
                                  'group1': 'refuted (wrong)', 'group2': 'supported (right)', 'label': lab, **rec})
    ds_sent = [r for r in path_rows if r['model'] == 'DeepSeek-V3.2' and r['sentence_kind'] == 'sentence']
    for wrong in ('wrong: other version (stale/future)', 'wrong: never existed'):
        for lab in ('hedged', 'flagged_any'):
            rec = boot_group_diff(ds_sent, 'instance_id', lambda r, wrong=wrong: r['status_group'] == wrong,
                                  lambda r: r['status_group'] == 'correct (exists at base commit)', lab)
            contrasts_cal.append({'data': 'pilot file paths named before seen, DeepSeek-V3.2 sentences',
                                  'subset': 'hidden or visible sentence', 'group1': wrong,
                                  'group2': 'correct (exists at base commit)', 'label': lab, **rec})
    write_csv('fig_calibration_contrasts.csv', contrasts_cal)

    print('Part 4: trap runs ...', flush=True)
    trap_rows, trap_skipped = analyze_trap_runs()
    write_csv('fig_trap_runs.csv', trap_rows)

    print('Part 3: validation sample ...', flush=True)
    sample = write_validation_sample(res)
    labels = read_or_init_labels(sample)
    validation = score_validation(labels, stratum_sizes)

    # --------- summary ---------
    def find(table, **kw):
        for r in table:
            if all(r.get(k) == v for k, v in kw.items()):
                return r
        return None

    key = {}
    for ds in ('swebv', 'faultyclaims', 'deepswe'):
        for model in sorted({r['model'] for r in traj_rows if r['dataset'] == ds}) + ['ALL']:
            for ch in CHANNELS:
                r = find(rates, dataset=ds, model=model, measure='A', channel=channel_name(ch))
                if r:
                    key[f'{ds}|{model}|A|{ch}'] = {k: r[k] for k in ('n_traj', 'mean_traj_rate', 'ci_lo', 'ci_hi',
                                                                    'share_traj_with_hit', 'share_ci_lo', 'share_ci_hi',
                                                                    'n_steps_with_text', 'n_steps_with_hit')}
    trap_summary = collections.defaultdict(dict)
    for set_name in sorted({r['run_set'] for r in trap_rows}):
        rs = [r for r in trap_rows if r['run_set'] == set_name]
        for col in ('pilot_regex_hidden', 'pilot_regex_visible', 'A_hidden', 'A_visible', 'any_hidden', 'any_visible'):
            vals = [r[col] for r in rs]
            trap_summary[set_name][col] = {'min': min(vals), 'max': max(vals), 'mean': round(sum(vals) / len(vals), 2)}
        trap_summary[set_name]['n_runs'] = len(rs)
    for cond in ('original', 'placebo', 'trap'):
        rs = [r for r in trap_rows if r['condition'] == cond]
        if rs:
            trap_summary[f'condition {cond}'] = {
                'n_runs': len(rs), 'mean_steps': round(sum(r['steps'] for r in rs) / len(rs), 2),
                'mean_A_hidden': round(sum(r['A_hidden'] for r in rs) / len(rs), 2),
                'mean_A_visible': round(sum(r['A_visible'] for r in rs) / len(rs), 2),
                'mean_share_of_steps_A_hidden': round(sum(r['A_hidden'] / r['steps'] for r in rs) / len(rs), 4)}
    all_runs = trap_rows
    trap_summary['ALL runs'] = {
        'n_runs': len(all_runs),
        'runs_with_more_A_steps_hidden_than_visible': sum(r['A_hidden'] > r['A_visible'] for r in all_runs),
        'runs_with_any_A_visible': sum(r['A_visible'] > 0 for r in all_runs),
        'total_A_steps_hidden': sum(r['A_hidden'] for r in all_runs),
        'total_A_steps_visible': sum(r['A_visible'] for r in all_runs),
        'total_any_steps_hidden': sum(r['any_hidden'] for r in all_runs),
        'total_any_steps_visible': sum(r['any_visible'] for r in all_runs)}

    targeted_files = sorted(glob.glob(os.path.join(TARGETED_RUNS, '*', '*.traj.json')))
    summary = {
        'what': 'RQ3 disclosure: does the reasoning reveal reliance on memory? (scripts/rq3_disclosure.py)',
        'run_started': started, 'run_finished': time.strftime('%Y-%m-%d %H:%M:%S'),
        'command': '.venv/bin/python -I scripts/rq3_disclosure.py',
        'bootstrap': f'{N_BOOT} resamples of trajectories, seed {SEED}, 95% percentile intervals',
        'lexicons': {
            'recall': [{'name': n, 'family': f, 'regex': rx, 'catches': d} for n, f, rx, d in RECALL_PATTERNS],
            'hedge': [{'name': n, 'regex': rx, 'catches': d} for n, rx, d in HEDGE_PATTERNS],
            'family_A': 'explicit memory statements + references to the real upstream project (main measure)',
            'family_B': 'version-history talk (weaker evidence of memory)',
            'matching': 'per sentence; hidden/visible text with ``` code blocks and typed fake tool outputs removed; '
                        'a step counts if any sentence in that channel matches'},
        'data': {
            'n_trajectories': dict(collections.Counter(r['dataset'] for r in traj_rows)),
            'n_trajectories_by_model': {f"{r[0]}|{r[1]}": v for r, v in sorted(
                collections.Counter((r['dataset'], r['model']) for r in traj_rows).items())},
            'n_steps': dict(collections.Counter({ds: sum(r['n_steps'] for r in traj_rows if r['dataset'] == ds)
                                                 for ds in ('swebv', 'faultyclaims', 'deepswe')})),
            'cleaning': dict(stats),
            'targeted_runs_snapshot': {'n_files': len(targeted_files),
                                       'latest_mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(
                                           max(os.path.getmtime(f) for f in targeted_files))) if targeted_files else None},
            'trap_runs_skipped_unreadable': trap_skipped},
        'part1_recall_rates_family_A': key,
        'part1_model_differences_hidden_A': model_diffs,
        'part1_contrasts_hidden_vs_visible_A': [r for r in contrasts if r['measure'] == 'A'
                                                and r['contrast'] == 'hidden reasoning minus visible message'],
        'part2_calibration_paths_sentences': path_tab_nl,
        'part2_calibration_paths_counts': dict(collections.Counter(
            (r['model'] + '|' + r['channel'] + '|' + r['status_group']) for r in path_rows)),
        'part2_calibration_claims_ALL': [r for r in claim_tab if r['model'] == 'ALL'],
        'part2_calibration_claims_manual': manual,
        'part2_calibration_wrong_vs_right': contrasts_cal,
        'part2_claim_sentence_location': dict(claim_stats),
        'part3_validation': validation,
        'part4_trap_runs': trap_summary,
    }
    with open(os.path.join(OUT, 'disclosure_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print('done ->', OUT)


if __name__ == '__main__':
    main()
