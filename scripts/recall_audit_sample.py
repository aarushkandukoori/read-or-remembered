"""RecallAudit sampling: pick the agent-text snippets to label for a small recall classifier.

RecallAudit is a small open classifier meant to flag EXPLICIT memory recall in coding-agent text
("I recall that Django's ...", "the upstream fix did ..."). RQ3's regex lexicons
(scripts/rq3_disclosure.py) catch such statements with only ~50% strict precision
(notes/rq3-disclosure.md), so we need labeled examples to train and test something better.
This script only SAMPLES the items. It makes no labels and no API calls.

What is an item?  The text of ONE channel of ONE agent step: the hidden reasoning or the visible
message (two separate items). Code blocks and DeepSeek's typed fake tool outputs are removed exactly
as in rq3_disclosure.py (clean_channels). A text longer than 1,200 characters is cut to a window of
about 1,200 characters around its most relevant sentence (see relevant_sentence and cut_window).

Lexicon label of a text (rq3_disclosure.py, sentence by sentence, with its rules 1 and 2):
  'A'    some sentence matches family A (explicit memory / real upstream fix)
  'B'    no family A, but some sentence matches family B (version-history talk)
  'none' no match ("non-matching")

Two pools:
  training pool (may be used for training): 1,200 items
      SWE-bench Verified runs in data/raw: DeepSeek-V3.2, Kimi K2.5, MiniMax M2.5
      You Wang et al.'s 400 trajectories (CC-BY-4.0): DeepSeek-V3.2, DeepSeek-V4.1-Flash, Qwen3.6-27B, Qwen3.8-27B
      -> 7 (dataset, model) runs; 45% lexicon-matched ('A' or 'B'), 55% random non-matching, each
         group shared evenly over the 7 runs as far as the data allows;
         split by TASK (instance_id): a seeded 20% of each benchmark's tasks are 'test', the rest
         'train', and 20% of each (run, group) quota is drawn from test tasks
  OOD pool (evaluation only, NEVER for training): 300 items, split 'ood'
      DeepSWE hidden reasoning from open-weight models whose reasoning text is complete (not a
      summary); same 45/55 stratification, balanced over models

Writes (DeepSWE text stays under data/; it must never go to outputs/, notes/ or docs/):
  data/recall_audit/batches/b_01.jsonl ... b_15.jsonl        100 items each, with all fields
  data/recall_audit/batches_blind/b_01.jsonl ... b_15.jsonl  the same items with only item_id and text
and prints every count (pool sizes, quotas, final composition).
Fields: item_id (ra0001...; carries no information), text, split (train/test/ood), source
(swebv/faultyclaims/deepswe), model, instance_id (the task; DeepSWE: task_name), step (1 = first agent
step), channel (hidden/visible), lexicon_match (A/B/none), plus lexicon_patterns (rq3 pattern names
found in the item text), traj_id (trajectory; DeepSWE: trial_name) and cut (text cut to a window).
Every batch has the same mix: 80 training-pool items and 20 OOD items.

Usage (from the project root; all data is read-only):
  .venv/bin/python -I scripts/recall_audit_sample.py                      # the real run
  .venv/bin/python -I scripts/recall_audit_sample.py --check-loader       # test the raw_output-skipping loader
  .venv/bin/python -I scripts/recall_audit_sample.py --limit 20 --out DIR # quick test on 20 trajectories per run
Run that made the current batches: 2026-11-13 11:11:55 to 11:15:18 (203 s, 247 MB peak memory);
a second run into a scratch folder gave byte-identical batch files.
"""

import argparse
import collections
import csv
import glob
import json
import os
import random
import re
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))  # with python -I the script folder is not on the path
import rq3_disclosure as rq3  # noqa: E402  (lexicons, trajectory loaders and text cleaning, reused unchanged)

SEED = 0
MAX_CHARS = 1200        # item window: about this many characters
MIN_CHARS = 20          # shorter items carry no signal (e.g. a lone "THOUGHT:")
N_TRAINING_POOL = 1200
N_OOD = 300
MATCHED_SHARE = 0.45    # share of items with lexicon label 'A' or 'B'
TEST_SHARE = 0.20       # share of tasks (per benchmark), and of each quota, that go to the test split
BATCH_SIZE = 100

# DeepSWE "open-weight": same vendor-family rule as scripts/rq1_fact_provenance.py (OPEN_FAMILIES):
# families that published weights for earlier versions; not verified per release. Qwen Plus/Max are
# hosted-only products, so they do not count.
OPEN_FAMILIES = ('deepseek', 'kimi', 'minimax', 'glm', 'mimo')
# "Full reasoning": the reasoning text has at least this many characters per reported reasoning token.
# Complete reasoning text is ~4 characters per token; summarized reasoning (GPT, grok, Gemini) is
# 0.4 to 1.2 (notes/rq3-disclosure.md, caveats).
MIN_CHARS_PER_REASONING_TOKEN = 3.0


# ---------------------------------------------------------------------------------------------
# 1. Reading trajectory files
# ---------------------------------------------------------------------------------------------

def load_json(path):
    with open(path) as fh:
        return json.load(fh)


RAW_KEY = '"raw_output": "'


def load_json_skip_raw_output(path, piece_chars=1 << 20):
    """json.load for You Wang et al.'s trajectory files, with every "raw_output" string emptied.

    Each tool message in these files keeps a second, untruncated copy of the command output under
    extra.raw_output (one file holds a single 134 MB copy). We never use it: the agent saw the shorter
    'content' field, which is what rq3_disclosure.py reads. A plain json.load of that file peaked at
    1.19 GB of resident memory, so we read it in pieces of 1M characters and copy everything except
    the inside of raw_output strings. A JSON string ends at the first quote that is not escaped (not preceded by an
    odd number of backslashes). Inside a JSON string every quote is escaped, so RAW_KEY can only match
    a real key. Checked with --check-loader: same steps and task text as json.load."""
    kept = []          # pieces of the file we keep
    skipping = False   # True while we are inside a raw_output string
    tail = ''          # end of the previous piece, looked at again together with the next piece
    with open(path, encoding='utf-8') as fh:
        while True:
            piece = fh.read(piece_chars)
            text = tail + piece
            tail = ''
            if not text:
                break
            i = 0
            while i < len(text):
                if not skipping:
                    j = text.find(RAW_KEY, i)
                    if j < 0:
                        # the key could start in the last few characters: look at them again next time
                        cut = max(i, len(text) - len(RAW_KEY) + 1) if piece else len(text)
                        kept.append(text[i:cut])
                        tail = text[cut:]
                        break
                    kept.append(text[i:j + len(RAW_KEY)])  # keep the key and the opening quote
                    i = j + len(RAW_KEY)
                    skipping = True
                else:
                    j = text.find('"', i)
                    if j < 0:
                        # the string goes on in the next piece; carry trailing backslashes (a split escape)
                        n_bs = 0
                        while n_bs < len(text) - i and text[-1 - n_bs] == '\\':
                            n_bs += 1
                        tail = text[len(text) - n_bs:]
                        break
                    k = j
                    while k > i and text[k - 1] == '\\':  # count the backslashes right before the quote
                        k -= 1
                    if (j - k) % 2 == 0:  # not escaped: the raw_output string ends here
                        kept.append('"')
                        skipping = False
                    i = j + 1
            if not piece:
                break
    return json.loads(''.join(kept))


def read_trajectory(source, path):
    """(agent steps, task text) of one trajectory, with rq3_disclosure.py's loaders."""
    if source == 'swebv':
        traj = load_json(path)
        return rq3.swebv_steps(traj), rq3.task_text(traj['messages'])
    if source == 'faultyclaims':
        traj = load_json_skip_raw_output(path)
        return rq3.chat_steps(traj), rq3.task_text(traj['messages'])
    traj = load_json(path)  # deepswe
    return rq3.deepswe_steps(traj), rq3.deepswe_task_text(traj)


def training_trajectories(limit=None):
    """The 7 training-pool runs: list of dicts (source, model, instance_id, traj_id, path)."""
    out = []
    for run, model in rq3.SWEBV_RUNS.items():
        for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'raw', run, 'trajs', '*', '*.traj.json')))[:limit]:
            iid = os.path.basename(f)[:-len('.traj.json')]
            out.append({'source': 'swebv', 'model': model, 'instance_id': iid, 'traj_id': iid, 'path': f})
    fc_dir = os.path.join(ROOT, 'data', 'faultyclaims', 'reasoning_replication', 'data')
    for folder, model in rq3.FC_MODELS.items():
        for f in sorted(glob.glob(os.path.join(fc_dir, folder, '*.json')))[:limit]:
            iid = os.path.basename(f)[:-len('.json')]
            out.append({'source': 'faultyclaims', 'model': model, 'instance_id': iid, 'traj_id': iid, 'path': f})
    return out


def ood_trajectories(limit=None):
    """DeepSWE trials of open-weight models whose reasoning text is complete.
    Returns (trials, per-model table with the characters-per-reasoning-token check)."""
    with open(os.path.join(ROOT, 'data', 'deepswe', 'sample_manifest.csv')) as fh:
        manifest = list(csv.DictReader(fh))
    table = collections.defaultdict(lambda: {'trials': [], 'reasoning_chars': 0, 'reasoning_tokens': 0})
    for m in manifest:
        if not m['model'].lower().startswith(OPEN_FAMILIES):
            continue
        p = os.path.join(ROOT, 'data', 'deepswe', 'trials', m['version'], m['trial_name'], 'trajectory.json')
        if not os.path.exists(p):
            continue
        traj = load_json(p)
        chars = sum(len(s['hidden']) for s in rq3.deepswe_steps(traj))
        tokens = ((traj.get('final_metrics') or {}).get('extra') or {}).get('total_reasoning_tokens')
        rec = table[m['model']]
        rec['trials'].append({'source': 'deepswe', 'model': m['model'], 'instance_id': m['task_name'],
                              'traj_id': m['trial_name'], 'path': p})
        if chars and tokens:  # same pooled ratio as the check in notes/rq3-disclosure.md
            rec['reasoning_chars'] += chars
            rec['reasoning_tokens'] += tokens
    trials, summary = [], {}
    for model in sorted(table):
        rec = table[model]
        ratio = rec['reasoning_chars'] / rec['reasoning_tokens'] if rec['reasoning_tokens'] else 0.0
        full = ratio >= MIN_CHARS_PER_REASONING_TOKEN
        summary[model] = {'trials': len(rec['trials']), 'chars_per_reasoning_token': round(ratio, 2),
                          'used': full}
        if full:
            trials += sorted(rec['trials'], key=lambda t: t['traj_id'])[:limit]
    return trials, summary


# ---------------------------------------------------------------------------------------------
# 2. Lexicon labels, the most relevant sentence, and the item window
# ---------------------------------------------------------------------------------------------

def sentence_labels(text, seen):
    """(start, end, family A matched, family B matched, pattern names) for each sentence of text.
    Same matching as rq3_disclosure.py part 1 (sentences shorter than 3 characters are skipped)."""
    out = []
    for a, b in rq3.sentence_spans(text):
        s = text[a:b].strip()
        if len(s) < 3:
            continue
        fa, fb, names = rq3.recall_families(s, seen)
        out.append((a, b, fa, fb, names))
    return out


def label_of(sents):
    """'A', 'B' or 'none' for a whole text, and the sorted pattern names that matched."""
    names = sorted({n for *_, ns in sents for n in ns})
    if any(fa for _, _, fa, _, _ in sents):
        return 'A', names
    if any(fb for _, _, _, fb, _ in sents):
        return 'B', names
    return 'none', names


def relevant_sentence(text, sents, label, rng_key):
    """The sentence the window is centred on, as (start, end, focus). focus = where the evidence sits
    inside the sentence (only used when the sentence alone is longer than the window).
      label A: the first family-A sentence      label B: the first family-B sentence
      none:    the first sentence with a broad memory-ish word (rq3's BROAD list: version, original,
               actual, commit, recall, ...), where missed recall statements would hide;
               if there is none, a random sentence (seeded per item)."""
    for a, b, fa, fb, names in sents:
        if (label == 'A' and fa) or (label == 'B' and fb):
            s = text[a:b]
            starts = [m.start() for n in names if (m := rq3.RECALL_RE[n].search(s))]
            return a, b, a + (min(starts) if starts else 0)
    if label == 'none':
        for a, b, *_ in sents:
            m = rq3.BROAD.search(text[a:b])
            if m:
                return a, b, a + m.start()
        a, b, *_ = random.Random(rng_key).choice(sents)
        return a, b, a
    raise ValueError('no sentence with the label')  # cannot happen: the label came from these sentences


def tidy(text):
    """Drop spaces at line ends and keep at most one empty line in a row (removed code blocks leave gaps)."""
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def cut_window(text, a, b, focus):
    """About MAX_CHARS characters of text that contain the sentence text[a:b], with the leftover room
    split evenly before and after it. If the sentence alone is longer than MAX_CHARS, the window is
    centred on `focus` instead. Cuts move to the nearest space (at most 40 characters, never into the
    sentence) so no word is split, and are marked with "[...]"."""
    n = len(text)
    if b - a >= MAX_CHARS:
        a = b = focus
    room = MAX_CHARS - (b - a)
    start = max(0, a - room // 2)
    end = min(n, start + MAX_CHARS)
    start = max(0, end - MAX_CHARS)  # near the end of the text: give the unused room to the front
    if start > 0:
        m = re.search(r'\s', text[start:min(a, start + 40)])
        if m:
            start += m.end()
    if end < n:
        k = max(text.rfind(c, max(b, end - 40), end) for c in ' \n\t')
        if k > 0:
            end = k
    piece = tidy(text[start:end])
    if text[:start].strip():
        piece = '[...] ' + piece
    if text[end:].strip():
        piece = piece + ' [...]'
    return piece


def make_item(text, seen, rng_key):
    """Everything about one channel text of one step.
    Returns (label of the full text, item text, label of the item text, pattern names in the item, cut?)."""
    sents = sentence_labels(text, seen)
    if not sents:
        return 'none', '', 'none', [], False
    label, _ = label_of(sents)
    if len(text) <= MAX_CHARS:
        item, cut = tidy(text), False
    else:
        a, b, focus = relevant_sentence(text, sents, label, rng_key)
        item, cut = cut_window(text, a, b, focus), True
    item_label, item_names = label_of(sentence_labels(item, seen))  # what the reader of the item will see
    return label, item, item_label, item_names, cut


def iter_items(source, steps, task, channels, key_prefix, stats, only=None):
    """Yield (step number, channel, make_item result) for every non-empty channel text of a trajectory.
    seen = the task text plus every tool output BEFORE this step (rq3's rule 2 uses it).
    only: a set of (step, channel) to compute; the others are skipped (used when re-reading)."""
    seen = task
    for step_no, raw in enumerate(steps, 1):
        st = rq3.clean_channels(raw, stats)
        for ch in channels:
            if only is not None and (step_no, ch) not in only:
                continue
            text = st[ch]
            if text.strip():
                yield step_no, ch, make_item(text, seen, f'{SEED}|{key_prefix}|{step_no}|{ch}')
        seen += '\n' + raw.get('obs', '')  # this step's tool output is readable from the next step on


# ---------------------------------------------------------------------------------------------
# 3. Splits, quotas, drawing
# ---------------------------------------------------------------------------------------------

def task_splits(trajs):
    """Split by TASK: per benchmark, a seeded random 20% of the instance_ids are 'test', the rest 'train'.
    Every trajectory (every model) of a task lands in the same split."""
    split = {}
    for source in sorted({t['source'] for t in trajs}):
        ids = sorted({t['instance_id'] for t in trajs if t['source'] == source})
        random.Random(f'{SEED}|split|{source}').shuffle(ids)
        n_test = round(TEST_SHARE * len(ids))
        for i, iid in enumerate(ids):
            split[(source, iid)] = 'test' if i < n_test else 'train'
    return split


def fair_shares(total, capacity):
    """Split `total` items as evenly as possible over the groups of `capacity` (group -> items available),
    never giving a group more than it has. Rounding leftovers go to groups in sorted order.
    Returns (shares, items that could not be placed anywhere)."""
    shares = {g: 0 for g in capacity}
    left = total
    while left > 0:
        open_groups = [g for g in sorted(capacity) if shares[g] < capacity[g]]
        if not open_groups:
            break
        each = max(1, left // len(open_groups))
        for g in open_groups:
            give = min(each, capacity[g] - shares[g], left)
            shares[g] += give
            left -= give
            if left == 0:
                break
    return shares, left


def matched_group(label):
    return 'matched' if label in ('A', 'B') else 'none'


def plan_quotas(cands, n_total, splits):
    """Quotas per (cell, matched group, split). cell = (source, model).
    45% matched / 55% non-matching overall; each group shared evenly over cells (fair_shares);
    inside a cell, 20% of the group's items come from test tasks (for the OOD pool there is one split)."""
    pool = collections.Counter((c['cell'], matched_group(c['label']), c['split']) for c in cands)
    cells = sorted({c['cell'] for c in cands})
    quotas, notes = {}, []
    n_matched = round(n_total * MATCHED_SHARE)
    for group, target in (('matched', n_matched), ('none', n_total - n_matched)):
        cap = {cell: sum(pool[(cell, group, s)] for s in splits) for cell in cells}
        shares, short = fair_shares(target, cap)
        if short:
            notes.append(f'NOT ENOUGH {group} items anywhere: {short} missing')
        for cell, q in shares.items():
            if len(splits) == 1:
                quotas[(cell, group, splits[0])] = q
                continue
            n_test = min(round(q * TEST_SHARE), pool[(cell, group, 'test')])
            n_train = min(q - n_test, pool[(cell, group, 'train')])
            n_test = q - n_train  # if the train side was short, take the rest from test (q <= both sides)
            quotas[(cell, group, 'train')] = n_train
            quotas[(cell, group, 'test')] = n_test
            if n_test != round(q * TEST_SHARE):
                notes.append(f'{cell} {group}: test share moved to {n_test} of {q} (pool train '
                             f'{pool[(cell, group, "train")]}, test {pool[(cell, group, "test")]})')
    return quotas, pool, notes


def draw(cands, quotas, pool_name):
    """Seeded uniform random sample without replacement inside each (cell, matched group, split)."""
    by = collections.defaultdict(list)
    for c in cands:
        by[(c['cell'], matched_group(c['label']), c['split'])].append(c)
    chosen = []
    for key in sorted(by):
        group = sorted(by[key], key=lambda c: (c['traj'], c['step'], c['channel']))
        random.Random(f'{SEED}|draw|{pool_name}|{key}').shuffle(group)
        chosen += group[:quotas.get(key, 0)]
    return chosen


# ---------------------------------------------------------------------------------------------
# 4. The passes
# ---------------------------------------------------------------------------------------------

def collect_candidates(trajs, channels, splits, stats, seen_texts):
    """Pass 1: label every channel text of every step; keep a small record for each usable item.
    Unusable: shorter than MIN_CHARS, a duplicate of an earlier item text, or an item window whose
    lexicon label differs from the full step's label (a cut that changed what the lexicon sees)."""
    cands = []
    t0 = time.time()
    for ti, t in enumerate(trajs):
        steps, task = read_trajectory(t['source'], t['path'])
        key_prefix = f"{t['source']}|{t['model']}|{t['traj_id']}"
        for step_no, ch, (label, item, item_label, _, cut) in iter_items(
                t['source'], steps, task, channels, key_prefix, stats):
            stats[f"{t['source']}: channel texts"] += 1
            if len(item) < MIN_CHARS:
                stats[f"{t['source']}: dropped, shorter than {MIN_CHARS} characters"] += 1
                continue
            if item_label != label:
                stats[f"{t['source']}: dropped, cut changed the lexicon label ({label} -> {item_label})"] += 1
                continue
            h = hash(item)  # same text = same hash within this run (only used for duplicates)
            if h in seen_texts:
                stats[f"{t['source']}: dropped, duplicate item text"] += 1
                continue
            seen_texts.add(h)
            cands.append({'traj': ti, 'step': step_no, 'channel': ch, 'label': label, 'hash': h,
                          'cell': (t['source'], t['model']), 'split': splits[(t['source'], t['instance_id'])]})
        del steps, task
        if (ti + 1) % 250 == 0:
            print(f'    {ti + 1}/{len(trajs)} trajectories ({time.time() - t0:.0f}s)', flush=True)
    return cands


def build_records(chosen, trajs, channels, stats):
    """Pass 2: re-read only the trajectories that hold chosen items and rebuild the item texts."""
    by_traj = collections.defaultdict(dict)
    for c in chosen:
        by_traj[c['traj']][(c['step'], c['channel'])] = c
    records = []
    for ti in sorted(by_traj):
        t = trajs[ti]
        steps, task = read_trajectory(t['source'], t['path'])
        key_prefix = f"{t['source']}|{t['model']}|{t['traj_id']}"
        want = by_traj[ti]
        for step_no, ch, (label, item, item_label, names, cut) in iter_items(
                t['source'], steps, task, channels, key_prefix, collections.Counter(), only=set(want)):
            c = want[(step_no, ch)]
            if hash(item) != c['hash'] or label != c['label'] or item_label != label:
                sys.exit(f'pass 2 did not reproduce pass 1 for {key_prefix} step {step_no} {ch}')
            records.append({'text': item, 'split': c['split'], 'source': t['source'], 'model': t['model'],
                            'instance_id': t['instance_id'], 'step': step_no, 'channel': ch,
                            'lexicon_match': label, 'lexicon_patterns': names, 'traj_id': t['traj_id'],
                            'cut': cut})
        del steps, task
    if len(records) != len(chosen):
        sys.exit(f'pass 2 found {len(records)} of {len(chosen)} chosen items')
    return records


def deal_into_batches(records):
    """Deal the items like cards into equal piles so every batch has the same mix of pool, lexicon
    label, model and split; then shuffle each pile and number the items (ids carry no information)."""
    rng = random.Random(f'{SEED}|batches')
    for r in records:
        r['_order'] = rng.random()
    records.sort(key=lambda r: (r['split'] == 'ood', r['lexicon_match'], r['source'], r['model'], r['split'],
                                r['_order']))
    n_batches = -(-len(records) // BATCH_SIZE)
    piles = [[] for _ in range(n_batches)]
    for i, r in enumerate(records):
        piles[i % n_batches].append(r)
    k = 0
    for pile in piles:
        rng.shuffle(pile)
        for r in pile:
            k += 1
            r['item_id'] = f'ra{k:04d}'
            del r['_order']
    return piles


FIELDS = ['item_id', 'text', 'split', 'source', 'model', 'instance_id', 'step', 'channel', 'lexicon_match',
          'lexicon_patterns', 'traj_id', 'cut']


def write_batches(piles, out_dir):
    full_dir, blind_dir = os.path.join(out_dir, 'batches'), os.path.join(out_dir, 'batches_blind')
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(blind_dir, exist_ok=True)
    paths = []
    for i, pile in enumerate(piles, 1):
        name = f'b_{i:02d}.jsonl'
        with open(os.path.join(full_dir, name), 'w') as fh:
            for r in pile:
                fh.write(json.dumps({k: r[k] for k in FIELDS}, ensure_ascii=False) + '\n')
        with open(os.path.join(blind_dir, name), 'w') as fh:
            for r in pile:
                fh.write(json.dumps({'item_id': r['item_id'], 'text': r['text']}, ensure_ascii=False) + '\n')
        paths.append(os.path.join(blind_dir, name))
    return paths


# ---------------------------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------------------------

def check_loader():
    """Check: for every You Wang et al. file under 20 MB, the raw_output-skipping loader gives exactly
    the same steps and task text as a plain json.load."""
    fc_dir = os.path.join(ROOT, 'data', 'faultyclaims', 'reasoning_replication', 'data')
    n_same = n_diff = n_big = 0
    for f in sorted(glob.glob(os.path.join(fc_dir, '*', '*.json'))):
        if os.path.basename(f) in ('resolution_status.json', 'sampled_instance_ids.json'):
            continue
        if os.path.getsize(f) > 20e6:
            n_big += 1
            continue
        a, b = load_json(f), load_json_skip_raw_output(f)
        same = (rq3.chat_steps(a) == rq3.chat_steps(b) and rq3.task_text(a['messages']) == rq3.task_text(b['messages']))
        n_same += same
        n_diff += not same
        if not same:
            print('DIFFERENT:', f)
    print(f'loader check: {n_same} files identical, {n_diff} different, {n_big} files over 20 MB not compared')


def show(title, counter):
    print(title)
    for k, v in sorted(counter.items(), key=lambda kv: str(kv[0])):
        print(f'    {k}: {v}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'recall_audit'))
    ap.add_argument('--limit', type=int, default=None, help='trajectories per run/model (quick test)')
    ap.add_argument('--check-loader', action='store_true')
    args = ap.parse_args()
    if args.check_loader:
        check_loader()
        return
    print('started', time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    stats = collections.Counter()
    seen_texts = set()

    # --- training pool ---
    train_trajs = training_trajectories(args.limit)
    splits = task_splits(train_trajs)
    print(f'training pool: {len(train_trajs)} trajectories; tasks per split:',
          dict(collections.Counter((s, v) for (s, _), v in splits.items())), flush=True)
    tr_cands = collect_candidates(train_trajs, ('hidden', 'visible'), splits, stats, seen_texts)
    tr_quota, tr_pool, tr_notes = plan_quotas(tr_cands, N_TRAINING_POOL, ('train', 'test'))
    tr_chosen = draw(tr_cands, tr_quota, 'training')

    # --- OOD pool (DeepSWE) ---
    ood_trajs, model_check = ood_trajectories(args.limit)
    print('DeepSWE open-weight models (characters per reasoning token; used = full reasoning):')
    for model, rec in model_check.items():
        print(f'    {model}: {rec}')
    train_tasks = {t['instance_id'] for t in train_trajs}
    leaked = [t for t in ood_trajs if t['instance_id'] in train_tasks]
    ood_trajs = [t for t in ood_trajs if t['instance_id'] not in train_tasks]
    print(f'OOD pool: {len(ood_trajs)} trajectories, {len({t["instance_id"] for t in ood_trajs})} tasks; '
          f'{len(leaked)} dropped because their task is in the training pool', flush=True)
    ood_splits = {(t['source'], t['instance_id']): 'ood' for t in ood_trajs}
    ood_cands = collect_candidates(ood_trajs, ('hidden',), ood_splits, stats, seen_texts)
    ood_quota, ood_pool, ood_notes = plan_quotas(ood_cands, N_OOD, ('ood',))
    ood_chosen = draw(ood_cands, ood_quota, 'ood')

    # --- pass 2 and output ---
    records = build_records(tr_chosen, train_trajs, ('hidden', 'visible'), stats) + \
        build_records(ood_chosen, ood_trajs, ('hidden',), stats)
    piles = deal_into_batches(records)
    paths = write_batches(piles, args.out)

    # --- report ---
    show('cleaning and filtering counts:', stats)
    for name, pool, quota, notes in (('training', tr_pool, tr_quota, tr_notes), ('ood', ood_pool, ood_quota, ood_notes)):
        print(f'{name} pool: usable items (cell, matched group, split) -> available / drawn')
        for key in sorted(set(pool) | set(quota)):
            print(f'    {key}: {pool.get(key, 0)} / {quota.get(key, 0)}')
        for n in notes:
            print('    note:', n)
    labels_avail = collections.Counter((c['cell'], c['label']) for c in tr_cands + ood_cands)
    show('usable items per cell and lexicon label (A / B / none):', labels_avail)
    show('final items by pool split:', collections.Counter(r['split'] for r in records))
    show('final items by split and lexicon label:', collections.Counter((r['split'], r['lexicon_match']) for r in records))
    show('final items by source, model:', collections.Counter((r['source'], r['model']) for r in records))
    show('final items by source, model, lexicon label:',
         collections.Counter((r['source'], r['model'], r['lexicon_match']) for r in records))
    show('final items by split, channel:', collections.Counter((r['split'], r['channel']) for r in records))
    show('final items by split, cut to a window:', collections.Counter((r['split'], r['cut']) for r in records))
    show('final tasks per split:', collections.Counter(s for s, _ in {(r['split'], (r['source'], r['instance_id']))
                                                                      for r in records}))
    lens = [len(r['text']) for r in records]
    print(f'item length (characters): min {min(lens)}, median {statistics.median(lens)}, max {max(lens)}')
    show('items per batch and pool:', collections.Counter(
        (i + 1, 'ood' if r['split'] == 'ood' else 'training') for i, p in enumerate(piles) for r in p))
    print('blind batches:')
    for p in paths:
        print('   ', p)
    print('finished', time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    main()
