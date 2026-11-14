# Agent-targeted trap candidates (Nov 13, 2026)

Script: `scripts/agent_targeted_candidates.py` (extract, then report). Data: the public mini-swe-agent runs on SWE-bench Verified in `data/raw` (DeepSeek-V3.2, Kimi K2.5, MiniMax M2.5), repos at each task's base commit. No API calls.

**Question:** which repo facts does each agent's own final patch rely on without the agent ever having looked at them? Those are the facts a trap should change (pilot lesson: generic traps on facts the agent can see or never uses do not bite).

**How:** for every added line of the patch (moved or re-indented old lines excluded), find uses of things the repo defined before the patch, outside the code the patch changed. Record the exact fact relied on. Then find the step where the agent first wrote that line, and check every tool output before it for the definition (signature with those parameters, `class E`, the import, the `raise`, or an `inspect.signature`/`help()` call on it). Three levels:

- **unseen**: no tool output before the write showed the definition.
- **no use seen** (the memory tier): unseen, and no other use of the name was shown before the write either, and the line did not just keep a call that was already on the line it replaced.
- **strong**: no use seen, and the definition never shows up anywhere in the run.

## Bottom line

- 1476 final patches, 697 of them rely on at least one earlier repo definition (1949 model-level facts).
- **The agent never saw the definition before writing for 608 facts (31%).** But 567 of those (93%) had another use of the same name in an earlier output, or kept the call from the line being edited. In the first spot check, 7 of 10 unseen facts were copied that way, the same thing that made the pilot's django-11149 trap fail.
- **Real memory reliance is rare: 41 facts (2%) are in the memory tier** (no definition and no use shown before the agent wrote it), in 17 tasks. 27 candidates in 14 tasks pass the trap-usability filter (not a `None` default or a public module name); each still needs a manual look (see caveats).
- **Models rarely agree on them:** 1 trap-usable memory-tier candidate(s) shared by 2+ models: `sympy__sympy-13878` hyper(...) takes (ap, bq, ...) in this order (KM).
- **Spot check of the final memory tier: 10/10 correct** (definition and uses really never shown before the write; with 10 checks the 95% lower bound is 69%). Earlier versions of the heuristic missed `inspect.signature` output, sibling-class calls and `sed -n` ranges; all three are handled now.
- 18 of the 41 memory-tier facts were first exercised in a test command before the patch line was written, so a trap on them would likely surface as an error during that probe (detected and repaired, like the pilot's sift run) rather than as a silent wrong patch.
- Memory errors that reached a final patch (a call or import that cannot work at the base commit): 2.

## Coverage

| model | tasks with a run | non-empty patch | patch has 1+ repo fact | facts (model x task x fact) | unseen | no use seen | strong |
|---|---|---|---|---|---|---|---|
| deepseek-v3.2 | 500 | 485 | 222 | 633 | 228 (36%) | 21 (3%) | 13 (2%) |
| kimi-k2.5 | 500 | 495 | 242 | 689 | 199 (29%) | 11 (2%) | 9 (1%) |
| minimax-m2.5 | 500 | 496 | 233 | 627 | 181 (29%) | 9 (1%) | 5 (1%) |

## Facts by type and model (relied on / unseen / no use seen / strong)

| fact type | deepseek-v3.2 | kimi-k2.5 | minimax-m2.5 | all |
|---|---|---|---|---|
| arg_order | 59 / 26 / 5 / 5 | 50 / 21 / 1 / 1 | 38 / 14 / 1 / 1 | 147 / 61 / 7 / 7 |
| default | 76 / 47 / 4 / 4 | 97 / 59 / 6 / 6 | 82 / 46 / 4 / 2 | 255 / 152 / 14 / 12 |
| keyword | 74 / 15 / 1 / 1 | 82 / 17 / 0 / 0 | 67 / 18 / 0 / 0 | 223 / 50 / 1 / 1 |
| import | 56 / 26 / 8 / 2 | 91 / 7 / 2 / 0 | 93 / 7 / 1 / 0 | 240 / 40 / 11 / 2 |
| raises | 0 / 0 / 0 / 0 | 1 / 1 / 0 / 0 | 0 / 0 / 0 / 0 | 1 / 1 / 0 / 0 |
| exception_class | 8 / 6 / 0 / 0 | 6 / 5 / 0 / 0 | 5 / 3 / 0 / 0 | 19 / 14 / 0 / 0 |
| attribute | 360 / 108 / 3 / 1 | 362 / 89 / 2 / 2 | 342 / 93 / 3 / 2 | 1064 / 290 / 8 / 5 |

## Cross-model agreement

411 candidates (task x fact unseen by at least one model) in 164 of 500 tasks; 36 of them are in the memory tier (no use seen) for at least one model, in 17 tasks.

| k models | candidates unseen by exactly k | tasks (>= k) | no use seen by exactly k | tasks (>= k) | strong by exactly k | tasks (>= k) |
|---|---|---|---|---|---|---|
| 1 | 278 | 164 | 31 | 17 | 17 | 11 |
| 2 | 69 | 73 | 5 | 3 | 5 | 3 |
| 3 | 64 | 33 | 0 | 0 | 0 | 0 |

Memory-tier candidates shared by 2+ models, by type: default 4, arg_order 1.

Target defined in a different file from the edit: 307 of 411 candidates (75%). Resolution of the target name: import 120, self 117, unique_name 64, module 59, same_signature 26, class 13, same_file 12.

Share of model-level facts that sit in a run whose patch resolved the task: memory tier 55%, unseen 67%, definition seen 63%.

## Memory errors that reached the final patch (not trap candidates)

Uses of things that do not exist at the base commit, as written: a keyword the function does not take, more positional arguments than it accepts, an import of a name the module does not have, or a module attribute that is not there. These are leads for "wrong-version recall"; each needs a manual look (dynamic attributes and `__getattr__` can fool the check).

| check | deepseek-v3.2 | kimi-k2.5 | minimax-m2.5 |
|---|---|---|---|
| keyword_not_in_signature | 0 | 0 | 0 |
| too_many_positional | 1 | 0 | 0 |
| import_unresolved | 0 | 0 | 1 |
| module_attr_unresolved | 0 | 0 | 0 |

- minimax-m2.5 `scikit-learn__scikit-learn-13328` (not resolved): from ..utils import FLOAT_DTYPES: FLOAT_DTYPES not found in sklearn.utils at the base commit - `from ..utils import FLOAT_DTYPES`
- deepseek-v3.2 `sympy__sympy-19040` (no run report): dmp_degree_list takes 2 positional args, code passes 3 - `deg_list = dmp_degree_list(h, u, K)`

## Top 40 tasks (best trap-usable candidate per task)

Score = memory signal (per model: 6 strong, 4 no use seen, 1 only unseen) + trap usability (arg_order/keyword/raises 3, default 3 [-6 if the function has more than 6 defaults or the default is None], import 2 [-6 for a package's top-level API like `from sympy import asin`], attribute/exception 1 [-6 for public module names like models.Model]) + name resolution (direct 2, unique method name 1, same-signature overrides 0) + 2 if defined in another file + 1 per memory-tier model whose run resolved the task + rewrite size (<= 60 sites +1, <= 200 0, <= 600 -3, more -6). Models: D = DeepSeek-V3.2, K = Kimi K2.5, M = MiniMax M2.5. "Relied on it" lists every model whose patch uses the fact: upper case = memory tier (no definition, no use seen before writing it), lower case = definition unseen but a use was visible (likely copied), in brackets = definition seen. Sites = lines using the name at the base commit (source/tests/docs), a rough size of the trap rewrite.

| # | task | type | fact the patch relies on | defined in | relied on it | strong | resolved | sites src/tests/docs | other cands |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sympy__sympy-13878 | arg_order | hyper(...) takes (ap, bq, ...) in this order; code passes 3 positional args | `sympy/functions/special/hyper.py:180` (other file) | KM | KM | K+ M+ | 69/178/4 | 2 |
| 2 | sympy__sympy-12419 | import | Relational importable from sympy.core.relational (from sympy.core.relational import Relational) | `sympy/core/relational.py:23` (other file) | D | D | D+ | 15/4/0 | 2 |
| 3 | sympy__sympy-19040 | arg_order | dmp_div takes (f, g, ...) in this order; code passes 4 positional args | `sympy/polys/densearith.py:1609` (other file) | D | D | D- | 14/2/0 | 19 |
| 4 | django__django-15741 | import | force_str importable from django.utils.encoding (from django.utils.encoding import force_str) | `django/utils/encoding.py:56` (other file) | (k)M | - | K+ M+ | 18/1/0 | 3 |
| 5 | django__django-11149 | arg_order | get_permission_codename takes (action, opts, ...) in this order; code passes 2 positional args | `django/contrib/auth/__init__.py:195` (other file) | dkm | - | D- K+ M+ | 8/30/2 | 0 |
| 6 | django__django-13794 | import | Promise importable from django.utils.functional (from django.utils.functional import Promise) | `django/utils/functional.py:68` (other file) | D(k)(m) | - | D- K- M- | 7/2/0 | 0 |
| 7 | django__django-15022 | attribute | Q.AND exists (code reads .AND) | `django/db/models/query_utils.py:29` (other file) | D | - | D+ | 4/0/0 | 3 |
| 8 | django__django-15563 | attribute | Options.get_ancestor_link exists (code calls .get_ancestor_link(...)) | `django/db/models/options.py:702` (other file) | K | K | K- | 7/1/0 | 5 |
| 9 | django__django-16560 | keyword | ValidationError(...) accepts keyword code= | `django/core/exceptions.py:136` (other file) | D(k)(m) | D | D+ K- M+ | 143/159/35 | 3 |
| 10 | django__django-16901 | arg_order | Exact(...) takes (lhs, rhs, ...) in this order; code passes 2 positional args | `django/db/models/lookups.py:25` (other file) | dkm | - | D+ K+ M+ | 2/9/0 (runs inherited `Lookup.__init__`: every subclass call too) | 2 |
| 11 | scikit-learn__scikit-learn-26323 | keyword | _safe_set_output accepts keyword transform= | `sklearn/utils/_set_output.py:244` (other file) | dkm | - | D+ K+ M+ | 3/6/0 | 0 |
| 12 | sphinx-doc__sphinx-7985 | attribute | BuildEnvironment.get_domain exists (code calls .get_domain(...)) | `sphinx/environment/__init__.py:508` (other file) | Km | K | K- M+ | 46/22/0 | 3 |
| 13 | django__django-15957 | attribute | Query.clear_limits exists (code calls .clear_limits(...)) | `django/db/models/sql/query.py:2081` (other file) | M | - | M+ | 2/0/0 | 1 |
| 14 | matplotlib__matplotlib-20676 | attribute | _AxesBase.get_xlim exists (code calls .get_xlim(...)) | `lib/matplotlib/axes/_base.py:3564` (other file) | M | M | M- | 33/62/6 | 3 |
| 15 | pydata__xarray-4094 | default | <2 classes>.drop_vars leaves out errors, relies on default errors='raise' | `xarray/core/dataarray.py:2012` | (d)M | M | D+ M+ | 9/65/1 | 5 |
| 16 | scikit-learn__scikit-learn-13779 | arg_order | has_fit_parameter takes (estimator, parameter, ...) in this order; code passes 2 positional args | `sklearn/utils/validation.py:784` (other file) | dk | - | D+ K+ | 10/5/0 | 0 |
| 17 | sphinx-doc__sphinx-7748 | keyword | prepare_docstring accepts keyword tabsize= | `sphinx/util/docstrings.py:48` (other file) | km | - | K- M- | 5/3/1 | 1 |
| 18 | sympy__sympy-15345 | arg_order | StrPrinter.stringify takes (args, sep, ...) in this order; code passes 2 positional args | `sympy/printing/str.py:35` (other file) | dm | - | D+ M+ | 16/0/0 | 2 |
| 19 | django__django-11138 | arg_order | localtime takes (value, timezone, ...) in this order; code passes 2 positional args | `django/utils/timezone.py:191` (other file) | k | - | K+ | 4/3/1 | 2 |
| 20 | django__django-11555 | arg_order | Cast(...) takes (expression, output_field, ...) in this order; code passes 2 positional args | `django/db/models/functions/comparison.py:10` (other file) | d | - | D+ | 9/19/4 | 3 |
| 21 | django__django-11734 | arg_order | Col(...) takes (alias, target, ...) in this order; code passes 2 positional args | `django/db/models/expressions.py:748` (other file) | k | - | K- | 2/6/0 | 7 |
| 22 | django__django-11885 | import | OR importable from django.db.models.sql.where (from django.db.models.sql.where import OR) | `django/db/models/sql/where.py:None` (other file) | dm | - | D- M- | 10/6/17 | 5 |
| 23 | django__django-13406 | import | ValuesListIterable importable from django.db.models.query (from django.db.models.query import ValuesListIterable) | `django/db/models/query.py:115` | D(m) | - | D- M- | 1/0/0 | 3 |
| 24 | django__django-15098 | arg_order | _lazy_re_compile takes (regex, flags, ...) in this order; code passes 2 positional args | `django/utils/regex_helper.py:340` (other file) | dm | - | D- M- | 73/5/0 | 1 |
| 25 | matplotlib__matplotlib-24177 | default | _AxesBase.update_datalim leaves out updatex, relies on default updatex=True | `lib/matplotlib/axes/_base.py:2521` (other file) | d | - | D- | 17/0/0 | 2 |
| 26 | matplotlib__matplotlib-24637 | attribute | Artist.get_gid exists (code calls .get_gid(...)) | `lib/matplotlib/artist.py:618` (other file) | dkm | - | D+ K+ M+ | 20/0/1 | 4 |
| 27 | psf__requests-1724 | keyword | Request(...) accepts keyword method= | `requests/models.py:196` (other file) | k | - | K+ | 3/18/2 | 0 |
| 28 | pytest-dev__pytest-10081 | default | Config.getoption leaves out skip, relies on default skip=False | `src/_pytest/config/__init__.py:1503` (other file) | dkm | - | D+ K+ M+ | 57/7/14 | 3 |
| 29 | pytest-dev__pytest-5787 | keyword | ReprExceptionInfo(...) accepts keyword reprcrash= | `src/_pytest/_code/code.py:902` (other file) | m | - | M- | 2/0/0 | 1 |
| 30 | sphinx-doc__sphinx-7440 | arg_order | make_refnode takes (builder, fromdocname, ...) in this order; code passes 5 positional args | `sphinx/util/nodes.py:539` (other file) | k | - | K+ | 19/0/0 | 6 |
| 31 | sphinx-doc__sphinx-7462 | arg_order | desc_sig_punctuation(...) takes (rawsource, text, ...) in this order; code passes 2 positional args | `sphinx/addnodes.py:181` (other file) | d | - | D- | 7/0/0 (runs inherited `desc_sig_element.__init__`: every subclass call too) | 1 |
| 32 | sphinx-doc__sphinx-8548 | arg_order | safe_getattr takes (obj, name, ...) in this order; code passes 2 positional args | `sphinx/util/inspect.py:397` (other file) | k | - | K+ | 33/4/0 | 6 |
| 33 | sphinx-doc__sphinx-8638 | keyword | PyTypedField(...) accepts keyword can_collapse= | `sphinx/util/docfields.py:165` (other file) | m | - | M+ | 3/0/0 (runs inherited `TypedField.__init__`: every subclass call too) | 6 |
| 34 | sympy__sympy-17630 | arg_order | ZeroMatrix(...) takes (m, n, ...) in this order; code passes 2 positional args | `sympy/matrices/expressions/matexpr.py:948` (other file) | (d)km | - | D- K- M- | 19/64/0 | 0 |
| 35 | sympy__sympy-21596 | arg_order | ConditionSet(...) takes (sym, condition, ...) in this order; code passes 3 positional args | `sympy/sets/conditionset.py:89` (other file) | dk | - | D- K- | 53/144/0 | 2 |
| 36 | django__django-11951 | arg_order | <3 classes>.bulk_batch_size takes (fields, objs, ...) in this order; code passes 2 positional args | `django/db/backends/base/operations.py:68` (other file) | (d)km | - | D- K+ M+ | 3/5/0 | 1 |
| 37 | django__django-12325 | import | OneToOneField importable from django.db.models.fields.related (from django.db.models.fields.related import OneToOneField) | `django/db/models/fields/related.py:994` (other file) | d | - | D- | 5/5/0 | 1 |
| 38 | django__django-12663 | import | empty importable from django.utils.functional (from django.utils.functional import empty) | `django/utils/functional.py:None` (other file) | k(m) | - | K+ M+ | 8/23/18 | 0 |
| 39 | django__django-13297 | import | empty importable from django.utils.functional (from django.utils.functional import empty) | `django/utils/functional.py:None` (other file) | d | - | D+ | 8/23/19 | 0 |
| 40 | django__django-14404 | arg_order | resolve takes (path, urlconf, ...) in this order; code passes 2 positional args | `django/urls/base.py:21` (other file) | d | - | D+ | 60/61/13 | 0 |

## Memory-tier facts that are not clean traps

Relied on without reading, but changing them would not make a fair, behavior-preserving trap: `None` defaults ("not set"), constructors with more than 6 defaults, public module names.

- `django__django-16901` Mod(...) leaves out output_field, relies on default output_field=None (DM)
- `django__django-15695` Index(...) leaves out condition, relies on default condition=None (DK)
- `django__django-15695` Index(...) leaves out db_tablespace, relies on default db_tablespace=None (DK)
- `django__django-15695` Index(...) leaves out include, relies on default include=None (DK)
- `django__django-16819` Index(...) leaves out condition, relies on default condition=None (K)
- `django__django-16819` Index(...) leaves out db_tablespace, relies on default db_tablespace=None (K)
- `django__django-16819` Index(...) leaves out include, relies on default include=None (K)
- `sympy__sympy-13878` asin importable from sympy (from sympy import asin) (K)
- `django__django-15022` Q(...) leaves out _connector, relies on default _connector=None (M)

## Spot check by hand

For each candidate: open the trajectory, find the step that first wrote the line, and read every earlier tool output and command for the definition (or the region of the file holding it, or an `inspect`/`help` call) and for other uses of the name. Two verdicts: is "definition unseen" right, and did the agent really rely on memory (neither the definition nor a use was visible)? `scripts/spot_check_candidate.py` prints what is needed for one candidate.

**Round 1** (v1 (definition regex only)): definition unseen correct 9/10, real memory reliance 2/10.

| model | task | fact | definition unseen? | memory? | what the trajectory shows |
|---|---|---|---|---|---|
| kimi-k2.5 | django__django-16901 | Exact(lhs, rhs) argument order | yes | no | Lookup.__init__ never shown, but the agent edited the existing call Exact(1, rhs_sum) it saw at step 5 (only the second argument changed) |
| deepseek-v3.2 | scikit-learn__scikit-learn-26323 | _safe_set_output(..., transform=) keyword | yes | no | definition never shown; the same call _safe_set_output(trans, transform=transform) sat two lines above the edit (seen at step 6) |
| minimax-m2.5 | sympy__sympy-13878 | hyper(ap, bq, z) argument order | yes | yes | written at step 94; no earlier output shows hyper's signature or any hyper(...) call (only 'hyper is available' from an import test) |
| kimi-k2.5 | matplotlib__matplotlib-24637 | Artist.get_gid method name | yes | no | def never shown, but a grep at step 18 listed many renderer.open_group(..., gid=self.get_gid()) lines; the patch line copies that pattern |
| minimax-m2.5 | django__django-15375 | ._output_field_or_none attribute | yes | no | the patch turned the existing line return Coalesce(..., output_field=c._output_field_or_none) into an assignment; the attribute was already on the line |
| deepseek-v3.2 | django__django-15563 | Query.add_filter(filter_lhs, filter_rhs) argument order | yes | no | def never shown; several query.add_filter("pk__in", ...) calls were shown at steps 7-32 and the patch copies them |
| minimax-m2.5 | matplotlib__matplotlib-24177 | update_datalim(..., updatex=) keyword | NO | no | MISS: at step 39 the agent ran inspect.signature(Axes.update_datalim) and saw (self, xys, updatex=True, updatey=True) before writing at step 42; the output has no 'def' line so the regex missed it (fixed in v2: introspection counts as seeing the definition) |
| deepseek-v3.2 | sympy__sympy-12419 | from sympy.core.relational import Relational | yes | yes | written at step 69; no tool output in the whole run contains 'Relational' |
| kimi-k2.5 | django__django-11951 | bulk_batch_size(fields, objs) argument order | yes | no | the patch kept ops.bulk_batch_size(fields, objs) from the line it replaced (shown at step 6) |
| deepseek-v3.2 | pytest-dev__pytest-10081 | Config.getoption default skip=False | yes | no | the patch kept self.config.getoption("usepdb") from the line it replaced |

**Round 2** (v2 (+ introspection, + kept-from-old-line, + no-use tier)): definition unseen correct 3/4, real memory reliance 2/4.

| model | task | fact | definition unseen? | memory? | what the trajectory shows |
|---|---|---|---|---|---|
| kimi-k2.5 | sympy__sympy-13878 | hyper(ap, bq, z) argument order | yes | yes | written at step 38; first output mentioning hyper is step 42 |
| deepseek-v3.2 | sympy__sympy-19040 | dmp_div(f, g, u, K) argument order | yes | yes | written at step 78; first output mentioning dmp_div is step 98 |
| minimax-m2.5 | sphinx-doc__sphinx-9602 | desc_sig_keyword(rawsource, text) argument order | yes | no | MISS for the memory tier: the agent saw addnodes.desc_sig_punctuation('', '|') at step 13; that sibling class runs the same inherited desc_sig_element.__init__, so the order was visible (fixed in v3: calls to any class sharing the constructor count as uses) |
| minimax-m2.5 | sympy__sympy-16792 | InputArgument(..., dimensions=) keyword | NO | no | MISS: sed -n '290,360p' at step 14 printed the body of Variable.__init__ (def at line 289, just outside the range) including 'if dimensions and ...'; sibling calls OutputArgument(..., dimensions=dims) were shown at steps 21-22 (fixed in v3: file-region rule + sibling classes) |

**Round 3** (v3 (final: + file region, + sibling constructors, + any mention for attributes)): definition unseen correct 10/10, real memory reliance 10/10.

| model | task | fact | definition unseen? | memory? | what the trajectory shows |
|---|---|---|---|---|---|
| kimi-k2.5 | sympy__sympy-13878 | hyper(ap, bq, z) argument order | yes | yes | same as round 2 (still in the memory tier) |
| deepseek-v3.2 | sympy__sympy-12419 | from sympy.core.relational import Relational | yes | yes | same as round 1 (still in the memory tier) |
| deepseek-v3.2 | sympy__sympy-19040 | dmp_div(f, g, u, K) argument order | yes | yes | same as round 2 (still in the memory tier) |
| kimi-k2.5 | django__django-15563 | Options.get_ancestor_link method name | yes | yes | written at step 27 (query.get_meta().model._meta.get_ancestor_link(...)); no earlier output mentions get_ancestor_link |
| deepseek-v3.2 | django__django-16560 | ValidationError(..., code=) keyword | yes | yes | __init__(self, message, code=None, params=None) never shown; only 'class ValidationError(Exception):' at step 19. Caveat: the issue text asks for an error 'code' on the ValidationError constructor, so the name is hinted in prose (a bad trap target) |
| minimax-m2.5 | django__django-15741 | from django.utils.encoding import force_str | yes | yes | first written at step 10 (inside a test command); nothing before mentions force_str |
| minimax-m2.5 | matplotlib__matplotlib-20676 | Axes.get_xlim method name | yes | yes | never read; caveat: the agent's own test at step 7 already ran ax.get_xlim() successfully, so a renamed method would have failed at that probe |
| minimax-m2.5 | pydata__xarray-4094 | drop_vars default errors='raise' | yes | yes | signature never shown; caveat: the patch guards with 'if dim in selected.coords', so the default never matters (useless as a trap) |
| kimi-k2.5 | sphinx-doc__sphinx-7985 | BuildEnvironment.get_domain method name | yes | yes | written at step 34; only lines 366-380 of environment/__init__.py were viewed (def at 508); first mention in an output is step 42 |
| deepseek-v3.2 | django__django-13794 | from django.utils.functional import Promise | yes | yes | a grep for 'Promise\|functional' in defaultfilters.py at step 31 found nothing; the import was written at step 34 |

## Caveats and what this means for the trap design

- Only final patches are analyzed. A fact recalled in an earlier attempt, or only in the reasoning, is not counted.
- Recall of the heuristic was not measured. The memory tier is cautious by design: any visible use of the name disqualifies a fact, so some real memory facts are surely excluded.
- When the receiver of `x.method()` cannot be resolved, the target is guessed only if exactly one repo class defines a method with that name ("unique_name"), excluding names that exist on builtins, numpy, pandas, docutils, scipy.sparse and pluggy (and all matplotlib names for seaborn). One wrong guess (seaborn `formatter.format_data` is matplotlib's, not seaborn's) was found and fixed; other guesses are not all checked.
- Classes built by metaclasses, star imports, module `__getattr__` and `globals()` tricks are skipped, which keeps false memory errors out but loses some facts.
- Argument-order traps only bite if the two parameters are not interchangeable (`dmp_mul(f, g)` is commutative; `dmp_div`, `dmp_sub` are not). Default traps only bite if the code path depends on the default (xarray-4094 guards the call so `errors=` never matters).
- Call-site counts are lines matching the name at the base commit (docs included), so they overcount common names. For inherited constructors (`Exact(...)` runs `Lookup.__init__`) they undercount: the real rewrite covers every subclass.
- The public runs had network access (a `pip install pytest` succeeded in at least one trajectory), but none of the memory-tier runs fetched upstream code (no curl/wget/git clone/pip download of the repo).
- **Design implication:** facts that agents rely on from memory are rare and idiosyncratic (almost no cross-model agreement), so traps cannot be picked once for all models from public runs. The plan from the pilot stands, with this script as its second step: run each model on the original repo with network off, several times, mine its own patches with `agent_targeted_candidates.py`, and trap facts in the memory tier that recur across runs of that model.

Reproduce: `.venv/bin/python -I scripts/prefetch_blobs.py data/meta/swebv/data/test-00000-of-00001.parquet data/repos src`, then `.venv/bin/python -I scripts/agent_targeted_candidates.py extract` and `... report` (the report downloads the test and doc files of the top tasks for the call-site counts). Files: `outputs/agent_targeted_facts.jsonl` (every fact), `outputs/agent_targeted_candidates.jsonl` (ranked candidates), `outputs/agent_targeted_spot_check.json` (hand verdicts).
