"""rr_traps: memory traps for coding-agent benchmarks ("Read or Remembered?").

A memory trap changes one fact a model has probably memorized about a repository (an argument
order, a default, a name, a module path, an exception, a return order, a keyword, a flag) and
carries the change through the whole repo so it still works. Each trap has a placebo: an edit of
the same kind at the same places that keeps the remembered fact true.

  transforms.py  the eight transforms T1-T8 (trap and placebo modes)
  common.py      helpers (file walking, libcst argument helpers, doctest/doc text rewriting)
  builder.py     builds a catalog entry for one SWE-bench task: worktree, trap/placebo diffs,
                 gold and test patches carried through the same transform, spec.json
"""
from .transforms import (TYPES, ArgSwap, DefaultChange, ExceptionSwap, FlagInvert, KeywordRename, Move,
                         Rename, Report, ReturnSwap, TrapError, make)

__all__ = ['TYPES', 'ArgSwap', 'DefaultChange', 'ExceptionSwap', 'FlagInvert', 'KeywordRename', 'Move',
           'Rename', 'Report', 'ReturnSwap', 'TrapError', 'make']
