"""Build the derived-data bundle attached to a GitHub release: release/rr-derived-data-v<version>.zip.

Same layout as v0.1 (paths relative to the repo root): the big RQ1 tables that are not in git, the
error-type annotation, the provenance batches, labels and summary, the RecallAudit metrics and labels
(labels only: item ids, no text), the Docker validation of the trap catalog, and the catalog itself.
Usage: .venv/bin/python -I scripts/build_release_data.py 0.2
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = [
    'outputs/rq1/claims_features.jsonl', 'outputs/rq1/facts_swebv.jsonl', 'outputs/rq1/facts_deepswe.jsonl',
    'outputs/annotation', 'outputs/provenance', 'outputs/recall_audit',
    'outputs/traps/validation.json', 'outputs/traps/validation.csv', 'outputs/traps/validation_breakdown.json',
    'data/recall_audit/labels', 'traps/catalog',
]
SKIP = {'.DS_Store', 'READY'}


def files(path):
    p = ROOT / path
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob('*') if f.is_file() and f.name not in SKIP and '__pycache__' not in f.parts)


def main():
    version = sys.argv[1]
    out = ROOT / f'release/rr-derived-data-v{version}.zip'
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in INCLUDE:
            found = files(path)
            if not found:
                sys.exit(f'missing: {path}')
            for f in found:
                z.write(f, f.relative_to(ROOT).as_posix())
                n += 1
    print(f'wrote {out.relative_to(ROOT)}: {n} files, {out.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
