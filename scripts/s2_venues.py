"""Print venue information from a Semantic Scholar batch answer (a JSON list).

Semantic Scholar's "venue" field is a second opinion on where a paper was published.
We cross-check it against the arXiv comment field and official proceedings pages,
so this script only prints; it decides nothing.

Usage:
    python3 -I scripts/s2_venues.py data/related_work_meta/s2_batch.json
"""
import json
import sys

papers = json.load(open(sys.argv[1]))
for paper in papers:
    if paper is None:  # Semantic Scholar did not know this ID
        print("(not found)")
        continue
    ids = paper.get("externalIds") or {}
    venue = paper.get("venue") or ""
    pub_venue = (paper.get("publicationVenue") or {}).get("name", "")
    journal = (paper.get("journal") or {}).get("name", "")
    print(
        f"{ids.get('ArXiv', '?'):11} | {paper.get('year')} | venue={venue!r} | "
        f"pubVenue={pub_venue!r} | journal={journal!r} | DBLP={ids.get('DBLP', '')} | "
        f"{(paper.get('title') or '')[:60]}"
    )
