"""Search OpenReview for paper titles and print the venue each match was accepted to.

OpenReview hosts ICLR, COLM and (recently) NeurIPS/ICML reviews, so its "venue" field is a
primary source for acceptance at those conferences. We save each raw JSON answer and print
the matches whose title looks like the one we searched for.

Usage:
    python3 -I scripts/openreview_venues.py titles.txt data/related_work_meta/openreview
"""
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

titles_file, out_dir = sys.argv[1], pathlib.Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)


def field(content, key):
    """OpenReview v2 stores fields as {"value": ...}; v1 stores them directly."""
    value = content.get(key)
    return value.get("value") if isinstance(value, dict) else value


def simple(text):
    """Lowercase and keep only letters/digits, so small punctuation changes still match."""
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


titles = [line.strip() for line in open(titles_file) if line.strip()]
for number, title in enumerate(titles):
    url = "https://api2.openreview.net/notes/search?" + urllib.parse.urlencode(
        {"term": title, "limit": 10}
    )
    request = urllib.request.Request(url, headers={"User-Agent": "related-work-check/0.1"})
    try:
        raw = urllib.request.urlopen(request, timeout=40).read()
    except Exception as error:
        print(f"### {title}\n   ERROR {error}\n")
        continue
    (out_dir / f"{number:02d}.json").write_bytes(raw)

    print(f"### {title}")
    for note in json.loads(raw).get("notes", []):
        content = note.get("content", {})
        found_title = field(content, "title")
        # Only show notes whose title starts like ours (skips unrelated hits).
        if simple(found_title)[:40] != simple(title)[:40]:
            continue
        print(f"   {note.get('id')} | {field(content, 'venue')} | {field(content, 'venueid')} | {found_title}")
    print()
    time.sleep(1.5)  # be polite to the free API
