"""Print the basic facts for each paper in an arXiv API answer (an Atom XML file).

Used for the related-work notes: it lets us check titles, authors, dates and the
"comment" / "journal_ref" fields (where authors often write the venue) straight from arXiv,
instead of trusting search-engine summaries.

Usage:
    python3 -I scripts/arxiv_meta.py [--abstract] data/related_work_meta/arxiv_core.xml [more.xml ...]

--abstract also prints each paper's abstract.
The XML files are downloaded data, so we only parse them; we never run anything inside them.
"""
import sys
import xml.etree.ElementTree as ET

# Optional flag: print abstracts too.
SHOW_ABSTRACT = "--abstract" in sys.argv
FILES = [arg for arg in sys.argv[1:] if arg != "--abstract"]

# The XML tags live in these two namespaces.
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def text(node, path):
    """Return the stripped text at `path` under `node`, or '' if it is missing."""
    found = node.find(path, NS)
    if found is None or found.text is None:
        return ""
    return " ".join(found.text.split())  # squash newlines and double spaces


for xml_file in FILES:
    root = ET.parse(xml_file).getroot()
    for entry in root.findall("a:entry", NS):
        arxiv_url = text(entry, "a:id")  # e.g. http://arxiv.org/abs/2506.12286v4
        authors = [text(a, "a:name") for a in entry.findall("a:author", NS)]
        categories = [c.get("term") for c in entry.findall("a:category", NS)]
        print("ID:       ", arxiv_url)
        print("TITLE:    ", text(entry, "a:title"))
        print("AUTHORS:  ", "; ".join(authors))
        print("PUBLISHED:", text(entry, "a:published"), " UPDATED:", text(entry, "a:updated"))
        print("CATS:     ", ", ".join(categories))
        print("COMMENT:  ", text(entry, "arxiv:comment"))
        print("JOURNAL:  ", text(entry, "arxiv:journal_ref"))
        print("DOI:      ", text(entry, "arxiv:doi"))
        if SHOW_ABSTRACT:
            print("ABSTRACT: ", text(entry, "a:summary"))
        print("-" * 80)
