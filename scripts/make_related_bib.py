"""Build notes/related.bib from arXiv metadata we downloaded, plus a hand-checked venue table.

Why a script: author lists and titles come straight from the arXiv API answers in
data/related_work_meta/*.xml, so the BibTeX cannot drift from what arXiv says. The only
hand-written part is the VENUES table below, and every row says where the venue was checked.

Usage:
    python3 -I scripts/make_related_bib.py data/related_work_meta notes/related.bib

The XML files are downloaded data: we only parse them.
"""
import glob
import sys
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# One row per paper: (bib key, arXiv id, booktitle or None, venue year or None, note or None).
# booktitle None  -> written as an arXiv preprint (@misc).
# The comment after each row says how the venue was checked (OR = OpenReview API, AC = the
# authors' own arXiv comment field, PROC = official proceedings page, ACL = ACL Anthology page).
VENUES = [
    # --- closest work: memorization / grounding in SWE agents ---
    ("liang2026swebenchillusion", "2506.12286",
     "Proceedings of the IEEE/ACM 48th International Conference on Software Engineering: "
     "Software Engineering in Practice (ICSE-SEIP)", 2026, None),  # conf.researchr.org ICSE 2026 SEIP page
    ("prathifkumar2025swebvmemory", "2512.10218", None, None, None),
    ("chen2026schrodingerrepo", "2609.27891", None, None, None),
    ("li2026repomirage", "2605.26177", None, None, None),  # NeurIPS 2026 listing not verified
    ("mohammadi2026workingset", "2608.16630", None, None, None),
    ("wang2026faultyclaims", "2610.32338", None, None, None),
    ("wang2026solvedissues", "2503.15223",
     "Proceedings of the IEEE/ACM 48th International Conference on Software Engineering (ICSE)",
     2026, None),  # conf.researchr.org ICSE 2026 author page lists it
    ("zheng2026sweproverified", "2609.08149", None, None, None),
    ("mehtiyev2026stalefiles", "2611.09802", None, None, None),
    ("tan2026swetouch", "2608.02499", None, None, None),
    ("balepur2026cabra", "2610.10610", None, None, None),
    # --- SWE benchmarks and their validity ---
    ("jimenez2024swebench", "2310.06770",
     "International Conference on Learning Representations (ICLR)", 2024, "Oral"),  # OR
    ("badertdinov2025swerebench", "2505.20411",
     "Advances in Neural Information Processing Systems 38 (NeurIPS 2025), Datasets and Benchmarks Track",
     2025, None),  # PROC
    ("zhang2025swebenchlive", "2505.23419",
     "Advances in Neural Information Processing Systems 38 (NeurIPS 2025), Datasets and Benchmarks Track",
     2025, None),  # PROC
    ("deng2025swebenchpro", "2509.16941", None, None, None),
    ("aleithan2024swebenchplus", "2410.06992", None, None, None),
    ("garg2026savingswebench", "2510.08996",
     "IEEE/ACM 5th International Conference on AI Engineering - Software Engineering for AI (CAIN)",
     2026, "Venue per the authors' arXiv comment"),  # AC
    ("sahoo2026agentlens", "2605.12925", None, None, None),
    ("kim2026coherencecollapse", "2603.24631", None, None, None),
    ("liu2023evalplus", "2305.01210",
     "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)", 2023, None),  # PROC
    # --- contamination detection ---
    ("shi2024mink", "2310.16789",
     "International Conference on Learning Representations (ICLR)", 2024, None),  # OR
    ("golchin2024timetravel", "2308.08493",
     "International Conference on Learning Representations (ICLR)", 2024, "Spotlight"),  # OR + AC
    ("deng2024tsguessing", "2311.09783",
     "Proceedings of the 2024 Conference of the North American Chapter of the Association for "
     "Computational Linguistics (NAACL)", 2024, "Venue per the authors' arXiv comment"),  # AC
    ("oren2024proving", "2310.17623",
     "International Conference on Learning Representations (ICLR)", 2024, "Oral"),  # OR
    ("roberts2024cutoff", "2310.10628",
     "International Conference on Learning Representations (ICLR)", 2024,
     "ICLR title: To the Cutoff... and Beyond? A Longitudinal Perspective on LLM Data Contamination"),  # OR
    ("yang2023rephrased", "2311.04850", None, None, None),
    ("riddell2024codecontamination", "2403.04811",
     "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics "
     "(Volume 1: Long Papers)", 2024, "Pages 14116--14137"),  # ACL
    ("zhou2025lessleak", "2502.06215", None, None, None),
    ("ramos2025memorizingbugs", "2411.13323", None, None, None),
    ("lan2026zerocot", "2605.21856", None, None, None),
    ("meeus2026functionalmem", "2606.12764", None, None, None),
    # --- perturbation / mutation / refactoring for code evaluation ---
    ("wang2023recode", "2212.10264",
     "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics "
     "(Volume 1: Long Papers)", 2023, None),  # ACL (2023.acl-long.773)
    ("xia2024evoeval", "2403.19114", "Conference on Language Modeling (COLM)", 2024, None),  # OR
    ("jain2025livecodebench", "2403.07974",
     "International Conference on Learning Representations (ICLR)", 2025, None),  # OR
    ("cao2024concerned", "2403.16898", None, None, None),
    ("cao2024codecleaner", "2411.10842", None, None, None),
    ("chen2025dycodeeval", "2503.04149",
     "Proceedings of the 42nd International Conference on Machine Learning (ICML)", 2025,
     "PMLR volume 267"),  # AC + PMLR page in search results
    ("miceli2023identifierswaps", "2305.15507",
     "Findings of the Association for Computational Linguistics: ACL 2023", 2023,
     "Pages 272--292"),  # ACL
    ("wu2024reasoningreciting", "2307.02477",
     "Proceedings of the 2024 Conference of the North American Chapter of the Association for "
     "Computational Linguistics (NAACL)", 2024, "Venue per the authors' arXiv comment"),  # AC
    ("mirzadeh2025gsmsymbolic", "2410.05229",
     "International Conference on Learning Representations (ICLR)", 2025, None),  # OR
    ("storek2026sense", "2505.13353", None, None,
     "Accepted to ACL 2026 (main) per the authors' arXiv comment"),  # AC
    # --- knowledge conflicts: context vs parametric memory ---
    ("longpre2021entity", "2109.05052",
     "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
     2021, "Venue per the authors' arXiv comment"),  # AC
    ("xie2024chameleon", "2305.13300",
     "International Conference on Learning Representations (ICLR)", 2024, "Spotlight"),  # OR
    ("wu2024clasheval", "2404.10198",
     "Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track",
     2024, None),  # PROC
    ("mallen2023popqa", "2212.10511",
     "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)",
     2023, "Venue per the authors' arXiv comment"),  # AC
    ("liu2024codeupdatearena", "2407.06249", None, None, None),
    ("wang2025deprecatedapis", "2406.09834",
     "Proceedings of the IEEE/ACM 47th International Conference on Software Engineering (ICSE)", 2025,
     "Venue per the authors' arXiv comment"),  # AC
    ("islah2024gitchameleon", "2411.05830", None, None, None),
    ("misra2025gitchameleon2", "2507.12367", None, None, None),
    ("ashik2026lagbehind", "2604.09515", None, None, None),
    # --- CoT faithfulness and monitoring ---
    ("turpin2023unfaithful", "2305.04388",
     "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)", 2023, None),  # PROC + AC
    ("lanham2023faithfulness", "2307.13702", None, None, None),
    ("chen2025reasoningmodels", "2505.05410", None, None, None),
    ("chua2025deepseekfaithful", "2501.08156", None, None, None),
    ("arcuschin2026wild", "2503.08679",
     "Proceedings of the 43rd International Conference on Machine Learning (ICML)", 2026, None),  # OR + AC
    ("baker2025monitoring", "2503.11926", None, None, None),
    ("korbak2025monitorability", "2507.11473", None, None, None),
    ("turpin2025verbalize", "2506.22777", None, None,
     "ICML 2025 Workshop on Reliable and Responsible Foundation Models, per the authors' arXiv comment"),
    ("chua2024bct", "2403.05518", None, None, None),
    # --- agent trajectories and fabricated observations ---
    ("bouzenia2025trajectories", "2506.18824",
     "Proceedings of the 40th IEEE/ACM International Conference on Automated Software Engineering (ASE)",
     2025, "Venue per the authors' arXiv comment"),  # AC
    ("cuadron2025overthinking", "2502.08235", None, None, None),
    ("cemri2025mast", "2503.13657",
     "Advances in Neural Information Processing Systems 38 (NeurIPS 2025), Datasets and Benchmarks Track",
     2025, None),  # PROC (first page of the proceedings PDF)
    ("kraishan2026narrative", "2609.12205", None, None, None),
    ("zhang2025miragebench", "2507.21017", None, None, None),
    ("gurram2026tooluse", "2604.16706", None, None, None),
    # --- exemplar empirical papers (structure models) not listed above ---
    ("shi2023distracted", "2302.00093",
     "Proceedings of the 40th International Conference on Machine Learning (ICML)", 2023,
     "Venue per the authors' arXiv comment"),  # AC
    ("zhang2024gsm1k", "2405.00332",
     "Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track",
     2024, "Venue per the authors' arXiv comment"),  # AC
    ("dominguezolmedo2025testtask", "2407.07890",
     "International Conference on Learning Representations (ICLR)", 2025, "Oral"),  # OR
]

# Two sources that are not arXiv papers; both pages were opened on 2026-11-13.
EXTRA = r"""
@misc{openai2026swebv,
  author       = {{OpenAI}},
  title        = {Why We No Longer Evaluate {SWE}-bench Verified},
  howpublished = {OpenAI blog post},
  year         = {2026},
  month        = feb,
  note         = {Published February 23, 2026. Accessed 2026-11-13},
  url          = {https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/}
}

@misc{swebench2025issue465,
  title        = {Repo State Loopholes During Agentic Evaluation ({SWE}-bench {GitHub} issue \#465)},
  organization = {SWE-bench},
  key          = {SWE-bench},
  howpublished = {GitHub issue, opened September 3, 2025},
  year         = {2025},
  note         = {Future commits reachable via git log --all; fixed for Verified in PR \#471 (swebench v4.1.0). Accessed 2026-11-13},
  url          = {https://github.com/SWE-bench/SWE-bench/issues/465}
}
"""


def text(node, path):
    found = node.find(path, NS)
    return "" if found is None or found.text is None else " ".join(found.text.split())


def protect(value):
    """Escape the few characters BibTeX treats specially inside titles and notes."""
    return value.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


# Accented letters that appear in our author names/titles, written as LaTeX accents so the
# .bib works with any engine (pdflatex, xelatex) and font. Add a row if a new one shows up.
ACCENTS = {
    "á": r"{\'a}", "ä": r'{\"a}', "ã": r"{\~a}", "ą": r"{\k{a}}", "ç": r"{\c{c}}",
    "é": r"{\'e}", "ė": r"{\.e}", "í": r"{\'i}", "ñ": r"{\~n}", "ó": r"{\'o}", "ö": r'{\"o}',
    "š": r"{\v{s}}", "Š": r"{\v{S}}", "ú": r"{\'u}", "ü": r'{\"u}', "ū": r"{\=u}",
}


def ascii_latex(value):
    """Replace each accented letter by its LaTeX accent command."""
    return "".join(ACCENTS.get(ch, ch) for ch in value)


# Read every arXiv answer we saved and index the entries by bare arXiv id (no version).
papers = {}
for xml_file in sorted(glob.glob(sys.argv[1] + "/arxiv_*.xml")):
    for entry in ET.parse(xml_file).getroot().findall("a:entry", NS):
        bare_id = text(entry, "a:id").rsplit("/", 1)[-1].split("v")[0]
        papers[bare_id] = entry

lines = [
    "% related.bib for 'Read or Remembered?' -- generated by scripts/make_related_bib.py on 2026-11-13.",
    "% Titles and authors come from the arXiv API; venues were checked as noted in the script.",
    "% Every entry was opened and checked; preprints are cited as arXiv preprints.",
    "% Accented letters are written as LaTeX accents, so the file is plain ASCII.",
    "",
]
missing = []
for key, arxiv_id, booktitle, venue_year, note in VENUES:
    entry = papers.get(arxiv_id)
    if entry is None:
        missing.append(arxiv_id)
        continue
    title = ascii_latex(protect(text(entry, "a:title")))
    authors = ascii_latex(" and ".join(text(a, "a:name") for a in entry.findall("a:author", NS)))
    first_year = text(entry, "a:published")[:4]
    category = entry.find("arxiv:primary_category", NS).get("term")
    fields = [("author", authors), ("title", "{" + title + "}")]
    if booktitle:
        kind = "inproceedings"
        fields += [("booktitle", booktitle), ("year", str(venue_year))]
    else:
        kind = "misc"
        fields += [("howpublished", f"arXiv preprint arXiv:{arxiv_id}"), ("year", first_year)]
    fields += [
        ("eprint", arxiv_id),
        ("archivePrefix", "arXiv"),
        ("primaryClass", category),
        ("url", f"https://arxiv.org/abs/{arxiv_id}"),
    ]
    if note:
        fields.append(("note", protect(note)))
    lines.append(f"@{kind}{{{key},")
    lines += [f"  {name:<13} = {{{value}}}," for name, value in fields]
    lines[-1] = lines[-1].rstrip(",")
    lines += ["}", ""]

with open(sys.argv[2], "w", encoding="utf-8") as out:
    out.write("\n".join(lines) + EXTRA)

print(f"wrote {len(VENUES) - len(missing)} arXiv entries + 2 web entries to {sys.argv[2]}")
if missing:
    print("MISSING arXiv metadata for:", ", ".join(missing))
