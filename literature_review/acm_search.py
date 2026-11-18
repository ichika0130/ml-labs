#!/usr/bin/env python3
"""
ACM Digital Library search, no API key needed.

ACM has no public search API, and its Crossref records rarely include abstracts, so a
title-only check finds very few matches. This script:

  1. collects ACM papers (DOI prefix 10.1145) about sign language from three free sources:
     Crossref, DBLP and Semantic Scholar, using the broad query "sign language";
  2. fills in missing abstracts by looking each DOI up in Semantic Scholar;
  3. applies the boolean query to title + abstract:
        ("sign language recognition" OR "sign language translation")
        AND ("pose" OR "landmark" OR "signer-independent")

Output CSV, one row per paper, with a "status" column:
    match                    title + abstract satisfy the query
    check manually           no abstract found anywhere; the title is about sign language
                             but the second keyword group can't be checked -> open the link
    (non-matching papers are dropped unless --keep-all is given)

Usage (from the ML folder):
    .venv/bin/python literature_review/acm_search.py --out literature_review/acm_papers.csv
"""
import argparse
import csv
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_search import get, matched_terms, merge, record  # noqa: E402

ACM_PREFIX = "10.1145/"
BROAD = "sign language"


def is_acm(doi):
    return (doi or "").lower().startswith(ACM_PREFIX)


def about_sign_language(title, abstract=""):
    return "sign language" in f"{title} {abstract}".lower()


# ---- 1a. Crossref: ACM records, relevance-ranked ----------------------------------------------
def from_crossref(y0, y1):
    out, cursor = [], "*"
    mailto = os.environ.get("CROSSREF_MAILTO")
    while cursor:
        params = {"query.bibliographic": BROAD, "rows": 100, "cursor": cursor,
                  "filter": f"prefix:10.1145,from-pub-date:{y0}-01-01,until-pub-date:{y1}-12-31",
                  "select": "DOI,title,author,issued,container-title,abstract"}
        if mailto:
            params["mailto"] = mailto
        msg = get("https://api.crossref.org/works", params).json()["message"]
        items = msg.get("items", [])
        page = []
        for it in items:
            title = (it.get("title") or [""])[0]
            authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in it.get("author", [])]
            year = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
            doi = it.get("DOI", "")
            page.append(record("Crossref", title, authors, year, (it.get("container-title") or [""])[0],
                               doi, f"https://dl.acm.org/doi/{doi}", it.get("abstract")))
        relevant = [r for r in page if about_sign_language(r["title"], r["abstract"])]
        out += relevant
        # results are ranked by relevance: stop once a whole page is off-topic
        if not items or not relevant:
            break
        cursor = msg.get("next-cursor")
        time.sleep(1)
    return out


# ---- 1b. DBLP: computer-science bibliography, title search -------------------------------------
def from_dblp(y0, y1):
    out, first = [], 0
    while True:
        data = get("https://dblp.org/search/publ/api",
                   {"q": BROAD, "format": "json", "h": 1000, "f": first}).json()
        hits = data.get("result", {}).get("hits", {})
        items = hits.get("hit", []) or []
        for h in items:
            info = h.get("info", {})
            doi = info.get("doi", "")
            year = int(info.get("year", 0) or 0)
            if not is_acm(doi) or not (y0 <= year <= y1):
                continue
            a = info.get("authors", {}).get("author", [])
            a = a if isinstance(a, list) else [a]
            authors = [x.get("text", "") if isinstance(x, dict) else str(x) for x in a]
            out.append(record("DBLP", info.get("title", "").rstrip("."), authors, year,
                              info.get("venue", ""), doi, f"https://dl.acm.org/doi/{doi}", ""))
        first += len(items)
        if not items or first >= int(hits.get("@total", 0)):
            break
        time.sleep(1)
    return out


# ---- 1c. Semantic Scholar: bulk search, keep ACM DOIs -------------------------------------------
def from_semanticscholar(y0, y1):
    out, token = [], None
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}
    while True:
        params = {"query": f'"{BROAD}"', "year": f"{y0}-{y1}",
                  "fields": "title,abstract,authors,year,venue,externalIds,url"}
        if token:
            params["token"] = token
        data = get("https://api.semanticscholar.org/graph/v1/paper/search/bulk", params, headers,
                   retries=6).json()
        for x in data.get("data") or []:
            doi = (x.get("externalIds") or {}).get("DOI", "")
            if is_acm(doi):
                out.append(record("Semantic Scholar", x.get("title"),
                                  [a.get("name", "") for a in x.get("authors") or []], x.get("year"),
                                  x.get("venue"), doi, f"https://dl.acm.org/doi/{doi}", x.get("abstract")))
        token = data.get("token")
        if not token or not data.get("data"):
            break
        time.sleep(1.5)
    return out


# ---- 2. fill in missing abstracts by DOI ----------------------------------------------------------
def fill_abstracts(rows):
    missing = [r for r in rows if not r["abstract"]]
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}
    filled = 0
    for i in range(0, len(missing), 400):
        chunk = missing[i:i + 400]
        for attempt in range(6):
            r = requests.post("https://api.semanticscholar.org/graph/v1/paper/batch",
                              params={"fields": "abstract,venue"},
                              json={"ids": [f"DOI:{x['doi']}" for x in chunk]},
                              headers={"User-Agent": "acm-search-script/1.0", **headers}, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            for row, paper in zip(chunk, r.json()):
                if paper and paper.get("abstract"):
                    row["abstract"] = " ".join(paper["abstract"].split())
                    filled += 1
                if paper and paper.get("venue") and not row["venue"]:
                    row["venue"] = paper["venue"]
            break
        time.sleep(1.5)
    return filled, len(missing)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-year", type=int, default=2024)
    p.add_argument("--to-year", type=int, default=2026)
    p.add_argument("--out", default="acm_papers.csv")
    p.add_argument("--keep-all", action="store_true", help="also keep sign-language papers that fail the query")
    args = p.parse_args()

    found = []
    for name, fn in (("Crossref", from_crossref), ("DBLP", from_dblp), ("Semantic Scholar", from_semanticscholar)):
        print(f"Collecting ACM sign-language papers from {name} ...")
        try:
            got = fn(args.from_year, args.to_year)
            print(f"  {len(got)} papers")
            found += got
        except Exception as e:  # keep going with the other sources
            print(f"  FAILED: {e}")

    papers = [r for r in merge(found) if is_acm(r["doi"])]
    print(f"\n{len(papers)} unique ACM papers about sign language")
    print("Filling in missing abstracts from Semantic Scholar ...")
    try:
        filled, missing = fill_abstracts(papers)
        print(f"  found {filled} of {missing} missing abstracts")
    except Exception as e:
        print(f"  FAILED: {e}")

    rows = []
    for r in papers:
        if not (r["year"].isdigit() and args.from_year <= int(r["year"]) <= args.to_year):
            continue
        terms = matched_terms(f"{r['title']} {r['abstract']}")
        if terms:
            r["status"] = "match"
        elif not r["abstract"] and about_sign_language(r["title"]):
            r["status"] = "check manually (no abstract)"
        elif args.keep_all:
            r["status"] = "no match"
        else:
            continue
        r["matched_terms"] = ", ".join(dict.fromkeys(terms)) if terms else ""
        rows.append(r)
    order = {"match": 0, "check manually (no abstract)": 1, "no match": 2}
    rows.sort(key=lambda r: (order[r["status"]], -int(r["year"]), r["title"]))

    fields = ["status", "title", "authors", "year", "venue", "doi", "url", "source", "matched_terms", "abstract"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n_match = sum(r["status"] == "match" for r in rows)
    n_check = sum(r["status"].startswith("check") for r in rows)
    print(f"\n{n_match} match the query, {n_check} need a manual check -> saved to {args.out}")


if __name__ == "__main__":
    main()
