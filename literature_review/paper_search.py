#!/usr/bin/env python3
"""
Search scholarly databases for:

    ("sign language recognition" OR "sign language translation")
    AND ("pose" OR "landmark" OR "signer-independent")

Results from all sources are merged, de-duplicated (by DOI, then by title),
re-checked against the boolean query locally, and saved to CSV.

Requirements:  pip install requests

NO KEY NEEDED (run by default):
    openalex         OpenAlex: ~250M works incl. IEEE, ACM, Springer, Elsevier; has abstracts
    semanticscholar  Semantic Scholar: ~200M papers; has abstracts
    arxiv            arXiv preprints
    acm              ACM Digital Library via Crossref (DOI prefix 10.1145)
  The "publisher" column shows where each paper was published (IEEE, ACM, Springer, Elsevier...),
  taken from its DOI, so IEEE/Springer papers are covered even without their own APIs.

Scopus / Web of Science without a key: run the search on their website (university login),
export the results, and merge them in with --import:
    Scopus:          Export -> CSV (tick "Abstract")           -> scopus.csv
    Web of Science:  Export -> Tab delimited file (Full record) -> savedrecs.txt
    python paper_search.py --import scopus.csv savedrecs.txt

OPTIONAL, with free API keys (skipped automatically when the key is not set):
    IEEE Xplore  -> https://developer.ieee.org            set IEEE_API_KEY
    Springer     -> https://dev.springernature.com        set SPRINGER_API_KEY
    Scopus       -> https://dev.elsevier.com              set SCOPUS_API_KEY
                    (works from your university network/VPN; off campus you may also need
                     SCOPUS_INSTTOKEN from the library)
    Web of Sci.  -> https://developer.clarivate.com       set WOS_API_KEY
                    (subscribe to the "Web of Science Starter API")
Optional:  CROSSREF_MAILTO / OPENALEX_MAILTO = you@example.com  (polite pools, faster)

Examples:
    python paper_search.py                                  # 2024-2026, every source it can use
    python paper_search.py --from-year 2023 --to-year 2026 --out papers.csv
    python paper_search.py --sources arxiv,acm              # only sources without keys
    python paper_search.py --sources scopus,wos             # just these two
"""
import argparse
import csv
import html
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import requests

# --------------------------------------------------------------------------------------------
# The query, defined once and translated into each database's syntax below.
# A paper matches if it contains at least one term from EVERY group.
# --------------------------------------------------------------------------------------------
GROUPS = [
    ["sign language recognition", "sign language translation"],
    ["pose", "landmark", "signer-independent"],
]

USER_AGENT = "paper-search-script/1.0 (literature review)"
TIMEOUT = 30


def boolean_query(quote='"', op_or="OR", op_and="AND", field=""):
    """Build ("a" OR "b") AND ("c" OR "d") in a database's syntax."""
    parts = []
    for group in GROUPS:
        terms = [f"{field}{quote}{t}{quote}" for t in group]
        parts.append("(" + f" {op_or} ".join(terms) + ")")
    return f" {op_and} ".join(parts)


# Local re-check. Hyphens and spaces are treated alike ("signer independent" also counts),
# and plurals count ("poses", "landmarks"), but "purpose" does not match "pose".
def _term_regex(term):
    words = re.split(r"[\s\-]+", term.lower())
    body = r"[\s\-]+".join(re.escape(w) for w in words)
    return re.compile(r"\b" + body + r"s?\b")


GROUP_REGEXES = [[(t, _term_regex(t)) for t in g] for g in GROUPS]


def matched_terms(text):
    """Return the matched terms if every group matches, else None."""
    text = (text or "").lower()
    found = []
    for group in GROUP_REGEXES:
        hits = [t for t, rx in group if rx.search(text)]
        if not hits:
            return None
        found += hits
    return found


def clean(text):
    """Strip HTML/JATS tags and collapse whitespace."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def get(url, params=None, headers=None, retries=3):
    headers = {"User-Agent": USER_AGENT, **(headers or {})}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"    retry after error: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:     # busy: wait and retry
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()                                  # other errors (401, 403...) fail at once
        return r
    raise RuntimeError(f"giving up on {url} (server busy)")


def record(source, title, authors, year, venue, doi, url, abstract):
    return {"title": clean(title), "authors": "; ".join(a for a in authors if a),
            "year": str(year or ""), "venue": clean(venue), "doi": (doi or "").strip(),
            "url": url or (f"https://doi.org/{doi}" if doi else ""),
            "source": source, "abstract": clean(abstract)}


# --------------------------------------------------------------------------------------------
# arXiv  (https://info.arxiv.org/help/api/)
# --------------------------------------------------------------------------------------------
def search_arxiv(y0, y1, limit):
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    query = boolean_query(field="all:")
    query += f" AND submittedDate:[{y0}01010000 TO {y1}12312359]"
    out, start, page = [], 0, 100
    while start < limit:
        r = get("https://export.arxiv.org/api/query",
                {"search_query": query, "start": start, "max_results": min(page, limit - start),
                 "sortBy": "submittedDate", "sortOrder": "descending"})
        root = ET.fromstring(r.content)
        entries = root.findall("a:entry", ns)
        total = int(root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults") or 0)
        for e in entries:
            doi = e.findtext("arxiv:doi", default="", namespaces=ns)
            journal = e.findtext("arxiv:journal_ref", default="", namespaces=ns)
            out.append(record(
                "arXiv", e.findtext("a:title", "", ns),
                [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
                e.findtext("a:published", "", ns)[:4], journal or "arXiv preprint", doi,
                e.findtext("a:id", "", ns), e.findtext("a:summary", "", ns)))
        start += len(entries)
        if not entries or start >= total:
            break
        time.sleep(3)          # arXiv asks for at most one request every 3 seconds
    return out


# --------------------------------------------------------------------------------------------
# IEEE Xplore  (https://developer.ieee.org/docs)
# --------------------------------------------------------------------------------------------
def search_ieee(y0, y1, limit):
    key = os.environ.get("IEEE_API_KEY")
    if not key:
        print("  skipped: set IEEE_API_KEY (free key at https://developer.ieee.org)")
        return []
    out, start, page = [], 1, 200  # 200 is the API maximum per request
    while len(out) < limit:
        r = get("https://ieeexploreapi.ieee.org/api/v1/search/articles",
                {"apikey": key, "querytext": boolean_query(), "format": "json",
                 "start_year": y0, "end_year": y1, "start_record": start,
                 "max_records": min(page, limit - len(out)),
                 "sort_field": "publication_year", "sort_order": "desc"})
        data = r.json()
        articles = data.get("articles", [])
        for a in articles:
            authors = [x.get("full_name", "") for x in a.get("authors", {}).get("authors", [])]
            out.append(record("IEEE Xplore", a.get("title"), authors, a.get("publication_year"),
                              a.get("publication_title"), a.get("doi"),
                              a.get("html_url") or a.get("pdf_url"), a.get("abstract")))
        start += len(articles)
        if not articles or start > int(data.get("total_records", 0)):
            break
        time.sleep(1)
    return out


# --------------------------------------------------------------------------------------------
# SpringerLink  (Springer Nature Meta API v2, https://dev.springernature.com)
# --------------------------------------------------------------------------------------------
def search_springer(y0, y1, limit):
    key = os.environ.get("SPRINGER_API_KEY")
    if not key:
        print("  skipped: set SPRINGER_API_KEY (free key at https://dev.springernature.com)")
        return []
    q = f"{boolean_query()} datefrom:{y0}-01-01 dateto:{y1}-12-31"
    out, start, page = [], 1, 25   # 25 per page is safe for free (basic) keys
    while len(out) < limit:
        r = get("https://api.springernature.com/meta/v2/json",
                {"q": q, "s": start, "p": min(page, limit - len(out)), "api_key": key})
        data = r.json()
        recs = data.get("records", [])
        for rec in recs:
            urls = rec.get("url") or []
            link = next((u.get("value") for u in urls if u.get("format") == "html"),
                        urls[0].get("value") if urls else "")
            out.append(record("SpringerLink", rec.get("title"),
                              [c.get("creator", "") for c in rec.get("creators", [])],
                              (rec.get("publicationDate") or "")[:4], rec.get("publicationName"),
                              rec.get("doi"), link, rec.get("abstract")))
        start += len(recs)
        total = int((data.get("result") or [{}])[0].get("total", 0))
        if not recs or start > total:
            break
        time.sleep(1)
    return out


# --------------------------------------------------------------------------------------------
# ACM Digital Library  (via Crossref, ACM DOI prefix 10.1145)
# Crossref has no boolean search, so it runs one keyword search per term combination and the
# local boolean check later removes the loose matches.
# --------------------------------------------------------------------------------------------
def search_acm(y0, y1, limit):
    mailto = os.environ.get("CROSSREF_MAILTO")
    out = []
    combos = [(a, b) for a in GROUPS[0] for b in GROUPS[1]]
    for a, b in combos:
        cursor, got = "*", 0
        while got < limit:
            params = {"query.bibliographic": f"{a} {b}", "rows": 100, "cursor": cursor,
                      "filter": f"prefix:10.1145,from-pub-date:{y0}-01-01,until-pub-date:{y1}-12-31",
                      "select": "DOI,title,author,issued,container-title,abstract,URL"}
            if mailto:
                params["mailto"] = mailto
            msg = get("https://api.crossref.org/works", params).json()["message"]
            items = msg.get("items", [])
            for it in items:
                authors = [f"{x.get('given', '')} {x.get('family', '')}".strip()
                           for x in it.get("author", [])]
                year = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
                doi = it.get("DOI", "")
                out.append(record("ACM Digital Library", (it.get("title") or [""])[0], authors,
                                  year, (it.get("container-title") or [""])[0], doi,
                                  f"https://dl.acm.org/doi/{doi}", it.get("abstract")))
            got += len(items)
            # Crossref ranks by relevance; after a few hundred hits the results stop being
            # about the topic, so stop at the first page with no local match.
            page_hits = sum(1 for it in out[-len(items):]
                            if matched_terms(f"{it['title']} {it['abstract']}")) if items else 0
            if not items or page_hits == 0 or len(items) < 100:
                break
            cursor = msg.get("next-cursor")
            time.sleep(1)
    return out


# --------------------------------------------------------------------------------------------
# Scopus  (Elsevier Scopus Search API, https://dev.elsevier.com)
# Needs institutional access: run it on the university network or VPN, or set SCOPUS_INSTTOKEN.
# The COMPLETE view (with abstracts and all authors) needs a subscription; if it is refused,
# the script falls back to the STANDARD view (first author, no abstract).
# --------------------------------------------------------------------------------------------
def search_scopus(y0, y1, limit):
    key = os.environ.get("SCOPUS_API_KEY")
    if not key:
        print("  skipped: set SCOPUS_API_KEY (free key at https://dev.elsevier.com)")
        return []
    headers = {"X-ELS-APIKey": key, "Accept": "application/json"}
    if os.environ.get("SCOPUS_INSTTOKEN"):
        headers["X-ELS-Insttoken"] = os.environ["SCOPUS_INSTTOKEN"]
    query = f"TITLE-ABS-KEY({boolean_query()}) AND PUBYEAR > {y0 - 1} AND PUBYEAR < {y1 + 1}"
    url = "https://api.elsevier.com/content/search/scopus"
    view, out, start = "COMPLETE", [], 0
    while len(out) < limit:
        params = {"query": query, "start": start, "count": 25, "view": view}
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT, **headers},
                         timeout=TIMEOUT)
        if r.status_code in (401, 403) and view == "COMPLETE":
            print("  COMPLETE view not allowed with this key, using STANDARD (no abstracts)")
            view = "STANDARD"
            continue
        if r.status_code in (401, 403):
            raise RuntimeError(f"Scopus refused the request ({r.status_code}). Run it on the "
                               "university network/VPN, or ask the library for an insttoken.")
        r.raise_for_status()
        res = r.json().get("search-results", {})
        entries = [e for e in res.get("entry", []) if "error" not in e]
        for e in entries:
            if e.get("author"):
                authors = [a.get("authname", "") for a in e["author"]]
            else:
                authors = [e.get("dc:creator", "")]
            link = next((l.get("@href") for l in e.get("link", []) if l.get("@ref") == "scopus"), "")
            out.append(record("Scopus", e.get("dc:title"), authors,
                              (e.get("prism:coverDate") or "")[:4], e.get("prism:publicationName"),
                              e.get("prism:doi"), link, e.get("dc:description")))
        start += len(entries)
        if not entries or start >= int(res.get("opensearch:totalResults", 0) or 0):
            break
        time.sleep(1)
    return out


# --------------------------------------------------------------------------------------------
# Web of Science  (Clarivate Web of Science Starter API, https://developer.clarivate.com)
# TS= searches title, abstract and keywords. The Starter API returns no abstracts.
# --------------------------------------------------------------------------------------------
def search_wos(y0, y1, limit):
    key = os.environ.get("WOS_API_KEY")
    if not key:
        print("  skipped: set WOS_API_KEY (Web of Science Starter API at https://developer.clarivate.com)")
        return []
    query = f"TS=({boolean_query()}) AND PY=({y0}-{y1})"
    out, page = [], 1
    while len(out) < limit:
        r = get("https://api.clarivate.com/apis/wos-starter/v1/documents",
                {"q": query, "db": "WOS", "limit": 50, "page": page},
                headers={"X-ApiKey": key})
        data = r.json()
        hits = data.get("hits", [])
        for h in hits:
            src = h.get("source", {})
            authors = [a.get("displayName") or a.get("wosStandard", "")
                       for a in h.get("names", {}).get("authors", [])]
            out.append(record("Web of Science", h.get("title"), authors, src.get("publishYear"),
                              src.get("sourceTitle"), h.get("identifiers", {}).get("doi"),
                              h.get("links", {}).get("record"), ""))
        total = int(data.get("metadata", {}).get("total", 0))
        if not hits or page * 50 >= total:
            break
        page += 1
        time.sleep(1)
    return out[:limit]


# --------------------------------------------------------------------------------------------
# OpenAlex  (https://help.openalex.org/api/) -- no key needed
# Boolean query on title, abstract and keywords. Abstracts come as an inverted index.
# --------------------------------------------------------------------------------------------
def _openalex_abstract(inv):
    if not inv:
        return ""
    words = {}
    for word, positions in inv.items():
        for i in positions:
            words[i] = word
    return " ".join(words[i] for i in sorted(words))


def search_openalex(y0, y1, limit):
    out, cursor = [], "*"
    while cursor and len(out) < limit:
        params = {"search": boolean_query(), "filter": f"publication_year:{y0}-{y1}",
                  "per_page": 200, "cursor": cursor,
                  "select": "doi,title,publication_year,authorships,primary_location,"
                            "abstract_inverted_index,id"}
        if os.environ.get("OPENALEX_MAILTO"):
            params["mailto"] = os.environ["OPENALEX_MAILTO"]
        if os.environ.get("OPENALEX_API_KEY"):
            params["api_key"] = os.environ["OPENALEX_API_KEY"]
        try:
            data = get("https://api.openalex.org/works", params).json()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403) and out:
                print(f"  OpenAlex stopped after {len(out)} results: paging further needs a free key "
                      "(https://openalex.org -> account -> API key, then set OPENALEX_API_KEY)")
                break
            raise
        results = data.get("results", [])
        for w in results:
            loc = w.get("primary_location") or {}
            src = loc.get("source") or {}
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            out.append(record("OpenAlex", w.get("title"),
                              [a.get("author", {}).get("display_name", "") for a in w.get("authorships", [])],
                              w.get("publication_year"), src.get("display_name"), doi,
                              loc.get("landing_page_url") or w.get("id"),
                              _openalex_abstract(w.get("abstract_inverted_index"))))
        cursor = data.get("meta", {}).get("next_cursor")
        if not results:
            break
        time.sleep(0.2)
    return out[:limit]


# --------------------------------------------------------------------------------------------
# Semantic Scholar  (https://api.semanticscholar.org) -- no key needed (shared rate limit,
# so it may answer "429 too many requests"; the script waits and retries)
# Bulk-search syntax: | = OR, + = AND, "..." = phrase.
# --------------------------------------------------------------------------------------------
def search_semanticscholar(y0, y1, limit):
    groups = ["(" + " | ".join(f'"{t}"' for t in g) + ")" for g in GROUPS]
    query = " + ".join(groups)
    headers = {}
    if os.environ.get("S2_API_KEY"):
        headers["x-api-key"] = os.environ["S2_API_KEY"]
    out, token = [], None
    while len(out) < limit:
        params = {"query": query, "year": f"{y0}-{y1}",
                  "fields": "title,abstract,authors,year,venue,externalIds,url"}
        if token:
            params["token"] = token
        data = get("https://api.semanticscholar.org/graph/v1/paper/search/bulk", params, headers,
                   retries=6).json()
        papers = data.get("data") or []
        for x in papers:
            ids = x.get("externalIds") or {}
            doi = ids.get("DOI") or (f"10.48550/arXiv.{ids['ArXiv']}" if ids.get("ArXiv") else "")
            out.append(record("Semantic Scholar", x.get("title"),
                              [a.get("name", "") for a in x.get("authors") or []],
                              x.get("year"), x.get("venue"), doi, x.get("url"), x.get("abstract")))
        token = data.get("token")
        if not papers or not token:
            break
        time.sleep(1.5)
    return out[:limit]


# --------------------------------------------------------------------------------------------
# Files exported by hand from the Scopus / Web of Science websites (--import)
# --------------------------------------------------------------------------------------------
def import_export_file(path):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        head = f.readline()
    out = []
    if head.startswith("PT\t") or "\tTI\t" in head:            # Web of Science tab-delimited
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                doi = row.get("DI", "")
                out.append(record("Web of Science (export)", row.get("TI"),
                                  (row.get("AF") or row.get("AU") or "").split("; "),
                                  row.get("PY"), row.get("SO"), doi,
                                  f"https://doi.org/{doi}" if doi else "", row.get("AB")))
    else:                                                        # Scopus CSV
        with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                authors = row.get("Author full names") or row.get("Authors") or ""
                out.append(record("Scopus (export)", row.get("Title"),
                                  [a.strip() for a in authors.split(";")], row.get("Year"),
                                  row.get("Source title"), row.get("DOI"), row.get("Link"),
                                  row.get("Abstract")))
    return out


# Publisher from the DOI prefix, so the CSV shows which library each paper belongs to.
DOI_PUBLISHERS = {"10.1109": "IEEE", "10.1145": "ACM", "10.1007": "Springer", "10.1016": "Elsevier",
                  "10.48550": "arXiv", "10.3390": "MDPI", "10.1038": "Nature", "10.1002": "Wiley",
                  "10.1080": "Taylor & Francis", "10.3233": "IOS Press", "10.1162": "MIT Press",
                  "10.18653": "ACL Anthology", "10.1371": "PLOS", "10.21437": "ISCA",
                  "10.1117": "SPIE", "10.1088": "IOP"}


# Kept by default. Everything else (small or predatory journals, unknown DOI prefixes, and MDPI,
# whose quality is debated) is dropped unless --all-publishers is given.
REPUTABLE = {"IEEE", "ACM", "Springer", "Elsevier", "ACL Anthology", "Nature", "Wiley",
             "Taylor & Francis", "MIT Press", "ISCA", "arXiv"}


def publisher_of(r):
    prefix = r["doi"].split("/")[0] if r["doi"] else ""
    if prefix in DOI_PUBLISHERS:
        return DOI_PUBLISHERS[prefix]
    if "arxiv.org" in r["url"]:
        return "arXiv"
    return prefix or ""

SOURCES = {"openalex": search_openalex, "semanticscholar": search_semanticscholar,
           "arxiv": search_arxiv, "ieee": search_ieee, "springer": search_springer, "acm": search_acm,
           "scopus": search_scopus, "wos": search_wos}

# Databases that apply the boolean query to title + abstract on their side. A record from one
# of them that comes back without an abstract is kept, because the local re-check could only
# look at its title. Everything else (and every record that has an abstract) is re-checked.
SERVER_CHECKED = {"Scopus", "Web of Science", "Scopus (export)", "Web of Science (export)"}


# --------------------------------------------------------------------------------------------
def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def merge(records):
    """De-duplicate by DOI, then by normalised title; keep the record with more information."""
    merged, by_key = [], {}
    for r in records:
        keys = [k for k in (("doi", r["doi"].lower()) if r["doi"] else None,
                            ("title", norm_title(r["title"]))) if k and k[1]]
        existing = next((by_key[k] for k in keys if k in by_key), None)
        if existing is None:
            merged.append(r)
            existing = r
        else:
            sources = set(existing["source"].split("; ")) | {r["source"]}
            existing["source"] = "; ".join(sorted(sources))
            for field in ("abstract", "doi", "venue", "authors", "year"):
                if len(r[field]) > len(existing[field]):
                    existing[field] = r[field]
        for k in keys:
            by_key[k] = existing
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-year", type=int, default=2024)
    p.add_argument("--to-year", type=int, default=2026)
    p.add_argument("--max-per-source", type=int, default=2000)
    p.add_argument("--sources", default="openalex,semanticscholar,arxiv,acm,ieee,springer,scopus,wos",
                   help="comma-separated: " + ",".join(SOURCES))
    p.add_argument("--out", default="papers.csv")
    p.add_argument("--import", dest="imports", nargs="*", default=[],
                   help="Scopus CSV / Web of Science tab-delimited files exported from their websites")
    p.add_argument("--all-publishers", action="store_true",
                   help="keep papers from every publisher, not only: " + ", ".join(sorted(REPUTABLE)))
    p.add_argument("--keep-unmatched", action="store_true",
                   help="also keep results that fail the local boolean check (marked in a column)")
    args = p.parse_args()

    print("Query:", boolean_query())
    print(f"Years: {args.from_year}-{args.to_year}\n")
    everything = []
    for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
        if name not in SOURCES:
            print(f"Unknown source '{name}', skipping")
            continue
        print(f"Searching {name} ...")
        try:
            found = SOURCES[name](args.from_year, args.to_year, args.max_per_source)
        except Exception as e:  # keep going with the other databases
            print(f"  FAILED: {e}")
            continue
        print(f"  {len(found)} raw results")
        everything += found

    for path in args.imports:
        print(f"Importing {path} ...")
        found = import_export_file(path)
        print(f"  {len(found)} records")
        everything += found

    merged = merge(everything)
    rows, dropped_publisher = [], 0
    for r in merged:
        terms = matched_terms(f"{r['title']} {r['abstract']}")
        year_ok = r["year"].isdigit() and args.from_year <= int(r["year"]) <= args.to_year
        trusted = not r["abstract"] and bool(SERVER_CHECKED & set(r["source"].split("; ")))
        r["matched_terms"] = (", ".join(dict.fromkeys(terms)) if terms
                              else "(matched by database; no abstract to re-check)" if trusted else "")
        r["passes_query"] = "yes" if (terms or trusted) and year_ok else "no"
        r["publisher"] = publisher_of(r)
        if r["publisher"] not in REPUTABLE and not args.all_publishers:
            dropped_publisher += r["passes_query"] == "yes"
            continue
        if r["passes_query"] == "yes" or args.keep_unmatched:
            rows.append(r)
    rows.sort(key=lambda r: (r["year"], r["title"]), reverse=True)

    fields = ["title", "authors", "year", "venue", "publisher", "doi", "url", "source",
              "matched_terms", "passes_query", "abstract"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(everything)} raw -> {len(merged)} after de-duplication "
          f"-> {sum(r['passes_query'] == 'yes' for r in rows)} match the query")
    counts = {}
    for r in rows:
        if r["passes_query"] == "yes":
            for s in r["source"].split("; "):
                counts[s] = counts.get(s, 0) + 1
    for s, c in sorted(counts.items()):
        print(f"  {s}: {c}")
    pubs = {}
    for r in rows:
        if r["passes_query"] == "yes":
            pubs[r["publisher"] or "unknown"] = pubs.get(r["publisher"] or "unknown", 0) + 1
    print("By publisher: " + ", ".join(f"{k} {v}" for k, v in sorted(pubs.items(), key=lambda x: -x[1])))
    if dropped_publisher:
        print(f"Dropped {dropped_publisher} matching papers from other publishers (use --all-publishers to keep them)")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
