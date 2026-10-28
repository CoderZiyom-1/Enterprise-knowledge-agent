"""
Rules written in plain code (not left to the LLM):
  - how much we trust a source
  - which citations the answer really used
  - the final confidence level
"""

import re
from urllib.parse import urlparse

AGGREGATORS = [
    "tracxn.com", "cbinsights.com", "owler.com", "crunchbase.com", "zoominfo.com",
    "craft.co", "growjo.com", "dnb.com", "comparably.com", "glassdoor.com",
    "ambitionbox.com", "pitchbook.com", "rocketreach.co", "leadiq.com",
    "globaldata.com", "tofler.in", "zaubacorp.com", "mapquest.com",
]


REPUTABLE = [
    "reuters.com", "wsj.com", "ft.com", "forbes.com", "techcrunch.com",
    "economictimes.indiatimes.com", "livemint.com", "business-standard.com",
    "moneycontrol.com", "thehindu.com", "businesswire.com", "prnewswire.com",
    "wikipedia.org", "sec.gov", "bloomberg.com", "cnbc.com",
]

ORDER = ["Low", "Medium", "High"]


def get_domain(url):
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def trust_tier(url, question):
    """Returns 'official', 'reputable', 'other' or 'aggregator'."""
    domain = get_domain(url)

    # company profile pages are aggregator style even on big sites
    if "/profile/" in url:
        return "aggregator"

    for site in AGGREGATORS:
        if domain == site or domain.endswith("." + site):
            return "aggregator"

   
    q = question.lower().replace(" ", "")
    for part in domain.split("."):
        if len(part) > 3 and part in q:
            return "official"

    for site in REPUTABLE:
        if domain == site or domain.endswith("." + site):
            return "reputable"
    if domain.endswith(".gov") or domain.endswith(".edu") or ".gov." in domain or ".ac." in domain:
        return "reputable"

    return "other"


def trust_score(url, question):
    tier = trust_tier(url, question)
    if tier == "official":
        return 3
    if tier == "reputable":
        return 2
    if tier == "other":
        return 1
    return 0


def find_citations(answer_text, max_number):
    """Find numbers like [1] or [2][3] or [1, 2] in the answer."""
    numbers = []
    groups = re.findall(r"\[([\d,\s]+)\]", answer_text)
    for g in groups:
        for piece in g.split(","):
            piece = piece.strip()
            if piece.isdigit():
                n = int(piece)
                if n >= 1 and n <= max_number and n not in numbers:
                    numbers.append(n)
    numbers.sort()
    return numbers


def remove_double_citations(text):
    """[1][1] becomes [1]"""
    return re.sub(r"(\[\d+\])\1+", r"\1", text)


def lower_level(a, b):
    if ORDER.index(a) <= ORDER.index(b):
        return a
    return b


def fix_confidence(model_level, cited_urls, question, has_conflicts):
    """The LLM guesses a confidence. This function can only lower it, never raise it."""
    if model_level not in ORDER:
        model_level = "Low"

    if len(cited_urls) == 0:
        return "Low", "the answer has no citations"

    domains = []
    tiers = []
    for url in cited_urls:
        d = get_domain(url)
        if d not in domains:
            domains.append(d)
        tiers.append(trust_tier(url, question))

    cap = "High"
    notes = []

    if len(domains) < 3:
        cap = "Medium"
        notes.append("only " + str(len(domains)) + " different website(s) cited")
    if len(domains) < 2:
        cap = "Low"

    if "official" not in tiers and "reputable" not in tiers:
        cap = lower_level(cap, "Medium")
        notes.append("no official or reputable source cited")

    
    all_aggregators = True
    for t in tiers:
        if t != "aggregator":
            all_aggregators = False
    if all_aggregators:
        cap = "Low"
        notes.append("all cited sources are data-aggregator sites that copy each other")

    if has_conflicts:
        cap = lower_level(cap, "Medium")
        notes.append("sources disagree on some points")

    final = lower_level(model_level, cap)
    if final != model_level:
        note = "lowered from " + model_level + " because " + "; ".join(notes)
    else:
        note = ""
    return final, note
