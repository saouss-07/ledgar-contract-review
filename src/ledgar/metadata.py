"""Subgroup attributes for the audit, extracted from the clause text with regular expressions.

LEDGAR (LexGLUE version) only gives `text` and `label`: no contract type, no party information. To audit
the models "across the kinds of contracts and parties" (project brief), we derive proxy attributes from the
words of each clause. They are heuristics, computed on a single clause: "inconnu" / "aucune" is a frequent
and honest answer, and the audit only reports groups with enough examples.

Usage:
    from ledgar.metadata import add_metadata
    df = add_metadata(ds["test"].to_pandas())     # adds the columns listed in METADATA_COLUMNS

    python -m ledgar.metadata                     # coverage of every attribute on LEDGAR
"""
import re

import pandas as pd

METADATA_COLUMNS = ["jurisdiction", "jurisdiction_group", "party_type", "contract_type", "gender",
                    "has_amount", "has_duration", "all_caps", "redacted", "n_words", "length_group"]

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas",
    "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
]
FOREIGN = [
    "England and Wales", "England", "United Kingdom", "Scotland", "Ireland", "Canada", "Ontario",
    "British Columbia", "Quebec", "Cayman Islands", "Bermuda", "British Virgin Islands", "Hong Kong",
    "Singapore", "People's Republic of China", "China", "Japan", "Israel", "Germany", "France",
    "the Netherlands", "Netherlands", "Luxembourg", "Switzerland", "Australia", "India", "Mexico", "Brazil",
]
TOP_STATES = ["Delaware", "New York", "California", "Texas"]   # kept as their own audit group

# Longest names first, so that "West Virginia" wins over "Virginia" and "England and Wales" over "England"
_PLACES = sorted(US_STATES + FOREIGN, key=len, reverse=True)
_PLACE_ALT = "|".join(re.escape(p) for p in _PLACES)
_JURISDICTION_RE = re.compile(
    rf"\b(?:laws?|state|commonwealth|province|courts?|jurisdiction)\s+of\s+(?:the\s+)?"
    rf"(?:state\s+of\s+|commonwealth\s+of\s+|province\s+of\s+)?({_PLACE_ALT})\b"
    rf"|\b({_PLACE_ALT})\s+(?:general\s+corporation\s+)?(?:law|courts?)\b",
    re.IGNORECASE,
)
_CANONICAL = {p.lower(): p for p in _PLACES}

# Contract families: the one with the most keyword hits wins; no hit -> "inconnu"
CONTRACT_KEYWORDS = {
    "emploi": r"employee|executive|employer|employment|salary|bonus|severance",
    "financement": r"lender|borrower|loan|credit agreement|administrative agent|collateral agent|guarantor|notes?\b",
    "acquisition": r"purchaser|buyer|seller|merger|stockholder|shareholder|acquisition|closing date",
    "bail": r"landlord|tenant|lessor|lessee|premises|lease",
    "licence": r"licensor|licensee|license|royalt",
    "services": r"supplier|vendor|customer|contractor|consultant|services",
}
_CONTRACT_RES = {k: re.compile(rf"\b(?:{v})", re.IGNORECASE) for k, v in CONTRACT_KEYWORDS.items()}

_PERSON_RE = re.compile(r"\b(?:employee|executive|individual|consultant|participant|grantee|optionee|director)\b",
                        re.IGNORECASE)
_FINANCE_RE = re.compile(r"\b(?:bank|lenders?|administrative agent|collateral agent|trustee|n\.a\.)", re.IGNORECASE)
_COMPANY_RE = re.compile(r"\b(?:company|corporation|corp\.|inc\.|llc|l\.l\.c\.|ltd\.?|l\.p\.|plc|limited)\b",
                         re.IGNORECASE)
_MASC_RE = re.compile(r"\b(?:he|him|his|himself)\b", re.IGNORECASE)
_FEM_RE = re.compile(r"\b(?:she|her|hers|herself)\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"[$€£]\s?\d|\b\d[\d,.]*\s?(?:dollars|usd|euros?)\b", re.IGNORECASE)

_NUMBER_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|"
                 "forty|forty-five|fifty|sixty|ninety|one hundred twenty|one hundred eighty")
# "30 days", "thirty days", "thirty (30) days", "ninety (90) calendar days", "six months"
DURATION_RE = re.compile(
    rf"\b(?:(?:{_NUMBER_WORDS})\s*\(\s*(\d+)\s*\)|(\d+)|({_NUMBER_WORDS}))\s*"
    rf"(?:calendar\s+|business\s+|consecutive\s+)?(days?|months?|years?|weeks?)\b",
    re.IGNORECASE,
)
_REDACTED_RE = re.compile(r"\[\s*\*+\s*\]")


def find_jurisdiction(text):
    """First governing place named in the clause ("Delaware", "England and Wales", ...), or None."""
    m = _JURISDICTION_RE.search(text)
    if m is None:
        return None
    return _CANONICAL[(m.group(1) or m.group(2)).lower()]


def jurisdiction_group(place):
    if place is None or pd.isna(place):   # Series.map passes NaN, not None, for clauses without a place
        return "aucune"
    if place in TOP_STATES:
        return place
    return "autre État US" if place in US_STATES else "hors US"


def find_durations(text):
    """List of (number_or_word, unit) found in the text, e.g. [("30", "days")]. Shared with the hybrid search."""
    return [(m.group(1) or m.group(2) or m.group(3).lower(), m.group(4).lower()) for m in DURATION_RE.finditer(text)]


def party_type(text):
    """Who the clause is about, by priority: a person, a financial institution, a company."""
    if _PERSON_RE.search(text) or _MASC_RE.search(text) or _FEM_RE.search(text):
        return "personne physique"
    if _FINANCE_RE.search(text):
        return "institution financière"
    if _COMPANY_RE.search(text):
        return "société"
    return "non précisé"


def contract_type(text):
    hits = {k: len(r.findall(text)) for k, r in _CONTRACT_RES.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else "inconnu"


def gender(text):
    masc, fem = bool(_MASC_RE.search(text)), bool(_FEM_RE.search(text))
    return {(True, True): "mixte", (True, False): "masculin", (False, True): "féminin"}.get((masc, fem), "aucun")


def all_caps(text):
    letters = [c for c in text if c.isalpha()]
    return len(letters) > 0 and sum(c.isupper() for c in letters) / len(letters) > 0.7


def add_metadata(df, text_col="text", length_bins=(0, 50, 150, 10**9)):
    """Return a copy of df with the METADATA_COLUMNS added (df must have a text column).

    length_group uses fixed word-count bins (court < 50 words <= moyen < 150 words <= long), the same on every
    split so that groups are comparable between validation and test.
    """
    out = df.copy()
    texts = out[text_col]
    out["jurisdiction"] = texts.map(find_jurisdiction)
    out["jurisdiction_group"] = out["jurisdiction"].map(jurisdiction_group)
    out["party_type"] = texts.map(party_type)
    out["contract_type"] = texts.map(contract_type)
    out["gender"] = texts.map(gender)
    out["has_amount"] = texts.str.contains(_AMOUNT_RE)
    out["has_duration"] = texts.map(lambda t: DURATION_RE.search(t) is not None)
    out["all_caps"] = texts.map(all_caps)
    out["redacted"] = texts.str.contains(_REDACTED_RE)
    out["n_words"] = texts.str.split().str.len()
    out["length_group"] = pd.cut(out["n_words"], bins=list(length_bins), labels=["court", "moyen", "long"],
                                 right=False)
    return out


if __name__ == "__main__":
    from pathlib import Path

    from ledgar.data import load_ledgar

    ds, _ = load_ledgar()
    test = add_metadata(ds["test"].to_pandas())
    rows = []
    for col in ["jurisdiction_group", "party_type", "contract_type", "gender", "length_group",
                "has_amount", "has_duration", "all_caps", "redacted"]:
        for value, n in test[col].value_counts().items():
            rows.append({"attribut": col, "valeur": value, "clauses_test": n, "part": round(n / len(test), 4)})
    coverage = pd.DataFrame(rows)
    print(coverage.to_string(index=False))
    out = Path(__file__).resolve().parents[2] / "results" / "tables" / "metadata_coverage.csv"
    coverage.to_csv(out, index=False)
    print("written", out)
