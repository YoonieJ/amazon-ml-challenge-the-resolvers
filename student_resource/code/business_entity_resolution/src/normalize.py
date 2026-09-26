"""Business name/address normalization: NFKC, lowercasing, legal-suffix
expansion, punctuation stripping, tokenization. No external transliteration
services -- see plan-v1.md section 8 for why (open question, treated as
excluded by default).
"""
import re
import unicodedata

# ponytail: hand-picked from abbreviations actually observed in the data
# (plan-v1.md section 3.1). Extend empirically as error analysis surfaces
# more, not preemptively.
_ABBREVIATIONS = {
    "corp": "corporation",
    "ltd": "limited",
    "pvt": "private",
    "inc": "incorporated",
    "co": "company",
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
}

_ABBR_RE = re.compile(r"\b(" + "|".join(_ABBREVIATIONS) + r")\.?\b")
_WS_RE = re.compile(r"\s+")


def _strip_punct(text: str) -> str:
    # str.isalnum()/\w miss combining marks (category Mn/Mc, e.g. Devanagari
    # matras), which would otherwise get stripped as "punctuation" and
    # corrupt non-Latin scripts.
    return "".join(
        ch if ch.isalnum() or ch.isspace() or unicodedata.category(ch)[0] == "M" else " "
        for ch in text
    )


def normalize_text(text) -> str:
    """Lowercase, NFKC-normalize, expand legal-suffix abbreviations, strip punctuation.

    Non-string input (NaN from pandas, None) normalizes to "" -- about 3% of
    source2/source3 business_address values are blank/NaN in the real data.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("&", " and ")
    text = _ABBR_RE.sub(lambda m: _ABBREVIATIONS[m.group(1)], text)
    text = _strip_punct(text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text) -> list[str]:
    """Normalize then split into whitespace-separated tokens."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def demo() -> None:
    """Sanity check against real noisy rows pulled from dataset/train (plan-v1.md build order step 1)."""
    cases = [
        # (raw, expected) -- pulled from train_source1.tsv / train_source2.tsv
        ("B+ Retail Inc", "b retail incorporated"),
        ("Shree Infracon Private Ltd", "shree infracon private limited"),
        ("Callicoat & Dailey Inc", "callicoat and dailey incorporated"),
        ("2062  SMITH HOLLOW RD", "2062 smith hollow road"),
        ("", ""),
        (float("nan"), ""),  # blank/NaN address, ~3% of source2/source3 rows
    ]
    for raw, expected in cases:
        got = normalize_text(raw)
        assert got == expected, f"{raw!r} -> {got!r}, expected {expected!r}"

    # Script mismatch (plan-v1.md section 2): Devanagari must survive intact,
    # we don't transliterate -- cross-script matching is an address/country
    # fallback, not a normalization job.
    devanagari = "राम मार्केटिंग प्राइवेट लिमिटेड"
    assert tokenize(devanagari) == ["राम", "मार्केटिंग", "प्राइवेट", "लिमिटेड"]

    print("normalize.py demo: all checks passed")


if __name__ == "__main__":
    demo()
