"""Tokenization used for BOTH document chunks and user questions.

Using the same function on both sides is essential for BM25: a question term
can only match a chunk term if both were normalised identically.

We deliberately keep preprocessing light (no stemming, no lemmatisation) so
technical terms such as 'bm25', 'k1' or 'python3' survive unchanged.
"""
from __future__ import annotations

import re
import unicodedata

# Numbers with decimals (e.g. 3.11, 0.75) stay in one piece; otherwise a token is
# a run of letters/digits/underscores. Hyphenated words split into their parts
# ("machine-learning" -> "machine", "learning") so both spellings match.
TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+)+|\w+")

# Small, conservative list: function words and question words only.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the of in on at to for from by with and or but if then than that this these those
    it its as is are was were be been being am do does did has have had not no
    i me my we our you your he she they them their there here also so such
    what which who whom whose when where why how
    can could will would should may might must
    about into over tell explain describe define give please
    """.split()
)


def normalize_text(text: str) -> str:
    """Unicode-normalise (NFKC) and lower-case."""
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Convert text to a list of lower-case tokens.

    >>> tokenize("Who created Python 3.11?")
    ['created', 'python', '3.11']
    """
    tokens = TOKEN_PATTERN.findall(normalize_text(text))
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens
