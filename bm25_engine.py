"""Thin wrapper around the `rank-bm25` library (BM25Okapi).

BM25 score of chunk D for query Q:

    score(D, Q) = sum over query terms t of
        IDF(t) * f(t, D) * (k1 + 1) / ( f(t, D) + k1 * (1 - b + b * |D| / avgdl) )

    f(t, D) : how often term t occurs in chunk D
    |D|     : chunk length in tokens, avgdl: average chunk length
    IDF(t)  : rarer terms (found in few chunks) weigh more
    k1, b   : saturation of term frequency / strength of length normalisation
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.utils import EmptyDocumentError


def create_bm25(tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> BM25Okapi:
    """Build a BM25 index from already-tokenized chunks."""
    if not tokenized_corpus or not any(tokenized_corpus):
        raise EmptyDocumentError("There is no searchable text to index.")
    return BM25Okapi(tokenized_corpus, k1=k1, b=b)


def score_query(bm25: BM25Okapi, query_tokens: list[str]) -> list[float]:
    """Return one BM25 score per indexed chunk (same order as the corpus)."""
    return [float(score) for score in bm25.get_scores(query_tokens)]
