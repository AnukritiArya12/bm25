"""Glue layer: document -> chunks -> BM25 index, and question -> ranked passages.

Retrieval is 100% lexical BM25. Nothing here calls a language model, computes
embeddings or uses a vector database.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from rank_bm25 import BM25Okapi

from src.bm25_engine import create_bm25, score_query
from src.chunker import Chunk, chunk_pages
from src.tokenizer import tokenize
from src.utils import (
    EmptyDocumentError,
    EmptyQuestionError,
    IndexNotBuiltError,
    LoadedDocument,
    clean_text,
    validate_search_settings,
)


@dataclass
class SearchIndex:
    """Everything needed to answer questions about one document."""

    bm25: BM25Okapi
    chunks: list[Chunk]
    tokenized_chunks: list[list[str]]


class SearchResult(TypedDict):
    rank: int
    score: float
    text: str
    source: str
    page: int | None
    section: str | None
    chunk_id: int
    matched_terms: list[str]


class SearchResponse(TypedDict):
    results: list[SearchResult]  # only passages that passed the relevance threshold
    best_score: float  # best BM25 score found (even if below the threshold)
    query_terms: list[str]


def build_search_index(chunks: list[Chunk]) -> SearchIndex:
    """Tokenize the chunks (dropping ones with no searchable words) and build BM25."""
    kept_chunks: list[Chunk] = []
    tokenized: list[list[str]] = []
    for chunk in chunks:
        tokens = tokenize(chunk["text"])
        if tokens:
            kept_chunks.append(chunk)
            tokenized.append(tokens)
    if not kept_chunks:
        raise EmptyDocumentError("The document has no searchable text after cleaning.")
    return SearchIndex(bm25=create_bm25(tokenized), chunks=kept_chunks, tokenized_chunks=tokenized)


def index_document(document: LoadedDocument, chunk_size: int = 120, overlap: int = 30) -> SearchIndex:
    """Full indexing pipeline: clean -> chunk -> tokenize -> BM25."""
    cleaned_pages = [{**page, "text": clean_text(page["text"])} for page in document["pages"]]
    cleaned_pages = [page for page in cleaned_pages if page["text"]]
    if not cleaned_pages:
        raise EmptyDocumentError("The extracted text is empty.")
    chunks = chunk_pages(cleaned_pages, document["name"], chunk_size, overlap)
    return build_search_index(chunks)


def search(index: SearchIndex | None, question: str, top_k: int = 3, min_score: float = 1.0) -> SearchResponse:
    """Rank chunks with BM25 and keep the top_k that reach `min_score`."""
    if index is None:
        raise IndexNotBuiltError("Please upload a document and build the index first.")
    validate_search_settings(top_k, min_score)
    if not question or not question.strip():
        raise EmptyQuestionError("Please type a question.")
    query_terms = tokenize(question)
    if not query_terms:
        raise EmptyQuestionError("Your question has no searchable keywords. Try including specific words from the topic.")

    scores = score_query(index.bm25, query_terms)
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))  # ties keep document order
    best_score = scores[ranked[0]] if ranked else 0.0

    results: list[SearchResult] = []
    for position in ranked[:top_k]:
        score = scores[position]
        if score <= 0 or score < min_score:
            break  # sorted, so everything after is lower too
        chunk = index.chunks[position]
        results.append(
            {
                "rank": len(results) + 1,
                "score": score,
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "section": chunk["section"],
                "chunk_id": chunk["chunk_id"],
                "matched_terms": sorted(set(query_terms) & set(index.tokenized_chunks[position])),
            }
        )
    return {"results": results, "best_score": best_score, "query_terms": query_terms}
