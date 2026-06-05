from __future__ import annotations

import hashlib
import math
import re
import time
from pathlib import Path
from typing import Any

from src import (
    ChunkingStrategyComparator,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    compute_similarity,
)


DATASET = [
    {
        "path": Path("data/caremark-oct2013.txt"),
        "source": "cnsdrugs local upload",
        "doc_type": "formulary_guide",
        "payer": "Caremark",
        "year": "2013",
        "language": "en",
    },
    {
        "path": Path("data/healthalliance-2013.txt"),
        "source": "cnsdrugs local upload",
        "doc_type": "formulary_guide",
        "payer": "Health Alliance",
        "year": "2013",
        "language": "en",
    },
    {
        "path": Path("data/humana_2014_wi.txt"),
        "source": "cnsdrugs local upload",
        "doc_type": "formulary_guide",
        "payer": "Humana",
        "year": "2014",
        "language": "en",
    },
    {
        "path": Path("data/uhc-cns-drugs-pdf-list.txt"),
        "source": "cnsdrugs local upload",
        "doc_type": "formulary_guide",
        "payer": "UnitedHealthcare",
        "year": "2013",
        "language": "en",
    },
    {
        "path": Path("data/medical_knowledge_base.txt"),
        "source": "local generated medical knowledge base",
        "doc_type": "drug_disease_knowledge_base",
        "payer": "N/A",
        "year": "N/A",
        "language": "vi",
    },
]


BENCHMARKS = [
    {
        "query": "What relationship does Rivastigmine have with Alzheimer's disease in the medical knowledge base?",
        "gold": "Rivastigmine has a DM clinical relationship with Alzheimer's disease and DrugBank ID DB00989.",
        "filter": {"doc_type": "drug_disease_knowledge_base"},
        "expected_source": "medical_knowledge_base.txt",
        "keywords": ["Rivastigmine", "DB00989", "DM", "Alzheimer"],
    },
    {
        "query": "In Health Alliance 2013, which anxiety medication is listed as Buspar?",
        "gold": "Buspirone is listed as buspirone (Buspar) under Anxiety.",
        "filter": {"payer": "Health Alliance"},
        "expected_source": "healthalliance-2013.txt",
        "keywords": ["buspirone", "Buspar", "ANXIETY"],
    },
    {
        "query": "What tier and quantity limit is ABILIFY 10 MG TABLET in Humana 2014 Wisconsin?",
        "gold": "ABILIFY 10 MG TABLET is listed as MO tier 4 with QL 30 per 30 days.",
        "filter": {"payer": "Humana"},
        "expected_source": "humana_2014_wi.txt",
        "keywords": ["ABILIFY 10 MG TABLET", "MO 4", "QL", "30 per 30 days"],
    },
    {
        "query": "UnitedHealthcare Morphine Sulfate Solution Oral Roxanol MS Contin",
        "gold": "The UHC CNS list includes Morphine Sulfate oral solution, Roxanol, MS Contin, and related Morphine Sulfate capsule entries.",
        "filter": {"payer": "UnitedHealthcare"},
        "expected_source": "uhc-cns-drugs-pdf-list.txt",
        "keywords": ["Morphine Sulfate", "Roxanol", "MS Contin", "NARCOTICS"],
    },
    {
        "query": "In Caremark October 2013, what note is listed for buspirone?",
        "gold": "Buspirone is listed with the note NP = 7.5 mg.",
        "filter": {"payer": "Caremark"},
        "expected_source": "caremark-oct2013.txt",
        "keywords": ["buspirone", "NP", "7.5 mg"],
    },
]

COSINE_QUERY_A = "What relationship does Rivastigmine have with Alzheimer's disease?"
COSINE_QUERY_B = "Which DrugBank ID is listed for Rivastigmine and Alzheimer's disease?"


def read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="ignore"), "cp1252"


def preview(text: str, limit: int = 110) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


class HashingTextEmbedder:
    """Small local lexical embedder for reproducible benchmark runs."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self._backend_name = f"hashing lexical embedder ({dim} dims)"

    def __call__(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            digest = hashlib.md5(token.encode()).hexdigest()
            index = int(digest, 16) % self.dim
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def make_documents(chunker: Any) -> list[Document]:
    documents: list[Document] = []
    for item in DATASET:
        text, encoding = read_text(item["path"])
        chunks = chunker.chunk(text)
        for chunk_index, chunk in enumerate(chunks):
            metadata = {
                "doc_id": item["path"].stem,
                "source_file": item["path"].name,
                "source": item["source"],
                "doc_type": item["doc_type"],
                "payer": item["payer"],
                "year": item["year"],
                "language": item["language"],
                "encoding": encoding,
                "chunk_index": chunk_index,
            }
            documents.append(Document(id=f"{item['path'].stem}-{chunk_index}", content=chunk, metadata=metadata))
    return documents


def make_store(chunker: Any) -> EmbeddingStore:
    store = EmbeddingStore(collection_name="medical_group_benchmark", embedding_fn=HashingTextEmbedder())
    store.add_documents(make_documents(chunker))
    return store


def is_relevant(results: list[dict[str, Any]], benchmark: dict[str, Any]) -> bool:
    combined = "\n".join(result["content"] for result in results[:3]).lower()
    has_source = any(result["metadata"].get("source_file") == benchmark["expected_source"] for result in results[:3])
    keyword_hits = sum(1 for keyword in benchmark["keywords"] if keyword.lower() in combined)
    return has_source and keyword_hits >= 2


def print_inventory() -> None:
    print("## DATA INVENTORY")
    print("| # | Tên tài liệu | Số ký tự | Encoding | Metadata |")
    print("|---:|---|---:|---|---|")
    for index, item in enumerate(DATASET, start=1):
        text, encoding = read_text(item["path"])
        metadata = (
            f"doc_type={item['doc_type']}; payer={item['payer']}; "
            f"year={item['year']}; language={item['language']}"
        )
        print(f"| {index} | {item['path'].name} | {len(text)} | {encoding} | {metadata} |")


def print_baseline() -> None:
    print("\n## CHUNKING BASELINE")
    print("| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |")
    print("|---|---|---:|---:|---|")
    labels = {
        "fixed_size": "FixedSizeChunker (`fixed_size`)",
        "by_sentences": "SentenceChunker (`by_sentences`)",
        "recursive": "RecursiveChunker (`recursive`)",
    }
    notes = {
        "fixed_size": "Trung bình: ổn định kích thước nhưng dễ cắt ngang dòng thuốc",
        "by_sentences": "Thấp với formulary: nhiều file ít dấu câu nên chunk quá dài",
        "recursive": "Tốt: ưu tiên đoạn/dòng, hợp với danh sách thuốc",
    }
    comparator = ChunkingStrategyComparator()
    for item in DATASET:
        text, _encoding = read_text(item["path"])
        result = comparator.compare(text)
        for key in ["fixed_size", "by_sentences", "recursive"]:
            stats = result[key]
            print(
                f"| {item['path'].name} | {labels[key]} | {stats['count']} | "
                f"{stats['avg_length']:.1f} | {notes[key]} |"
            )


def print_cosine_query_comparison() -> None:
    embedder = HashingTextEmbedder()
    query_a_embedding = embedder(COSINE_QUERY_A)
    query_b_embedding = embedder(COSINE_QUERY_B)
    similarity = compute_similarity(query_a_embedding, query_b_embedding)
    interpretation = "High similarity: both queries ask about Rivastigmine and Alzheimer's disease"

    print("\n## COSINE QUERY COMPARISON")
    print("| Query A | Query B | Embedding Method | Cosine Similarity | Interpretation |")
    print("|---|---|---|---:|---|")
    print(
        f"| {COSINE_QUERY_A} | {COSINE_QUERY_B} | "
        f"{embedder._backend_name} | {similarity:.4f} | {interpretation} |"
    )


def score_strategy(name: str, chunker: Any) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    build_start = time.perf_counter()
    store = make_store(chunker)
    build_ms = (time.perf_counter() - build_start) * 1000

    rows = []
    score = 0
    relevant_count = 0
    query_times = []
    for index, benchmark in enumerate(BENCHMARKS, start=1):
        query_start = time.perf_counter()
        if benchmark["filter"]:
            results = store.search_with_filter(
                benchmark["query"],
                top_k=3,
                metadata_filter=benchmark["filter"],
            )
        else:
            results = store.search(benchmark["query"], top_k=3)
        query_times.append((time.perf_counter() - query_start) * 1000)

        relevant = is_relevant(results, benchmark)
        if relevant:
            score += 2
            relevant_count += 1
        elif any(result["metadata"].get("source_file") == benchmark["expected_source"] for result in results[:3]):
            score += 1

        top = results[0] if results else {"content": "", "score": 0.0, "metadata": {}}
        rows.append(
            {
                "index": index,
                "strategy": name,
                "query": benchmark["query"],
                "gold": benchmark["gold"],
                "top1": preview(top["content"]),
                "score": top["score"],
                "source_file": top["metadata"].get("source_file", ""),
                "relevant": relevant,
                "answer": preview("\n".join(result["content"] for result in results[:3]), 180),
            }
        )
    metrics = {
        "stored_chunks": store.get_collection_size(),
        "build_ms": build_ms,
        "avg_query_ms": sum(query_times) / len(query_times) if query_times else 0.0,
        "max_query_ms": max(query_times) if query_times else 0.0,
        "precision_at_3": relevant_count / len(BENCHMARKS) if BENCHMARKS else 0.0,
    }
    return score, rows, metrics


def print_strategy_scores() -> None:
    print("\n## STRATEGY SCORES")
    strategies = {
        "FixedSizeChunker(500, overlap=50)": FixedSizeChunker(chunk_size=500, overlap=50),
        "SentenceChunker(3 sentences)": SentenceChunker(max_sentences_per_chunk=3),
        "RecursiveChunker(500)": RecursiveChunker(chunk_size=500),
    }
    print("| Strategy | Stored Chunks | Build Time (ms) | Avg Query Time (ms) | Max Query Time (ms) | Precision@3 | Retrieval Score (/10) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    all_rows = []
    for name, chunker in strategies.items():
        score, rows, metrics = score_strategy(name, chunker)
        all_rows.extend(rows)
        print(
            f"| {name} | {metrics['stored_chunks']} | {metrics['build_ms']:.2f} | "
            f"{metrics['avg_query_ms']:.2f} | {metrics['max_query_ms']:.2f} | "
            f"{metrics['precision_at_3']:.2f} | {score} |"
        )

    print("\n## MY RESULTS - RecursiveChunker(500)")
    recursive_rows = [row for row in all_rows if row["strategy"] == "RecursiveChunker(500)"]
    print("| # | Query | Top-1 Source | Top-1 Retrieved Chunk | Score | Relevant? | Agent Answer |")
    print("|---:|---|---|---|---:|---|---|")
    for row in recursive_rows:
        relevant = "Yes" if row["relevant"] else "No"
        print(
            f"| {row['index']} | {row['query']} | {row['source_file']} | {row['top1']} | "
            f"{row['score']:.3f} | {relevant} | {row['answer']} |"
        )


def main() -> None:
    print_inventory()
    print_baseline()
    print_cosine_query_comparison()
    print_strategy_scores()


if __name__ == "__main__":
    main()
