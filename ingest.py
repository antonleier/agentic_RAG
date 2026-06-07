"""Embed the corpus subset with OpenAI and load it into a Redis vector index.

Steps:
  1. Read data/corpus_subset.jsonl
  2. Embed (title + text) with OpenAI, batched, cached to data/doc_embeddings.npy
  3. Create a redisvl SearchIndex (FLAT, cosine, float32)
  4. Load each doc as a hash with its float32 vector

Idempotent: embeddings are cached; the index is recreated and keyed on doc_id,
so re-running overwrites cleanly. Delete data/doc_embeddings.npy + the Redis
index to force a full rebuild.

    python ingest.py
"""
import json
import time

import numpy as np
from openai import OpenAI, RateLimitError
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema

import config

MAX_DOC_CHARS = 8_000  # keep well under the 8191-token embedding limit


def load_subset() -> list[dict]:
    with open(config.CORPUS_SUBSET) as f:
        return [json.loads(line) for line in f]


def embed_texts(texts: list[str]) -> np.ndarray:
    client = OpenAI()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        batch = texts[i : i + config.EMBED_BATCH]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=config.EMBED_MODEL, input=batch)
                break
            except RateLimitError:
                wait = 2 ** attempt
                print(f"  rate limited, retrying in {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError("Embedding failed after retries")
        vectors.extend(d.embedding for d in resp.data)
        print(f"  embedded {i + len(batch)}/{len(texts)}")
    return np.array(vectors, dtype=np.float32)


def get_embeddings(docs: list[dict]) -> np.ndarray:
    """Return embeddings, using the on-disk cache when it matches the subset."""
    if config.EMB_CACHE.exists() and config.IDS_CACHE.exists():
        cached_ids = json.load(open(config.IDS_CACHE))
        if cached_ids == [d["_id"] for d in docs]:
            print(f"Using cached embeddings ({config.EMB_CACHE.name})")
            return np.load(config.EMB_CACHE)
        print("Cache stale (subset changed); re-embedding.")

    texts = [
        (f"{d.get('title', '')}\n{d.get('text', '')}")[:MAX_DOC_CHARS] for d in docs
    ]
    embeddings = embed_texts(texts)
    np.save(config.EMB_CACHE, embeddings)
    json.dump([d["_id"] for d in docs], open(config.IDS_CACHE, "w"))
    return embeddings


def build_index() -> SearchIndex:
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": config.INDEX_NAME,
                "prefix": config.KEY_PREFIX,
                "storage_type": "hash",
            },
            "fields": [
                {"name": "doc_id", "type": "tag"},
                {"name": "title", "type": "text"},
                {"name": "text", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": config.EMBED_DIMS,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )
    index = SearchIndex(schema, redis_url=config.require_redis_url())
    index.create(overwrite=True)
    return index


def main() -> None:
    docs = load_subset()
    print(f"Loaded {len(docs)} docs from subset")

    embeddings = get_embeddings(docs)
    assert embeddings.shape == (len(docs), config.EMBED_DIMS), embeddings.shape

    index = build_index()
    records = [
        {
            "doc_id": d["_id"],
            "title": d.get("title", ""),
            "text": d.get("text", ""),
            "embedding": embeddings[i].astype(np.float32).tobytes(),
        }
        for i, d in enumerate(docs)
    ]
    keys = index.load(records, id_field="doc_id")
    print(f"Loaded {len(keys)} docs into Redis index '{config.INDEX_NAME}'")


if __name__ == "__main__":
    main()
