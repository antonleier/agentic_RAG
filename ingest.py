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
import redis
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
            # Only doc_id and the vector are INDEXED. title/text are still
            # written to each hash (and returned by queries) but not indexed,
            # which saves the RediSearch inverted-index RAM -> more docs fit.
            "fields": [
                {"name": "doc_id", "type": "tag"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": config.EMBED_DIMS,
                        "distance_metric": "cosine",
                        "algorithm": config.VECTOR_ALGORITHM,
                        "datatype": "float32",
                    },
                },
            ],
        }
    )
    index = SearchIndex(schema, redis_url=config.require_redis_url())
    index.create(overwrite=True)
    return index


LOAD_BATCH = 5_000


def main() -> None:
    docs = load_subset()
    print(f"Loaded {len(docs)} docs from subset (gold docs first)")

    embeddings = get_embeddings(docs)
    assert embeddings.shape == (len(docs), config.EMBED_DIMS), embeddings.shape

    index = build_index()
    mem_client = redis.from_url(config.require_redis_url())

    # Load in batches; stop once Redis memory approaches the cap. Because the
    # subset is written gold-docs-first, the gold docs are always loaded.
    loaded = 0
    for start in range(0, len(docs), LOAD_BATCH):
        batch = docs[start : start + LOAD_BATCH]
        records = [
            {
                "doc_id": d["_id"],
                "title": d.get("title", ""),
                "text": d.get("text", ""),
                "embedding": embeddings[start + j].astype(np.float32).tobytes(),
            }
            for j, d in enumerate(batch)
        ]
        try:
            index.load(records, id_field="doc_id")
        except Exception as e:  # e.g. Redis OOM if we bump the hard limit
            print(f"Stopping: Redis rejected a write at {loaded:,} docs ({type(e).__name__}: {e}).")
            break
        loaded += len(batch)
        used = mem_client.info("memory")["used_memory"]
        print(f"  loaded {loaded:,}/{len(docs):,} | redis used {used/1e9:.2f} GB")
        if used >= config.REDIS_MEM_CAP_BYTES:
            print(f"Reached memory cap ({config.REDIS_MEM_CAP_BYTES/1e9:.1f} GB); stopping.")
            break

    print(f"Done. Indexed {loaded:,} docs in '{config.INDEX_NAME}'.")
    if loaded < len(docs):
        print(f"NOTE: {len(docs) - loaded:,} distractor docs were skipped to stay under the cap.")


if __name__ == "__main__":
    main()
