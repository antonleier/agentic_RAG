"""Central configuration for the HotpotQA RAG pipeline.

Single source of truth for paths, sampling sizes, model names, and secrets.
Everything else imports from here.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# --- dataset paths (BEIR-format HotpotQA) -------------------------------
HOTPOT = ROOT / "hotpotqa"
CORPUS = HOTPOT / "corpus.jsonl"
QUERIES = HOTPOT / "queries.jsonl"
QRELS_TEST = HOTPOT / "qrels" / "test.tsv"

# --- generated artifacts ------------------------------------------------
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
CORPUS_SUBSET = DATA / "corpus_subset.jsonl"
QUERIES_SUBSET = DATA / "queries_subset.jsonl"
QRELS_SUBSET = DATA / "qrels_subset.tsv"
EMB_CACHE = DATA / "doc_embeddings.npy"
IDS_CACHE = DATA / "doc_ids.json"

# --- subset sizing ------------------------------------------------------
# N_QUERIES = None -> use ALL test queries (7,405), the full leaderboard set.
# An int -> sample that many (handy for quick local runs).
# TARGET_CORPUS is an UPPER bound on docs to index; the real limiter is the
# Redis memory cap below (ingest loads all gold docs first, then distractors
# until memory runs out), so we can "use as much of the 5GB as fits" safely.
N_QUERIES = None
TARGET_CORPUS = 300_000
SEED = 42

# Stop loading distractors once Redis used_memory exceeds this (bytes).
# ~2.2GB leaves headroom under a 2.5GB plan for index/fragmentation overhead.
REDIS_MEM_CAP_BYTES = 2_200_000_000

# --- embeddings (OpenAI) ------------------------------------------------
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
EMBED_BATCH = 256

# --- Redis vector index -------------------------------------------------
# Several Redis Cloud databases may be configured; default to the 2.5GB one
# (it has the Search & Query module enabled).
REDIS_URL_1GB = os.environ.get("REDIS_URL_1GB")
REDIS_URL_2_5GB = os.environ.get("REDIS_URL_2_5GB")
REDIS_URL_5GB = os.environ.get("REDIS_URL_5GB")
REDIS_URL = (
    REDIS_URL_2_5GB or REDIS_URL_5GB or REDIS_URL_1GB or os.environ.get("REDIS_URL")
)
INDEX_NAME = "hotpot_rag"
KEY_PREFIX = "doc:"
# FLAT = exact KNN (fine for a few thousand docs); HNSW = approximate, needed
# once the corpus is hundreds of thousands of docs so queries stay fast.
VECTOR_ALGORITHM = "hnsw"
TOP_K = 5

# --- answer LLM (W&B Inference, OpenAI-compatible) ----------------------
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
ANSWER_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
WANDB_PROJECT = "vejaleier-lss/weavehacks-rag"
WANDB_ENTITY = "vejaleier-lss"


def require_redis_url() -> str:
    """Fail fast with a helpful message if REDIS_URL isn't configured."""
    if not REDIS_URL:
        raise SystemExit(
            "REDIS_URL is not set. Add the connection string from your Redis Cloud\n"
            "console ('Connect' page) to .env, e.g.:\n"
            "  REDIS_URL=redis://default:<password>@<host>:<port>"
        )
    return REDIS_URL
