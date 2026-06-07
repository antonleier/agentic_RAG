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
# Redis Cloud free tier (~30MB) only fits a small index. Vectors cost
# n_docs * EMBED_DIMS * 4 bytes. 2500 docs ~= 15MB of vectors.
# Bump these if your Redis database is larger; nothing else changes.
N_QUERIES = 150        # eval queries sampled from the test split
TARGET_CORPUS = 2_500  # total docs to index (gold docs + random distractors)
SEED = 42

# --- embeddings (OpenAI) ------------------------------------------------
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
EMBED_BATCH = 256

# --- Redis vector index -------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL")  # redis://default:<pw>@<host>:<port>
INDEX_NAME = "hotpot_rag"
KEY_PREFIX = "doc:"
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
