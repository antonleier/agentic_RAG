"""RedisRagModel: a real RAG pipeline wrapped as a weave.Model.

predict(question):
  1. embed the question with OpenAI
  2. KNN-retrieve top_k docs from the Redis vector index
  3. build a context block from the retrieved docs
  4. answer with Llama-3.3-70B via W&B Inference (OpenAI-compatible, Weave-traced)
  -> {"answer": str, "retrieved_ids": list[str]}

Clients are module-level singletons (built lazily) because weave.Model is a
pydantic model and shouldn't hold arbitrary client objects as fields.
"""
import functools

import numpy as np
import weave
from openai import OpenAI
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

import config


@functools.lru_cache(maxsize=1)
def _openai() -> OpenAI:
    return OpenAI()  # OPENAI_API_KEY from env


@functools.lru_cache(maxsize=1)
def _llm() -> OpenAI:
    # W&B Inference speaks the OpenAI protocol; routing it through the OpenAI
    # SDK lets Weave auto-trace the generation call.
    return OpenAI(
        base_url=config.WANDB_INFERENCE_BASE_URL,
        api_key=config.WANDB_API_KEY,
        # W&B Inference's OpenAI-Project header must be the full "team/project"
        # path; the bare entity fails auth.
        project=config.WANDB_PROJECT,
    )


@functools.lru_cache(maxsize=1)
def _index() -> SearchIndex:
    return SearchIndex.from_existing(
        config.INDEX_NAME, redis_url=config.require_redis_url()
    )


PROMPT = (
    "Answer the question with a short, factual answer (a few words). "
    "Use only the context below.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


class RedisRagModel(weave.Model):
    top_k: int = config.TOP_K

    @weave.op
    def retrieve(self, question: str) -> list[dict]:
        qvec = (
            _openai()
            .embeddings.create(model=config.EMBED_MODEL, input=[question])
            .data[0]
            .embedding
        )
        query = VectorQuery(
            vector=np.array(qvec, dtype=np.float32).tobytes(),
            vector_field_name="embedding",
            num_results=self.top_k,
            return_fields=["doc_id", "title", "text"],
        )
        return _index().query(query)

    @weave.op
    def predict(self, question: str) -> dict:
        hits = self.retrieve(question)
        context = "\n\n".join(f"[{h['title']}] {h['text']}" for h in hits)
        resp = _llm().chat.completions.create(
            model=config.ANSWER_MODEL,
            temperature=0,
            max_tokens=64,
            messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
        )
        return {
            "answer": resp.choices[0].message.content.strip(),
            "retrieved_ids": [h["doc_id"] for h in hits],
        }
