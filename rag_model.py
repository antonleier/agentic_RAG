"""RedisRagModel: a real RAG pipeline wrapped as a weave.Model.

predict(question):
  1. embed the question with OpenAI
  2. KNN-retrieve top_k docs from the Redis vector index
  3. present each retrieved doc as title + sentence-indexed lines
  4. ask Llama-3.3-70B (via W&B Inference) for a JSON object containing the
     answer AND the supporting sentences it used, as [title, sentence_idx] pairs
  -> {"answer": str, "supporting_facts": list[[title, idx]], "retrieved_ids": [...]}

Emitting sentence-level supporting facts lets us score the official HotpotQA
Sup and Joint metrics (see hotpot_metrics.py). Clients are module-level
singletons (built lazily) because weave.Model is a pydantic model and shouldn't
hold arbitrary client objects as fields.
"""
import functools
import json
import re

import numpy as np
import weave
from openai import OpenAI
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

import config
from hotpot_metrics import split_sentences


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
    "You answer multi-hop questions using only the provided context.\n"
    "The context lists documents; each has a Title and sentences numbered [0], [1], ...\n\n"
    "Respond with a single JSON object and nothing else:\n"
    '{{"answer": "<short factual answer, a few words>", '
    '"supporting_facts": [["<exact document title>", <sentence number>], ...]}}\n\n'
    "Rules:\n"
    "- The answer must be as short as possible (a name, place, date, or yes/no).\n"
    "- supporting_facts must list ONLY the sentences you actually used, each as\n"
    "  [title, sentence_number] using titles and numbers exactly as shown below.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nJSON:"
)


def _format_context(hits: list[dict]) -> str:
    """Render retrieved docs as title + sentence-indexed lines for the prompt."""
    blocks = []
    for h in hits:
        lines = [f"Title: {h['title']}"]
        for i, sent in enumerate(split_sentences(h["text"])):
            lines.append(f"[{i}] {sent}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parse_response(content: str, valid_titles: set[str]) -> tuple[str, list]:
    """Extract answer + supporting_facts from the model's JSON reply.

    Robust to code fences / stray prose. Keeps only supporting facts whose title
    was actually in the retrieved context. Falls back to (raw text, []) on failure.
    """
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return content.strip(), []
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return content.strip(), []

    answer = str(obj.get("answer", "")).strip()
    sp = []
    for item in obj.get("supporting_facts", []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            title, idx = item
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            if title in valid_titles:
                sp.append([title, idx])
    return answer, sp


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
        context = _format_context(hits)
        resp = _llm().chat.completions.create(
            model=config.ANSWER_MODEL,
            temperature=0,
            max_tokens=256,  # room for the answer + supporting_facts JSON
            messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
        )
        valid_titles = {h["title"] for h in hits}
        answer, supporting_facts = _parse_response(
            resp.choices[0].message.content, valid_titles
        )
        return {
            "answer": answer,
            "supporting_facts": supporting_facts,
            "retrieved_ids": [h["doc_id"] for h in hits],
        }
