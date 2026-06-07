"""Evaluate the Redis RAG pipeline against HotpotQA labels with Weave.

Two scorers:
  retrieval_scorer  -- did we retrieve the gold docs? (recall, precision,
                       full_support = all gold docs retrieved, the multi-hop bar)
  answer_scorer     -- does the generated answer match the gold answer?
                       (normalized exact match + lenient containment)

    python evaluate.py
"""
import asyncio
import csv
import json
import re
import string
from collections import defaultdict

import weave

import config
from rag_model import RedisRagModel


def build_rows() -> list[dict]:
    """Eval rows carrying the question, gold answer, and gold corpus-ids."""
    q2gold: dict[str, set[str]] = defaultdict(set)
    with open(config.QRELS_SUBSET) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for qid, cid, _score in reader:
            q2gold[qid].add(cid)

    rows = []
    with open(config.QUERIES_SUBSET) as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["_id"]
            rows.append(
                {
                    "question": obj["text"],
                    "answer": str(obj["metadata"]["answer"]),  # some answers are non-str
                    "gold_ids": sorted(q2gold[qid]),
                }
            )
    return rows


def _norm(s: str) -> str:
    """SQuAD/HotpotQA-style normalization: lowercase, drop punct + articles."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


@weave.op
def retrieval_scorer(gold_ids: list, output: dict) -> dict:
    gold, retrieved = set(gold_ids), set(output["retrieved_ids"])
    if not gold:
        return {}
    hit = gold & retrieved
    return {
        "recall_at_k": len(hit) / len(gold),
        "precision_at_k": len(hit) / max(1, len(retrieved)),
        "full_support": float(gold.issubset(retrieved)),
    }


@weave.op
def answer_scorer(answer: str, output: dict) -> dict:
    gold, pred = _norm(answer), _norm(output["answer"])
    return {
        "exact_match": float(gold == pred),
        "contains": float(bool(gold) and (gold in pred or pred in gold)),
    }


async def main() -> None:
    weave.init(config.WANDB_PROJECT)
    rows = build_rows()
    print(f"Evaluating {len(rows)} queries (top_k={config.TOP_K})")
    evaluation = weave.Evaluation(
        name="hotpot-rag",
        dataset=rows,
        scorers=[retrieval_scorer, answer_scorer],
    )
    results = await evaluation.evaluate(RedisRagModel())
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
