"""Evaluate the Redis RAG pipeline against HotpotQA labels with Weave.

Scorers:
  hotpot_scorer     -- the OFFICIAL HotpotQA leaderboard metrics, computed with
                       the ported scoring code in hotpot_metrics.py:
                         Answer EM / F1, Supporting-fact EM / F1, Joint EM / F1
  retrieval_scorer  -- supplementary IR diagnostics for the retriever:
                       recall@k, precision@k, full_support (all gold docs found)

    python evaluate.py
"""
import asyncio
import csv
import json
from collections import defaultdict

import weave

import config
from hotpot_metrics import score_example
from rag_model import RedisRagModel


def build_rows() -> list[dict]:
    """Eval rows: question, gold answer, gold supporting_facts, gold corpus-ids."""
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
            meta = obj["metadata"]
            rows.append(
                {
                    "question": obj["text"],
                    "answer": str(meta["answer"]),  # some answers are non-str
                    "supporting_facts": meta["supporting_facts"],  # [[title, sent_idx], ...]
                    "gold_ids": sorted(q2gold[qid]),
                }
            )
    return rows


@weave.op
def hotpot_scorer(answer: str, supporting_facts: list, output: dict) -> dict:
    """Official HotpotQA metrics: answer, supporting-fact, and joint EM/F1."""
    return score_example(
        pred_answer=output["answer"],
        gold_answer=answer,
        pred_sp=output.get("supporting_facts", []),
        gold_sp=supporting_facts,
    )


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


async def main() -> None:
    weave.init(config.WANDB_PROJECT)
    rows = build_rows()
    print(f"Evaluating {len(rows)} queries (top_k={config.TOP_K})")
    evaluation = weave.Evaluation(
        name="hotpot-rag",
        dataset=rows,
        scorers=[hotpot_scorer, retrieval_scorer],
    )
    results = await evaluation.evaluate(RedisRagModel())
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
