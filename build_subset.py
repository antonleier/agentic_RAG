"""Build a small, indexable subset of the 5.23M-doc HotpotQA corpus.

Strategy: sample N_QUERIES from the test split, collect their gold documents
(from qrels), then add random "distractor" documents up to TARGET_CORPUS.
The 2GB corpus.jsonl is streamed in a single pass and never fully loaded.

Idempotent: skips work if all three outputs already exist.

    python build_subset.py
"""
import csv
import json
import random
from collections import defaultdict

import config


def load_test_qrels() -> dict[str, set[str]]:
    """query-id -> set(corpus-id) from the test qrels TSV (skips header)."""
    q2gold: dict[str, set[str]] = defaultdict(set)
    with open(config.QRELS_TEST) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for qid, cid, _score in reader:
            q2gold[qid].add(cid)
    return q2gold


def main() -> None:
    if (
        config.CORPUS_SUBSET.exists()
        and config.QUERIES_SUBSET.exists()
        and config.QRELS_SUBSET.exists()
    ):
        print("Subset already built; skipping. Delete data/ to rebuild.")
        return

    rng = random.Random(config.SEED)
    q2gold = load_test_qrels()

    if config.N_QUERIES is None:
        sampled_qids = sorted(q2gold)  # ALL test queries (the full leaderboard set)
        print(f"Using all {len(sampled_qids)} test queries")
    else:
        sampled_qids = rng.sample(list(q2gold), config.N_QUERIES)
    gold_ids: set[str] = set().union(*(q2gold[q] for q in sampled_qids))
    print(f"{len(sampled_qids)} queries -> {len(gold_ids)} gold docs")

    n_distract = max(0, config.TARGET_CORPUS - len(gold_ids))

    # Single streaming pass: keep every gold doc, reservoir-sample distractors.
    kept: dict[str, str] = {}   # gold _id -> raw json line
    reservoir: list[str] = []   # distractor raw json lines
    seen_non_gold = 0
    with open(config.CORPUS) as f:
        for line in f:
            cid = json.loads(line)["_id"]
            if cid in gold_ids:
                kept[cid] = line
            else:
                seen_non_gold += 1
                if len(reservoir) < n_distract:
                    reservoir.append(line)
                else:
                    j = rng.randint(0, seen_non_gold - 1)
                    if j < n_distract:
                        reservoir[j] = line

    missing = gold_ids - kept.keys()
    if missing:
        print(f"WARNING: {len(missing)} gold docs not found in corpus (first few: "
              f"{list(missing)[:5]})")

    with open(config.CORPUS_SUBSET, "w") as out:
        for line in kept.values():
            out.write(line)
        for line in reservoir:
            out.write(line)
    print(f"Wrote {len(kept) + len(reservoir)} docs -> {config.CORPUS_SUBSET}")

    sampled_set = set(sampled_qids)
    n_q = 0
    with open(config.QUERIES) as qf, open(config.QUERIES_SUBSET, "w") as out:
        for line in qf:
            if json.loads(line)["_id"] in sampled_set:
                out.write(line)
                n_q += 1
    print(f"Wrote {n_q} queries -> {config.QUERIES_SUBSET}")

    with open(config.QRELS_SUBSET, "w") as out:
        out.write("query-id\tcorpus-id\tscore\n")
        for qid in sampled_qids:
            for cid in q2gold[qid]:
                out.write(f"{qid}\t{cid}\t1\n")
    print(f"Wrote qrels -> {config.QRELS_SUBSET}")


if __name__ == "__main__":
    main()
