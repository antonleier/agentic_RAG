"""Official HotpotQA metric functions (ported from hotpot_evaluate_v1.py).

These reproduce the exact scoring used by the HotpotQA leaderboard so our
numbers are directly comparable:

  - Answer:          EM + token-level F1
  - Supporting facts: EM + F1 over the set of (title, sentence_idx) pairs
  - Joint:           combines answer and supporting-fact correctness

Reference: https://github.com/hotpotqa/hotpot (hotpot_evaluate_v1.py)
"""
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Token-level F1, precision, recall (with HotpotQA's yes/no/noanswer rule)."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO = (0.0, 0.0, 0.0)
    # yes/no/noanswer only score if they match exactly.
    if normalized_prediction in {"yes", "no", "noanswer"} and \
            normalized_prediction != normalized_ground_truth:
        return ZERO
    if normalized_ground_truth in {"yes", "no", "noanswer"} and \
            normalized_prediction != normalized_ground_truth:
        return ZERO

    pred_tokens = normalized_prediction.split()
    gold_tokens = normalized_ground_truth.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def sp_metrics(prediction: list, gold: list) -> tuple[float, float, float, float]:
    """Supporting-fact EM, precision, recall, F1 over (title, sent_idx) pairs."""
    cur = set(map(tuple, prediction))
    gold_set = set(map(tuple, gold))
    tp = len(cur & gold_set)
    fp = len(cur - gold_set)
    fn = len(gold_set - cur)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return em, prec, recall, f1


def score_example(
    pred_answer: str,
    gold_answer: str,
    pred_sp: list,
    gold_sp: list,
) -> dict:
    """Full official scoring for one example: answer, sup, and joint metrics."""
    ans_f1, ans_prec, ans_recall = f1_score(pred_answer, gold_answer)
    ans_em = exact_match_score(pred_answer, gold_answer)

    sp_em, sp_prec, sp_recall, sp_f1 = sp_metrics(pred_sp, gold_sp)

    # Joint: precision/recall are the products; EM requires both EMs.
    joint_prec = ans_prec * sp_prec
    joint_recall = ans_recall * sp_recall
    joint_f1 = (
        2 * joint_prec * joint_recall / (joint_prec + joint_recall)
        if joint_prec + joint_recall > 0
        else 0.0
    )
    joint_em = ans_em * sp_em

    return {
        "ans_em": ans_em,
        "ans_f1": ans_f1,
        "sup_em": sp_em,
        "sup_f1": sp_f1,
        "sup_prec": sp_prec,
        "sup_recall": sp_recall,
        "joint_em": joint_em,
        "joint_f1": joint_f1,
    }


def split_sentences(text: str) -> list[str]:
    """Sentence splitter aligned with the gold supporting_facts indices.

    Verified against the BEIR HotpotQA corpus: 368/369 gold (title, sent_idx)
    entries resolve in-range with this split. Used both to present indexed
    sentences to the model and (implicitly) to keep predicted indices
    comparable to the gold ones.
    """
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p for p in parts if p]
