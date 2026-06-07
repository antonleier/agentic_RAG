# Evaluation Metrics

We track two groups of metrics: **retrieval** (did we find the right documents?) and **answer + supporting facts** (did we answer correctly and cite the right evidence?). The second group matches the official HotpotQA leaderboard exactly.

---

## Example query (used throughout)

> "In what city did the 'Prince of tenors' star in a film based on an opera by Giacomo Puccini?"

- **Gold answer:** `Rome`
- **Gold supporting facts:**
  - `["Franco Corelli", sentence 2]` — *"Dubbed the 'Prince of tenors', Corelli..."*
  - `["Tosca (1956 film)", sentence 2]` — *"It was made at Cinecittà in Rome."*
- **Gold documents:** `353102` (Franco Corelli), `50924944` (Tosca 1956 film)

This is a **multi-hop** question: no single document contains the answer. You need both documents.

---

## Group 1 — Retrieval (IR diagnostics)

Measures whether the retriever (Redis vector search) fetched the right documents.

Let **G** = gold doc IDs, **R** = top-k retrieved doc IDs, **hits** = G ∩ R.

| Metric | Formula | Example (k=5, both gold docs retrieved) |
|---|---|---|
| **recall@k** | hits / \|G\| | 2/2 = **1.0** |
| **precision@k** | hits / \|R\| | 2/5 = **0.40** |
| **full_support** | 1 if G ⊆ R else 0 | **1.0** (both gold docs in top 5) |

**full_support is the key retrieval metric for multi-hop QA.** Recall gives partial credit (finding 1 of 2 gold docs = 0.5); full_support is all-or-nothing. If full_support = 0 for a query, the LLM never saw all the evidence it needed — the answer cannot be fully correct regardless of how good the model is.

Precision is naturally low when k > number of gold docs (at most 2 gold docs, k=5 → ceiling is 0.40). It's a sanity check, not a target.

---

## Group 2 — Answer + Supporting Facts (official HotpotQA leaderboard metrics)

These match the scoring code at [hotpotqa/hotpot](https://github.com/hotpotqa/hotpot) (hotpot_evaluate_v1.py) and are directly comparable to the leaderboard.

All text is normalized before comparison: lowercased, punctuation removed, articles (a/an/the) dropped.

### Answer metrics

| Metric | What it measures | Example |
|---|---|---|
| **Ans EM** | Exact match: 1 if normalized strings are identical, else 0 | `"Rome"` vs `"Rome"` → **1.0** |
| **Ans F1** | Token-level overlap (harmonic mean of precision & recall over words) | `"the city of Rome"` vs `"Rome"` → **0.5** |

EM is strict; F1 gives partial credit for verbose but correct answers. The gap between them (e.g. F1=0.77, EM=0.54) is almost always formatting — the model answers in a sentence rather than a bare phrase.

### Supporting facts metrics

The model must also predict which **sentences** it used: `[document title, sentence index]`. These are scored against the gold supporting facts from HotpotQA.

Let **G** = gold (title, sent_idx) pairs, **P** = predicted pairs.

| Metric | Formula | Example (predicted both gold + 1 extra) |
|---|---|---|
| **Sup EM** | 1 if P == G exactly, else 0 | extra sentence → **0** |
| **Sup F1** | F1 over the set of (title, idx) pairs | 2 correct / 3 predicted → prec=0.67, recall=1.0 → **F1=0.80** |

Sup EM is strict — any extra or missing sentence makes it 0. Sup F1 gives partial credit.

### Joint metrics

Combines answer and supporting facts: you only get credit when **both** are right.

| Metric | Formula |
|---|---|
| **Joint EM** | Ans EM × Sup EM (both must be 1) |
| **Joint F1** | Computed from joint precision (Ans\_prec × Sup\_prec) and recall (Ans\_recall × Sup\_recall) |

Joint EM is the headline "fully solved the multi-hop task" rate. It is always ≤ min(Ans EM, Sup EM).

---

## Our results (7,405 test queries, 139k-doc corpus)

| Metric | Score | Leaderboard #1 |
|---|---|---|
| Ans EM | 38.8 | 66.7 |
| Ans F1 | 48.8 | 79.7 |
| Sup EM | 16.3 | 70.7 |
| Sup F1 | 48.2 | 85.3 |
| Joint EM | 10.1 | 51.4 |
| Joint F1 | 31.2 | 66.4 |
| recall@5 | 62.8 | — |
| full_support | 42.4 | — |

**The main bottleneck is retrieval** (full_support 42.4%). When the gold documents are not retrieved, the model cannot produce the correct answer or supporting facts. The leaderboard leaders use iterative multi-hop retrieval and fine-tuned dense retrievers; we use a single-shot vector search with a general-purpose embedding model.

Note: the leaderboard uses the full 5.23M-doc Wikipedia corpus. We searched 139k docs (~2.7% coverage), making retrieval meaningfully harder than a small toy subset but still easier than true fullwiki scale.
