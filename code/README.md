# Reproducibility Guide

EviRAG has two execution paths.

## No-API evaluation

The reported local scores can be checked without a hosted API or secret key. The deterministic scorer compares a gold file with a prediction file:

```bash
python3 code/Score.py path/to/val_gold.csv path/to/predictions.csv
```

The archived summaries are in `code/results/`:

- `evidence_budget_ablation_summary.csv`
- `source_disjoint_five_fold_summary.csv`
- `bm25_qasper_scifact_2026-11-23.json`

The evidence-budget helper reproduces the context-size operation used in the experiment:

```bash
python3 code/limit_retrieval_candidates.py \
  --input reranked.jsonl \
  --output reranked_top3.jsonl \
  --k 3
```

## Full local pipeline

The core pipeline scripts are included here: corpus extraction, hybrid retrieval, neural reranking, structured generation, post-processing, scoring, and transfer-set preparation. A full run additionally requires the WattBot documents, processed chunks, model weights, and a GPU. Those large assets are intentionally not committed to this report repository.

The main stages are:

1. `extract_corpus.py` creates page-aware PDF chunks.
2. `build_hybrid_candidates.py` combines BM25 and dense retrieval.
3. `rerank_candidates.py` applies the neural passage reranker.
4. `generate_answers.py` produces the structured answer record.
5. `postprocess_wattbot_predictions.py` applies deterministic normalization.
6. `Score.py` evaluates answer values, units, citations, and abstentions.

## API and Kaggle boundaries

Hosted-model scripts are optional diagnostics. They require an environment variable such as `OPENROUTER_API_KEY`; no credential is stored in this repository, and the key must never be committed or printed. The no-API path is the recommended reproducibility path for grading.

The Kaggle score is a blind public-test result. It cannot be recreated locally because the test labels are hidden. The paper reports it separately from the reproducible local ablations and identifies the public leaderboard as approximately 56% of the test set.
