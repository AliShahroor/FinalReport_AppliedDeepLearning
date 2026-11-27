# Reproducibility Guide

EviRAG has two execution paths.

## Quick start

From the repository root, install the small dependency used by the local scorer and run the no-API smoke test:

```bash
python3 -m pip install -r code/requirements-minimal.txt
python3 code/reproducibility_check.py
```

This verifies the scorer, the evidence-budget helper, and the archived result summaries. It does not need an API key, GPU, or Kaggle access.

## No-API evaluation

The reported local scores can be checked without a hosted API or secret key. The deterministic scorer compares a gold file with a prediction file:

```bash
python3 code/Score.py path/to/val_gold.csv path/to/predictions.csv
```

The archived summaries are in `code/results/`:

- `evidence_budget_ablation_summary.csv`
- `source_disjoint_five_fold_summary.csv`
- `bm25_qasper_scifact_2026-11-23.json`
- `error_analysis_summary.json` and `error_analysis_75q.csv`
- `source_disjoint_bootstrap_ci.json`
- `wattbot2025_transfer_scores.csv`

The evidence-budget helper reproduces the context-size operation used in the experiment:

```bash
python3 code/limit_retrieval_candidates.py \
  --input reranked.jsonl \
  --output reranked_top3.jsonl \
  --k 3
```

The held-out error analysis can be regenerated when the WattBot validation files are available:

```bash
python3 code/error_analysis.py \
  --gold path/to/val_gold.csv \
  --predictions path/to/predictions.csv \
  --manifest path/to/val_manifest.jsonl \
  --output-csv error_analysis.csv \
  --output-json error_analysis.json
```

The script deliberately calls the retrieval category `retrieval_risk`: it only means that a gold source was absent from the saved candidate set, not that retrieval is proven to be the causal failure.

Bootstrap intervals for the five grouped folds can be regenerated with:

```bash
python3 code/confidence_intervals.py \
  --gold-dir path/to/ms_swift \
  --pred-dir path/to/grouped_results \
  --output source_disjoint_bootstrap_ci.json
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

## Source links

- Repository: <https://github.com/AliShahroor/FinalReport_AppliedDeepLearning>
- University report source: <https://github.com/AliShahroor/FinalReport_AppliedDeepLearning/blob/main/main.tex>
- Pipeline code: <https://github.com/AliShahroor/FinalReport_AppliedDeepLearning/tree/main/code/pipeline>
- Reproducibility check: <https://github.com/AliShahroor/FinalReport_AppliedDeepLearning/blob/main/code/reproducibility_check.py>
