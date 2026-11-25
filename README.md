# EviRAG

Final project for Applied Deep Learning at Hamad Bin Khalifa University.

The project studies EviRAG, an evidence-first retrieval-augmented generation system for precision-critical scientific question answering. The report evaluates hybrid retrieval, neural reranking, structured generation, citation validation, abstention, and deterministic answer normalization on the WattBot benchmark and related transfer data.

## Authors

- Ali Shahroor
- Mohammed Al-Ani
- Yin Yang

Ali Shahroor and Mohammed Al-Ani contributed equally.

## Files

- `main.tex`: LaTeX source for the university report.
- `custom.bib`: bibliography.
- `evidencefirst_course_report.pdf`: compiled report.
- `eacl/`: anonymous EACL/CHOMPS submission source and compiled PDF.
- `figures/`: report figures.
- `code/`: scoring, transfer-evaluation, and core EviRAG pipeline scripts.
- `code/README.md`: no-API reproduction instructions and API/Kaggle boundaries.

## Compile

The report uses the EACL 2023 style files. With a LaTeX installation:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The public WattBot 2026 result reported in the paper is the blind Kaggle score; local development and transfer experiments are reported separately.

## Scope and limitations

The main evaluation is English scientific QA about AI infrastructure and environmental impact. WattBot 2025 is a small 41-question transfer set, while SciFact and QASPER are retrieval-focused transfer diagnostics rather than full WattBot evaluations. The Kaggle test labels are hidden, so the public score cannot be reproduced locally. Hosted-model calls are optional and require the user's own credentials; the saved no-API outputs and deterministic scorers are the reproducibility path. Results should therefore be read as evidence for this benchmark and transfer setup, not as a universal guarantee across domains, languages, or model families.
