# EvidenceFirst

Final project for Applied Deep Learning at Hamad Bin Khalifa University.

The project studies an evidence-controlled retrieval-augmented generation system for precision-critical scientific question answering. The report evaluates hybrid retrieval, neural reranking, structured generation, citation validation, abstention, and deterministic answer normalization on the WattBot benchmark and related transfer data.

## Authors

- Ali Shahroor
- Mohammed Al-Ani
- Yin Yang

Ali Shahroor and Mohammed Al-Ani contributed equally.

## Files

- `main.tex`: LaTeX source for the university report.
- `custom.bib`: bibliography.
- `evidencefirst_course_report.pdf`: compiled report.
- `figures/`: report figures.
- `code/`: selected scoring and transfer-evaluation scripts.

## Compile

The report uses the EACL 2023 style files. With a LaTeX installation:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The public WattBot 2026 result reported in the paper is the blind Kaggle score; local development and transfer experiments are reported separately.
