# Human citation-support audit

This is a small validation study, not a replacement for the benchmark score.
Two annotators independently inspect the same blinded rows. For each row, they
answer `yes`, `no`, or `unclear` for:

1. Is the answer value correct?
2. Does the cited source support the answer?
3. Is the unit or numerical interpretation correct?
4. Should the system have abstained because the evidence is insufficient?

Annotators should see the question, generated answer, citation identifiers,
supporting quote, and the relevant source excerpt. They should not see the
model or condition name. The two annotators then compare disagreements and
record one adjudicated label. Report the percentage of `yes` labels for each
criterion and Cohen's kappa before adjudication. A sample of 50--100 rows is
enough for a transparent course-project audit; it should be described as a
small blinded audit rather than a population-level human evaluation.

The template is created with `scripts/select_human_audit_sample.py`. The
annotators may be the two project members, provided they follow the same
written rubric and disclose that the audit was performed by the authors.
