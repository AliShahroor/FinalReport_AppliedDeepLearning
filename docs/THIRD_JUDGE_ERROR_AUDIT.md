# Two-model error audit

The automatic error categories are useful for triage, but the paper should not call them human-validated labels unless annotators actually inspect the evidence. The recommended audit is:

1. Select 40--60 held-out questions where two independent systems disagree on the answer value, citation set, or abstention decision.
2. Give both annotators the question, gold answer, retrieved evidence, and the two predictions. Hide model names and prediction order.
3. Have Ali and Mohammed independently assign one primary cause: retrieval, reasoning, numerical/unit, citation, abstention, formatting, or no error.
4. Record a short evidence-based justification and whether the gold source was present in the retrieved candidates.
5. Compute Cohen's kappa for each annotator pair, adjudicate disagreements, and report the final distribution.
6. Use a third model only as a blinded tie-breaker. Its decision is advisory; it cannot override both human annotations without adjudication.

This design keeps the deterministic scorer as the primary evaluation, uses model judging as a low-cost review layer, and reserves human adjudication for genuinely ambiguous cases.
