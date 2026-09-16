# Sample and evidence directory

**Start with [input/](input/) and [output/](output/).** They contain the three
primary examples requested by the assignment.

## How to read an output

For a given name, such as `classic`:

- `classic.png`: invoice with merged field boxes and a side legend.
- `classic.json`: extracted values, confidence, and original-pixel boxes.
- `classic.trace.json`: OCR, model scores, candidate decisions, and selected evidence.
- `evaluation.json`: measured results for that batch, when present.

Ground-truth files contain the expected answers used for evaluation. They are
not inference results and are not used by `predict.py` to extract values.

## Main and final reviewed results

| Location | What it contains | Associated input / answer key |
|---|---|---|
| [output/](output/) | Three main development examples: classic, right header, ledger. | [input/](input/), [ground_truth.json](ground_truth.json) |
| [review/output/](review/output/) | Final reruns of the two previously inspected layouts. | [heldout/input/](heldout/input/), [answer key](heldout/ground_truth.json) |
| [stress_review/output/](stress_review/output/) | Final stress results: faint total, wrapped name, currency glyph, blank page, missing total. | [stress/input/](stress/input/), [answer key](stress/ground_truth.json) |
| [edge_cases/output/](edge_cases/output/) | Corrected half-resolution example. | [edge_cases/input/](edge_cases/input/) |
| [edge_cases/audit_report.json](edge_cases/audit_report.json) | Final 20-case CPU audit summary with extraction-source hash. | Regenerate with `scripts/audit_submission.py`. |
| [runtime/](runtime/) | CPU example and window/blank/rotation checks. | See [validation](../VALIDATION.md). |

## Historical evidence and comparisons

These directories intentionally preserve earlier results, including mistakes.
They support the error analysis; they are not the primary final examples.

| Location | Why it is retained |
|---|---|
| [heldout/output/](heldout/output/) | First results on two untouched layouts, before reviewing their failures. |
| [stress/output/](stress/output/) | Earlier stress results before the final repairs. |
| [error_analysis/](error_analysis/) | Original checkpoint failures, early adapted runs, and the original low-resolution failure. |
| [ablation/](ablation/) | Model-only, rules-only, and original-checkpoint comparisons. |

Once a layout has been reviewed, a later run is a regression check rather than
a new unseen-layout test. All examples are synthetic. The distinction is
explained in [VALIDATION.md](../VALIDATION.md) and
[ERROR_ANALYSIS.md](../ERROR_ANALYSIS.md).
