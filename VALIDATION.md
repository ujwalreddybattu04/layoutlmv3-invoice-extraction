# Validation

## Scope and measurements

The final extraction code was frozen at `b39062ce619319fd15c67e74cc03ca68f4fa6461`.
The model adaptation is unchanged after its eight-invoice validation selection.
All bundled invoices are synthetic. The company supplied no invoice dataset.

Exact match compares the **raw assembled value**, including punctuation and line
breaks. It does not silently normalize mistakes away. Null is correct only when
the ground truth is absent. Box IoU measures the overlap between the predicted
rectangle and the independently rendered text rectangle; a missing prediction
counts as zero for a present field.

| Evaluation | Invoices | Exact fields, including correct nulls | Mean IoU on present fields | Evidence |
|---|---:|---:|---:|---|
| Development examples (final) | 3 | 15/15 | 0.9856 | [Report](samples/output/evaluation.json) |
| First untouched-layout test | 2 | 7/10 | 0.7881 | [Report](samples/heldout/output/evaluation.json) |
| Same layouts after review; regression only | 2 | 9/10 | 0.9860 | [Report](samples/review/output/evaluation.json) |
| Stress cases after review | 5 | 23/25 | 0.8792 | [Report](samples/stress_review/output/evaluation.json) |

The first held-out test used two layouts not used to tune the model or rules,
at commit `4846c2901f63f5fe1cabe6cd2ad84bf61bb75c99`. Its **7/10** result is retained.
After those results were inspected, the name/address rules were improved. The
**9/10** rerun is a regression result on now-seen cases, not another unseen-layout
estimate. Development results likewise are not an unbiased accuracy estimate.
The remaining sidebar error is an OCR character error even though its box is close.

On the five stress cases, **17/19 present fields** match exactly and **6/6 absent
fields** correctly stay null. Faint print and the rupee glyph remain failures.
The two-line issuer name now stays separate from its postal address. Earlier
stress outputs are kept in `samples/stress/output`; the final reruns are in
`samples/stress_review/output`.

## What the model adds

All rows below use the same three development invoices and the same final
aggregation implementation. Removing either candidate source loses one field.
This comparison demonstrates that the model path executes and contributes; three
synthetic examples are insufficient for a general accuracy claim.

| Pipeline | Exact fields | Mean IoU | Evidence |
|---|---:|---:|---|
| Adapted model + rules | 15/15 | 0.9856 | [Report](samples/output/evaluation.json) |
| Adapted model candidates only | 14/15 | 0.9200 | [Report](samples/ablation/model_only/evaluation.json) |
| Rules only; model disabled | 14/15 | 0.9200 | [Report](samples/ablation/heuristic_only/evaluation.json) |
| Original checkpoint + current rules | 14/15 | 0.9200 | [Report](samples/ablation/original_checkpoint/evaluation.json) |

The original public model with the initial aggregation achieved 11/15 exact
fields; those actual outputs remain in `samples/error_analysis/initial`.
The adapted model's **0.9720 validation field-token macro-F1** is a different
metric from end-to-end field exact match. It is reported separately in
`assets/adaptation/training_report.json`.

## Installation and runtime checks

- Windows 11; a fresh isolated Python **3.12.12** environment with
  `include-system-site-packages = false`.
- Pinned PyTorch **2.6.0+cpu**, Transformers **4.57.6**, Pillow **12.0.0**,
  PyMuPDF **1.27.2**, and Tesseract **5.5.3.20260724** with English language data.
- Direct dependency installation succeeded in the fresh environment. The complete
  resolved CPU environment is recorded in `requirements-lock.txt`.
- CPU CLI inference succeeded with the default checkpoint identity, bundled
  adaptation, verified local Hugging Face cache, and `--local-files-only`. All
  five classic fields were found. See `samples/runtime/default_cpu.json`.
- GPU batch evaluation used Python 3.13.0, PyTorch 2.6.0+cu124, and an RTX 4050
  Laptop GPU. Each evaluation report records its runtime and source revision.
- A cached CPU CLI startup took **81.36 seconds** in one fresh-process run on
  this Windows machine; model initialization/import dominates that number.
  Warm GPU development runs in the final batch took approximately 0.71–1.21
  seconds per page. Batch timings exclude one-time model initialization.

### Checks actually run

1. **39 unit/regression tests pass** in the isolated Python 3.12 environment.
   They cover grouping, fused/split tokens, multi-line values, negative keys,
   missing fields, confidence, conflicting candidates, character boxes, date
   ambiguity, box normalization, inverse rotation, and CLI path guards.
2. A **750-word** model input spans **two windows**; every word receives exactly
   one resolved prediction, including the last word.
3. A blank page produces all five nulls and no model forward pass.
4. A sideways invoice corrected with `--rotate 90` has the same values as the
   upright input; all final rectangles map exactly to the original sideways pixels.
5. PDF loading was checked against the supplied four-page assignment PDF: only
   page 1 was rasterized, at **1240×1755** pixels for 150 DPI. This is a PDF input
   handling check, not an invoice extraction accuracy case.
6. Annotated development outputs were visually inspected for readable labels,
   one primary box per field, distinct colors, and no legend text over the invoice.

Machine-readable long-input/blank/rotation evidence is in
`samples/runtime/runtime_checks.json`. The schema and bounds checker is
`scripts/check_submission.py`.

## Reproduce

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/evaluate.py --device cpu
python scripts/evaluate.py --manifest samples/heldout/ground_truth.json --output-dir samples/review/output --device cpu
python scripts/evaluate.py --manifest samples/stress/ground_truth.json --output-dir samples/stress_review/output --device cpu
python scripts/verify_runtime.py --device cpu
python scripts/check_submission.py
```

Add `--tesseract-cmd "path/to/tesseract"` to inference/evaluation commands when
the executable is not on PATH. `scripts/verify_runtime.py` writes fresh runtime
fixtures under ignored `artifacts/runtime-checks`.

## Practical limits

No private or real-world vendor test set was available. The two initially
held-out layouts and eight validation invoices are too few for a reliable
production estimate. The outputs and failure analysis support a reproducible
take-home engineering demonstration; they do not establish perfect extraction.
