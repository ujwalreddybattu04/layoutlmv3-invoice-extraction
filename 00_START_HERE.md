# Evaluator guide

**Invoice field extraction with LayoutLMv3**

Start with the three examples below, then follow the setup and code walkthrough.
The project extracts the issuer name/address, invoice number, issue date, and
grand total. Each detected value has one merged original-pixel bounding box;
unresolved fields remain `null`.

## 1. Inspect the main examples

These are the three primary deliverable examples. Each row links the original
invoice, the annotated result, and the structured JSON.

| Layout | Input | Annotated result | JSON |
|---|---|---|---|
| Classic invoice | [Image](samples/input/classic.png) | [Result](samples/output/classic.png) | [Fields](samples/output/classic.json) |
| Right-hand issuer header | [Image](samples/input/right_header.png) | [Result](samples/output/right_header.png) | [Fields](samples/output/right_header.json) |
| Ledger layout | [Image](samples/input/ledger.png) | [Result](samples/output/ledger.png) | [Fields](samples/output/ledger.json) |

For decision-level evidence, open the matching `.trace.json` in `samples/output`.
The [sample directory guide](samples/README.md) explains the additional cases.

## 2. Run the submission

Extract the complete ZIP, open a terminal in `invoice-extraction/`, and follow
[README section 1](README.md#1-setup). Prerequisites are Python 3.12, Tesseract 5
with English language data, and the pinned Python packages. The first base-model
download is approximately 504 MB. CPU inference is supported.

After setup, the assignment's CLI contract is:

```bash
python predict.py --input samples/input/classic.png --output result.jpg --json result.json
```

`result.jpg` is the annotated image; `result.json` contains `meta` and `fields`.
If Tesseract is not on PATH, add `--tesseract-cmd "path/to/tesseract"`.

## 3. Review the core implementation

```text
predict.py
  -> src/pipeline.py
       -> src/ocr.py       words, confidence, original pixel boxes
       -> src/model.py     LayoutLMv3 token labels and scores
       -> src/aggregate.py grouping, cleanup, ranking, abstention
       -> src/merge.py     geometry helpers and box unions
       -> src/visualize.py annotated image and side legend
```

Start with [Pipeline.run](src/pipeline.py), then
[aggregation](src/aggregate.py) and [box merging](src/merge.py).
[README section 5](README.md#5-token-to-field-merging) explains the rules and
thresholds in detail. The [source guide](src/README.md) maps all modules.

## 4. Check measurements and limitations

| Document | Purpose |
|---|---|
| [README.md](README.md) | The assignment's seven required sections, in order. |
| [FINAL_AUDIT.md](FINAL_AUDIT.md) | Final extracted-ZIP checks: 42 tests and 20 CPU cases. |
| [VALIDATION.md](VALIDATION.md) | Evaluation methodology, model comparisons, and installation evidence. |
| [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) | Actual failures, screenshots, root causes, and fixes. |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | Dependencies, model licensing, attribution, and assistance disclosure. |

The final diagnostic batch matched 97/100 field results, including correct nulls.
All fixtures are synthetic and include reviewed layouts and transformations.
This is not an unseen real-world accuracy estimate. Remaining failures involve
faint print, currency-symbol OCR, and inserted characters in one address.

## 5. Reproduce the checks

After installing `requirements-dev.txt`:

```bash
python -m pytest -q
python scripts/check_submission.py
python scripts/audit_submission.py --device cpu
```

The full audit writes generated inputs, JSON, annotated images, traces, and its
summary under `artifacts/final-audit/`. See the [script guide](scripts/README.md)
for evaluation and training commands.

## Archive map

```text
invoice-extraction/
  00_START_HERE.md          this review guide
  README.md                 main technical submission
  ERROR_ANALYSIS.md          real failures and screenshots
  FINAL_AUDIT.md             final verification summary
  VALIDATION.md              methodology and measured results
  THIRD_PARTY_NOTICES.md     attribution and licensing
  predict.py                required CLI entry point
  train.py                  optional adaptation training
  requirements*.txt         dependencies and tested environment
  schema.json               output JSON schema
  pyproject.toml            test configuration
  src/                      modular extraction implementation
  samples/
    input/                  three main invoices
    output/                 their annotated images, JSON, traces
    README.md               guide to all supporting evidence
    ...                     edge cases, comparisons, historical failures
  assets/                   small adaptation and licensed fonts
  scripts/                  evaluation, audit, and data generation
  tests/                    unit and regression tests
```

The archive contains one project folder. Base weights are fetched separately;
virtual environments, download caches, OCR installers, and local working files
are excluded. Training is optional when reviewing or running prediction.
