# Final submission audit

The submission was extracted into a fresh folder and exercised with the real
LayoutLMv3 model and bundled adaptation on CPU, using the isolated Python 3.12
environment. Model weights came from a verified local cache. All examples are
synthetic; transformed cases are checks of robustness, not a new unseen-vendor
accuracy estimate.

## Results

| Cases | Exact field results |
|---|---:|
| Three development layouts | 15/15 |
| Two previously reviewed layouts | 9/10 |
| Five stress cases | 23/25 |
| Ten additional transformed or missing-field cases | 50/50 |
| Combined diagnostic batch | 97/100 |

The additional cases cover JPEG quality 65, grayscale, 600 x 800 resolution,
RGBA input, explicit 90/180/270-degree rotation correction, missing invoice
number, missing invoice date, and a two-page PDF that must use page 1 only.
The repeated identity/layouts and correct nulls make the combined percentage
unsuitable as a production accuracy estimate.

- **42 unit/regression tests pass** in the extracted project.
- Every generated result passes JSON schema, original-pixel box bounds, and
  multi-line union checks.
- The CLI runs as a separate process, writes JSON and an annotated JPG, and
  returns exit code 0. Its field values, boxes, and confidences agree with the
  pipeline results.
- All **750 supplied OCR words** receive model predictions across two windows.
  This specifically checks model window coverage, using a constructed word list.
- The three rotation cases preserve values and correctly map rectangles back
  to original sideways/upside-down image coordinates.
- Missing input, corrupt image, invalid DPI/output extension, output-path
  collisions, and attempted input overwrite produce clear errors with expected
  exit codes. The source image stays unchanged.
- Dependency compatibility, archive integrity, and the 100 MB asset limit pass.

## Issue found and fixed during this audit

At half resolution, the postal address incorrectly included the fragment `Emai`.
The email label and email value had become separate OCR segments. A bounded
same-row contact check now prevents that orphan fragment joining the address.
Three regression tests protect the fix, legitimate postal lines, and separate
contact columns. Annotation outlines also scale down on small images to preserve
the readability of small text.

The original failure is preserved in
`samples/error_analysis/low_resolution_initial`. The corrected result is in
`samples/edge_cases/output/half_resolution.json` and its matching PNG.

## Remaining errors

1. The sidebar address contains inserted OCR characters/punctuation.
2. A very faint printed total cannot be recovered and remains null.
3. The rupee glyph is read as `%`; the invalid amount is rejected and remains null.

These failures are retained in the report. They are not hidden by returning the
subtotal, inventing a currency, or rewriting the ground-truth answer.

## Reproduce

Install `requirements-dev.txt`, make Tesseract available, then run from the
project root:

```bash
python -m pytest -q
python scripts/check_submission.py
python scripts/audit_submission.py --device cpu
```

When Tesseract is not on PATH, add `--tesseract-cmd "path/to/tesseract"`.
The audit defaults to `artifacts/final-audit`, preserving each generated input,
JSON result, annotated image, and decision trace. Its summary is `audit_report.json`.
The submitted run summary is in `samples/edge_cases/audit_report.json` and includes
a hash of the extraction source files so the tested code can be identified.
