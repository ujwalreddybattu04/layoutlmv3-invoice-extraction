# Evaluation and supporting scripts

Run commands from the project root, after the main README setup.
Checking and audit commands require `requirements-dev.txt`.

| Script | Purpose | Example |
|---|---|---|
| [check_submission.py](check_submission.py) | Required files, result schema, box bounds, line unions, asset limits. | `python scripts/check_submission.py` |
| [audit_submission.py](audit_submission.py) | Full 20-case synthetic CPU audit, rotations, long input, and CLI error handling. | `python scripts/audit_submission.py --device cpu` |
| [evaluate.py](evaluate.py) | Measure field exact match and box IoU against an answer key. | `python scripts/evaluate.py --device cpu --output-dir artifacts/evaluation` |
| [verify_runtime.py](verify_runtime.py) | Blank page, 750-word window coverage, and rotation checks. | `python scripts/verify_runtime.py --device cpu` |
| [generate_samples.py](generate_samples.py) | Generate the development invoice fixtures. | `python scripts/generate_samples.py --help` |
| [generate_training.py](generate_training.py) | Generate the synthetic training/validation data. | `python scripts/generate_training.py --seed 1729` |

For another invoice set, give `evaluate.py` its `--manifest` and a fresh
`--output-dir`. Keeping new evaluation outputs under `artifacts/` preserves the
recorded evidence in `samples/`.

Inference/evaluation commands accept `--tesseract-cmd` when Tesseract is not on
PATH. The audit also accepts `--model`, `--local-files-only`, and `--output-dir`.
Use each script's `--help` for its exact options.

The adaptation training entry point is [train.py](../train.py); its methodology
and reproduction command are in [README section 3](../README.md#3-model).
