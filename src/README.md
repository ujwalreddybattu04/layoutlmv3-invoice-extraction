# Source walkthrough

Read [pipeline.py](pipeline.py) first: `Pipeline.run` connects the stages.

| Module | Responsibility |
|---|---|
| [pipeline.py](pipeline.py) | Coordinate loading, OCR, model inference, aggregation, rotation mapping, and annotation. |
| [types.py](types.py) | Shared field names and word, prediction, piece, and candidate data structures. |
| [ocr.py](ocr.py) | Load images or PDF page 1; parse Tesseract words, confidence, and character boxes. |
| [model.py](model.py) | Load pinned LayoutLMv3 and adaptation; normalize boxes; resolve subtokens and overlapping windows. |
| [layout.py](layout.py) | Reading order, visual rows, text segments, measured word slices, and text stitching. |
| [aggregate.py](aggregate.py) | Build candidates, remove keys, check layout/content, combine lines, rank, and abstain. |
| [merge.py](merge.py) | Box normalization, clipping, overlap, union, and inverse rotation. |
| [visualize.py](visualize.py) | Draw one primary rectangle per field and a readable side legend. |
| [__init__.py](__init__.py) | Python package marker. |

The CLI and output serialization live in [predict.py](../predict.py).
The algorithm and its thresholds are explained in
[README section 5](../README.md#5-token-to-field-merging).

To follow one concrete decision, compare
[classic.json](../samples/output/classic.json) with
[classic.trace.json](../samples/output/classic.trace.json).
