# Adapted model weights: attribution and license notice

The adapted model tensors are provided under
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).
See the [full license text](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Upstream work: **LayoutLMv3**, Microsoft; Yupan Huang, Tengchao Lv, Lei Cui,
Yutong Lu, and Furu Wei. Original checkpoint and license declaration:
https://huggingface.co/microsoft/layoutlmv3-base

Intermediate checkpoint: **Kapilydv6/layoutlmv3-invoice-parser**, revision
`e464cfcadf6b766e105119c46e06f81ff1e4836a`:
https://huggingface.co/Kapilydv6/layoutlmv3-invoice-parser

Changes made for this assignment: the final encoder block was fine-tuned on
36 synthetic invoices, and the classifier was replaced and trained for six flat
labels. Only these changed tensors are distributed in `weights.safetensors`.
Eight synthetic validation invoices were used to select the training epoch.
See `config.json`, `training_report.json`, and the repository's `train.py`.

This notice applies to adapted model weights. Third-party code, dependencies,
and fonts retain their own licenses; see `THIRD_PARTY_NOTICES.md`.
