# Bundled assets

| Folder | Contents | Purpose |
|---|---|---|
| [adaptation/](adaptation/) | `weights.safetensors`, `config.json`, `training_report.json`, `LICENSE.md` | The 28.4 MB trained changes: final encoder block and six-label classifier. |
| [fonts/](fonts/) | DejaVu regular/bold fonts and their license | Consistent readable annotations and generated fixtures. |

The approximately 504 MB base checkpoint is downloaded separately on the first
model run and is not included in the archive. The adaptation requires the exact
base identity recorded in its configuration. No training is required to use it.

See [README section 3](../README.md#3-model) for model provenance and training,
and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for attribution and license
conditions. No bundled asset exceeds the assignment's 100 MB limit.
